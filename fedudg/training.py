import copy
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .core import (
    ProjectionHead, aggregate_models, average_anchors, feature_dimension,
    optimizer_state_to_cpu, semantic_admission, semantic_anchor_logits,
    soft_cross_entropy, state_dict_to_cpu, update_ema_anchors, update_ema_model,
)
from .data import ClientDataset
from .io import atomic_checkpoint, atomic_json, load_checkpoint, restore_rng, rng_state


class LabeledBundle(nn.Module):
    def __init__(self, model, anchors, temperature):
        super().__init__()
        self.model = model
        self.anchors = nn.Parameter(anchors)
        self.temperature = temperature

    def forward(self, images):
        features, logits = self.model(images)
        return features, logits, semantic_anchor_logits(features, self.anchors, self.temperature)


class UnlabeledBundle(nn.Module):
    def __init__(self, model, projection):
        super().__init__()
        self.model = model
        self.projection = projection

    def forward(self, images):
        features, logits = self.model(images)
        return features, logits, self.projection(features)


def make_optimizer(bundle, config, learning_rate, state):
    optimizer = torch.optim.SGD(
        [parameter for parameter in bundle.parameters() if parameter.requires_grad],
        lr=learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    if state:
        optimizer.load_state_dict(copy.deepcopy(state))
        device = next(bundle.parameters()).device
        for values in optimizer.state.values():
            for name, value in list(values.items()):
                if torch.is_tensor(value):
                    values[name] = value.to(device)
    for group in optimizer.param_groups:
        group['lr'] = float(learning_rate)
    return optimizer


def train_labeled_client(config, loader, global_model, anchors, optimizer_state, learning_rate):
    device = next(global_model.parameters()).device
    model = copy.deepcopy(global_model).to(device)
    model.train()
    bundle = LabeledBundle(model, anchors.detach().clone().to(device), config.anchor_temperature).to(device)
    optimizer = make_optimizer(bundle, config, learning_rate, optimizer_state)
    totals = {'classification_loss': 0.0, 'anchor_loss': 0.0, 'total_loss': 0.0,
              'batches': 0, 'samples': 0}
    for _ in range(config.local_ep):
        for _, _, views, labels in loader:
            images = views[0].to(device, non_blocking=True)
            targets = labels.to(device, non_blocking=True).argmax(dim=1)
            optimizer.zero_grad(set_to_none=True)
            _, logits, anchor_logits = bundle(images)
            classification = F.cross_entropy(logits, targets)
            anchor_loss = F.cross_entropy(anchor_logits, targets)
            loss = classification + config.anchor_weight * anchor_loss
            loss.backward()
            optimizer.step()
            totals['classification_loss'] += float(classification.detach())
            totals['anchor_loss'] += float(anchor_loss.detach())
            totals['total_loss'] += float(loss.detach())
            totals['batches'] += 1
            totals['samples'] += int(targets.numel())
    denominator = max(totals['batches'], 1)
    metrics = {key: value / denominator if key.endswith('_loss') else value
               for key, value in totals.items()}
    return (state_dict_to_cpu(model.state_dict()),
            F.normalize(bundle.anchors.detach(), dim=1, eps=1e-8).cpu(),
            optimizer_state_to_cpu(optimizer.state_dict()), metrics)


def proximal_penalty(model, reference):
    penalty = None
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or name.startswith('classifier.') or '.classifier.' in name:
            continue
        value = (parameter - reference[name]).pow(2).sum()
        penalty = value if penalty is None else penalty + value
    return next(model.parameters()).sum() * 0.0 if penalty is None else penalty


def masked_semantic_loss(values, mask):
    if bool(mask.any()):
        return values[mask].mean()
    return values.sum() * 0.0


def train_unlabeled_client(config, loader, global_model, teacher, teacher_anchors,
                           projection_state, optimizer_state, learning_rate, semantic_weight):
    device = next(global_model.parameters()).device
    model = copy.deepcopy(global_model).to(device)
    model.train()
    teacher.eval()
    projection = ProjectionHead(feature_dimension(model), config.projection_hidden).to(device)
    if projection_state:
        projection.load_state_dict(projection_state, strict=True)
    bundle = UnlabeledBundle(model, projection)
    optimizer = make_optimizer(bundle, config, learning_rate, optimizer_state)
    reference = {name: value.detach().clone().to(device) for name, value in model.named_parameters()}
    totals = {'distillation_loss': 0.0, 'semantic_loss': 0.0, 'proximal_loss': 0.0,
              'total_loss': 0.0, 'admitted': 0, 'confidence_sum': 0.0, 'js_sum': 0.0,
              'samples': 0, 'batches': 0}
    for _ in range(config.unlabeled_local_ep):
        for _, _, views, _ in loader:
            weak1, weak2, strong = [image.to(device, non_blocking=True) for image in views]
            with torch.no_grad():
                teacher_features, logits1 = teacher(weak1)
                probability1 = F.softmax(logits1.float(), dim=1)
                _, logits2 = teacher(weak2)
                anchor_logits = semantic_anchor_logits(
                    teacher_features, teacher_anchors, config.anchor_temperature)
                probability2 = F.softmax(logits2.float(), dim=1)
                anchor_probability = F.softmax(anchor_logits.float(), dim=1)
                mask, target, js, confidence = semantic_admission(
                    probability1, probability2, anchor_probability,
                    config.confidence_threshold, config.js_threshold)
            optimizer.zero_grad(set_to_none=True)
            _, student_logits, projected = bundle(strong)
            distillation = (1.0 - F.cosine_similarity(
                projected.float(), teacher_features.detach().float(), dim=1)).mean()
            semantic = masked_semantic_loss(soft_cross_entropy(student_logits, target.detach()), mask)
            proximal = proximal_penalty(model, reference)
            loss = (config.distillation_weight * distillation + semantic_weight * semantic
                    + 0.5 * config.proximal_mu * proximal)
            loss.backward()
            optimizer.step()
            totals['distillation_loss'] += float(distillation.detach())
            totals['semantic_loss'] += float(semantic.detach())
            totals['proximal_loss'] += float(proximal.detach())
            totals['total_loss'] += float(loss.detach())
            totals['admitted'] += int(mask.sum())
            totals['confidence_sum'] += float(confidence.sum())
            totals['js_sum'] += float(js.sum())
            totals['samples'] += int(strong.shape[0])
            totals['batches'] += 1
    batches = max(totals['batches'], 1)
    samples = max(totals['samples'], 1)
    metrics = {name: totals[name] / batches for name in
               ('distillation_loss', 'semantic_loss', 'proximal_loss', 'total_loss')}
    metrics.update(samples=totals['samples'], admission_rate=totals['admitted'] / samples,
                   mean_confidence=totals['confidence_sum'] / samples, mean_js=totals['js_sum'] / samples)
    return (state_dict_to_cpu(model.state_dict()), state_dict_to_cpu(projection.state_dict()),
            optimizer_state_to_cpu(optimizer.state_dict()), metrics)


def evaluate(model, dataset, config):
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False,
                        num_workers=config.num_workers, pin_memory=torch.cuda.is_available())
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    correct, count = 0, 0
    with torch.no_grad():
        for _, _, images, labels in loader:
            _, logits = model(images.to(device, non_blocking=True))
            targets = labels.to(device, non_blocking=True).argmax(dim=1)
            correct += int((logits.argmax(dim=1) == targets).sum())
            count += int(targets.numel())
    model.train(was_training)
    if count == 0:
        raise ValueError('Cannot evaluate an empty dataset')
    return correct / count


def weighted_metric(metrics, key):
    if not metrics:
        return 0.0
    count = sum(max(int(item['samples']), 1) for item in metrics)
    return sum(float(item[key]) * max(int(item['samples']), 1) for item in metrics) / count


def learning_rate(config, round_index):
    return config.base_lr * (0.1 if round_index >= int(config.rounds * 0.8) else 1.0)


def checkpoint_config(config):
    ignored = {'data_dir', 'output_dir', 'device', 'num_workers', 'checkpoint_every'}
    return {key: value for key, value in config.to_dict().items() if key not in ignored}


def train(config, protocol, model, run_dir, resume=''):
    run_dir = Path(run_dir)
    device = next(model.parameters()).device
    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    anchors = F.normalize(model.classifier.weight.detach().float().clone().to(device), dim=1, eps=1e-8)
    teacher_anchors = anchors.detach().clone()
    loaders = {client: DataLoader(
        ClientDataset(protocol.train, indices), batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=torch.cuda.is_available())
        for client, indices in protocol.clients.items()}
    labeled_optimizers = {client: None for client in protocol.labeled_ids}
    unlabeled_optimizers = {client: None for client in protocol.unlabeled_ids}
    projections = {client: None for client in protocol.unlabeled_ids}
    history = []
    best_score, best_round, start_round = float('-inf'), None, 0
    if resume:
        saved = load_checkpoint(resume)
        if saved.get('method') != 'FedUDG' or saved.get('schema_version') != 1:
            raise ValueError('Not a FedUDG release checkpoint')
        if checkpoint_config(config) != checkpoint_config(type(config)(**saved['config'])):
            raise ValueError('The training configuration differs from the recovery checkpoint')
        if saved['partition'] != protocol.manifest():
            raise ValueError('Client partitions differ from the recovery checkpoint')
        model.load_state_dict(saved['student_state_dict'], strict=True)
        teacher.load_state_dict(saved['state_dict'], strict=True)
        anchors = saved['anchors'].to(device)
        teacher_anchors = saved['teacher_anchors'].to(device)
        labeled_optimizers = saved['labeled_optimizers']
        unlabeled_optimizers = saved['unlabeled_optimizers']
        projections = saved['projections']
        history = saved['history']
        best_score, best_round = saved['best_source_validation'], saved['best_round']
        start_round = saved['round'] + 1
        restore_rng(saved['rng_state'])
        logging.info('Resuming from round %s', start_round)
    started = time.time()
    for round_index in range(start_round, config.rounds):
        lr = learning_rate(config, round_index)
        ramp = min(1.0, max(0, round_index - config.warmup_rounds + 1) / config.rampup_rounds)
        mix = config.unlabeled_mix * ramp
        labeled_states, labeled_anchors, labeled_metrics = [], [], []
        for client in protocol.labeled_ids:
            state, anchor, optimizer, metrics = train_labeled_client(
                config, loaders[client], model, anchors, labeled_optimizers[client], lr)
            labeled_states.append(state)
            labeled_anchors.append(anchor)
            labeled_optimizers[client] = optimizer
            labeled_metrics.append(metrics)
            logging.info('Labeled client %d loss %.6f', client, metrics['total_loss'])
        unlabeled_states, unlabeled_metrics = [], []
        if round_index >= config.warmup_rounds:
            for client in protocol.unlabeled_ids:
                state, projection, optimizer, metrics = train_unlabeled_client(
                    config, loaders[client], model, teacher, teacher_anchors,
                    projections[client], unlabeled_optimizers[client], lr, config.semantic_weight * ramp)
                unlabeled_states.append(state)
                projections[client], unlabeled_optimizers[client] = projection, optimizer
                unlabeled_metrics.append(metrics)
                logging.info('Unlabeled client %d loss %.6f admission %.4f',
                             client, metrics['total_loss'], metrics['admission_rate'])
        model.load_state_dict(aggregate_models(labeled_states, unlabeled_states, mix), strict=True)
        anchors = average_anchors(labeled_anchors).to(device)
        steps = max(1, int(round(np.mean([item['batches'] for item in labeled_metrics]))))
        decay = config.ema_decay ** steps
        update_ema_model(teacher, model, decay)
        update_ema_anchors(teacher_anchors, anchors, decay)
        by_domain = {domain: evaluate(teacher, dataset, config)
                     for domain, dataset in protocol.validation.items()}
        source_score = float(np.mean(list(by_domain.values())))
        base = {'schema_version': 1, 'method': 'FedUDG', 'round': round_index,
                'state_dict': state_dict_to_cpu(teacher.state_dict()), 'config': config.to_dict(),
                'classes': list(protocol.train.classes), 'source_validation': source_score,
                'source_validation_by_domain': by_domain}
        if source_score > best_score:
            best_score, best_round = source_score, round_index
            atomic_checkpoint(base, run_dir / 'best.pth')
        row = {'round': round_index, 'learning_rate': lr, 'unlabeled_mix': mix,
               'semantic_weight': config.semantic_weight * ramp, 'ema_decay': decay,
               'source_validation': source_score, 'source_validation_by_domain': by_domain}
        for prefix, metrics in [('labeled', labeled_metrics), ('unlabeled', unlabeled_metrics)]:
            keys = ('classification_loss', 'anchor_loss', 'total_loss') if prefix == 'labeled' else (
                'distillation_loss', 'semantic_loss', 'total_loss', 'admission_rate', 'mean_confidence', 'mean_js')
            for key in keys:
                row[prefix + '_' + key] = weighted_metric(metrics, key)
        history.append(row)
        atomic_json(history, run_dir / 'history.json')
        logging.info('Round %d/%d source_val %.6f best %.6f@%s',
                     round_index + 1, config.rounds, source_score, best_score, best_round)
        if (round_index + 1) % config.checkpoint_every == 0 and round_index + 1 < config.rounds:
            base.update(student_state_dict=state_dict_to_cpu(model.state_dict()), anchors=anchors.cpu(),
                        teacher_anchors=teacher_anchors.cpu(), labeled_optimizers=labeled_optimizers,
                        unlabeled_optimizers=unlabeled_optimizers, projections=projections, history=history,
                        best_source_validation=best_score, best_round=best_round,
                        partition=protocol.manifest(), rng_state=rng_state())
            atomic_checkpoint(base, run_dir / 'last.pth')
    if not math.isfinite(best_score):
        raise RuntimeError('No valid source-validation checkpoint was produced')
    atomic_json({'method': 'FedUDG', 'completed_rounds': config.rounds,
                 'best_source_validation': best_score, 'best_round': best_round,
                 'training_seconds_this_session': time.time() - started}, run_dir / 'train_summary.json')
    recovery = run_dir / 'last.pth'
    if recovery.is_file():
        recovery.unlink()
    logging.info('Training complete. Test checkpoint: %s', run_dir / 'best.pth')
