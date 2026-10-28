import copy

import torch
from torch import nn
import torch.nn.functional as F


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def feature_dimension(model):
    base = unwrap_model(model)
    classifier = getattr(base, 'classifier', None)
    if classifier is None or not hasattr(classifier, 'in_features'):
        raise ValueError('FedUDG requires a model with a linear classifier.in_features')
    return int(classifier.in_features)


def semantic_anchor_logits(features, anchors, temperature=0.1):
    if features.ndim != 2 or anchors.ndim != 2:
        raise ValueError('features and anchors must both be rank-2 tensors')
    if features.shape[1] != anchors.shape[1]:
        raise ValueError('feature and anchor dimensions do not match')
    if float(temperature) <= 0:
        raise ValueError('anchor temperature must be positive')
    normalized_features = F.normalize(features.float(), dim=1, eps=1e-08)
    normalized_anchors = F.normalize(anchors.float(), dim=1, eps=1e-08)
    return normalized_features @ normalized_anchors.t() / float(temperature)


def multi_distribution_js(probabilities, eps=1e-08):
    if len(probabilities) < 2:
        raise ValueError('at least two probability tensors are required')
    stacked = torch.stack([p.float().clamp_min(eps) for p in probabilities], dim=0)
    if not torch.isfinite(stacked).all():
        raise ValueError('probabilities contain NaN or Inf')
    stacked = stacked / stacked.sum(dim=-1, keepdim=True).clamp_min(eps)
    mixture = stacked.mean(dim=0).clamp_min(eps)
    divergence = stacked * (stacked.log() - mixture.log().unsqueeze(0))
    return divergence.sum(dim=-1).mean(dim=0)


def geometric_mean_target(probabilities, eps=1e-08):
    stacked = torch.stack([p.float().clamp_min(eps) for p in probabilities], dim=0)
    log_target = stacked.log().mean(dim=0)
    target = torch.exp(log_target - log_target.max(dim=1, keepdim=True).values)
    return target / target.sum(dim=1, keepdim=True).clamp_min(eps)


def semantic_admission(class_prob_1, class_prob_2, anchor_prob,
                       confidence_threshold=0.8, js_threshold=0.1):
    probabilities = [class_prob_1, class_prob_2, anchor_prob]
    shapes = [tuple(p.shape) for p in probabilities]
    if len(set(shapes)) != 1:
        raise ValueError('all semantic probability tensors must have the same shape')
    predictions = torch.stack([p.argmax(dim=1) for p in probabilities], dim=0)
    agreement = (predictions == predictions[0:1]).all(dim=0)
    confidences = torch.stack([p.max(dim=1).values for p in probabilities], dim=0)
    minimum_confidence = confidences.min(dim=0).values
    js = multi_distribution_js(probabilities)
    mask = agreement & (minimum_confidence >= float(confidence_threshold))
    mask = mask & (js <= float(js_threshold))
    target = geometric_mean_target(probabilities)
    return (mask, target, js, minimum_confidence)


def soft_cross_entropy(logits, soft_targets):
    if logits.shape != soft_targets.shape:
        raise ValueError('logits and soft targets must have identical shapes')
    return -(soft_targets * F.log_softmax(logits.float(), dim=1)).sum(dim=1)


class ProjectionHead(nn.Module):

    def __init__(self, feature_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(feature_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(feature_dim)),
        )

    def forward(self, features):
        return self.net(features)


def _mean_state_values(values):
    first = values[0]
    if torch.is_floating_point(first) or torch.is_complex(first):
        return torch.stack(
            [value.to(dtype=first.dtype) for value in values], dim=0,
        ).mean(dim=0)
    return torch.stack(values, dim=0).max(dim=0).values


def average_state_dicts(states):
    if not states:
        raise ValueError('at least one state dict is required')
    keys = list(states[0].keys())
    if any((list(state.keys()) != keys for state in states[1:])):
        raise ValueError('state dict keys do not match')
    result = {}
    for key in keys:
        result[key] = _mean_state_values([state[key] for state in states])
    return result


def average_anchors(anchor_tensors):
    if not anchor_tensors:
        raise ValueError('at least one anchor tensor is required')
    average = torch.stack([tensor.float() for tensor in anchor_tensors], dim=0).mean(dim=0)
    return F.normalize(average, dim=1, eps=1e-08)


def aggregate_models(labeled_states, unlabeled_states, weight):
    labeled = average_state_dicts(labeled_states)
    if not unlabeled_states or weight == 0:
        return labeled
    if not 0 <= weight < 1:
        raise ValueError('The unlabeled mixing weight must be in [0, 1)')
    unlabeled = average_state_dicts(unlabeled_states)
    return {
        name: (1.0 - weight) * value + weight * unlabeled[name]
        if torch.is_floating_point(value) else value
        for name, value in labeled.items()
    }


@torch.no_grad()
def update_ema_model(ema_model, student_model, decay):
    decay = float(decay)
    ema_state = unwrap_model(ema_model).state_dict()
    student_state = unwrap_model(student_model).state_dict()
    for (key, ema_value) in ema_state.items():
        student_value = student_state[key].to(device=ema_value.device)
        if torch.is_floating_point(ema_value):
            ema_value.mul_(decay).add_(student_value, alpha=1.0 - decay)
        else:
            ema_value.copy_(student_value)


@torch.no_grad()
def update_ema_anchors(ema_anchors, student_anchors, decay):
    ema_anchors.mul_(float(decay)).add_(student_anchors, alpha=1.0 - float(decay))
    ema_anchors.copy_(F.normalize(ema_anchors, dim=1, eps=1e-08))


def state_dict_to_cpu(state):
    return {key: value.detach().cpu().clone() for (key, value) in state.items()}


def optimizer_state_to_cpu(state):
    result = copy.deepcopy(state)
    for entry in result.get('state', {}).values():
        for (key, value) in list(entry.items()):
            if torch.is_tensor(value):
                entry[key] = value.detach().cpu()
    return result

