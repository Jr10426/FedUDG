import argparse
from pathlib import Path

import torch

from fedudg.config import Config
from fedudg.data import build_target
from fedudg.io import atomic_json, load_checkpoint
from fedudg.model import ResNet18
from fedudg.training import evaluate


def main():
    parser = argparse.ArgumentParser(description='Evaluate a source-selected FedUDG EMA checkpoint')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', default='')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    saved = load_checkpoint(args.checkpoint)
    if saved.get('method') != 'FedUDG' or saved.get('schema_version') != 1:
        raise ValueError('Expected a checkpoint from this FedUDG release')
    config = Config(**saved['config'])
    config.device, config.batch_size, config.num_workers = args.device, args.batch_size, args.num_workers
    if args.data_dir:
        config.data_dir = args.data_dir
    config.validate()
    target = build_target(config, saved['classes'])
    model = ResNet18(len(saved['classes']), drop_rate=config.drop_rate, pretrained=False)
    model.load_state_dict(saved['state_dict'], strict=True)
    model.to(torch.device(config.device))
    accuracy = evaluate(model, target, config)
    result = {'method': 'FedUDG', 'dataset': config.dataset, 'target_domain': config.target_domain,
              'setting': config.setting, 'dirichlet_alpha': config.dirichlet_alpha,
              'seed': config.seed, 'round': saved['round'],
              'accuracy': accuracy, 'accuracy_percent': 100.0 * accuracy,
              'source_validation': saved['source_validation'], 'target_samples': len(target)}
    atomic_json(result, Path(args.checkpoint).resolve().parent / 'test_results.json')
    print('Target accuracy: {:.2f}%'.format(100.0 * accuracy))


if __name__ == '__main__':
    main()
