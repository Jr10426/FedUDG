import logging
import sys
from pathlib import Path

import torch

from fedudg.config import Config, training_parser
from fedudg.data import build_protocol
from fedudg.io import atomic_json, seed_everything
from fedudg.model import ResNet18
from fedudg.training import train


def main():
    options = training_parser().parse_args()
    resume, preflight = options.resume, options.preflight
    values = vars(options).copy()
    del values['resume'], values['preflight']
    config = Config(**values).validate()
    if not config.output_dir:
        suffix = '_a{}'.format(format(config.dirichlet_alpha, 'g')) if (
            config.dataset != 'pacs_domainnet' or config.setting == 'dirichlet4') else ''
        config.output_dir = str(Path('runs') / '{}_{}_{}{}_s{}'.format(
            config.dataset, config.target_domain, config.setting, suffix, config.seed))
    if config.device.startswith('cuda') and not torch.cuda.is_available() and not preflight:
        raise RuntimeError('CUDA is unavailable. Use --device cpu for CPU execution.')
    seed_everything(config.seed)
    protocol = build_protocol(config)
    if preflight:
        print('Preflight OK: {} labeled clients, {} unlabeled clients, {} classes'.format(
            len(protocol.labeled_ids), len(protocol.unlabeled_ids), protocol.train.num_classes))
        print(protocol.manifest())
        return
    run_dir = Path(config.output_dir).resolve()
    if resume:
        if Path(resume).resolve().parent != run_dir or not Path(resume).is_file():
            raise ValueError('--resume must refer to last.pth in this output directory')
        if not (run_dir / 'best.pth').is_file():
            raise FileNotFoundError('Keep best.pth alongside last.pth when resuming')
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError('Use an empty --output_dir or resume from its last.pth')
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(str(run_dir / 'train.log'), mode='a' if resume else 'w'),
                                  logging.StreamHandler(sys.stdout)])
    atomic_json(config.to_dict(), run_dir / 'config.json')
    atomic_json(protocol.manifest(), run_dir / 'client_partition.json')
    model = ResNet18(protocol.train.num_classes, drop_rate=config.drop_rate, pretrained=True)
    model = model.to(torch.device(config.device))
    model.train()
    train(config, protocol, model, run_dir, resume)


if __name__ == '__main__':
    main()
