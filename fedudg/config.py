import argparse
import math
from dataclasses import asdict, dataclass
from typing import Optional


PACS_DOMAINS = ('pacs_photo', 'pacs_art_painting', 'pacs_cartoon', 'pacs_sketch')
DOMAINNET_DOMAINS = ('real', 'painting', 'clipart', 'quickdraw', 'infograph', 'sketch')
VLCS_DOMAINS = ('caltech', 'labelme', 'pascal', 'sun')
SHARED_CLASSES = ('dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person')
JOINT_SETTINGS = ('single', 'random2', 'random4', 'dirichlet4')


@dataclass
class Config:
    dataset: str
    target_domain: str
    setting: str = ''
    data_dir: str = 'data'
    output_dir: str = ''
    device: str = 'cuda:0'
    seed: int = 1337
    rounds: int = 80
    local_ep: int = 5
    batch_size: int = 16
    num_workers: int = 4
    base_lr: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.0005
    image_size: int = 224
    drop_rate: float = 0.2
    validation_fraction: float = 0.2
    dirichlet_alpha: Optional[float] = None
    min_client_samples: int = 16
    checkpoint_every: int = 5
    warmup_rounds: int = 10
    unlabeled_local_ep: int = 1
    ema_decay: float = 0.999
    confidence_threshold: float = 0.8
    js_threshold: float = 0.1
    anchor_temperature: float = 0.1
    anchor_weight: float = 1.0
    distillation_weight: float = 0.5
    semantic_weight: float = 1.0
    unlabeled_mix: float = 0.5
    proximal_mu: float = 0.001
    rampup_rounds: int = 20
    projection_hidden: int = 512
    strong_num_ops: int = 2
    strong_magnitude: int = 10

    def validate(self):
        if self.dataset not in ('pacs_domainnet', 'domainnet', 'vlcs'):
            raise ValueError('Unsupported dataset')
        if self.dirichlet_alpha is None:
            self.dirichlet_alpha = 0.5 if self.dataset == 'pacs_domainnet' else 10.0
        domains = PACS_DOMAINS if self.dataset == 'pacs_domainnet' else (
            DOMAINNET_DOMAINS if self.dataset == 'domainnet' else VLCS_DOMAINS
        )
        if self.dataset == 'pacs_domainnet' and not self.target_domain.startswith('pacs_'):
            self.target_domain = 'pacs_' + self.target_domain
        if self.target_domain not in domains:
            raise ValueError('Target must be one of {}'.format(domains))
        if not self.setting:
            self.setting = 'single' if self.dataset == 'pacs_domainnet' else 'within_domain'
        valid_settings = JOINT_SETTINGS if self.dataset == 'pacs_domainnet' else ('within_domain',)
        if self.setting not in valid_settings:
            raise ValueError('Choose a setting from {}'.format(valid_settings))
        for name in ('rounds', 'local_ep', 'batch_size', 'image_size', 'min_client_samples',
                     'checkpoint_every', 'unlabeled_local_ep', 'rampup_rounds', 'projection_hidden'):
            if getattr(self, name) <= 0:
                raise ValueError('{} must be positive'.format(name))
        if self.num_workers < 0 or self.warmup_rounds < 0:
            raise ValueError('Worker and warm-up counts must be nonnegative')
        for name in ('base_lr', 'anchor_temperature', 'anchor_weight', 'distillation_weight',
                     'semantic_weight', 'dirichlet_alpha'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError('{} must be positive and finite'.format(name))
        if not 0 < self.validation_fraction < 1 or not 0 < self.unlabeled_mix < 1:
            raise ValueError('Validation and unlabeled mixing fractions must be in (0, 1)')
        if not 0 <= self.confidence_threshold <= 1 or self.js_threshold < 0:
            raise ValueError('Invalid consensus thresholds')
        if not 0 <= self.ema_decay < 1:
            raise ValueError('EMA decay must be in [0, 1)')
        return self

    def to_dict(self):
        return asdict(self)


def training_parser():
    parser = argparse.ArgumentParser(description='FedUDG training for Tables II-IV')
    parser.add_argument('--dataset', required=True, choices=('pacs_domainnet', 'domainnet', 'vlcs'))
    parser.add_argument('--target_domain', required=True)
    parser.add_argument('--setting', default='', choices=JOINT_SETTINGS + ('within_domain',))
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--rounds', type=int, default=80)
    parser.add_argument('--local_ep', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--dirichlet_alpha', type=float, default=None)
    parser.add_argument('--resume', default='')
    parser.add_argument('--preflight', action='store_true')
    return parser
