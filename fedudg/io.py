import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def atomic_json(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp-{}'.format(os.getpid()))
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    os.replace(str(temporary), str(path))


def atomic_checkpoint(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp-{}'.format(os.getpid()))
    torch.save(value, str(temporary))
    os.replace(str(temporary), str(path))


def load_checkpoint(path):
    try:
        return torch.load(str(path), map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location='cpu')


def rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state.get('cuda'):
        torch.cuda.set_rng_state_all(state['cuda'])
