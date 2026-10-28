import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .config import DOMAINNET_DOMAINS, PACS_DOMAINS, SHARED_CLASSES, VLCS_DOMAINS


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class TrainingViews:
    def __init__(self, weak, strong):
        self.weak = weak
        self.strong = strong

    def __call__(self, image):
        return self.weak(image), self.weak(image), self.strong(image)


def training_transform(config):
    weak = transforms.Compose([
        transforms.RandomResizedCrop(config.image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    strong = transforms.Compose([
        transforms.RandomResizedCrop(config.image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=config.strong_num_ops, magnitude=config.strong_magnitude),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return TrainingViews(weak, strong)


def evaluation_transform(config):
    return transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def required_root(candidates):
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError('Expected a dataset folder at: {}'.format(
        ', '.join(str(path) for path in candidates)))


def domain_roots(config):
    root = Path(config.data_dir)
    if config.dataset == 'pacs_domainnet':
        pacs = required_root([root / 'PACS' / 'kfold', root / 'PACS', root / 'kfold'])
        domainnet = required_root([root / 'DomainNet', root / 'domainnet'])
        return dict(
            [(domain, pacs / domain[len('pacs_'):]) for domain in PACS_DOMAINS]
            + [('domainnet_' + domain, domainnet / domain) for domain in DOMAINNET_DOMAINS]
        )
    if config.dataset == 'domainnet':
        domainnet = required_root([root / 'DomainNet', root / 'domainnet'])
        return {domain: domainnet / domain for domain in DOMAINNET_DOMAINS}
    vlcs = required_root([root / 'vlcs', root / 'VLCS'])
    return {name: vlcs / folder for name, folder in
            [('caltech', 'CALTECH'), ('labelme', 'LABELME'), ('pascal', 'PASCAL'), ('sun', 'SUN')]}


def image_files(root):
    extensions = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}
    return sorted(path for path in root.rglob('*')
                  if path.is_file() and path.suffix.lower() in extensions)


class DomainDataset(Dataset):
    def __init__(self, roots, domains, transform, seed, validation_fraction,
                 domain_splits, class_names=None):
        self.domains = list(domains)
        self.transform = transform
        self.samples = []
        self.client_domains = []
        self.client_domain_counts = {}
        self.domain_to_indices = defaultdict(list)
        class_to_idx = {name: i for i, name in enumerate(class_names)} if class_names else None
        for domain in self.domains:
            root = roots[domain]
            if not root.is_dir():
                raise FileNotFoundError('Missing domain folder: {}'.format(root))
            per_class = defaultdict(list)
            if class_names is None:
                images = datasets.ImageFolder(str(root))
                if class_to_idx is None:
                    class_to_idx = images.class_to_idx
                elif images.class_to_idx != class_to_idx:
                    raise ValueError('Class folders differ in {}'.format(domain))
                for path, label in images.samples:
                    per_class[label].append((path, label, domain))
            else:
                is_domainnet = domain.startswith('domainnet_') or domain in DOMAINNET_DOMAINS
                for name, label in class_to_idx.items():
                    folder = 'face' if is_domainnet and name == 'person' else name
                    class_root = root / folder
                    if not class_root.is_dir():
                        raise FileNotFoundError('Missing class folder: {}'.format(class_root))
                    for path in image_files(class_root):
                        per_class[label].append((str(path), label, domain))
            rng = random.Random(seed)
            selected = []
            for _, samples in sorted(per_class.items()):
                rng.shuffle(samples)
                cutoff = int(len(samples) * (1.0 - validation_fraction))
                split = domain_splits[domain]
                if split == 'all':
                    selected.extend(samples)
                elif split == 'train':
                    selected.extend(samples[:cutoff])
                elif split == 'test':
                    selected.extend(samples[cutoff:])
                else:
                    raise ValueError('Invalid domain split: {}'.format(split))
            for sample in selected:
                self.domain_to_indices[domain].append(len(self.samples))
                self.samples.append(sample)
        self.class_to_idx = class_to_idx or {}
        self.classes = [name for name, _ in sorted(self.class_to_idx.items(), key=lambda item: item[1])]
        self.num_classes = len(self.classes)

    @classmethod
    def from_records(cls, records, classes, transform, domains):
        instance = cls.__new__(cls)
        instance.samples = list(records)
        instance.classes = list(classes)
        instance.class_to_idx = {name: i for i, name in enumerate(classes)}
        instance.num_classes = len(classes)
        instance.transform = transform
        instance.domains = list(domains)
        instance.domain_to_indices = defaultdict(list)
        for index, (_, _, domain) in enumerate(instance.samples):
            instance.domain_to_indices[domain].append(index)
        return instance

    def summarize_client_domains(self, dict_users):
        return {
            client: dict(sorted(Counter(self.samples[int(i)][2] for i in indices).items()))
            for client, indices in sorted(dict_users.items())
        }

    def build_client_splits(self, client_domains, seed=1337, partition='random', dirichlet_alpha=0.5, min_client_samples=16):
        if partition not in ('random', 'dirichlet', 'dirichlet_standard'):
            raise ValueError('Unknown client partition: {}'.format(partition))
        if partition in ('dirichlet', 'dirichlet_standard') and dirichlet_alpha <= 0:
            raise ValueError('dirichlet_alpha must be positive')
        self.client_domains = list(client_domains)
        rng = random.Random(seed)
        dict_users = {}
        domain_to_clients = defaultdict(list)
        for (client_id, domain) in enumerate(self.client_domains):
            domain_to_clients[domain].append(client_id)
        for (domain, client_ids) in domain_to_clients.items():
            indices = list(self.domain_to_indices[domain])
            if len(indices) < len(client_ids):
                raise ValueError('Domain {} has {} training samples for {} clients'.format(domain, len(indices), len(client_ids)))
            if partition == 'random' or len(client_ids) == 1:
                rng.shuffle(indices)
                for (position, client_id) in enumerate(client_ids):
                    dict_users[client_id] = set(indices[position::len(client_ids)])
                continue
            if partition == 'dirichlet_standard':
                import numpy as np
                np_rng = np.random.RandomState(rng.randrange(2 ** 31))
                minimum = int(min_client_samples)
                if minimum < 1 or len(indices) < minimum * len(client_ids):
                    raise ValueError('Insufficient samples for Dirichlet minimum in {}'.format(domain))
                by_class = defaultdict(list)
                for index in indices:
                    by_class[self.samples[index][1]].append(index)
                for attempt in range(10000):
                    buckets = [[] for _ in client_ids]
                    for label in sorted(by_class):
                        class_indices = np_rng.permutation(by_class[label])
                        proportions = np_rng.dirichlet([dirichlet_alpha] * len(client_ids))
                        sizes = np_rng.multinomial(len(class_indices), proportions)
                        for (position, chunk) in enumerate(np.split(class_indices, np.cumsum(sizes)[:-1])):
                            buckets[position].extend((int(value) for value in chunk))
                    if min(map(len, buckets)) >= minimum:
                        break
                else:
                    raise RuntimeError('Dirichlet allocation failed after 10000 attempts in {}'.format(domain))
                for (client_id, bucket) in zip(client_ids, buckets):
                    dict_users[client_id] = set(bucket)
                continue
            class_to_indices = defaultdict(list)
            for index in indices:
                class_to_indices[self.samples[index][1]].append(index)
            client_count = len(client_ids)
            capacities = [len(indices) // client_count + int(position < len(indices) % client_count) for position in range(client_count)]
            buckets = [set() for _ in client_ids]
            class_ids = sorted(class_to_indices)
            rng.shuffle(class_ids)
            for class_id in class_ids:
                class_indices = class_to_indices[class_id]
                rng.shuffle(class_indices)
                preferences = [rng.gammavariate(dirichlet_alpha, 1.0) for _ in client_ids]
                for index in class_indices:
                    weights = [preference * capacity for (preference, capacity) in zip(preferences, capacities)]
                    if not any(weights):
                        weights = list(capacities)
                    position = rng.choices(range(client_count), weights=weights, k=1)[0]
                    buckets[position].add(index)
                    capacities[position] -= 1
            if any(capacities):
                raise RuntimeError('Dirichlet partition failed to allocate every sample')
            for (client_id, bucket) in zip(client_ids, buckets):
                dict_users[client_id] = bucket
        for client_id in range(len(self.client_domains)):
            dict_users.setdefault(client_id, set())
        self.client_domain_counts = self.summarize_client_domains(dict_users)
        return dict_users

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, domain = self.samples[index]
        with Image.open(path) as source:
            image = source.convert('RGB')
        image = self.transform(image)
        target = torch.zeros(self.num_classes)
        target[label] = 1.0
        return '{}:{}'.format(domain, path), index, image, target


class ClientDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


@dataclass
class ProtocolData:
    train: DomainDataset
    clients: dict
    client_domains: list
    labeled_ids: list
    unlabeled_ids: list
    validation: dict
    partition: str

    def manifest(self):
        heldout = {domain: len(data) for domain, data in self.validation.items()}
        return {
            'partition': self.partition,
            'classes': self.train.classes,
            'validation_samples_by_domain': heldout,
            'clients': [
                {'id': i, 'domain': domain,
                 'role': 'labeled' if i in self.labeled_ids else 'unlabeled',
                 'samples': len(self.clients[i]),
                 'class_counts': dict(sorted(Counter(
                     int(self.train.samples[j][1]) for j in self.clients[i]).items()))}
                for i, domain in enumerate(self.client_domains)
            ],
        }


def build_protocol(config):
    roots = domain_roots(config)
    joint = config.dataset == 'pacs_domainnet'
    all_domains = list(PACS_DOMAINS) + ['domainnet_' + d for d in DOMAINNET_DOMAINS] if joint else (
        list(DOMAINNET_DOMAINS) if config.dataset == 'domainnet' else list(VLCS_DOMAINS))
    sources = [domain for domain in all_domains if domain != config.target_domain]
    classes = SHARED_CLASSES if config.dataset != 'vlcs' else None
    if joint:
        factor = {'single': 1, 'random2': 2, 'random4': 4, 'dirichlet4': 4}[config.setting]
        client_domains = [domain for domain in sources for _ in range(factor)]
        labeled = [i for i, domain in enumerate(client_domains) if domain.startswith('pacs_')]
        partition = 'dirichlet' if config.setting == 'dirichlet4' else 'random'
        domain_splits = {domain: 'train' if domain.startswith('pacs_') else 'all' for domain in sources}
    else:
        client_domains = [domain for domain in sources for _ in range(3)]
        labeled = list(range(0, len(client_domains), 3))
        partition = 'dirichlet_standard'
        domain_splits = {domain: 'all' for domain in sources}
    train = DomainDataset(roots, sources, training_transform(config), config.seed,
                          config.validation_fraction, domain_splits, classes)
    expected_classes = 5 if config.dataset == 'vlcs' else 7
    if train.num_classes != expected_classes:
        raise ValueError('Expected {} classes, found {}'.format(expected_classes, train.classes))
    clients = train.build_client_splits(
        client_domains, seed=config.seed, partition=partition,
        dirichlet_alpha=config.dirichlet_alpha, min_client_samples=config.min_client_samples)
    validation_records = []
    if joint:
        labeled_domains = [domain for domain in sources if domain.startswith('pacs_')]
        validation_data = DomainDataset(
            roots, labeled_domains, evaluation_transform(config), config.seed,
            config.validation_fraction, {domain: 'test' for domain in labeled_domains}, classes)
        validation_records = validation_data.samples
    else:
        labeled_domains = sources
        held_out = []
        for client in labeled:
            rng = np.random.RandomState(config.seed + 104729 * (client + 1))
            by_class = {}
            for index in sorted(clients[client]):
                by_class.setdefault(int(train.samples[index][1]), []).append(index)
            selected = []
            for _, indices in sorted(by_class.items()):
                indices = rng.permutation(indices).tolist()
                count = min(len(indices) - 1, max(1, int(len(indices) * config.validation_fraction))) if len(indices) > 1 else 0
                selected.extend(indices[:count])
            if not selected:
                raise ValueError('Labeled client {} has no validation samples'.format(client))
            clients[client] -= set(selected)
            held_out.extend(selected)
        validation_records = [train.samples[i] for i in sorted(held_out)]
    validation = {
        domain: DomainDataset.from_records(
            [record for record in validation_records if record[2] == domain],
            train.classes, evaluation_transform(config), [domain])
        for domain in labeled_domains
    }
    if any(not indices for indices in clients.values()) or any(not len(ds) for ds in validation.values()):
        raise ValueError('Every training client and validation domain must contain samples')
    unlabeled = [i for i in range(len(client_domains)) if i not in labeled]
    return ProtocolData(train, clients, client_domains, labeled, unlabeled, validation, partition)


def build_target(config, expected_classes):
    classes = SHARED_CLASSES if config.dataset != 'vlcs' else None
    target = DomainDataset(
        domain_roots(config), [config.target_domain], evaluation_transform(config), config.seed,
        config.validation_fraction, {config.target_domain: 'all'}, classes)
    if list(target.classes) != list(expected_classes):
        raise ValueError('Target and checkpoint class vocabularies differ')
    if not len(target):
        raise ValueError('The target dataset is empty')
    return target

