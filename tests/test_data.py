from pathlib import Path
import tempfile
import unittest

from PIL import Image

from fedudg.config import Config, DOMAINNET_DOMAINS, PACS_DOMAINS, SHARED_CLASSES
from fedudg.data import build_protocol, build_target


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        groups = [(cls.root / 'PACS' / 'kfold' / domain[len('pacs_'):], SHARED_CLASSES)
                  for domain in PACS_DOMAINS]
        groups += [(cls.root / 'DomainNet' / domain,
                    tuple('face' if name == 'person' else name for name in SHARED_CLASSES))
                   for domain in DOMAINNET_DOMAINS]
        groups += [(cls.root / 'vlcs' / domain, tuple(str(i) for i in range(5)))
                   for domain in ('CALTECH', 'LABELME', 'PASCAL', 'SUN')]
        for domain, classes in groups:
            for name in classes:
                folder = domain / name
                folder.mkdir(parents=True)
                for i in range(12):
                    Image.new('RGB', (12, 12), (i * 15, 80, 110)).save(folder / '{}.png'.format(i))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def config(self, dataset, target, **values):
        return Config(dataset, target, data_dir=str(self.root), num_workers=0, **values).validate()

    def check_disjoint(self, protocol):
        training = [set(indices) for indices in protocol.clients.values()]
        for i, indices in enumerate(training):
            for other in training[i + 1:]:
                self.assertFalse(indices & other)
        train_paths = {protocol.train.samples[i][0] for indices in training for i in indices}
        val_paths = {record[0] for dataset in protocol.validation.values() for record in dataset.samples}
        self.assertFalse(train_paths & val_paths)

    def test_joint_settings_and_source_only_validation(self):
        for setting, factor in [('single', 1), ('random2', 2), ('random4', 4), ('dirichlet4', 4)]:
            protocol = build_protocol(self.config('pacs_domainnet', 'pacs_photo', setting=setting))
            self.assertEqual(len(protocol.labeled_ids), 3 * factor)
            self.assertEqual(len(protocol.unlabeled_ids), 6 * factor)
            self.assertNotIn('pacs_photo', protocol.client_domains)
            self.assertTrue(all(name.startswith('pacs_') for name in protocol.validation))
            self.check_disjoint(protocol)

    def test_domainnet_role_assignment(self):
        for alpha in (10.0, 1.0, 0.1):
            protocol = build_protocol(self.config('domainnet', 'sketch', dirichlet_alpha=alpha))
            self.assertEqual(len(protocol.labeled_ids), 5)
            self.assertEqual(len(protocol.unlabeled_ids), 10)
            self.assertEqual(protocol.labeled_ids, [0, 3, 6, 9, 12])
            self.assertEqual(protocol.train.classes[-1], 'person')
            self.check_disjoint(protocol)

    def test_vlcs_role_assignment(self):
        protocol = build_protocol(self.config('vlcs', 'sun', dirichlet_alpha=0.1))
        self.assertEqual(len(protocol.labeled_ids), 3)
        self.assertEqual(len(protocol.unlabeled_ids), 6)
        self.assertEqual(protocol.train.num_classes, 5)
        self.check_disjoint(protocol)

    def test_target_uses_all_records_and_one_evaluation_view(self):
        config = self.config('domainnet', 'sketch')
        protocol = build_protocol(config)
        target = build_target(config, protocol.train.classes)
        self.assertEqual(len(target), 7 * 12)
        _, _, image, label = target[0]
        self.assertEqual(tuple(image.shape), (3, 224, 224))
        self.assertEqual(tuple(label.shape), (7,))

    def test_partition_is_reproducible(self):
        config = self.config('vlcs', 'sun', dirichlet_alpha=1.0)
        self.assertEqual(build_protocol(config).clients, build_protocol(config).clients)


if __name__ == '__main__':
    unittest.main()
