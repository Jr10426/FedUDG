import copy
import unittest

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from fedudg.config import Config
from fedudg.core import (
    aggregate_models, average_anchors, geometric_mean_target, semantic_admission,
    semantic_anchor_logits, soft_cross_entropy, update_ema_anchors, update_ema_model,
)
from fedudg.training import (
    learning_rate, masked_semantic_loss, train_labeled_client, train_unlabeled_client,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 8)
        self.classifier = nn.Linear(8, 3)

    def forward(self, images):
        features = self.encoder(images.mean(dim=(2, 3)))
        return features, self.classifier(features)


class TinyDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        image = torch.full((3, 8, 8), (index + 1) / 4)
        label = F.one_hot(torch.tensor(index % 3), 3).float()
        return str(index), index, (image, image.clone(), image.clone()), label


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1337)

    def test_soft_target_is_a_distribution(self):
        probabilities = [torch.tensor([[0.9, 0.1], [0.2, 0.8]]),
                         torch.tensor([[0.8, 0.2], [0.1, 0.9]])]
        target = geometric_mean_target(probabilities)
        torch.testing.assert_close(target.sum(dim=1), torch.ones(2))
        self.assertTrue(bool((target > 0).all()))

    def test_consensus_requires_all_three_predictions(self):
        first = torch.tensor([[0.99, 0.01], [0.99, 0.01]])
        second = torch.tensor([[0.99, 0.01], [0.01, 0.99]])
        mask, target, js, _ = semantic_admission(first, second, first)
        self.assertEqual(mask.tolist(), [True, False])
        self.assertEqual(tuple(target.shape), (2, 2))
        self.assertAlmostEqual(float(js[0]), 0.0)

    def test_anchor_prediction_has_no_feature_scale_dependence(self):
        features, anchors = torch.randn(4, 8), torch.randn(3, 8)
        torch.testing.assert_close(semantic_anchor_logits(features, anchors),
                                   semantic_anchor_logits(5 * features, anchors))

    def test_soft_cross_entropy_matches_manual_expression(self):
        logits = torch.randn(4, 3)
        targets = torch.softmax(torch.randn(4, 3), dim=1)
        torch.testing.assert_close(soft_cross_entropy(logits, targets),
                                   -(targets * F.log_softmax(logits, dim=1)).sum(dim=1))

    def test_empty_admission_keeps_a_zero_differentiable_loss(self):
        values = torch.randn(4, requires_grad=True)
        loss = masked_semantic_loss(values, torch.zeros(4, dtype=torch.bool))
        loss.backward()
        self.assertEqual(float(loss), 0.0)
        torch.testing.assert_close(values.grad, torch.zeros(4))

    def test_group_aggregation_updates_encoder_and_classifier(self):
        left = {'encoder.weight': torch.tensor([2.0]), 'classifier.weight': torch.tensor([4.0]),
                'counter': torch.tensor(7)}
        right = {'encoder.weight': torch.tensor([6.0]), 'classifier.weight': torch.tensor([8.0]),
                 'counter': torch.tensor(20)}
        result = aggregate_models([left], [right], 0.5)
        self.assertEqual(float(result['encoder.weight']), 4.0)
        self.assertEqual(float(result['classifier.weight']), 6.0)
        self.assertEqual(int(result['counter']), 7)

    def test_anchors_are_normalized_after_group_averaging(self):
        anchors = average_anchors([torch.randn(3, 8), torch.randn(3, 8)])
        torch.testing.assert_close(anchors.norm(dim=1), torch.ones(3))

    def test_teacher_and_anchor_updates(self):
        model = TinyModel()
        teacher = copy.deepcopy(model)
        previous = {key: value.clone() for key, value in teacher.state_dict().items()}
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1)
        update_ema_model(teacher, model, 0.75)
        for key, value in teacher.state_dict().items():
            torch.testing.assert_close(value, previous[key] + 0.25)
        anchors = F.normalize(torch.randn(3, 8), dim=1)
        update_ema_anchors(anchors, F.normalize(torch.randn(3, 8), dim=1), 0.75)
        torch.testing.assert_close(anchors.norm(dim=1), torch.ones(3))

    def test_labeled_client_updates_model_and_anchors(self):
        config = Config('vlcs', 'sun', local_ep=1, num_workers=0)
        model = TinyModel()
        anchors = F.normalize(model.classifier.weight.detach().clone(), dim=1)
        state, updated_anchors, optimizer, metrics = train_labeled_client(
            config, DataLoader(TinyDataset(), batch_size=2), model, anchors, None, 0.01)
        self.assertFalse(torch.equal(state['classifier.weight'], model.classifier.weight))
        self.assertFalse(torch.equal(updated_anchors, anchors))
        self.assertEqual(metrics['batches'], 2)
        self.assertTrue(optimizer['state'])

    def test_unlabeled_client_keeps_teacher_and_anchors_fixed(self):
        config = Config('vlcs', 'sun', num_workers=0)
        model = TinyModel()
        teacher = copy.deepcopy(model)
        anchors = F.normalize(model.classifier.weight.detach().clone(), dim=1)
        old_anchors = anchors.clone()
        old_teacher = {key: value.clone() for key, value in teacher.state_dict().items()}
        state, projection, optimizer, metrics = train_unlabeled_client(
            config, DataLoader(TinyDataset(), batch_size=2), model, teacher, anchors,
            None, None, 0.01, 1.0)
        self.assertFalse(torch.equal(state['encoder.weight'], model.encoder.weight))
        self.assertTrue(projection)
        self.assertTrue(optimizer['state'])
        self.assertGreaterEqual(metrics['admission_rate'], 0.0)
        torch.testing.assert_close(anchors, old_anchors)
        for key, value in teacher.state_dict().items():
            torch.testing.assert_close(value, old_teacher[key])

    def test_learning_rate_changes_at_eighty_percent(self):
        config = Config('vlcs', 'sun')
        self.assertEqual(learning_rate(config, 63), 0.001)
        self.assertEqual(learning_rate(config, 64), 0.0001)

    def test_protocol_defaults(self):
        self.assertEqual(Config('pacs_domainnet', 'photo').validate().target_domain, 'pacs_photo')
        self.assertEqual(Config('domainnet', 'sketch').validate().setting, 'within_domain')
        self.assertEqual(Config('domainnet', 'sketch').validate().dirichlet_alpha, 10.0)
        with self.assertRaises(ValueError):
            Config('vlcs', 'sun', setting='random2').validate()


if __name__ == '__main__':
    unittest.main()
