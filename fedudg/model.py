import torch
from torch import nn
from torchvision import models


class ResNet18(nn.Module):

    def __init__(self, out_size, drop_rate=0.2, pretrained=True):
        super(ResNet18, self).__init__()
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
        except (AttributeError, TypeError):
            backbone = models.resnet18(pretrained=bool(pretrained))
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.pool = backbone.avgpool
        self.drop_layer = nn.Dropout(p=drop_rate)
        self.classifier = nn.Linear(backbone.fc.in_features, out_size)

    def forward(self, x):
        for stage in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            x = stage(x)
        features = self.pool(x)
        features = torch.flatten(features, 1)
        features = self.drop_layer(features)
        logits = self.classifier(features)
        return (features, logits)

