"""
Model Architectures for Baseline and Transfer Learning.
Supports:
1. BaselineCNN: 4-block ConvNet trained from scratch
2. MobileNetV3-Large (ImageNet-pretrained)
3. EfficientNet-B0 (ImageNet-pretrained)
4. ResNet50 (ImageNet-pretrained)
5. ConvNeXt-Tiny (ImageNet-pretrained)
"""

import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    resnet50, ResNet50_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights
)


class BaselineCNN(nn.Module):
    """Small custom 4-block CNN trained from scratch as a benchmark baseline."""
    def __init__(self, num_classes=37):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 112x112
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 56x56
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 28x28
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)) # 1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_model(model_name: str, num_classes: int = 37, pretrained: bool = True):
    """Factory creating and configuring architectures for transfer learning."""
    model_name = model_name.lower().replace("-", "_")

    if model_name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes)

    elif model_name == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model

    elif model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    elif model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    elif model_name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        return model

    else:
        raise ValueError(f"Unknown model architecture: {model_name}")


def freeze_backbone(model, model_name: str, freeze: bool = True):
    """Freezes or unfreezes backbone parameters for two-stage transfer learning."""
    model_name = model_name.lower().replace("-", "_")

    if model_name == "baseline_cnn":
        return  # No freezing for baseline from scratch

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = not freeze

    # Ensure head parameters always require gradients
    if model_name == "mobilenet_v3_large":
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif model_name == "efficientnet_b0":
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif model_name == "resnet50":
        for param in model.fc.parameters():
            param.requires_grad = True
    elif model_name == "convnext_tiny":
        for param in model.classifier.parameters():
            param.requires_grad = True
