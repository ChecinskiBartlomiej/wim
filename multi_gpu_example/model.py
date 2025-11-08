"""
Simple CNN model for multi-GPU training demonstration.
A basic ResNet-like model for CIFAR-10 classification.
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network for CIFAR-10 (32x32 RGB images, 10 classes).

    This is intentionally simple to focus on multi-GPU training concepts,
    not state-of-the-art performance.
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # First block: 3 -> 64 channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16
        )

        # Second block: 64 -> 128 channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8
        )

        # Third block: 128 -> 256 channels
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8 -> 4x4
        )

        # Global average pooling instead of flatten - reduces parameters
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor [batch_size, 3, 32, 32]

        Returns:
            logits: Output tensor [batch_size, num_classes]
        """
        x = self.conv1(x)       # [B, 64, 16, 16]
        x = self.conv2(x)       # [B, 128, 8, 8]
        x = self.conv3(x)       # [B, 256, 4, 4]
        x = self.global_pool(x) # [B, 256, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # [B, 256]
        x = self.fc(x)          # [B, num_classes]
        return x


def count_parameters(model):
    """Count total trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    model = SimpleCNN(num_classes=10)
    x = torch.randn(4, 3, 32, 32)  # Batch of 4 CIFAR-10 images
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {count_parameters(model):,}")
