import torch.nn as nn
from torchvision import models


class EfficientNetB0Regression(nn.Module):
    def __init__(self, num_classes=2):
        super(EfficientNetB0Regression, self).__init__()
        # Load pre-trained EfficientNetB0
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Remove the classifier layer
        self.features = self.efficientnet.features  # Extract features only
        # Add a segmentation head for pixel-wise classification
        self.segmentation_head = nn.Conv2d(1280, num_classes, kernel_size=1)  # 1280 comes from EfficientNet's last feature map

    def forward(self, x):
        # Extract features using EfficientNet
        x = self.features(x)  # Shape: [batch_size, 1280, H/32, W/32]
        # Apply the segmentation head
        x = self.segmentation_head(x)  # Shape: [batch_size, num_classes, H/32, W/32]
        # Upsample to match the input dimensions
        x = nn.functional.interpolate(x, size=(288, 384), mode="bilinear", align_corners=False)  # Shape: [batch_size, num_classes, 288, 384]
        return x