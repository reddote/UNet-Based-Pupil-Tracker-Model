import torch.nn as nn
from torchvision import models


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        # Initial downsampling layer
        # self.initial_conv = nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1)  # 384x288 to 192x144 ( 224x168 )
        # Load pre-trained EfficientNetB0
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # self.vit_net = models.vit_h_14(weights=models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)  # resize 518
        # Replace the final layer with a regression layer for 5 outputs
        # Output 5 values = center_x, center_y, angle, width, height
        self.efficientnet.classifier = nn.Sequential(
            nn.Linear(self.efficientnet.classifier[1].in_features, 5)
        )

    def forward(self, x):
        # x = self.initial_conv(x)
        x = self.efficientnet(x)
        return x