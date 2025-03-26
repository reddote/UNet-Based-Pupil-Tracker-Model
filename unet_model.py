import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x_down = self.down(x)
        return x, x_down


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNet384x288(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = EncoderBlock(3, 16)     # 384x288 → 192x144
        self.enc2 = EncoderBlock(16, 32)    # 192x144 → 96x72
        self.enc3 = EncoderBlock(32, 64)    # 96x72 → 48x36
        self.enc4 = EncoderBlock(64, 128)   # 48x36 → 24x18

        self.bottleneck = ConvBlock(128, 256)

        self.dec4 = DecoderBlock(256, 128)  # 24x18 → 48x36
        self.dec3 = DecoderBlock(128, 64)   # 48x36 → 96x72
        self.dec2 = DecoderBlock(64, 32)    # 96x72 → 192x144
        self.dec1 = DecoderBlock(32, 16)    # 192x144 → 384x288

        self.final = nn.Conv2d(16, 2, kernel_size=1)  # 2 sınıf

    def forward(self, x):
        x1, x = self.enc1(x)
        x2, x = self.enc2(x)
        x3, x = self.enc3(x)
        x4, x = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec4(x, x4)
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)

        return self.final(x)
