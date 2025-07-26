import torch
import torch.nn as nn

# Basic block: Conv + BatchNorm + ReLU with 3x3 kernel
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# Encoder: ConvBlock + Downsampling using strided convolution
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x_down = self.down(x)
        return x, x_down

# Decoder: Upsampling + Skip Connection + ConvBlock
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)  # input is upsampled + skip (concat)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

# Main U-Net model
class UNet384x288(nn.Module):
    def __init__(self):
        super().__init__()

        # Stem block (entry feature extractor)
        self.stem = ConvBlock(3, 16)  # input RGB image (3 channels) → 16 feature channels

        # Encoder path
        self.enc1 = EncoderBlock(16, 32)    # 384x288 → 192x144
        self.enc2 = EncoderBlock(32, 64)    # 192x144 → 96x72
        self.enc3 = EncoderBlock(64, 128)   # 96x72 → 48x36
        self.enc4 = EncoderBlock(128, 256)  # 48x36 → 24x18

        # Bottleneck block
        self.bottleneck = ConvBlock(256, 512)

        # Decoder path
        self.dec4 = DecoderBlock(512, 256)  # 24x18 → 48x36
        self.dec3 = DecoderBlock(256, 128)  # 48x36 → 96x72
        self.dec2 = DecoderBlock(128, 64)   # 96x72 → 192x144
        self.dec1 = DecoderBlock(64, 32)    # 192x144 → 384x288

        # Final 1x1 Conv layer to produce output
        self.final = nn.Conv2d(32, 2, kernel_size=1)  # 2 output classes (e.g., background + pupil)

        # Output refine block to clean/polish the segmentation output with 3x3 kernel
        self.output_refine = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 2, kernel_size=1)
        )

    def forward(self, x):
        x = self.stem(x)             # Input → 16 channels

        x1, x = self.enc1(x)         # 16 → 32 → 192x144
        x2, x = self.enc2(x)         # 32 → 64 → 96x72
        x3, x = self.enc3(x)         # 64 → 128 → 48x36
        x4, x = self.enc4(x)         # 128 → 256 → 24x18

        x = self.bottleneck(x)       # 256 → 512

        x = self.dec4(x, x4)         # Decoder with skip connections
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)

        x = self.final(x)            # Final output layer (2 classes)
        x = self.output_refine(x)    # Refine the output (optional but helpful)

        return x
