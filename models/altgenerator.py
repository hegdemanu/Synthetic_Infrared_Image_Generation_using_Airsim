# models/

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Residual block with instance normalization."""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    """Generator network with U-Net architecture."""
    def __init__(self, input_channels=3, output_channels=3, num_filters=64, num_residuals=9):
        super(Generator, self).__init__()
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, 7, 1, 3),
            nn.InstanceNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )

        # Encoding blocks
        self.down1 = self._make_encoder_block(num_filters, num_filters * 2)
        self.down2 = self._make_encoder_block(num_filters * 2, num_filters * 4)
        self.down3 = self._make_encoder_block(num_filters * 4, num_filters * 8)

        # Residual blocks
        self.residuals = nn.Sequential(*[
            ResidualBlock(num_filters * 8) for _ in range(num_residuals)
        ])

        # Decoding blocks with skip connections
        self.up3 = self._make_decoder_block(num_filters * 8, num_filters * 4)
        self.up2 = self._make_decoder_block(num_filters * 4, num_filters * 2)
        self.up1 = self._make_decoder_block(num_filters * 2, num_filters)

        # Final convolution
        self.final = nn.Sequential(
            nn.Conv2d(num_filters, output_channels, 7, 1, 3),
            nn.Tanh()
        )

    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Initial convolution
        x1 = self.initial(x)
        
        # Encoder
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Residual blocks
        x4 = self.residuals(x4)
        
        # Decoder with skip connections
        x = self.up3(x4) + x3
        x = self.up2(x) + x2
        x = self.up1(x) + x1
        
        # Final convolution
        return self.final(x)
