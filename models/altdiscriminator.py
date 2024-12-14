# models/
import torch
import torch.nn as nn

class DilatedConvBlock(nn.Module):
    """Dilated convolution block with instance normalization."""
    def __init__(self, in_channels, out_channels, dilation_rate):
        super(DilatedConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1,
            padding=dilation_rate,
            dilation=dilation_rate
        )
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

class Discriminator(nn.Module):
    """Discriminator network with dilated convolutions."""
    def __init__(self, input_channels=3, num_filters=64):
        super(Discriminator, self).__init__()

        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Dilated convolutions with different dilation rates
        self.dilated1 = DilatedConvBlock(num_filters, num_filters * 2, dilation_rate=1)
        self.dilated2 = DilatedConvBlock(num_filters * 2, num_filters * 4, dilation_rate=2)
        self.dilated3 = DilatedConvBlock(num_filters * 4, num_filters * 8, dilation_rate=4)

        # Standard convolutions
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1),
            nn.InstanceNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # PatchGAN output
        self.conv2 = nn.Conv2d(num_filters * 8, 1, 4, 1, 1)

    def forward(self, x):
        # Initial convolution
        x = self.initial(x)
        
        # Dilated convolutions
        x = self.dilated1(x)
        x = self.dilated2(x)
        x = self.dilated3(x)
        
        # Standard convolutions
        x = self.conv1(x)
        
        # PatchGAN output
        return self.conv2(x)
