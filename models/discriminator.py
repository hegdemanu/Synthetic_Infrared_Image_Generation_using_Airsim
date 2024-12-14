import torch
import torch.nn as nn

class DilatedConvBlock(nn.Module):
    """Dilated convolution block with instance normalization."""
    def __init__(self, in_channels, out_channels, dilation_rate=1):
        super(DilatedConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1,
            padding=dilation_rate,  # Maintain spatial dimensions
            dilation=dilation_rate
        )
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.lrelu(self.instance_norm(self.conv(x)))

class Discriminator(nn.Module):
    """Discriminator network with dilated convolutions for spatial hierarchy preservation."""
    def __init__(self, input_channels=3, ndf=64):
        super(Discriminator, self).__init__()
        
        # Initial convolution without instance normalization
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Dilated convolution layers with increasing dilation rates
        self.dilated1 = DilatedConvBlock(ndf, ndf * 2, dilation_rate=1)
        self.dilated2 = DilatedConvBlock(ndf * 2, ndf * 4, dilation_rate=2)
        self.dilated3 = DilatedConvBlock(ndf * 4, ndf * 8, dilation_rate=4)

        # Standard convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # PatchGAN classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()  # For binary classification
        )

    def forward(self, x):
        # Initial convolution
        x = self.initial(x)
        
        # Dilated convolutions
        x = self.dilated1(x)
        x = self.dilated2(x)
        x = self.dilated3(x)
        
        # Standard convolution
        x = self.conv1(x)
        
        # PatchGAN classification
        x = self.classifier(x)
        
        return x

    def init_weights(self, init_type='normal', gain=0.02):
        """Initialize network weights."""
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)
