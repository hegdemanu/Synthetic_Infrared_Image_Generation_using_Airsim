import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    """A basic U-Net block consisting of Conv-InstanceNorm-ReLU."""
    def __init__(self, in_channels, out_channels, use_instance_norm=True):
        super(UNetBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        if use_instance_norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class ResnetBlock(nn.Module):
    """Residual block that preserves spatial dimensions."""
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    """U-Net based generator with skip connections and residual blocks."""
    def __init__(self, input_channels=3, output_channels=3, num_filters=64, num_resnet_blocks=9):
        super(Generator, self).__init__()
        
        # Initial convolution
        self.initial = UNetBlock(input_channels, num_filters, use_instance_norm=False)
        
        # Encoding path
        self.down1 = UNetBlock(num_filters, num_filters * 2)
        self.down2 = UNetBlock(num_filters * 2, num_filters * 4)
        self.down3 = UNetBlock(num_filters * 4, num_filters * 8)
        
        # Residual blocks
        resnet_blocks = []
        for _ in range(num_resnet_blocks):
            resnet_blocks.append(ResnetBlock(num_filters * 8))
        self.resnet_blocks = nn.Sequential(*resnet_blocks)
        
        # Decoding path with skip connections
        self.up3 = UNetBlock(num_filters * 16, num_filters * 4)  # *16 due to skip connection
        self.up2 = UNetBlock(num_filters * 8, num_filters * 2)   # *8 due to skip connection
        self.up1 = UNetBlock(num_filters * 4, num_filters)       # *4 due to skip connection
        
        # Final convolution
        self.final = nn.Sequential(
            nn.Conv2d(num_filters * 2, output_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # Upsampling layer
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Initial convolution
        x1 = self.initial(x)
        
        # Encoding path
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Residual blocks
        x4 = self.resnet_blocks(x4)
        
        # Decoding path with skip connections
        x = self.upsample(x4)
        x = self.up3(torch.cat([x, x3], dim=1))
        x = self.upsample(x)
        x = self.up2(torch.cat([x, x2], dim=1))
        x = self.upsample(x)
        x = self.up1(torch.cat([x, x1], dim=1))
        
        # Final convolution
        x = self.final(torch.cat([x, x1], dim=1))
        
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
