import torch
from torch import nn
import torch.nn.functional as F

# --- Residual Block ---
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.down = None
        if stride != 1 or in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.block(x)
        if self.down is not None:
            identity = self.down(x)
        out += identity
        out = self.relu(out)
        return out

# --- Convolution Block ---
class ConvBlock(nn.Module):
    """
    A generic convolution block that applies two conv-BN-ReLU layers.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

# --- Encoder Block ---
class EncoderBlock(nn.Module):
    """
    Encoder block that downsamples the input with a residual block and then applies additional
    residual blocks at the same resolution. It also saves the block input as a skip connection.
    """
    def __init__(self, in_channels, out_channels, n_blocks):
        super(EncoderBlock, self).__init__()
        layers = []
        # First block downsamples (stride=2)
        layers.append(ResNetBlock(in_channels, out_channels, stride=2))
        # Additional blocks keep the same resolution (stride=1)
        for _ in range(1, n_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, stride=1))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        skip = x  # store the input as skip connection
        out = self.block(x)
        return out, skip

# --- Decoder Block ---
class DecoderBlock(nn.Module):
    """
    Decoder block that upsamples, concatenates a skip connection,
    and then applies a convolution block.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels,
                                     kernel_size=2, stride=2)
        # After concatenation, the number of channels is (out_channels + skip_channels)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)
        
    def forward(self, x, skip):
        x = self.up(x)
        # Adjust spatial size if necessary.
        if x.size()[2:] != skip.size()[2:]:
            x = F.interpolate(x, size=skip.size()[2:], mode='bilinear',
                              align_corners=True)
        # Concatenate along the channel dimension.
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x

# --- ResNet34_UNet Model ---
class ResNet34_UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet34_UNet, self).__init__()
        # Stem
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # Encoder blocks
        self.encoder2 = EncoderBlock(64, 64, n_blocks=3)    # Skip from encoder2: 64 channels
        self.encoder3 = EncoderBlock(64, 128, n_blocks=4)     # Skip from encoder3: 64 channels (input)
        self.encoder4 = EncoderBlock(128, 256, n_blocks=6)    # Skip from encoder4: 128 channels
        self.encoder5 = EncoderBlock(256, 512, n_blocks=3)    # Skip from encoder5: 256 channels
        
        # Center block: a simple conv-BN-ReLU sequence.
        self.center = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Decoder blocks with corrected channel parameters:
        # Decoder4: center (256) and skip4 (256) → out: 128 channels
        self.decoder4 = DecoderBlock(in_channels=256, skip_channels=256, out_channels=128)
        # Decoder3: dec4 (128) and skip3 (128) → out: 64 channels
        self.decoder3 = DecoderBlock(in_channels=128, skip_channels=128, out_channels=64)
        # Decoder2: dec3 (64) and skip2 (64) → out: 32 channels
        self.decoder2 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=32)
        # Decoder1: dec2 (32) and skip1 (64) → out: 32 channels
        self.decoder1 = DecoderBlock(in_channels=32, skip_channels=64, out_channels=32)
        
        # Output block: further upsample to the desired output resolution.
        self.output = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            ConvBlock(32, 32),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            ConvBlock(32, 32),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )
    
    def forward(self, x):
        # Encoder pathway
        enc1 = self.encoder1(x)                     # Stem output: 64 channels.
        enc2, skip1 = self.encoder2(enc1)             # Skip1: 64 channels.
        enc3, skip2 = self.encoder3(enc2)             # Skip2: 64 channels.
        enc4, skip3 = self.encoder4(enc3)             # Skip3: 128 channels.
        enc5, skip4 = self.encoder5(enc4)             # Skip4: 256 channels.
        # Center block.
        center = self.center(enc5)                    # Center output: 256 channels.
        
        # Decoder pathway.
        dec4 = self.decoder4(center, skip4)           # Combines center with skip4.
        dec3 = self.decoder3(dec4, skip3)             # Combines dec4 with skip3.
        dec2 = self.decoder2(dec3, skip2)             # Combines dec3 with skip2.
        dec1 = self.decoder1(dec2, skip1)             # Combines dec2 with skip1.
        
        return self.output(dec1)
