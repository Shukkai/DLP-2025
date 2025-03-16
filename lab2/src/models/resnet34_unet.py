import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.down  = None
        if stride != 1 or in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        out += identity
        return self.relu(out)

class EncoderBlock(nn.Module):
    """
    Applies a series of ResNetBlocks, where the first block downsamples the input.
    Returns:
      - out: the output after the block (downsampled features)
      - skip: the original input to be used as a skip connection.
    """
    def __init__(self, in_channels, out_channels, n_blocks):
        super(EncoderBlock, self).__init__()
        layers = []
        # Downsample in the first block (stride=2)
        layers.append(ResNetBlock(in_channels, out_channels, stride=2))
        # The remaining blocks keep the same resolution (stride=1)
        for _ in range(1, n_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, stride=1))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        skip = x  # store input for skip connection
        out = self.block(x)
        return out, skip

class DecoderBlock(nn.Module):
    """
    Upsamples the input features and fuses them with the corresponding encoder skip.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # Upsample the low-resolution feature map
        self.up = nn.ConvTranspose2d(in_channels, out_channels,
                                     kernel_size=2, stride=2)
        # After upsampling, concatenate with skip (channels = skip_channels + out_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(skip_channels + out_channels, out_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip):
        x = self.up(x)
        # Ensure skip and x have the same spatial dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class ResNet34_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        """
        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of output segmentation channels.
                                For binary segmentation, use 1.
        """
        super(ResNet34_UNet, self).__init__()
        # Stem: similar to ResNet's initial layers.
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )  # Output: (B, 64, 256/4, 256/4) = (B, 64, 64, 64)
        
        # Encoder blocks.
        self.encoder2 = EncoderBlock(64, 64, n_blocks=3)    # Output: (B, 64, 32, 32)
        self.encoder3 = EncoderBlock(64, 128, n_blocks=4)     # Output: (B, 128, 16, 16)
        self.encoder4 = EncoderBlock(128, 256, n_blocks=6)    # Output: (B, 256, 8, 8)
        self.encoder5 = EncoderBlock(256, 512, n_blocks=3)    # Output: (B, 512, 4, 4)
        
        # Center block
        self.center = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )  # Keeps resolution 4×4
        
        # Decoder blocks.
        self.decoder4 = DecoderBlock(in_channels=256, skip_channels=512, out_channels=32)  # Upsample from 4x4 to 8x8
        self.decoder3 = DecoderBlock(in_channels=32,  skip_channels=256, out_channels=32)  # 8x8 -> 16x16
        self.decoder2 = DecoderBlock(in_channels=32,  skip_channels=128, out_channels=32)  # 16x16 -> 32x32
        self.decoder1 = DecoderBlock(in_channels=32,  skip_channels=64,  out_channels=32)  # 32x32 -> 64x64
        
        # Final output block to upsample from 64×64 to 256×256.
        self.output = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),  # 64x64 -> 128x128
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),  # 128x128 -> 256x256
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)           # Final segmentation map
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)              # (B, 64, 64, 64)
        enc2, skip1 = self.encoder2(enc1)      # (B, 64, 32, 32); skip1: (B, 64, 64, 64)
        enc3, skip2 = self.encoder3(enc2)      # (B, 128, 16, 16); skip2: (B, 64, 32, 32)
        enc4, skip3 = self.encoder4(enc3)      # (B, 256, 8, 8); skip3: (B, 128, 16, 16)
        enc5, skip4 = self.encoder5(enc4)      # (B, 512, 4, 4); skip4: (B, 256, 8, 8)
        
        # Center
        center = self.center(enc5)           # (B, 256, 4, 4)
        
        # Decoder (fusing with corresponding encoder outputs)
        dec4 = self.decoder4(center, enc5)    # (B, 32, 8, 8)
        dec3 = self.decoder3(dec4, enc4)      # (B, 32, 16, 16)
        dec2 = self.decoder2(dec3, enc3)      # (B, 32, 32, 32)
        dec1 = self.decoder1(dec2, enc2)      # (B, 32, 64, 64)
        
        # Final upsampling to recover original resolution 256×256.
        out = self.output(dec1)              # (B, out_channels, 256, 256)
        return out

# Example usage:
if __name__ == "__main__":
    model = ResNet34_UNet(in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 256, 256)  # Dummy input
    y = model(x)
    print("Output shape:", y.shape)  # Expected: (1, 1, 256, 256)
