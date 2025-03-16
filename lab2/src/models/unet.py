import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    """
    An encoder block that applies two 3x3 convolutions with BatchNorm and ReLU,
    then downsamples using 2x2 max pooling.

    Returns:
        pooled: Downsampled feature map.
        features: Feature map before pooling (for skip connections).
    """
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = self.conv_block(x)
        pooled = self.pool(features)
        return pooled, features


class DecoderBlock(nn.Module):
    """
    A decoder block that upsamples the input, crops and concatenates the corresponding
    encoder feature map (skip connection), and then applies two 3x3 convolutions
    with BatchNorm and ReLU.
    """
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # in_channels here is the number of channels from the previous layer (to be upsampled)
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After upsampling, we concatenate with the skip connection,
        # so the number of input channels for the conv block is out_channels*2.
        self.conv_block = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        # Center-crop the skip connection if necessary to match x's spatial size.
        if x.size()[2:] != skip.size()[2:]:
            diff_y = skip.size(2) - x.size(2)
            diff_x = skip.size(3) - x.size(3)
            skip = skip[:, :, diff_y // 2: diff_y // 2 + x.size(2), diff_x // 2: diff_x // 2 + x.size(3)]
        # Concatenate along the channel dimension.
        x = torch.cat([skip, x], dim=1)
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    """
    U-Net architecture that consists of:
      - Four encoder blocks.
      - A center block.
      - Four decoder blocks.
      - A final 1x1 convolution mapping features to the desired number of classes.
    """
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        # Encoder: downsampling
        self.encoder1 = EncoderBlock(in_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        
        # Center block (bottom of the U)
        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        
        # Decoder: upsampling
        self.decoder4 = DecoderBlock(1024, 512)
        self.decoder3 = DecoderBlock(512, 256)
        self.decoder2 = DecoderBlock(256, 128)
        self.decoder1 = DecoderBlock(128, 64)
        
        # Final 1x1 convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path with skip connections
        enc1, skip1 = self.encoder1(x)  # enc1: pooled, skip1: features from block 1
        enc2, skip2 = self.encoder2(enc1)
        enc3, skip3 = self.encoder3(enc2)
        enc4, skip4 = self.encoder4(enc3)
        
        center = self.center(enc4)
        
        # Decoder path with skip connection concatenation
        dec4 = self.decoder4(center, skip4)
        dec3 = self.decoder3(dec4, skip3)
        dec2 = self.decoder2(dec3, skip2)
        dec1 = self.decoder1(dec2, skip1)
        
        out = self.final_conv(dec1)
        return out
