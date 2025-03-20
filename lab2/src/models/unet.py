import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    Basic convolutional block:
    - Two 3x3 convolutions (with padding=1) 
    - Each convolution is followed by BatchNorm and ReLU.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    """
    Encoder block that applies a ConvBlock and then downsamples via max pooling.
    
    Returns:
        pooled: The downsampled feature map.
        features: The output of the ConvBlock (to be used as a skip connection).
    """
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        features = self.conv_block(x)
        pooled = self.pool(features)
        return pooled, features


class DecoderBlock(nn.Module):
    """
    Decoder block that upsamples the input, concatenates it with the corresponding 
    encoder features (skip connection), and applies a ConvBlock.
    
    Args:
        in_channels (int): Number of channels in the input to be upsampled.
        out_channels (int): Number of channels after upsampling and processing.
    """
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # Upsample the input to double its spatial dimensions
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After upsampling, the skip connection (from the encoder) has out_channels.
        # Their concatenation yields 2*out_channels channels.
        self.conv_block = ConvBlock(out_channels * 2, out_channels)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        # If necessary, center-crop the skip connection to match x's spatial size.
        if x.size()[2:] != skip.size()[2:]:
            diff_y = skip.size(2) - x.size(2)
            diff_x = skip.size(3) - x.size(3)
            skip = skip[:, :, diff_y // 2: diff_y // 2 + x.size(2),
                          diff_x // 2: diff_x // 2 + x.size(3)]
        # Concatenate along the channel dimension and apply the conv block.
        x = torch.cat([skip, x], dim=1)
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation. It comprises:
      - Four encoder blocks for downsampling.
      - A bottleneck center block (a ConvBlock without pooling).
      - Four decoder blocks for upsampling and merging skip connections.
      - A final 1x1 convolution mapping to the desired number of output classes.
    
    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of segmentation classes (or output channels).
    """
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        # Encoder: Downsample the image while saving skip connections.
        self.enc1 = EncoderBlock(in_channels, 64)    # Output: features with 64 channels.
        self.enc2 = EncoderBlock(64, 128)              # Output: features with 128 channels.
        self.enc3 = EncoderBlock(128, 256)             # Output: features with 256 channels.
        self.enc4 = EncoderBlock(256, 512)             # Output: features with 512 channels.
        
        # Center block (Bottleneck): No pooling.
        self.center = ConvBlock(512, 1024)
        
        # Decoder: Upsample and merge with skip connections.
        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)
        
        # Final 1x1 convolution to map to the desired number of output classes.
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder pathway: Each block returns the pooled features and the skip connection.
        enc1_pool, skip1 = self.enc1(x)   # Resolution: 1/2 of input.
        enc2_pool, skip2 = self.enc2(enc1_pool)  # Resolution: 1/4 of input.
        enc3_pool, skip3 = self.enc3(enc2_pool)  # Resolution: 1/8 of input.
        enc4_pool, skip4 = self.enc4(enc3_pool)  # Resolution: 1/16 of input.
        
        # Center bottleneck block.
        center = self.center(enc4_pool)
        
        # Decoder pathway: Upsample and concatenate with corresponding skip connections.
        dec4 = self.dec4(center, skip4)   # Upsampled to resolution 1/8.
        dec3 = self.dec3(dec4, skip3)       # Upsampled to resolution 1/4.
        dec2 = self.dec2(dec3, skip2)       # Upsampled to resolution 1/2.
        dec1 = self.dec1(dec2, skip1)       # Upsampled to full resolution.
        
        # Final output layer.
        out = self.final_conv(dec1)
        return out  