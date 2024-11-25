import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetWithDropout(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetWithDropout, self).__init__()

        # Encoder (downsampling)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Decoder (upsampling)
        self.dec4 = self.conv_block(512, 256)
        self.dec3 = self.conv_block(256, 128)
        self.dec2 = self.conv_block(128, 64)
        self.dec1 = nn.Conv2d(64, out_channels, kernel_size=1)

        self.dropout = nn.Dropout(p=0.5)  # Dropout layer with 50% probability

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Decoder with skip connections and dropout
        dec4 = self.dropout(self.dec4(enc4))
        dec3 = self.dropout(self.dec3(dec4 + enc3))  # Skip connection
        dec2 = self.dropout(self.dec2(dec3 + enc2))  # Skip connection
        dec1 = self.dec1(dec2 + enc1)  # Skip connection

        return dec1