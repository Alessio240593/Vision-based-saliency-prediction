import torch
import torch.nn as nn


# Unet con skip connections + inception + se

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class InceptionBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        hidden = out_channels // 4

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True)
        )

        self.branch_dilated = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True)
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.se = SEBlock(out_channels)

    def forward(self, x):
        x1 = self.branch1(x)
        x3 = self.branch3(x)
        xd = self.branch_dilated(x)
        xp = self.branch_pool(x)

        x = torch.cat([x1, x3, xd, xp], dim=1)
        x = self.fuse(x)
        x = self.se(x)
        return x


class DoubleConvSkip(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class EncoderBlockSkip(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = DoubleConvSkip(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip_features = self.double_conv(x)
        pooled_output = self.pool(skip_features)
        return skip_features, pooled_output


class DecoderBlockSkip(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.double_conv = DoubleConvSkip(in_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.up(x)
        x = torch.cat([x, skip_connection], dim=1)
        return self.double_conv(x)


class InceptionSeUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.enc1 = EncoderBlockSkip(in_channels, 64)
        self.enc2 = EncoderBlockSkip(64, 128)
        self.enc3 = EncoderBlockSkip(128, 256)
        self.enc4 = EncoderBlockSkip(256, 512)

        self.bottleneck = InceptionBottleneck(512, 1024)

        self.dec1 = DecoderBlockSkip(1024 + 512, 512)
        self.dec2 = DecoderBlockSkip(512 + 256, 256)
        self.dec3 = DecoderBlockSkip(256 + 128, 128)
        self.dec4 = DecoderBlockSkip(128 + 64, 64)

        self.head = nn.Sequential(
            nn.Conv2d(64, out_channels, 1),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            torch.nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec1(x, s4)
        x = self.dec2(x, s3)
        x = self.dec3(x, s2)
        x = self.dec4(x, s1)

        return self.head(x)
