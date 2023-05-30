import torch
import torch.nn as nn


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1, use_bn=True):
        super(DiscriminatorBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=strides, padding=1)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None
        self.relu = nn.LeakyReLU(negative_slope=0.3)

    def forward(self, x):
        out = self.conv(x)
        if self.bn is not None:
            out = self.bn(out)
        out = self.relu(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, df=32):
        super(Discriminator, self).__init__()

        self.d1 = DiscriminatorBlock(in_channels, df, use_bn=False)
        self.d2 = DiscriminatorBlock(df, df, strides=2)
        self.d3 = DiscriminatorBlock(df, df*2)
        self.d4 = DiscriminatorBlock(df*2, df*2, strides=2)
        self.d5 = DiscriminatorBlock(df*2, df*4)
        self.d6 = DiscriminatorBlock(df*4, df*4, strides=2)
        self.d7 = DiscriminatorBlock(df*4, df*8)
        self.d8 = DiscriminatorBlock(df*8, df*8, strides=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(df*8*4*4, df*16)
        self.relu = nn.LeakyReLU(negative_slope=0.3)
        self.fc2 = nn.Linear(df*16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.d1(x)
        out = self.d2(out)
        out = self.d3(out)
        out = self.d4(out)
        out = self.d5(out)
        out = self.d6(out)
        out = self.d7(out)
        out = self.d8(out)

        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        validity = self.sigmoid(out)

        return validity