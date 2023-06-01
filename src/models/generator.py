import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms


# Define the generator network for stylized output
class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.pad = SymmetricPadding2D()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=(5, 5), padding=0)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=0)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.5)
        self.prelu = nn.PReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=0)
        self.bn3 = nn.BatchNorm2d(64, momentum=0.5)

    def forward(self, ip):
        res_model = self.pad(ip)
        res_model = self.pad(res_model)
        res_model = self.conv1(res_model)
        res_model = self.bn1(res_model)

        res_model = self.pad(res_model)
        res_model = self.conv2(res_model)
        res_model = self.bn2(res_model)
        res_model = self.prelu(res_model)

        res_model = self.pad(res_model)
        res_model = self.conv3(res_model)
        res_model = self.bn3(res_model)

        return ip + res_model


class DeepResBlock(nn.Module):
    def __init__(self):
        super(DeepResBlock, self).__init__()
        self.pad = SymmetricPadding2D()
        self.conv1 = nn.Conv2d(64, 256, kernel_size=(3, 3), padding=0)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(256, 64, kernel_size=(3, 3), padding=0)
        self.bn = nn.BatchNorm2d(64, momentum=0.5)
        # self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.prelu2 = nn.PReLU()

    def forward(self, ip):
        up_model = self.pad(ip)
        up_model = self.conv1(up_model)
        up_model = self.prelu1(up_model)

        up_model = self.pad(up_model)
        up_model = self.conv2(up_model)
        up_model = self.bn(up_model)
        # up_model = self.upsample(up_model)
        up_model = self.prelu2(up_model)

        return up_model


class Generator(nn.Module):
    def __init__(self, num_res_block, num_deep_res_block):
        super(Generator, self).__init__()
        self.pad = SymmetricPadding2D()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=(9, 9), padding=0)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(5, 5), padding=0)
        self.prelu2 = nn.PReLU()

        self.res_blocks = nn.ModuleList([ResBlock() for _ in range(num_res_block)])

        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=0)
        self.bn = nn.BatchNorm2d(64, momentum=0.5)

        self.deep_res_blocks = nn.ModuleList([DeepResBlock() for _ in range(num_deep_res_block)])

        self.conv4 = nn.Conv2d(64, 3, kernel_size=(9, 9), padding=0)

    def forward(self, gen_ip):
        layers = self.pad(gen_ip)
        layers = self.pad(layers)
        layers = self.pad(layers)
        layers = self.pad(layers)
        layers = self.conv1(layers)
        layers = self.prelu1(layers)

        layers = self.pad(layers)
        layers = self.pad(layers)
        layers = self.conv2(layers)
        layers = self.prelu2(layers)

        temp = layers

        for res_block in self.res_blocks:
            layers = res_block(layers)

        layers = self.pad(layers)
        layers = self.conv3(layers)
        layers = self.bn(layers)

        layers = layers + temp

        for deep_res_block in self.deep_res_blocks:
            layers = deep_res_block(layers)

        layers = self.pad(layers)
        layers = self.pad(layers)
        layers = self.pad(layers)
        layers = self.pad(layers)

        op = self.conv4(layers)

        return op


# Custom layer inherited from nn.Module
class SymmetricPadding2D(nn.Module):
    def __init__(self, padding=[1, 1], data_format="channels_first"):
        super(SymmetricPadding2D, self).__init__()
        self.padding = padding
        self.data_format = data_format

    def forward(self, inputs):
        if self.data_format == "channels_last":
            # (batch, height, width, channels)
            pad = nn.ReflectionPad2d(self.padding)
        elif self.data_format == "channels_first":
            # (batch, channels, height, width)
            pad = nn.ReflectionPad2d([self.padding[1], self.padding[1], self.padding[0], self.padding[0]])
        else:
            raise ValueError("Invalid data format. Expected 'channels_last' or 'channels_first'.")

        return pad(inputs)
