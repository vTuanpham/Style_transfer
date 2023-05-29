import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


# Define the generator network for stylized output
class Generator(nn.Module):
    def __init__(self, num_residual: int):
        super(Generator, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4)
        self.relu1 = nn.ReLU()

        # Downsampling layers
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            ResidualBlock(128, num_residual=num_residual)
        )

        # Upsampling layers
        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu5 = nn.ReLU()

        # Final convolutional layer
        self.conv6 = nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.residual_blocks(x)
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.conv6(x)
        return x


# Define the residual block
class ResidualBlock(nn.Module):
    def __init__(self, channels, num_residual: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.num_residual = num_residual

    def forward(self, x):
        for layer in range(self.num_residual):
            residual = x
            x = self.relu1(self.conv1(x))
            x = self.relu2(self.conv2(x))
            x = x + residual  # Element-wise addition with the residual
        return x
