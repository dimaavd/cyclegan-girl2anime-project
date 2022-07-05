import torch
import torch.nn as nn
import torch.nn.functional as F

# "Reflection padding was used to reduce artifacts"
# reflection padding is used wherever possible in the generator

class Residual_block(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1,padding_mode = 'reflect'),
            nn.InstanceNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1,padding_mode = 'reflect'),
            nn.InstanceNorm2d(num_channels),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv_block = nn.Sequential(
            # 3х128х128
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # 64х128х128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            # 128х64х64
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            # 256х32х32
            # "We use 6 residual blocks for 128 × 128 training images..."
            Residual_block(256),
            Residual_block(256),
            Residual_block(256),
            Residual_block(256),
            Residual_block(256),
            Residual_block(256),
            # 256х32х32
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            # 128х64х64
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # 64х128х128
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3, padding_mode='reflect')
            # 3х128х128
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = torch.tanh(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_block = nn.Sequential(
            # 3x128x128
            # "We do not use InstanceNorm for the first C64 layer."
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64x64x64
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128x32x32
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256x16x16
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 512x15x15
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
            # 1x14x14
        )
    def forward(self, x):
        x = self.conv_block(x)
        x = torch.mean(x,(2,3))
        return x

# code from dcgan tutorial
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)