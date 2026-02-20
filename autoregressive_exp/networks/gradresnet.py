""" GradResNet architecture. """

from typing import *

import torch
from torch import nn

from data import DatasetInfo

from networks.conditioning import ConditionalBlock, ConditionalSequential, NoiseVarEmbedding
from networks.normalization import Normalization


nonlinearity = lambda x: nn.functional.gelu(x)
C_PER_GROUPS = 8
GroupNormNoBias = lambda num_groups, num_channels, conditioning_channels, affine: Normalization(
    num_channels, conditioning_channels, homogeneous=True, learnable=affine, center=True, use_statistics="input", num_groups=num_groups)


class BasicBlock(ConditionalBlock):
    expansion = 1

    def __init__(self, in_planes, planes, emb_channels, stride=1, transpose=False):
        """_summary_
        Args:
            in_planes (_type_): _description_
            planes (_type_): _description_
            emb_channels (_type_): _description_
            stride (int, optional): _description_. Defaults to 1.
            transpose (bool, optional): if True, use transposed convolutions. WIP
        """
        super(BasicBlock, self).__init__()

        rev_if_t = lambda *args: reversed(args) if transpose else args

        def conv(in_channels, out_channels, kernel_size, stride, padding, bias):
            return (nn.ConvTranspose2d if transpose else nn.Conv2d)(
                *rev_if_t(in_channels, out_channels), kernel_size=kernel_size,
                stride=stride, padding=padding, bias=bias,
            )

        self.conv1 = conv(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn = GroupNormNoBias(planes // C_PER_GROUPS, planes, emb_channels, affine=True)
        self.conv2 = conv(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = ConditionalSequential(*rev_if_t(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                GroupNormNoBias(self.expansion*planes // C_PER_GROUPS, self.expansion*planes, emb_channels, affine=True)
            ))
        else:
            self.shortcut = ConditionalSequential()

    def forward(self, x, noise_cond=None):
        out = nonlinearity(self.conv1(x))
        out = self.conv2(out) + self.shortcut(x, noise_cond)
        out = nonlinearity(out)
        out = self.gn(out, noise_cond)
        return out


class GradResNet(ConditionalBlock):
    def __init__(self, dataset_info: DatasetInfo, block=BasicBlock,
                 width_multiplier: int = 1, num_blocks=[2, 2, 2, 2], strides=[1, 1, 2, 2, 2]):
        super().__init__()

        num_input_channels = dataset_info.num_channels
        w = lambda width: int(width_multiplier * width)
        self.in_planes = w(64)

        self.fourier_dim = 64
        self.time_embed_dim = 256
        self.noise_var_embedding = NoiseVarEmbedding(fourier_dim=self.fourier_dim, time_embed_dim=self.time_embed_dim)

        self.conv1 = nn.Conv2d(num_input_channels, w(64), kernel_size=3,
                               stride=strides[0], padding=1, bias=False)
        self.layer1 = self._make_layer(block, w(64), num_blocks[0], stride=strides[1])
        self.layer2 = self._make_layer(block, w(128), num_blocks[1], stride=strides[2])
        self.layer3 = self._make_layer(block, w(256), num_blocks[2], stride=strides[3])
        self.layer4 = self._make_layer(block, w(512), num_blocks[3], stride=strides[4])

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(w(64*block.expansion), w(32))
        self.fc2 = nn.Linear(w(128*block.expansion), w(64))
        self.fc3 = nn.Linear(w(256*block.expansion), w(128))

        self.nc_last = w(512*block.expansion) + w(32) + w(64) + w(128)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.time_embed_dim, stride))
            self.in_planes = planes * block.expansion
        return ConditionalSequential(*layers)

    def forward(self, x: torch.Tensor, noise_conditioning: torch.Tensor = None) -> torch.Tensor:
        if noise_conditioning is not None:
            noise_conditioning = self.noise_var_embedding(noise_conditioning)
        feats = self.conv1(x)
        feats = nonlinearity(feats)
        feats = self.layer1(feats, noise_conditioning)
        out1 = self.fc1(self.pool(feats).view(feats.size(0), -1))
        feats = self.layer2(feats, noise_conditioning)
        out2 = self.fc2(self.pool(feats).view(feats.size(0), -1))
        feats = self.layer3(feats, noise_conditioning)
        out3 = self.fc3(self.pool(feats).view(feats.size(0), -1))
        feats = self.layer4(feats, noise_conditioning)
        feats = self.pool(feats)
        feats = feats.view(feats.size(0), -1)
        fx = torch.cat([feats, out3, out2, out1], dim=-1)
        energy = (fx**2).mean(dim=-1) / 2  # (B,)
        return energy
