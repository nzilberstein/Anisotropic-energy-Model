""" DnCNN architecture. """

from typing import *

import torch
from torch import nn

from data import DatasetInfo
from networks.normalization import Normalization


class DnCNN(nn.Module):
    """ DnCNN architecture (no downsamplings). """

    def __init__(self, dataset_info: DatasetInfo, num_hidden_layers: int = 20, num_hidden_channels: int = 64, kernel_size: int = 3,
                 normalization: str = "pre", learnable_normalization: bool = True, normalize_by: Literal["mean", "stddev", "scale"] = "scale",
                 remove_mean: bool = False, skip=True, homogeneous=True):
        super().__init__()

        self.num_hidden_layers = num_hidden_layers
        self.dataset_info = dataset_info
        self.remove_mean = remove_mean
        self.skip = skip
        self.homogeneous = homogeneous

        num_input_channels = dataset_info.num_channels
        padding = (kernel_size - 1) // 2

        self.layers = []
        for l in range(self.num_hidden_layers):
            self.layers.append(nn.Conv2d(num_input_channels if l == 0 else num_hidden_channels, num_hidden_channels, kernel_size, padding=padding, bias=not self.homogeneous))

            norm = Normalization(num_hidden_channels, homogeneous=self.homogeneous, learnable=learnable_normalization, normalize_by=normalize_by)
            nonlin = nn.ReLU()
            if normalization == "pre":
                self.layers.extend([norm, nonlin])
            elif normalization == "post":
                self.layers.extend([nonlin, norm])
            elif normalization == "none":
                self.layers.append(nonlin)
            else:
                raise ValueError(f"Unknown normalization: {normalization}")

        self.layers.append(nn.Conv2d(num_hidden_channels, num_input_channels, kernel_size, padding=padding, bias=not self.homogeneous))

        self.module = nn.Sequential(*self.layers)

        self.apply(weights_init_kaiming)

    def extra_repr(self) -> str:
        return f"num_hidden_layers={self.num_hidden_layers}, skip={self.skip}, homogeneous={self.homogeneous}"

    def forward(self, x: torch.Tensor, noise_conditioning: torch.Tensor = None) -> torch.Tensor:
        """ (B, *, C, H, W) to (B, *, C, H, W). Ignores noise conditioning. """
        batch_shape = x.shape[:-3]
        x = x.reshape((-1, *x.shape[-3:]))  # (B*, C, H, W)

        if self.remove_mean:
            x = x - self.dataset_info.mean

        y = self.module(x)
        if self.skip:
            y = y + x  # Note: adds x with zero mean if self.remove_mean is True (correct behavior).

        if self.remove_mean:
            y = y + self.dataset_info.mean

        y = y.reshape((*batch_shape, *y.shape[-3:]))  # (B, *, C, H, W)
        return y


def weights_init_kaiming(module):
    """ Initialization with Gaussian weights and fan-in normalization. """
    if any(isinstance(module, cl) for cl in [nn.Conv2d, nn.Linear]):
        nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
