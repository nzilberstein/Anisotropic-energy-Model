""" Conditioning modules (e.g., on noise level). """

from typing import *

import math
import torch
from torch import nn

import pdb


class ConditionalBlock(nn.Module):
    """ Any module where forward() takes an optional conditioning as a second argument. """
    def forward(self, x, conditioning=None):
        raise NotImplementedError


class ConditionalSequential(nn.Sequential, ConditionalBlock):
    """ A sequential module that passes conditioning information to the children that support it as an extra input. """
    def forward(self, x, conditioning=None):
        for layer in self:
            if isinstance(layer, ConditionalBlock):
                x = layer(x, conditioning)
            else:
                x = layer(x)
        return x


def sinusoidal_embedding(timesteps, dim, t_min=1, t_max=100, log_scale=False):
    """ Create sinusoidal embeddings. Typically timesteps are noise variances, compared to pixel value range in [0, 1].
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
    :param dim: the dimension of the output.
    :param t_min, t_max: estimated minimum and maximal values of the timesteps (widest values should be 1e-5 and 1e2, defaults are here for backwards compatibility).
    :param log_scale: if True, puts t in log space with linear frequencies, otherwise puts t in linear space with log frequencies.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    assert dim % 2 == 0, "dim must be even"
    if log_scale:
        # Put t in log space with linear frequencies.
        timesteps = torch.log(timesteps)
        freqs = (1 + torch.arange(dim // 2, dtype=torch.float32, device=timesteps.device)) / torch.log(t_max / t_min)  # (F,), ranges from 1/t_min to 1/t_max linearly.
        assert False, "not implemented because t = 0 behvaior is undefined"
    else:
        # Use log frequencies.
        freqs = torch.exp(-torch.linspace(math.log(t_min), math.log(t_max), dim // 2, dtype=torch.float32, device=timesteps.device))  # (F,), ranges from 1/t_min to 1/t_max logarithmically.
    args = timesteps[:, None] * freqs[None]  # (B, F)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, 2F)
    return embedding


class NoiseVarEmbedding(nn.Module):
    """ Embedding module for noise variance. """
    def __init__(self, fourier_dim=64, time_embed_dim=256, t_min=1, t_max=100):
        super().__init__()
        self.fourier_dim = fourier_dim
        self.time_embed_dim = time_embed_dim
        self.t_min = t_min
        self.t_max = t_max
        self.time_embed = nn.Sequential(
            nn.Linear(self.fourier_dim, self.time_embed_dim),
            # nn.Sigmoid(),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            # nn.Sigmoid(),
            nn.SiLU(),
            # nn.Linear(self.time_embed_dim, self.time_embed_dim),
            # nn.Sigmoid()
        )

    def forward(self, noise_var: torch.Tensor):
        # noise_var = torch.log10(noise_var.view(noise_var.shape[0], -1))  # Flatten the input to ensure it is 1D.
        # return self.time_embed(noise_var)
        return self.time_embed(sinusoidal_embedding(timesteps=noise_var, dim=self.fourier_dim, t_min=self.t_min, t_max=self.t_max))

    def extra_repr(self):
        return f"fourier_dim={self.fourier_dim}, time_embed_dim={self.time_embed_dim}, t_min={self.t_min}, t_max={self.t_max}"

    def my_named_parameters(self, reduced=True, with_grad=True, prefix="") -> Dict[str, torch.Tensor]:
        """ More convenient version of nn.Module.named_parameters. Overridden by some modules to provide more helpful names.
        Possiblity to return a reduced list (for more concise logging) or filtering parameters that have gradient only.
        For NoiseVarEmbedding, all parameters are included (reduced is ignored).
        """
        parameters = {}

        # Just add both weight and bias parameters of both linear layers.
        for i in range(2):
            layer = self.time_embed[2 * i]
            parameters[f"{prefix}linear{i+1}.weight"] = layer.weight
            parameters[f"{prefix}linear{i+1}.bias"] = layer.bias

        return parameters

class SpatioNoiseVarEmbedding(nn.Module):
    """
    Implements the following operation:
    1. Pixel-wise sinusoidal embedding of a spatial map (average over channels).
    2. Convolutional projection.
    """
    def __init__(self, fourier_dim=64, time_embed_dim=256, t_min=1, t_max=100):
        super().__init__()
        self.fourier_dim = fourier_dim
        self.time_embed_dim = time_embed_dim
        self.t_min = t_min
        self.t_max = t_max
        
        self.conv = nn.Sequential(
            nn.Conv2d(self.fourier_dim, self.time_embed_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(self.time_embed_dim, self.time_embed_dim, kernel_size=1),
            nn.SiLU(),
        )

    def forward(self, t_map: torch.Tensor) -> torch.Tensor:
        """
        :param t_map: The conditioning tensor of shape [B, C, H, W].
        :return: A processed conditioning map of shape [B, 256, H, W].
        """
        b, c, h, w = t_map.shape
        
        # --- Step 1: Pixel-wise Sinusoidal Embedding ---
        
        # a. Aggregate the channel dimension to get a single value per pixel.
        t_map_agg = t_map.mean(dim=1) #[B, H, W]
        
        # b. Flatten the spatial map into a 1D tensor of pixel values.
        t_map_flat = t_map_agg.view(-1) #[B*H*W]
        
        # c. Apply the sinusoidal embedding.
        sin_embedding = sinusoidal_embedding(
            t_map_flat,
            dim=self.fourier_dim,
            t_min=self.t_min,
            t_max=self.t_max
        ) # [B*H*W, 64]
        
        # d. Reshape the result back into a spatial map.
        sin_map = sin_embedding.view(b, h, w, self.fourier_dim).permute(0, 3, 1, 2) # [B, 64, H, W]

        # --- Step 2: Convolutional Projection ---
        
        # Apply the conv to get the final conditioning map.
        final_conditioning_map = self.conv(sin_map) # [B, 256, H, W]
        
        return final_conditioning_map
    
# --- 2. Small Convolutional ResNet Block for Gamma Embedding ---
class ConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels) # Use GroupNorm for stability
        self.silu = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return self.silu(h + self.shortcut(x))


# --- 3. Anisotropic Gamma Embedding Network ---
class AnisotropicGammaEmbeddingNet(nn.Module):
    def __init__(self, fourier_dim, t_min, t_max, base_channels, num_res_blocks, out_embedding_channels):
        super().__init__()
        self.fourier_dim = fourier_dim
        self.t_min = t_min
        self.t_max = t_max

        self.initial_conv = nn.Conv2d(self.fourier_dim, base_channels, kernel_size=3, padding=1)

        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            self.res_blocks.append(ConvResBlock(base_channels, base_channels))

        self.final_conv = nn.Conv2d(base_channels, out_embedding_channels, kernel_size=1)

    def forward(self, t_map):
        # pe_gamma_map: [B, D, H, W] (D is dim from positional encoding)
        b, c, h, w = t_map.shape
        
        # --- Step 1: Pixel-wise Sinusoidal Embedding ---
        
        # a. Aggregate the channel dimension to get a single value per pixel.
        t_map_agg = t_map.mean(dim=1) #[B, H, W]
        
        # b. Flatten the spatial map into a 1D tensor of pixel values.
        t_map_flat = t_map_agg.view(-1) #[B*H*W]
        
        # c. Apply the sinusoidal embedding.
        sin_embedding = sinusoidal_embedding(
            t_map_flat,
            dim=self.fourier_dim,
            t_min=self.t_min,
            t_max=self.t_max
        ) # [B*H*W, 64]
        
        # d. Reshape the result back into a spatial map.
        sin_map = sin_embedding.view(b, h, w, self.fourier_dim).permute(0, 3, 1, 2) # [B, 64, H, W]

        x = self.initial_conv(sin_map)
        for block in self.res_blocks:
            x = block(x)
        gamma_embedding = self.final_conv(x) # [B, out_embedding_channels, H, W]
        
        return gamma_embedding