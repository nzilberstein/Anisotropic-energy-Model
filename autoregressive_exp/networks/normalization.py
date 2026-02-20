""" Normalization modules and functions. """

from typing import *

import torch
from torch import nn
import torch.nn.functional as F
from einops import *
import pdb

from networks.conditioning import ConditionalBlock


class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x


class Normalization(ConditionalBlock):
    """ Normalization layer (like a BatchNorm/GroupNorm, but better). Optionally, the input is scaled (gain-controlled) and shifted according to a learned embedding of a conditioning. """
    def __init__(self, num_channels: int, conditioning_channels: Optional[int] = None,
                 homogeneous: bool = False, learnable: bool = True, conditioning_shift: bool = False,
                 center: bool = True, normalize_by: Literal["mean", "stddev", "scale"] = "stddev", preserve_scale: bool = False, eps: float = 1e-5, symmetric_eps: bool = False,
                 use_statistics: str = "batch", momentum: float = 0.9, num_groups: Optional[int] = None, group_size: Optional[int] = None,
                 anisotropic: bool = True):
        """ Normalization layer.
        Args:
            num_channels: number of channels in the input.
            conditioning_channels: number of channels in the given conditioning (optional).
            homogeneous: if True, forces a linear transform (disable input centering when using batch statistics, preserve input global scale when using input statistics, and learned bias).
            learnable: whether to include learnable parameters (gain, and bias if not homogeneous).
            conditioning_shift: whether to add a shift depending on the conditioning value (if not homogeneous).
            center: whether to remove the mean of the input (disabled when homogeneous = True and use_statistics = "batch").
            normalize_by: whether to divide by the mean, standard deviation, or scale (root mean square) of the input.
            eps: small constant to avoid numerical instabilities.
            symmetric_eps: slightly different expression using eps (should be True, but False by default for backwards compatibility).
            use_statistics: "batch" or "input".
            momentum: momentum parameter for the update of the running stats (only used when use_statistics = "batch").
            num_groups, group_size: if not None (only one of them may be specified), divide channels into groups and averages inside a group (defaults to group_size=1).
        """
        super().__init__()

        self.num_channels = num_channels
        self.conditioning_channels = conditioning_channels
        self.num_groups = num_groups if num_groups is not None else num_channels // (group_size if group_size is not None else 1)
        self.group_size = num_channels // self.num_groups

        self.use_statistics = use_statistics
        if self.use_statistics not in ["batch", "input"]:
            raise ValueError(f"Unknown statistics: {self.use_statistics}")
        self.dims: Tuple[int] = dict(batch=(0,), input=())[self.use_statistics] + (2, 3, 4)  # Quickly average over the correct axes.

        self.homogeneous = homogeneous
        # Hyper-parameter overrides from homogeneity.
        self.center = center and ((not homogeneous) or self.use_statistics == "input")  # Homogeneous batch norms cannot center their input.
        self.preserve_scale = preserve_scale or (homogeneous and self.use_statistics == "input")  # Homogeneous group norms have to preserve the input scale.
        self.learnable_weight = learnable
        self.learnable_bias = learnable and not homogeneous  # Homogeneous layers are bias-free.
        self.conditioning_shift = conditioning_shift and not homogeneous  # Homogeneous layers are bias-free.

        self.normalize_by = normalize_by
        self.eps = eps
        self.symmetric_eps = symmetric_eps
        self.anisotropic = anisotropic

        # Running statistics.
        if self.use_statistics == "batch":
            self.momentum = momentum
            self.register_buffer("running_scale", torch.ones(self.num_groups))
            if self.center:  # disabled by homogeneous = True because batch statistics.
                self.register_buffer("running_mean", torch.zeros(self.num_groups))

        # Learned weights.
        if self.learnable_weight:
            self.weight = nn.Parameter(torch.ones(num_channels))
        if self.learnable_bias:
            self.bias = nn.Parameter(torch.zeros(num_channels))

        # Conditioning.
        self.noise_conditioning = conditioning_channels is not None
        if self.noise_conditioning:
            if self.anisotropic is False:
                self.conditioning_embedding = nn.Linear(conditioning_channels, num_channels * (1 + self.conditioning_shift))
            else:
                self.conditioning_embedding = nn.Conv2d(conditioning_channels, num_channels * (1 + self.conditioning_shift), kernel_size=1)

        self.group_norm = GroupNorm(num_channels, num_groups=self.num_groups)


    def extra_repr(self) -> str:
        return f"num_channels={self.num_channels}={self.num_groups} groups of size {self.group_size}, conditioning_channels={self.conditioning_channels}, homogeneous={self.homogeneous}, " \
            f"learnable_weight={self.learnable_weight}, learnable_bias={self.learnable_bias}, center={self.center}, normalize_by={self.normalize_by}, preserve_scale={self.preserve_scale}, use_statistics={self.use_statistics}"

    def compute_scale(self, x: torch.Tensor):
        """ Returns the (typically quadratic) scale vector ([B,] G) computed from x (B, G, S, H, W). """
        if self.normalize_by == "mean":
            pass
        elif self.normalize_by == "stddev":
            scale = (x - x.mean(dim=self.dims, keepdim=True)) ** 2
        elif self.normalize_by == "scale":
            scale = x ** 2
        else:
            raise ValueError(f"Unknown normalization: {self.normalize_by}")
        return scale.mean(dim=self.dims)  # ([B,] G)

    def normalizing_factor(self, scale: torch.Tensor):
        """ Converts ([B,] G) scale to ([B,] G) normalizing factor, taking care of potential square root, normalization for homogeneity, and stable inverse. """
        norm_factor = 1 / (scale + self.eps)
        # NOTE: sqrt also needs the norm_factor to be bounded away from zero for numerical stability.
        # We add epsilon to the fraction or to the numerator based on self.symmetric_eps.

        if self.preserve_scale:
            # Normalize the average of the scales to 1.
            norm_factor = (scale.mean(dim=1, keepdim=True) + (self.eps if self.symmetric_eps else 0)) * norm_factor

        if self.normalize_by != "mean":
            norm_factor = torch.sqrt(norm_factor + (self.eps if not self.symmetric_eps else 0))

        return norm_factor

    def forward(self, x: torch.Tensor, conditioning=None) -> torch.Tensor:
        # Expand the input into groups.
        # x = rearrange(x, "b (g s) h w -> b g s h w", g=self.num_groups)

        # # Compute stats from input or use running stats.
        # if self.use_statistics == "batch" and not self.training:  # Use running stats.
        #     if self.center:
        #         mean = self.running_mean

        #     scale = self.running_scale
        # else:  # Use batch statistics and update running stats if needed.
        #     if self.center:
        #         mean = x.mean(dim=self.dims)  # ([B,] G)
        #         if self.use_statistics == "batch":  # self.training is guaranteed
        #             with torch.no_grad():
        #                 self.running_mean.copy_(self.momentum * self.running_mean.data + (1 - self.momentum) * mean)  # (G,)

        #     scale = self.compute_scale(x)  # ([B,] G)
        #     if self.use_statistics == "batch":  # self.training is guaranteed
        #         with torch.no_grad():
        #             self.running_scale.copy_(self.momentum * self.running_scale.data + (1 - self.momentum) * scale)  # (G,)

        # # Whiten x.
        # if self.center:
        #     x = x - mean[..., None, None, None]
        # x = x * self.normalizing_factor(scale)[..., None, None, None]

        # # Collapse groups (learned parameters and conditioning are done per-channel, ignoring groups).
        # x = rearrange(x, "b g s h w -> b (g s) h w")

        # # Apply learned parameters.
        # if self.learnable_weight:
        #     x = x * self.weight[..., None, None]
        # if self.learnable_bias:
        #     x = x + self.bias[..., None, None]

        x = self.group_norm(x)

        # Conditioning.
        if self.noise_conditioning and conditioning is not None:
            if self.anisotropic is False:
                embedding = self.conditioning_embedding(conditioning)[..., None, None]  # (B, C or 2C, 1, 1)
                x = x * (1 + embedding[:, :self.num_channels])
                # -----------------------------------------
            else:
                embedding = self.conditioning_embedding(conditioning)  # (B, C or 2C, 1, 1)
                if embedding.shape[-2:] != x.shape[-2:]:
                    embedding = F.interpolate(
                        embedding, 
                        size=x.shape[-2:],  # Target size is the H, W of x
                        mode='bilinear',    # Bilinear is a good default for feature maps
                        align_corners=False
                    )
                x = x * (1 + embedding[:, :self.num_channels])
            # # -----------------------------------------
            if self.conditioning_shift:
                x = x + embedding[:, self.num_channels:]

        return x

    def my_named_parameters(self, reduced=True, with_grad=True, prefix="") -> Dict[str, torch.Tensor]:
        """ More convenient version of nn.Module.named_parameters. Overridden by some modules to provide more helpful names.
        Possiblity to return a reduced list (for more concise logging) or filtering parameters that have gradient only.
        For Normalization, all parameters are included (reduced is ignored).
        """
        parameters = {}

        # Running statistics: if batch statistics only (no gradients).
        if self.use_statistics == "batch" and not with_grad:
            parameters[f"{prefix}running_scale"] = self.running_scale
            if self.center:
                parameters[f"{prefix}running_mean"] = self.running_mean

        # Learned weights (have gradient).
        if self.learnable_weight:
            parameters[f"{prefix}weight"] = self.weight
        if self.learnable_bias: 
            parameters[f"{prefix}bias"] = self.bias

        # Conditioning (have gradient, linear layer always has a bias).
        if self.noise_conditioning:
            parameters[f"{prefix}conditioning_embedding.weight"] = self.conditioning_embedding.weight
            parameters[f"{prefix}conditioning_embedding.bias"] = self.conditioning_embedding.bias

        return parameters
