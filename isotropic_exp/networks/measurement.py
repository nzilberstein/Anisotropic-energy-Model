""" Normalization modules and functions. """

from typing import *

import torch
from torch import nn
import torch.nn.functional as F
from einops import *
import pdb

from networks.conditioning import ConditionalBlock


def downsample_strideconv(
    in_channels=64,
    out_channels=64,
    padding=0,
    bias=True,
    mode="2R",
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
        "4",
        "8",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 4BR."
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], "C")
    down1 = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode,
    )
    return down1

def conv(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="CBR",
):
    L = []
    for t in mode:
        if t == "C":
            L.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            )
        elif t == "T":
            L.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            )
        elif t == "R":
            L.append(nn.ReLU(inplace=True))
        else:
            raise NotImplementedError("Undefined type: ".format(t))
    return sequential(*L)

def sequential(*args):
    """Advanced nn.Sequential.
    Args:
        nn.Sequential, nn.Module
    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def upsample_convtranspose(
    in_channels=64,
    out_channels=3,
    padding=0,
    bias=True,
    mode="2R",
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
        "4",
        "8",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 4BR."
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], "T")
    up1 = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode,
    )
    return up1


class HeadBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True, depth=2, relu_in=False, skip_in=False):
        super(HeadBlock, self).__init__()

        padding = kernel_size // 2

        c = out_channels if depth < 2 else in_channels

        self.convin = torch.nn.Conv2d(in_channels, c, kernel_size, padding=padding, bias=bias)
        self.zero_conv_skip = torch.nn.Conv2d(in_channels, c, 1, bias=False)
        self.depth = depth
        self.nl_1 = torch.nn.ReLU(inplace=False)
        self.nl_2 = torch.nn.ReLU(inplace=False)
        self.relu_in = relu_in
        self.skip_in = skip_in

        for i in range(depth-1):
            if i < depth - 2:
                c_in, c = in_channels, in_channels
            else:
                c_in, c = in_channels, out_channels

            setattr(self, f"conv1{i}", torch.nn.Conv2d(c_in, c_in, kernel_size, padding=padding, bias=bias))
            setattr(self, f"conv2{i}", torch.nn.Conv2d(c_in, c, kernel_size, padding=padding, bias=bias))
            setattr(self, f"skipconv{i}", torch.nn.Conv2d(c_in, c, 1, bias=False))


    def forward(self, x):

        if self.skip_in and self.relu_in:
            x = self.nl_1(self.convin(x)) + self.zero_conv_skip(x)
        elif self.skip_in and not self.relu_in:
            x = self.convin(x) + self.zero_conv_skip(x)
        else:
            x = self.convin(x)

        for i in range(self.depth-1):
            aux = getattr(self, f"conv1{i}")(x)
            aux = self.nl_2(aux)
            aux_0 = getattr(self, f"conv2{i}")(aux)
            aux_1 = getattr(self, f"skipconv{i}")(x)
            x = aux_0 + aux_1

        return x

class Heads(torch.nn.Module):
    def __init__(self, in_channels, out_channels, depth=2, scale=1, bias=True, mode="bilinear", c_mult=1, c_add=0, relu_in=False, skip_in=False):
        super(Heads, self).__init__()
        self.in_channels = in_channels # * (c_mult + c_add)
        self.scale = scale
        self.mode = mode
        setattr(self, f"head0", HeadBlock(self.in_channels, out_channels, depth=depth, bias=bias, relu_in=relu_in, skip_in=skip_in))

        if self.mode == "":
            self.nl = torch.nn.ReLU(inplace=False)
            if self.scale != 1:
                setattr(self, f"down0", downsample_strideconv(self.in_channels, self.in_channels, bias=False, mode=str(self.scale)))

    def forward(self, x):

        if self.scale != 0:
            if self.mode == "bilinear":
                x = torch.nn.functional.interpolate(x, scale_factor=1/self.scale, mode='bilinear', align_corners=False)
            else:
                x = getattr(self, f"down0")(x)
                x = self.nl(x)

        # find index
        x = getattr(self, f"head0")(x)

        return x



class Tails(torch.nn.Module):
    def __init__(self, in_channels, out_channels, depth=2, scale=1, bias=True, mode="bilinear", c_mult=1, relu_in=False, skip_in=False):
        super(Tails, self).__init__()
        self.out_channels = out_channels
        self.scale = scale
        setattr(self, f"tail", HeadBlock(in_channels, out_channels * c_mult, depth=depth, bias=bias, relu_in=relu_in, skip_in=skip_in))

        self.mode = mode
        if self.mode == "":
            self.nl = torch.nn.ReLU(inplace=False)
            if self.scale != 1:
                setattr(self, f"up", upsample_convtranspose(out_channels * c_mult, out_channels * c_mult, bias=bias, mode=str(self.scale)))

    def forward(self, x):
        x = getattr(self, f"tail")(x)
        if self.scale != 0:
            if self.mode == "bilinear":
                x = torch.nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
            else:
                x = getattr(self, f"up")(x)

        return x
    
class MeasurementTransformation(torch.nn.Module):
    def __init__(self, img_channels, h, w):
        super(MeasurementTransformation, self).__init__()
        self.h = h
        self.w = w
        self.img_channels = img_channels
        self.input_dim = img_channels * 2 * h * w
        self.output_dim = img_channels * h * w
        self.linear = torch.nn.Linear(self.input_dim, self.output_dim)
        self.non_linear = torch.nn.SiLU()

        
    def forward(self, x):
        x = self.linear(x)
        x = self.non_linear(x)
        return x
    

class MeasCondBlock(torch.nn.Module):
    def __init__(self, in_channels, img_channels, h, w, depth=1, scale=1, c_mult=1, depth_encoding=1, N=1, relu_in=False, skip_in=True):
        super(MeasCondBlock, self).__init__()
        self.in_channels = in_channels
        self.depth = depth
        self.relu_in = relu_in
        self.skip_in = skip_in
        self.img_channels = img_channels
        self.h = h
        self.w = w
        self.scale = scale
        self.c_mult = c_mult
        self.depth_encoding = depth_encoding
        self.N = N

        self.decoding_conv = Tails(self.in_channels, self.img_channels, depth=1, scale=self.scale, bias=False, c_mult=c_mult)
        self.meas_transformation = MeasurementTransformation(img_channels=self.img_channels, h=self.h, w=self.w)
        self.encoding_conv = Heads(self.img_channels, self.in_channels,  depth=self.depth_encoding, scale=self.scale, bias=False, c_mult=self.c_mult*self.N, c_add=self.N, relu_in=False, skip_in=True)

    # def forward(self, x_noisy, feature, covariance):
    #     batch_size = x_noisy.shape[0]   
    #     # Decode the feature to the pixel space
    #     x = self.decoding_conv(feature)
    #     # Apply the degradation model (related to the colored covariance)
    #     cov_feature = covariance.apply_inv_sqrt(x)
    #     cov_feature = cov_feature.view(batch_size, self.img_channels, self.h, self.w) # Reshape to original dimensions
    #     # Concat and apply the transformation between the measurement and the pixel representation of the feature
    #     meas_input = torch.cat([x_noisy, cov_feature], dim=1).view(batch_size, -1)
    #     meas_output = self.meas_transformation(meas_input).view(batch_size, self.img_channels, self.h, self.w)
    #     # Encode the measurement back to the feature space
    #     z1_meas = self.encoding_conv(meas_output)

    #     return z1_meas
    

    def forward(self, x_noisy, feature, covariance):
        batch_size = x_noisy.shape[0]   
        # Decode the feature to the pixel space
        x = self.decoding_conv(feature)
        # Apply the degradation model (related to the colored covariance)
        cov_feature = covariance.apply_inv_sqrt(x)
        cov_feature = cov_feature.view(batch_size, self.img_channels, self.h, self.w) # Reshape to original dimensions
        cov_x_noisy = covariance.apply_inv_sqrt(x_noisy)
        cov_x_noisy = cov_x_noisy.view(batch_size, self.img_channels, self.h, self.w) # Reshape to original dimensions
        # Concat and apply the transformation between the measurement and the pixel representation of the feature
        meas_input = torch.cat([cov_x_noisy, cov_feature], dim=1).view(batch_size, -1)
        meas_output = self.meas_transformation(meas_input).view(batch_size, self.img_channels, self.h, self.w)
        meas_output = covariance.apply_sqrt(meas_output)  # Apply the square root of the covariance to the measurement output
        # Encode the measurement back to the feature space
        z1_meas = self.encoding_conv(meas_output)

        return z1_meas
    
    