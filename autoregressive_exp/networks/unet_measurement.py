""" UNet architecture. """

from typing import *

import torch
from torch import nn

from data import DatasetInfo

from networks.conditioning import ConditionalBlock, ConditionalSequential, NoiseVarEmbedding, SpatioNoiseVarEmbedding, AnisotropicGammaEmbeddingNet
from networks.normalization import Normalization
from networks.measurement import MeasCondBlock

import pdb

class SequentialBuilder:
    """ Simple utility for building sequential modules which keeps track of the number of channels. """
    def __init__(self, num_channels):
        self.layers = []
        self.num_channels = num_channels

    def add_layer(self, layer, out_channels=None, **kwargs):
        in_channels = self.num_channels
        if out_channels is not None:
            kwargs["out_channels"] = out_channels
            self.num_channels = out_channels
        self.layers.append(layer(in_channels, **kwargs))
        return self  # for convenient chaining

    def get_module(self):
        """ Returns a ConditionalSequential module with the added layers, and empties the layers for reuse. """
        module = ConditionalSequential(*self.layers)
        self.layers = []
        return module


class UNet_measurement(ConditionalBlock):
    """ UNet architecture with customizable non-linearity and normalization, and noise conditioning. """
    def __init__(self, dataset_info: DatasetInfo, noise_conditioning=True, t_min=1, t_max=100,
                 num_scales=3, base_width=64, kernel_size=3, padding=1, homogeneous=True,
                 num_layers_encoder_block=2, num_layers_mid_block=2, num_layers_decoder_block=2,
                 non_linearity=nn.GELU, normalization=Normalization, **normalization_kwargs):
        super().__init__()

        num_input_channels = dataset_info.num_channels
        self.num_scales = num_scales
        self.num_layers_encoder_block = num_layers_encoder_block
        self.num_layers_mid_block = num_layers_mid_block
        self.num_layers_decoder_block = num_layers_decoder_block
        self.num_layers_encoder_measurement = 1
        self.num_layers_mid_measurement = 1
        self.num_layers_decoder_measurement = 1

        self.noise_conditioning = noise_conditioning
        if self.noise_conditioning:
            self.fourier_dim = 64
            self.time_embed_dim = 256
            self.noise_var_embedding = NoiseVarEmbedding(fourier_dim=self.fourier_dim, time_embed_dim=self.time_embed_dim, t_min=t_min, t_max=t_max)
            # self.noise_var_embedding = SpatioNoiseVarEmbedding(fourier_dim=self.fourier_dim, time_embed_dim=self.time_embed_dim, t_min=t_min, t_max=t_max)
            # self.noise_var_embedding = AnisotropicGammaEmbeddingNet(self.fourier_dim, t_min, t_max, self.time_embed_dim, 3, self.time_embed_dim)

        conv = lambda in_channels, out_channels, transpose=False, kernel_size=kernel_size, stride=1, padding=padding: (nn.ConvTranspose2d if transpose else nn.Conv2d)(
            in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=False)

        normalization_kwargs.setdefault("group_size", 8)  # Groups can be disabled by setting group_size to 1 or None.
        normalization_kwargs.setdefault("use_statistics", "input")
        norm = lambda num_channels: normalization(num_channels=num_channels, conditioning_channels=self.time_embed_dim if self.noise_conditioning else None,
                                                  homogeneous=homogeneous, **normalization_kwargs)
        builder = SequentialBuilder(num_input_channels)

        def build_block(num_layers, scale, is_encoder, is_decoder):
            """ Builds a block of the UNet (see below). Skip-connections run from before every downsampling layer to after every upsampling layer.
            ------------     |                                                                              |     ------------
                enc0         |                                                                              |         dec0
                             |-----------     |                                            |     -----------|
                                 enc1         |                                            |         dec1
                                              |-----------     |          |     -----------|
                                                  enc2         |          |         dec2
                                                               |----------|
                                                                  middle
            :param num_layers: number of layers in the block.
            :param scale: scale of block (corresponds to block index for encoder and reverse index for decoder).
            :param is_encoder: whether to start the block with a downsampling layer (except for largest scale).
            :param is_decoder: whether to end the block with an upsampling layer (except for largest scale).
            """
            # Add standard layers (with last layer of last decoder block being an exception).
            for layer in range(num_layers):
                if is_decoder and scale == 0 and layer == num_layers - 1:
                    builder.add_layer(conv, out_channels=num_input_channels)
                else:
                    if is_encoder and scale > 0 and layer == 0:
                        # Encoder blocks begin with downsampling layer (unless largest scale).
                        builder.add_layer(conv, out_channels=base_width * 2**scale, kernel_size=2, stride=2, padding=0)
                    elif is_decoder and scale > 0 and layer == num_layers - 1:
                        # Decoder blocks end with upsampling layer (unless largest scale).
                        builder.add_layer(conv, out_channels=base_width * 2**(scale - 1), kernel_size=2, stride=2, padding=0, transpose=True)
                    else:
                        builder.add_layer(conv, out_channels=base_width * 2**scale)

                    builder.add_layer(norm)
                    builder.layers.append(non_linearity())  # Bypasses add_layer because non_linearity doesn't take number of channels as input.
            return builder.get_module()

        self.encoder = nn.ModuleList([None] * self.num_scales)
        self.meas_block_encoder = nn.ModuleList([None] * self.num_scales)
        self.gains_encoder = [None] * self.num_scales
        # self.norm_encoder = nn.ModuleList([None] * self.num_scales)
        for scale in range(self.num_scales):
            self.encoder[scale] = build_block(num_layers=num_layers_encoder_block, scale=scale, is_encoder=True, is_decoder=False)
            # Add measurement conditioning block for each scale.
            h, w = 32, 32
            if scale > 0:
                self.meas_block_encoder[scale] = MeasCondBlock(in_channels= base_width * 2**scale, 
                                                               img_channels=3, h=h, w=w, 
                                                               depth=self.num_layers_encoder_measurement, c_mult=1,
                                                               scale=2 ** scale,
                                                               depth_encoding=self.num_layers_encoder_measurement, N=1)
                self.gains_encoder[scale] = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True).cuda()
            else:
                self.meas_block_encoder[scale] = MeasCondBlock(in_channels=base_width * 2**scale, img_channels=3, h=h, w=w,
                                                                depth=self.num_layers_encoder_measurement, c_mult=1,
                                                                scale=scale,
                                                                depth_encoding=self.num_layers_encoder_measurement, N=1, skip_in=False, relu_in=False)
                self.gains_encoder[scale] = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True).cuda()
            #

        self.middle_block = build_block(num_layers=num_layers_mid_block, scale=self.num_scales, is_encoder=True, is_decoder=True)  # Middle block has both an upsampling and downsampling layer.
        self.meas_block_middle = MeasCondBlock(in_channels=base_width * 2 ** (self.num_scales - 1), 
                                               img_channels=3, h=h, w=w,
                                               scale=2 ** (self.num_scales-1),
                                               depth=self.num_layers_encoder_measurement, c_mult=1, depth_encoding=self.num_layers_mid_measurement, N=1)
        self.gains_middle = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True).cuda()  # Gain for the middle block.
        # self.norm_middle = norm(base_width * 2 ** (self.num_scales - 1))

        self.decoder = nn.ModuleList([None] * self.num_scales)
        self.meas_block_decoder = nn.ModuleList([None] * self.num_scales)
        self.gains_decoder = [None] * self.num_scales
        # self.norm_decoder = nn.ModuleList([None] * self.num_scales)
        for scale in range(self.num_scales - 1, -1, -1):
                builder.num_channels *= 2  # Account for concatenation with skip-connection.
                self.decoder[scale] = build_block(num_layers=num_layers_decoder_block, scale=scale, is_encoder=False, is_decoder=True)
                # Add measurement conditioning block for each scale.
                h, w = 32, 32
                if scale > 0:
                    self.meas_block_decoder[scale] = MeasCondBlock(in_channels=base_width * 2**(scale - 1), img_channels=3, 
                                                                   h=h, w=w,
                                                                   scale=2 ** (scale-1), depth=self.num_layers_encoder_measurement, c_mult=1,
                                                                   depth_encoding=self.num_layers_decoder_measurement, N=1)
                    self.gains_decoder[scale] = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True).cuda()
                #

    def forward(self, x: torch.Tensor, noise_conditioning: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
        if self.noise_conditioning:
            noise_conditioning = self.noise_var_embedding(noise_conditioning)

        x_noisy = torch.clone(x)        
        features = []
        for scale in range(self.num_scales):
            x = self.encoder[scale](x, noise_conditioning)
            x_meas = self.meas_block_encoder[scale](x_noisy, x, covariance)  # Apply measurement conditioning block.
            x = x + self.gains_encoder[scale] * x_meas
            # x = self.norm_encoder[scale](x, noise_conditioning)
            features.append(x)
            
        x = self.middle_block(x, noise_conditioning)
        x_meas = self.meas_block_middle(x_noisy, x, covariance)
        x = x + self.gains_middle * x_meas
        # x = self.norm_middle(x, noise_conditioning)

        for scale in range(self.num_scales - 1, -1, -1):
            x = torch.cat((x, features[scale]), dim=1)
            x = self.decoder[scale](x, noise_conditioning)
            if scale != 0:
                x_meas = self.meas_block_decoder[scale](x_noisy, x, covariance)
                x = x + self.gains_decoder[scale] * x_meas
            # x = self.norm_decoder[scale](x, noise_conditioning)

        return x

    def my_named_parameters(self, reduced=True, with_grad=True, prefix="") -> Dict[str, torch.Tensor]:
        """ More convenient version of nn.Module.named_parameters. Overridden by some modules to provide more helpful names.
        Possiblity to return a reduced list (for more concise logging) or filtering parameters that have gradient only.
        For UNet, the reduced parameters are only the first layer in each block.
        """
        parameters = {}

        scales = range(self.num_scales)
        named_blocks = {f"encoder.scale{scale + 1}": (self.encoder[scale], self.num_layers_encoder_block) for scale in scales} \
            | {"middle": (self.middle_block, self.num_layers_mid_block)} \
            | {f"decoder.scale{scale + 1}": (self.decoder[scale], self.num_layers_decoder_block) for scale in reversed(scales)}  # Reverse order for decoder for convenience.
        for block_name, (block, num_layers) in named_blocks.items():

            layers = [0] if reduced else range(num_layers)
            for layer in layers:
                layer_prefix = f"{prefix}{block_name}.layer{layer+1}."

                # First layer of block is convolution: add the weight (there is never a bias and it always has gradient).
                conv = block[3 * layer]
                parameters[f"{layer_prefix}conv.weight"] = conv.weight

                # Second layer of block is normalization: logic is implemented in Normalization.
                try:
                    norm = block[3 * layer + 1]
                    parameters.update(norm.my_named_parameters(reduced=reduced, with_grad=with_grad, prefix=f"{layer_prefix}norm."))
                except IndexError:
                    pass  # The last layer of the decoder does not have a normalization layer.

                # Third layer of block is non-linearity: no parameters.

        # Finally add parameters from noise var embedding.
        if self.noise_conditioning:
            parameters.update(self.noise_var_embedding.my_named_parameters(reduced=reduced, with_grad=with_grad, prefix=f"{prefix}noise_var_embedding."))

        return parameters
