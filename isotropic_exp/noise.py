""" Noise-related functions. """

from typing import *

import numpy as np
import torch

from data import DatasetInfo
from tensor_ops import rand_power_uniform, rand_uniform, rand_log_uniform, to_tensor
from trackers import TimeTracker
from covariance_generation import *

import deepinv as dinv
import pdb


class NoiseLevel:
    """ Represents an input noise level in different units and handles the conversion logic.
    - variance: noise variance per pixel/frequency
    - stddev: noise standard deviation per pixel/frequency
    - time: reverse diffusion time (inverse of variance)
    There also "effective" versions which correspond to the infinite-dimensional equivalent.
    """
    units = ["var", "fvar", "std", "fstd", "t", "ft"]

    def __init__(self, dataset_info: DatasetInfo = None, variance: torch.Tensor = None, effective_variance: torch.Tensor = None,
                 stddev: torch.Tensor = None, effective_stddev: torch.Tensor = None,
                 time: torch.Tensor = None, effective_time: torch.Tensor = None) -> None:
        self.dataset_info: DatasetInfo = dataset_info

        # We store the variance per pixel as it is the most convenient quantity.
        if stddev is not None:
            variance = stddev ** 2
        if effective_stddev is not None:
            effective_variance = effective_stddev ** 2
        if time is not None:
            variance = self.dataset_info.mean / time
        if effective_time is not None:
            effective_variance = self.dataset_info.mean / effective_time
        if effective_variance is not None:
            variance = effective_variance * self.dataset_info.dimension
        self._variance: torch.Tensor = to_tensor(variance)

    @staticmethod
    def from_unit(dataset_info: DatasetInfo, x: torch.Tensor, unit: str) -> "NoiseLevel":
        if unit in NoiseLevel.units:
            kwarg = dict(var="variance", fvar="effective_variance", std="stddev", fstd="effective_stddev", t="time", ft="effective_time")[unit]
            return NoiseLevel(dataset_info=dataset_info, **{kwarg: x})
        elif unit in DenoisingError.units:
            return DenoisingError.from_unit(dataset_info, x, unit).to_noise_level()
        else:
            raise ValueError(f"Unknown unit: {unit}.")

    @property
    def replace_variance(self, new_variance) -> torch.Tensor:
        """ Returns the noise variance per pixel/frequency in finite dimensions. """
        self._variance = to_tensor(new_variance)
    
    @property
    def variance(self) -> torch.Tensor:
        """ Returns the noise variance per pixel/frequency in finite dimensions. """
        return self._variance

    @property
    def effective_variance(self) -> torch.Tensor:
        """ Returns the noise variance per frequency in infinite dimensions. """
        return self._variance / self.dataset_info.dimension

    @property
    def stddev(self) -> torch.Tensor:
        """ Returns the noise standard deviation per pixel/frequency in finite dimensions. """
        return torch.sqrt(self.variance)

    @property
    def effective_stddev(self) -> torch.Tensor:
        """ Returns the noise standard deviation per frequency in infinite dimensions. """
        return torch.sqrt(self.effective_variance)

    @property
    def time(self) -> torch.Tensor:
        """ Returns the reverse diffusion time in finite dimensions. """
        return self.dataset_info.mean / self.variance

    @property
    def effective_time(self) -> torch.Tensor:
        """ Returns the reverse diffusion time in infinite dimensions. """
        return self.dataset_info.mean / self.effective_variance

    def to_unit(self, unit: str) -> torch.Tensor:
        """ Returns the noise level in the given units. """
        if unit == "var":
            return self.variance
        elif unit == "fvar":
            return self.effective_variance
        elif unit == "std":
            return self.stddev
        elif unit == "fstd":
            return self.effective_stddev
        elif unit == "t":
            return self.time
        elif unit == "ft":
            return self.effective_time
        elif unit in DenoisingError.units:
            return self.to_error().to_unit(unit)
        else:
            raise ValueError(f"Unknown unit: {unit}.")

    def __getitem__(self, item) -> "NoiseLevel":
        return NoiseLevel(dataset_info=self.dataset_info, variance=self.variance[item])

    def __len__(self) -> int:
        return len(self.variance)

    @property
    def shape(self) -> torch.Size:
        return self.variance.shape

    @property
    def ndim(self) -> int:
        return self.variance.ndim

    @property
    def T(self) -> "NoiseLevel":
        return NoiseLevel(dataset_info=self.dataset_info, variance=self.variance.T)

    def to(self, device: torch.device) -> "NoiseLevel":
        return NoiseLevel(dataset_info=self.dataset_info, variance=self.variance.to(device))

    def to_error(self) -> "DenoisingError":
        """ Converts to a denoising error with the corresponding input MSE. """
        return DenoisingError(dataset_info=self.dataset_info, mse=self.variance)


class DenoisingError:
    """ Represents an output denoising error in different units and handles the conversion logic.
    - MSE: mean squared error (per pixel)
    - SNR: signal-to-noise ratio
    - PSNR: peak signal-to-noise ratio
    """
    units = ["mse", "snr", "psnr"]

    def __init__(self, dataset_info: DatasetInfo, mse: torch.Tensor = None, snr: torch.Tensor = None, psnr: torch.Tensor = None) -> None:
        self.dataset_info: DatasetInfo = dataset_info

        # We store the MSE as it is the most convenient quantity.
        if psnr is not None:
            mse = 10 ** (-psnr / 10)
        if snr is not None:
            mse = self.dataset_info.variance * 10 ** (-snr / 10)
        self._mse: torch.Tensor = to_tensor(mse)

    @staticmethod
    def from_unit(dataset_info: DatasetInfo, x: torch.Tensor, unit: str) -> "DenoisingError":
        if unit in DenoisingError.units:
            kwarg = dict(mse="mse", snr="snr", psnr="psnr")[unit]
            return DenoisingError(dataset_info=dataset_info, **{kwarg: x})
        elif unit in NoiseLevel.units:
            return NoiseLevel.from_unit(dataset_info, x, unit).to_error()
        else:
            raise ValueError(f"Unknown unit: {unit}.")

    @property
    def mse(self) -> torch.Tensor:
        """ Returns the MSE per pixel. """
        return self._mse

    @property
    def snr(self) -> torch.Tensor:
        """ Returns the SNR. """
        return 10 * torch.log10(self.dataset_info.variance / self.mse)

    @property
    def psnr(self) -> torch.Tensor:
        """ Returns the PSNR. """
        return -10 * torch.log10(self.mse)

    def to_unit(self, unit: str) -> torch.Tensor:
        """ Returns the denoising error in the given units. """
        if unit == "mse":
            return self.mse
        elif unit == "snr":
            return self.snr
        elif unit == "psnr":
            return self.psnr
        elif unit in NoiseLevel.units:
            return self.to_noise_level().to_unit(unit)
        else:
            raise ValueError(f"Unknown unit: {unit}.")

    def __getitem__(self, item) -> "DenoisingError":
        return DenoisingError(dataset_info=self.dataset_info, mse=self.mse[item])

    def __len__(self) -> int:
        return len(self.mse)

    @property
    def shape(self) -> torch.Size:
        return self.mse.shape

    @property
    def ndim(self) -> int:
        return self.mse.ndim

    def to(self, device: torch.device) -> "DenoisingError":
        return DenoisingError(dataset_info=self.dataset_info, mse=self.mse.to(device))

    def to_noise_level(self) -> NoiseLevel:
        """ Converts to a noise level with the corresponding input MSE. """
        return NoiseLevel(dataset_info=self.dataset_info, variance=self.mse)



class NoiseLevelSampler:
    """ Class that holds the logic for sampling noise levels. """
    def sample_noise_levels(self, batch_shape: torch.Size, device: torch.device) -> NoiseLevel:
        """ Samples noise levels for a batch of images, for a given batch shape (B...,).
        Return shape will always be (B..., T...) (no broadcast necessary to play nicely with model assumptions and UnionNoiseLevelSampler). """
        raise NotImplementedError


class UniformPower(NoiseLevelSampler):
    def __init__(self, min: NoiseLevel, max: NoiseLevel, alpha: float, unit: str) -> None:
        self.min: NoiseLevel = min
        self.max: NoiseLevel = max
        self.alpha: float = alpha
        self.unit: str = unit

    def sample_noise_levels(self, batch_shape: torch.Size, device: torch.device) -> NoiseLevel:
        return NoiseLevel.from_unit(dataset_info=self.min.dataset_info, unit=self.unit,
                                    x=rand_power_uniform(batch_shape, self.min.to_unit(self.unit), self.max.to_unit(self.unit), alpha=self.alpha, device=device))


class UniformStddev(UniformPower):
    """ Sample the standard deviation uniformly. """
    def __init__(self, min: NoiseLevel, max: NoiseLevel) -> None:
        super().__init__(min, max, alpha=1, unit="std")


class UniformSqrtStddev(UniformPower):
    """ Sample the square root of the standard deviation uniformly. """
    def __init__(self, min: NoiseLevel, max: NoiseLevel) -> None:
        super().__init__(min, max, alpha=2, unit="std")


class UniformVariance(UniformPower):
    """ Sample the variance uniformly. """
    def __init__(self, min: NoiseLevel, max: NoiseLevel) -> None:
        super().__init__(min, max, alpha=1, unit="var")


class UniformTime(UniformPower):
    """ Sample the reverse time (inverse of variance) uniformly."""
    def __init__(self, min: NoiseLevel, max: NoiseLevel) -> None:
        super().__init__(min, max, alpha=1, unit="t")


class UniformLog(UniformPower):
    """ Sample the PSNR uniformly (i.e., log std or var is uniform). """
    def __init__(self, min: NoiseLevel, max: NoiseLevel) -> None:
        super().__init__(min, max, alpha=1, unit="psnr")


class UniformDim(UniformPower):
    """ Sample the dimensionality (1 / sigma) uniformly. """
    def __init__(self, min: NoiseLevel, max: NoiseLevel) -> None:
        super().__init__(min, max, alpha=-1, unit="std")


class FixedNoiseLevelSampler(NoiseLevelSampler):
    """ For each image, returns the same noise levels. """
    def __init__(self, noise_levels: NoiseLevel):
        self.noise_levels: NoiseLevel = noise_levels  # (T...)

    def sample_noise_levels(self, batch_shape: torch.Size, device: torch.device) -> NoiseLevel:
        return NoiseLevel(dataset_info=self.noise_levels.dataset_info, variance=self.noise_levels.variance.to(device=device).repeat(batch_shape + (1,) * self.noise_levels.ndim))  # (B..., T...)


class UnionNoiseLevelSampler(NoiseLevelSampler):
    def __init__(self, *samplers: NoiseLevelSampler) -> None:
        self.samplers: Tuple[NoiseLevelSampler, ...] = samplers

    def sample_noise_levels(self, batch_shape: torch.Size, device: torch.device) -> NoiseLevel:
        noise_levels = [sampler.sample_noise_levels(batch_shape, device) for sampler in self.samplers]  # (B..., T...)

        # Manually broadcast to (B..., T).
        ndim = max(2, max(noise_level.ndim for noise_level in noise_levels))
        assert ndim == 2
        noise_levels = [noise_level[(...,) + (None,) * (ndim - noise_level.ndim)] for noise_level in noise_levels]  # (B..., T)

        # Concatenate along last axis.
        return NoiseLevel(dataset_info=noise_levels[0].dataset_info, variance=torch.cat([noise_level.variance for noise_level in noise_levels], dim=-1))  # (B..., T)


class Covariance:
    """ Class for representing an implicitly-defined covariance matrix. """
    def apply_power(self, x: torch.Tensor, p: float):
        raise NotImplementedError

    def apply(self, x: torch.Tensor):
        return self.apply_power(x, p=1)

    def apply_inv(self, x: torch.Tensor):
        return self.apply_power(x, p=-1)

    def apply_sqrt(self, x: torch.Tensor):
        return self.apply_power(x, p=0.5)

    def apply_inv_sqrt(self, x: torch.Tensor):
        return self.apply_power(x, p=-0.5)


class IdentityCovariance(Covariance):
    """ Identity covariance: all functions are no-ops. """
    def apply_power(self, x: torch.Tensor, p: float):
        return x

    def get_matrix(self, shape: int, device: torch.device) -> torch.Tensor:
        """ Returns the covariance matrix in pixel space, of shape (C, H, W). """
        return torch.ones(shape, shape, device=device)
    
    def get_inv_matrix(self, shape = None, device = None) -> torch.Tensor:
        """ Returns the covariance matrix in pixel space, of shape (C, H, W). """
        return self.matrix ** -1


class StationaryCovariance(Covariance):
    """ Stationary covariance defined by a Fourier spectrum. Will assume Hermitian symmetry and discard imaginary part after IFFT. """
    def __init__(self, spectrum: torch.Tensor):
        """ Spectrum should be of shape ([C,], H, W), non-negative real numbers of mean 1. """
        self.spectrum: torch.Tensor = spectrum

    def apply_power(self, x: torch.Tensor, p: float):
        # return torch.real(torch.fft.ifft2(torch.fft.fft2(x, norm="ortho") * self.spectrum ** p, norm="ortho"))
        return torch.real(torch.fft.ifft2(torch.fft.fft2(x) * self.spectrum ** p))
    
    def get_matrix(self) -> torch.Tensor:
        """ Returns the covariance matrix in pixel space, of shape (C, H, W). """
        return self.spectrum
    
    def get_inv_matrix(self) -> torch.Tensor:
        """ Returns the covariance matrix in pixel space, of shape (C, H, W). """
        return self.spectrum ** -1    
    
    def get_covariance_matrix_pixel(self) -> torch.Tensor:
        """ Returns the covariance matrix in pixel space, of shape (C, H, W). """
        return torch.real(torch.fft.ifft2(self.spectrum, norm="ortho"))

    def get_bin_spectrum(self, min_freq, max_freq) -> torch.Tensor:
        """ Returns the binned spectrum, of shape (C, num_bins). """
        # Compute the bin edges
        bin_spectrum = log_binned_power_law_stds(min_freq_power=min_freq, max_freq_power=max_freq, alpha=torch.tensor([2.3]), c=torch.tensor([0.1]), device=self.spectrum.device)
        return bin_spectrum

def power_law_covariance(dataset_info: DatasetInfo, alpha: float, c: float, device: torch.device) -> Covariance:
    return StationaryCovariance(power_law_spectrum(spatial_size=dataset_info.spatial_size, alpha=alpha, c=c, device=device))

def power_law_covariance_from_shape(spatial_size: int, alpha: float, c: float, device: torch.device, noise_level: torch.Tensor) -> Covariance:
    return StationaryCovariance(noise_level * power_law_spectrum(spatial_size=spatial_size, alpha=alpha, c=c, device=device))

def deblurring_covariance_from_shape(spatial_size: int, kernel_size: float, kernel_std: float, device: torch.device, noise_level: torch.Tensor) -> Covariance:
    blur_kernel = Blurkernel(kernel_size=kernel_size, std=kernel_std, device=device)
    fourier_inv_blur_kernel = blur_kernel.get_inverse_reg_fourier(img_dim=spatial_size, epsilon = 1e-6).abs() ** 2
    # fourier_inv_blur_kernel = blur_kernel.get_inverse_fourier(img_dim=spatial_size).abs() ** 2
    fourier_inv_blur_kernel = fourier_inv_blur_kernel.to(device)
    # pdb.set_trace()
    return StationaryCovariance(noise_level * fourier_inv_blur_kernel)

def sr_covariance_from_shape(spatial_size: int, kernel_size: float,device: torch.device, noise_level: torch.Tensor) -> Covariance:
    sr_kernel = DownSampling(kernel_size=kernel_size)
    fourier_inv_sr_kernel = sr_kernel.get_inverse_reg_fourier(img_dim=spatial_size, epsilon = 1e-6).abs() ** 2
    fourier_inv_sr_kernel = fourier_inv_sr_kernel.to(device) + 1
    # pdb.set_trace()
    return StationaryCovariance(noise_level * fourier_inv_sr_kernel)


class SpatialCorrCovariance(Covariance):
    """ Stationary covariance defined by a Fourier spectrum. Will assume Hermitian symmetry and discard imaginary part after IFFT. """
    def __init__(self, matrix: torch.Tensor):
        """ Spectrum should be of shape ([C,], H, W), non-negative real numbers of mean 1. """
        self.matrix: torch.Tensor = matrix

    def apply_power(self, x: torch.Tensor, p: float):
        return x * self.matrix ** p
    
    def get_matrix(self, shape = None, device = None) -> torch.Tensor:
        """ Returns the covariance matrix in pixel space, of shape (C, H, W). """
        return self.matrix
    
    def get_inv_matrix(self, shape = None, device = None) -> torch.Tensor:
        """ Returns the covariance matrix in pixel space, of shape (C, H, W). """
        return self.matrix ** -1

    # def get_diag(self) -> torch.Tensor:
    #     """ Returns the covariance matrix in pixel space, of shape (C, H, W). """
    #     return torch.diag(self.matrix)

def spatial_corr_covariance_testing(spatial_size: int, box_size: int, var_box: float, device: torch.device, half_box_size: float, var_clean: float = 1e-3, 
                                    inp_mask_type = "half", matrix = None, 
                                    p_n_missing = 0.8, missing_indices_input = None) -> Covariance:
    """ Returns a spatial correlation covariance with the given diagonal. """
    # Normalize diagonal to have unit variance per pixel.
    if inp_mask_type == "half":
        matrix = build_half_mask(spatial_size, device=device, var_up=var_box, var_down=var_clean, half_box_size = half_box_size)
    elif inp_mask_type == "box":
        matrix = build_box_mask(spatial_size, box_size, var_box, device=device, var_clean=var_clean)
    elif inp_mask_type == "random":
        matrix = build_random_mask(spatial_size, var_box, p_n_missing = p_n_missing, device=device, var_clean=var_clean, missing_indices_input=missing_indices_input)    
    elif inp_mask_type == "constant":
        matrix = var_box * torch.ones(spatial_size, spatial_size).to(device=device)
    elif inp_mask_type == "combined":
        type_mask = torch.randint(0, 2, (1,)).item()
        if type_mask == 0:
            matrix = build_box_mask(spatial_size, box_size, var_box, device=device, var_clean=var_clean)
        elif type_mask == 1:
            matrix = build_half_mask(spatial_size, device=device, var_up=var_box, var_down=var_clean)
    elif inp_mask_type == "input":
        matrix = matrix 
    else:
        raise ValueError(f"Unknown mask type: {inp_mask_type}.")
    return SpatialCorrCovariance(matrix=matrix.to(device=device))


def spatial_corr_covariance(spatial_size: int, box_size: int, var_box: float, device: torch.device, half_box_size: float, var_clean: float = 1e-3) -> Covariance:
    """ Returns a spatial correlation covariance with the given diagonal. """
    # Normalize diagonal to have unit variance per pixel.
    type_mask = torch.randint(0, 2, (1,)).item()
    if type_mask == 0:
        matrix = build_box_mask(spatial_size, box_size, var_box, device=device, var_clean=var_clean)
    elif type_mask == 1:
        matrix = build_half_mask(spatial_size, device=device, var_up=var_box, var_down=var_clean, half_box_size = half_box_size)
    # matrix = build_half_mask(spatial_size, device=device, var_up=var_box, var_down=var_clean)
    # matrix = var_box * torch.ones(spatial_size, spatial_size)
    # matrix = build_box_mask(spatial_size, box_size, var_box, device=device, var_clean=var_clean)
    return SpatialCorrCovariance(matrix=matrix.to(device=device))

def spectral_corr_covariance(spatial_size: int, kernel_size: int, kernel_std: int, var: float, device: torch.device) -> Covariance:
    """ Returns a spatial correlation covariance with the given diagonal. """
    # Normalize diagonal to have unit variance per pixel.
    type_mask = torch.randint(0, 2, (1,)).item()
    if type_mask == 0:
        matrix = deblurring_covariance_from_shape(spatial_size=spatial_size, kernel_size=kernel_size, kernel_std=kernel_std, device=device, noise_level=var)
    elif type_mask == 1:
        kernel_size = int(kernel_size / 2)
        matrix = sr_covariance_from_shape(spatial_size=spatial_size, kernel_size=kernel_size, device=device, noise_level=var)
    # matrix = build_half_mask(spatial_size, device=device, var_up=var_box, var_down=var_clean)
    # matrix = var_box * torch.ones(spatial_size, spatial_size)
    # matrix = build_box_mask(spatial_size, box_size, var_box, device=device, var_clean=var_clean)
    return matrix

def sample_covariances_from_distribution(batch_size: int, spatial_size: int, noise_level: NoiseLevel, frequency_component: bool, spatial_component: bool) -> List[Covariance]:
    covariances = []
    if frequency_component == True:
        for i in range(batch_size):
            # alpha = np.random.rand() * 5 + 0.0  # Sample alpha from [0, 5]
            # c = np.random.rand() * 1 + 0.01     # Sample c from [0.01, 1]
            # covariance = power_law_covariance_from_shape(spatial_size=spatial_size, alpha=alpha, c=c, device="cuda", noise_level=noise_level.variance[i].flatten().item())
            # covariances.append(covariance)
            kernel_size = 8  # Sample alpha from [0, 5]
            kernel_std = 0.8    # Sample c from [0.01, 1]
            # covariance = deblurring_covariance_from_shape(spatial_size=spatial_size, kernel_size=kernel_size, kernel_std=kernel_std, device="cuda", noise_level=noise_level.variance[i].flatten().item())
            covariance = spectral_corr_covariance(spatial_size=spatial_size, kernel_size=kernel_size, kernel_std=kernel_std, var=noise_level.variance[i].flatten().item(), device="cuda")
            covariances.append(covariance)
    if spatial_component == True:
        for i in range(batch_size):
            box_size = spatial_size #torch.randint(low=1, high=spatial_size+1, size=(1,)).item() #spatial_size#torch.randint(low=1, high=spatial_size+1, size=(1,)).item()  # Random box size between 1 and spatial_size
            half_box_size =  spatial_size # torch.randint(low=0, high=spatial_size+1, size=(1,)).item() #spatial_size#torch.randint(low=0, high=spatial_size+1, size=(1,)).item()
            noise_var = noise_level.variance[i].flatten().item()
            noise_var_clean = noise_level.variance[i+1].flatten().item()
            covariance = spatial_corr_covariance(spatial_size=spatial_size, box_size=box_size, var_box=noise_var, half_box_size = half_box_size, device="cuda", var_clean=noise_var_clean)
            covariances.append(covariance)
    return covariances 

def get_tensor_from_list_covariances(covariances: List[Covariance], noise_levels: NoiseLevel = None) -> torch.Tensor:
    """ Converts a list of Covariance objects to a tensor of shape (B, C, H, W) """
    if noise_levels is not None:
        matrices = [noise_levels.variance[i].flatten().item() * cov.get_matrix() for i, cov in enumerate(covariances)]
    else:
        matrices = [cov.get_matrix() for cov in covariances]  # List of (C, H, W)
    return torch.stack(matrices, dim=0)  # (B, C, H, W)

def get_inv_tensor_from_list_covariances(covariances: List[Covariance], noise_levels: NoiseLevel = None) -> torch.Tensor:
    """ Converts a list of Covariance objects to a tensor of shape (B, C, H, W) """
    if noise_levels is not None:
        matrices = [noise_levels.variance[i].flatten().item() * cov.get_matrix() for i, cov in enumerate(covariances)]
    else:
        matrices = [cov.get_inv_matrix() for cov in covariances]  # List of (C, H, W)
    return torch.stack(matrices, dim=0)  # (B, C, H, W)

# TODO: Compute with noise level as input for data score
def apply_power_to_list_covariances(covariances: List[Covariance], x: torch.Tensor, p: float) -> torch.Tensor:
    """ Applies the power p to each covariance in the list to the corresponding x in the batch. x should be of shape (B, C, H, W). Returns a tensor of shape (B, C, H, W) """
    results = [cov.apply_power(x[i:i+1], p) for i, cov in enumerate(covariances)]  # List of (1, C, H, W)
    return torch.cat(results, dim=0)  # (B, C, H, W)

class SignalShape:
    def __init__(self, signal_ndim: int):
        self.signal_ndim: int = signal_ndim  # Number of signal dimensions (typically 3 for CHW).
        self.signal_dims: Tuple[int, ...] = tuple(range(-signal_ndim, 0))  # (-3, -2, -1) for CHW.

    def batch_shape(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """ Returns the batch shape of the given shape. """
        return shape[:len(shape) - self.signal_ndim]  # Cannot use negative indexing if signal_ndim = 0.

    def signal_shape(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """ Returns the signal shape of the given shape. """
        return shape[len(shape) - self.signal_ndim:]  # Cannot use negative indexing if signal_ndim = 0.

    def unsqueeze(self, x: torch.Tensor) -> torch.Tensor:
        """ Appends 1... to the shape of x. (N...,) to (N..., 1...) """
        return x[(...,) + (None,) * self.signal_ndim]

s = SignalShape(signal_ndim=3)  # Number of signal dimensions (typically 3 for CHW).


class NoisySampler:
    """ Class that holds the logic for sampling noisy images. """
    def sample_noisy(self, clean: torch.Tensor, noise_level: NoiseLevel, noise_shape: torch.Size = ()) -> torch.Tensor:
        """ Samples a noisy image conditioned on the given clean image. Each clean image + noise level can optionally get multiple noise samples. If not None, sampled noises will have additional noise dimensions of shape N....
        :param clean: (B..., D...)
        :param noise_level: (B..., T...)
        :param noise_shape: optional (N...) shape to generate multiple noise samples per clean image + noise level.
        :return: (B..., T..., N..., D...)
        """
        raise NotImplementedError

    def full_noise_shape(self, clean_shape: torch.Size, noise_level_shape: torch.Size, noise_shape: torch.Size) -> torch.Size:
        """ (B..., D...) and (B..., T...) to (B..., T..., N..., D...). """
        return noise_level_shape + noise_shape + s.signal_shape(clean_shape)

    def broadcast(self, clean: torch.Tensor, noise_level: NoiseLevel, noise_ndim: int) -> Tuple[torch.Tensor, NoiseLevel]:
        """ (B..., D...), (B..., T...) to (B, 1..., 1..., D...) and (B, T..., 1..., 1...). """
        noise_level = noise_level[(...,) + (None,) * (noise_ndim + s.signal_ndim)]  # (B..., T..., 1..., 1...)
        # print(f"{s.signal_ndim=} {clean.shape=} {noise_ndim=} {noise_level.shape=}")
        clean = clean[(...,) + (None,) * (noise_level.ndim - clean.ndim) + (slice(None),) * s.signal_ndim]  # (B..., 1..., 1..., D...)
        return clean, noise_level


class AdditiveNoisySampler(NoisySampler):
    def sample_noisy(self, clean: torch.Tensor, noise_level: NoiseLevel, noise_shape: torch.Size = ()) -> torch.Tensor:
        # Handle shapes.
        full_noise_shape = self.full_noise_shape(clean.shape, noise_level.shape, noise_shape)  # (B..., T..., N..., D...)
        clean, noise_level = self.broadcast(clean, noise_level, len(noise_shape))  # broadcast to (B..., T..., 1..., D...)

        # Sample unit noise and add it scaled by noise variance.
        noise = self.sample_noise(shape=full_noise_shape, batch_size=clean.shape[0], noise_level=noise_level, device=clean.device)  # (B..., T..., N..., D...)
        # noise = self.sample_noise(shape=full_noise_shape, device=clean.device)
        # return clean + noise_level.stddev * noise  # (B..., T..., N..., D...)
        return clean + noise  # (B..., T..., N..., D...)

    def sample_noise(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        """ Samples a noise image of given shape and device, with unit variance per pixel. """
        raise NotImplementedError


class WhiteGaussianSampler(AdditiveNoisySampler):
    """ Superseded by ColoredGaussianSampler with identity covariance, but left here for backwards compatibility. """
    def sample_noise(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        return torch.randn(shape, device=device)


class ColoredGaussianSampler(WhiteGaussianSampler):
    def __init__(self, noise_covariance: Covariance):
        """ Initialize the sampler with a given power spectrum ([C,], H, W). """
        super().__init__()
        self.noise_covariance: Covariance = noise_covariance

    def get_noise_covariances(self):
        return self.noise_covariance

    def sample_noise(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        white_noise = super().sample_noise(shape=shape, device=device)  # (B..., T..., N..., C, H, W)
        return self.noise_covariance.apply_sqrt(white_noise)   
    
class MultipleColoredGaussianSamplerWithInput(WhiteGaussianSampler):
    def __init__(self, noise_covariance: Covariance, batch_size: int = 1):
        """ Initialize the sampler with a given power spectrum ([C,], H, W). """
        super().__init__()
        self.noise_covariance: List[Covariance] = [noise_covariance] * batch_size

    def get_noise_covariances(self):
        return self.noise_covariance

    def sample_noise(self, shape: torch.Size,  batch_size: int, device: torch.device, noise_level: NoiseLevel) -> torch.Tensor:
        self.noise_covariances = self.noise_covariance  # List of covariance objects, length B...
        noises = []
        for _, cov in enumerate(self.noise_covariances):
            white_noise = super().sample_noise(shape=shape[-3:], device=device)  # (B..., T..., N..., C, H, W)
            white_noise = white_noise.unsqueeze(0)
            colored_noise = cov.apply_sqrt(white_noise)
            noises.append(colored_noise)

        noises = torch.cat(noises, dim=0)
        return noises  

class MultipleColoredGaussianSampler(WhiteGaussianSampler):
    def __init__(self, frequency_component = False, spatial_component = False):
        """ Initialize the sampler with a given power spectrum ([C,], H, W). """
        super().__init__()
        self.frequency_component = frequency_component  # StationaryCovariance
        self.spatial_component = spatial_component  # SpatialCorrCovariance

        self.sampler_covariance = lambda batch_size, shape, noise_level = None: sample_covariances_from_distribution(batch_size, shape, noise_level, self.frequency_component, self.spatial_component)  # List of covariance objects, length B...

    def get_noise_covariances(self):
        return self.noise_covariances
    
    # TODO: I think I can unify the logic of both cases.
    def sample_noise(self, shape: torch.Size, batch_size: int, device: torch.device, noise_level: NoiseLevel) -> torch.Tensor:
        if len(noise_level.stddev.shape) == 5:
            ## If noise level shape has dimension 5, then we are in validation.
            # self.noise_covariances = IdentityCovariance() # TODO: Change this
            self.noise_covariances = spatial_corr_covariance(spatial_size=shape[-2], box_size=12, var_box=1, device=device, var_clean=1e-3)  # Single covariance for the whole batch
            white_noise = super().sample_noise(shape=shape, device=device)  # (B..., T..., N..., C, H, W)
            noises = self.noise_covariances.apply_sqrt(white_noise)
            noises = noise_level.stddev * noises
        else:
            self.noise_covariances = self.sampler_covariance(batch_size, shape[-1], noise_level)  # List of covariance objects, length B...
            noises = []
            for i, cov in enumerate(self.noise_covariances):
                white_noise = super().sample_noise(shape=shape[-3:], device=device)  # (B..., T..., N..., C, H, W)
                white_noise = white_noise.unsqueeze(0)
                # !! Here I am multiplying by the stddev because I detach the t from the covariance def
                # colored_noise = noise_level.stddev[i] * cov.apply_sqrt(white_noise)
                colored_noise = cov.apply_sqrt(white_noise) # This is the original one
                noises.append(colored_noise)

            noises = torch.cat(noises, dim=0)
        return noises  # (B..., T..., N..., C, H, W)
    
class UnionPinkWhiteGaussianSampler(WhiteGaussianSampler):
    def __init__(self, noise_covariance: Covariance):
        """ Initialize the sampler with a given power spectrum ([C,], H, W). """
        super().__init__()
        self.noise_covariance: Covariance = noise_covariance

    def sample_noise(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        size_white = torch.randint(0, shape[0], (1,), device=device).item()
        size_pink = shape[0] - size_white

        white_noise_pink = super().sample_noise(shape=(size_pink, *shape[1:]), device=device)  # (B..., T..., N..., C, H, W)
        pink_noise = self.noise_covariance.apply_sqrt(white_noise_pink)

        white_noise = torch.randn((size_white, *shape[1:]), device=device)
        noise = torch.cat([white_noise, pink_noise], dim=0)  # (2, B..., T..., N..., C, H, W)
        return noise



class PoissonNoisySampler(NoisySampler):
    def sample_noisy(self, clean: torch.Tensor, noise_level: NoiseLevel, noise_shape: torch.Size = ()) -> torch.Tensor:
        # Handle shapes.
        clean, noise_level = self.broadcast(clean, noise_level, len(noise_shape))  # broadcast to (B..., T..., 1..., D...)
        time = noise_level.time  # (B..., T..., 1..., 1...)
        # Repeat if necessary to convert the 1... into N....
        if noise_shape.ndim > 0:
            time = time.repeat((1,) * (time.shape - len(noise_shape)) + noise_shape)  # (B..., T..., N...)
        counts = torch.poisson(clean * time)  # (B..., T..., N..., D...)
        return counts / time



class Batch:
    """ Named-tuple-like object containing a batch of data (clean, noisy, noise-level info, etc). """
    def __init__(self, noise_level: NoiseLevel = None, clean: torch.Tensor = None, noisy: torch.Tensor = None, noise_covariance: List[Covariance] = None) -> None:
        # Uses:
        # - training: one noisy per clean, noise level info for potential weighting
        # - validation: perhaps several noisy per clean, noise_level info for plotting
        self.noise_level: NoiseLevel = noise_level # (B..., T...)
        self.clean: torch.Tensor = clean  # (B..., D...)
        self.noisy: torch.Tensor = noisy  # (B..., T..., N..., D...)
        self.noise_covariance: List[Covariance] = noise_covariance  # List of covariance objects, length B... (only for MultipleColoredGaussianSampler)

    def __getitem__(self, idx):
        idx_if_not_none = lambda x: x[idx] if x is not None else None
        return Batch(noise_level=idx_if_not_none(self.noise_level), clean=idx_if_not_none(self.clean), noisy=idx_if_not_none(self.noisy), noise_covariance=idx_if_not_none(self.noise_covariance))


def noisy_batch(clean: torch.Tensor, noise_level: NoiseLevelSampler | torch.Tensor | List[float] | float = 0, noisy_sampler: NoisySampler = WhiteGaussianSampler(), num_noises: torch.Size | int = ()) -> Batch:
    """ Returns a batch based on the given clean data and optional noise level and noisy samplers.
    Args:
        clean: (B..., D...) clean data
        noise_level: sampler for the noise levels, or fixed noise level variances (T...) (defaults to zero)
        noisy_sampler: sampler for the noisy data (defaults to additive white Gaussian noise)
        num_noises: shape or number of noise samples per clean data (N...) (defaults to empty shape)
    Returns:
        Batch: a batch of noisy data and noise levels, of shape (B..., T..., N..., D...)
    """
    batch_shape = s.batch_shape(clean.shape)  # (B...,)
    if hasattr(noise_level, "sample_noise_levels"):  # For some reason isinstance() does not always work? (maybe with autoreload in notebooks).
        noise_level = noise_level.sample_noise_levels(torch.Size([int(batch_shape[0] * 2)]), device=clean.device)  # (B..., T...)
        # noise_level = noise_level.sample_noise_levels(batch_shape, device=clean.device)  # (B..., T...)
    else:
        if noise_level is None:
            noise_level = 0
        if not isinstance(noise_level, torch.Tensor):
            noise_level = torch.tensor(noise_level, dtype=clean.dtype, device=clean.device)
        # print(f"{noise_level.shape=} {clean.shape=} {s.signal_ndim=} {batch_shape=}")
        noise_level = NoiseLevel(variance=noise_level[(None,) * (clean.ndim - s.signal_ndim)].expand(batch_shape + noise_level.shape))  # (B..., T...)
    if isinstance(num_noises, int):
        num_noises = (num_noises,)
    noisy = noisy_sampler.sample_noisy(clean, noise_level, num_noises)  # (B..., T..., N..., D...)
    covariances = noisy_sampler.get_noise_covariances()
    return Batch(noise_level=noise_level, clean=clean, noisy=noisy, noise_covariance=covariances)


def noisy_loader(dataloader: torch.utils.data.DataLoader, noise_level_sampler: NoiseLevelSampler,
                 noisy_sampler: NoisySampler, time_tracker: TimeTracker, batch_size: int = None) -> Iterator[Batch]:
    """ Yields (B, [L,] [N,] D...) batches based on the given noise level and noisy samplers. Optionally restricts batch size. """
    # TODO would be nice if this function returned an iterator that had a len() for nice integration with tqdm.
    # Maybe can be done with itertools.map?

    # Nicolas' addition: I add the cov as input
    time_tracker.switch("dataloading")
    for clean in dataloader:
        # Drop class/other information if provided in dataset.
        if isinstance(clean, (tuple, list)):
            clean = clean[0]
        # Decrease batch size if necessary.
        if batch_size is not None:
            clean = clean[:batch_size]

        time_tracker.switch("cuda")
        clean = clean.cuda()  # (B, D...)

        time_tracker.switch("noise")
        yield noisy_batch(clean=clean, noise_level=noise_level_sampler, noisy_sampler=noisy_sampler)

        time_tracker.switch("dataloading")
    

# define an operator that converts color images into grayscale ones.
class PowerLaw(dinv.physics.LinearPhysics):
    r"""
    Converts RGB images to grayscale.

    Signals must be tensors with 3 colour (RGB) channels, i.e. [*,3,*,*]
    The measurements are grayscale images.

    """

    def __init__(self, spatial_size, **kwargs):
        super().__init__(**kwargs)
        self.noise_model = dinv.physics.GaussianNoise(sigma=0.1)
        # the RGB to grayscale coefficients
        coefficients = power_law_spectrum(spatial_size=spatial_size, alpha=2.3, c=0.1, device='cuda')

        # register the coefficients as a buffer
        self.register_buffer("coefficients", coefficients)

    def A(self, x, theta=None):  # theta is an optional parameter that is not used here
        y = (
            torch.real(torch.fft.ifft2(torch.fft.fft2(x, norm="ortho") * self.coefficients ** (-0.5), norm="ortho"))
        )  # apply coefficients to each channel
        return y  # sum over the color channels
    
    def get_A(self):
        return self.coefficients

    def A_adjoint(self, y, theta=None):
        return torch.real(torch.fft.ifft2(torch.fft.fft2(y, norm="ortho") * self.coefficients ** (-0.5), norm="ortho"))


# define an operator that converts color images into grayscale ones.
class BoxInpainting(dinv.physics.LinearPhysics):
    r"""
    Converts RGB images to grayscale.

    Signals must be tensors with 3 colour (RGB) channels, i.e. [*,3,*,*]
    The measurements are grayscale images.

    """

    def __init__(self, spatial_size, **kwargs):
        super().__init__(**kwargs)
        self.noise_model = dinv.physics.GaussianNoise(sigma=0.1)
        # the RGB to grayscale coefficients
        coefficients = build_random_box_mask(spatial_size, box_size=5, num_boxes=5,device='cuda')

        # register the coefficients as a buffer
        self.register_buffer("coefficients", coefficients)

    def A(self, x, theta=None):  # theta is an optional parameter that is not used here
        y = (
            x * self.coefficients ** (-0.5)
        )  # apply coefficients to each channel
        return y  # sum over the color channels
    
    def get_A(self):
        return self.coefficients

    def A_adjoint(self, y, theta=None):
        return y * self.coefficients ** (-0.5)


