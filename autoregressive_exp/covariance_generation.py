from typing import *

import numpy as np
import torch

import scipy


def build_half_mask(spatial_size, device, var_up, half_box_size, var_down=1e-3):
    # size_box = spatial_size // 2
    size_box = half_box_size
    # size_box = torch.randint(low=0, high=spatial_size+1, size=(1,)).item() # Random box size between 0 and spatial_size
    # Create a 2D boolean mask for the ima
    noise_up = var_up * torch.ones((size_box, spatial_size), device=device)
    noise_down = var_down * torch.ones((spatial_size - size_box, spatial_size), device=device)

    noise_diag = torch.cat((noise_up, noise_down), dim=0)

    return noise_diag

def build_box_mask(spatial_size, box_size, var_box, device, var_clean=1e-3):
    start_coord = (spatial_size - box_size) // 2
    end_coord = start_coord + box_size
    # Create a 2D boolean mask for the image
    # Initialize with all False (representing the area outside the box)
    noise_diag = var_clean * torch.ones((spatial_size, spatial_size), device=device)

    # Set the area inside the box to True
    noise_diag[start_coord:end_coord, start_coord:end_coord] = var_box

    return noise_diag

def build_random_box_mask(spatial_size, box_size, num_boxes, device):
    # Create a 2D boolean mask for the image
    # Initialize with all False (representing the area outside the box)
    H = torch.ones((spatial_size, spatial_size), dtype=torch.bool, device=device) * 1e-3

    top = 1
    left = 3
    var_inp = 1
    for _ in range(num_boxes):
        # Random top-left corner ensuring the box fits inside the image

        # Set the area inside the box to False (0)
        H[top:top+box_size, left:left+box_size] = var_inp
        top = top + 5
        left = left + 5
        var_inp = var_inp * 0.7e-1

    noise_diag = H

    return noise_diag

def build_dynamic_noise_mask(spatial_size, num_boxes, noise_levels, device, overlap_mode='max'):
    """
    Build a spatial noise mask where each box gets its own noise level.
    
    Args:
        spatial_size: Size of the square spatial dimension (e.g., 28 for MNIST)
        num_boxes: Number of boxes to partition the image
                   - num_boxes = 1: whole image gets same noise level
                   - num_boxes = spatial_size: each pixel gets its own noise level
                   - in between: tiles/boxes of varying sizes
        noise_levels: Tensor of shape (num_boxes,) with noise variance for each box
        device: PyTorch device
        overlap_mode: 'max', 'sum', or 'non_overlap' (default: 'max')
    
    Returns:
        noise_mask: 2D tensor of shape (spatial_size, spatial_size) with spatial noise variance map
        box_size: Size of each box used
        centers: List of (y, x) centers for each box
    """
    # Calculate box size based on num_boxes
    # Strategy: aim for approximately sqrt(num_boxes) boxes per row
    boxes_per_row = int(torch.sqrt(torch.tensor(num_boxes, dtype=torch.float32)).item())
    boxes_per_row = max(1, min(boxes_per_row, spatial_size))
    
    box_size = spatial_size // boxes_per_row
    box_size = max(1, box_size)
    
    # Adjust num_boxes to match actual grid capacity
    actual_max_boxes = boxes_per_row * boxes_per_row
    
    # Initialize noise mask with zeros
    noise_mask = torch.zeros((spatial_size, spatial_size), device=device)
    
    # Generate box centers
    if overlap_mode == 'non_overlap':
        # Grid-based placement for non-overlapping
        centers = []
        for i in range(boxes_per_row):
            for j in range(boxes_per_row):
                if len(centers) >= num_boxes:
                    break
                cy = i * box_size + box_size // 2
                cx = j * box_size + box_size // 2
                centers.append((cy, cx))
            if len(centers) >= num_boxes:
                break
    else:
        # Random placement for overlapping modes
        half_box = box_size // 2
        min_coord = half_box
        max_coord = spatial_size - half_box
        
        centers = []
        for _ in range(min(num_boxes, len(noise_levels))):
            cy = torch.randint(min_coord, max_coord, (1,), device=device).item()
            cx = torch.randint(min_coord, max_coord, (1,), device=device).item()
            centers.append((cy, cx))
    
    # Place each box with its corresponding noise level
    for idx, (cy, cx) in enumerate(centers):
        if idx >= len(noise_levels):
            break
            
        start_y = max(0, cy - box_size // 2)
        end_y = min(spatial_size, start_y + box_size)
        start_x = max(0, cx - box_size // 2)
        end_x = min(spatial_size, start_x + box_size)

        noise_level = noise_levels.variance[idx].flatten().item()

        if overlap_mode == 'max':
            noise_mask[start_y:end_y, start_x:end_x] = torch.maximum(
                noise_mask[start_y:end_y, start_x:end_x],
                torch.tensor(noise_level, device=device)
            )
        elif overlap_mode == 'sum':
            noise_mask[start_y:end_y, start_x:end_x] += noise_level
        else:  # non_overlap
            noise_mask[start_y:end_y, start_x:end_x] = noise_level
    
    return noise_mask, box_size, centers

def build_dynamic_noise_mask_from_array(spatial_size, num_boxes, noise_levels, device, overlap_mode='max'):
    """
    Build a spatial noise mask where each box gets its own noise level.
    
    Args:
        spatial_size: Size of the square spatial dimension (e.g., 28 for MNIST)
        num_boxes: Number of boxes to partition the image
                   - num_boxes = 1: whole image gets same noise level
                   - num_boxes = spatial_size: each pixel gets its own noise level
                   - in between: tiles/boxes of varying sizes
        noise_levels: Tensor of shape (num_boxes,) with noise variance for each box
        device: PyTorch device
        overlap_mode: 'max', 'sum', or 'non_overlap' (default: 'max')
    
    Returns:
        noise_mask: 2D tensor of shape (spatial_size, spatial_size) with spatial noise variance map
        box_size: Size of each box used
        centers: List of (y, x) centers for each box
    """
    # Calculate box size based on num_boxes
    # Strategy: aim for approximately sqrt(num_boxes) boxes per row
    boxes_per_row = int(torch.sqrt(torch.tensor(num_boxes, dtype=torch.float32)).item())
    boxes_per_row = max(1, min(boxes_per_row, spatial_size))
    
    box_size = spatial_size // boxes_per_row
    box_size = max(1, box_size)
    
    # Adjust num_boxes to match actual grid capacity
    actual_max_boxes = boxes_per_row * boxes_per_row
    
    # Initialize noise mask with zeros
    noise_mask = torch.zeros((spatial_size, spatial_size), device=device)
    
    # Generate box centers
    if overlap_mode == 'non_overlap':
        # Grid-based placement for non-overlapping
        centers = []
        for i in range(boxes_per_row):
            for j in range(boxes_per_row):
                if len(centers) >= num_boxes:
                    break
                cy = i * box_size + box_size // 2
                cx = j * box_size + box_size // 2
                centers.append((cy, cx))
            if len(centers) >= num_boxes:
                break
    else:
        # Random placement for overlapping modes
        half_box = box_size // 2
        min_coord = half_box
        max_coord = spatial_size - half_box
        
        centers = []
        for _ in range(min(num_boxes, len(noise_levels))):
            cy = torch.randint(min_coord, max_coord, (1,), device=device).item()
            cx = torch.randint(min_coord, max_coord, (1,), device=device).item()
            centers.append((cy, cx))
    
    # Place each box with its corresponding noise level
    for idx, (cy, cx) in enumerate(centers):
        if idx >= len(noise_levels):
            break
            
        start_y = max(0, cy - box_size // 2)
        end_y = min(spatial_size, start_y + box_size)
        start_x = max(0, cx - box_size // 2)
        end_x = min(spatial_size, start_x + box_size)

        noise_level = noise_levels[idx]

        if overlap_mode == 'max':
            noise_mask[start_y:end_y, start_x:end_x] = torch.maximum(
                noise_mask[start_y:end_y, start_x:end_x],
                torch.tensor(noise_level, device=device)
            )
        elif overlap_mode == 'sum':
            noise_mask[start_y:end_y, start_x:end_x] += noise_level
        else:  # non_overlap
            noise_mask[start_y:end_y, start_x:end_x] = noise_level
    
    return noise_mask, box_size, centers

def power_law_values(omega: torch.Tensor, c: float, alpha: float) -> torch.Tensor:
    omega = omega.float()
    # stable_omega = torch.where(omega == 0, torch.tensor(1e-6, device=omega.device), omega)
    
    value = 1 / (c + omega ** alpha)
    return value

def log_binned_power_law_stds(min_freq_power: float, max_freq_power: float,
                              alpha: torch.Tensor, c: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Calculates the standard deviations of power-law noise within logarithmically spaced frequency bins.
    These standard deviations represent the spectral shape for the noise embedding.

    Args:
        min_freq_power (int): The lowest power of 2 for bin edges (e.g., -2 for 2^-2 = 0.25).
        max_freq_power (int): The highest power of 2 for bin edges (e.g., 3 for 2^3 = 8).
        alpha (torch.Tensor): A batch of power-law exponents. Shape (batch_size,).
        c (torch.Tensor): A batch of constants for the power-law. Shape (batch_size,).
        device (torch.device): The device on which to create tensors.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, num_bins) where each entry
                      is the standard deviation of noise for a specific bin and batch item.
    """
    
    # 1. Define bin edges based on powers of 2
    # Create bin edges on CPU first as Python list, then convert to tensor
    bin_edges_list = [2**p for p in range(min_freq_power, max_freq_power + 1)]
    bin_edges = torch.tensor(bin_edges_list, dtype=torch.float32, device=device)
    # Ensure bin_edges are unique and sorted if there were overlaps/duplicates
    bin_edges = torch.unique(bin_edges).sort().values

    num_bins = len(bin_edges) - 1

    representative_freqs = torch.zeros(num_bins, dtype=torch.float32, device=device)
    for i in range(num_bins):
        lower_edge = bin_edges[i]
        upper_edge = bin_edges[i+1]
        
        if lower_edge == 0:
            # For the first bin starting at 0, use a fraction of the upper bound
            # e.g., 0.5 * upper_edge, or 0.75 * upper_edge
            representative_freqs[i] = 0.5 * upper_edge
        else:
            # For other bins, use geometric mean (sqrt(lower * upper)) for logarithmic scales
            representative_freqs[i] = torch.sqrt(lower_edge * upper_edge)

    
    # Make sure they are on the correct device
    alpha = alpha.to(device)
    c = c.to(device)

    # Expand representative_freqs to match batch_size for element-wise calculation
    # (num_bins) -> (1, num_bins) -> (batch_size, num_bins)
    representative_freqs_expanded = representative_freqs

    # 3. Calculate spectral density at representative frequencies for each batch item
    # This will result in a (batch_size, num_bins) tensor
    spectral_vals = power_law_values(representative_freqs_expanded, c, alpha)
    
    spectral_vals = spectral_vals / spectral_vals.mean()  # Normalize each batch item's spectrum to have mean 1
    # 4. Take square root to get standard deviations
    # per_bin_stds_tensor = torch.sqrt(spectral_vals)

    return spectral_vals

def power_law_spectrum(spatial_size: int, device: torch.device, c: float, alpha: float) -> torch.Tensor:
    """ Returns a (H, W) tensor with the power law spectrum 1 / (c + ||omega|| ** alpha), normalized to have unit variance per pixel. """
    x, y = torch.meshgrid(*(spatial_size * torch.fft.fftfreq(spatial_size, device=device) for _ in range(2)), indexing="xy")  # (H, W) both
    omega_norm = torch.sqrt(x ** 2 + y ** 2)  # (H, W)
    # c = c.to(device)
    # alpha = alpha.to(device)
    # Compute spectrum and normalize.
    spectrum = 1 / (c + omega_norm ** alpha)
    return spectrum / spectrum.mean()


class Blurkernel(torch.nn.Module):
    def __init__(self, blur_type='gaussian', kernel_size=5, std=3.0, device=None):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.seq = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(self.kernel_size//2),
            torch.nn.Conv2d(3, 3, self.kernel_size, stride=1, padding=0, bias=False, groups=3)
        )

        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2,self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k
    
    def get_fourier(self, img_dim):
        kernel_padded = torch.zeros((img_dim, img_dim))
        kernel_padded[:self.kernel_size, :self.kernel_size] = self.k
        # kernel_padded_shift = torch.fft.ifftshift(kernel_padded, dim=[-1, -2])
        k_fourier = torch.fft.fft2(kernel_padded, dim=[-1, -2])
        return k_fourier
    
    def get_fourier_shift(self, img_dim):
        kernel_padded = torch.zeros((img_dim, img_dim))
        kernel_padded[:self.kernel_size, :self.kernel_size] = self.k
        k_fourier = torch.fft.fftshift(torch.fft.fft2(kernel_padded), dim=[-1, -2])
        return k_fourier

    def get_inverse_fourier(self, img_dim):
        # Get the FFT of the padded kernel
        f_kernel = self.get_fourier(img_dim)

        return f_kernel ** -1
    
    def get_inverse_reg_fourier(self, img_dim, epsilon = 1e-8):
        # Get the FFT of the padded kernel
        f_kernel = self.get_fourier(img_dim)

        # Get the magnitude squared
        f_kernel_mag_sq = torch.abs(f_kernel)**2

        # Add a small epsilon for numerical stability
        stabilized_denominator = f_kernel_mag_sq + epsilon

        # Compute the inverse filter
        # The inverse filter is the complex conjugate of the original filter
        # divided by the magnitude squared (plus epsilon)
        inverse_filter_fourier = torch.conj(f_kernel) / stabilized_denominator

        return inverse_filter_fourier


class DownSampling(torch.nn.Module):
    """
    A PyTorch module that combines blurring and downsampling into a single
    convolutional operation.
    """
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size

        # Create the Conv2d layer
        self.conv_layer = torch.nn.Conv2d(in_channels=3,
                         out_channels=3,
                         kernel_size=self.kernel_size,
                         stride=self.stride,
                         padding=0,
                         groups=3) # groups=in_channels makes it a depthwise convolution

        self.init_weights()

    def init_weights(self):
        avg_kernel_value = 1.0 / (self.kernel_size * self.kernel_size)
        self.k = torch.full((3, 1, self.kernel_size, self.kernel_size), avg_kernel_value)      
          # Set the weights and disable gradient updates
        self.conv_layer.weight = torch.nn.Parameter(self.k, requires_grad=False)
          
          # Set the bias to zero and disable gradient updates
        self.conv_layer.bias = torch.nn.Parameter(torch.zeros(3), requires_grad=False)

    def forward(self, input_tensor):
        # Apply the convolution to the input tensor
        output = self.conv_layer(input_tensor)
  
        return output

    
    def get_kernel(self):
        return self.k
    
    def get_fourier(self, img_dim):
        kernel_padded = torch.zeros((img_dim, img_dim))
        kernel_padded[:self.kernel_size, :self.kernel_size] = self.k[0,0,:,:]
        k_fourier = torch.fft.fft2(kernel_padded)
        return k_fourier

    def get_inverse_fourier(self, img_dim):
        # Get the FFT of the padded kernel
        f_kernel = self.get_fourier(img_dim)

        return f_kernel ** -1
    
    def get_inverse_reg_fourier(self, img_dim, epsilon = 1e-8):
        # Get the FFT of the padded kernel
        f_kernel = self.get_fourier(img_dim)

        # Get the magnitude squared
        f_kernel_mag_sq = torch.abs(f_kernel)**2

        # Add a small epsilon for numerical stability
        stabilized_denominator = f_kernel_mag_sq + epsilon

        # Compute the inverse filter
        # The inverse filter is the complex conjugate of the original filter
        # divided by the magnitude squared (plus epsilon)
        inverse_filter_fourier = torch.conj(f_kernel) / stabilized_denominator

        return inverse_filter_fourier
