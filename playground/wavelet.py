import numpy as np
import torch
import pytorch_wavelets
import pywt

class OneLevelWaveletTransform(torch.nn.Module):
    """ Performs one-level wavelet decomposition (self.decompose) or reconstruction (self.reconstruct).
    Uses pytorch_wavelets (supports GPU and autodiff) when possible, but falls back to pywt when pytorch_wavelets is
    unreliable (it has border effects on small image sizes, see test_wavelets).
    Wavelet coefficients at scale j are represented as (*, 4, L/2^j, L/2^j) tensors.
    The first channel corresponds to low frequencies x_j,
    and the following 3 channels correspond to high frequencies x_j_bar.
    """
    def __init__(self, wavelet="db2", mode="periodization"):
        super().__init__()
        self.mode = mode
        self.wavelet = wavelet
        self.ok_shapes = {}  # (spatial shape, decompose?) -> can use pytorch_wavelets over pywt
        self.decompose_module = pytorch_wavelets.DWTForward(wave=self.wavelet, mode=self.mode).cuda()
        self.reconstruct_module = pytorch_wavelets.DWTInverse(wave=self.wavelet, mode=self.mode).cuda()

    def decompose_numpy(self, x):
        """ One-level wavelet decomposition, (*, L, L) to (*, 4, L/2, L/2), using numpy (CPU, no autodiff). """
        x_np = x.cpu().numpy()
        low, high = pywt.dwt2(x_np, wavelet=self.wavelet, mode=self.mode)
        # low is a (*, L/2, L/2) array, high is a tuple of (*, L/2, L/2) arrays.
        y = np.stack((low,) + high, axis=-3)  # (*, 4, L/2, L/2)
        return torch.from_numpy(y).to(dtype=x.dtype, device=x.device)

    def decompose_pytorch(self, x):
        """ One-level wavelet decomposition, (*, L, L) to (*, 4, L/2, L/2), using pytorch (GPU, autodiff). """
        # pytorch_wavelets wants (B, 1, L, L) input.
        batch_shape = x.shape[:-2]
        low, high = self.decompose_module(x.reshape((-1, 1) + x.shape[-2:]))
        # low is (B, 1, L/2, L/2), high is a list whose single element is (B, 1, 3, L/2, L/2).
        y = torch.cat((low, high[0][:, 0]), dim=1)  # (B, 4, L/2, L/2)
        return y.reshape(batch_shape + y.shape[-3:])  # (*, 4, L/2, L/2)

    def decompose(self, x):
        """ One-level wavelet decomposition, (*, L, L) to (*, 4, L/2, L/2). Uses pytorch if possible. """
        if self.is_shape_ok(x, decompose=True):
            return self.decompose_pytorch(x)
        else:
            return self.decompose_numpy(x)

    def reconstruct_numpy(self, x):
        """ One-level wavelet reconstruction, (*, 4, L/2, L/2) to (*, L, L), using numpy (CPU, no autodiff). """
        x_np = x.cpu().numpy()
        channels = tuple(x_np[..., c, :, :] for c in range(4))
        y = pywt.idwt2((channels[0], channels[1:]), wavelet=self.wavelet, mode=self.mode)  # (*, L, L)
        return torch.from_numpy(y).to(dtype=x.dtype, device=x.device)

    def reconstruct_pytorch(self, x):
        """ One-level wavelet reconstruction, (*, 4, L/2, L/2) to (*, L, L), using pytorch (GPU, autodiff). """
        # pytorch_wavelets wants ((B, 1, L/2, L/2), [(B, 1, 3, L/2, L/2))] input.
        batch_shape = x.shape[:-3]
        x = x.reshape((-1,) + x.shape[-3:])  # (B, 4, L/2, L/2)
        y = self.reconstruct_module((x[:, :1], (x[:, None, 1:],)))  # (B, 1, L, L)
        return y.reshape(batch_shape + y.shape[-2:])  # (*, L/2, L/2)

    def reconstruct(self, x):
        """ One-level wavelet reconstruction, (*, 4, L/2, L/2) to (*, L, L). Uses pytorch if possible. """
        if self.is_shape_ok(x, decompose=False):
            return self.reconstruct_pytorch(x)
        else:
            return self.reconstruct_numpy(x)

    def is_shape_ok(self, x, decompose):
        """ Returns whether pytorch_wavelets can be safely used over numpy wavelets for the given spatial shape.
        @param x: pytorch tensor of shape (*, [4,] L, L)
        @param decompose: whether to test decomposition or reconstruction
        @return: whether pytorch_wavelets returns the same output as pywt
        """
        shape = x.shape[-2:]  # (L, L)
        if (shape, decompose) not in self.ok_shapes:
            x = torch.rand((() if decompose else (4,)) + shape, device=x.device)  # (L, L) or (4, L, L)
            numpy = self.decompose_numpy if decompose else self.reconstruct_numpy
            pytorch = self.decompose_pytorch if decompose else self.reconstruct_numpy
            ok = torch.allclose(numpy(x), pytorch(x), atol=1e-06)
            if not ok:
                print(f"Warning: fallback to numpy wavelet {'decomposition' if decompose else 'reconstruction'} for spatial shape {shape}")
            self.ok_shapes[shape, decompose] = ok
        return self.ok_shapes[shape, decompose]
    
def wavelet_transform(input, target_shape):
    wavelets = OneLevelWaveletTransform()

    # shape_x = 4
    # x = torch.randn(1,1,shape_x,shape_x)

    # x_wavelet_prev = covariance.get_matrix()[None, None, None, :, :].cpu()
    # while x_wavelet_prev.shape[-1] != x.shape[-1]:
    if len(input.shape) == 4:
        input = input[:,:,None,:,:]
    x_wavelet_prev = input
    if x_wavelet_prev.shape[-1] == target_shape:
        x_wavelet = x_wavelet_prev.squeeze(1)
    else:
        while x_wavelet_prev.shape[-1] != target_shape:
            x_wavelet = wavelets.decompose(x_wavelet_prev[:,:,0,:,:])
            x_wavelet_prev = x_wavelet
        x_wavelet = x_wavelet.squeeze(1)
    return x_wavelet 