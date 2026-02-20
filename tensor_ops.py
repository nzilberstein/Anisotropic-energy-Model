""" Useful tensor operations. """

from typing import *

import numpy as np
import torch
from einops import *


def unsqueeze(x, dim, num):
    """ Adds `num` dimensions of size 1 at position `dim`. """
    if dim < 0:
        dim += x.ndim + 1  # For instance, dim=-1 should unsqueeze at the last dimension of a tensor.
    return x[(slice(None),) * dim + (None,) * num]


def transpose_view(x, axes, names=None):
    """ Performs a general transpose of x. Returns transposed tensor and new names.
    :param x: numpy array to transpose
    :param names: names of each axis, in the order of x
    :param transpose: iterable of integers corresponding to the new order of axes.
    None correspond to new unsqueezed axes, and axes not present are squeezed (should have a size of 1).
    :return: the transposed array, and names of axes in the new order (if provided)
    """
    # First transpose: put not present at the top, remove Nones.
    not_present = tuple(i for i in range(x.ndim) if i not in axes)
    assert all(x.shape[i] == 1 for i in not_present)
    transpose = not_present + tuple(i for i in axes if i is not None)
    # Then view: indexing and unsqueezes.
    view = (0,) * len(not_present) + tuple(None if i is None else slice(None) for i in axes)

    x = x.transpose(transpose)[view]

    if names is None:
        return x
    else:
        # Handle names: new axes are replaced by dots.
        names = [r"\cdot" if axes[i] is None else names[axes[i]] for i in range(len(axes))]
        return x, names


def channel_reshape(x, channel_shape):
    """ (B, *, H, W) to (B, custom, H, W) """
    return x.reshape((x.shape[0],) + channel_shape + x.shape[-2:])


def space_to_batch(x, shape=None, kernel_size=None, stride=1, padding=0, dilation=1):
    """
    :param x: (B, C') or (B, C, M, N)
    :param shape: optional (M, N) spatial shape to give to 2d input
    :param kernel_size: optional (K, K) patch size to extract from 2d input
    :return: (BP, CK²) where P is the number of patches extracted from (M, N) (defined by input shape or shape param)
    """
    # Do a reshape of spatial shape.
    if shape is None:
        if x.ndim == 2:
            shape = (1, 1)
        elif x.ndim == 4:
            shape = x.shape[-2:]  # (M, N)
        else:
            assert False

    # Permute if there is a spatial shape.
    # Alternative is shape was given as (1, 1) or None and x is 2D
    if shape != (1, 1):
        x = x.reshape((x.shape[0], -1) + shape)  # (B, C, M, N)
        # Now extract patches
        if kernel_size not in [None, 1, (1, 1)]:
            x = torch.nn.functional.unfold(x, kernel_size=kernel_size, stride=stride,
                                           padding=padding, dilation=dilation)  # (B, CK², P)
            x = x.permute(0, 2, 1)  # (B, P, CK^2)
        else:
            x = x.permute(0, 2, 3, 1)  # (B, M, N, C)

    # Flatten and return.
    return x.reshape((-1, x.shape[-1]))  # (BMN, C) or (BP, CK^2)


def flatten_space(x):
    """ (B, C, M, N) to (B, CMN) """
    return x.reshape((x.shape[0], -1))


def optimized_cat(tensors, dim):
    """ Avoids creating a new tensor for lists of length 1. """
    if len(tensors) > 1:
        return torch.cat(tensors, dim)
    else:
        return tensors[0]


class Indexer:
    """ Dummy class used to construct slices with the colon syntax. """
    def __getitem__(self, item):
        return item


idx = Indexer()


def rand_uniform(shape: Tuple[int], low: float | torch.Tensor = 0, high: float | torch.Tensor = 1, device: Optional[torch.device] = None):
    """ Draw from the uniform distribution. """
    return low + (high - low) * torch.rand(size=shape, device=device)

def rand_log_uniform(shape: Tuple[int], low: float | torch.Tensor, high: float | torch.Tensor, device: Optional[torch.device] = None):
    """ Draw from the exponential of a uniform distribution (so that its log is uniform). """
    return torch.exp(rand_uniform(shape, low=log(low), high=log(high), device=device))

def rand_power_uniform(shape: Tuple[int], low: float | torch.Tensor, high: float | torch.Tensor, alpha: float | torch.Tensor, device: Optional[torch.device] = None):
    """ Draw from the power alpha of a uniform distribution (so that its power 1/alpha is uniform). """
    # Invert bounds if alpha is negative.
    if alpha < 0:
        low, high = high, low
    return rand_uniform(shape, low=low ** (1 / alpha), high=high ** (1 / alpha), device=device) ** alpha


def sqrt(x: float | np.ndarray | torch.Tensor) -> float | np.ndarray | torch.Tensor:
    """ Type-agnostic sqrt. """
    if isinstance(x, torch.Tensor):
        return x.sqrt()
    else:
        return np.sqrt(x)

def log(x: float | np.ndarray | torch.Tensor) -> float | np.ndarray | torch.Tensor:
    """ Type-agnostic log. """
    if isinstance(x, torch.Tensor):
        return x.log()
    else:
        return np.log(x)


def detach(x: float | np.ndarray | torch.Tensor) -> float | np.ndarray | torch.Tensor:
    """ Make sure x has no gradient (if it is a tensor). """
    if isinstance(x, torch.Tensor):
        return x.detach()
    else:
        return x



def to_numpy(data):
    """ Converts a tensor/array/list to numpy array.
    Recurse over dictionaries and tuples. Values are left as-is. """
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, list):
        return np.array(data)
    elif isinstance(data, dict):
        return {k: to_numpy(v) for k, v in data.items()}
    elif isinstance(data, tuple):
        return tuple(to_numpy(v) for v in data)
    else:
        return data


def to_tensor(x: float | np.ndarray | torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        return torch.tensor(x)


default_backend = torch


def get_backend(x):
    """ Returns the backend adapted to a given tensor or array. """
    if isinstance(x, torch.Tensor):
        return torch
    else:
        return np


def relative_error(target, estimation):
    """ Computes the relative error between a target and an estimation. """
    def norm(x):
        backend = get_backend(x)
        return backend.linalg.norm(x.flatten())
    return norm(target - estimation) / norm(target)


def contract(tensor, matrix, axis):
    """ tensor is (..., D, ...), matrix is (P, D), returns (..., P, ...). """
    backend = get_backend(tensor)
    t = backend.moveaxis(tensor, source=axis, destination=-1)  # (..., D)
    r = t @ matrix.T  # (..., P)
    return backend.moveaxis(r, source=-1, destination=axis)  # (..., P, ...)


def matmul(x, y, batch_dims=0, contracting_axes=1):
    """ Computes a batched matrix multiplication. This is a transposed dot, but eschews the transpose.
    :param x: (B..., N..., D...)
    :param y: (B..., D..., M...)
    :param batch_dims: number of batch axes (B...,)
    :param contracting_axes: number of contracting axes (D...,)
    :return: (B..., N..., M...)
    """
    d = np.prod(x.shape[-contracting_axes])

    x_shape = x.shape[batch_dims:-contracting_axes]  # (N...,)
    x = x.reshape(x.shape[:batch_dims] + (-1, d))  # (B..., N, D)

    y_shape = y.shape[batch_dims + contracting_axes:]  # (M...,)
    y = y.reshape(y.shape[:batch_dims] + (d, -1))  # (B...., D, M)

    res = x @ y  # (B..., N, M)
    res = res.reshape(res.shape[:batch_dims] + x_shape + y_shape)  # (B..., N..., M...)
    return res


def transpose(matrix):
    """ Transpose a stack of matrices: (*, N, M) to (*, M, N). """
    backend = get_backend(matrix)
    return backend.swapaxes(matrix, -1, -2)


def diagonal(matrix):
    """ Extract the diagonal of a stack of matrices: (B..., D, D) to (B..., D, D). """
    backend = get_backend(matrix)
    assert backend is torch
    return torch.diagonal(matrix, dim1=-1, dim2=-2)


def trace(matrix):
    """ Extract the trace of a stack of matrices: (B..., D, D) to (B...,). """
    return backend.sum(diagonal(matrix), -1)


def decompose(matrix, decomposition="eigh", rank=None):
    """ Performs the decomposition matrix = eigenvectors.T @ diag(eigenvalues) @ dual_eigenvectors.
    matrix is (*, C, D), returns eigenvalues in descending order (*, N), eigenvectors (*, N, C) and their duals (*, N, D).
    N can be smaller than D because we could prune small eigenvalues.
    There are three decompositions:
    - "svd": compute singular values (non-negative) and left and right singular vectors (orthogonal bases)
    - "eig": compute eigenvalues and eigenvectors, dual eigenvectors are the inverse transpose of eigenvectors (requires C = D and real eigenvalues)
    - "eigh": Hermitian case, equivalent to both "svd" and "eig" but faster and more stable (requires C = D)
    rank is an optional upper bound used to prune the number of eigenvalues and eigenvectors.
    """
    def _decompose(matrix):
        backend = get_backend(matrix)
        if decomposition == "svd":
            eigenvectors, eigenvalues, dual_eigenvectors = backend.linalg.svd(matrix, full_matrices=False)
            # Shapes are (*, N), (*, C, N), (*, N, D) with N = min(C, D). Singular values are in descending order.
            eigenvectors = transpose(eigenvectors)  # (*, N, C)
        else:
            sym = dict(eig=False, eigh=True)[decomposition]
            method = backend.linalg.eigh if sym else backend.linalg.eig
            eigenvalues, eigenvectors = method(matrix)  # eigenvalues (*, N) ascending, eigenvectors (*, C, N)

            # Sort in descending order.
            if sym:
                if backend == np:
                    eigenvalues, eigenvectors = eigenvalues[..., ::-1], eigenvectors[..., ::-1]
                else:
                    eigenvalues, eigenvectors = eigenvalues.flip(-1), eigenvectors.flip(-1)
            else:
                assert np.isreal(eigenvalues.dtype)  # Complex eigenvalues not dealt with for now.
                I = backend.argsort(-eigenvalues, axis=-1)  # (*, N)
                take_along = torch.take_along_dim if backend == torch else np.take_along_axis
                eigenvalues, eigenvectors = take_along(eigenvalues, I, axis=-1), \
                                            take_along(eigenvectors, I[..., None, :], axis=-1)

            eigenvectors = transpose(eigenvectors)  # (*, N, C)

            if sym:
                dual_eigenvectors = eigenvectors
            else:
                dual_eigenvectors = transpose(backend.linalg.inv(eigenvectors))  # (*, N, D)

        # Prune eigenvalues that are theoretically zero because of low-rank.
        if rank is not None:
            eigenvalues = eigenvalues[..., :rank]
            eigenvectors = eigenvectors[..., :rank, :]
            dual_eigenvectors = dual_eigenvectors[..., :rank, :]

        # Prune eigenvalues that are too small (deprecated because dual_eigenvectors and batch axes)
        # I = eigenvalues >= eigenvalues[0] / 1e6
        # eigenvalues, eigenvectors = eigenvalues[I], eigenvectors[I]

        return eigenvalues, eigenvectors, dual_eigenvectors

    try:
        eig = _decompose(matrix)
    except RuntimeError as ex:
        print(f"RuntimeError ({ex}) while computing {decomposition} decomposition of shape {matrix.shape}, retrying with numpy")
        matrix_np = matrix.cpu().numpy()
        eigs_np = _decompose(matrix_np)
        eig = tuple(torch.from_numpy(e_np).to(dtype=matrix.dtype, device=matrix.device) for e_np in eigs_np)

    return eig


def reconstruct(eigenvalues, eigenvectors, dual_eigenvectors):
    """ eigenvalues is (*, N,), eigenvectors are (*, N, C) and duals are (*, N, D). Returns (*, C, D) matrices.
    Assumes real eigenvalues and eigenvectors. """
    # Reconstruct with eigenvectors.T @ diag(eigenvalues) @ dual_eigenvectors.
    return transpose(eigenvectors) @ (eigenvalues[..., :, None] * dual_eigenvectors)


class DecomposedMatrix:
    """ Class for holding decomposed matrices (diagonalization or SVD). Only real arrays are supported.
    Typical usage: DecomposedMatrix(m).apply(some function).matrix.
    """

    def __init__(self, matrix=None, decomposition="eigh", rank=float("inf"),
                 eigenvalues=None, eigenvectors=None, dual_eigenvectors=None):
        """ One typically either gives matrix or eigenvalues, eigenvectors and dual_eigenvectors.
        :param matrix: (*, C, D)
        :param decomposition: any of "eig", "eigh", or "svd"
        :param rank: optional upper bound on the rank to limit the number of eigenvalues and eigenvectors
        :param eigenvalues: (*, R) where R is the rank
        :param eigenvectors: (*, R, C)
        :param dual_eigenvectors (*, R, D)
        """
        self._matrix = matrix  # (*, C, D)
        self.decomposition = decomposition
        self.rank: int = min(min(*matrix.shape[-2:]) if matrix is not None else float("inf"), eigenvalues.shape[-1] if eigenvalues is not None else float("inf"), rank)  # Never inf.
        self._eigenvalues = eigenvalues  # (*, R), descending
        self._eigenvectors = eigenvectors  # (*, R, C)
        self._dual_eigenvectors = eigenvectors if dual_eigenvectors is None else dual_eigenvectors  # (*, R, D)

    @property
    def backend(self):
        return get_backend(self._matrix if self._matrix is not None else self._eigenvectors)

    @property
    def matrix(self):
        return self.reconstruct()._matrix

    @property
    def eigenvalues(self):
        """ (*, N) """
        return self.decompose()._eigenvalues

    @property
    def eigenvectors(self):
        """ (*, N, C) """
        return self.decompose()._eigenvectors

    @property
    def dual_eigenvectors(self):
        """ (*, N, D) """
        return self.decompose()._dual_eigenvectors

    @property
    def singular_values(self):
        """ (*, N), alias for eigenvalues. """
        return self.eigenvalues

    @property
    def left_singular_vectors(self):
        """ (*, N, C), alias for eigenvectors. """
        return self.eigenvectors

    @property
    def right_singular_vectors(self):
        """ (*, N, D), alias for dual_eigenvectors. """
        return self.dual_eigenvectors

    def decompose(self, do=True) -> "DecomposedMatrix":
        """ Diagonalizes the matrix (if not None and not cached). do=False is a no-op provided for convenience. """
        if do and self._eigenvalues is None:
            self._eigenvalues, self._eigenvectors, self._dual_eigenvectors = decompose(self._matrix, decomposition=self.decomposition, rank=self.rank)
            # Sets the true rank (min(rank, C, D)) as opposed to the optional upper bound provided.
            self.rank = self._eigenvalues.shape[-1]
        return self

    def reconstruct(self, do=True) -> "DecomposedMatrix":
        """ Reconstructs the matrix (if not None and not cached). do=False is a no-op provided for convenience. """
        if do and self._matrix is None:
            self._matrix = reconstruct(self._eigenvalues, self._eigenvectors, self._dual_eigenvectors)
        return self

    def with_eigenvalues(self, eigenvalues) -> "DecomposedMatrix":
        """ Replaces the eigenvalues with a new set (*, N). * can be different as long as it broadcasts. """
        return DecomposedMatrix(
            decomposition=self.decomposition, rank=self.rank,
            eigenvalues=eigenvalues, eigenvectors=self.eigenvectors, dual_eigenvectors=self.dual_eigenvectors,
        )

    def with_eigenvectors(self, eigenvectors) -> "DecomposedMatrix":
        """ Replaces the eigenvectors with a new set (*, N, D) (assumes symmetric).
        * can be different as long as it broadcasts. """
        assert self.decomposition == "eigh"
        return DecomposedMatrix(
            decomposition=self.decomposition, rank=self.rank,
            eigenvalues=self.eigenvalues, eigenvectors=eigenvectors,
        )

    def apply(self, f) -> "DecomposedMatrix":
        """ Applies a function on the matrix via its eigenvalues.
        :param f: a scalar function (which can depend on the matrix) : (*, N) -> (*, N)
        """
        return self.with_eigenvalues(eigenvalues=f(self.eigenvalues))

    def sqrt(self) -> "DecomposedMatrix":
        """ Computes the symmetric matrix square root. """
        return self.apply(lambda t: self.backend.sqrt(self.backend.abs(t)))

    def rsqrt(self) -> "DecomposedMatrix":
        """ Computes the reciprocal of the symmetric matrix square root. """
        return self.apply(lambda t: 1 / self.backend.sqrt(t))

    def orthogonalize(self) -> "DecomposedMatrix":
        """ Computes the projection to orthogonal matrices by setting all eigenvalues to 1. """
        return self.apply(self.backend.ones_like)

    def project(self, rank) -> "DecomposedMatrix":
        """ Lowers the rank of the matrix. rank=None is a no-op. """
        if rank is None or (self.rank is not None and self.rank <= rank):
            return self
        else:
            self.decompose()  # We need to decompose the matrix
            return DecomposedMatrix(
                decomposition=self.decomposition, rank=rank if self.rank is None else min(rank, self.rank),
                eigenvalues=self.eigenvalues[..., :rank],  eigenvectors=self.eigenvectors[..., :rank, :],
                dual_eigenvectors=self.dual_eigenvectors[..., :rank, :],
            )

    def random_rotation(self) -> "DecomposedMatrix":
        """ Performs a random rotation A -> O A O^T. Assumes symmetric.
        Note that each matrix in the batch has a different rotation. """
        assert self.decomposition == "eigh"
        if self._matrix is None:
            # Replace eigenvectors with random ones
            return self.with_eigenvectors(random_orthogonal_like(self._eigenvectors))
        else:
            # Rotate the matrix, and its eigenvectors if pre-computed.
            o = random_orthogonal_like(self._matrix)  # (..., D_new, D_old)
            return DecomposedMatrix(
                matrix=o @ self._matrix @ o.mT,  # (..., D_new, D_new)
                decomposition=self.decomposition, rank=self.rank,
                eigenvalues=self._eigenvalues,
                eigenvectors=(None if self._eigenvectors is None else self._eigenvectors @ o.mT),  # (..., R, D_new)
            )

    def participation_ratio(self):
        # NOTE: could also be computed from the matrix entries for symmetric matrices, but at a quadratic cost as opposed to linear after diagonalization.
        return self.eigenvalues.sum(-1) ** 2 / (self.eigenvalues ** 2).sum(-1)

    def __getitem__(self, item) -> "DecomposedMatrix":
        """ Indexes simultaneously matrices, eigenvalues and eigenvectors to select a subset along the batch axis. """
        def index(m):
            return m[item] if m is not None else None
        return DecomposedMatrix(
            matrix=index(self._matrix), decomposition=self.decomposition, rank=self.rank,
            eigenvalues=index(self._eigenvalues), eigenvectors=index(self._eigenvectors),
            dual_eigenvectors=index(self._dual_eigenvectors),
        )

    @property
    def T(self) -> "DecomposedMatrix":
        """ Returns a transposed view of this DecomposedMatrix (swaps eigenvectors and dual_eigenvectors). """
        return DecomposedMatrix(
            matrix=self._matrix.mT if self._matrix is not None else None, decomposition=self.decomposition, rank=self.rank,
            eigenvalues=self._eigenvalues, eigenvectors=self._dual_eigenvectors, dual_eigenvectors=self._eigenvectors,
        )

    @property
    def trace(self):
        # Prefers eigenvalues to leverage small rank.
        if self._eigenvalues is not None:
            return self._eigenvalues.sum(-1)
        else:
            return einsum(self._matrix, "... i i -> ...")

    def to(self, dtype=None, device=None) -> "DecomposedMatrix":
        """ Casts the DecomposedMatrix into a different dtype/device. """
        def t(m):
            return m.to(dtype=dtype, device=device) if m is not None else None
        return DecomposedMatrix(
            matrix=t(self._matrix), decomposition=self.decomposition, rank=self.rank,
            eigenvalues=t(self._eigenvalues), eigenvectors=t(self._eigenvectors),
            dual_eigenvectors=t(self._dual_eigenvectors),
        )

    @property
    def dtype(self):
        return self._matrix.dtype if self._matrix is not None else self._eigenvectors.dtype

    @property
    def device(self):
        return self._matrix.device if self._matrix is not None else self._eigenvectors.device


def orthogonalize(matrix):
    """ Returns an orthogonalized version of (*, C, D) matrix with the same shape. """
    return DecomposedMatrix(matrix, decomposition="svd").orthogonalize().matrix


def empirical_covariance(x, rank=float("inf")) -> DecomposedMatrix:
    """ Compute covariance along second-to-last axis: (*, N, D) to (*, D, D). rank is an optional upper-bound on the rank.
    Efficient optimization: performs an SVD of x if N < D.
    """
    backend = get_backend(x)
    assert backend == torch

    n, d = x.shape[-2:]
    rank = min(n, d, rank)

    if n < d:
        # SVD more efficient: we do not compute the large (D, D) matrix.
        x = DecomposedMatrix(x, decomposition="svd", rank=rank)
        # x = U S V^T -> C = X^T X / N = V S²/N V^T
        return DecomposedMatrix(eigenvalues=x.eigenvalues ** 2 / n, eigenvectors=x.dual_eigenvectors,
                                dual_eigenvectors=x.dual_eigenvectors,
                                decomposition="eigh", rank=rank)
    else:
        # Computing the covariance matrix is more efficient.
        return DecomposedMatrix(x.mT @ x / n, decomposition="eigh",
                                rank=n if rank is None else min(x.shape[-2], rank))


def random_gaussian(shape, dtype=None, device=None, cov=None, backend=torch):
    """ Returns Gaussian samples of the given covariance.
    :param shape: (*, D)
    :param cov: optional, (*, D, D) DecomposedMatrix or tensor (should be full-rank or suitably pruned)
    :return: Gaussian samples of given shape
    """

    if cov is not None:
        if not isinstance(cov, DecomposedMatrix):
            cov = DecomposedMatrix(cov, decomposition="eigh")
        backend = cov.backend

    if backend == torch:
        if cov is not None:
            dtype = cov.matrix.dtype
            device = cov.matrix.device
        x = torch.randn(shape, dtype=dtype, device=device)
    else:
        x = np.random.standard_normal(shape)

    if cov is not None:
        x = x @ cov.sqrt().matrix

    return x


def random_gaussian_like(x):
    """ Returns Gaussian samples of the same shape and covariance as x (*, D) (covariance is (D, D)). """
    return random_gaussian(shape=x.shape, cov=empirical_covariance(x))


def random_orthogonal(shape, dtype=None, device=None, backend=torch):
    """ Returns random orthogonal matrices of the given shape (*, C, D). """
    return orthogonalize(random_gaussian(shape, dtype=dtype, device=device, backend=backend))


def random_orthogonal_like(x):
    """ Returns random orthogonal matrices of the same shape as x (*, C, D). """
    return random_orthogonal(shape=x.shape, dtype=x.dtype, device=x.device)
