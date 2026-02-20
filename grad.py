""" Gradient-related functions. """

from typing import *

import numpy as np
import torch
from torch.func import jacrev, jacfwd, vmap

from tensor_ops import DecomposedMatrix

import pdb

def compute_grad(f, xs: Tuple[torch.Tensor], type_cov_list: List[str], grad_output=None, create_graph=True, **kwargs):
    """ Compute the gradient (or vector Jacobian product) of f at x, in a differentiable manner.
    @param f: a differentiable function, in_shape to out_shape
    @param xs: tuple of inputs point at which to estimate the gradient, in_shape
    @param grad_output: vector in vector Jacobian product, out_shape
    (similar to computing gradient of <w, f(x)>, optional if f is a scalar function)
    @param create_graph: whether to create a graph to differentiate through the gradients (default)
    @param kwargs: additional keyword arguments are passed to f
    @return: f(x), out_shape, and (grad_output^T) grad f(x), in_shape
    """
    x_requires_grad = tuple(x.requires_grad for x in xs)    
    for x in xs:
        x.requires_grad = True
    with torch.set_grad_enabled(True):
        y = f(type_cov_list, *xs, **kwargs)
        grads = torch.autograd.grad(y, xs, create_graph=create_graph, grad_outputs=grad_output, materialize_grads=True)
    # Compute norm of grads[1]
    # norm_grad1 = torch.norm(grads[1])
    # Print norm of grads[1]
    # print(f"Norm of grads[1]: {norm_grad1.item()}")
    for i, x in enumerate(xs):
        x.requires_grad = x_requires_grad[i]
    return y, grads


def compute_jacobian(denoiser: torch.nn.Module, x: torch.Tensor, t: torch.Tensor, full_batch=False,
                     symmetrize=True, backward=True) -> DecomposedMatrix:
    """ Compute the denoiser Jacobian on a batch of images, optionally symmetrizing it, and diagonalizing it.

    Args:
        model: denoiser
        x: (*, C, H, W)
        full_batch: whether batch the entire computation or loop over single images (default due to memory constraints)
        symmetrize: whether to make the Jacobian symmetric
        backward: whether to use backward or forward differentiation

    Returns:
        jacobian, of shape (*, CHW, CHW), as a DecomposedMatrix
    """
    batch_shape, signal_shape = x.shape[:-3], x.shape[-3:]  # (*) and (C, H, W)
    signal_dim = np.prod(signal_shape)

    x = x.reshape((-1, signal_dim))  # (B*, CHW)
    forward = lambda y, t: denoiser(y.reshape((1, *signal_shape)), t.reshape((1,))).reshape((signal_dim,))  # (CHW,) to (CHW,)
    jacobian_single = (jacrev if backward else jacfwd)(forward)  # (H, W) to (H, W, H, W)

    if full_batch:
        jacobian_batched = vmap(jacobian_single)  # (B, CHW) to (B, CHW, CHW)
        jacobian = jacobian_batched(x, t)  # (B*, CHW, CHW)
    else:
        jacobian = torch.stack([jacobian_single(y, s) for y, s in zip(x, t)])  # (B*, CHW, CHW)

    jacobian = jacobian.reshape((*batch_shape, signal_dim, signal_dim))  # (*, CHW, CHW)
    if symmetrize:
        jacobian = (jacobian + jacobian.mT) / 2  # (*, CHW, CHW)
    return DecomposedMatrix(jacobian, decomposition="eigh" if symmetrize else "svd")
