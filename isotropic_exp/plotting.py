""" Plotting functions (for tensorboard visualization). """

import math
from typing import *

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
from matplotlib import pyplot as plt
import torch
from einops import *

from noise import NoiseLevel, DenoisingError
from tensor_ops import *


def plot_lines(*lines, x=None, same_x=False, label="", smoothing=None, show_points=True, show_mean=True, show_std=True, subsample_factor=None,
               title=None, xscale="linear", yscale="linear",
               has_fig=False, figsize=None, colormap=None, colorbar=False, color=None, xlabel=None, ylabel=None, save=None, legend=True, aspect=None,
               xlim=None, ylim=None, alpha=None, xticks=None, xtick_labels=None, yticks=None, ytick_labels=None, marker=None, linestyle=None, linewidth=None, grid=True, **named_lines):
    """ Creates a simple one-off plots with lines.
    :param lines: sequence of unnamed arrays of y-values, assuming x going from 1 to n, or (x, y)
    :param x: optional x to use for all lines (should be the same)
    :param same_x: whether to rescale the x's to [0, 1]
    :param title: title of the plot
    :param colors: list of colors for each line
    :param colormap: assign a color to lines based on its rank
    :param xscale: scale of the x-axis
    :param yscale: scale of the y-axis
    :param has_fig: whether to create/show the plot (useful for subplots)
    :param figsize: size of figure (width, height) in inches, if has_fig is False
    :param named_lines: dictionary of named lines
    """
    if not has_fig:
        plt.figure(figsize=figsize)

    if isinstance(xscale, str):
        xscale = (xscale, {})
    plt.xscale(xscale[0], **xscale[1])
    if isinstance(yscale, str):
        yscale = (yscale, {})
    scale = matplotlib.scale.scale_factory(yscale[0], plt.gca().yaxis, **yscale[1])
    plt.yscale(yscale[0], **yscale[1])

    all_lines = [(label, line) for line in lines] + list(named_lines.items())
    all_lines = [(name, to_numpy(line)) for name, line in all_lines]

    if not isinstance(linestyle, list):
        linestyle = [linestyle] * len(all_lines)
    if not isinstance(linewidth, list):
        linewidth = [linewidth] * len(all_lines)

    if x is not None:
        x = to_numpy(x)

    if color is None and colormap is not None:
        color = matplotlib.cm.get_cmap(colormap)(np.linspace(0, 1, len(all_lines)))
        if colorbar:
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=matplotlib.colors.Normalize(vmin=1, vmax=len(all_lines)))
            plt.colorbar(sm)
    elif not isinstance(color, list):
        color = [color] * len(all_lines)

    for i, (name, line) in enumerate(all_lines):
        if isinstance(line, tuple):
            this_x, line = line
        elif x is not None:
            this_x = x
        elif same_x:
            this_x = np.linspace(0, 1, len(line))
        else:
            this_x = 1 + np.arange(len(line))

        this_color = color[i] if color is not None and i < len(color) else None
        this_linestyle = linestyle[i]
        this_linewidth = linewidth[i]

        if smoothing is not None:
            sigma = smoothing * len(line)
            smooth = lambda y: gaussian_filter(y, sigma=sigma, mode="nearest")

            line_y = scale.get_transform().transform(line)
            line_mean_y = smooth(line_y)
            line_std_y = np.sqrt(smooth((line_y - line_mean_y) ** 2))
            line_minus, line_mean, line_plus = [scale.get_transform().inverted().transform(line_mean_y + s * line_std_y) for s in [-1, 0, 1]]

            if show_mean:
                plotted_line, = plt.plot(this_x, line_mean, label=name, color=this_color, linestyle=this_linestyle,
                                         linewidth=this_linewidth)
                this_color = plotted_line.get_color()
            if show_std:
                plt.fill_between(this_x, line_minus, line_plus, color=this_color, alpha=0.3)
            if show_points:
                if subsample_factor is not None:
                    this_x = this_x[::subsample_factor]
                    line = line[::subsample_factor]
                plt.scatter(this_x, line, marker="+", color=this_color)
        else:
            plt.plot(this_x, line, label=name, color=this_color, marker=marker, alpha=alpha,
                     linestyle=this_linestyle, linewidth=this_linewidth)

    plt.grid(visible=grid, which="both", axis="both")
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.xticks(xticks, xtick_labels)
    plt.yticks(yticks, ytick_labels)
    if title is not None:
        plt.title(title)
    if legend and len(named_lines) > 0:
        plt.legend()
    if aspect is not None:
        plt.gca().set_aspect(aspect)
    # plt.tight_layout()

    if save is not None:
        savefig(save)
    if not has_fig:
        plt.show()


def plot_img(img, title=None, has_fig=False, figsize=None, inches_per_pixel=0.1, save=None,
             xticks=None, xtick_labels=None, yticks=None, ytick_labels=None, xlabel=None, ylabel=None, contour=False, sigma=3,
             cmap=None, bound=None, vmin=None, vmax=None, colorbar=False, p=1, l=0.6, s=1.0, alpha_max=1.0):
    """ Plots the image `img` (H..., W..., [C,]). The axes of H and W represent nested subimages.
    `bound` is the optional scale to use for the colormap. Also add an optional title `title` and subdivision lines.
    :param (x|y)ticks: tick labels for the x/y axes at every pixel location
    ;param (x|y)label: legend for the x/y axes.
    """
    img = handle_img(to_numpy(img), p, l, s, alpha_max)
    n = img.ndim // 2

    if cmap == "gray":
        img = rescale_imgs(img, soft=True, dims=(n - 1, 2 * n - 1))  # Do a soft rescale to [0, 1] for each tiny image.

    H_shape, W_shape, C = img.shape[:n], img.shape[n:-1], img.shape[-1]
    H, W = np.prod(H_shape), np.prod(W_shape)
    img = img.reshape((H, W, C))

    if not has_fig:
        if figsize is None:
            # TODO change dpi of figure instead
            figsize = inches_per_pixel * W, inches_per_pixel * H
        plt.figure(figsize=figsize)

    if C == 1:
        if cmap is None:
            cmap = "bwr"
        if cmap == "bwr" and bound is None:
            bound = np.abs(img).max()
        if bound is not None:
            vmin = -bound
            vmax = bound
        kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        kwargs = dict()
    im = plt.imshow(img, **kwargs, interpolation="nearest")
    if colorbar:
        add_colorbar(im)

    if contour:
        img_contour = img[:, :, 0]
        img_contour = gaussian_filter(img_contour, sigma=(sigma, sigma), mode="nearest")
        plt.contour(img_contour, cmap=cmap, vmin=vmin, vmax=vmax)

    # Draw lines to signify sub-images.
    for shape, method in [(H_shape, plt.axhline), (W_shape, plt.axvline)]:
        strides = np.cumprod(shape[::-1])  # number of pixels of sub-images of all depths, starting from smallest to largest
        # Last element of stride is the size of the entire image.
        for i, stride in enumerate(strides[:-1], start=1):
            for pos in np.arange(0, strides[-1] + 1, stride):
                method(pos - 0.5, c="gray", lw=i * 1.5)  # pos is the center of the pixel, -0.5 gives the bottom left corner.
                # 1.5 is the default linewidth, multiply by the nesting level to get larger and larger lines.

    format_plot(has_fig=has_fig, save=save, title=title, xticks=xticks, xtick_labels=xtick_labels,
                yticks=yticks, ytick_labels=ytick_labels, xlabel=xlabel, ylabel=ylabel)


def format_plot(has_fig=False, save=None, title=None, xticks=None, xtick_labels=None,
                yticks=None, ytick_labels=None, xlabel=None, ylabel=None, xscale=None, yscale=None):
    """ Format axes and labels, and show the plot if necessary. """
    if xticks is not None:
        plt.xticks(ticks=xticks, labels=xtick_labels, rotation="vertical")
    else:
        plt.xticks(ticks=[], labels=[])
    if yticks is not None:
        plt.yticks(ticks=yticks, labels=ytick_labels)
    else:
        plt.yticks(ticks=[], labels=[])
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if xscale is not None:
        plt.xscale(xscale)
    if yscale is not None:
        plt.yscale(yscale)
    if title is not None:
        plt.title(title)
    if save is not None:
        savefig(save)
    if not has_fig:
        plt.show()


def add_colorbar(im, aspect=20, pad_fraction=1.0, label=None, **kwargs):
    """ Add a vertical color bar to an image plot.
    :param im: output of imshow
    :param aspect: aspect ratio of the colorbar
    :param pad_fraction: padding between the image and the colorbar, as a fraction of the colorbar's width
    :param label: add a label to the colorbar
    :param kwargs: additional kwargs to pass to the colorbar call
    """
    from mpl_toolkits import axes_grid1
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    colorbar = im.axes.figure.colorbar(im, cax=cax, **kwargs)

    if label is not None:
        colorbar.ax.get_yaxis().labelpad = 15
        colorbar.ax.set_ylabel(label, rotation=90)

    return colorbar


def arrange_images(images, aspect_ratio=np.sqrt(2)):
    """ (N, H..., W..., [C]) to (h, H..., w, W..., [C]), where w*W/h*H is approximately aspect_ratio. """
    images = to_numpy(images)

    n = (images.ndim - 1) // 2
    N, H_shape, W_shape, C_shape = images.shape[0], images.shape[1:n+1], images.shape[n+1:2*n+1], images.shape[2*n+1:]
    # print(f"{N=} {H_shape=} {W_shape=} {C_shape=}")

    H, W = np.prod(H_shape), np.prod(W_shape)
    # w/h approx H/W * aspect_ratio, and w*h >= N
    # Leads to w = sqrt(N * H/W * apect_ratio) then rounding up
    w = math.ceil(np.sqrt(N * H / W * aspect_ratio))
    h = math.ceil(N / w)
    # Sometimes w can be reduced without changing h:
    w = math.ceil(N / h)

    res = np.concatenate((images, np.zeros((h * w - N,) + H_shape + W_shape + C_shape)))  # (h * w, H..., W...)
    res = np.moveaxis(res.reshape((h, w) + H_shape + W_shape + C_shape), source=1, destination=n + 1)  # (h, H..., w, W...)
    # print(f"Handled {images.shape=} to {res.shape=}")
    return res


def handle_img(img, p=1, l=0.6, s=1.0, alpha_max=1.0):
    """ Put one or several real/complex images in the right format: (H..., W..., [C,]) to (H..., W..., C) real. """
    if img.dtype in [np.complex64, np.complex128]:
        img = complex_to_rgb(img, p, l, s, alpha_max)
    if img.ndim % 2 == 0:
        img = img[..., None]  # grayscale image
    return img


def complex_to_rgb(x, p, l, s, alpha_max):
    """ Compute the color of the complex coefficients,
    with argument -> hue and modulus -> luminance, at fixed saturation. (*) to (*, 3). """
    from colorsys import hls_to_rgb

    mod = np.abs(x) ** p  # (*)
    alpha = alpha_max * mod / mod.max()  # (*)

    arg = np.angle(x)  # (*)
    h = (arg + np.pi) / (2 * np.pi) + 0.5  # (*)

    c = np.array(np.vectorize(hls_to_rgb)(h, l, s)).transpose(tuple(1 + i for i in range(x.ndim)) + (0,))  # (*, 3)
    c = np.concatenate((c, alpha[..., None]), axis=-1)  # (*, 4)
    return c


def rescale_imgs(x: torch.Tensor | np.ndarray, soft: bool = False, dims=(-1, -2, -3)) -> torch.Tensor | np.ndarray:
    """ Rescale to [0, 1] over the given axes. If soft, leave alone images already in [0, 1]. """
    backend = get_backend(x)
    kwargs = dict(dim=dims, keepdim=True) if backend == torch else dict(axis=dims, keepdims=True)

    lower = backend.amin(x, **kwargs)
    upper = backend.amax(x, **kwargs)

    if soft:
        lower = backend.minimum(lower, backend.zeros_like(lower))
        upper = backend.maximum(upper, backend.ones_like(upper))

    x = (x - lower) / (upper - lower)
    return x


def rescale_img_diffs(x: torch.Tensor | np.ndarray, dims=(-1, -2, -3)) -> torch.Tensor | np.ndarray:
    """ Rescale to fit in [-1, 1] over the given axes, preserving zero. """
    backend = get_backend(x)
    kwargs = dict(dim=dims, keepdim=True) if backend == torch else dict(axis=dims, keepdims=True)
    bound = backend.amax(backend.abs(x), **kwargs)
    return x / bound


def plot_correlations(vectors, title=None, figsize=(10, 10), kernel=None, normalize=True, abs=True,
                      plot=True, save=None, colorbar=True, labels=None):
    """ Plot the correlation between sets of vectors.
    :param vectors: (N, M, D) for N experiments, each with M vectors in dimension D
    :param kernel: optional kernel for dot products, can be a dictionary of kernels
    :return: the full correlation matrix (N, M, N, M)
    """
    if kernel is not None:
        assert not normalize  # Normalization not supported for (non-symmetric) kernels.
    if not isinstance(kernel, dict):
        kernel = {(i, j): kernel for i in range(len(vectors)) for j in range(len(vectors))}


    matrix = torch.stack([
        torch.stack([EuclideanDotProduct(kernel=kernel[i,j]).dot(vectors[i], vectors[j], normalize=normalize, abs=abs)
                  for j in range(len(vectors))], dim=1) for i in range(len(vectors))], dim=0)  # (N, M, N, M)
    if plot:
        num_exps, num_vectors = len(vectors), len(vectors[0])  # vectors might be a list of arrays
        ticks = num_vectors * (1/2 + np.arange(num_exps))
        plot_img(matrix, colorbar=colorbar, cmap="viridis", figsize=figsize, vmin=0, vmax=1,
                 title=title, save=save, xticks=ticks, yticks=ticks, xtick_labels=labels, ytick_labels=labels)
    return matrix


def plot_performance_curves(curves, x="psnr", y="psnr", plot_baselines=True) -> plt.Figure:
    """ Plot one or several performance curves, and return the figure.

    Args:
        curves: list of (noise_level, denoising_error, plot_kwargs) (kwargs are passed to plot_lines)
        x, y: units for x and y axes
    """
    # TODO: indicate limiting scales based on imge size and resolution, and training noise level distribution/weighting.
    unit_to_scale = lambda unit: "linear" if unit in ["snr", "psnr"] else "log"
    unit_to_name = dict(snr="SNR", psnr="PSNR", mse="MSE", var="variance", std="standard deviation")
    x_prefix = "Input" if x in DenoisingError.units else "Noise"
    y_prefix = "Output"
    assert y in DenoisingError.units

    plt.figure(figsize=(7, 7))
    for noise_levels, denoising_errors, line_kwargs in curves:
        noise_levels: NoiseLevel
        denoising_errors: DenoisingError
        line_kwargs: dict[str, Any]
        plot_lines(denoising_errors.to_unit(y), x=noise_levels.to_unit(x), **line_kwargs, has_fig=True, marker=".")

    # Save plot limits.
    xlim = plt.xlim()
    ylim = plt.ylim()

    # Also plot two trivial performance curves: identity and mean.
    if plot_baselines:
        noise_range = noise_levels[[0, -1]]
        plot_lines(noise_range.to_unit(y), x=noise_range.to_unit(x), color="black", has_fig=True)
        dataset_info = noise_range.dataset_info
        plot_lines(DenoisingError(dataset_info=dataset_info, mse=torch.full((2,), dataset_info.variance)).to_unit(y), x=noise_range.to_unit(x), color="black", has_fig=True)

    format_plot(has_fig=True,
        xlabel=f"{x_prefix} {unit_to_name[x]}", xscale=unit_to_scale(x),
        ylabel=f"{y_prefix} {unit_to_name[y]}", yscale=unit_to_scale(y),
    )

    # Restore plot limits.
    plt.xlim(xlim)
    plt.ylim(ylim)
    # plt.xlim(sorted(noise_range.to_unit(x)))
    # What to put at ylim?

    plt.tight_layout()
    return plt.gcf()


def plot_jacobian_eigenvalues(eigenvalues: torch.Tensor, noise_levels: NoiseLevel, unit: str, yscale="log") -> plt.Figure:
    """ Plot the eigenvalues of the denoiser Jacobian or the Hessian (L, R). """
    lines = {f"{unit} = {noise_level.to_unit(unit)}": eigenvalues for noise_level, eigenvalues in zip(noise_levels, eigenvalues)}

    plt.figure(figsize=(7, 7))
    plot_lines(**lines , xscale="log", yscale=yscale, xlabel="Rank", ylabel="Eigenvalue", colormap="viridis", has_fig=True)
    if yscale == "log":
        plt.ylim(1e-2, max(line.max().item() for line in lines.values()))
    plt.tight_layout()
    return plt.gcf()


def arrange_jacobian_eigenvectors(jacobian: DecomposedMatrix, noise_levels: NoiseLevel) -> torch.Tensor:
    """ Arrange the eigenvectors of the Jacobian (L, CHW, CHW) into a (C, LH, NW) image. """
    dataset_info = noise_levels.dataset_info

    eigenvectors = jacobian.eigenvectors  # (L, R, CHW)
    idx = 2 ** torch.arange(np.log2(eigenvectors.shape[1] + 0.5), dtype=torch.int) - 1  # (N,)
    eigenvectors = eigenvectors[:, idx, :]  # (L, N, CHW)

    # Rescale to [0, 1].
    bound = torch.abs(eigenvectors).amax(dim=-1, keepdim=True)  # (L, N)
    eigenvectors = 0.5 + 0.5 * eigenvectors / bound  # (L, N, CHW)
    # Apply colormap (ignore alpha channel).
    cmap = matplotlib.colormaps.get_cmap("bwr")
    eigenvectors = torch.from_numpy(cmap(eigenvectors.detach().cpu().numpy())).to(device=jacobian.device)[..., :3]  # (L, N, CHW, 3)

    eigenvectors = rearrange(eigenvectors, "l n (h w) c -> c (l h) (n w)", h=dataset_info.spatial_size, w=dataset_info.spatial_size)

    return eigenvectors
