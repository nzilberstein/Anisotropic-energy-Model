""" Main file for training a diffusion model. """

import argparse
from collections import defaultdict
import json
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import *

from torch.nn.utils import clip_grad_norm_

import numpy as np
import torch
from torch.nn import ReLU, GELU  # for eval in parse_args
from einops import *

from data import *
from grad import *
from networks import *
from noise import *
from plotting import *
from printing import *
from tensor_ops import *
from trackers import *
from utils import *


class TrainingContext:
    """ Named-tuple-like object for holding things to pass around during training. Also ensures proper clean-up in case of error. """

    def __init__(self, *additional_args, step=None, key_remap=None, seed=0, dataloaders=True, writer=True, **additional_kwargs) -> None:
        """ Create a training context from command-line (and potential additional) arguments.
        If step is provided, load a previous checkpoint from the experiment folder. Can be an integer, "best", or "last".
        key_remap is an optional key remapping function for the model state_dict, which is passed to load_checkpoint.
        seed is the random seed for data loading (ordering can be fixed by passing seed=None).
        No side-effects beyond creating experiments folder if does not exist.
        """
        self.time_tracker: TimeTracker = TimeTracker()
        self.time_tracker.switch("initialization")

        args = parse_args(*additional_args, **additional_kwargs)
        self.args: argparse.Namespace = args

        self.dir: Path = Path(args.dir) / args.name
        if not self.dir.exists():
            self.dir.mkdir(parents=True)

        if writer:
            self.logger: logging.Logger = get_logger(log_file=self.dir / "log.txt")
            # Only load writer on request to speed up imports?
            from torch.utils import tensorboard
            self.writer: tensorboard.SummaryWriter = tensorboard.SummaryWriter(log_dir=self.dir)
        self.noise_unit = "psnr"  # TODO: convert to effective_variance once comparisons have been made.
        self.error_unit = "psnr"

        self.device: torch.device =  torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        train_dataloader, test_dataloader, dataset_info = load_data(
            dataset=args.dataset, spatial_size=args.spatial_size, grayscale=args.grayscale, horizontal_flip=args.horizontal_flip, data_subset=eval(args.data_subset),
            train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size, num_workers=args.num_workers, seed=seed, load_only_info=not dataloaders,
        )
        self.train_dataloader: DataLoader = train_dataloader
        self.test_dataloader: DataLoader = test_dataloader
        self.dataset_info: DatasetInfo = dataset_info
        if args.dataset == "Celeba":
            self.dataset_info.spatial_size = 64
        
        self.min_noise_level: NoiseLevel = NoiseLevel.from_unit(dataset_info=self.dataset_info, **args.min_noise_level)
        self.max_noise_level: NoiseLevel = NoiseLevel.from_unit(dataset_info=self.dataset_info, **args.max_noise_level)
        self.noise_level_sampler: NoiseLevelSampler = eval(args.noise_level_sampler)(min=self.min_noise_level, max=self.max_noise_level)
        if args.noise_covariance is not None:
            # self.noise_covariance: Covariance = eval(args.noise_covariance)
            # self.noise_covariance: Covariance = spatial_corr_covariance(dataset_info=self.dataset_info, box_size=5, device=self.device)            # self.noisy_sampler: NoisySampler = ColoredGaussianSampler(noise_covariance=self.noise_covariance)
            # self.noisy_sampler: NoisySampler = UnionPinkWhiteGaussianSampler(noise_covariance=self.noise_covariance)
            self.noisy_sampler: NoisySampler = MultipleColoredGaussianSampler(frequency_component=args.frequency_component, spatial_component=args.spatial_component)
        else:
            self.noise_covariance: Covariance = IdentityCovariance()
            self.noisy_sampler: NoisySampler = eval(args.noise_sampler)

        network_kwargs = eval(args.network_kwargs)
        if args.embed_noise_level_in_range:
            network_kwargs.update(t_min=self.min_noise_level.variance, t_max=self.max_noise_level.variance)
        # network = eval(args.network)(dataset_info=self.dataset_info, **network_kwargs)
        if args.dataset == "ImageNet64" or args.dataset == "Celeba":
            if args.size_network == "small":
                network = SongUNet(img_resolution=64, in_channels=3, out_channels=3,
                                model_channels=128,  # 64 or 128,  # Base multiplier for the number of channels.
                                channel_mult=[2,2,2],    # Per-resolution multipliers for the number of channels.
                                num_blocks=3
                                )
            elif args.size_network == "large":
                network = SongUNet(img_resolution=64, in_channels=3, out_channels=3, channel_mult=[1,2,2,2])
                # network = DhariwalUNet(img_resolution=64, in_channels=3, out_channels=3)
        elif args.dataset == "MNIST":
            if args.size_network == "small":
                network = SongUNet(img_resolution=28, in_channels=1, out_channels=1,
                                model_channels=64,  # 64 or 128,  # Base multiplier for the number of channels.
                                channel_mult=[2,2,2],    # Per-resolution multipliers for the number of channels.
                                num_blocks=3,
                                )
            elif args.size_network == "large":
                # network = SongUNet(img_resolution=32, in_channels=3, out_channels=3, adaptive_scale = args.adaptive_scale, channel_mult=[2,2,2],embedding_type = 'positional', encoder_type = 'residual', dropout = 0.13)
                network = SongUNet(img_resolution=28, in_channels=1, out_channels=1)
        else:
            if args.size_network == "small":
                network = SongUNet(img_resolution=32, in_channels=3, out_channels=3,
                                model_channels=64,  # 64 or 128,  # Base multiplier for the number of channels.
                                channel_mult=[2,2,2],    # Per-resolution multipliers for the number of channels.
                                num_blocks=3,
                                adaptive_scale = args.adaptive_scale
                                )
            elif args.size_network == "large":
                # network = SongUNet(img_resolution=32, in_channels=3, out_channels=3, adaptive_scale = args.adaptive_scale, channel_mult=[2,2,2],embedding_type = 'positional', encoder_type = 'residual', dropout = 0.13)
                network = SongUNet(img_resolution=32, in_channels=3, out_channels=3, adaptive_scale = args.adaptive_scale)
        network = Reparameterization(network, dataset_info=self.dataset_info, **eval(args.reparam_kwargs))  # Note: Reparameterization could be updated to deal with non-identity noise covariances.
        self.model: Model = eval(args.model)(network=network, dataset_info=self.dataset_info, **eval(args.model_kwargs))  # Puts on right device and uses DataParallel.
        self.network: nn.Module = self.model.network.module  # Provides access to underlying network without DataParallel (but on gpu).

        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        self.step: int = 0
        self.best_test_loss: float = float("inf")

        if step is not None:
            train_perf, test_perf = load_checkpoint(self, step=step, key_remap=key_remap)
            # Save train and test performance information just in case
            self.train_perf: PerformanceInfo = train_perf
            self.test_perf: PerformanceInfo = test_perf

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            crash_msg = f"Exception occurred: {exc_type}: {exc_val}"
            self.logger.error(crash_msg, exc_info=True, stack_info=True)
        logging.shutdown()

    def new_dataloader(self, train: bool, batch_size: int, num_workers: Optional[int] = None, seed: Optional[int] = None, num_samples: Optional[int] = None, num_epochs: Optional[int] = 1, data_subset: Optional[slice | Iterable[int]] = None) -> DataLoader:
        """ Returns a new dataloader which can be infinite or not (default: 1 epoch) and shuffled or not (default: unshuffled). """
        dataset = self.train_dataloader.dataset if train else self.test_dataloader.dataset
        dataset = take_subset(dataset, data_subset)
        return get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers or self.args.num_workers, seed=seed, num_samples=num_samples, num_epochs=num_epochs)


def parse_args(*additional_args, **additional_kwargs) -> argparse.Namespace:
    # NOTE: we cannot have eval-type arguments because they are not JSON-serializable.

    parser = argparse.ArgumentParser(description="Train a diffusion model.")

    # Parsers
    def noise_level(arg: str):
        unit, value = arg.split("=")
        return dict(x=float(value), unit=unit)

    # Experiment arguments
    parser.add_argument("--name", default="test", help="name of the experiment")
    parser.add_argument("--dir", default="models", help="directory to save the training runs")

    # Dataset arguments
    parser.add_argument("--dataset", default="ImageNet64", help="name of dataset or path to image folder")
    parser.add_argument("--data-subset", default="None", help="expression (typically a slice) for taking a subset of training data")
    parser.add_argument("--grayscale", default=True, action=argparse.BooleanOptionalAction, help="use grayscale images")
    parser.add_argument("--horizontal-flip", default=True, action=argparse.BooleanOptionalAction, help="use horizontal flip data augmentation")
    parser.add_argument("--spatial-size", default=None, type=int, help="size of patches to extract")

    # Noise arguments
    parser.add_argument("--noise-sampler", default="WhiteGaussianSampler()", help="noise sampler, as a Python expression--if noise-covariance is specified, overriden")
    parser.add_argument("--noise-covariance", default=None, help="noise covariance matrix, as a Python expression")
    parser.add_argument("--noise-level-sampler", default="UniformStddev", help="noise level sampler for training")
    parser.add_argument("--min-noise-level", default=dict(x=30, unit="psnr"), type=noise_level, help="minimum noise level as unit=value")
    parser.add_argument("--max-noise-level", default=dict(x=0, unit="psnr"), type=noise_level, help="maximum noise level as unit=value")
    parser.add_argument("--embed-noise-level-in-range", default=False, action=argparse.BooleanOptionalAction, help="pass noise range to noise level embedding parameters")
    parser.add_argument("--frequency-component",default=True, action=argparse.BooleanOptionalAction, help="frequency component for noise covariance, as a Python expression")
    parser.add_argument("--spatial-component", default=False, action=argparse.BooleanOptionalAction, help="spatial component for noise covariance, as a Python expression")

    # Network arguments
    parser.add_argument("--model", choices=["DenoiserModel", "EnergyModel"], help="model interface")
    parser.add_argument("--model-kwargs", default="dict()", help="keyword arguments for model interface, as a Python expression")
    parser.add_argument("--network", choices=["DnCNN", "UNet", "UNet_measurement", "GradResNet", "Song"], help="network architecture")
    parser.add_argument("--size-network", choices=["small", "large"], help="size of the network")
    parser.add_argument("--network-kwargs", default="dict()", help="keyword arguments for network architecture, as a Python expression")
    parser.add_argument("--reparam-kwargs", default="dict()", help="keyword arguments for reparameterization, as a Python expression")

    # Objective arguments
    parser.add_argument("--mse-var-exponent", type=float, default=0, help="scale the MSE by the value of the noise variance to this exponent")
    parser.add_argument("--noise-score-var-exponent", type=float, default=0, help="scale the noise score objective by the value of the noise variance to this exponent")
    parser.add_argument("--train-noise-score", type=float, default=None, help="scalar multiplier for the noise score loss")

    # Training arguments
    parser.add_argument("--train-batch-size", type=int, default=512, help="batch size for training")
    parser.add_argument("--num-workers", type=int, default=0, help="number of workers for data loading")
    parser.add_argument("--num-training-steps", type=int, default=100_000, help="number of steps to train on")
    parser.add_argument("--lr", type=float, default=0.1, help="initial learning rate")
    parser.add_argument("--lr-decay-every", type=int, default=10_000, help="halve learning rate every N steps")
    parser.add_argument("--warmup-steps", type=int, default=10000, help="number of steps for linear learning rate warmup")

    # Testing arguments
    parser.add_argument("--test-batch-size", type=int, default=64, help="batch size for evaluation")
    parser.add_argument("--num-testing-steps", type=int, default=100, help="number of steps to test on")
    parser.add_argument("--test-every", type=int, default=50_00, help="test every N steps")
    parser.add_argument("--log-gradients", default=False, action=argparse.BooleanOptionalAction, help="log gradients statistics during evaluation")

    # Ignored arguments
    parser.add_argument("-f", "--f")  # This is used by Jupyter notebook with custom kernels.

    # Parse arguments from command line and optional additional ones.
    parser.set_defaults(**additional_kwargs)
    args = parser.parse_args(*additional_args)
    return args


def train_network(ctx: TrainingContext) -> None:
    """ Training-evaluate loop."""
    for batch in noisy_loader(ctx.train_dataloader, ctx.noise_level_sampler, ctx.noisy_sampler, ctx.time_tracker):  # Infinite loop.
        
        # Evaluate the model and save checkpoint.
        if ctx.step % ctx.args.test_every == 0 and ctx.args.num_testing_steps > 0:
            ctx.model.eval()
            # evaluate_and_save_checkpoint(ctx)
            save_checkpoint(ctx, train_perf_info=None, test_perf_info=None, is_best_on_test=False)  # TODO: compute and pass performance info.
            ctx.model.train()

        # Stop if we are done.
        if ctx.step == ctx.args.num_training_steps:
            break

        # Adjust learning rate.
        # if ctx.step % ctx.args.lr_decay_every == 0:
        #     lr = ctx.args.lr / (2 ** (ctx.step // ctx.args.lr_decay_every))  # Recalculate learning rate to work nicely with resumes.
        #     for param_group in ctx.optimizer.param_groups:
        #         param_group["lr"] = lr

        if ctx.step < ctx.args.warmup_steps:
            # Linear warm-up
            lr = ctx.args.lr  * ctx.step / ctx.args.warmup_steps
            for param_group in ctx.optimizer.param_groups:
                param_group["lr"] = lr
        elif (ctx.step - ctx.args.warmup_steps) % ctx.args.lr_decay_every == 0:
            # Exponential decay after warm-up
            decay_factor = 2 ** ((ctx.step - ctx.args.warmup_steps) // ctx.args.lr_decay_every)
            lr = ctx.args.lr / decay_factor
            for param_group in ctx.optimizer.param_groups:
                param_group["lr"] = lr

        # Training step.
        training_step(ctx, batch)
        ctx.step += 1


def compute_metrics(ctx, batch, output, train_test = "train"):
    """ Returns per-sample metrics and loss. Batch axis should be (B, [1 + L]). """
    t = batch.noise_level.variance  # (B, [1 + L])
    d = ctx.dataset_info.dimension

    metrics = {}
    loss = 0
    expand = lambda x: x[:, None] if t.ndim == 2 else x  # (B, [1 + L], ...) to (B, 1, ...)
    if train_test == "train":
        # e = ctx.noise_covariance.apply_sqrt(output.denoised - expand(batch.clean))
        # e = ctx.noise_covariance.apply_inv_sqrt(output.denoised - expand(batch.clean))
        if len(t.shape) == 2:
            e = output.denoised - expand(batch.clean)
            mse = torch.mean(e ** 2, dim=(-1, -2, -3))  # (B, ...) 
            metrics["denoising_mse"] = mse
            loss = loss + mse * t ** ctx.args.mse_var_exponent  # (B, [1 + L])
        else:
            e = apply_power_to_list_covariances(batch.noise_covariance, output.denoised - expand(batch.clean), p=-0.5)
            # e = output.denoised - expand(batch.clean)
            mse = torch.mean(e ** 2, dim=(-1, -2, -3))  # (B, [1 + L]) 
            metrics["denoising_mse"] = mse
            metrics["iso_denoising_mse"] = torch.mean((output.denoised - expand(batch.clean)) ** 2, dim=(-1, -2, -3))
            loss = loss + mse #When considering random matrices, we don't have to multiply by t because it is embedding in the covariance
            # e = apply_power_to_list_covariances(batch.noise_covariance, expand(batch.noisy) - expand(batch.clean), p=-1)
            # e = apply_power_to_list_covariances(batch.noise_covariance, output.data_score - e, p=0.5)
            # # e = output.denoised - expand(batch.clean)
            # mse = torch.mean(e ** 2, dim=(-1, -2, -3))  # (B, [1 + L]) 
            # metrics["denoising_mse"] = mse
            # loss = loss + mse 
            # ----------------------------- #
    elif train_test == "test":
        e = output.denoised - expand(batch.clean)
        mse = torch.mean(e ** 2, dim=(-1, -2, -3))
        metrics["denoising_mse"] = mse
        loss = loss + mse * t ** ctx.args.mse_var_exponent  # (B, [1 + L])
    if output.noise_score is not None:
        if train_test == "train":
            if len(t.shape) == 1:
                z = apply_power_to_list_covariances(batch.noise_covariance, batch.noisy - expand(batch.clean), p=-1)
                phi_inv = get_inv_tensor_from_list_covariances(batch.noise_covariance).unsqueeze(1)
                # pdb.set_trace()
                if ctx.args.spatial_component == True:
                    phi_inv_minus_z = 0.5 * (phi_inv - z ** 2)
                else:
                    phi_inv_minus_z = 0.5 * (torch.real(torch.fft.ifft2(phi_inv - torch.fft.fft2(z) ** 2)))
                e = apply_power_to_list_covariances(batch.noise_covariance, output.noise_score - phi_inv_minus_z, p=1)
                noise_loss = torch.mean(e ** 2, dim=(-1, -2, -3)) / d  # (B, ...)
                metrics["norm_mse"] = noise_loss  # XXX: multiply by 4 for backwards compatibility
                if ctx.args.train_noise_score is not None:
                    loss = loss + ctx.args.train_noise_score * noise_loss
                # quadratic_form = lambda z: torch.mean(z * apply_power_to_list_covariances(batch.noise_covariance, z, p=-1), dim=(-1, -2, -3))  # (..., C, H, W) to (...,)
                # target = 1 - quadratic_form(batch.noisy - expand(batch.clean)) # (B, [1 + L])
                # # pdb.set_trace()
                # noise_loss = (target / 2 - t * output.noise_score / d) ** 2  # (B, [1 + L])
                # metrics["norm_mse"] = noise_loss * 4  # XXX: multiply by 4 for backwards compatibility
                # if ctx.args.train_noise_score is not None:
                #     loss = loss + ctx.args.train_noise_score * noise_loss * t ** ctx.args.noise_score_var_exponent
    # pdb.set_trace()
    # Also add energy (for cross-entropy/NLL).
    if output.energy is not None:
        metrics["energy"] = output.energy  # (B, [1 + L])

    metrics["loss"] = loss
    return metrics


def training_step(ctx: TrainingContext, batch: Batch) -> None:
    """ Do a single training step. """
    ctx.time_tracker.switch("forward")
    input = ModelInput(noisy=batch.noisy, noise_level=batch.noise_level, covariance=batch.noise_covariance)  # (B, C, H, W)
    output: ModelOutput = ctx.model(input)  # (B, C, H, W)
    losses = compute_metrics(ctx, batch, output)  # metric -> (B, [1 + L])
    loss = losses["loss"].mean()  # (B,) to ()

    if ctx.step % 20 == 0:
        ctx.logger.info(f"Step {ctx.step}: denoising_mse {losses['denoising_mse'].mean():.4f}")
        ctx.logger.info(f"Step {ctx.step}: Isotropic_denoising_mse {losses['iso_denoising_mse'].mean():.4f}")
        if ctx.args.model == "EnergyModel":
            ctx.logger.info(f"Step {ctx.step}: norm_mse {losses['norm_mse'].mean():.4f}")
        ctx.logger.info(f"Step {ctx.step}: loss {loss:.4f}")
        ctx.logger.info(f"Step {ctx.step}: lr {ctx.optimizer.param_groups[0]['lr']:.4f}")

    if ctx.step % 100 == 0:
        image = torch.stack([batch.noisy[0:8], output.denoised[0:8]], dim=0)  # (2, L, C, H, W)
        image = rescale_imgs(image, soft=True)
        image = rearrange(image, "n l c h w -> c (l h) (n w)")
        ctx.writer.add_image(f"denoising/train_masks", image, ctx.step)
        ctx.writer.add_scalar(f"loss_dmse", losses["denoising_mse"].mean(), ctx.step)
        ctx.writer.add_scalar(f"loss_iso_dmse", losses["iso_denoising_mse"].mean(), ctx.step)
        if ctx.args.model == "EnergyModel":
            ctx.writer.add_scalar(f"loss_tmse", losses["norm_mse"].mean(), ctx.step)
        ctx.writer.add_scalar(f"loss", loss, ctx.step)
        

    ctx.time_tracker.switch("backward")
    ctx.optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(ctx.model.parameters(), max_norm=10.0) # max_norm is the clipping threshold
    ctx.optimizer.step()


class PerformanceInfo:
    """ Named-tuple-like object for holding performance information. """

    def __init__(self, loss: float, noise_levels: NoiseLevel, denoising_errors: DenoisingError, norm_errors: Optional[DenoisingError] = None,
                 log_normalization_constant: Optional[float] = None, cross_entropies: Optional[LogTensor] = None) -> None:
        self.loss: float = loss
        self.noise_levels: NoiseLevel = noise_levels  # (L,)
        self.denoising_errors: DenoisingError = denoising_errors  # (L,)
        self.norm_errors: Optional[DenoisingError] = norm_errors  # (L,)
        self.log_normalization_constant: Optional[float] = log_normalization_constant
        self.cross_entropies: Optional[LogTensor] = cross_entropies  # (L,)


def evaluate_and_save_checkpoint(ctx: TrainingContext):
    """ Evaluate the model on both train and test set and log a few things. Returns the validation loss. """
    # Evaluate on both training and test set.
    train_perf_info = evaluate_on_dataloader(ctx, ctx.train_dataloader, train_test="train")
    test_perf_info = evaluate_on_dataloader(ctx, ctx.test_dataloader, train_test="test")

    # Higher-resolution PSNR plots.
    ctx.time_tracker.switch("plotting")
    ctx.writer.add_figure("denoising_curves", plot_performance_curves(curves=[
            (train_perf_info.noise_levels, train_perf_info.denoising_errors, dict(linestyle="--", color="tab:blue")),
            (test_perf_info.noise_levels, test_perf_info.denoising_errors, dict(linestyle="-", color="tab:red")),
        ], x=ctx.error_unit, y=ctx.noise_unit), ctx.step)
    if train_perf_info.norm_errors is not None:
        ctx.writer.add_figure("norm_curves", plot_performance_curves(curves=[
                (train_perf_info.noise_levels, train_perf_info.norm_errors, dict(linestyle="--", color="tab:blue")),
                (test_perf_info.noise_levels, test_perf_info.norm_errors, dict(linestyle="-", color="tab:red")),
            ], x=ctx.error_unit, y=ctx.noise_unit, plot_baselines=False), ctx.step)

    # Update best test loss and save checkpoint.
    ctx.time_tracker.switch("checkpointing")
    is_best_on_test = test_perf_info.loss < ctx.best_test_loss
    if is_best_on_test:
        ctx.best_test_loss = test_perf_info.loss
    # save_checkpoint(ctx, train_perf_info, test_perf_info, is_best_on_test)

    # Do a bit of logging.
    ctx.time_tracker.switch("logging")
    ctx.logger.info(f"Step {ctx.step}: train loss {train_perf_info.loss:.4f}, test loss {test_perf_info.loss:.4f}")
    ctx.logger.info(f"Time breakdown so far:\n{ctx.time_tracker.pretty_print()}")
    sys.stdout.flush()

    # Exit training if train loss is NaN (done only now to still do the plotting and checkpointing just in case).
    if np.isnan(train_perf_info.loss):
        raise ValueError("Training loss is NaN!")


def evaluate_on_dataloader(ctx: TrainingContext, loader: DataLoader, train_test: str) -> PerformanceInfo:
    """ Evaluate the model on the given loader. Writes results to tensorboard and logger. Returns a summary of the performance for checkpointing. """
    if ctx.noise_unit == "psnr":
        noise_format = ".0f"
        psnr_min = -30
        psnr_max = 90
        psnrs = torch.linspace(psnr_min, psnr_max, 17, device=ctx.device)  # (L,), steps of 7.5dB
        noise_levels = DenoisingError(dataset_info=ctx.dataset_info, psnr=psnrs).to_noise_level()  # (L,)
        idx_subset = np.arange(0, len(noise_levels), 2)  # steps of 15dB
        noise_levels_subset = noise_levels[idx_subset]
    else:
        assert False
    fixed_noise_level_sampler = FixedNoiseLevelSampler(noise_levels=noise_levels)
    noise_level_sampler = UnionNoiseLevelSampler(ctx.noise_level_sampler, fixed_noise_level_sampler)  # Will sample 1 + L noise levels.

    metrics_tracker = defaultdict(BatchMeanTracker)

    log_gradients = ctx.args.log_gradients and train_test == "train"
    if log_gradients:
        gradients_tracker = defaultdict(MeanTracker)  # We directly compute summed gradients for efficiency.
        param_dict = ctx.network.my_named_parameters(reduced=True, with_grad=True)

    with torch.set_grad_enabled(log_gradients):
        for i, batch in enumerate(noisy_loader(loader, noise_level_sampler, ctx.noisy_sampler, ctx.time_tracker), start=1):
            ctx.time_tracker.switch("evaluation")

            # NOTE: we could set up hooks to record statistics of activations.
            input = ModelInput(noisy=batch.noisy, noise_level=batch.noise_level, covariance=batch.noise_covariance)  # (B, C, H, W)
            output: ModelOutput = ctx.model(input, create_graph=log_gradients)  # (B, C, H, W)
            denoised = output.denoised

            # Update tracker metrics.
            metrics = compute_metrics(ctx, batch, output, train_test=train_test)  # metric -> (B, 1 + L)
            for key, metric in metrics.items():
                metrics_tracker[key].update(metric)

            if log_gradients:
                # Update gradient metrics.
                ctx.time_tracker.switch("avg_gradients")
                batch_size = batch.noisy.shape[0]
                loss = metrics["loss"].sum(0)[1:]  # (L,)
                # NOTE: could compute gradients separately for each term in the loss (denoising MSE vs noise score MSE).

                for noise_level_idx in idx_subset:  # Only compute gradients for noise levels in the subset for efficiency.
                    # Compute gradients for this noise level.
                    ctx.optimizer.zero_grad()
                    loss[noise_level_idx].backward(retain_graph=noise_level_idx != idx_subset[-1])  # Retain the graph for future noise levels unless this is the last noise level.

                    # Update trackers.
                    for param_name, param in param_dict.items():
                        gradients_tracker[param_name, noise_level_idx].update(sum=param.grad, count=batch_size)

            if i == ctx.args.num_testing_steps:  # Cannot be zero (this function is not called otherwise)
                break
    
    loss = metrics_tracker["loss"].mean()[0].item()  # ()
    denoising_mse = metrics_tracker["denoising_mse"].mean()  # (1 + L,)
    denoising_errors = DenoisingError(dataset_info=ctx.dataset_info, mse=denoising_mse[1:])  # (1,)
    performance_info = PerformanceInfo(loss=loss, noise_levels=noise_levels, denoising_errors=denoising_errors)

    # Plot metrics corresponding to average noise level as scalars: loss, denoising MSE, and norm MSE.
    ctx.time_tracker.switch("plotting")
    ctx.writer.add_scalar(f"loss/{train_test}", loss, ctx.step)
    for i in idx_subset:
        ctx.writer.add_scalar(f"denoising_{ctx.error_unit}_{ctx.noise_unit}{noise_levels[i].to_unit(ctx.noise_unit).item():{noise_format}}/{train_test}", denoising_errors[i].psnr, ctx.step)
    if "norm_mse" in metrics_tracker:
        norm_mse = metrics_tracker["norm_mse"].mean()  # (1 + L,)
        norm_errors = DenoisingError(dataset_info=ctx.dataset_info, mse=norm_mse[1:])  # (L,)
        performance_info.norm_errors = norm_errors
        for i in idx_subset:
            ctx.writer.add_scalar(f"norm_{ctx.error_unit}_{ctx.noise_unit}{noise_levels[i].to_unit(ctx.noise_unit).item():{noise_format}}/{train_test}", norm_errors[i].psnr, ctx.step)
    if "energy" in metrics_tracker:
        average_energy = metrics_tracker["energy"].mean()[1:]  # (L,), large noise first
        # Compute log normalization constant (from large noise) and normalize energies.
        log_normalization_constant = ctx.dataset_info.dimension/2 * torch.log(2 * np.pi * np.e * noise_levels[0].variance) - average_energy[0]
        cross_entropies = LogTensor(average_energy + log_normalization_constant, d=ctx.dataset_info.dimension)
        performance_info.log_normalization_constant = log_normalization_constant.item()
        performance_info.cross_entropies = cross_entropies  # (L,)
        # Log cross entropy in bits per dimension (+ log2(256) = 8 to convert to discrete)
        for i in idx_subset:
            ctx.writer.add_scalar(f"crossent_bpd_{ctx.noise_unit}{noise_levels[i].to_unit(ctx.noise_unit).item():{noise_format}}/{train_test}", cross_entropies.bpd[i] + 8, ctx.step)

    # Plot average gradients.
    if log_gradients:
        key = lambda param_name, idx_subset: f"avg_gradient/{param_name}_{ctx.noise_unit}{noise_levels[idx_subset].to_unit(ctx.noise_unit).item():{noise_format}}"

        for param_name in param_dict:
            # Compute norms and cosines.
            gradient_means = torch.stack([gradients_tracker[param_name, noise_level_idx].mean().flatten() for noise_level_idx in idx_subset], dim=0)  # (L_small, N)
            norms = torch.linalg.norm(gradient_means, dim=1)  # (L,)
            cosines = (gradient_means @ gradient_means.T) / (norms[:, None] * norms[None, :])  # (L, L)

            # Plot each norm as a scalar.
            for i, noise_level_idx in enumerate(idx_subset):
                ctx.writer.add_scalar(key(param_name, noise_level_idx), norms[i].item(), ctx.step)
            # Publish custom scalar layout (one panel for parameter with all noise levels on it) on first step.
            if ctx.step == 0:
                layout = {}
                layout[param_name] = ["Multiline", [key(param_name, noise_level_idx) for noise_level_idx in idx_subset]]
                ctx.writer.add_custom_scalars({f"avg_gradient/{param_name}": layout})

            # Plot norms as a function of noise level as a figure.
            plt.figure()
            plt.plot(psnrs[idx_subset].cpu().numpy(), (norms / norms.max()).cpu().numpy(), marker=".")  # Set maximum norm to 1 (absolute value meaningless for Adam).
            plt.yscale("log")
            plt.xlabel("Input PSNR")
            plt.ylabel("Norm of average gradient")
            ctx.writer.add_figure(f"avg_gradient_norm/{param_name}", plt.gcf(), ctx.step)

            # Plot cosines as an image.
            cosines_01 = (cosines + 1) / 2  # Map from [-1, 1] to [0, 1].
            cosines_img = plt.get_cmap("bwr")(cosines_01.cpu().numpy())[..., :3]  # (L, L, 3), remove alpha channel.
            ctx.writer.add_image(f"avg_gradient_cosine/{param_name}", cosines_img, ctx.step, dataformats="HWC")

    # Select one sample of clean, noisy, denoised images as (num_noise_levels, C, H, W) tensors.
    index = idx[0, 1 + idx_subset]
    image = torch.stack([batch.noisy[index], denoised[index]], dim=0)  # (2, L, C, H, W)
    image = rescale_imgs(image, soft=True)
    image = rearrange(image, "n l c h w -> c (l h) (n w)")
    ctx.writer.add_image(f"denoising/{train_test}", image, ctx.step)

    # Evaluate Jacobian of denoiser (disabled for now).
    #     ctx.time_tracker.switch("jacobian")
    #     if isinstance(ctx.model, DenoiserModel):
    #         # Disable DataParallel for Jacobian.
    #         jacobian = compute_jacobian(ctx.network, batch.noisy[index], batch.noise_level.variance[index], full_batch=False, symmetrize=True)  # (L, CHW, CHW)
    #     elif isinstance(ctx.model, EnergyModel):
    #         # Disable DataParallel for Hessian.
    #         model_dp = ctx.model.network
    #         ctx.model.network = model_dp.module
    #         hessian = rearrange(ctx.model(input[index], compute_hessian=True).data_hessian, "l c1 h1 w1 c2 h2 w2 -> l (c1 h1 w1) (c2 h2 w2)")  # (L, CHW, CHW)
    #         jacobian = torch.eye(hessian.shape[-1], device=hessian.device) - input[index].noise_level.variance[..., None, None] * hessian  # (L, CHW, CHW)
    #         jacobian = DecomposedMatrix(jacobian, decomposition="eigh")  # (L, CHW, CHW)
    #         ctx.model.network = model_dp
    #     # Plot eigenvalues of Jacobian and Hessian
    #     ctx.writer.add_figure(f"jacobian_eigenvalues/{train_test}", plot_jacobian_eigenvalues(
    #         jacobian.eigenvalues, noise_levels=noise_levels_subset, unit=ctx.noise_unit,
    #         ), ctx.step)
    #     ctx.writer.add_figure(f"hessian_eigenvalues/{train_test}", plot_jacobian_eigenvalues(
    #         (1 - jacobian.eigenvalues) / input[index].noise_level.variance[..., None], noise_levels=noise_levels_subset, unit=ctx.noise_unit, yscale="linear",
    #         ), ctx.step)
    #     # Plot eigenvectors.
    #     ctx.writer.add_image(f"jacobian_eigenvectors/{train_test}", arrange_jacobian_eigenvectors(
    #         jacobian, noise_levels=noise_levels_subset,
    #     ), ctx.step)

    # NOTE: we could plot statistics of the weights and optimizer state (running gradient mean and variances).
    # Disabled for now (obsolete, would need to update merge with the above).
    # ctx.time_tracker.switch("magnitudes")
    # plot_magnitudes(ctx, batch)

    return performance_info


def plot_magnitudes(ctx: TrainingContext, batch: Batch):
    """ Plot magnitudes of optimization-related quantities: weights, activations, and gradients. """
    # We compute norms of input/output activations, weights, and gradients.
    # Norms are computed over all axes but the last one (batch/neurons), and we plot their mean.
    # NOTE: we could plot many more things: raw gradients (esp. at initialization), noise-level-dependent activations and gradients, higher-order moments/histograms, etc.

    # Disable gradient magnitudes for non-DnCNN for now.
    if not isinstance(ctx.network, DnCNN):
        return

    model: DnCNN = ctx.network  # No DataParallel here (interacts badly with hooks).
    tensors: Dict[str, Dict[str, torch.Tensor]] = dict(
        input={},
        output={},
        weight={},
        grad_mean={},
        grad_root_mean_square={},
        grad_step={},
    ) # quantity -> parameter name -> values (B, C, H, W) / (N, [C, K, K])


    # Collect module and parameters to monitor.

    # Modules for which we monitor norms of input and output activations.
    monitored_modules: Dict[str, nn.Module] = {}
    # Parameters/buffers for which we monitor norms, and optimizer state if applicable.
    monitored_parameters: Dict[str, torch.Tensor] = {}
    # We define the layer index as starting from 1 and increasing after each convolutional layer.
    # Plot things every 5 layers: typically 1, 6, 11, 16, 21.
    monitored_layers = 1 + np.arange(0, 1 + model.num_hidden_layers, 5)

    layer = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            layer += 1

        if layer in monitored_layers:
            if isinstance(module, nn.Conv2d):
                monitored_modules[f"conv{layer}"] = module
                monitored_parameters[f"conv{layer}"] = module.weight
            elif isinstance(module, Normalization):
                monitored_modules[f"norm{layer}"] = module
                monitored_parameters[f"norm{layer}"] = module.running_scale
                monitored_parameters[f"gain{layer}"] = module.weight


    # Collect activations on given batch.

    def get_hook(name: str):
        def hook(self, inputs, output):  # inputs is a tuple, we assume it is of length 1
            tensors["input"][name] = inputs[0]
            tensors["output"][name] = output
            return output
        return hook

    handles = []
    for name, module in monitored_modules.items():
        handles.append(module.register_forward_hook(get_hook(name)))
    _ = model(batch.noisy[:, 0])  # Only look at activations for one noise level.

    # Remove hooks: handles do not seem to work?
    for module in monitored_modules.values():
        module._forward_hooks.clear()


    # Collect weights and optimizer state.

    assert isinstance(ctx.optimizer, torch.optim.Adam), "optimizer state is class-dependent (for SGD, should plot momentum)"
    optimizer_state = ctx.optimizer.state_dict()
    assert len(optimizer_state["param_groups"]) == 1, "logic for param-dependent lr not implemented"
    lr = optimizer_state["param_groups"][0]["lr"]
    eps = optimizer_state["param_groups"][0]["eps"]

    def get_param_id(parameter: torch.Tensor | nn.Parameter) -> Optional[int]:
        # Returns the optimizer id of the given parameter, or None.
        # Unfortunately there is no better way to get that than iterating through the parameter list used to build the optimizer.
        for i, param in enumerate(ctx.optimizer.param_groups[0]["params"]):
            if param is parameter:
                return i
        return None

    for name, parameter in monitored_parameters.items():
        tensors["weight"][name] = parameter  # (N, ...)

        if ctx.step > 0 and ((param_id := get_param_id(parameter)) is not None):
            grad_mean = optimizer_state["state"][param_id]["exp_avg"]  # (N, ...)
            tensors["grad_mean"][name] = grad_mean

            grad_root_mean_square = torch.sqrt(optimizer_state["state"][param_id]["exp_avg_sq"])  # (N, ...)
            tensors["grad_root_mean_square"][name] = grad_root_mean_square

            grad_step = lr * grad_mean / (torch.sqrt(grad_root_mean_square) + eps)  # (N, ...)
            tensors["grad_step"][name] = grad_step


    # Compute aggregate statistics and send to tensorboard.

    key = lambda quantity_name, layer_name: f"magnitudes/{quantity_name}_msqn/{layer_name}"

    for quantity_name, values in tensors.items():
        # Plot root means of squares, and forget about other things (higher-order moments, histograms) for now.
        # Reduce using mean for activations but sum for weights and gradients.
        reduction = "mean" if quantity_name in ["input", "output"] else "sum"
        square_norms = {layer_name: reduce(tensor ** 2, "n ... -> n", reduction) for layer_name, tensor in values.items()}
        root_mean_square_norms = {layer_name: tensor.mean().sqrt() for layer_name, tensor in square_norms.items()}
        # ctx.writer.add_scalars(f"magnitudes/{quantity_name}_msqn", mean_square_norms, ctx.step)
        for layer_name, root_mean_square_norm in root_mean_square_norms.items():
            ctx.writer.add_scalar(key(quantity_name, layer_name), root_mean_square_norm, ctx.step)

    if ctx.step == ctx.args.test_every:  # Publish layout after first train epoch.
        layout = {}
        for quantity_name, values in tensors.items():
            layout[quantity_name] = ["Multiline", [key(quantity_name, layer_name) for layer_name in values.keys()]]
        ctx.writer.add_custom_scalars({"magnitudes_agg": layout})


def checkpoint_filename(ctx: TrainingContext, step="last"):
    """ Small utility function to create checkpoint filenames. `step' can be an integer, "best", or "last" (default). """
    if step == "last":
        suffix = ""
    elif step == "best":
        suffix = "best"
    elif isinstance(step, int):
        suffix = f"step{step}"
    else:
        raise ValueError(f"Unknown step: {step}")

    filename = ctx.dir / f"{str_concat(['model', suffix], sep='_')}.pth.tar"
    return filename


def save_checkpoint(ctx: TrainingContext, train_perf_info: PerformanceInfo, test_perf_info: PerformanceInfo, is_best_on_test: bool):
    """ Update the "last" checkpoint, and save a copy at appropriate steps or if it is the best so far. """
    state = dict(
        step=ctx.step,
        args=ctx.args,
        state_dict=ctx.network.state_dict(),  # Save without DataParallel.
        # Don't save optimize state (momentum) to save space at the cost of exact reproducibility.
        # optimizer=ctx.optimizer.state_dict(),
        train_perf=train_perf_info,
        test_perf=test_perf_info,
    )
    torch.save(state, checkpoint_filename(ctx))

    if should_save_checkpoint(ctx.step, ctx.args.test_every):
        shutil.copyfile(checkpoint_filename(ctx), checkpoint_filename(ctx, ctx.step))
    if is_best_on_test:
        shutil.copyfile(checkpoint_filename(ctx), checkpoint_filename(ctx, "best"))


def load_checkpoint(ctx: TrainingContext, step="last", key_remap=None) -> Tuple[PerformanceInfo, PerformanceInfo]:
    """ Loads a given checkpoint (updates model parameters, current step, and best test loss).
    key_remap is an optional function old_key -> new_key to remap keys in the model state_dict.
    Returns train and test performance info. """
    state = torch.load(checkpoint_filename(ctx, step), weights_only=False)

    if key_remap is not None:
        state["state_dict"] = {key_remap(k): v for k, v in state["state_dict"].items()}

    ctx.step = state["step"]
    # ctx.args = state["args"]  # Do not load arguments, because new arguments might have been added since the last run.
    ctx.network.load_state_dict(state["state_dict"])  # Load without DataParallel.
    # We didn't save optimizer state to save space at the cost of exact reproducibility.
    # ctx.optimizer.load_state_dict(state["optimizer"])

    train_perf: PerformanceInfo = state["train_perf"]
    test_perf: PerformanceInfo = state["test_perf"]
    # ctx.best_test_loss = train_perf.loss
    return train_perf, test_perf


def steps_to_save(test_every: int) -> Generator[int, None, None]:
    """ Infinite generator for the steps to save on: test_every times 1|2|5 times a power of 10 (or 0). """
    yield 0
    step = test_every
    while True:
        yield step
        yield 2 * step
        yield 5 * step
        step *= 10


def should_save_checkpoint(step: int, test_every: int) -> bool:
    """ Returns True if the checkpoint should be saved at this step. """
    for to_save in steps_to_save(test_every):
        if step == to_save:
            return True
        elif step < to_save:
            return False
    assert False



def get_logger(log_file: Optional[str]) -> logging.Logger:
    """ Create a logger for the experiment, which logs to stdout and an optional log file. """

    logging.captureWarnings(True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Do not propagate to root logger, otherwise we see double output on the console.

    formatter = logging.Formatter("[{asctime}] {levelname:8} {message}\t\t(File \"{filename}\", line {lineno}, in {funcName})",
                                  datefmt="%Y-%m-%d %H:%M:%S", style='{')

    handlers = [logging.StreamHandler(stream=sys.stdout)]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, mode="a"))

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def print_cmd_line_and_save_args(ctx: TrainingContext):
    ctx.logger.info(f"Running on {os.environ.get('SLURM_JOB_NODELIST')} with slurm job id {os.environ.get('SLURM_JOB_ID')}")
    ctx.logger.info(f"Command line: {' '.join(sys.argv)}")
    ctx.logger.info(f"Experiment running in {ctx.dir}")
    ctx.logger.info(f"Arguments: {vars(ctx.args)}")

    with (ctx.dir / "args.json").open("w") as f:
        json.dump(ctx.args.__dict__, f, indent=2)

    ctx.writer.add_text("cmd", " ".join(sys.argv), 0)


def print_model_and_params(ctx: TrainingContext):
    s = []

    s.append("Model:")
    s.append(str(ctx.model))
    ctx.writer.add_text("model", str(ctx.model), 0)

    s.append("Model parameters:")
    num_parameters = 0
    for name, param in ctx.model.named_parameters():
        if not param.requires_grad:
            continue
        num = param.numel()
        num_parameters += num
        s.append(f"- {name}: {param.shape} => {num:_} parameters")
    s.append(f"Total: {num_parameters:_} parameters")

    # Also log basic statistics of the training data.
    s.append(f"Dataset:")
    s.append(f"- Training set: {len(ctx.train_dataloader.dataset):_} samples")
    s.append(f"- Test set: {len(ctx.test_dataloader.dataset):_} samples")
    s.append(f"- Data info: shape {(ctx.dataset_info.num_channels, ctx.dataset_info.spatial_size, ctx.dataset_info.spatial_size)} mean {ctx.dataset_info.mean:.2f} stddev {ctx.dataset_info.stddev:.2f}")

    ctx.logger.info("\n".join(s))

def main():
    ctx = TrainingContext()
    with ctx as _:
        print_cmd_line_and_save_args(ctx)
        print_model_and_params(ctx)
        sys.stdout.flush()

        assert torch.cuda.is_available()

        train_network(ctx)


if __name__ == "__main__":
    main()
