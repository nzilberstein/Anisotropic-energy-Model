from noise import *
from pathlib import Path 
import json
from data import *
from trackers import *
from main import *

from networks.conditioning import *

import matplotlib.pyplot as plt
import numpy as np

def load_args(name, step="last", log=True, dataloaders=False):
    """ Load an experiment with a given name. step can be an integer, "best", or "last" (default). """
    exp_dir = Path("models") / name

    with open(exp_dir / "args.json") as f:
        args_dict = json.load(f)

    return args_dict


def load_exp(name, step="last", log=True, dataloaders=False):
    """ Load an experiment with a given name. step can be an integer, "best", or "last" (default). """
    exp_dir = Path("models") / name

    with open(exp_dir / "args.json") as f:
        args_dict = json.load(f)

    ctx = TrainingContext(**args_dict, step=step, key_remap=None, seed=None, dataloaders=dataloaders, writer=False)
    if log:
        print(f"{name}: retrieved model at step {ctx.step}") # and test loss {ctx.test_perf.loss:.2e}")

    # Disable DataParallel (needed for Hessian computation)
    ctx.model.network = ctx.model.network.module

    # Put in eval mode and disable gradients with respect to all parameters.
    ctx.model.eval()
    for p in ctx.model.parameters():
        p.requires_grad = False

    # Normalize energies.
    # ctx.network.network.log_normalization_constant = ctx.test_perf.log_normalization_constant

    return ctx



def compute_and_plot_single_denoising(ctxs, test_dataloader, noise_level_sampler, noisy_sampler, time_tracker, test_batch_size=512, var = 1):
    for batch in noisy_loader(test_dataloader, noise_level_sampler, noisy_sampler, time_tracker, batch_size=1):

        input = ModelInput(noisy=batch.noisy, noise_level=batch.noise_level, covariance=batch.noise_covariance)
        break

    num_figs = len(ctxs)
    # Save output
    fig, axs = plt.subplots(1, num_figs, figsize=(12, 4))
    for idx_fig in range(num_figs):
        key = list(ctxs.keys())[idx_fig]
        output = ctxs[key].model.forward(input, create_graph=True)
        axs[idx_fig].imshow(np.clip(output.denoised[0].view(3, 32, 32).permute(1, 2, 0).detach().cpu().numpy(), 0, 1))
        e = (output.denoised[0] - batch.clean[0])
        print(var, key, torch.mean(e ** 2))
        axs[idx_fig].set_title(f'Denoised ({key})')
        axs[idx_fig].axis('off')

    plt.tight_layout()
    plt.savefig(f'plots/denoising_comparison{var}.png')


def compute_denoising_error(ctx, test_dataloader, noise_level_sampler, noisy_sampler, time_tracker, test_batch_size=32):
    print(len(test_dataloader.dataset))
    denoising_error = []
    ctx.model.eval()
    with torch.no_grad():
        for i, batch in enumerate(noisy_loader(test_dataloader, noise_level_sampler, noisy_sampler, time_tracker, batch_size=test_batch_size), start=1):
            # print(i)
            input = ModelInput(noisy=batch.noisy, noise_level=batch.noise_level, covariance=batch.noise_covariance)
            # break
            output = ctx.model.forward(input, create_graph=True)
            # e = apply_power_to_list_covariances(batch.noise_covariance, output.denoised - expand(batch.clean), p=-0.5) 
            e = output.denoised - batch.clean
            mse = torch.mean(e ** 2, dim=(-1, -2, -3))  # (B, [1 + L]) 
            denoising_error.append(mse)
            if i == 7:  # Cannot be zero (this function is not called otherwise)
                break

        denoising_error = torch.cat(denoising_error, dim=0)
        return denoising_error


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    # torch.set_printoptions(precision=10, sci_mode=False)

    args = load_args("multigpu/combined_cov/denoiser_song_anisoEmb_groupNorm_lr1.5e-4_lrdecay15000_1000warmup_d", step = "last")
    step = "last"
    ctxs = {
        # "denoiser-anisotropic-newloss-tinbox": load_exp("test_mult_anisotropic_newloss_tinbox", step = "last"),
        # "denoiser-anisotropic-oldloss": load_exp("test_mult_anisotropic_oldloss", step = "last"),
        "denoiser-anisotropic-newloss-tinbox": load_exp("multigpu/combined_cov/denoiser_song_anisoEmb_groupNorm_lr1.5e-4_lrdecay15000_1000warmup_d", step = step),
        "energy-anisotropic-newloss": load_exp("multigpu/combined_cov/energy_song_anisoEmb_groupNorm_lambda1_lr1.5e-4_lrdecay15000_1000warmup_d", step = step),
        # "denoiser-anisotropic-newloss-tinbox": load_exp("test_denoiser_mult_anisotropic_newloss_newTweedie_tinbox_song_groupNorm", step = "last"),
        # "energy-anisotropic-newloss": load_exp("test_energy_mult_anisotropic_newloss_newTweedie_tinbox_song_groupNorm", step = "last"),
    }

    default_ctx = ctxs["denoiser-anisotropic-newloss-tinbox"]
    device = default_ctx.device
    dataset_info = default_ctx.dataset_info
    d = dataset_info.dimension


    # Load data
    test_batch_size = 32
    CHW = 3 * 32 * 32
    H = 32

    train_dataloader, test_dataloader, dataset_info = load_data(
        dataset=args["dataset"], spatial_size=args["spatial_size"], grayscale=args["grayscale"], horizontal_flip=args["horizontal_flip"], data_subset=eval(args["data_subset"]),
        train_batch_size=test_batch_size, test_batch_size=test_batch_size, num_workers=args["num_workers"], seed=2
    )

    images = next(iter(test_dataloader))  # load for testing things.

    shape = (images[0].shape[0], CHW)

    time_tracker: TimeTracker = TimeTracker()
    time_tracker.switch("initialization")

    min_noise_level: NoiseLevel = NoiseLevel.from_unit(dataset_info=dataset_info, **args["min_noise_level"]) # This calls denoising error
    max_noise_level: NoiseLevel = NoiseLevel.from_unit(dataset_info=dataset_info, **args["max_noise_level"])
    noise_level_sampler: NoiseLevelSampler = eval(args["noise_level_sampler"])(min=min_noise_level, max=max_noise_level)

    covariance = spatial_corr_covariance(spatial_size=shape[-2], box_size=12, var_box=1, device=device, var_clean=1e-3)
    noisy_sampler = MultipleColoredGaussianSamplerWithInput(noise_covariance=covariance)
    # noisy_sampler = MultipleColoredGaussianSampler()

    scale_noise_level = 1
    num_variances = 17
    psnr_min = -30
    psnr_max = 90
    psnrs = torch.linspace(psnr_min, psnr_max, num_variances , device=default_ctx.device)  # (L,), steps of 7.5dB
    noise_levels = DenoisingError(dataset_info=default_ctx.dataset_info, psnr=psnrs).to_noise_level().variance  # (L,)
    denoising_error_all_scales = torch.zeros(num_variances)
    energy_denoising_error_all_scales = torch.zeros(num_variances)

    for idx in range(num_variances):
        covariance = spatial_corr_covariance(spatial_size=H, box_size=12, var_box=noise_levels[idx] * scale_noise_level, device=device, var_clean = 1e-3 * scale_noise_level)
        noisy_sampler = MultipleColoredGaussianSamplerWithInput(noise_covariance=covariance)

        denoising_error = compute_denoising_error(ctxs['denoiser-anisotropic-newloss-tinbox'], test_dataloader, noise_level_sampler, noisy_sampler, time_tracker, test_batch_size=test_batch_size)
        print("Denoising model",noise_levels[idx], denoising_error.shape, denoising_error.mean())
        denoising_error_all_scales[idx] = denoising_error.mean()

        noisy_sampler_energy = MultipleColoredGaussianSamplerWithInput(noise_covariance=covariance, batch_size = test_batch_size)
        energy_denoising_error = compute_denoising_error(ctxs['energy-anisotropic-newloss'], test_dataloader, noise_level_sampler, noisy_sampler_energy, time_tracker, test_batch_size=test_batch_size)
        print("Energy model", noise_levels[idx], energy_denoising_error.shape, energy_denoising_error.mean())
        energy_denoising_error_all_scales[idx] = energy_denoising_error.mean()

        compute_and_plot_single_denoising(ctxs, test_dataloader, noise_level_sampler, noisy_sampler, time_tracker, var = noise_levels[idx])

# 

    # Save in the torch array in a file
    torch.save({
        'psnrs': psnrs,
        'denoising_error_all_scales': denoising_error_all_scales,
        'energy_denoising_error_all_scales': energy_denoising_error_all_scales,
    }, f'plots/denoising_error_anisotropic_multigpu_newloss_tinbox_{step}.pt') 
    compute_and_plot_single_denoising(ctxs, test_dataloader, noise_level_sampler, noisy_sampler, time_tracker)
