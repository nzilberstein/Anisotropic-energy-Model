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
        print(f"{name}: retrieved model at step {ctx.step} and test loss {ctx.test_perf.loss:.2e}")
        print(f"Residual: {ctx.network.residual}")

    # Disable DataParallel (needed for Hessian computation)
    ctx.model.network = ctx.model.network.module

    # Put in eval mode and disable gradients with respect to all parameters.
    ctx.model.eval()
    for p in ctx.model.parameters():
        p.requires_grad = False

    # Normalize energies.
    ctx.network.network.log_normalization_constant = ctx.test_perf.log_normalization_constant

    return ctx


def compute_denoising_error(ctx, test_dataloader, noise_level_sampler, noisy_sampler, time_tracker, test_batch_size=512):
    print(len(test_dataloader.dataset))
    denoising_error = []
    for i, batch in enumerate(noisy_loader(test_dataloader, noise_level_sampler, noisy_sampler, time_tracker, batch_size=test_batch_size), start=1):
        print(i)
        input = ModelInput(noisy=batch.noisy, noise_level=batch.noise_level, covariance=batch.noise_covariance)
        # break

        output_isotropic_oldloss = ctx.model.forward(input, create_graph=True)
        diff = output_isotropic_oldloss.denoised.view(test_batch_size, -1) - batch.clean.view(test_batch_size, -1)
        error = torch.linalg.vector_norm(diff, ord=2, dim=1)
        denoising_error.append(error)

        if i == 7:  # Cannot be zero (this function is not called otherwise)
            break

    denoising_error = torch.cat(denoising_error, dim=0)
    return denoising_error


def compute_and_plot_single_denoising(ctxs, test_dataloader, noise_level_sampler, noisy_sampler, time_tracker, test_batch_size=512):
    for batch in noisy_loader(train_dataloader, noise_level_sampler, noisy_sampler, time_tracker):

        input = ModelInput(noisy=batch.noisy, noise_level=batch.noise_level, covariance=batch.noise_covariance)
        break

    output_isotropic_oldloss = ctxs['denoiser-isotropic-newloss-tinbox'].model.forward(input, create_graph=True)

    # Save output
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(np.clip(input.noisy[0].view(3, 32, 32).permute(1, 2, 0).detach().cpu().numpy(), 0, 1))
    axs[0].set_title('Noisy')
    axs[0].axis('off')
    axs[1].imshow(np.clip(output_isotropic_oldloss.denoised[0].view(3, 32, 32).permute(1, 2, 0).detach().cpu().numpy(), 0, 1))
    denoising_error = (output_isotropic_oldloss.denoised[0].view(-1) - batch.clean[0].view(-1)).norm().item()
    print(denoising_error)
    axs[1].set_title('Denoised (Isotropic Old Loss)')
    axs[1].axis('off')
    plt.tight_layout()
    plt.savefig('denoising_comparison_iso.png')


if __name__ == "__main__":
    args = load_args("test_mult_isotropic_newloss_tinbox", step = "best")

    ctxs = {
        "denoiser-isotropic-newloss-tinbox": load_exp("test_mult_isotropic_newloss_tinbox", step = "last"),
    }

    default_ctx = ctxs["denoiser-isotropic-newloss-tinbox"]
    device = default_ctx.device
    dataset_info = default_ctx.dataset_info
    d = dataset_info.dimension


    # Load data
    batch_test = 10
    CHW = 3 * 32 * 32
    H = 32

    train_dataloader, test_dataloader, dataset_info = load_data(
        dataset=args["dataset"], spatial_size=args["spatial_size"], grayscale=args["grayscale"], horizontal_flip=args["horizontal_flip"], data_subset=eval(args["data_subset"]),
        train_batch_size=batch_test, test_batch_size=args["test_batch_size"], num_workers=args["num_workers"], seed=2
    )

    images = next(iter(test_dataloader))  # load for testing things.

    shape = (images[0].shape[0], CHW)

    time_tracker: TimeTracker = TimeTracker()
    time_tracker.switch("initialization")

    min_noise_level: NoiseLevel = NoiseLevel.from_unit(dataset_info=dataset_info, **args["min_noise_level"]) # This calls denoising error
    max_noise_level: NoiseLevel = NoiseLevel.from_unit(dataset_info=dataset_info, **args["max_noise_level"])
    noise_level_sampler: NoiseLevelSampler = eval(args["noise_level_sampler"])(min=min_noise_level, max=max_noise_level)

    scale_noise_level = 1
    num_variances = 17
    psnr_min = -30
    psnr_max = 90
    psnrs = torch.linspace(psnr_min, psnr_max, num_variances , device=default_ctx.device)  # (L,), steps of 7.5dB
    noise_levels = DenoisingError(dataset_info=default_ctx.dataset_info, psnr=psnrs).to_noise_level().variance  # (L,)
    denoising_error_all_scales = torch.zeros(num_variances)
    print(noise_levels)
    for idx in range(num_variances ):
        covariance = spatial_corr_covariance(spatial_size=H, box_size=12, var_box=noise_levels[idx] * scale_noise_level, device=device, var_clean = 1e-3 * scale_noise_level)
        noisy_sampler = MultipleColoredGaussianSamplerWithInput(noise_covariance=covariance)

        denoising_error = compute_denoising_error(ctxs['denoiser-isotropic-newloss-tinbox'], test_dataloader, noise_level_sampler, noisy_sampler, time_tracker)
        print(noise_levels[idx], denoising_error.shape, denoising_error.mean())

        denoising_error_all_scales[idx] = denoising_error.mean()


    # Save in the torch array in a file
    torch.save({
        'psnrs': psnrs,
        'denoising_error_all_scales': denoising_error_all_scales,
    }, 'plots/denoising_error_isotropic_newloss_tinbox.pt')