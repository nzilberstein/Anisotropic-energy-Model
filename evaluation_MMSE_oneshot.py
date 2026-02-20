from noise import *
from pathlib import Path 
import json
from data import *
from trackers import *
from main import *
import random

from networks.conditioning import *

import matplotlib.pyplot as plt
import numpy as np

import lpips

seed = 20
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_args(name, step="last", log=True, dataloaders=False):
    """ Load an experiment with a given name. step can be an integer, "best", or "last" (default). """
    exp_dir = Path("models") / name

    with open(exp_dir / "args.json") as f:
        args_dict = json.load(f)

    args_dict['num_workers'] = 8
    return args_dict


def load_exp(name, step="last", log=True, dataloaders=False):
    """ Load an experiment with a given name. step can be an integer, "best", or "last" (default). """
    exp_dir = Path("models") / name

    with open(exp_dir / "args.json") as f:
        args_dict = json.load(f)

    args_dict['size_network'] = "large"
    
    ctx = TrainingContext(**args_dict, step=step, key_remap=None, seed=None, dataloaders=dataloaders, writer=False)
    if log:
        print(f"{name}: retrieved model at step {ctx.step}")

    # Disable DataParallel (needed for Hessian computation)
    # ctx.model.network = ctx.model.network.module

    # Put in eval mode and disable gradients with respect to all parameters.
    ctx.model.eval()
    for p in ctx.model.parameters():
        p.requires_grad = False

    # Normalize energies.
    # ctx.network.network.log_normalization_constant = ctx.test_perf.log_normalization_constant

    return ctx



def compute_and_plot_single_denoising(ctxs, test_dataloader, noise_level_sampler, noisy_sampler, time_tracker, test_batch_size=1, var = 1):
    for batch in noisy_loader(test_dataloader, noise_level_sampler, noisy_sampler, time_tracker, batch_size=1):

        input = ModelInput(noisy=batch.noisy, noise_level=batch.noise_level, covariance=batch.noise_covariance)
        break

    num_figs = len(ctxs)
    # Save output
    fig, axs = plt.subplots(1, num_figs+2, figsize=(12, 4))
    axs[0].imshow(np.clip(batch.clean[0].view(3, 32, 32).permute(1, 2, 0).detach().cpu().numpy(), 0, 1))
    axs[0].set_title(f'Clean image')
    axs[0].axis('off')
    axs[1].imshow(np.clip(batch.noisy[0].view(3, 32, 32).permute(1, 2, 0).detach().cpu().numpy(), 0, 1))
    axs[1].set_title(f'Noisy image $\sigma_t = 1')
    axs[1].axis('off')
    for idx_fig in range(num_figs):
        key = list(ctxs.keys())[idx_fig]
        output = ctxs[key].model.forward(input, create_graph=True)
        axs[idx_fig+2].imshow(np.clip(output.denoised[0].view(3, 32, 32).permute(1, 2, 0).detach().cpu().numpy(), 0, 1))
        e = (output.denoised[0] - batch.clean[0])
        print(var, key, torch.mean(e ** 2))
        axs[idx_fig+2].set_title(f'{key}')
        axs[idx_fig+2].axis('off')

    plt.tight_layout()
    plt.savefig(f'plots/denoising_comparison{var}.pdf')


def compute_denoising_error(ctx, test_dataloader, noise_level_sampler, noisy_sampler, time_tracker, test_batch_size=32, lpips=None):
    # print(len(test_dataloader.dataset))
    denoising_error = []
    ctx.model.eval()
    loss_lpips = 0
    with torch.no_grad():
        for i, batch in enumerate(noisy_loader(test_dataloader, noise_level_sampler, noisy_sampler, time_tracker, batch_size=test_batch_size), start=1):
            # print(i)
            input = ModelInput(noisy=batch.noisy, noise_level=batch.noise_level, covariance=batch.noise_covariance)
            # break
            output = ctx.model.forward(input, create_graph = False)
            # e = apply_power_to_list_covariances(batch.noise_covariance, output.denoised - batch.clean, p=-0.5) 
            e = output.denoised - batch.clean
            mse = torch.mean(e ** 2, dim=(-1, -2, -3))  # (B, [1 + L]) 
            denoising_error.append(mse)
            if lpips != None:
                loss_lpips = 0
                for idx in range(batch.clean.shape[0]):
                    loss_lpips = loss_lpips + lpips(output.denoised[idx:idx+1,:,:,:], batch.clean[idx:idx+1,:,:,:])
            if i == 10:  # Cannot be zero (this function is not called otherwise)
                break
        
        denoising_error = torch.cat(denoising_error, dim=0)
        print(f"Evaluated on {denoising_error.shape[0]} images")
        if lpips != None:
            print(loss_lpips/denoising_error.shape[0])
        return denoising_error


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    # torch.set_printoptions(precision=10, sci_mode=False)
    # args = load_args("multigpu/freq_cov/denoiser_song_score_anisoEmb_groupNorm_mult_lr1.5e-4_lrdecay50000_1000warmup", step = "last")
    args = load_args("multigpu/all_together/energy_song_dual_truncatedFreq_celeba", step = "last")
    step = "last"
    ctxs = {
        "Energy-Single": load_exp("multigpu/all_together/energy_song_dual_truncatedFreq_celeba", step = 250000),
        "Energy-Dual": load_exp("multigpu/all_together/energy_song_single_truncatedFreq_celeba", step = step),
    }

    default_ctx = ctxs["Energy-Dual"]
    device = default_ctx.device
    dataset_info = default_ctx.dataset_info
    d = dataset_info.dimension


    # Load data
    test_batch_size = 32
    H = 64
    CHW = 3 * H * H

    train_dataloader, test_dataloader, dataset_info = load_data(
        dataset=args["dataset"], spatial_size=args["spatial_size"], grayscale=args["grayscale"], horizontal_flip=False, data_subset=eval(args["data_subset"]),
        train_batch_size=test_batch_size, test_batch_size=test_batch_size, num_workers=args["num_workers"], seed=seed
    )
    images = next(iter(test_dataloader))  # load for testing things.

    shape = (images[0].shape[0], CHW)

    time_tracker: TimeTracker = TimeTracker()
    time_tracker.switch("initialization")

    min_noise_level: NoiseLevel = NoiseLevel.from_unit(dataset_info=dataset_info, **args["min_noise_level"]) # This calls denoising error
    max_noise_level: NoiseLevel = NoiseLevel.from_unit(dataset_info=dataset_info, **args["max_noise_level"])
    noise_level_sampler: NoiseLevelSampler = eval(args["noise_level_sampler"])(min=min_noise_level, max=max_noise_level)

    # covariance = spatial_corr_covariance(spatial_size=shape[-2], box_size=15, var_box=1, device=device, var_clean=1e-3)
    # noisy_sampler = MultipleColoredGaussianSamplerWithInput(noise_covariance=covariance)
    # noisy_sampler = MultipleColoredGaussianSampler()

    scale_noise_level = 1
    num_variances = 20
    psnr_min = -30
    psnr_max = 90
    psnrs = torch.linspace(psnr_min, psnr_max, num_variances , device=default_ctx.device)  # (L,), steps of 7.5dB
    noise_levels = DenoisingError(dataset_info=default_ctx.dataset_info, psnr=psnrs).to_noise_level().variance  # (L,)
    #
    # low, high = 1e-9, 1e3
    # log_low, log_high = torch.log10(torch.tensor(low)), torch.log10(torch.tensor(high))
    
    # # Sample uniformly in the log domain
    # log_noise_levels = torch.empty(17).uniform_(log_low, log_high)
    # noise_levels = 10 ** log_noise_levels
    # log_noise_levels_2 = torch.empty(17).uniform_(log_low, log_high)
    # noise_levels_2 = 10 ** log_noise_levels_2
    # print(noise_levels, noise_levels_2)
    #
    energyDual_denoiser_error_all_scales = torch.zeros(num_variances)
    energy_denoiser_error_all_scales = torch.zeros(num_variances)
    denoiser_error_all_scales = torch.zeros(num_variances)

    loss_fn_alex = None #lpips.LPIPS(net='alex').cuda() # best forward scores

    inp_mask_type = 'box'  #'box' 'inpainting_box' 'inpainting_random'
    with torch.no_grad():
        for idx in range(num_variances):
            box_size = 20 #torch.randint(0,H+1,(1,)).item()
            covariance = spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=noise_levels[idx] * scale_noise_level, device=device, var_clean = 1e-9 * scale_noise_level, inp_mask_type=inp_mask_type, half_box_size = box_size, missing_indices_input = None)
            
            # kernel_size = 8
            # kernel_std = 0.8
            # covariance = deblurring_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, kernel_std=kernel_std, device=device, noise_level=noise_levels[idx])
    
            noisy_sampler = MultipleColoredGaussianSamplerWithInput(noise_covariance=covariance, batch_size = test_batch_size)
    
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
            energyDual_denoising_error = compute_denoising_error(ctxs['Energy-Dual'], test_dataloader, noise_level_sampler, noisy_sampler, time_tracker, test_batch_size=test_batch_size, lpips=loss_fn_alex)
            print(f"MSE for energy-dual at noise level {noise_levels[idx]}: {energyDual_denoising_error.mean()}")
            energyDual_denoiser_error_all_scales[idx] = energyDual_denoising_error.mean()
    
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
            energy_denoising_error = compute_denoising_error(ctxs['Energy-Single'], test_dataloader, noise_level_sampler, noisy_sampler, time_tracker, test_batch_size=test_batch_size, lpips=loss_fn_alex)
            print(f"MSE for energy-single at noise level {noise_levels[idx]}: {energy_denoising_error.mean()}") 
            energy_denoiser_error_all_scales[idx] = energy_denoising_error.mean()
    
            # torch.manual_seed(seed)
            # torch.cuda.manual_seed_all(seed)
    
            # denoiser_denoising_error = compute_denoising_error(ctxs['Denoiser'], test_dataloader, noise_level_sampler, noisy_sampler, time_tracker, test_batch_size=test_batch_size, lpips=loss_fn_alex)
            # print(f"MSE for denoiser at noise level {noise_levels[idx]}: {denoiser_denoising_error.mean()}")
            # denoiser_error_all_scales[idx] = denoiser_denoising_error.mean()
    
            # compute_and_plot_single_denoising(ctxs, test_dataloader, noise_level_sampler, noisy_sampler, time_tracker, var = noise_levels[idx])
        # print(energy_reg_denoising_error_all_scales.mean())
        print(denoiser_error_all_scales.mean())
    # # 
    
        # Save in the torch array in a file
        results_file_name = f"denoising_error_energyvsdenoiser_song_{step}.pt"
        torch.save({
            'psnrs': psnrs,
            'energy_denoiser_denoising': energy_denoiser_error_all_scales,
            'energyDual_denoiser_denoising': energyDual_denoiser_error_all_scales,
            'mask_type': inp_mask_type,
        }, f'plots/{results_file_name}') 
        print(f"Saved in plots/{results_file_name}")
        # compute_and_plot_single_denoising(ctxs, test_dataloader, noise_level_sampler, noisy_sampler, time_tracker)
