from sympy import var
from noise import *
from pathlib import Path 
import json
from data import *
from trackers import *
from main import *
import random
import torchvision

from networks.conditioning import *

import lpips


seed = 1000
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

    return args_dict


def load_exp(name, step="last", log=True, dataloaders=False):
    """ Load an experiment with a given name. step can be an integer, "best", or "last" (default). """
    exp_dir = Path("models") / name

    with open(exp_dir / "args.json") as f:
        args_dict = json.load(f)

    args_dict['size_network'] = "large"
    args_dict['frequency_component'] = False
    args_dict['spatial_component'] = True
    
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

def get_sigmas(sigma_begin, sigma_end, num_classes, device, sigma_dist):
    if sigma_dist == 'geometric':
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end),
                               num_classes))).float().to(device)
    elif sigma_dist == 'uniform':
        sigmas = torch.tensor(
            np.linspace(sigma_begin, sigma_end, num_classes)
        ).float().to(device)

    else:
        raise NotImplementedError('sigma distribution not supported')

    return sigmas

def anneal_Langevin_dynamics(x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008, batch_size = 1,
                             final_only=False, verbose=False, denoise=True, factor_sigma_2=1, data_score = False,
                             box_size=10, H=32, inp_mask_type='box'):
    images = []

    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            step_size = step_lr * (1/sigmas[-1]) ** 2
            
            # adjacent_sigma = sigmas[c + 1] if c + 1 < len(sigmas) else sigma
            # step_size = (sigma**2 - adjacent_sigma**2)
            
            # if sigma/factor_sigma_2 > sigmas[-1]:
            #     covariance = [spatial_corr_covariance(spatial_size=H, box_size=10, var_box=sigma**2, device=device, var_clean = (sigma/factor_sigma_2)**2)] * batch_size
            #     # covariance = [spatial_corr_covariance(spatial_size=H, box_size=13, var_box=(sigma/factor_sigma_2)**2, device=device, var_clean = sigma**2)] * batch_size
            # else:
            #     covariance = [spatial_corr_covariance(spatial_size=H, box_size=10, var_box=sigma**2, device=device, var_clean = (sigmas[-1])**2)] * batch_size
            #     # covariance = [spatial_corr_covariance(spatial_size=H, box_size=13, var_box=(sigmas[-1])**2, device=device, var_clean = (sigma)**2)] * batch_size
            covariance = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=sigma**2, device=device, var_clean = 1e-7, inp_mask_type=inp_mask_type)] * batch_size
            # covariance = [spatial_corr_covariance(spatial_size=H, box_size=15, var_box=(sigma/factor_sigma_2)**2, device=device, var_clean = (sigma)**2)] * batch_size
            noisy_sampler = MultipleColoredGaussianSamplerWithInput(noise_covariance=covariance, batch_size = batch_size)
            noise_level = NoiseLevel(variance=sigma**2)
            
            for s in range(n_steps_each):
                input = ModelInput(noisy=x_mod, noise_level=noise_level, covariance=covariance)
                model_output = scorenet.forward(input, create_graph = False)
                if data_score:
                    grad = -model_output.data_score
                else:
                    denoised = model_output.denoised
                    grad = covariance[0].apply_power(denoised - x_mod, p = -1)
    
                noise = torch.randn_like(x_mod)
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                x_mod = x_mod + step_size * covariance[0].apply_power(grad, p=1) + torch.sqrt(step_size) * covariance[0].apply_power(noise, p=0.5)

                image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
                snr = torch.sqrt(step_size / 2.) * grad_norm / noise_norm
                grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

                if not final_only:
                    images.append(x_mod.to('cpu'))
                if verbose:
                    print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                        c, step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))
                
            if c % 50 == 0:
                samples = torch.clamp(x_mod[0], 0.0, 1.0)
                # samples = inverse_data_transform(config, x_mod[0])
                plt.figure(figsize=(2, 2))
                plt.imshow(samples.cpu().numpy().transpose(1, 2, 0))
                plt.axis('off')
                plt.savefig(f'samples/sample_step{c}_{ctx.args.model}.pdf')

                # Compute mse between samples and clean images
                # mse = torch.mean((samples - batch.clean[0]) ** 2)
                # print(f"Step {c}, MSE: {mse.item():.4f}", f"MSE[dB]: {-10 * torch.log10(mse)}")

        if final_only:
            return [x_mod.to('cpu')]
        else:
            return images

def PC_sampler(x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008, batch_size = 1,
                 final_only=False, verbose=False, denoise=True, data_score = False,
                 inp_mask_type="half", var_clean = 1e-7, box_size=12, H=32):
    images = []
    with torch.no_grad():
        for c, sigma in enumerate(sigmas[:-2]):

            # Predictor step
            sigma_curr = sigmas[c]
            sigma_next = sigmas[c+1]

            # Compute score with sigma_{i+1}
            covariance = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=sigma_curr**2, device=device, var_clean = var_clean, inp_mask_type=inp_mask_type)] * batch_size
            noisy_sampler = MultipleColoredGaussianSamplerWithInput(noise_covariance=covariance, batch_size = batch_size)
            noise_level = NoiseLevel(variance=sigma_curr**2)
            input = ModelInput(noisy=x_mod, noise_level=noise_level, covariance=covariance)
            model_output = scorenet.forward(input, create_graph = False)
            grad = -model_output.data_score


            # Update x by using sigma_{i+1} - sigma_{i}
            covariance_step = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=(sigma_curr**2 - sigma_next**2), device=device, var_clean = var_clean, inp_mask_type=inp_mask_type)] * batch_size
            x_mod = x_mod + covariance_step[0].apply_power(grad, p=1) 
            noise = torch.randn_like(x_mod)
            x_mod = x_mod + covariance_step[0].apply_power(noise, p=0.5)

            # Corrector steps
            for s in range(n_steps_each):
                covariance = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=sigma_next**2, device=device, var_clean = var_clean, inp_mask_type=inp_mask_type)] * batch_size
                noisy_sampler = MultipleColoredGaussianSamplerWithInput(noise_covariance=covariance, batch_size = batch_size)
                noise_level = NoiseLevel(variance=(sigma_next**2))                
                input = ModelInput(noisy=x_mod, noise_level=noise_level, covariance=covariance)
                model_output = scorenet.forward(input, create_graph = False)
                grad = -model_output.data_score

    
                noise = torch.randn_like(x_mod)
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                r = 0.17
                step_size = 2 * (r * noise_norm / grad_norm) ** 2
                covariance_step = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=(sigma_curr - sigma_next), device=device, var_clean = var_clean, inp_mask_type=inp_mask_type)] * batch_size
                x_mod = x_mod + covariance_step[0].apply_power(grad, p=1) + covariance_step[0].apply_power(noise, p=0.5)

                image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
                grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

                if not final_only:
                    images.append(x_mod.to('cpu'))
                if verbose:
                    print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                        c, step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))
                
            if c % 50 == 0:
                samples = torch.clamp(x_mod[0], 0.0, 1.0)
                # samples = inverse_data_transform(config, x_mod[0])
                plt.figure(figsize=(2, 2))
                plt.imshow(samples.cpu().numpy().transpose(1, 2, 0))
                plt.axis('off')
                plt.show()

                # Compute mse between samples and clean images
                # mse = torch.mean((samples - batch.clean[0]) ** 2)
                # print(f"Step {c}, MSE: {mse.item():.4f}", f"MSE[dB]: {-10 * torch.log10(mse)}")

        if final_only:
            return [x_mod.to('cpu')]
        else:
            return images


if __name__ == "__main__":

    ## Load models

    args = load_args("multigpu/combined_cov_tworandvars/energy_song_anisoEmb_groupNorm_lambda1_mult_lr2e-4_lrdecay50000_1000warmup_d2", step = "best")
    # args = load_args("multigpu/combined_cov_tworandvars/energy_songWavelet_anisoEmb_groupNorm_lambda1_lr1.5e-4_lrdecay15000_1000warmup_d2", step = "last")
    step = "last"
    ctxs = {
        # "energy-non-reg": load_exp("multigpu/combined_cov_tworandvars/energy_song_anisoEmb_groupNorm_lambda0_lr1.5e-4_lrdecay15000_1000warmup_d2", step = step),
        # "Energy-Dual": load_exp("multigpu/combined_cov_tworandvars/energy_song_anisoEmb_groupNorm_lambda0_lr1.5e-4_lrdecay15000_1000warmup_d2", step = step),
        # "Denoiser": load_exp("multigpu/combined_cov_tworandvars/denoiser_song_anisoEmb_groupNorm_lr1.5e-4_lrdecay15000_1000warmup", step = step),
        # "Energy-Dual": load_exp("multigpu/combined_cov_tworandvars/energy_songSmall_anisoEmb_groupNorm_lambda1_mult_lr1.5e-4_lrdecay50000_1000warmup_d2", step = step),
        # "Denoiser": load_exp("multigpu/combined_cov_tworandvars/denoiser_song_scoreSmall_anisoEmb_groupNorm_mult_lr1.5e-4_lrdecay50000_1000warmup", step = step),
        "Energy-Dual": load_exp("multigpu/combined_cov_tworandvars/energy_song_anisoEmb_groupNorm_lambda1_mult_lr2e-4_lrdecay50000_1000warmup_d2", step = step),
        "Denoiser": load_exp("multigpu/combined_cov_tworandvars/denoiser_song_score_anisoEmb_groupNorm_mult_lr1.5e-4_lrdecay50000_1000warmup", step = step),
        # "energy-reg-d": load_exp("multigpu/combined_cov_tworandvars/energy_song_anisoEmb_groupNorm_lambda1_lr1.5e-4_lrdecay15000_1000warmup_d2", step = step),
        # "denoiser-anisotropic-newloss-tinbox": load_exp("test_denoiser_mult_anisotropic_newloss_newTweedie_tinbox_song_groupNorm", step = "last"),
        # "energy-anisotropic-newloss": load_exp("test_energy_mult_anisotropic_newloss_newTweedie_tinbox_song_groupNorm", step = "last"),
    }
    loss_fn_alex = lpips.LPIPS(net='alex').cuda() # best forward scores

    time_tracker: TimeTracker = TimeTracker()
    time_tracker.switch("initialization")


    default_ctx = ctxs["Energy-Dual"]
    device = default_ctx.device
    dataset_info = default_ctx.dataset_info
    d = dataset_info.dimension

    ## Load data 

    test_batch_size = 10
    img_size = 32
    CHW = 3 * img_size * img_size
    
    train_dataloader, test_dataloader, dataset_info = load_data(
        dataset=args["dataset"], spatial_size=args["spatial_size"], grayscale=args["grayscale"], data_subset=eval(args["data_subset"]),
        train_batch_size=test_batch_size, test_batch_size=test_batch_size, num_workers=args["num_workers"], seed=2
    )

    images = next(iter(test_dataloader))  # load for testing things.
    shape = (images[0].shape[0], CHW)
 
    ## Sampling and noise parameters
    sigma_begin = 1 * 30
    num_classes = 1000
    sigma_dist = 'geometric'
    sigma_end = 1e-2
    n_steps_each = 2
    step_lr = 0.8e-6 #0.0000008
    batch_size = test_batch_size
    factor_sigma_2 = 1
    
    box_size = 12
    sigma_box_init = sigma_begin**2
    sigma_clean_init = 1e-7 #(sigma_begin/factor_sigma_2)**2 #1e-3
    inp_mask_type='half'
    sampler = "anneal_lang"

    # sigma_box_init = (sigma_begin/factor_sigma_2)**2
    # sigma_clean_init = sigma_begin**2 #1e-3

    if sampler == "PC_sampler":
        timesteps = torch.arange(num_classes, device=device) / (num_classes - 1)
        sigmas = torch.tensor([sigma_begin**2 * (sigma_end**2 / sigma_begin**2) ** t for t in timesteps])
    else:
        sigmas = get_sigmas(sigma_begin, sigma_end, num_classes, device, sigma_dist)

    # img_set_total = 1
    # mse = 0
    # lpips = 0
    # for img_set in range(img_set_total):
        
    cov = spatial_corr_covariance_testing(spatial_size=img_size, box_size=box_size, var_box=sigma_box_init, device=device, var_clean = sigma_clean_init, inp_mask_type=inp_mask_type)

    idx_img = 0 #torch.randint(0, images[0].shape[0], (1,)).item()
    clean_images = images[0][idx_img:idx_img+batch_size,:,:,:].cuda()

    # x_init = images[0][idx_img:idx_img+batch_size,:,:,:].cuda() + cov.apply_power(torch.randn_like(images[0][0:batch_size,:,:,:]).cuda(), p = 0.5)
    # # x_init = cov.apply_power(torch.randn_like(images[0][idx_img:idx_img+batch_size,:,:,:]).cuda(), p = 0.5)
    # x_init = x_init[0:batch_size]

    x_lists = {f'samples_{ctxs["Energy-Dual"].args.model}': None, 
            f'samples_{ctxs["Denoiser"].args.model}': None
        }

    ## Run sampler
    print("Running sampling...")
    for model, ctx in ctxs.items():
        mse = 0
        lpips = 0
        for data_score_flag in [True]:
            seed = 100
            # Reset seed and regenerate the same x_init
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            x_init = clean_images + cov.apply_power(torch.randn_like(images[0][0:batch_size,:,:,:]).cuda(), p=0.5)

            print(f"Running {ctx.args.model} with data score {data_score_flag}...")

            if sampler == "PC_sampler":
                x = PC_sampler(x_init, ctx.model, sigmas, n_steps_each=n_steps_each, step_lr=step_lr, batch_size=batch_size, 
                                             verbose = False, final_only=True, denoise=False, data_score = data_score_flag,
                                             inp_mask_type=inp_mask_type, var_clean=sigma_clean_init, box_size=box_size, H=img_size)
            else:
                x = anneal_Langevin_dynamics(x_init, ctx.model, sigmas, n_steps_each=n_steps_each, step_lr=step_lr, batch_size=batch_size, 
                                             verbose = False, final_only=True, denoise=False, factor_sigma_2 = factor_sigma_2, 
                                             data_score = data_score_flag,
                                             box_size=box_size, H=img_size, inp_mask_type=inp_mask_type)

            
            
            samples = torch.clamp(x[0], 0.0, 1.0)
            mse = mse + torch.mean((images[0][idx_img:idx_img+batch_size,:,:,:] - samples[0:batch_size])**2, dim=(-1, -2, -3)).sum() 
            for ii in range(batch_size):
                lpips += loss_fn_alex(images[0][ii:ii+1,:,:,:].cuda(),  samples[ii:ii+1].cuda())
            
            print("MSE", mse / (batch_size))
            print("LPIPS", lpips / (batch_size))
            
            print("Saving generated samples...")
            plt.figure(figsize=(2,2))
            plt.imshow(samples[0].cpu().numpy().transpose(1, 2, 0))
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'samples/generated_sample_{ctx.args.model}.pdf')


            # Save in dictonary
            x_lists[f'samples_{ctx.args.model}'] = x[0]
 
 
        # images = next(iter(test_dataloader)) 
            
    # print(mmse_total.mean())
    # print(lpips_total.mean())
    
    grid = torchvision.utils.make_grid(x_init.cpu(), nrow=8, padding=2)
    grid_np = grid.permute(1, 2, 0).numpy()

    # Display the grid
    plt.figure(figsize=(10, 10)) # Adjust figure size as needed
    plt.imshow(grid_np)
    plt.axis('off') # Hide axes
    plt.title("Noisy images")
    plt.tight_layout()
    plt.savefig(f'samples/noisy_images_batch_{seed}.pdf')

    samples = torch.clamp(x_lists[f'samples_{ctxs["Energy-Dual"].args.model}'], 0.0, 1.0)
    grid = torchvision.utils.make_grid(samples, nrow=8, padding=2, normalize=False)
    grid_np = grid.permute(1, 2, 0).numpy()

    # Display the grid
    plt.figure(figsize=(10, 10)) # Adjust figure size as needed
    plt.imshow(grid_np)
    plt.axis('off') # Hide axes
    plt.title("Generated images - Energy")
    plt.tight_layout()
    plt.savefig(f'samples/generated_sample_batch_energy_{seed}.pdf')

    samples = torch.clamp(x_lists[f'samples_{ctxs["Denoiser"].args.model}'], 0.0, 1.0)
    grid = torchvision.utils.make_grid(samples, nrow=8, padding=2, normalize=False)
    grid_np = grid.permute(1, 2, 0).numpy()

    # Display the grid
    plt.figure(figsize=(10, 10)) # Adjust figure size as needed
    plt.imshow(grid_np)
    plt.axis('off') # Hide axes
    plt.title("Generated images - Denoiser")
    plt.tight_layout()
    plt.savefig(f'samples/generated_sample_batch_denoiser_{seed}.pdf')