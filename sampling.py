import random
import json

import lpips
import torchvision
from torchmetrics.image.kid import KernelInceptionDistance

from noise import *
from pathlib import Path 
from data import *
from trackers import *
from main import *
from networks.conditioning import *
from samplers import *





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
    args_dict['adaptive_scale'] = False

    ctx = TrainingContext(**args_dict, step=step, key_remap=None, seed=None, dataloaders=dataloaders, writer=False)
    if log:
        print(f"{name}: retrieved model at step {ctx.step}")

    # Put in eval mode and disable gradients with respect to all parameters.
    ctx.model.eval()
    for p in ctx.model.parameters():
        p.requires_grad = False

    # Normalize energies.
    # ctx.network.network.log_normalization_constant = ctx.test_perf.log_normalization_constant

    return ctx

def get_sigmas(sigma_begin, sigma_end, num_classes, device, sigma_dis):
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


def compute_sigma(y, model, inp_mask_type = 'box', type_cov = 'spatial'):
    num_variances = 1 #30
    psnr_min = 90
    psnr_max = -30
    psnrs = torch.linspace(psnr_min, psnr_max, num_variances , device=default_ctx.device)  # (L,), steps of 7.5dB
    noise_levels = DenoisingError(dataset_info=default_ctx.dataset_info, psnr=psnrs).to_noise_level().variance  # (L,)
    noise_level_2 = 1e-9
    H = y.shape[-1]

    output_reg_d2 =  torch.zeros(num_variances)
    min_energy = np.zeros(H-1)
    estimated_var = np.zeros(H-1)
    
    box_idx = 0
    if type_cov == 'spatial':
        type_cov_tensor = torch.tensor(0)
    elif type_cov == 'freq':
        type_cov_tensor = torch.tensor(1)
    with torch.no_grad():
        for box in range(0, H - 1):
            for idx in range(num_variances):
                covariance = spatial_corr_covariance_testing(spatial_size=H, box_size=box, var_box=0, device=device, var_clean = 1, inp_mask_type=inp_mask_type, half_box_size = box)
                output_reg_d2[idx] = model.network(y, covariance.get_matrix()[None,None,:,:], type_cov_list =type_cov_tensor)
            
            min_energy[box_idx] = np.min(output_reg_d2.cpu().numpy())
            estimated_var[box_idx] = noise_levels[np.argmin(output_reg_d2.cpu().numpy()).item()].cpu().numpy()
            box_idx = box_idx + 1
    
    return np.argmin(min_energy)

if __name__ == "__main__":

    ## Load models
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=False, choices=['CIFAR10', 'Celeba', 'ImageNet64', 'AFHQ'])

    # range(0, 10)
    for seed in [20]:
        # Seed parameterers
        seed = seed#20
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


        args_cmd = parser.parse_args()
        print(args_cmd)
        if args_cmd.dataset == "CIFAR10":
            args = load_args("multigpu/all_together/energy_song_dual_truncatedFreq_CIFAR10_2", step = "best")
            step = "last"
            ctxs = {
                "Energy-Dual": load_exp("multigpu/all_together/energy_song_dual_truncatedFreq_CIFAR10_2", step = step),
            }
        elif args_cmd.dataset == "Celeba":
            args = load_args("multigpu/all_together/energy_song_dual_truncatedFreq_celeba", step = "best")
            step = "last"
            ctxs = {
                "Energy-Dual": load_exp("multigpu/all_together/energy_song_dual_truncatedFreq_celeba", step = step),
            }
        elif args_cmd.dataset == "ImageNet64":
            args = load_args("multigpu/all_together/energy_song_dual_truncatedFreq_imagenet", step = "best")
            step = "last"
            ctxs = {
                "Energy-Dual": load_exp("multigpu/all_together/energy_song_dual_truncatedFreq_imagenet", step = step),
            }
        elif args_cmd.dataset == "AFHQ":
            args = load_args("multigpu/all_together/energy_song_dual_truncatedFreq_afhq_192", step = "best")
            step = "last"
            ctxs = {
                "Energy-Dual": load_exp("multigpu/all_together/energy_song_dual_truncatedFreq_afhq_192", step = step),
            }

        # Load metrics
        loss_fn_alex = lpips.LPIPS(net='alex').cuda() # best forward scores
        kid = KernelInceptionDistance(subset_size=5)

        time_tracker: TimeTracker = TimeTracker()
        time_tracker.switch("initialization")

        # Hyperparametrs
        default_ctx = ctxs["Energy-Dual"]
        device = default_ctx.device
        dataset_info = default_ctx.dataset_info
        d = dataset_info.dimension

        # Load data 
        test_batch_size = 3 # 20
        if args_cmd.dataset == "AFHQ":
            img_size = 192
        else:
            img_size = dataset_info.spatial_size
        CHW = 3 * img_size * img_size
        
        train_dataloader, test_dataloader, dataset_info = load_data(
            dataset=args["dataset"], spatial_size=args["spatial_size"], grayscale=args["grayscale"], horizontal_flip=False, data_subset=eval(args["data_subset"]),
            train_batch_size=test_batch_size, test_batch_size=test_batch_size, num_workers=args["num_workers"], seed=seed #2
        )

    
        # Sampling and noise parameters
        sigma_begin = 10 #best 8
        sigma_end = 1e-3 #1e-2 
        num_classes = 1000 #400
        sigma_dist = 'geometric'
        sigma_measurement = 1e-3
        sigmas = get_sigmas(sigma_begin, sigma_end, num_classes, device, sigma_dist)
        
        n_steps_each = 1
        batch_size = test_batch_size
        snr = 0.15 #0.13 #0.2
        temp = 1.0 #1
        blind = False
        sampler = "PC" #PC_adaptive, PC_mala 

        # Parameters of the degradation
        sigma_box_init = sigma_begin**2
        domain = 'pixel' #'freq' #
        inp_mask_type = "box"
        box_size = 25 #13
        # For domain = 'freq'
        kernel_size = 4
        kernel_std = 0.8

        if inp_mask_type == "random":
            total_pixels = img_size * img_size
            n_missing = int(0.7 * total_pixels)
            missing_indices = torch.randperm(total_pixels, device=device)[:n_missing]
            torch.save(missing_indices, "missing_indices.pt")
            
        else:
            missing_indices = None


        
        # Build covariance and load images
        idx_img = 0
        if domain == "freq":
            # deblurring_id = deblurring_covariance_from_shape(spatial_size=img_size, kernel_size=kernel_size, kernel_std=kernel_std, device=device, noise_level=1)
            # cov = deblurring_covariance_from_shape(spatial_size=img_size, kernel_size=kernel_size, kernel_std=kernel_std, device=device, noise_level=sigma_begin**2)
            cov = sr_covariance_from_shape(spatial_size=img_size, kernel_size=kernel_size, device=device, noise_level=1)    
        else:
            # cov = spatial_corr_covariance_testing(spatial_size=img_size, box_size=box_size, var_box=sigma_box_init, device=device, var_clean = sigma_clean_init, half_box_size = box_size, inp_mask_type=inp_mask_type)
            cov = spatial_corr_covariance_testing(spatial_size=img_size, box_size=box_size, var_box=0, device=device, var_clean = 1, half_box_size = box_size, inp_mask_type=inp_mask_type, missing_indices_input=missing_indices)

        x_lists = {f'samples_{ctxs["Energy-Dual"].args.model}': None, 
            }

        ## Run sampler
        print("Running sampling...")

        mse = 0
        lpips_ = 0

        gamma_cov = str(0.6) + "snr" + str(snr) + "box_25" #"dual_test" # + "posterior_exp" + str(idx_img) #6e-2
        batch_idx_post = seed
        dataset_name = args["dataset"]
        if not Path(f"samples/{dataset_name}_{box_size}_{gamma_cov}_{sampler}_{domain}").exists():
            Path(f"samples/{dataset_name}_{box_size}_{gamma_cov}_{sampler}_{domain}").mkdir(parents=True)
        for batch_idx, images in enumerate(test_dataloader):
            clean_images = images[0][idx_img:idx_img+batch_size,:,:,:].cuda()
            for model_, ctx in ctxs.items():

                mse_batch = 0
                lpips_batch = 0
                print(f"Running {model_} ... with batch idx {batch_idx}")
                
                if domain == "freq":
                    x_deblur = cov.apply_power(clean_images, p = -0.5)
                    x_init = cov.apply_power(x_deblur + 0.01 * torch.randn_like(images[0][0:batch_size,:,:,:]).cuda(), p = 0.5)
                    # x_init = x_init
                else:
                    # x_init = clean_images + cov.apply_power(torch.randn_like(clean_images).cuda(), p=0.5)
                    x_init = cov.apply_power(clean_images, p = 1) + sigma_measurement * torch.randn_like(images[0][0:batch_size,:,:,:]).cuda()

                # Compute the size of the box
                if blind == True:
                    box_size_hat = compute_sigma(x_init, ctx)
                else:
                    box_size_hat = box_size
                print(f"Box size estimated: {box_size_hat}, True box size:{box_size}")

                # Run sampler
                if sampler == "PC":
                    x = PC_sampler(x_init, ctx.model, sigmas, n_steps_each=n_steps_each, final_only=True,
                                inp_mask_type=inp_mask_type, temp = temp, box_size = box_size_hat,
                                snr = snr, domain = domain, kernel_size = kernel_size, kernel_std = kernel_std, device = device, missing_indices = missing_indices)
                elif sampler == "adaptive":           
                    x = adaptive_sampler(x_init, ctx.model, sigma_init = sigma_begin, n_updates = 1000 , n_steps_each=1, step_cov_update = 1.5e-2, 
                        temp = 0.75, final_only=True, inp_mask_type=inp_mask_type, box_size = box_size, device = device, missing_indices = missing_indices)
                elif sampler == 'PC_adaptive':
                    x = PC_sampler_adapted(x_init, ctx.model, sigmas, n_steps_each=n_steps_each, final_only=True,
                                inp_mask_type=inp_mask_type, temp = temp, box_size = box_size_hat,
                                snr = snr, domain = domain, kernel_size = kernel_size, kernel_std = kernel_std, device = device, missing_indices = missing_indices)
                elif sampler == "PC_mala":
                    x = PC_sampler_mala(x_init, ctx.model, sigmas, n_steps_each=n_steps_each, final_only=True,
                                inp_mask_type=inp_mask_type, temp = temp, box_size = box_size_hat,
                                snr = snr, domain = domain, kernel_size = kernel_size, kernel_std = kernel_std, device = device, missing_indices = missing_indices)
                
                
                samples = torch.clamp(x[0], 0.0, 1.0)
                ald_data = {
                    'samples': samples,
                    'y': x_init,
                    'x': clean_images[0],
                }
                torch.save(ald_data, f"samples/{dataset_name}_{box_size}_{gamma_cov}_{sampler}_{domain}/samples_posterior_{batch_idx}.pt")
                
                mse_batch = torch.mean((images[0][idx_img:idx_img+batch_size,:,:,:] - samples[0:batch_size])**2, dim=(-1, -2, -3)).sum() 
                for ii in range(batch_size):
                    lpips_batch += loss_fn_alex(images[0][ii+idx_img:ii+idx_img+1,:,:,:].cuda(),  samples[ii:ii+1].cuda())
                
                mse += mse_batch.item()
                lpips_ += lpips_batch.item()

                # 1. Scale by 255.0
                # x_true_scaled = images[0][idx_img:idx_img+batch_size,:,:,:] * 255.0
                # x_true_uint8 = x_true_scaled.to(torch.uint8)
                # kid.update(x_true_uint8, real=True)
                # x_samples_scaled = samples * 255.0
                # x_samples_uint8 = x_samples_scaled.to(torch.uint8)
                # kid.update(x_samples_uint8, real=False)
                
                print("MSE", mse_batch / (batch_size))
                print("LPIPS", lpips_batch / (batch_size))
                # print("KID", kid.compute())
                x_lists[f'samples_{ctx.args.model}'] = x[0]

                nrow = 16
                # grid = torchvision.utils.make_grid(clean_images.cuda().cpu(), nrow=nrow, padding=2)
                # grid_clean = grid.permute(1, 2, 0).numpy()
                if not Path(f"samples/{dataset_name}_{box_size}_{gamma_cov}_{sampler}_{domain}/clean").exists():
                    Path(f"samples/{dataset_name}_{box_size}_{gamma_cov}_{sampler}_{domain}/clean").mkdir(parents=True)
                for ii in range(clean_images.shape[0]):
                    # Save img individually
                    torchvision.utils.save_image(clean_images[ii:ii+1], f'samples/{dataset_name}_{box_size}_{gamma_cov}_{sampler}_{domain}/clean/clean_image_batch{batch_idx}_img{ii}.png', normalize=False)
                

                # # Display the grid
                # plt.figure(figsize=(10, 10)) # Adjust figure size as needed
                # plt.imshow(grid_clean)
                # plt.axis('off') # Hide axes
                # # plt.title("Clean images")
                # plt.tight_layout() # Hide axes
                # plt.savefig(f'samples/clean_images_batch_{batch_idx}.pdf')

                # if domain == "freq":
                #     # cov = [deblurring_covariance_from_shape(spatial_size=img_size, kernel_size=kernel_size, kernel_std=kernel_std, device=device, noise_level=1)] * batch_size
                #     cov = [sr_covariance_from_shape(spatial_size=img_size, kernel_size=kernel_size, device=device, noise_level=1)] * batch_size
                #     x_init = apply_power_to_list_covariances(cov, x_init, p=-0.5).cpu()
                #     grid = torchvision.utils.make_grid(x_init, nrow=nrow, padding=2)
                # else:
                #     grid = torchvision.utils.make_grid(x_init.cpu(), nrow=nrow, padding=2)
                # grid_np = grid.permute(1, 2, 0).numpy()
            
                # # Display the grid
                # plt.figure(figsize=(10, 10)) # Adjust figure size as needed
                # plt.imshow(grid_np)
                # plt.axis('off') # Hide axes
                # # plt.title("Noisy images")
                # plt.tight_layout()
                # plt.savefig(f'samples/noisy_images_batch_{batch_idx}.pdf')
            
                samples = torch.clamp(x_lists[f'samples_{ctxs["Energy-Dual"].args.model}'], 0.0, 1.0)
                if not Path(f"samples/{dataset_name}_{box_size}_{gamma_cov}_{sampler}_{domain}/generated").exists():
                    Path(f"samples/{dataset_name}_{box_size}_{gamma_cov}_{sampler}_{domain}/generated").mkdir(parents=True)
                for ii in range(samples.shape[0]):
                    # Save img individually
                    torchvision.utils.save_image(samples[ii:ii+1], f'samples/{dataset_name}_{box_size}_{gamma_cov}_{sampler}_{domain}/generated/sample_batch{batch_idx}_img{ii}.png', normalize=False)
                
                grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=2, normalize=False)
                grid_np = grid.permute(1, 2, 0).numpy()
            
                # Display the grid
                plt.figure(figsize=(10, 10)) # Adjust figure size as needed
                plt.imshow(grid_np)
                plt.axis('off') # Hide axes
                # plt.title("Generated images - Energy")
                plt.tight_layout()
                plt.savefig(f'samples/{dataset_name}_{box_size}_{gamma_cov}_{sampler}_{domain}/generated/generated_sample_batch_energy_{model_}_{batch_idx}_ula.pdf')
            
            print("MSE total", mse / (batch_size * (batch_idx+1)))
            print("LPIPS total", lpips_ / (batch_size * (batch_idx+1)))
            if batch_idx % 20 == 0 and batch_idx > 0: #20
            # if batch_idx == 0:
            # if batch_idx == 0:
                break
            