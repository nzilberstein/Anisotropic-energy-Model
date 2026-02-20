import random
import json

import lpips
import torchvision

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

    # Disable DataParallel (needed for Hessian computation)
    # ctx.model.network = ctx.model.network.module

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




if __name__ == "__main__":

    for seed in [20]:
        # Seed parameterers
        seed = seed #20
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        ## Load models

        args = load_args("multigpu/inpainting_imagenet/denoiser_song_score_anisoEmb_groupNorm_mult_lr2e-4_lrdecay50000_1000warmup_128bs", step = "best")
        # args = load_args("multigpu/inpainting/denoiser_song_score_anisoEmb_groupNorm_mult_lr2e-4_lrdecay50000_1000warmup", step = "best")
        # args = load_args("multigpu/inpainting_celeba/denoiser_song_score_anisoEmb_groupNorm_mult_lr2e-4_lrdecay50000_1000warmup_128bs", step = "best")
        step = "last"
        ctxs = {
            "Denoiser": load_exp("multigpu/inpainting_imagenet/denoiser_song_score_anisoEmb_groupNorm_mult_lr2e-4_lrdecay50000_1000warmup_128bs", step = step),
            # "Denoiser": load_exp("multigpu/inpainting/denoiser_song_score_anisoEmb_groupNorm_mult_lr2e-4_lrdecay50000_1000warmup", step = step),
            # "Denoiser": load_exp("multigpu/inpainting_celeba/denoiser_song_score_anisoEmb_groupNorm_mult_lr2e-4_lrdecay50000_1000warmup_128bs", step = step),
            # "Denoiser": load_exp("multigpu/inpainting_afhq/denoiser_song_score_anisoEmb_groupNorm_mult_lr2e-4_lrdecay50000_1000warmup_128bs", step = step),
 
        }


        loss_fn_alex = lpips.LPIPS(net='alex').cuda() # best forward scores

        time_tracker: TimeTracker = TimeTracker()
        time_tracker.switch("initialization")

        default_ctx = ctxs["Denoiser"]
        device = default_ctx.device
        dataset_info = default_ctx.dataset_info
        d = dataset_info.dimension

        ## Load data 
        test_batch_size = 20 #50 #20
        img_size = dataset_info.spatial_size
        CHW = 3 * img_size * img_size
        
        train_dataloader, test_dataloader, dataset_info = load_data(
            dataset=args["dataset"], spatial_size=args["spatial_size"], grayscale=args["grayscale"], horizontal_flip=False, data_subset=eval(args["data_subset"]),
            train_batch_size=test_batch_size, test_batch_size=test_batch_size, num_workers=args["num_workers"], seed=seed
        )

        # images = next(iter(test_dataloader))  # load for testing things.
        # shape = (images[0].shape[0], CHW)
    
        ## Sampling and noise parameters
        sigma_begin = 25 #27
        sigma_end = 1e-2
        num_classes = 1000
        sigma_dist = 'geometric'
        sigmas = get_sigmas(sigma_begin, sigma_end, num_classes, device, sigma_dist)
        # Sampling parameters
        batch_size = test_batch_size #64
        step_lr = None
        temp = 1
        # Parameters of the degradation
        sigma_box_init = sigma_begin**2
        sigma_clean_init = 1e-2
        inp_mask_type = "box"
        box_size = 25
        sampler = "dps"
        domain = 'freq'    
        kernel_size = 8
        kernel_std = 0.8

        if inp_mask_type == "random":
            total_pixels = img_size * img_size
            n_missing = int(0.7 * total_pixels)
            missing_indices = torch.randperm(total_pixels, device=device)[:n_missing]
            torch.save(missing_indices, "missing_indices.pt")
            
        else:
            missing_indices = None

        # sigma_begin = 25 #27
        # sigma_end = 1e-2
        # num_classes = 500 # 400
        # sigma_dist = 'geometric'
        # sigmas = get_sigmas(sigma_begin, sigma_end, num_classes, device, sigma_dist)
        # # Sampling parameters
        # batch_size = test_batch_size #64
        # step_lr = None
        # temp = 1
        # # Parameters of the degradation
        # sigma_box_init = sigma_begin**2
        # sigma_clean_init = 1e-2 
        # inp_mask_type = "box"
        # box_size = 45
        # sampler = "reddiff"
        # domain = 'freq'    
        # kernel_size = 8
        # kernel_std = 0.8
            
        # Build covariance and load images
        idx_img = 0
        if domain == "freq":
            deblurring_id = deblurring_covariance_from_shape(spatial_size=img_size, kernel_size=kernel_size, kernel_std=kernel_std, device=device, noise_level=1)
            cov = deblurring_id #deblurring_covariance_from_shape(spatial_size=img_size, kernel_size=kernel_size, kernel_std=kernel_std, device=device, noise_level=sigma_begin**2)
            # cov = sr_covariance_from_shape(spatial_size=img_size, kernel_size=kernel_size, device=device, noise_level=1)    
        else:
            # cov = spatial_corr_covariance_testing(spatial_size=img_size, box_size=box_size, var_box=sigma_box_init, device=device, var_clean = sigma_clean_init, half_box_size = box_size, inp_mask_type=inp_mask_type)
            cov = spatial_corr_covariance_testing(spatial_size=img_size, box_size=box_size, var_box=0, device=device, var_clean = 1, half_box_size = box_size, inp_mask_type=inp_mask_type, missing_indices_input=missing_indices)

        x_lists = {
                f'samples_{ctxs["Denoiser"].args.model}': None
            }

        ## Run sampler
        print("Running sampling...")

        mse = 0
        lpips_ = 0

        domain_name = domain + "wpt_files" #"steps" + str(1200) + "n" # + "posterior_exp" + str(idx_img) #6e-2
        batch_idx_post = seed
        dataset_name = args["dataset"]
        if not Path(f"samples/{dataset_name}_{box_size}_{sampler}_{domain_name}").exists():
            Path(f"samples/{dataset_name}_{box_size}_{sampler}_{domain_name}").mkdir(parents=True)
    
        for batch_idx, images in enumerate(test_dataloader):
            # clean_images = images[0][idx_img:idx_img+batch_size,:,:,:].cuda()
            # N = 32
            clean_images = images[0][idx_img:idx_img+batch_size,:,:,:].cuda()#.repeat(N,1,1,1)
            # print(clean_images.shape)
            print(clean_images.shape)
            for model_, ctx in ctxs.items():
                mse_batch = 0
                lpips_batch = 0
                print(f"Running {model_} ... with batch idx {batch_idx}")
                
                # x_init = clean_images + cov.apply_power(torch.randn_like(clean_images).cuda(), p=0.5)
                if domain == "freq":
                    x_deblur = cov.apply_power(clean_images, p = -0.5)
                    y = x_deblur + 0.01 * torch.randn_like(images[0][0:batch_size,:,:,:]).cuda()
                    x_init = torch.randn_like(clean_images).cuda()
                else:
                    print(domain)
                    y = cov.apply_power(clean_images, p = 1) + sigma_clean_init * torch.randn_like(clean_images).cuda()
                    x_init = torch.randn_like(clean_images).cuda()

                # Run sampler
                if sampler == 'dps':
                    print(domain)
                    x = DPS_sampler(x_init, y, img_size, cov, ctx.model, sigmas, 
                                inp_mask_type=inp_mask_type, 
                                box_size = box_size, device=device, domain = domain, missing_indices=missing_indices)
                elif sampler == "reddiff":
                    x = reddiff_sampler(x_init, y, img_size, cov, ctx.model, sigmas, 
                                inp_mask_type=inp_mask_type, 
                                box_size = box_size, device=device, domain = domain, missing_indices=missing_indices)
                elif sampler == "daps":
                    x = DAPS_sampler(x_init, y, img_size, cov, ctx.model, sigmas, std_noise = sigma_clean_init,
                                inp_mask_type=inp_mask_type, lr_langevin = 2e-6,
                                box_size = box_size, device=device, domain = domain, missing_indices=missing_indices)
                samples = torch.clamp(x[0], 0.0, 1.0)
                ald_data = {
                    'samples': samples,
                    'y': x_init,
                    'x': clean_images[0],
                }
                
                torch.save(ald_data, f"samples/{dataset_name}_{box_size}_{sampler}_{domain_name}/samples_posterior_{batch_idx}.pt")            
                
                samples = torch.clamp(x[0], 0.0, 1.0)
                mse_batch = torch.mean((images[0][idx_img:idx_img+batch_size,:,:,:] - samples[0:batch_size])**2, dim=(-1, -2, -3)).sum() 
                for ii in range(batch_size):
                    lpips_batch += loss_fn_alex(images[0][ii+idx_img:ii+idx_img+1,:,:,:].cuda(),  samples[ii:ii+1].cuda())
                
                mse += mse_batch.item()
                lpips_ += lpips_batch.item()

                print("MSE", mse_batch / (batch_size))
                print("LPIPS", lpips_batch / (batch_size))
                x_lists[f'samples_{ctx.args.model}'] = x[0]

                # grid = torchvision.utils.make_grid(clean_images.cuda().cpu(), nrow=16, padding=2)
                # grid_clean = grid.permute(1, 2, 0).numpy()
                dataset_name = args["dataset"]
                if not Path(f"samples/{dataset_name}_{box_size}_{sampler}_{domain_name}/clean").exists():
                    Path(f"samples/{dataset_name}_{box_size}_{sampler}_{domain_name}/clean").mkdir(parents=True)
                for ii in range(clean_images.shape[0]):
                    # Save img individually
                    torchvision.utils.save_image(clean_images[ii:ii+1], f'samples/{dataset_name}_{box_size}_{sampler}_{domain_name}/clean/clean_image_batch{batch_idx}_img{ii}.png', normalize=False)
                
                grid = torchvision.utils.make_grid(y.cpu(), nrow=16, padding=2)
                grid_np = grid.permute(1, 2, 0).numpy()
            
                # # Display the grid
                plt.figure(figsize=(10, 10)) # Adjust figure size as needed
                plt.imshow(grid_np)
                plt.axis('off') # Hide axes
                # plt.title("Noisy images")
                plt.tight_layout()
                plt.savefig(f'samples/{dataset_name}_{box_size}_{sampler}_{domain_name}/noisy_images_batch_{batch_idx}.pdf')
            
                samples = torch.clamp(x_lists[f'samples_{ctxs["Denoiser"].args.model}'], 0.0, 1.0)
                if not Path(f'samples/{dataset_name}_{box_size}_{sampler}_{domain_name}/generated_{sampler}').exists():
                    Path(f'samples/{dataset_name}_{box_size}_{sampler}_{domain_name}/generated_{sampler}').mkdir(parents=True)
                for ii in range(samples.shape[0]):
                    torchvision.utils.save_image(samples[ii:ii+1], f'samples/{dataset_name}_{box_size}_{sampler}_{domain_name}/generated_{sampler}/sample_batch{batch_idx}_img{ii}.png', normalize=False)
                grid = torchvision.utils.make_grid(samples, nrow=16, padding=2, normalize=False)
                grid_np = grid.permute(1, 2, 0).numpy()
            
                # Display the grid
                plt.figure(figsize=(10, 10)) # Adjust figure size as needed
                plt.imshow(grid_np)
                plt.axis('off') # Hide axes
                # plt.title("Generated images - Energy")
                plt.tight_layout()
                plt.savefig(f'samples/{dataset_name}_{box_size}_{sampler}_{domain_name}/generated_{sampler}/generated_sample_batch_energy_dps_{batch_idx}.pdf')

            print("MSE total", mse / (batch_size * (batch_idx+1)))
            print("LPIPS total", lpips_ / (batch_size * (batch_idx+1)))
            if batch_idx % 20 == 0 and batch_idx > 0:
                break\
            # if batch_idx == 0:
            #     break
            # if batch_idx == 0:
            #     break
            