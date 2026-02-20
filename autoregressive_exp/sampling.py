from sympy import var
from pathlib import Path 
import random
import torchvision
import lpips

from utils import *
from data import *
from trackers import *
from main import *
from networks.conditioning import *
from noise import *
from networks.classifier import ImageClassifier
from visualization import *
# from samplers import dual_sampler

seed = 0
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

    args_dict['size_network'] = "small"
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


def dual_sampler(x_mod_ald, x_mod_pc, scorenet, sigmas, n_steps_each=200, step_lr=0.000008, batch_size=1,
                 final_only=False, verbose=True, denoise=True, factor_sigma_2=1, data_score=False,
                 inp_mask_type="half", x_clean=None, half_box_size=12, snr=0.5, init_cov=None,
                 box_size=12, temp=2, num_boxes=1, perm=None, num_high=784, sigma_init_ald=2):
    """
    Runs both Annealed Langevin Dynamics and PC Sampler simultaneously with shared noise.
    
    Parameters:
    -----------
    x_mod_ald : torch.Tensor
        Initial samples for Annealed Langevin Dynamics
    x_mod_pc : torch.Tensor
        Initial samples for PC Sampler
    
    Returns:
    --------
    dict containing results from both samplers
    """
    # Storage for both methods
    results = {
        'ald': {
            'images': [],
            'mse_list': [],
            'samples_to_save': [],
            'cov_to_save': [],
            'delta_to_save': [],
            'energy_values': torch.zeros(len(sigmas), batch_size)
        },
        'pc': {
            'images': [],
            'samples_to_save': [],
            'cov_to_save': [],
            'iter_print_img': 0
        }
    }

    var_clean = 1e-7
    num_low = num_boxes - num_high
    H = x_mod_ald.shape[-1]
    # ===========================
    # Initialize for ALD
    # ===========================
    noise_levels = torch.cat([ #2
        torch.full((num_high,), sigma_init_ald**2, device=device),
        torch.full((num_low,), (1e-7)**2, device=device)
    ])
    
    noise_levels = noise_levels[perm]
    init_cov, box_size, centers = build_dynamic_noise_mask_from_array(
                spatial_size=H,
                num_boxes=num_boxes,
                noise_levels=noise_levels,
                device=device,
                overlap_mode='non_overlap'
            )
    ald_covariance = [SpatialCorrCovariance(matrix=init_cov.to(device=device))] * batch_size
    noise_levels_id = torch.cat([
        torch.full((num_high,), 1, device=device),
        torch.full((num_low,), 0, device=device)
    ])
    noise_levels_id = noise_levels_id[perm]
    
    ald_covariance_id = [spatial_corr_covariance_testing(
        spatial_size=H, box_size=box_size, var_box=0, device=device, var_clean=1, 
        half_box_size=box_size, inp_mask_type="autoregressive", num_boxes=num_boxes, 
        noise_levels=noise_levels_id)]
    
    ald_cov_sample = ald_covariance[0].get_matrix()
    ald_cov_sample = ald_cov_sample[None, None, :,:].repeat(batch_size, 1, 1, 1)
    ald_current_variance = sigmas[0]
    ald_noise_level = NoiseLevel(variance=sigmas[0]**2)
    gamma_cov = 4e-2#1e-2
    temp_ald = 1 #10e-1
    # clamp_val = 1e0
    max_norm = 5
    
    # ===========================
    # Initialize for PC
    # ===========================
    pc_noise_levels_id = torch.cat([
        torch.full((num_high,), 1, device=device),
        torch.full((num_low,), 0, device=device)
    ])
    pc_noise_levels_id = pc_noise_levels_id[perm]
    
    pc_covariance_id = [spatial_corr_covariance_testing(
        spatial_size=H, box_size=box_size, var_box=0, device=device, var_clean=1, 
        half_box_size=box_size, inp_mask_type=inp_mask_type, num_boxes=num_boxes, 
        noise_levels=pc_noise_levels_id)]
    
    with torch.no_grad():
        # Iterate through sigma levels (PC uses [:-2], ALD uses all)
        for c in range(len(sigmas) - 2):  # Use PC's shorter range as base
            
            # ============================================
            # PC SAMPLER STEP (using shared noise)
            # ============================================
            sigma_curr = sigmas[c]
            sigma_next = sigmas[c+1]


            # Setup PC covariance for current sigma
            pc_noise_levels = torch.cat([
                torch.full((num_high,), sigma_curr**2, device=device),
                torch.full((num_low,), var_clean, device=device)
            ])
            pc_noise_levels = pc_noise_levels[perm]
            pc_covariance = [spatial_corr_covariance_testing(
                spatial_size=H, box_size=box_size, var_box=0, device=device, var_clean=1, 
                half_box_size=box_size, inp_mask_type="autoregressive", num_boxes=num_boxes, 
                noise_levels=pc_noise_levels)] * batch_size
            
            pc_noise_level = NoiseLevel(variance=sigma_curr**2)
            pc_input = ModelInput(noisy=x_mod_pc, noise_level=pc_noise_level, covariance=pc_covariance)
            pc_model_output = scorenet.forward(pc_input, create_graph=False)
            pc_grad = -pc_model_output.data_score
            
            # Setup step covariance for PC
            diff = sigma_curr**2 - sigma_next**2
            pc_step_noise_levels = torch.cat([
                torch.full((num_high,), diff, device=device),
                torch.full((num_low,), var_clean, device=device)
            ])
            pc_step_noise_levels = pc_step_noise_levels[perm]
            pc_covariance_step = [spatial_corr_covariance_testing(
                spatial_size=H, box_size=box_size, var_box=0, device=device, var_clean=1, 
                half_box_size=box_size, inp_mask_type="autoregressive", num_boxes=num_boxes, 
                noise_levels=pc_step_noise_levels)] * batch_size
            
            # SHARED NOISE: Generate once for both samplers
            shared_noise = torch.randn_like(x_mod_pc)
            
            # Update PC with shared noise
            x_mod_pc = x_mod_pc + pc_covariance_step[0].apply_power(pc_grad, p=1) + \
                       pc_covariance_step[0].apply_power(shared_noise, p=0.5)

            for s in range(n_steps_each):
                pc_step_noise_levels = torch.cat([
                    torch.full((num_high,), sigma_next**2, device=device),
                    torch.full((num_low,), var_clean, device=device)
                ])
                pc_step_noise_levels = pc_step_noise_levels[perm]
                pc_covariance = [spatial_corr_covariance_testing(
                    spatial_size=H, box_size=box_size, var_box=0, device=device, var_clean=1, 
                    half_box_size=box_size, inp_mask_type="autoregressive", num_boxes=num_boxes, 
                    noise_levels=pc_step_noise_levels)] * batch_size
                
                noise_level = NoiseLevel(variance=(sigma_next**2))                
                input = ModelInput(noisy=x_mod_pc, noise_level=noise_level, covariance=pc_covariance)
                model_output = scorenet.forward(input, create_graph = False)
                grad = -model_output.data_score

    
                noise = torch.randn_like(x_mod_pc)
                grad_norm = torch.norm(pc_covariance_id[0].apply_power(grad, p = 1).view(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(pc_covariance_id[0].apply_power(noise, p = 1).view(noise.shape[0], -1), dim=-1).mean()
                step_size = (snr * noise_norm / grad_norm) ** 2 * 2
                pc_step_noise_levels = torch.cat([
                    torch.full((num_high,), step_size**2, device=device),
                    torch.full((num_low,), var_clean, device=device)
                ])
                pc_step_noise_levels = pc_step_noise_levels[perm]
                pc_covariance_step = [spatial_corr_covariance_testing(
                    spatial_size=H, box_size=box_size, var_box=0, device=device, var_clean=1, 
                    half_box_size=box_size, inp_mask_type="autoregressive", num_boxes=num_boxes, 
                    noise_levels=pc_step_noise_levels)] * batch_size
                # covariance_step = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=var_clean, device=device, var_clean = step_size, inp_mask_type=inp_mask_type, half_box_size = box_size)] * batch_size
                x_mod_pc = x_mod_pc + pc_covariance_step[0].apply_power(grad, p=1) + np.sqrt(temp) * pc_covariance_step[0].apply_power(noise, p=0.5)

            
            results['pc']['cov_to_save'].append(pc_covariance_step[0].get_matrix())
            results['pc']['iter_print_img'] += 1
            
            # ============================================
            # ALD STEP (using same shared noise)
            # ============================================
            if c < len(sigmas):  # ALD can run for full range
                # if c % 200 == 0 and c > 0:
                #     gamma_cov = gamma_cov * 1.5
                #     if gamma_cov > 5e-2:
                #         gamma_cov = 5e-2
                    
                ald_input = ModelInput(noisy=x_mod_ald, noise_level=ald_noise_level, covariance=ald_covariance)
                ald_model_output = scorenet.forward(ald_input, create_graph=False)
                
                if data_score:
                    ald_grad = -ald_model_output.data_score
                    ald_grad_cov = -ald_model_output.noise_score
                    # print(ald_grad_cov.max)
                    # ald_grad_cov = ald_grad_cov.sign() * torch.clamp(ald_grad_cov, min=-clamp_val, max=clamp_val)
                    torch.nn.utils.clip_grad_norm_([ald_grad_cov], max_norm=max_norm)
                    ald_grad_cov = ald_grad_cov.sign() * ald_grad_cov
                    results['ald']['energy_values'][c, :] = ald_model_output.energy
                else:
                    ald_denoised = ald_model_output.denoised
                    ald_grad = ald_covariance[0].apply_power(ald_denoised - x_mod_ald, p=-1)
                
                # Check for NaN/Inf
                if data_score and (torch.isnan(ald_grad_cov).any() or torch.isinf(ald_grad_cov).any()):
                    print(f"!!! Step {c}: ALD grad_cov from scorenet is NaN or Inf!")
                    import sys; sys.exit()
                
                if data_score:
                    ald_delta_matrix = gamma_cov * (ald_cov_sample * ald_grad_cov * ald_cov_sample)
                    
                    if torch.isnan(ald_delta_matrix).any() or torch.isinf(ald_delta_matrix).any():
                        print(f"!!! Step {c}: ALD delta_matrix exploded to NaN or Inf!")
                        import sys; sys.exit()
                    
                    # Update ALD covariance
                    ald_cov_sample = ald_cov_sample - ald_delta_matrix
                    ald_cov_sample = torch.clamp(ald_cov_sample, min=1e-9)
                    
                    # Create new covariance objects
                    ald_covariance_update = []
                    ald_covariance = []
                    for bss in range(batch_size):
                        ald_covariance_update.append(SpatialCorrCovariance(matrix=ald_delta_matrix[bss,0,:,:].to(device=device)))
                        ald_covariance.append(SpatialCorrCovariance(matrix=ald_cov_sample[bss,0,:,:].to(device=device)))

                    ald_input = ModelInput(noisy=x_mod_ald, noise_level=ald_noise_level, covariance=ald_covariance)
                    ald_model_output = scorenet.forward(ald_input, create_graph=False)
                    ald_grad = -ald_model_output.data_score
                    
                    # Update ALD with SAME shared noise as PC
                    for bss in range(batch_size):
                        x_mod_ald[bss:bss+1] = x_mod_ald[bss:bss+1] + \
                                               step_lr * ald_covariance_update[bss].apply_power(ald_grad[bss:bss+1], p=1) + \
                                               np.sqrt(temp_ald * step_lr) * ald_covariance_update[bss].apply_power(shared_noise[bss:bss+1], p=0.5)
                    
                    # Get updated gradient
                    for inner_step in range(n_steps_each):
                        noise = torch.randn_like(x_mod_ald)
                        ald_input = ModelInput(noisy=x_mod_ald, noise_level=ald_noise_level, covariance=ald_covariance)
                        ald_model_output = scorenet.forward(ald_input, create_graph=False)
                        if data_score:
                            ald_grad = -ald_model_output.data_score
                        else:
                            ald_denoised = ald_model_output.denoised
                            ald_grad = ald_covariance[0].apply_power(ald_denoised - x_mod_ald, p=-1)
                        
                        # Update ALD with SAME shared noise as PC
                        for bss in range(batch_size):
                            x_mod_ald[bss:bss+1] = x_mod_ald[bss:bss+1] + \
                                                   step_lr * ald_covariance_update[bss].apply_power(ald_grad[bss:bss+1], p=1) + \
                                                   np.sqrt(temp * step_lr) * ald_covariance_update[bss].apply_power(noise[bss:bss+1], p=0.5)
                        

                    # Update ALD noise level
                    ald_current_variance = torch.trace(ald_cov_sample[0,0,:,:]) / ald_cov_sample.shape[0]
                    ald_noise_level = NoiseLevel(variance=ald_current_variance)
                    
                    assert (ald_cov_sample >= 0).all(), "Error: ALD Covariance matrix is not PSD!"
                    results['ald']['delta_to_save'].append(torch.log(torch.abs(ald_delta_matrix.float()) / gamma_cov))
            
            # ============================================
            # Save results for both
            # ============================================
            if not final_only:
                results['ald']['images'].append(x_mod_ald.to('cpu'))
                results['pc']['images'].append(x_mod_pc.to('cpu'))
            
            ald_samples = torch.clamp(x_mod_ald, 0.0, 1.0)
            # ald_samples = torch.clamp(ald_model_output.denoised, 0.0, 1.0)
            pc_samples = torch.clamp(x_mod_pc, 0.0, 1.0)
            # pc_samples = torch.clamp(pc_model_output.denoised, 0.0, 1.0)
            results['ald']['samples_to_save'].append(ald_samples)
            results['pc']['samples_to_save'].append(pc_samples)
            
            if data_score:
                results['ald']['cov_to_save'].append(ald_cov_sample)
            
            # Optional: MSE computation
            if x_clean is not None:
                ald_e = x_mod_ald - x_clean
                pc_e = x_mod_pc - x_clean
                ald_mse = torch.mean(ald_e ** 2, dim=(-1, -2, -3))
                pc_mse = torch.mean(pc_e ** 2, dim=(-1, -2, -3))
                results['ald']['mse_list'].append(ald_mse.cpu())
                results['pc']['mse_list'] = results['pc'].get('mse_list', [])
                results['pc']['mse_list'].append(pc_mse.cpu())
            
            # Optional: Visualization
            if verbose and c % 100 == 0:
                print(f"Step {c} - ALD Energy: {ald_model_output.energy if data_score else 'N/A'}")
                print(f"Step {c} - ALD Variance: {ald_cov_sample.mean()}")
                print(f"Step {c} - PC Variance: {sigma_curr**2}")

                    # Visualize both samplers
                # visualize_dual_grids(ald_samples, pc_samples, ald_delta_matrix if data_score else None, 
                #                     c, gamma_cov, nrow=30)
    
    # Final processing
    if final_only:
        # Save ALD results
        ald_data = {
            'samples': results['ald']['samples_to_save'],
            'covariances': results['ald']['cov_to_save'],
            'delta_cov': results['ald']['delta_to_save'],
            'denoiser_output': ald_model_output.denoised,
        }
        torch.save(ald_data, "samples_cov_adapted_dual.pt")
        
        # Save PC results
        pc_data = {
            'samples': results['pc']['samples_to_save'],
            'covariances': results['pc']['cov_to_save'],
            'denoiser_output': pc_model_output.denoised,
        }
        torch.save(pc_data, "samples_cov_classical_dual.pt")
        
        print(f"Successfully saved dual sampler results")
        
        return {
            'ald': ([x_mod_ald.to('cpu')], results['ald']['mse_list'], results['ald']['energy_values']),
            'pc': ([x_mod_pc.to('cpu')], results['pc'].get('mse_list', []))
        }
    else:
        return {
            'ald': results['ald']['images'],
            'pc': results['pc']['images']
        }

if __name__ == "__main__":

    ## Load models
    args = load_args("multigpu/inpainting_mnist/energy_songSmall_autoregressive_MNIST", step = "best")
    step = "last"
    ctxs = {
        "Energy-Dual": load_exp("multigpu/inpainting_mnist/energy_songSmall_autoregressive_MNIST", step = step),
    }

    loss_fn_alex = lpips.LPIPS(net='alex').cuda() # best forward scores
    classifier = ImageClassifier().to('cuda')

    with open('model_state.pt', 'rb') as f: 
        classifier.load_state_dict(torch.load(f))  
        
    ## General parameters
    time_tracker: TimeTracker = TimeTracker()
    time_tracker.switch("initialization")


    default_ctx = ctxs["Energy-Dual"]
    device = default_ctx.device
    dataset_info = default_ctx.dataset_info
    d = dataset_info.dimension

    ## Load data 

    test_batch_size = 256 #32
    img_size = 28
    CHW = 1 * img_size * img_size
    
    train_dataloader, test_dataloader, dataset_info = load_data(
        dataset=args["dataset"], spatial_size=args["spatial_size"], grayscale=args["grayscale"], data_subset=eval(args["data_subset"]), horizontal_flip = False,
        train_batch_size=test_batch_size, test_batch_size=test_batch_size, num_workers=args["num_workers"], seed=seed
    )
 
    ## Sampling and noise parameters
    sigma_begin = 1 * 10
    sigma_end = 1e-2
    num_classes = 500
    sigma_dist = 'geometric'
    n_steps_each = 5
    batch_size = test_batch_size
    idx_img = 0 #torch.randint(0, images[0].shape[0], (1,)).item()

    
    snr = 0.1
    sigma_box_init = sigma_begin**2
    sigma_clean_init = 1e-7**2#sigma_begin**2 #1e-7
    inp_mask_type = "input"

    sigmas = get_sigmas(sigma_begin, sigma_end, num_classes, device, sigma_dist)

    ## Build covariance
    idx_img = 0 #torch.randint(0, images[0].shape[0], (1,)).item()
    num_boxes_options = [1, 4, 16, 49, 196, 784] # For MNIST
    num_boxes = 784 #np.random.choice(num_boxes_options)
    num_high = num_boxes - 300#// 2
    num_low = num_boxes - num_high

    # min_noise_level: NoiseLevel = NoiseLevel.from_unit(dataset_info=dataset_info, x = 1, unit = 'mse') # This calls denoising error
    # max_noise_level: NoiseLevel = NoiseLevel.from_unit(dataset_info=dataset_info, x = 2**2, unit = 'mse')
    # noise_level_sampler: NoiseLevelSampler = eval(args["noise_level_sampler"])(min=min_noise_level, max=max_noise_level)

    # noise_levels = torch.cat([
    #     torch.full((num_high,), sigma_box_init, device=device),
    #     torch.full((num_low,), sigma_clean_init, device=device)
    # ])

    noise_levels = torch.cat([
        torch.full((num_high,), 0, device=device),
        torch.full((num_low,), 1, device=device)
    ])

    perm = torch.randperm(num_boxes, device=device)
    noise_levels = noise_levels[perm]

    matrix, box_size, centers = build_dynamic_noise_mask_from_array(
                spatial_size=img_size,
                num_boxes=num_boxes,
                noise_levels=noise_levels,
                device=device,
                overlap_mode='non_overlap'
            )
        
    cov = spatial_corr_covariance_testing(spatial_size=img_size, box_size=box_size, var_box=0, device=device, var_clean = 1, half_box_size = box_size, inp_mask_type=inp_mask_type, matrix = matrix)

    ## Build dict and run the sampler
    x_lists = {f'samples_{ctxs["Energy-Dual"].args.model}': None, 
            }

    total_classification_error_ald = 0
    total_classification_error_pc = 0
    total_classification_error_gt = 0

    for batch_idx, images in enumerate(test_dataloader):
        clean_images = images[0].cuda()
        x_init = cov.apply_power(clean_images, p = 1) + np.sqrt(sigma_clean_init) * torch.randn_like(images[0][0:batch_size,:,:,:]).cuda()
        x_init_same = x_init[0:batch_size].clone()

        # Initialize both starting points (can be the same or different)
        x_init_ald = x_init_same.clone()
        x_init_pc = x_init_same.clone()
        
        # nrow = 5
        # fig, ax = plt.subplots(1, 2)
        # ax[0].set_title('Noisy Image')
        # ax[1].set_title('Clean Image')
        # grid = torchvision.utils.make_grid(x_init, nrow=nrow, padding=2)
        # grid_np = grid.permute(1, 2, 0).cpu().numpy()
        # ax[0].imshow(grid_np)
        # grid = torchvision.utils.make_grid(images[0][idx_img:idx_img+batch_size], nrow=nrow, padding=2, normalize=False)
        # grid_np = grid.permute(1, 2, 0).cpu().numpy()
        # ax[1].imshow(grid_np)
        # ax[0].axis("off")
        # ax[1].axis("off")
        # plt.savefig("samples/initial_noisy_clean.png")

        for model, ctx in ctxs.items():
            print(f"Running...")
            results = dual_sampler(
                x_mod_ald=x_init_ald,
                x_mod_pc=x_init_pc,
                scorenet=ctx.model,
                sigmas=sigmas,
                n_steps_each=n_steps_each,
                step_lr=1e-0,
                batch_size=batch_size,
                final_only=True,
                data_score=True,
                init_cov=matrix,
                box_size=box_size,
                num_boxes=num_boxes,
                perm=perm,
                num_high=num_high,
                snr=snr,
                x_clean=clean_images,
                sigma_init_ald=0.5, #2 for 50 boxes,
                temp = 0.5
            )

                # Access results
            ald_samples, ald_mse, ald_energy = results['ald']
            pc_samples, pc_mse = results['pc']

            samples_ald = torch.clamp(ald_samples[0], 0.0, 1.0).cuda()
            samples_pc = torch.clamp(pc_samples[0], 0.0, 1.0).cuda()
            output_ald = classifier(samples_ald[:,:,:,:])
            output_pc = classifier(samples_pc[:,:,:,:])
            output_gt = classifier(images[0].cuda())
            
            predicted_label_ald = torch.argmax(output_ald, dim = 1)
            total_classification_error_ald += (predicted_label_ald.cpu() != images[1][idx_img:idx_img+batch_size]).sum() 
            predicted_label_pc = torch.argmax(output_pc, dim = 1)
            total_classification_error_pc += (predicted_label_pc.cpu() != images[1][idx_img:idx_img+batch_size]).sum()
            predicted_label_gt = torch.argmax(output_gt, dim = 1)
            total_classification_error_gt += (predicted_label_gt.cpu() != images[1]).sum()

            print(f"Total errors adaptive: {total_classification_error_ald / ((batch_idx + 1) * batch_size)}")

            print(f"Total errors classical: {total_classification_error_pc / ((batch_idx + 1) * batch_size)}")
            
            print(f"Total errors classical: {total_classification_error_gt / ((batch_idx + 1) * batch_size)}")
        
        if batch_idx > 0 and batch_idx % 10 == 0:
            break

    results = {'errors_adaptive': total_classification_error_ald.item() / ((batch_idx + 1) * batch_size),
               'errors_classical': total_classification_error_pc.item() / ((batch_idx + 1) * batch_size)
              }
    # results = {'errors_gt': total_classification_error_gt.item() / ((batch_idx + 1) * batch_size),
    #           }

    torch.save(results, f"classification_sampling_results_dual_{300}_{(batch_idx + 1) * test_batch_size}samples_sigmiainit{0.5}_temp_{0.5}.pt")

    # torch.save(results, f"classification_sampling_results_gt.pt")