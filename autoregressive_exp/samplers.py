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
from visualization import *



def dual_sampler(x_mod_ald, x_mod_pc, scorenet, sigmas, n_steps_each=200, step_lr=0.000008, batch_size=1,
                 final_only=False, verbose=True, data_score=False, device = 'cpu',
                 inp_mask_type="half", x_clean=None, snr=0.5, init_cov=None,
                 box_size=12, temp=1, num_boxes=1, perm=None, num_high=784, sigma_init_ald=2):
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

    img_size = x_mod_ald.shape[-1]
    var_clean = 1e-7
    num_low = num_boxes - num_high
    # ===========================
    # Initialize for ALD
    # ===========================
    noise_levels = torch.cat([ #2
        torch.full((num_high,), sigma_init_ald**2, device=device),
        torch.full((num_low,), var_clean**2, device=device)
    ])
    
    noise_levels = noise_levels[perm]
    init_cov, box_size, _ = build_dynamic_noise_mask_from_array(
                spatial_size=img_size,
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
    
    ald_cov_sample = ald_covariance[0].get_matrix()
    ald_cov_sample = ald_cov_sample[None, None, :,:].repeat(batch_size, 1, 1, 1)
    ald_current_variance = sigmas[0]
    ald_noise_level = NoiseLevel(variance=sigmas[0]**2)
    gamma_cov = 6e-2#1e-2
    temp_ald = 10e-1
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
        spatial_size=img_size, box_size=box_size, var_box=0, device=device, var_clean=1, 
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
                spatial_size=img_size, box_size=box_size, var_box=0, device=device, var_clean=1, 
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
                spatial_size=img_size, box_size=box_size, var_box=0, device=device, var_clean=1, 
                half_box_size=box_size, inp_mask_type="autoregressive", num_boxes=num_boxes, 
                noise_levels=pc_step_noise_levels)] * batch_size
            
            # SHARED NOISE: Generate once for both samplers
            shared_noise = torch.randn_like(x_mod_pc)
            
            # Update PC with shared noise
            x_mod_pc = x_mod_pc + pc_covariance_step[0].apply_power(pc_grad, p=1) + \
                       pc_covariance_step[0].apply_power(shared_noise, p=0.5)

            ## Correctors
            for s in range(n_steps_each):
                pc_step_noise_levels = torch.cat([
                    torch.full((num_high,), sigma_next**2, device=device),
                    torch.full((num_low,), var_clean, device=device)
                ])
                pc_step_noise_levels = pc_step_noise_levels[perm]
                pc_covariance = [spatial_corr_covariance_testing(
                    spatial_size=img_size, box_size=box_size, var_box=0, device=device, var_clean=1, 
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
                    spatial_size=img_size, box_size=box_size, var_box=0, device=device, var_clean=1, 
                    half_box_size=box_size, inp_mask_type="autoregressive", num_boxes=num_boxes, 
                    noise_levels=pc_step_noise_levels)] * batch_size
                # covariance_step = [spatial_corr_covariance_testing(spatial_size=img_size, box_size=box_size, var_box=var_clean, device=device, var_clean = step_size, inp_mask_type=inp_mask_type, half_box_size = box_size)] * batch_size
                x_mod_pc = x_mod_pc + pc_covariance_step[0].apply_power(grad, p=1) + np.sqrt(2 * temp) * pc_covariance_step[0].apply_power(noise, p=0.5)

            
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
                if data_score:
                    ald_grad = -ald_model_output.data_score
                else:
                    ald_denoised = ald_model_output.denoised
                    ald_grad = ald_covariance[0].apply_power(ald_denoised - x_mod_ald, p=-1)
                
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
                                                np.sqrt(0.5 * step_lr) * ald_covariance_update[bss].apply_power(noise[bss:bss+1], p=0.5)
                    

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
                visualize_dual_grids(ald_samples, pc_samples, ald_delta_matrix if data_score else None, 
                                    c, gamma_cov, nrow=5)
    
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
