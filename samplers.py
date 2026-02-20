import torchvision
from pathlib import Path 

from noise import *
from pathlib import Path 
from data import *
from trackers import *
from main import *
from networks.conditioning import *


def PC_sampler(x_mod, scorenet, sigmas, 
                n_steps_each=200, final_only=False,
                inp_mask_type="half", box_size = 12, temp = 1, 
                snr = 0.1, domain='pixel',
                kernel_size = 8, kernel_std = 0.8,
                device = 'cpu', missing_indices = None):
    
    images = []
    var_clean = 1e-9**2
    H = x_mod.shape[-1]
    batch_size = x_mod.shape[0]
    covariance_id = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=1, device=device, var_clean = 0, inp_mask_type=inp_mask_type, half_box_size = box_size, missing_indices_input=missing_indices)]
    # covariance_id = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=0, device=device, var_clean = 1, inp_mask_type=inp_mask_type, half_box_size = box_size)]
    iter_print_img = 0

    with torch.no_grad():
        for c, sigma in enumerate(sigmas[:-2]):

            # Predictor step
            sigma_curr = sigmas[c]
            sigma_next = sigmas[c+1]
            
            if domain == "freq":
                # covariance = [deblurring_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, kernel_std=kernel_std, device=device, noise_level=sigma_curr**2)] * batch_size
                covariance = [sr_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, device=device, noise_level=sigma_curr**2)] * batch_size
            else:
                covariance = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=sigma_curr**2, device=device, var_clean = var_clean, inp_mask_type=inp_mask_type, half_box_size = box_size, missing_indices_input=missing_indices)] * batch_size
                # covariance = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=var_clean, device=device, var_clean = sigma_curr**2, inp_mask_type=inp_mask_type, half_box_size = box_size)] * batch_size
            noise_level = NoiseLevel(variance=sigma_curr**2)
            input = ModelInput(noisy=x_mod, noise_level=noise_level, covariance=covariance)
            model_output = scorenet.forward(input, create_graph = False)
            grad = -model_output.data_score

            diff = sigma_curr**2 - sigma_next**2
            if domain == "freq":
                # covariance_step = [deblurring_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, kernel_std=kernel_std, device=device, noise_level=diff)] * batch_size
                # covariance_step_noise = [deblurring_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, kernel_std=kernel_std, device=device, noise_level=diff)] * batch_size
                covariance_step = [sr_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, device=device, noise_level=diff)] * batch_size
                covariance_step_noise = [sr_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, device=device, noise_level=diff)] * batch_size
            else:
                covariance_step = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=diff, device=device, var_clean = var_clean, inp_mask_type=inp_mask_type, half_box_size = box_size, missing_indices_input=missing_indices)] * batch_size
                covariance_step_noise = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=(diff * sigma_next ** 2) / sigma_curr ** 2, device=device, var_clean = var_clean, inp_mask_type=inp_mask_type, half_box_size = box_size, missing_indices_input=missing_indices)] * batch_size
                # covariance_step = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=var_clean, device=device, var_clean = diff, inp_mask_type=inp_mask_type, half_box_size = box_size)] * batch_size
                # covariance_step_noise = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=var_clean, device=device, var_clean = (diff * sigma_next ** 2) / sigma_curr ** 2, inp_mask_type=inp_mask_type, half_box_size = box_size)] * batch_size
            noise = torch.randn_like(x_mod)
            x_mod = x_mod + covariance_step[0].apply_power(grad, p=1) + covariance_step[0].apply_power(noise, p=0.5)

            # Corrector steps
            for s in range(n_steps_each):
                if domain == "freq":
                    covariance = [deblurring_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, kernel_std=kernel_std, device=device, noise_level=sigma_next**2)] * batch_size
                    # covariance = [sr_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, device=device, noise_level=sigma_next**2)] * batch_size
                else:
                    covariance = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=sigma_next**2, device=device, var_clean = var_clean, inp_mask_type=inp_mask_type, half_box_size = box_size, missing_indices_input=missing_indices)] * batch_size
                    # covariance = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=var_clean, device=device, var_clean = sigma_next**2, inp_mask_type=inp_mask_type, half_box_size = box_size)] * batch_size
                noise_level = NoiseLevel(variance=(sigma_next**2))                
                input = ModelInput(noisy=x_mod, noise_level=noise_level, covariance=covariance)
                model_output = scorenet.forward(input, create_graph = False)
                grad = -model_output.data_score

    
                noise = torch.randn_like(x_mod)
                if domain == "freq":
                    grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                    noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                else:
                    grad_norm = torch.norm(covariance_id[0].apply_power(grad, p = 1).view(grad.shape[0], -1), dim=-1).mean()
                    noise_norm = np.sqrt(box_size * box_size * 3)
                    # noise_norm = torch.norm(covariance_id[0].apply_power(noise, p = 1).view(noise.shape[0], -1), dim=-1).mean()
                step_size = (snr * noise_norm / grad_norm) ** 2 * 2
                if domain == "freq":
                    covariance_step = [deblurring_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, kernel_std=kernel_std, device=device, noise_level=step_size)] * batch_size
                    # covariance_step = [sr_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, device=device, noise_level=step_size)] * batch_size
                else:
                    covariance_step = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=step_size, device=device, var_clean = var_clean, inp_mask_type=inp_mask_type, half_box_size = box_size, missing_indices_input=missing_indices)] * batch_size
                # covariance_step = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=var_clean, device=device, var_clean = step_size, inp_mask_type=inp_mask_type, half_box_size = box_size)] * batch_size
                x_mod = x_mod + covariance_step[0].apply_power(grad, p=1) + np.sqrt(2 * temp) * covariance_step[0].apply_power(noise, p=0.5)

                if not final_only:
                    images.append(x_mod.to('cpu'))
                                
                if iter_print_img % 200 == 0:
                    samples = torch.clamp(x_mod, 0.0, 1.0)
                    nrow = 4
                    grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=2, normalize=False)
                    grid_np = grid.permute(1, 2, 0).cpu().numpy()
                     
                    # Display the grid
                    plt.figure(figsize=(7, 7)) # Adjust figure size as needed
                    plt.imshow(grid_np)
                    plt.axis('off') # Hide axes
                    plt.title("Generated images - Energy")
                    plt.savefig(f'samples/sampling_intermediate_iter_{iter_print_img}.pdf')
                    print(f"Saved intermediate sampling image at iteration {iter_print_img}")
                iter_print_img = iter_print_img + 1


            if n_steps_each == 0:
                if iter_print_img % 500 == 0:
                    samples = torch.clamp(x_mod, 0.0, 1.0)
                    nrow = 4
                    grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=2, normalize=False)
                    grid_np = grid.permute(1, 2, 0).cpu().numpy()
                     
                    # Display the grid
                    plt.figure(figsize=(7, 7)) # Adjust figure size as needed
                    plt.imshow(grid_np)
                    plt.axis('off') # Hide axes
                    plt.title("Generated images - Energy")
                    plt.savefig(f'samples/sampling_intermediate_iter_{iter_print_img}.pdf')
                    print(f"Saved intermediate sampling image at iteration {iter_print_img}")
                iter_print_img = iter_print_img + 1

        if final_only:
            return [x_mod.to('cpu')]
        else:
            return images


def PC_sampler_mala(x_mod, scorenet, sigmas, 
                n_steps_each=200, final_only=False,
                inp_mask_type="half", box_size = 12, temp = 1, 
                snr = 0.1, domain='pixel',
                kernel_size = 8, kernel_std = 0.8,
                device = 'cpu', missing_indices = None):
    
    images = []
    var_clean = 1e-9**2
    H = x_mod.shape[-1]
    batch_size = x_mod.shape[0]
    covariance_id = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=1, device=device, var_clean = 0, inp_mask_type=inp_mask_type, half_box_size = box_size, missing_indices_input=missing_indices)]
    # covariance_id = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=0, device=device, var_clean = 1, inp_mask_type=inp_mask_type, half_box_size = box_size)]
    iter_print_img = 0

    with torch.no_grad():
        for c, sigma in enumerate(sigmas[:-2]):

            # Predictor step
            sigma_curr = sigmas[c]
            sigma_next = sigmas[c+1]
            
            if domain == "freq":
                # covariance = [deblurring_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, kernel_std=kernel_std, device=device, noise_level=sigma_curr**2)] * batch_size
                covariance = [sr_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, device=device, noise_level=sigma_curr**2)] * batch_size
            else:
                covariance = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=sigma_curr**2, device=device, var_clean = var_clean, inp_mask_type=inp_mask_type, half_box_size = box_size, missing_indices_input=missing_indices)] * batch_size
                # covariance = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=var_clean, device=device, var_clean = sigma_curr**2, inp_mask_type=inp_mask_type, half_box_size = box_size)] * batch_size
            noise_level = NoiseLevel(variance=sigma_curr**2)
            input = ModelInput(noisy=x_mod, noise_level=noise_level, covariance=covariance)
            model_output = scorenet.forward(input, create_graph = False)
            grad = -model_output.data_score

            diff = sigma_curr**2 - sigma_next**2
            if domain == "freq":
                # covariance_step = [deblurring_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, kernel_std=kernel_std, device=device, noise_level=diff)] * batch_size
                # covariance_step_noise = [deblurring_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, kernel_std=kernel_std, device=device, noise_level=diff)] * batch_size
                covariance_step = [sr_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, device=device, noise_level=diff)] * batch_size
                covariance_step_noise = [sr_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, device=device, noise_level=diff)] * batch_size
            else:
                covariance_step = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=diff, device=device, var_clean = var_clean, inp_mask_type=inp_mask_type, half_box_size = box_size, missing_indices_input=missing_indices)] * batch_size
                covariance_step_noise = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=(diff * sigma_next ** 2) / sigma_curr ** 2, device=device, var_clean = var_clean, inp_mask_type=inp_mask_type, half_box_size = box_size, missing_indices_input=missing_indices)] * batch_size
                # covariance_step = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=var_clean, device=device, var_clean = diff, inp_mask_type=inp_mask_type, half_box_size = box_size)] * batch_size
                # covariance_step_noise = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=var_clean, device=device, var_clean = (diff * sigma_next ** 2) / sigma_curr ** 2, inp_mask_type=inp_mask_type, half_box_size = box_size)] * batch_size
            noise = torch.randn_like(x_mod)
            x_mod = x_mod + covariance_step[0].apply_power(grad, p=1) + covariance_step[0].apply_power(noise, p=0.5)

            # Corrector steps
            for s in range(n_steps_each):
                if domain == "freq":
                    # covariance = [deblurring_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, kernel_std=kernel_std, device=device, noise_level=sigma_next**2)] * batch_size
                    covariance = [sr_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, device=device, noise_level=sigma_next**2)] * batch_size
                else:
                    covariance = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=sigma_next**2, device=device, var_clean = var_clean, inp_mask_type=inp_mask_type, half_box_size = box_size, missing_indices_input=missing_indices)] * batch_size
                    # covariance = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=var_clean, device=device, var_clean = sigma_next**2, inp_mask_type=inp_mask_type, half_box_size = box_size)] * batch_size
                noise_level = NoiseLevel(variance=(sigma_next**2))                
                input = ModelInput(noisy=x_mod, noise_level=noise_level, covariance=covariance)
                model_output = scorenet.forward(input, create_graph = False)
                grad = -model_output.data_score
                log_p_old = -model_output.energy  # log p(x_old)

    
                noise = torch.randn_like(x_mod)
                if domain == "freq":
                    grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                    noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                else:
                    grad_norm = torch.norm(covariance_id[0].apply_power(grad, p = 1).view(grad.shape[0], -1), dim=-1).mean()
                    # noise_norm = np.sqrt(box_size * box_size * 3)
                    noise_norm = torch.norm(covariance_id[0].apply_power(noise, p = 1).view(noise.shape[0], -1), dim=-1).mean()
                step_size = (snr * noise_norm / grad_norm) ** 2 * 2
                if domain == "freq":
                    # covariance_step = [deblurring_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, kernel_std=kernel_std, device=device, noise_level=step_size)] * batch_size
                    covariance_step = [sr_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, device=device, noise_level=step_size)] * batch_size
                else:
                    covariance_step = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=step_size, device=device, var_clean = var_clean, inp_mask_type=inp_mask_type, half_box_size = box_size, missing_indices_input=missing_indices)] * batch_size
                # covariance_step = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=var_clean, device=device, var_clean = step_size, inp_mask_type=inp_mask_type, half_box_size = box_size)] * batch_size
                x_proposed = x_mod + covariance_step[0].apply_power(grad, p=1) + np.sqrt(2 * temp) * covariance_step[0].apply_power(noise, p=0.5)
                
                # Evaluate model at proposed state
                input_proposed = ModelInput(noisy=x_proposed, noise_level=noise_level, covariance=covariance)
                model_output_proposed = scorenet.forward(input_proposed, create_graph=False)

                residual_forward = x_proposed - (x_mod + covariance_step[0].apply_power(grad, p=1))
                whitened_forward = covariance_step[0].apply_power(residual_forward, p=-0.5) / np.sqrt(2 * temp)

                log_p_new = -model_output_proposed.energy  # log p(x_proposed)
                grad_new = -model_output_proposed.data_score
                # Compute new step size for the backward move
                if domain == "freq":
                    grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                    noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                else:
                    grad_norm = torch.norm(covariance_id[0].apply_power(grad_new, p = 1).view(grad.shape[0], -1), dim=-1).mean()
                    # noise_norm = np.sqrt(box_size * box_size * 3)
                    noise_norm = torch.norm(covariance_id[0].apply_power(noise, p = 1).view(noise.shape[0], -1), dim=-1).mean()
                step_size_new = (snr * noise_norm / grad_norm) ** 2 * 2
                if domain == "freq":
                    # covariance_step = [deblurring_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, kernel_std=kernel_std, device=device, noise_level=step_size)] * batch_size
                    covariance_step_new = [sr_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, device=device, noise_level=step_size)] * batch_size
                else:
                    covariance_step_new = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=step_size_new, device=device, var_clean = var_clean, inp_mask_type=inp_mask_type, half_box_size = box_size)] * batch_size
                
                residual_backward = x_mod - (x_proposed + covariance_step_new[0].apply_power(grad_new, p=1))
                whitened_backward = covariance_step_new[0].apply_power(residual_backward, p=-0.5) / np.sqrt(2 * temp)

                # Compute log densities
                log_q_forward = -0.5 * (whitened_forward ** 2).sum(dim=[1,2,3])
                log_q_backward = -0.5 * (whitened_backward ** 2).sum(dim=[1,2,3])

                # Compute acceptance probability
                # For symmetric proposal (Langevin), we only need p(x_new) / p(x_old)
                log_alpha = (log_p_new - log_p_old) / temp  + (log_q_backward - log_q_forward) # Shape: (batch_size,)
                alpha = torch.exp(log_alpha).clamp(max=1.0)  # min(1, p_new/p_old)
                
                # Accept or reject for each sample in batch
                accept = torch.rand(batch_size, device=device) < alpha
                accept = accept.view(-1, 1, 1, 1)  # Reshape for broadcasting
                
                # Update: accept proposal or keep current state
                x_mod = torch.where(accept, x_proposed, x_mod)      

                if c % 50 == 0:
                    print(f"Acceptance rate: {accept.float().mean():.3f}")
                    # print(f"Log p change: {(log_p_new - log_p_old).mean():.3f}")
                    # print(f"Proposal asymmetry: {(log_q_backward - log_q_forward).mean():.3f}")
                    # print(f"Log alpha: {log_alpha.mean():.3f}")

                if not final_only:
                    images.append(x_mod.to('cpu'))
                                
                if iter_print_img % 200 == 0:
                    samples = torch.clamp(x_mod, 0.0, 1.0)
                    nrow = 4
                    grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=2, normalize=False)
                    grid_np = grid.permute(1, 2, 0).cpu().numpy()
                     
                    # Display the grid
                    plt.figure(figsize=(7, 7)) # Adjust figure size as needed
                    plt.imshow(grid_np)
                    plt.axis('off') # Hide axes
                    plt.title("Generated images - Energy")
                    plt.savefig(f'samples/sampling_intermediate_iter_{iter_print_img}.pdf')
                    print(f"Saved intermediate sampling image at iteration {iter_print_img}")
                iter_print_img = iter_print_img + 1


            if n_steps_each == 0:
                if iter_print_img % 500 == 0:
                    samples = torch.clamp(x_mod, 0.0, 1.0)
                    nrow = 4
                    grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=2, normalize=False)
                    grid_np = grid.permute(1, 2, 0).cpu().numpy()
                     
                    # Display the grid
                    plt.figure(figsize=(7, 7)) # Adjust figure size as needed
                    plt.imshow(grid_np)
                    plt.axis('off') # Hide axes
                    plt.title("Generated images - Energy")
                    plt.savefig(f'samples/sampling_intermediate_iter_{iter_print_img}.pdf')
                    print(f"Saved intermediate sampling image at iteration {iter_print_img}")
                iter_print_img = iter_print_img + 1

        if final_only:
            return [x_mod.to('cpu')]
        else:
            return images

def PC_sampler_adapted(x_mod, scorenet, sigmas, 
                n_steps_each=200, final_only=False,
                inp_mask_type="half", box_size = 12, temp = 1, 
                snr = 0.1, domain='pixel',
                kernel_size = 8, kernel_std = 0.03, step_lr = None,
                device = 'cpu', missing_indices = None):
    
    images = []
    var_clean = 0 #1e-9**2
    H = x_mod.shape[-1]
    batch_size = x_mod.shape[0]
    covariance_id = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=1, device=device, var_clean = 0, inp_mask_type=inp_mask_type, half_box_size = box_size, missing_indices_input=missing_indices)]
    # covariance_id = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=0, device=device, var_clean = 1, inp_mask_type=inp_mask_type, half_box_size = box_size)]
    iter_print_img = 0

    with torch.no_grad():
        for c, sigma in enumerate(sigmas[:-2]):

            # Predictor step
            sigma_curr = sigmas[c]
            sigma_next = sigmas[c+1]
            
            if domain == "freq":
                # covariance = [deblurring_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, kernel_std=kernel_std, device=device, noise_level=sigma_curr**2)] * batch_size
                covariance = [sr_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, device=device, noise_level=sigma_curr**2)] * batch_size
            else:
                covariance = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=sigma_curr**2, device=device, var_clean = var_clean, inp_mask_type=inp_mask_type, half_box_size = box_size, missing_indices_input=missing_indices)] * batch_size
                # covariance = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=var_clean, device=device, var_clean = sigma_curr**2, inp_mask_type=inp_mask_type, half_box_size = box_size)] * batch_size
            noise_level = NoiseLevel(variance=sigma_curr**2)
            input = ModelInput(noisy=x_mod, noise_level=noise_level, covariance=covariance)
            model_output = scorenet.forward(input, create_graph = False)
            grad = -model_output.data_score

            diff = sigma_curr**2 - sigma_next**2
            if domain == "freq":
                # covariance_step = [deblurring_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, kernel_std=kernel_std, device=device, noise_level=diff)] * batch_size
                # covariance_step_noise = [deblurring_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, kernel_std=kernel_std, device=device, noise_level=diff)] * batch_size
                covariance_step = [sr_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, device=device, noise_level=diff)] * batch_size
                covariance_step_noise = [sr_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, device=device, noise_level=diff)] * batch_size
            else:
                covariance_step = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=diff, device=device, var_clean = var_clean, inp_mask_type=inp_mask_type, half_box_size = box_size, missing_indices_input=missing_indices)] * batch_size
                covariance_step_noise = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=(diff * sigma_next ** 2) / sigma_curr ** 2, device=device, var_clean = var_clean, inp_mask_type=inp_mask_type, half_box_size = box_size, missing_indices_input=missing_indices)] * batch_size
                # covariance_step = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=var_clean, device=device, var_clean = diff, inp_mask_type=inp_mask_type, half_box_size = box_size)] * batch_size
                # covariance_step_noise = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=var_clean, device=device, var_clean = (diff * sigma_next ** 2) / sigma_curr ** 2, inp_mask_type=inp_mask_type, half_box_size = box_size)] * batch_size
            noise = torch.randn_like(x_mod)
            x_mod = x_mod + covariance_step[0].apply_power(grad, p=1) + covariance_step[0].apply_power(noise, p=0.5)

            # Corrector steps
            covariance = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=sigma_next**2, device=device, var_clean = var_clean, half_box_size = box_size, inp_mask_type=inp_mask_type, missing_indices_input=missing_indices)] * batch_size
            cov_sample = covariance[0].get_matrix()
            cov_sample = cov_sample[None, None, :,:].repeat(batch_size, 1, 1, 1)
            for s in range(n_steps_each):
                noise_level = NoiseLevel(variance=sigmas[0]**2)
                with torch.no_grad():
                    # for c, sigma in enumerate(sigmas):
                    gamma_cov = 0.01e-2 #6e-2, and 0.6e-2 for 5 steps
                    clamp_val = 5e1 #10e1
                    max_norm = 20

                    input = ModelInput(noisy=x_mod, noise_level=noise_level, covariance=covariance)
                    model_output = scorenet.forward(input, create_graph = False)
                
                    grad = -model_output.data_score
                    grad_cov = -model_output.noise_score
                    
                    #
                    log_p_old = -model_output.energy  # log p(x_old)
                    #
                    grad_cov = grad_cov.sign() * torch.clamp(grad_cov, min=-clamp_val, max=clamp_val) # Clamp to a large but not infinite value
                    # torch.nn.utils.clip_grad_norm_([grad_cov], max_norm=max_norm)
                    # grad_cov = grad_cov.sign() * grad_cov

                    delta_matrix = gamma_cov * (cov_sample * grad_cov * cov_sample)
                    cov_sample = cov_sample - delta_matrix
     
                    cov_sample = torch.clamp(cov_sample, min=1e-9)
                    
                    # === Create New Covariance Object ===
                    # covariance_update = [SpatialCorrCovariance(matrix=delta_matrix.to(device=device))] * batch_size
                    covariance_update = []
                    covariance = []
                    # x_proposed = torch.zeros_like(x_mod)
                    for bss in range(batch_size):
                        covariance_update.append(SpatialCorrCovariance(matrix=delta_matrix[bss,0,:,:].to(device=device)))
                        covariance.append(SpatialCorrCovariance(matrix=cov_sample[bss,0,:,:].to(device=device)))
            
                        noise = torch.randn_like(x_mod[bss:bss+1])
                        x_mod[bss:bss+1] = x_mod[bss:bss+1] + covariance_update[bss].apply_power(grad[bss:bss+1], p=1) + np.sqrt(2 * temp) * covariance_update[bss].apply_power(noise, p=0.5)
                        # x_proposed[bss:bss+1] = x_mod[bss:bss+1] + covariance_update[bss].apply_power(grad[bss:bss+1], p=1) + np.sqrt(2 * temp) * covariance_update[bss].apply_power(noise, p=0.5)


                    # Evaluate model at proposed state
                    # input_proposed = ModelInput(noisy=x_proposed, noise_level=noise_level, covariance=covariance)
                    # model_output_proposed = scorenet.forward(input_proposed, create_graph=False)
                    # log_p_new = -model_output_proposed.energy  # log p(x_proposed)

                    # # Compute acceptance probability
                    # # For symmetric proposal (Langevin), we only need p(x_new) / p(x_old)
                    # log_alpha = log_p_new - log_p_old  # Shape: (batch_size,)
                    # alpha = torch.exp(log_alpha).clamp(max=1.0)  # min(1, p_new/p_old)
                    
                    # # Accept or reject for each sample in batch
                    # accept = torch.rand(batch_size, device=device) > alpha
                    # accept = accept.view(-1, 1, 1, 1)  # Reshape for broadcasting
                    
                    # # Update: accept proposal or keep current state
                    # x_mod = torch.where(accept, x_proposed, x_mod)    

                if not final_only:
                    images.append(x_mod.to('cpu'))
                                
                if iter_print_img % 200 == 0:
                    samples = torch.clamp(x_mod, 0.0, 1.0)
                    nrow = 4
                    grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=2, normalize=False)
                    grid_np = grid.permute(1, 2, 0).cpu().numpy()
                     
                    # Display the grid
                    plt.figure(figsize=(7, 7)) # Adjust figure size as needed
                    plt.imshow(grid_np)
                    plt.axis('off') # Hide axes
                    plt.title("Generated images - Energy")
                    plt.savefig(f'samples/sampling_intermediate_iter_{iter_print_img}.pdf')
                    print(f"Saved intermediate sampling image at iteration {iter_print_img}")
                iter_print_img = iter_print_img + 1


            if n_steps_each == 0:
                if iter_print_img % 500 == 0:
                    samples = torch.clamp(x_mod, 0.0, 1.0)
                    nrow = 4
                    grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=2, normalize=False)
                    grid_np = grid.permute(1, 2, 0).cpu().numpy()
                     
                    # Display the grid
                    plt.figure(figsize=(7, 7)) # Adjust figure size as needed
                    plt.imshow(grid_np)
                    plt.axis('off') # Hide axes
                    plt.title("Generated images - Energy")
                    plt.savefig(f'samples/sampling_intermediate_iter_{iter_print_img}.pdf')
                    print(f"Saved intermediate sampling image at iteration {iter_print_img}")
                iter_print_img = iter_print_img + 1

        if final_only:
            return [x_mod.to('cpu')]
        else:
            return images

# def leapfrog_integrator(x, v, grad_U, M_inv, epsilon, L):
#     """
#     Leapfrog integrator for Hamiltonian dynamics
    
#     Args:
#         x: Current position (state)
#         v: Current momentum
#         grad_U: Gradient of potential energy (negative score)
#         M_inv: Inverse mass matrix (covariance)
#         epsilon: Step size
#         L: Number of leapfrog steps
    
#     Returns:
#         x_new: New position
#         v_new: New momentum
#     """
#     # Make a half step for momentum
#     v_half = v - 0.5 * epsilon * grad_U
    
#     # Alternate full steps for position and momentum
#     x_new = x.clone()
#     v_new = v_half.clone()
    
#     for i in range(L):
#         # Full step for position
#         x_new = x_new + epsilon * M_inv.apply_power(v_new, p=1)
        
#         # Compute new gradient at new position
#         # This needs to be done by calling the score network
#         # For now, we'll update this in the main loop
#         if i < L - 1:
#             # Full step for momentum (except at end of trajectory)
#             # Note: grad_U needs to be recomputed here with x_new
#             # This will be handled in the main corrector loop
#             pass
    
#     # Make a final half step for momentum
#     # Note: This uses the gradient at the final position x_new
#     # which will be computed in the main loop
    
#     return x_new, v_new

#  # Helper function to compute potential energy gradient (negative score)
# def compute_grad_U(x, domain, sigma_next, kernel_size, kernel_std, device, batch_size,device = device, missing_indices = None):
#     if domain == "freq":
#         covariance = [deblurring_covariance_from_shape(
#             spatial_size=H, kernel_size=kernel_size, kernel_std=kernel_std, 
#             device=device, noise_level=sigma_next**2)] * batch_size
#     else:
#         covariance = [spatial_corr_covariance_testing(
#             spatial_size=H, box_size=box_size, var_box=sigma_next**2, 
#             device=device, var_clean=var_clean, inp_mask_type=inp_mask_type, 
#             half_box_size=box_size, missing_indices_input=missing_indices)] * batch_size
    
#     noise_level = NoiseLevel(variance=(sigma_next**2))
#     input = ModelInput(noisy=x, noise_level=noise_level, covariance=covariance)
#     model_output = scorenet.forward(input, create_graph=False)
    
#     # Gradient of potential U = -score (since score = -grad log p)
#     grad_U = -model_output.data_score
#     return grad_U, covariance

# # Helper function to compute probability density
# def compute_log_p(x, covariance):
#     """Compute log probability p(x) using model_output.energy"""
#     noise_level = NoiseLevel(variance=(sigma_next**2))
#     input = ModelInput(noisy=x, noise_level=noise_level, covariance=covariance)
#     model_output = scorenet.forward(input, create_graph=False)
    
#     # Use the energy from model output as log probability
#     return model_output.energy


# def PHMC_sampler(x_mod, scorenet, sigmas, 
#                 n_steps_each=200, final_only=False,
#                 inp_mask_type="half", box_size = 12, temp = 1, 
#                 snr = 0.1, domain='pixel',
#                 kernel_size = 8, kernel_std = 0.8,
#                 device = 'cpu'):
    
#     images = []
#     var_clean = 1e-9**2
#     H = x_mod.shape[-1]
#     batch_size = x_mod.shape[0]
#     covariance_id = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=1, device=device, var_clean = 0, inp_mask_type=inp_mask_type, half_box_size = box_size)]
#     # covariance_id = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=0, device=device, var_clean = 1, inp_mask_type=inp_mask_type, half_box_size = box_size)]
#     iter_print_img = 0

#     with torch.no_grad():
#         for c, sigma in enumerate(sigmas[:-2]):

#             # Predictor step
#             sigma_curr = sigmas[c]
#             sigma_next = sigmas[c+1]
            
#             if domain == "freq":
#                 # covariance = [deblurring_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, kernel_std=kernel_std, device=device, noise_level=sigma_curr**2)] * batch_size
#                 covariance = [sr_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, device=device, noise_level=sigma_curr**2)] * batch_size
#             else:
#                 covariance = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=sigma_curr**2, device=device, var_clean = var_clean, inp_mask_type=inp_mask_type, half_box_size = box_size)] * batch_size
#                 # covariance = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=var_clean, device=device, var_clean = sigma_curr**2, inp_mask_type=inp_mask_type, half_box_size = box_size)] * batch_size
#             noise_level = NoiseLevel(variance=sigma_curr**2)
#             input = ModelInput(noisy=x_mod, noise_level=noise_level, covariance=covariance)
#             model_output = scorenet.forward(input, create_graph = False)
#             grad = -model_output.data_score

#             diff = sigma_curr**2 - sigma_next**2
#             if domain == "freq":
#                 # covariance_step = [deblurring_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, kernel_std=kernel_std, device=device, noise_level=diff)] * batch_size
#                 # covariance_step_noise = [deblurring_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, kernel_std=kernel_std, device=device, noise_level=diff)] * batch_size
#                 covariance_step = [sr_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, device=device, noise_level=diff)] * batch_size
#                 covariance_step_noise = [sr_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, device=device, noise_level=diff)] * batch_size
#             else:
#                 covariance_step = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=diff, device=device, var_clean = var_clean, inp_mask_type=inp_mask_type, half_box_size = box_size)] * batch_size
#                 covariance_step_noise = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=(diff * sigma_next ** 2) / sigma_curr ** 2, device=device, var_clean = var_clean, inp_mask_type=inp_mask_type, half_box_size = box_size)] * batch_size
#                 # covariance_step = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=var_clean, device=device, var_clean = diff, inp_mask_type=inp_mask_type, half_box_size = box_size)] * batch_size
#                 # covariance_step_noise = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=var_clean, device=device, var_clean = (diff * sigma_next ** 2) / sigma_curr ** 2, inp_mask_type=inp_mask_type, half_box_size = box_size)] * batch_size
#             noise = torch.randn_like(x_mod)
#             x_mod = x_mod + covariance_step[0].apply_power(grad, p=1) + covariance_step[0].apply_power(noise, p=0.5)

#             # Corrector steps
#             for i in range(n_steps_each):
#                 # Sample momentum from N(0, M)
#                 v = torch.randn_like(x_mod)
#                 # TODO: Define correctly the mass matrix
#                 v = M[0].apply_power(v, p=0.5)  # v ~ N(0, M)
                
#                 # Store initial state
#                 x_old = x_mod.clone()
#                 v_old = v.clone()
                
#                 # Compute initial gradient
#                 grad_U_old, covariance_old = compute_grad_U(x_old)
                
#                 # Leapfrog integration
#                 # Half step for momentum
#                 v_half = v_old - 0.5 * epsilon * grad_U_old
                
#                 # L leapfrog steps
#                 x_new = x_old.clone()
#                 v_new = v_half.clone()
                
#                 for step in range(L):
#                     # Full step for position
#                     x_new = x_new + epsilon * M[0].apply_power(v_new, p=1)
                    
#                     # Compute gradient at new position
#                     grad_U_new, covariance_new = compute_grad_U(x_new)
                    
#                     # Full step for momentum (except at end)
#                     if step < L - 1:
#                         v_new = v_new - epsilon * grad_U_new
                
#                 # Final half step for momentum
#                 v_new = v_new - 0.5 * epsilon * grad_U_new
                
#                 # Negate momentum for reversibility (optional, makes trajectory symmetric)
#                 v_new = -v_new
                
#                 # Compute acceptance probability
#                 # H(x,v) = U(x) + 0.5*v^T*M^{-1}*v (Hamiltonian = potential + kinetic energy)
#                 # We want exp(-H_new) / exp(-H_old) = exp(H_old - H_new)
                
#                 # Kinetic energy terms: K = 0.5 * v^T * M^{-1} * v
#                 # Since v ~ N(0, M), we have M^{-1/2}*v ~ N(0, I)
#                 v_old_normalized = M[0].apply_power(v_old, p=-0.5)
#                 v_new_normalized = M[0].apply_power(v_new, p=-0.5)
                
#                 K_old = 0.5 * torch.sum(v_old_normalized**2, dim=[1, 2, 3])
#                 K_new = 0.5 * torch.sum(v_new_normalized**2, dim=[1, 2, 3])
                
#                 # Potential energy (using negative log probability)
#                 # U(x) = -log p(x) = 0.5 * ||score||^2 (approximately)
#                 U_old = -compute_log_p(x_old, covariance_old)
#                 U_new = -compute_log_p(x_new, covariance_new)
                
#                 # Acceptance probability: min(1, exp(-(H_new - H_old)))
#                 H_old = U_old + K_old
#                 H_new = U_new + K_new
                
#                 # Compute acceptance probability for each sample in batch
#                 log_alpha = -(H_new - H_old)
#                 alpha = torch.exp(log_alpha).clamp(max=1.0)
                
#                 # Accept or reject for each sample in batch
#                 accept = torch.rand(batch_size, device=device) < alpha
#                 accept = accept.view(-1, 1, 1, 1)  # Reshape for broadcasting
                
#                 # Update x_mod based on acceptance
#                 x_mod = torch.where(accept, x_new, x_old)
#                 if not final_only:
#                     images.append(x_mod.to('cpu'))
                                
#                 if iter_print_img % 200 == 0:
#                     samples = torch.clamp(x_mod, 0.0, 1.0)
#                     nrow = 4
#                     grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=2, normalize=False)
#                     grid_np = grid.permute(1, 2, 0).cpu().numpy()
                     
#                     # Display the grid
#                     plt.figure(figsize=(7, 7)) # Adjust figure size as needed
#                     plt.imshow(grid_np)
#                     plt.axis('off') # Hide axes
#                     plt.title("Generated images - Energy")
#                     plt.savefig(f'samples/sampling_intermediate_iter_{iter_print_img}.pdf')
#                     print(f"Saved intermediate sampling image at iteration {iter_print_img}")
#                 iter_print_img = iter_print_img + 1


#             if n_steps_each == 0:
#                 if iter_print_img % 500 == 0:
#                     samples = torch.clamp(x_mod, 0.0, 1.0)
#                     nrow = 4
#                     grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=2, normalize=False)
#                     grid_np = grid.permute(1, 2, 0).cpu().numpy()
                     
#                     # Display the grid
#                     plt.figure(figsize=(7, 7)) # Adjust figure size as needed
#                     plt.imshow(grid_np)
#                     plt.axis('off') # Hide axes
#                     plt.title("Generated images - Energy")
#                     plt.savefig(f'samples/sampling_intermediate_iter_{iter_print_img}.pdf')
#                     print(f"Saved intermediate sampling image at iteration {iter_print_img}")
#                 iter_print_img = iter_print_img + 1

#         if final_only:
#             return [x_mod.to('cpu')]
#         else:
#             return images

def adaptive_sampler(x_mod, scorenet, sigma_init, n_updates = 1000 ,n_steps_each=3, step_lr=1, step_cov_update = 1.5e-2, 
                     temp = 1, final_only=False, inp_mask_type="half", box_size = 12, device = 'cpu', missing_indices = None):


    images = []
    mse_list = []
    samples_to_save = []
    cov_to_save = []
    img_size = x_mod.shape[-1]
    batch_size = x_mod.shape[0]
    
    covariance = [spatial_corr_covariance_testing(spatial_size=img_size, box_size=box_size, var_box=sigma_init**2, device=device, var_clean = 1e-7, half_box_size = box_size, inp_mask_type=inp_mask_type)] * batch_size
    
    # covariance = [spatial_corr_covariance_testing(spatial_size=H, box_size=17, var_box=1e-7, device=device, var_clean = sigmas[0]**2, half_box_size = half_box_size, inp_mask_type=inp_mask_type)] * batch_size    
    cov_sample = covariance[0].get_matrix()
    cov_sample = cov_sample[None, None, :,:].repeat(batch_size, 1, 1, 1)
    # current_variance = sigmas[0]
    noise_level = NoiseLevel(variance=sigma_init**2)
    energy_values = torch.zeros(n_updates, batch_size)
    clamp_val = 5e1

    with torch.no_grad():
        for c_ in range(n_updates):
            step_size = step_lr
            input = ModelInput(noisy=x_mod, noise_level=noise_level, covariance=covariance)
            model_output = scorenet.forward(input, create_graph = False)
            # Calculate gradients
            grad = -model_output.data_score
            grad_cov = -model_output.noise_score
            grad_cov = grad_cov.sign() * torch.clamp(grad_cov, min=-clamp_val, max=clamp_val) # Clamp to a large but not infinite value
            # grad_cov = grad_cov.sign() * grad_cov / (grad_cov.norm() + 1e-8)
            energy_values[c_, :] = model_output.energy
            # energy_old = energy_values[c_, :]


            # 1. Check the network output immediately
            if torch.isnan(grad_cov).any() or torch.isinf(grad_cov).any():
                print(f"!!! Step {s}: grad_cov from scorenet is NaN or Inf!")
                print(grad_cov)
                # You might want to break or exit here to inspect
                import sys; sys.exit()

            delta_matrix = step_cov_update * (cov_sample * grad_cov * cov_sample)
            
            if torch.isnan(delta_matrix).any() or torch.isinf(delta_matrix).any():
                print(f"!!! Step {s}: delta_matrix exploded to NaN or Inf!")
                print("Norm of grad_cov:", torch.norm(grad_cov))
                print("Norm of cov_sample:", torch.norm(cov_sample))
                import sys; sys.exit()

            # Update with gradient descent
            cov_sample = cov_sample - delta_matrix
            cov_sample = torch.clamp(cov_sample, min=1e-9)
            
            # Create covariance object
            covariance_update = []
            covariance = []
            for bss in range(batch_size):
                covariance_update.append(SpatialCorrCovariance(matrix=delta_matrix[bss,0,:,:].to(device=device)))
                covariance.append(SpatialCorrCovariance(matrix=cov_sample[bss,0,:,:].to(device=device)))
                noise = torch.randn_like(x_mod[bss:bss+1])
                x_mod[bss:bss+1] = x_mod[bss:bss+1] + covariance_update[bss].apply_power(grad[bss:bss+1], p=1) + covariance_update[bss].apply_power(noise, p=0.5)

            for inner_step in range(n_steps_each):
                input = ModelInput(noisy=x_mod, noise_level=noise_level, covariance=covariance)
                model_output = scorenet.forward(input, create_graph = False)
                
                grad = -model_output.data_score
                grad_cov = model_output.noise_score

                # TODO: use the built in function apply power from batch.
                for bss in range(batch_size):  
                    noise = torch.randn_like(x_mod[bss:bss+1])
                    x_mod[bss:bss+1] = x_mod[bss:bss+1] + step_lr * covariance_update[bss].apply_power(grad[bss:bss+1], p=1) + np.sqrt(temp * 2 * step_lr) * covariance_update[bss].apply_power(noise, p=0.5)
        
            # === ENERGY CHECK: Accept/Reject ===
            # input_new = ModelInput(noisy=x_mod, noise_level=noise_level, covariance=covariance)
            # model_output_new = scorenet.forward(input_new, create_graph=False)
            # energy_new = model_output_new.energy
            
            # # Check if energy improved (or use Metropolis acceptance for sampling)
            # energy_increase = (torch.abs(energy_new) - torch.abs(energy_old).cuda())
            # print(energy_increase)
            # # print(energy_new, energy_values.shape)
            # # Option A: Hard rejection if energy increases too much
            # if energy_increase > 0.1 * torch.abs(energy_old.cuda()):  # 10% threshold
            #     # print(f"Step {c_}: Energy increased {energy_old:.4f} -> {energy_new:.4f}, REJECTING")
            #     print(f"Step: Energy increased {energy_old}, {energy_new}")
            #     x_mod = x_mod_prev
            #     cov_sample = cov_sample_prev
            #     # Reduce step size for next iteration
            #     step_lr = step_lr * 0.5
            #     gamma_cov = gamma_cov * 0.5
            #     continue
            
            # Update noise level based on new covariance
            current_variance = torch.trace(cov_sample[0,0,:,:]) / cov_sample.shape[0]
            noise_level = NoiseLevel(variance=current_variance)

            if not final_only:
                images.append(x_mod.to('cpu'))

            assert (cov_sample >= 0).all(), "Error: Covariance matrix is not PSD!"

            if cov_sample.max() < 5:
                step_cov_update = 5e-2
            
            if c_ % 100 == 0:                
                # nrow = 8
                # samples = torch.clamp(x_mod, 0.0, 1.0)
                # grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=2, normalize=False)
                # grid_np = grid.permute(1, 2, 0).cpu().numpy()

                # plt.figure(figsize=(7, 7))
                # plt.imshow(grid_np)
                # plt.axis('off')
                # plt.show()

                # grid = torchvision.utils.make_grid(cov_sample.float(), nrow=nrow, padding=2, normalize=True)
                # # print(grid.shape)
                # grid_np = grid.permute(1, 2, 0).cpu().numpy()
                # plt.figure(figsize=(7, 7))
                # plt.imshow(grid_np, cmap='plasma')
                # plt.axis('off')
                # plt.show()
                samples = torch.clamp(x_mod, 0.0, 1.0)
                nrow = 4
                grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=2, normalize=False)
                grid_np = grid.permute(1, 2, 0).cpu().numpy()
                    
                # Display the grid
                plt.figure(figsize=(7, 7)) # Adjust figure size as needed
                plt.imshow(grid_np)
                plt.axis('off') # Hide axes
                plt.title("Generated images - Energy")
                plt.savefig(f'samples/sampling_intermediate_iter_{c_}.pdf')
                print(f"Saved intermediate sampling image at iteration {c_}")

                samples_to_save.append(samples)
                cov_to_save.append(cov_sample)
                print(f"Iter:{c_} , Energy: {model_output.energy}")
                print(f"Variance:{cov_sample.max()}")

    if final_only:
        data_to_save = {
            'samples': samples_to_save,
            'covariances': cov_to_save
        }
        torch.save(data_to_save, "samples_cov_adapted_randombox.pt")
        print(f"Successfully saved data to samples_cov_adapted_randomhalf.pt")
        return [x_mod.to('cpu')]
    else:
        return images


def DPS_sampler(x_mod, y, img_size, mask, scorenet, sigmas, 
                batch_size=1, inp_mask_type="half",
                box_size=12, device = 'cpu'):
    
    iter_print_img = 0
    
    for c, sigma in enumerate(sigmas[:-2]):
        sigma_curr = sigmas[c]
        sigma_next = sigmas[c+1]
        
        x_mod = x_mod.detach().requires_grad_(True)
        
        # Uniform covariance
        covariance = [spatial_corr_covariance_testing(
            spatial_size=img_size, box_size=box_size, var_box=sigma_curr**2, 
            device=device, var_clean=sigma_curr**2, 
            inp_mask_type=inp_mask_type, half_box_size=box_size
        )] * batch_size
        
        noise_level = NoiseLevel(variance=sigma_curr**2)
        input = ModelInput(noisy=x_mod, noise_level=noise_level, covariance=covariance)
        
        model_output = scorenet.forward(input, create_graph=False)
        grad = -model_output.data_score
        x0_pred = model_output.denoised
        
        # Compute gradient efficiently
        residual = y - mask.apply_power(x0_pred, p=1)
        mat_norm = (residual.reshape(batch_size, -1) ** 2).sum(dim=1).sqrt()
        mat = (residual.reshape(batch_size, -1) ** 2).sum()
        
        # Compute gradient w.r.t. x_mod directly (backprop through network)
        grad_term = torch.autograd.grad(mat, x_mod, retain_graph=False)[0]
        
        grad = grad.detach()
        grad_term = grad_term.detach()
        mat_norm = mat_norm.detach()
        coeff = 0.5 # / mat_norm.reshape(-1, 1, 1, 1)
        
        with torch.no_grad():
            diff = sigma_curr**2 - sigma_next**2
            noise = torch.randn_like(x_mod)
            x_mod = x_mod + diff * grad + torch.sqrt(diff) * noise - coeff * grad_term

        if c % 10 == 0: 
            torch.cuda.empty_cache()
        
        # Visualization code (unchanged)
        if iter_print_img % 200 == 0:
            with torch.no_grad():
                samples = torch.clamp(x_mod, 0.0, 1.0)
                nrow = 5
                grid = torchvision.utils.make_grid(samples[0:20], nrow=nrow, 
                                                  padding=2, normalize=False)
                grid_np = grid.permute(1, 2, 0).cpu().numpy()
                plt.figure(figsize=(5, 5))
                plt.imshow(grid_np)
                plt.axis('off')
                plt.title("Generated images - DPS")
                plt.savefig(f'samples/sampling_intermediate_iter_{iter_print_img}.pdf')
        iter_print_img += 1
    
    return [x_mod.detach().to('cpu')]