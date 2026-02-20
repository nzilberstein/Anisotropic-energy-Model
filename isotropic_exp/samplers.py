import torchvision
from pathlib import Path 

from noise import *
from pathlib import Path 
from data import *
from sampling import get_sigmas
from trackers import *
from main import *
from networks.conditioning import *


def PC_sampler(x_mod, scorenet, sigmas, 
                n_steps_each=200, final_only=False,
                inp_mask_type="half", box_size = 12, temp = 1, 
                snr = 0.1, domain='pixel',
                kernel_size = 8, kernel_std = 0.8,
                device = 'cpu'):
    
    images = []
    var_clean = 1e-9
    H = x_mod.shape[-1]
    batch_size = x_mod.shape[0]
    covariance_id = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=1, device=device, var_clean = 0, inp_mask_type=inp_mask_type, half_box_size = box_size)]
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
                covariance = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=sigma_curr**2, device=device, var_clean = var_clean, inp_mask_type=inp_mask_type, half_box_size = box_size)] * batch_size
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
                covariance_step = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=diff, device=device, var_clean = var_clean, inp_mask_type=inp_mask_type, half_box_size = box_size)] * batch_size
                covariance_step_noise = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=(diff * sigma_next ** 2) / sigma_curr ** 2, device=device, var_clean = var_clean, inp_mask_type=inp_mask_type, half_box_size = box_size)] * batch_size
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
                    covariance = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=sigma_next**2, device=device, var_clean = var_clean, inp_mask_type=inp_mask_type, half_box_size = box_size)] * batch_size
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
                    # noise_norm = np.sqrt(box_size * box_size * 3)
                    noise_norm = torch.norm(covariance_id[0].apply_power(noise, p = 1).view(noise.shape[0], -1), dim=-1).mean()
                step_size = (snr * noise_norm / grad_norm) ** 2 * 2
                if domain == "freq":
                    # covariance_step = [deblurring_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, kernel_std=kernel_std, device=device, noise_level=step_size)] * batch_size
                    covariance_step = [sr_covariance_from_shape(spatial_size=H, kernel_size=kernel_size, device=device, noise_level=step_size)] * batch_size
                else:
                    covariance_step = [spatial_corr_covariance_testing(spatial_size=H, box_size=box_size, var_box=step_size, device=device, var_clean = var_clean, inp_mask_type=inp_mask_type, half_box_size = box_size)] * batch_size
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


def adaptive_sampler(x_mod, scorenet, sigma_init, n_updates = 1000 ,n_steps_each=3, step_lr=1, step_cov_update = 5e-2, 
                     temp = 1, final_only=False, np_mask_type="half", box_size = 12, device = 'cpu'):


    images = []
    mse_list = []
    samples_to_save = []
    cov_to_save = []
    img_size = x_mod.shape[-1]
    batch_size = x_mod.shape[0]
    
    init_cov = [spatial_corr_covariance_testing(spatial_size=img_size, box_size=box_size, var_box=sigma_init**2, device=device, var_clean = 1e-7, half_box_size = box_size, inp_mask_type=np_mask_type)] * batch_size
    
    # covariance = [spatial_corr_covariance_testing(spatial_size=H, box_size=17, var_box=1e-7, device=device, var_clean = sigmas[0]**2, half_box_size = half_box_size, inp_mask_type=inp_mask_type)] * batch_size    
    cov_sample = init_cov[0].get_matrix()
    cov_sample = cov_sample[None, None, :,:].repeat(batch_size, 1, 1, 1)
    # current_variance = sigmas[0]
    noise_level = NoiseLevel(variance=sigma_init**2)
    energy_values = torch.zeros(n_updates, batch_size)
    clamp_val = 10e1

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
           
            for inner_step in range(n_steps_each):
                input = ModelInput(noisy=x_mod, noise_level=noise_level, covariance=covariance)
                model_output = scorenet.forward(input, create_graph = False)
                
                grad = -model_output.data_score
                grad_cov = model_output.noise_score

                # TODO: use the built in function apply power from batch.
                for bss in range(batch_size):  
                    noise = torch.randn_like(x_mod[bss:bss+1])
                    x_mod[bss:bss+1] = x_mod[bss:bss+1] + step_lr * covariance_update[bss].apply_power(grad[bss:bss+1], p=1) + np.sqrt(temp * step_lr) * covariance_update[bss].apply_power(noise, p=0.5)
        
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
            
            if c_ % 100 == 0:                
                nrow = 8
                samples = torch.clamp(x_mod, 0.0, 1.0)
                grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=2, normalize=False)
                grid_np = grid.permute(1, 2, 0).cpu().numpy()

                plt.figure(figsize=(7, 7))
                plt.imshow(grid_np)
                plt.axis('off')
                plt.show()

                grid = torchvision.utils.make_grid(cov_sample.float(), nrow=nrow, padding=2, normalize=True)
                # print(grid.shape)
                grid_np = grid.permute(1, 2, 0).cpu().numpy()
                plt.figure(figsize=(7, 7))
                plt.imshow(grid_np, cmap='plasma')
                plt.axis('off')
                plt.show()

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
        return [x_mod.to('cpu')], mse_list, energy_values
    else:
        return images


def DPS_sampler(x_mod, y, img_size, mask, scorenet, sigmas,
                inp_mask_type="half",
                box_size=12, device = 'cpu',
                domain = 'pixel', missing_indices=None):
    
    iter_print_img = 0
    batch_size = x_mod.shape[0]
    for c, sigma in enumerate(sigmas[:-2]):
        sigma_curr = sigmas[c]
        sigma_next = sigmas[c+1]
        
        x_mod = x_mod.detach().requires_grad_(True)
        
        # Uniform covariance
        covariance = [spatial_corr_covariance_testing(
            spatial_size=img_size, box_size=box_size, var_box=sigma_curr**2, 
            device=device, var_clean=sigma_curr**2, 
            inp_mask_type=inp_mask_type, half_box_size=box_size, missing_indices_input=missing_indices
        )] * batch_size
        
        noise_level = NoiseLevel(variance=sigma_curr**2)
        input = ModelInput(noisy=x_mod, noise_level=noise_level, covariance=covariance)
        
        model_output = scorenet.forward(input, create_graph=False)
        grad = -model_output.data_score
        x0_pred = model_output.denoised
        
        # Compute gradient efficiently
        if domain == "freq":
            residual = y - mask.apply_power(x0_pred, p=-1)
        else:
            residual = y - mask.apply_power(x0_pred, p=1)

        mat_norm = (residual.reshape(batch_size, -1) ** 2).sum(dim=1).sqrt()
        mat = (residual.reshape(batch_size, -1) ** 2).sum()
        
        # Compute gradient w.r.t. x_mod directly (backprop through network)
        grad_term = torch.autograd.grad(mat, x_mod, retain_graph=False)[0]
        
        grad = grad.detach()
        grad_term = grad_term.detach()
        mat_norm = mat_norm.detach()
        if domain == "freq":
            coeff = 0.01
        else:
            coeff = 0.05 #0.05 #0.5 # / mat_norm.reshape(-1, 1, 1, 1)
        
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
                print(f"Saved intermediate sampling image at iteration {iter_print_img}")
        iter_print_img += 1
    
    return [x_mod.detach().to('cpu')]


def reddiff_sampler(x_mod, y, img_size, mask, scorenet, sigmas, 
                batch_size=1, inp_mask_type="half",
                box_size=12, device = 'cpu',
                domain = 'pixel',
                missing_indices=None):
    
    iter_print_img = 0

    mu = x_mod.requires_grad_(True)

    #optimizer
    dtype = torch.FloatTensor
    optimizer = torch.optim.Adam([mu], lr=0.1, betas=(0.9, 0.99), weight_decay=0.0)   #original: 0.999
    
    for c, sigma in enumerate(sigmas[:-2]):
        sigma_curr = sigma

        # Build covariance
        covariance = [spatial_corr_covariance_testing(
            spatial_size=img_size, box_size=box_size, var_box=sigma_curr**2, 
            device=device, var_clean=sigma_curr**2, 
            inp_mask_type=inp_mask_type, half_box_size=box_size, missing_indices_input=missing_indices
        )] * batch_size

        # Compute x_t
        noise_xt = torch.randn_like(mu).to(device)
        x_t = mu + sigma_curr * noise_xt

        # Evaluate the score
        noise_level = NoiseLevel(variance=sigma_curr**2)
        input = ModelInput(noisy=x_t, noise_level=noise_level, covariance=covariance)
        model_output = scorenet.forward(input, create_graph=False)

        
        et = -(-model_output.data_score * sigma_curr)
        # Compute gradient efficiently
        if domain == "freq":
            residual = y - mask.apply_power(mu, p=-1)
        else:
            residual = y - mask.apply_power(mu, p=1)
        # residual = y - mask.apply_power(mu, p=1)
        loss_obs = (residual**2).mean()/2
        loss_noise = torch.mul((et - noise_xt).detach(), mu).mean()

        snr_inv = sigma_curr/1  #1d torch tensor
    
        w_t = 2 * snr_inv   #0.25 #2
        v_t = 1

        loss = w_t*loss_noise + v_t*loss_obs
        
        #adam step
        optimizer.zero_grad()  #initialize
        loss.backward()
        optimizer.step()
                
  
        
        # Visualization code (unchanged)
        if iter_print_img % 200 == 0:
            with torch.no_grad():
                samples = torch.clamp(mu, 0.0, 1.0)
                nrow = 5
                grid = torchvision.utils.make_grid(samples[0:20], nrow=nrow, 
                                                  padding=2, normalize=False)
                grid_np = grid.permute(1, 2, 0).cpu().numpy()
                plt.figure(figsize=(5, 5))
                plt.imshow(grid_np)
                plt.axis('off')
                plt.title("Generated images - DPS")
                plt.savefig(f'samples/sampling_intermediate_iter_{iter_print_img}.pdf')
                print(f"Saved intermediate sampling image at iteration {iter_print_img}")
        iter_print_img += 1
    
    return [mu.detach().to('cpu')]



def ODE_flow(x_mod, H, scorenet, sigmas, c, device, batch_size=1,
               inp_mask_type="half", box_size=12):

    for c_ode, sigma in enumerate(sigmas[:-1]):
        sigma_curr = sigmas[c_ode]
        sigma_next = sigmas[c_ode + 1]
        
        x_mod = x_mod.detach()
        
        covariance = [spatial_corr_covariance_testing(
            spatial_size=H, box_size=box_size, var_box=sigma_curr**2, 
            device=device, var_clean=sigma_curr**2, 
            inp_mask_type=inp_mask_type, half_box_size=box_size
        )] * batch_size


        noise_level = NoiseLevel(variance=sigma_curr**2)
        input = ModelInput(noisy=x_mod, noise_level=noise_level, covariance=covariance)
        
        model_output = scorenet.forward(input, create_graph=False)
        grad = -model_output.data_score
        
        with torch.no_grad():
            # noise = torch.randn_like(x_mod)
            # diff = sigma_curr**2 - sigma_next**2
            # x_mod = x_mod + 0.5 * diff * grad #+ torch.sqrt(diff) * noise 
            x0_hat = x_mod + sigma_curr**2 * grad
            d = (x_mod - x0_hat) / sigma_curr
            x_mod = x_mod + (sigma_next - sigma_curr) * d
        
    return x_mod


def DAPS_sampler(x_mod, y, H, mask, scorenet, sigmas, num_steps_langevin=100, batch_size=1, std_noise = 1e-3,
               inp_mask_type="half", box_size=12, lr_langevin=2e-6, device = 'cpu', domain = 'pixel', missing_indices=None):
    
    for c, sigma in enumerate(sigmas[:-1]):
        
        # Run the ODE to get x_0
        sigmas_ODE = get_sigmas(sigma.detach().cpu().numpy(), sigmas[-1].detach().cpu().numpy(), 5, device, sigma_dist = 'geometric')
        x_pred = ODE_flow(x_mod, H, scorenet, sigmas_ODE, c, device, batch_size, inp_mask_type, box_size)

        # -- Run Langevin sampling
        x_cond_pred = x_pred.clone().detach().requires_grad_(True)
        # Parameters
        r_t = (2 * sigmas[c] ** 2)
        p = 1
        ratio = c/len(sigmas)
        coeff = (1  + ratio * (0.01 * (1/p) - 1)) ** p * lr_langevin

        for _ in range(num_steps_langevin):
            
            # Measurement residual
            if domain == "freq":
                residual = mask.apply_power(x_cond_pred, p=-1) - y
            else:
                residual = mask.apply_power(x_cond_pred, p=1) - y
            residual_y_norm = (residual.reshape(batch_size, -1) ** 2).sum() / (2 * std_noise ** 2) #1e-3 is the noise level of the measurement

            # Prior residual
            residual_prior = (x_cond_pred - x_pred.detach()) 
            residual_prior_norm = (residual_prior.reshape(batch_size, -1) ** 2).sum() / r_t

            # Aggregate
            loss = residual_y_norm + residual_prior_norm
            grad = torch.autograd.grad(loss, x_cond_pred, retain_graph=False)[0]
            
            # Update
            noise = torch.randn_like(x_pred)
            x_cond_pred = x_cond_pred - coeff * (grad) + np.sqrt(2 * coeff) * noise

        # Sample a noisy version of the next noise level
        x_mod = x_cond_pred + sigmas[c+1] * torch.randn_like(x_cond_pred)
        
        # Visualization code (unchanged)
        if c % 10 == 0:
            torch.cuda.empty_cache()
            with torch.no_grad():
                samples = torch.clamp(x_mod, 0.0, 1.0)
                nrow = 5
                grid = torchvision.utils.make_grid(samples[0:20], nrow=nrow, 
                                                  padding=2, normalize=False)
                grid_np = grid.permute(1, 2, 0).cpu().numpy()
                plt.figure(figsize=(5, 5))
                plt.imshow(grid_np)
                plt.axis('off')
                plt.title("Generated images - DAPS")
                plt.savefig(f'samples/sampling_intermediate_iter_{c}.pdf')
                print(f"Saved intermediate sampling image at iteration {c}")
                
    return [x_cond_pred.detach().to('cpu')]
