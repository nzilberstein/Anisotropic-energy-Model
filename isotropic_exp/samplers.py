import torchvision
from pathlib import Path 

from noise import *
from pathlib import Path 
from data import *
from sampling import get_sigmas
from trackers import *
from main import *
from networks.conditioning import *

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
            residual = y - mask.apply_power(x0_pred, p=-0.5)
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
            residual = y - mask.apply_power(mu, p=-0.5)
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
                residual = mask.apply_power(x_cond_pred, p=-0.5) - y
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
