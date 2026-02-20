from pathlib import Path 
import json
import torch
import numpy as np


def visualize_dual_grids(ald_samples, pc_samples, delta_matrix=None, iteration=0, gamma_cov=None, nrow=5):
    """
    Visualize grids of images from both samplers, similar to the original implementations.
    
    Parameters:
    -----------
    ald_samples : torch.Tensor
        Samples from Annealed Langevin Dynamics
    pc_samples : torch.Tensor
        Samples from PC Sampler
    delta_matrix : torch.Tensor, optional
        Covariance delta for ALD visualization
    iteration : int
        Current iteration number
    gamma_cov : float, optional
        Gamma coefficient for covariance scaling
    nrow : int
        Number of images per row in grid
    """
    import matplotlib.pyplot as plt
    import torchvision
    
    # Create ALD samples grid
    ald_grid = torchvision.utils.make_grid(ald_samples, nrow=nrow, padding=2, normalize=False)
    ald_grid_np = ald_grid.permute(1, 2, 0).cpu().numpy()
    
    # Create PC samples grid
    pc_grid = torchvision.utils.make_grid(pc_samples, nrow=nrow, padding=2, normalize=False)
    pc_grid_np = pc_grid.permute(1, 2, 0).cpu().numpy()
    
    # Determine number of subplots
    num_plots = 3 if delta_matrix is not None else 2
    
    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 7))
    
    # Plot ALD samples
    axes[0].imshow(ald_grid_np)
    axes[0].set_title(f'ALD Samples - Iteration {iteration}')
    axes[0].axis('off')
    
    # Plot PC samples
    axes[1].imshow(pc_grid_np)
    axes[1].set_title(f'PC Samples - Iteration {iteration}')
    axes[1].axis('off')
    
    # Plot delta matrix if available
    if delta_matrix is not None and gamma_cov is not None:
        delta_grid = torchvision.utils.make_grid(
            torch.log(torch.abs(delta_matrix.float()) / gamma_cov), 
            nrow=nrow, padding=2, normalize=True
        )
        # delta_grid = torchvision.utils.make_grid(
        #    delta_matrix.float() / gamma_cov, 
        #     nrow=nrow, padding=2, normalize=True
        # )
        delta_grid_np = delta_grid.permute(1, 2, 0).cpu().numpy()
        axes[2].imshow(delta_grid_np[:,:,0], cmap='plasma')
        axes[2].set_title(f'ALD Delta Covariance (log scale)')
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"samples/dual_sampler_iteration_{iteration}.png")
    # plt.show()


def visualize_final_comparison(results_ald, results_pc, x_init, x_clean, idx_img, batch_size, nrow=5):
    """
    Create final comparison visualization similar to the original implementation.
    
    Parameters:
    -----------
    results_ald : tuple
        (samples, mse_list, energy_values) from ALD
    results_pc : tuple
        (samples, mse_list) from PC
    x_init : torch.Tensor
        Initial noisy images
    x_clean : torch.Tensor
        Clean reference images
    idx_img : int
        Starting index for images
    batch_size : int
        Number of images to display
    nrow : int
        Number of images per row
    """
    import matplotlib.pyplot as plt
    import torchvision
    
    ald_samples, ald_mse, ald_energy = results_ald
    pc_samples, pc_mse = results_pc
    
    # Clamp samples
    ald_final = torch.clamp(ald_samples[0], 0.0, 1.0)
    pc_final = torch.clamp(pc_samples[0], 0.0, 1.0)
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # Initial noisy image
    axes[0, 0].imshow(x_init[0].permute(1, 2, 0).cpu().numpy())
    axes[0, 0].set_title('Initial Noisy Image')
    axes[0, 0].axis('off')
    
    # Clean images
    clean_grid = torchvision.utils.make_grid(
        x_clean[idx_img:idx_img+batch_size], 
        nrow=nrow, padding=2, normalize=False
    )
    clean_grid_np = clean_grid.permute(1, 2, 0).cpu().numpy()
    axes[0, 1].imshow(clean_grid_np)
    axes[0, 1].set_title('Clean Images')
    axes[0, 1].axis('off')
    
    # ALD final samples
    ald_grid = torchvision.utils.make_grid(ald_final, nrow=nrow, padding=2, normalize=False)
    ald_grid_np = ald_grid.permute(1, 2, 0).cpu().numpy()
    axes[1, 0].imshow(ald_grid_np)
    
    # Calculate and display ALD MSE
    if len(ald_mse) > 0:
        ald_mse_final = ald_mse[-1].mean().item()
        axes[1, 0].set_title(f'ALD Final - MSE: {ald_mse_final:.6f}')
    else:
        axes[1, 0].set_title('ALD Final Samples')
    axes[1, 0].axis('off')
    
    # PC final samples
    pc_grid = torchvision.utils.make_grid(pc_final, nrow=nrow, padding=2, normalize=False)
    pc_grid_np = pc_grid.permute(1, 2, 0).cpu().numpy()
    axes[1, 1].imshow(pc_grid_np)
    
    # Calculate and display PC MSE
    if len(pc_mse) > 0:
        pc_mse_final = pc_mse[-1].mean().item()
        axes[1, 1].set_title(f'PC Final - MSE: {pc_mse_final:.6f}')
    else:
        axes[1, 1].set_title('PC Final Samples')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig("samples/final_comparison.png")
    
    # Print statistics
    print("\n" + "="*50)
    print("FINAL RESULTS COMPARISON")
    print("="*50)
    if len(ald_mse) > 0:
        print(f"ALD - Final MSE: {ald_mse[-1].mean():.6f}")
        print(f"ALD - MSE [dB]: {-10 * torch.log10(ald_mse[-1].mean()):.2f}")
    if len(pc_mse) > 0:
        print(f"PC  - Final MSE: {pc_mse[-1].mean():.6f}")
        print(f"PC  - MSE [dB]: {-10 * torch.log10(pc_mse[-1].mean()):.2f}")
    print("="*50)


def visualize_single_samples(samples, title="Samples", figsize=(2, 2)):
    """
    Visualize a single sample image.
    
    Parameters:
    -----------
    samples : torch.Tensor
        Image tensor to visualize
    title : str
        Title for the plot
    figsize : tuple
        Figure size
    """
    import matplotlib.pyplot as plt
    
    samples_clamped = torch.clamp(samples, 0.0, 1.0)
    plt.figure(figsize=figsize)
    plt.imshow(samples_clamped[0].cpu().numpy().transpose(1, 2, 0))
    plt.title(title)
    plt.axis('off')
    plt.show()
