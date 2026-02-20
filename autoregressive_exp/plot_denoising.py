from noise import *
from pathlib import Path 
import json
from data import *
from trackers import *
from main import *

from networks.conditioning import *

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

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
    
    ctx = TrainingContext(**args_dict, step=step, key_remap=None, seed=None, dataloaders=dataloaders, writer=False)
    # if log:
    #     print(f"{name}: retrieved model at step {ctx.step} and test loss {ctx.test_perf.loss:.2e}")

    # Disable DataParallel (needed for Hessian computation)
    # ctx.model.network = ctx.model.network.module

    # Put in eval mode and disable gradients with respect to all parameters.
    ctx.model.eval()
    for p in ctx.model.parameters():
        p.requires_grad = False

    # Normalize energies.
    # ctx.network.network.log_normalization_constant = ctx.test_perf.log_normalization_constant

    return ctx


def split_function_tensor_clamp(input_tensor, threshold):
    return torch.clamp_min(input_tensor, threshold)

# args = load_args("test_mult_anisotropic_newloss", step = "best")
step = "last"
ctxs = {
    # "denoiser-anisotropic-newloss": load_exp("test_mult_anisotropic_newloss", step = "last"),
    # "denoiser-anisotropic-oldloss": load_exp("test_mult_anisotropic_oldloss", step = "last"),
    # "denoiser-anisotropic-newloss-tinbox": load_exp("multigpu/test_denoiser_mult_anisotropic_newloss_newTweedie_tinbox_song_groupNorm_multigpu_warmup", step = "last"),
    # "denoiser-anisotropic-newloss-tinbox": load_exp("multigpu/combined_cov_tworandvars/energy_songSmall_anisoEmb_groupNorm_lambda1_mult_lr1.5e-4_lrdecay50000_1000warmup_d2", step = step),
    "denoiser-anisotropic-newloss-tinbox": load_exp("multigpu/deblurring_sr/energy_songSmall_score_anisoEmb_groupNorm_lambda0_lr2e-4_lrdecay50000_1000warmup_d2_maxnoise10", step = step),
}

denoiser_eval = True
size_network = "large"
variance = 1e-3

default_ctx = ctxs["denoiser-anisotropic-newloss-tinbox"]
# device = default_ctx.device
# dataset_info = default_ctx.dataset_info
# d = dataset_info.dimension

# Load data
if denoiser_eval is True:
    data_newloss = torch.load(f"plots/denoising_error_energyvsdenoiser_song_{step}.pt")
else:
    data_newloss = torch.load(f"plots/denoising_error_regvsnonreg_song_{step}.pt")

t = 10 ** (-data_newloss['psnrs'] / 10).detach().cpu()
t_id = torch.zeros(t.shape[0])

for ii in range(t.shape[0]):
    # covariance = deblurring_covariance_from_shape(spatial_size=32, kernel_size=8, kernel_std=0.8, device="cpu", noise_level=t[ii])
    covariance = sr_covariance_from_shape(spatial_size=32, kernel_size=4, device="cpu", noise_level=t[ii])
    t_id[ii] = torch.sum(covariance.get_matrix()) / (32**2)
    print(t[ii], t_id[ii])

# denoising_error_anisotropic_newloss = data_newloss['denoising_error_all_scales'].detach().cpu()
# energy_denoising_error_anisotropic_newloss = data_newloss['energy_denoising_error_all_scales'].detach().cpu()
if denoiser_eval is True:
    # denoiser = data_newloss['denoiser_denoising'].detach().cpu()
    energy_reg = data_newloss['energyDual_denoiser_denoising'].detach().cpu()
    energy_nonreg = data_newloss['energy_denoiser_denoising'].detach().cpu()
    # print(t, denoising_error_anisotropic_newloss)
    # print(t, energy_denoising_error_anisotropic_newloss)
else:
    energy_reg = data_newloss['energy_denoising_reg'].detach().cpu()
    # energy_nonreg = data_newloss['energy_denoising_non_reg'].detach().cpu()
    denoiser = data_newloss['denoiser_denoising'].detach().cpu()
    # print(t, denoising_error_anisotropic_newloss)
    # print(t, energy_denoising_error_anisotropic_newloss)

# print(t, data_newloss['psnrs'])
plt.figure(figsize=(4, 3), layout="constrained")
ax = plt.gca()
if denoiser_eval is True:
    # plt.plot(t, denoiser, color="tab:red", marker=".", label="Denoiser")
    plt.plot(t, energy_reg, color="tab:cyan", marker=".", label="Energy-Dual") 
    plt.plot(t, energy_nonreg, color="tab:blue", marker=".", label="Energy-Single") 
    plt.plot(t, t_id, color="black", label="Identity - $Sum(\Sigma_t)/d$", zorder=-1)
else:
    plt.plot(t, energy_reg, color="tab:red", marker=".", label="Energy-Dual")
    # plt.plot(t, energy_nonreg, color="tab:blue", marker=".", label="Energy-Single") 
    plt.plot(t, denoiser, color="tab:green", marker=".", label="Denoiser")
    plt.plot(t, split_function_tensor_clamp(t, energy_reg[-1]), color="black", label="Identity", zorder=-1)
# plt.plot(t, denoising_error_anisotropic_oldloss, color="tab:red", linestyle="dashed", marker=".", label="Old loss - Old Tweedie")
# plt.plot(t, denoising_error_anisotropic_oldloss_newTweedie, color="tab:orange", linestyle="dashed", marker=".", label="Old loss - New Tweedie")
# plt.plot(t, denoising_error_isotropic_newloss, color="tab:green", marker=".", label="Isotropic - New loss")
plt.plot(t, torch.full_like(t, default_ctx.dataset_info.variance), color="gray", label="Data variance", zorder=-1)
plt.xlabel("Noise variance $\sigma_2$")
plt.xscale("log")
plt.xlim(1e-9, 1e1)
# plt.xticks(10. ** np.arange(-9, 4), fontsize=8)
plt.ylabel("Denoising MSE")
plt.yscale("log")
# plt.ylim(5e-10, 2e0)
# plt.title(r"Box inpainting ($\sigma_1=10^{-3}$)")
plt.title("Downsampling x 4")
plt.legend()
if denoiser_eval is True:
    plt.savefig(f"plots/MMSE_denoiservsenergy_song{size_network}_{step}_box_{variance}_half.pdf", transparent=True, bbox_inches="tight", pad_inches=0)
else:
    plt.savefig(f"plots/MMSE_regvsnonreg_song{size_network}_{step}.pdf", transparent=True, bbox_inches="tight", pad_inches=0)