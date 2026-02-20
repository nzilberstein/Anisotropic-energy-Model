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

    args_dict['num_workers'] = 8
    return args_dict


def load_exp(name, step="last", log=True, dataloaders=False):
    """ Load an experiment with a given name. step can be an integer, "best", or "last" (default). """
    exp_dir = Path("models") / name

    with open(exp_dir / "args.json") as f:
        args_dict = json.load(f)

    args_dict['size_network'] = "large"
    
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
    "Energy-Dual": load_exp("multigpu/all_together/energy_song_dual_truncatedFreq_celeba", step = step),
}

denoiser_eval = True
size_network = "large"
variance = 1e-3

default_ctx = ctxs["Energy-Dual"]
# device = default_ctx.device
# dataset_info = default_ctx.dataset_info
# d = dataset_info.dimension

# Load data
if denoiser_eval is True:
    data_newloss = torch.load(f"plots/denoising_error_energyvsdenoiser_song_{step}.pt")
    # data_posterior = torch.load(f"plots/denoising_error_posteriorsampling_energyvsdenoiser_song_{step}.pt")
else:
    data_newloss = torch.load(f"plots/denoising_error_regvsnonreg_song_{step}.pt")

t = 10 ** (-data_newloss['psnrs'] / 10).detach().cpu()
# denoiser = data_newloss['denoiser_denoising'].detach().cpu()
energy_reg = data_newloss['energyDual_denoiser_denoising'].detach().cpu()
energy_nonreg = data_newloss['energy_denoiser_denoising'].detach().cpu()
# energy_reg_posterior = data_posterior['energyDual_denoiser_denoising'].detach().cpu()
# print(t, energy_r)
print(energy_reg.shape, energy_nonreg.shape)
print(t, data_newloss['psnrs'])
plt.figure(figsize=(4, 3), layout="constrained")
ax = plt.gca()
# plt.plot(t, denoiser, color="tab:red", marker=".", label="Denoiser")
plt.plot(t, energy_reg, color="tab:cyan", marker=".", label="Energy-Dual") 
plt.plot(t, energy_nonreg, color="tab:blue", marker=".", label="Energy-Single") 
# plt.plot(t[0:15], energy_reg_posterior[0:15], color="tab:orange", marker=".", label="Energy-Dual (Posterior Sampling)")
# plt.plot(t, t, color="black", label="Identity", zorder=-1)
plt.plot(t, torch.full_like(t, default_ctx.dataset_info.variance), color="gray", label="Data variance", zorder=-1)
plt.xlabel("Noise variance $\sigma_1^2$")
plt.xscale("log")
plt.xlim(1e-3, 1e1)
# plt.xticks(10. ** np.arange(-9, 4), fontsize=8)
plt.ylabel("Denoising MSE")
plt.yscale("log")
# plt.ylim(5e-10, 2e0)
# plt.title(r"Box inpainting ($\sigma_1=10^{-3}$)")
# plt.title("Box impainting - Celeba")
plt.legend()
plt.savefig(f"plots/MMSE_DualvsSingle_networkSize{size_network}_{step}_boxType_{data_newloss['mask_type']}.pdf", transparent=True, bbox_inches="tight", pad_inches=0)
