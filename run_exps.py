""" This file generates the commands to train the models used in the paper. """

from typing import *
import shlex
import subprocess
import argparse
import numpy as np
from printing import format_sig_digits
from utils import si_to_num


parser = argparse.ArgumentParser()
parser.add_argument("--print", action="store_true", help="only print the commands instead of running them")
parser.add_argument("-p", action="store_true", help="only print the names instead of the full commands")
args = parser.parse_args()


class Exps:
    def __init__(self, exps: List[Tuple[List[str], List[str]]]):
        self.exps = exps  # List of exps, each exp being represented as tuple(names, cmds) which represent name and cmd parts to join.

    def __or__(self, other: Union["Exps", "SubExps"]) -> "Exps":
        return Exps(self.exps + to_exps(other).exps)

    def __xor__(self, other: Union["Exps", "SubExps"]) -> "Exps":
        """ Weak product (do product with first experiment in the list). """
        other = to_exps(other)
        return (self * Exps(other.exps[:1])) | (Exps(self.exps[:1]) * Exps(other.exps[1:]))

    def __mul__(self, other: Union["Exps", "SubExps"]) -> "Exps":
        return Exps([(name1 + name2, cmd1 + cmd2) for name1, cmd1 in self.exps for name2, cmd2 in to_exps(other).exps])

    def __getitem__(self, item) -> "Exps":
        " For selecting a subset of experiments. item should be a slice rather than an int. "
        if isinstance(item, int):
            if item < 0:
                item = len(self.exps) + item
            item = slice(item, item + 1)
        return Exps(self.exps[item])

    def run(self):
        for i, (names, cmds) in enumerate(self.exps, start=1):
            name = "_".join(filter(None, names))
            cmd = " ".join(filter(None, cmds))
            print(f"[{i:{len(str(len(self.exps)))}}/{len(self.exps)}] {name}:")
            if not args.p:
                print(f"\t{cmd}\t")
                if not args.print:
                    cmd = f"{cmd} --name {name}"
                    print(subprocess.run(shlex.split(cmd), check=True, capture_output=True).stdout.decode("utf-8").strip())
                print("")  # Newline

    def anonymize(self) -> "Exps":
        """ Return self, without empty names. """
        return Exps([([], cmds) for _, cmds in self.exps])


def exp(name="", cmd=""):
    return Exps([([name], [cmd])])


class literal:
    """ Convenience class when generating kwarg dicts. Typically used so that str(kwargs) will correctly handle class names. """
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return self.value


class SubExps:
    def __init__(self, subs: List[Tuple[List[str], str, Dict[str, Any]]]):
        self.subs = subs  # List of sub-exps, each sub-exp being represented as tuple(names, header, kwarg dict)

    def __or__(self, other: "SubExps") -> "SubExps":
        return SubExps(self.subs + other.subs)

    def __xor__(self, other: "SubExps") -> "SubExps":
        """ Weak product (do product with first experiment in the list). """
        return (self * SubExps(other.subs[:1])) | (SubExps(self.subs[:1]) * SubExps(other.subs[1:]))

    def __mul__(self, other: "SubExps") -> "SubExps":
        # We always use the header of self, which should be OK.
        return SubExps([(name1 + name2, hdr1, dict1 | dict2) for name1, hdr1, dict1 in self.subs for name2, hdr2, dict2 in other.subs])  # This allows dict2 to overwrite dict1.

    def to_exps(self) -> Exps:
        return Exps([(names, [f"{header} \"{str(kwargs).replace(' ', '')}\""]) for names, header, kwargs in self.subs])


def sub(name="", header="", **kwargs):
    return SubExps([([name], header, kwargs)])


def to_exps(exps: Exps | SubExps):
    if isinstance(exps, SubExps):
        return exps.to_exps()
    else:
        return exps


empty_exps = Exps([])  # An empty list of experiments
single_empty_exp = exp()  # A single exp with no arguments nor name
opt = lambda e, without="": exp(name=without) | e  # Simple A/B testing for an experiment

def ors(exps):
    # Complicated implementation in case exps is a list of SubExps.
    exps = list(exps)
    if len(exps) == 0:
        return empty_exps
    res = exps[0]
    for exp in exps[1:]:
        res = res | exp
    return res



# Arguments sets

base = exp(cmd="python main.py")  # Replace by SLURM command, etc... (all experiments were run on a single H100 GPU).

# Dataset
dataset = lambda name, path=None: exp(name=name.lower(), cmd=f"--dataset {path if path is not None else name}")  # default dataset BSD, default grayscale
color = exp(name="color", cmd="--no-grayscale")
imagenet64 = dataset("ImageNet64")
subset = lambda n, flip=False: exp(name=f"subN{n}{'B' if flip else 'A'}", cmd=f"--data-subset \"slice({-n if flip else None}, {None if flip else n})\"")

# Models
denoiser = exp(name="denoiser", cmd=f"--model DenoiserModel")
energy = exp(name="energy", cmd=f"--model EnergyModel")
reparam = sub(header="--reparam-kwargs")

# Architectures
unet = exp(name="unet", cmd="--network UNet")

# Network arguments
net_kwargs = sub(header="--network-kwargs")

# Objectives
mse_exponent = lambda alpha: exp(name=f"mseexp{alpha}", cmd=f"--mse-var-exponent {alpha}")
noise_score_exponent = lambda alpha: exp(name=f"nsexp{alpha}", cmd=f"--noise-score-var-exponent {alpha}")
noise_score = lambda mult: exp(name=f"noisescore{format_sig_digits(mult)}", cmd=f"--train-noise-score {mult}")

# Noise level sampling
noise_range = lambda min, max, unit="psnr": exp(name=f"noiserange{min}to{max}{unit}", cmd=f"--min-noise-level {unit}={min} --max-noise-level {unit}={max}")
log_sampler = exp(name="logsampler", cmd="--noise-level-sampler UniformLog")

# Training args
lr = lambda lr: exp(name=f"lr{lr}", cmd=f"--lr {lr}")
batch_size = lambda train=256, test=64: exp(name=f"bs{train}-{test}", cmd=f"--train-batch-size {train} --test-batch-size {test}")
num_steps = lambda train_steps, num_decays=10: exp(name=f"{train_steps}steps{num_decays}decays", cmd=f"--num-training-steps {si_to_num(train_steps):_} --lr-decay-every {si_to_num(train_steps) // num_decays:_}")

# Commonly used combinations
subsetsAB = ors(subset(n, flip) for n in [10, 100, 1_000, 10_000, 100_000] for flip in [False, True])
deep_unet = unet * (net_kwargs * sub(num_scales=3) * sub(group_size=1) * sub(name="deep", num_layers_encoder_block=3, num_layers_mid_block=3, num_layers_decoder_block=3))
default_denoiser = denoiser * (reparam * sub(name="psco", output_scaling=1, residual=True))
default_energy = energy * (reparam * sub(name=f"psco_ip", conversion="inner_product"))


# Experiments

exps = empty_exps

# Energy model on ImageNet64 (both grayscale and color)
exps |= exp(name="finalclean") * base * imagenet64 * opt(color, without="grayscale") * (batch_size(128, 32) * deep_unet * default_energy * noise_range(90, -30) * log_sampler * mse_exponent(-1) * noise_score(1) * noise_score_exponent(0)).anonymize() * lr(0.0005) * num_steps("1M")
# Denoiser on ImageNet64 color
exps |= exp(name="finalclean") * base * imagenet64 * color * (batch_size(128, 32) * deep_unet).anonymize() * default_denoiser * (noise_range(90, -30) * log_sampler * mse_exponent(-1)).anonymize() * lr(0.0005) * num_steps("1M")
# Convergence experiments on ImageNet64 color
exps |= exp(name="finalcvg") * base * imagenet64 * color * subsetsAB * (batch_size(128, 32) * deep_unet * default_energy * noise_range(90, -30) * log_sampler * mse_exponent(-1) * noise_score(1) * noise_score_exponent(0)).anonymize() * lr(0.0002) * num_steps("1M")

exps.run()
