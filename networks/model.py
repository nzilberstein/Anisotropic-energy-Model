""" Model interface: abstractions over architectures for denoising, energy prediction, etc, in a common interface. """

from typing import *

import numpy as np
import torch
from torch import nn
from functorch import vmap, jacrev, jacfwd
from einops import *

from data import DatasetInfo
import noise
from noise import NoiseLevel, NoiseLevelSampler, Covariance, IdentityCovariance, NoisySampler, WhiteGaussianSampler, Batch, noisy_batch, get_tensor_from_list_covariances, apply_power_to_list_covariances
from grad import compute_grad, compute_jacobian

import pdb

class LogTensor:
    """ Class used to represent logarithmic quantities (typically energies or log-probabilities). Takes care of unit conversions. """
    def __init__(self, total_energy_nats, d: int):
        self.total_energy_nats = total_energy_nats
        self.d = d

    def to(self, base: Literal["nats", "bits", "dBs"], sign: Literal["energy", "logp"] = "energy", per_dimension=True):
        """
        Args:
            base: base of logarithm (with potential multiplicative constant). Must be one of "nats", "bits", or "dBs".
            sign: "energy" or "logp".
            per_dimension: whether to divide the quantity by the underlying dimension or not.
        """
        value = self.total_energy_nats
        if base == "nats":
            pass
        elif base == "bits":
            value = value / np.log(2)
        elif base == "dBs":
            value = 10 * value / np.log(10)
        else:
            raise ValueError(f"Unknown base: {base}")
        if sign == "energy":
            pass
        elif sign == "logp":
            value = -value
        else:
            raise ValueError(f"Unknown sign: {sign}")
        if per_dimension:
            value = value / self.d
        return value

    @property
    def bpd(self):
        return self.to("bits", sign="energy", per_dimension=True)

    @property
    def dBpd(self):
        return self.to("dBs", sign="energy", per_dimension=True)


class ModelInput:
    """ Named-tuple-like object containing a batch of model input data (noisy, noise-level info, etc).
    This class provides a stable API unifying past and future models which might be conditioned on different data. """
    def __init__(self, noisy: torch.Tensor, noise_level: NoiseLevel = None, covariance: List[Covariance] = None, type_cov: List[str] = None) -> None:
        self.noisy: torch.Tensor = noisy  # (B..., D...)
        self.noise_level: NoiseLevel = noise_level # (B...,)
        self.noise_covariance: List[Covariance] = covariance if covariance is not None else IdentityCovariance()  # (B..., D..., D...)
        self.type_cov: List[str] = type_cov  # List of strings, length B... (only for MultipleColoredGaussianSampler)

    def __getitem__(self, idx):
        return ModelInput(noisy=self.noisy[idx], noise_level=self.noise_level[idx])

    @property
    def batch_shape(self) -> Tuple[int]:
        return self.noise_level.shape  # (B...)

    @property
    def signal_shape(self) -> Tuple[int]:
        return self.noisy.shape[len(self.batch_shape):]  # (D...,)

    @property
    def full_shape(self) -> Tuple[int]:
        return self.noisy.shape  # (B..., D...)

    @staticmethod
    def from_batch(batch: Batch) -> "ModelInput":
        """ Creates a ModelInput from a batch, taking care of broadcasting. """
        noisy = batch.noisy  # (B..., T..., N..., D...)
        noise_level = batch.noise_level  # (B/1..., T...)
        # print(f"{noisy.shape=} {noise_level.shape=} {noise.s.signal_ndim=}")
        noise_level = NoiseLevel(noise_level.dataset_info, variance=noise_level.variance[(...,) + (None,) * (noisy.ndim - noise_level.ndim - noise.s.signal_ndim)].expand(noise.s.batch_shape(noisy.shape)))  # (B..., T..., N...)
        # print(f"{noise_level.shape=}")
        return ModelInput(noisy=noisy, noise_level=noise_level)  # (B..., T..., N..., D...)


def model_input(clean: torch.Tensor, noise_level: NoiseLevelSampler | torch.Tensor | List[float] | float = 0, noisy_sampler: NoisySampler = WhiteGaussianSampler(), num_noises: torch.Size | int = ()) -> ModelInput:
    """ Returns a ModelInput based on the given clean data and optional noise level and noisy samplers.
    Args:
        clean: (B..., D...) clean data
        noise_level: sampler for the noise levels, or fixed noise level variances (T...) (defaults to zero)
        noisy_sampler: sampler for the noisy data (defaults to additive white Gaussian noise)
        num_noises: shape or number of noise samples per clean datum (N...) (defaults to empty shape)
    Returns:
        ModelInput: composed noisy data and noise levels, of shape (B..., T..., N..., D...) (no broadcasting)
    """
    return ModelInput.from_batch(noisy_batch(clean, noise_level, noisy_sampler, num_noises))


class ModelOutput:
    """ Named-tuple-like object containing a batch of model output data (score, denoised data, etc).
    This class provides a stable API unifying past and future models which might predict different quantities. """
    def __init__(self, energy: torch.Tensor = None, data_score: torch.Tensor = None, data_hessian: torch.Tensor = None,
                 noise_score: torch.Tensor = None, denoised: torch.Tensor = None) -> None:
        self.energy: torch.Tensor = energy  # (B...,)
        self.data_score: torch.Tensor = data_score  # (B..., D...)
        self.data_hessian: torch.Tensor = data_hessian  # (B..., D..., D...)
        self.noise_score: torch.Tensor = noise_score  # (B...,) for scalar noise level
        self.denoised: torch.Tensor = denoised  # (B..., D...)

    def stack(outputs: List["ModelOutput"], dim: int) -> "ModelOutput":
        """ Stacks every component, taking care of potential None's. """
        stack = lambda xs: None if xs[0] is None else torch.stack(xs, dim=dim)
        return ModelOutput(**{key: stack([output.__dict__[key] for output in outputs])
                            for key in ModelOutput().__dict__})

    def __getitem__(self, idx):
        """ Indexes the whole output (only reliably works for batch indices). """
        return ModelOutput(energy=self.energy[idx], data_score=self.data_score[idx], data_hessian=self.data_hessian[idx],
                           noise_score=self.noise_score[idx], denoised=self.denoised[idx])


class Model(nn.Module):
    """ Abstract wrapper over underlying network which uses the above interface. Takes care of DataParallel and multi-dimensional batching. """
    def __init__(self, network: nn.Module, dataset_info: DatasetInfo):
        super().__init__()

        if torch.cuda.is_available():
            network = nn.DataParallel(network).cuda()
        self.network: nn.Module = network  # Expose the internal network, can always be useful.
        self.dataset_info: DatasetInfo = dataset_info

    def network_forward(self, type_cov_list, *xs):
        """ Performs a network forward, taking care of multiple batch dimensions. By convention xs[0] is the noisy data. """
        batch_shape = noise.s.batch_shape(xs[0].shape)  # (B...,)
        #
        # xs = tuple(x.reshape((-1, *x.shape[len(batch_shape):])) for x in xs)  # (B, ...)
        #
        cov = xs[-1]
        xs = tuple(x.reshape((-1, *x.shape[len(batch_shape):])) for x in xs[:-1])  # (B, ...)
        xs = xs + (cov,)  # (B, ..., D) or (B, ..., D, D) for colored covariance
        #
        ys = self.network(*xs, type_cov_list)  # (B, ...)

        if isinstance(ys, tuple):
            ys = tuple(y.reshape(batch_shape + noise.s.signal_shape(y.shape)) for y in ys)  # (B..., D...)
        else:
            ys = ys.reshape((*batch_shape, *ys.shape[1:]))  # (B..., D...)
        return ys

    def forward(self, x: ModelInput) -> ModelOutput:
        """ Forward with the correct interface. Subclasses are supposed to call network_forward. """
        raise NotImplementedError

    def my_named_parameters(self, reduced=True, with_grad=True, prefix="") -> Dict[str, torch.Tensor]:
        """ More convenient version of nn.Module.named_parameters. Overridden by some modules to provide more helpful names.
        Possiblity to return a reduced list (for more concise logging) or filtering parameters that have gradient only.
        """
        return self.network.my_named_parameters(reduced=reduced, with_grad=with_grad, prefix=prefix)


class DenoiserModel(Model):
    """ Model which uses a (blind or not) denoiser as underlying network. """
    def __init__(self, network: nn.Module, dataset_info: DatasetInfo):
        super().__init__(network, dataset_info)
        self.anisotropic_network = True
        self.new_tweedie = True

    def forward(self, x: ModelInput, create_graph=None, compute_hessian=False, full_batch=False, backward=True, symmetrize=True) -> ModelOutput:
        """ create_graph argument is here just for convenience. """
        t = x.noise_level.variance  # (B...)
        if len(t.shape) == 2:
            if self.anisotropic_network is True:
                t_input = t.view(-1)
                t_input = t_input[:, None, None] * x.noise_covariance.get_matrix(shape = x.noisy.shape[-1], device = x.noisy.device)[None, :, :]
                t_input = t_input.unsqueeze(1)  # (B..., D, D) or (B..., D) for identity covariance
                output = self.network_forward(x.noisy, t_input)
                if self.new_tweedie is True:
                    # If use new tweedie, we have residual true and return the denoiser output (which is scaled by the covariance)
                    denoised = output
                    data_score = x.noise_covariance.apply_inv(x.noisy - denoised) / t[..., None, None, None]  # (B..., D...)
                else:
                    # Else, we return the score and apply just the t scalar
                    denoised = x.noisy - output * t[..., None, None, None]
                    data_score = output
            else:
                ## If we use isotropic, then t is just a scalar and we apply the correct tweedie outside the reparametrization.
                t_input = t.view(-1)
                output = self.network_forward(x.noisy, t_input)
                denoised = x.noisy - x.noise_covariance.apply(output) * t[..., None, None, None]
                data_score = output
        else:
            if self.anisotropic_network is True:
                ## If use anisotropic, we use the covariance directly
                t_input = get_tensor_from_list_covariances(x.noise_covariance)
                t_input = t_input.unsqueeze(1)
                output = self.network_forward(x.noisy, t_input)
                if self.new_tweedie is True:
                    # If use new tweedie, we have residual true and return the denoiser output (which is scaled by the covariance)
                    if self.network.module.residual is True:
                        denoised = output
                        data_score = apply_power_to_list_covariances(x.noise_covariance, x.noisy - denoised, p=-1)  # (B..., D...)
                    else:
                        # print("Using the score as output")
                        data_score = output
                        denoised = x.noisy - apply_power_to_list_covariances(x.noise_covariance, data_score, p=1)  # (B..., D...)
                else:
                    # Else, we return the score and apply just the t scalar
                    denoised = x.noisy - output * t[..., None, None, None]
                    # denoised = x.noisy - apply_power_to_list_covariances(x.noise_covariance, output, p=1)
                    data_score = output
            else:
                ## If we use isotropic, then t is just a scalar and we apply the correct tweedie outside the reparametrization.
                t_input = t
                output = self.network_forward(x.noisy, t_input)
                denoised = x.noisy - apply_power_to_list_covariances(x.noise_covariance, output, p=1)
                data_score = output

        if compute_hessian:
            if isinstance(self.noise_covariance, IdentityCovariance):
                raise NotImplementedError  # Formula below need to be modified to handle non-identity covariance.

            jacobian = compute_jacobian(self.network_forward, x.noisy, t, full_batch=full_batch,
                                        backward=backward, symmetrize=symmetrize).matrix  # (B..., D, D)
            hessian = torch.reshape((torch.eye(jacobian.shape[-1], device=jacobian.device) - jacobian) / t[..., None, None],
                                    t.shape + noise.s.signal_shape(x.noisy.shape) * 2)
        else:
            hessian = None

        return ModelOutput(denoised=denoised, data_score=data_score, data_hessian=hessian)


class EnergyModel(Model):
    """ Model which uses an energy network as underlying network. """
    def __init__(self, network: nn.Module, dataset_info: DatasetInfo):
        super().__init__(network, dataset_info)
        self.log_normalization_constant = None  # Here for convenience in notebooks (automatically returns normalized energies if set).
        self.anisotropic_network = True
        # self.new_tweedie = True

    def forward(self, x: ModelInput, create_graph=True, compute_scores=True, compute_hessian=False, full_batch=False, backward1=True, backward2=False, symmetrize=None) -> ModelOutput:
        """ Perform a network forward, compute score, and optionally Hessian.
        :param create_graph: whether to create a graph to allow differentiating through the scores.
        :param compute_scores: whether to compute the gradients of the energy with respect to the input.
        :param compute_hessian: whether to compute the Hessian of the energy with respect to the input.
        :param full_batch: if True, computes the Hessian for the full batch at once (faster but requires more memory).
        :param symmetrize: ignored argument, here for convenience.
        """
        if compute_hessian:
            create_graph = False
        noisy = x.noisy  # (B..., D...)
        t = x.noise_level.variance  # (B...,)

        ## Get the type of covariance and covert to a list of tensors to pass to the network
        t_input, type_cov_input = get_tensor_from_list_covariances(x.noise_covariance)
        type_map = {"spatial": 0, "frequency": 1}
        # Convert the list of strings to a list of numbers
        type_cov_numbers = [type_map[t] for t in type_cov_input]
        # Convert the list of numbers to a tensor on the correct device
        type_cov_tensor = torch.tensor(type_cov_numbers, dtype=torch.long, device=noisy.device)

        t_input = t_input.unsqueeze(1)
        batch_shape = t_input.shape[0]  # (B...,)
        input_tensors = (noisy, t_input)
        if compute_scores:
            ones = torch.ones(batch_shape, device=noisy.device)  # (B...)
            energy, scores = compute_grad(self.network_forward, input_tensors, type_cov_list=type_cov_tensor, grad_output=ones, create_graph=create_graph)  # (B...), (B..., D...), [(B...,)]
            data_score, noise_score = scores  # (B..., D...), (B..., D...)
            denoised = x.noisy - apply_power_to_list_covariances(x.noise_covariance, data_score, p=1)  # (B..., D...)
        else:
            energy = self.network_forward(*input_tensors)
            data_score = None
            noise_score = None
            denoised = None
        if compute_hessian:
            compute_hessian = (jacrev if backward2 else jacfwd)((jacrev if backward1 else jacfwd)(self.network_forward))  # (D...,) to (D..., D...)

            if full_batch:
                batch = vmap
            else:
                batch = lambda f: lambda *ys: torch.stack([f(*(y[i] for y in ys)) for i in range(len(ys[0]))])
            for _ in range(len(batch_shape)):
                compute_hessian = batch(compute_hessian)
            hessian = compute_hessian(*input_tensors)  # (B..., D..., D...)
        else:
            hessian = None
        if self.log_normalization_constant is not None:
            energy = energy + self.log_normalization_constant

        return ModelOutput(energy=energy, data_score=data_score, data_hessian=hessian, noise_score=noise_score, denoised=denoised)


class Reparameterization(nn.Module):
    """ Reparameterization of a vector field to play with noise scaling, skip-connections, and energy conversion. """
    def __init__(self, network: nn.Module, dataset_info: DatasetInfo, input_scaling=None, output_scaling=None,
                 residual=False, conversion=None, final_scaling=None, additive_normalization=False, edm_like=False):
        """ Wrapper around a network to reparameterize its input and output as a function of t.
        Args:
            input_scaling: scale the input to the network by t^input_scaling.
            output_scaling: scale the output of the network by t^output_scaling.
            residual: whether to add a global skip-connection to the network.
            conversion: optional conversion from a vector-valued network to a scalar-valued network (inner_product or squared_norm).
            final_scaling: scale the global output (including potential skip-connection) by t^final_scaling.
            additive_normalization: whether to add the normalizing constant of the Gaussian or not at the end.
            edm_like: whether to use a hardcoded EDM-like scaling (overwrites everything else).
        """
        super().__init__()
        self.network = network
        self.dataset_info = dataset_info

        if conversion not in [None, "inner_product", "squared_norm"]:
            raise ValueError(f"Unknown conversion: {conversion}")

        self.input_scaling = input_scaling
        self.output_scaling = output_scaling
        self.residual = residual
        self.conversion = conversion
        self.final_scaling = final_scaling
        self.additive_normalization = additive_normalization
        self.edm_like = edm_like

    def extra_repr(self) -> str:
        if self.edm_like:
            output = "<y, t*y + f(y/√1+t)> / 2(1+t)² + d/2 log(1 + t)"
        else:
            def t_str(s, alpha, has_denom=False):
                slash = "" if has_denom else "/"
                return {None: s, -1: f"{s}{slash}t", -1/2: f"{s}{slash}√t", 0: s, 1/2: f"√t*{s}", 1: f"t*{s}"}[alpha]

            input = t_str("y", self.input_scaling)
            output = t_str(f"f({input})", self.output_scaling)
            if self.residual:
                output = f"y - {output}"
            if self.conversion == "inner_product":
                output = f"<y, {output}> / 2"
            elif self.conversion == "squared_norm":
                output = f"||{output}||^2 / 2"
            output = t_str(output, self.final_scaling, has_denom=self.conversion is not None)
            if self.additive_normalization:
                output = f"{output} + d/2 log t"

        return f"expr=\"{output}\""


    def forward(self, x: torch.Tensor, t: torch.Tensor, type_cov_list: List[str]) -> torch.Tensor:
        if self.edm_like:
            # This reparameterization ensures that the energy has the right asymptotic behavior at t -> 0 and t -> ∞, for both the energy and the denoiser.
            # This uses mean and variance of the dataset to ensure that the (white) Gaussian behavior is respected.
            x = x - self.dataset_info.mean
            t_unsqeeze = t[(...,) + (None,) * (x.ndim - t.ndim)]
            input = x / torch.sqrt(self.dataset_info.variance + t_unsqeeze)
            output = self.network(input, t)
            skip = t_unsqeeze * x + output
            scalar = torch.sum(x * skip, dim=noise.s.signal_dims) / (2 * (1 + t) ** 2)
            constant = self.dataset_info.dimension * torch.log(1 + t) / 2
            return scalar + constant
        scale = lambda z, alpha: z if alpha is None else z * t[(...,) + (None,) * (z.ndim - t.ndim)] ** alpha

        input = scale(x, self.input_scaling)
        output = self.network(input, t, type_cov_list)
        output = scale(output, self.output_scaling)
        output = x - output if self.residual else output
        if self.conversion == "inner_product":
            output = torch.sum(x * output, dim=noise.s.signal_dims)  / 2
        elif self.conversion == "squared_norm":
            output = torch.sum(output ** 2, dim=noise.s.signal_dims) / 2

        output = scale(output, self.final_scaling)

        if self.additive_normalization:
            output = output + self.dataset_info.dimension * torch.log(t) / 2

        return output

    def my_named_parameters(self, reduced=True, with_grad=True, prefix="") -> Dict[str, torch.Tensor]:
        """ More convenient version of nn.Module.named_parameters. Overridden by some modules to provide more helpful names.
        Possiblity to return a reduced list (for more concise logging) or filtering parameters that have gradient only.
        """
        return self.network.my_named_parameters(reduced=reduced, with_grad=with_grad, prefix=prefix)
