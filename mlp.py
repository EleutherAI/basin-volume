# %%
from flax import linen as nn
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from dataclasses import dataclass
from typing import Callable

# %%


def param_normal(fan_in: int, norm_scale: float = 1.0):
    """Kernel/bias initializer with variance 1/fan_in, untruncated"""
    return nn.initializers.normal(stddev=norm_scale * (fan_in) ** -0.5)


@dataclass
class Raveler:
    raveled: jnp.ndarray
    unravel: Callable

    def __init__(self, params, unravel=None):
        if isinstance(params, dict):
            self.raveled, self.unravel = ravel_pytree(params)
        else:
            assert isinstance(params, jnp.ndarray), "params must be a JAX array or a dict"
            self.raveled = params
            assert unravel is not None, "unravel must be provided if params are raveled"
            self.unravel = unravel
    
    @property
    def unraveled(self):
        return self.unravel(self.raveled)
    
    @property
    def norm(self):
        return jnp.linalg.norm(self.raveled)
    

# %%
class MLP(nn.Module):
    hidden_sizes: tuple[int, ...]
    out_features: int
    norm_scale: float

    @nn.compact
    def __call__(self, x):
        fan_in = x.shape[-1]

        for i, feat in enumerate(self.hidden_sizes):
            x = nn.Dense(
                    feat, 
                    bias_init=param_normal(fan_in, self.norm_scale), 
                    kernel_init=param_normal(fan_in, self.norm_scale)
                )(x)
            x = self.perturb(f'a_{i}', x)
            x = nn.gelu(x)
            x = self.perturb(f'h_{i}', x)

            fan_in = feat

        x = nn.Dense(
                self.out_features, 
                bias_init=param_normal(fan_in, self.norm_scale), 
                kernel_init=param_normal(fan_in, self.norm_scale)
            )(x)
        x = self.perturb(f'a_L', x)
        return x
    
# def ellipsoid_norm(params: Params, spherical: bool = False):
#     bias_coef = 1 if spherical else 3
#     params = params.unraveled
#     out = 0
#     for layer in params['params']:
#         ker = params['params'][layer]['kernel']
#         bias = params['params'][layer]['bias']
#         out += jnp.sum(ker**2) + bias_coef * jnp.sum(bias**2)
#     return jnp.sqrt(out)

# def typicalize(params: Params, norm_scale: float):
#     pu = params.unraveled
#     out_params = {}
#     for layer in pu['params']:
#         ker = pu['params'][layer]['kernel']
#         bias = pu['params'][layer]['bias']
#         ker /= jnp.sqrt(ker.shape[0]) * jnp.std(ker) / norm_scale
#         bias /= jnp.sqrt(3 * ker.shape[0]) * jnp.std(bias) / norm_scale
#         out_params[layer] = {'kernel': ker, 'bias': bias}
#     return Params({'params': out_params})


# %%
