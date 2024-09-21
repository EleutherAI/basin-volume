# %%
from flax import linen as nn
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def kernel_normal(fan_in: int, norm_scale: float = 1.0):
    """Kernel initializer with variance 1/fan_in, untruncated"""
    return nn.initializers.normal(stddev=norm_scale * (fan_in) ** -0.5)

def bias_normal(fan_in: int, norm_scale: float = 1.0):
    """Initializer for bias with same variance as PyTorch's init"""
    return nn.initializers.normal(stddev=norm_scale * (3 * fan_in) ** -0.5)


# %%
class MLP(nn.Module):
    hidden_sizes: tuple[int, ...]
    out_features: int
    norm_scale: float

    @nn.compact
    def __call__(self, x):
        fan_in = x.shape[-1]

        for feat in self.hidden_sizes:
            x = nn.Dense(feat, bias_init=bias_normal(fan_in, self.norm_scale), kernel_init=kernel_normal(fan_in, self.norm_scale))(x)
            x = nn.gelu(x)

            fan_in = feat

        x = nn.Dense(self.out_features, bias_init=bias_normal(fan_in, self.norm_scale), kernel_init=kernel_normal(fan_in, self.norm_scale))(x)
        return x
    

def typicalize(params, norm_scale):
    out_params = {}
    for layer in params['params']:
        ker = params['params'][layer]['kernel']
        bias = params['params'][layer]['bias']
        ker /= jnp.sqrt(ker.shape[0]) * jnp.std(ker) / norm_scale
        bias /= jnp.sqrt(3 * ker.shape[0]) * jnp.std(bias) / norm_scale
        out_params[layer] = {'kernel': ker, 'bias': bias}
    return {'params': out_params}


