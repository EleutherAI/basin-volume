# %%
from flax import linen as nn
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def kernel_normal(fan_in: int):
    """Kernel initializer with variance 1/fan_in, untruncated"""
    return nn.initializers.normal(stddev=2 *(fan_in) ** -0.5)

def bias_normal(fan_in: int):
    """Initializer for bias with same variance as PyTorch's init"""
    return nn.initializers.normal(stddev=2 * (3 * fan_in) ** -0.5)


# %%
class MLP(nn.Module):
    hidden_sizes: tuple[int, ...]
    out_features: int

    @nn.compact
    def __call__(self, x):
        fan_in = x.shape[-1]

        for feat in self.hidden_sizes:
            x = nn.Dense(feat, bias_init=bias_normal(fan_in), kernel_init=kernel_normal(fan_in))(x)
            x = nn.gelu(x)

            fan_in = feat

        x = nn.Dense(self.out_features, bias_init=bias_normal(fan_in), kernel_init=kernel_normal(fan_in))(x)
        return x
    

def typicalize(params):
    out_params = {}
    for layer in params['params']:
        ker = params['params'][layer]['kernel']
        bias = params['params'][layer]['bias']
        ker /= jnp.sqrt(ker.shape[0]) * jnp.std(ker) / 2
        bias /= jnp.sqrt(3 * ker.shape[0]) * jnp.std(bias) / 2
        out_params[layer] = {'kernel': ker, 'bias': bias}
    return {'params': out_params}


def force_init(params):
    # ravel the params
    raveled_params, unravel = ravel_pytree(params)
    # sample normal distribution in shape of raveled_params
    # unravel the params
    params = unravel(jnp.random.normal(raveled_params.shape))
    # then typicalize
    params = typicalize(params)
    return params
