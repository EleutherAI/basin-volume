# %%
from flax import linen as nn
# %%

def param_normal(fan_in: int, norm_scale: float = 1.0):
    """Kernel/bias initializer with variance 1/fan_in, untruncated"""
    return nn.initializers.normal(stddev=norm_scale * (fan_in) ** -0.5)
    

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

# %%
