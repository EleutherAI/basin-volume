import jax
import jax.numpy as jnp
from dataclasses import dataclass
from jax.flatten_util import ravel_pytree
from typing import Callable
from jax.numpy.linalg import norm
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch

def unit(v, **kwargs):
    return v / norm(v, **kwargs)

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
    

def orthogonal_complement(r):
    r = unit(r)
    eye = jnp.eye(r.shape[0])
    u = eye[0] - r
    u = unit(u)
    hou = eye - 2 * jnp.outer(u, u)
    return hou[:, 1:]
def logrectdet(M):
    return jnp.sum(jnp.log(jnp.linalg.svdvals(M)))

def rectdet(M):
    return jnp.exp(logrectdet(M))
def logspace(start, end, num):
    return 10**jnp.linspace(jnp.log10(start), jnp.log10(end), num)
linspace = jnp.linspace
def logspace_indices(length, num):
    # logarithically spaced indices from each end towards the middle
    num_beginning = num // 2 + 1
    num_end = num - num_beginning
    beginning = logspace(1, length // 2 + 1, num_beginning)
    beginning -= 1
    end = length - logspace(1, (length - length // 2) + 1, num_end + 1)
    end = end[-2::-1]
    return jnp.concatenate([beginning, end]).astype(int)    


def normal_probability_plot(data, figsize=(10, 6), title="Normal Probability Plot"):
    """
    Create a normal probability plot for the given data.
    
    Parameters:
    - data: JAX array or list of data points
    - figsize: tuple, size of the figure (width, height)
    - title: str, title of the plot
    
    Returns:
    - fig, ax: matplotlib figure and axis objects
    """
    # Convert to JAX array if it's not already
    if not isinstance(data, jnp.ndarray):
        data = jnp.array(data)
    
    # Normalize the data
    normalized_data = (data - jnp.mean(data)) / jnp.std(data)
    
    # Sort the normalized data
    sorted_data = jnp.sort(normalized_data)
    
    # Calculate theoretical quantiles
    n = len(sorted_data)
    theoretical_quantiles = stats.norm.ppf((jnp.arange(1, n+1) - 0.5) / n)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the points
    ax.scatter(theoretical_quantiles, sorted_data, alpha=0.5)
    
    # Plot the line y=x
    ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".3")
    
    # Set labels and title
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles (Normalized)")
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return fig, ax

# Example usage:
# fig, ax = normal_probability_plot(estimates_100k)
# plt.show()
def summarize(obj, size_limit=10, str_limit=100):
    if type(obj) in [int, float, bool]:
        return obj
    if isinstance(obj, str) and len(obj) <= str_limit:
        return obj
    out = {}
    out['type'] = type(obj)
    out['size'] = get_size(obj)
    info = get_info(obj)
    if info is not None:
        out['info'] = info
    if out['size'] <= size_limit:
        out['contents'] = get_contents(obj, size_limit)
    return out

def get_info(obj):
    if isinstance(obj, jax.Array) or isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):
        return {'shape': obj.shape, 'dtype': obj.dtype}
    else:
        return None

def get_contents(obj, size_limit):
    if isinstance(obj, torch.nn.parameter.Parameter):
        return obj.tolist()
    elif isinstance(obj, jax.Array) or isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return [{'key': summarize(k, size_limit), 'value': summarize(v, size_limit)} for k, v in obj.items()]
    elif any(isinstance(obj, t) for t in [list, tuple, set]):
        return [summarize(v, size_limit) for v in obj]
    elif isinstance(obj, str):
        return obj
    else:
        raise ValueError(f"Unsupported type: {type(obj)}")

def get_size(obj):
    if isinstance(obj, torch.nn.parameter.Parameter):
        return obj.numel()
    elif isinstance(obj, jax.Array) or isinstance(obj, np.ndarray):
        return obj.size
    elif isinstance(obj, torch.Tensor):
        return obj.numel()
    elif any(isinstance(obj, t) for t in [dict, list, tuple, set, str]):
        return len(obj)
    else:
        raise ValueError(f"Unsupported type: {type(obj)}")

def flatten_dict(d):
    new_d = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in flatten_dict(v).items():
                new_d[(k,) + k2] = v2
        else:
            new_d[(k,)] = v
    return new_d