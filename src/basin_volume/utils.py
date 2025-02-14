import gc
from typing import Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch
from torch import norm

BASIN_VOLUME_DIR = "/mnt/ssd-1/adam/basin-volume"


def unit(v, **kwargs):
    return v / norm(v, **kwargs)


# https://github.com/PhilippDahlinger/torch_weighted_logsumexp
def weighted_logsumexp(
    logx: torch.Tensor,
    w: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    return_sign: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    from https://github.com/PhilippDahlinger/torch_weighted_logsumexp
    
    This is a Pytorch port of the Tensorflow function `reduce_weighted_logsumexp` from
    https://www.tensorflow.org/probability/api_docs/python/tfp/math/reduce_weighted_logsumexp
    Computes log(abs(sum(weight * exp(elements across tensor dimensions)))) in a numerically stable way.
    Right now, it only supports to perform the operation over 1 dimension. (mandatory parameter)
    :param logx: Tensor to reduce
    :param w: weights, has to be same shape as logx
    :param dim: dimension to reduce
    :param keep_dim: if True, retains reduced dimensions with length 1
    :param return_sign: if True, return the sign of weight * exp(elements across tensor dimensions)))
    :return: Either the reduced tensor or a tuple of the reduced tensor and the sign
    """
    log_absw_x = logx + torch.log(torch.abs(w))
    max_log_absw_x = torch.amax(log_absw_x, dim=dim, keepdim=True)
    max_log_absw_x = torch.where(
        torch.isinf(max_log_absw_x),
        torch.zeros(torch.Size([]), dtype=max_log_absw_x.dtype, device=max_log_absw_x.device),
        max_log_absw_x)
    wx_over_absw_x = torch.sign(w) * torch.exp(log_absw_x - max_log_absw_x)
    sum_wx_over_max_absw_x = torch.sum(wx_over_absw_x, dim=dim, keepdim=keepdim)
    if not keepdim:
        max_log_absw_x = torch.squeeze(max_log_absw_x, dim=dim)
    sgn = torch.sign(sum_wx_over_max_absw_x)
    lswe = max_log_absw_x + torch.log(sgn * sum_wx_over_max_absw_x)
    if return_sign:
        return lswe, sgn
    else:
        return lswe


# @dataclass
# class Raveler:
#     raveled: jnp.ndarray
#     unravel: Callable

#     def __init__(self, params, unravel=None):
#         if isinstance(params, dict):
#             self.raveled, self.unravel = ravel_pytree(params)
#         else:
#             assert isinstance(params, jnp.ndarray), "params must be a JAX array or a dict"
#             self.raveled = params
#             assert unravel is not None, "unravel must be provided if params are raveled"
#             self.unravel = unravel
    
#     @property
#     def unraveled(self):
#         return self.unravel(self.raveled)
    
#     @property
#     def norm(self):
#         return jnp.linalg.norm(self.raveled)
    

def orthogonal_complement(r):
    r = unit(r)
    eye = torch.eye(r.shape[0], device=r.device)
    u = eye[0] - r
    u = unit(u)
    # Householder matrix
    hou = eye - 2 * torch.outer(u, u)
    return hou[:, 1:]

def logrectdet(M):
    return torch.sum(torch.log(torch.linalg.svdvals(M)))

def rectdet(M):
    return torch.exp(logrectdet(M))

def logspace(start, end, num):
    return 10**np.linspace(np.log10(start), np.log10(end), num)

linspace = np.linspace

def logspace_indices(length, num):
    # logarithically spaced indices from each end towards the middle
    num_beginning = num // 2 + 1
    num_end = num - num_beginning
    beginning = logspace(1, length // 2 + 1, num_beginning)
    beginning -= 1
    end = length - logspace(1, (length - length // 2) + 1, num_end + 1)
    end = end[-2::-1]
    return np.concatenate([beginning, end]).astype(int)    


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
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # Normalize the data
    normalized_data = (data - np.mean(data)) / np.std(data)
    
    # Sort the normalized data
    sorted_data = np.sort(normalized_data)
    
    # Calculate theoretical quantiles
    n = len(sorted_data)
    theoretical_quantiles = stats.norm.ppf((np.arange(1, n+1) - 0.5) / n)
    
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
    if isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):  # or isinstance(obj, jax.Array)
        return {'shape': obj.shape, 'dtype': obj.dtype, 'device': obj.device}
    else:
        return None

def get_contents(obj, size_limit):
    if isinstance(obj, torch.nn.parameter.Parameter):
        return obj.tolist()
    elif isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):  # or isinstance(obj, jax.Array)
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
    elif isinstance(obj, np.ndarray):  # or isinstance(obj, jax.Array)
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

def scaled_histogram(values, label, settings, nbins=None):
    if nbins is None:
        nbins = int(np.sqrt(len(values)))
    counts, bins = np.histogram(values, bins=nbins)
    counts = counts / counts.max()
    plt.stairs(counts, bins, **dict(settings, label=label))


def list_largest_tensors():
    # Get all tensor objects
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensors.append(obj)
        except:
            pass
    
    # Group tensors by memory location
    memory_dict = {}
    for t in tensors:
        if t.device.type == 'cuda':
            location = t.data_ptr()
            if location not in memory_dict:
                memory_dict[location] = []
            memory_dict[location].append(t)
    
    # Calculate sizes and sort by memory usage
    tensor_sizes = []
    for location, tensor_list in memory_dict.items():
        # Take the first tensor from each memory location
        tensor = tensor_list[0]
        size_mb = tensor.nelement() * tensor.element_size() / (1024 * 1024)
        tensor_sizes.append((size_mb, tensor.size(), tensor.dtype, len(tensor_list)))
    
    # Sort by size in descending order
    tensor_sizes.sort(reverse=True)
    
    # Calculate cumulative sizes relative to largest tensor
    if tensor_sizes:
        largest_size = tensor_sizes[0][0]
        cumulative = 0
    
    # Print results
    print(f"{'Size (MB)':>10} {'Cumul.(x)':>10} {'Shape':>20} {'Type':>10} {'Aliases':>8}")
    print("-" * 60)
    for size, shape, dtype, num_tensors in tensor_sizes:
        cumulative += size
        relative_cumul = cumulative / largest_size
        print(f"{size:10.2f} {relative_cumul:10.2f} {str(shape):>20} {str(dtype):>10} {num_tensors:>8}")

    
def print_gpu_memory():
    print()
    if torch.cuda.is_available():
        print(f"Current GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Max GPU memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        print(f"Current GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print(f"Max GPU memory reserved: {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB")
    else:
        print("No GPU available")