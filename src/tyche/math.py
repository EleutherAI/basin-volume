# import jax
# import jax.numpy as jnp
import numpy as np
# import jax.scipy as jsp 
import scipy as sp
# from jax.scipy.special import logsumexp
import torch

from .utils import weighted_logsumexp


def if_tensor(torch_fn, np_fn):
    def wrapper(x):
        if isinstance(x, torch.Tensor):
            return torch_fn(x)
        else:
            return np_fn(x)
    return wrapper

sqrt = if_tensor(torch.sqrt, np.sqrt)
log = if_tensor(torch.log, np.log)
exp = if_tensor(torch.exp, np.exp)
sin = if_tensor(torch.sin, np.sin)
cos = if_tensor(torch.cos, np.cos)
sinc = if_tensor(torch.sinc, np.sinc)
erfc = if_tensor(torch.special.erfc, sp.special.erfc)
erfcx = if_tensor(torch.special.erfcx, sp.special.erfcx)


def log_hyperball_volume(dim):
    return (dim / 2) * log(np.pi) - sp.special.gammaln(dim / 2 + 1)

def log_hypersphere_area(dim):
    return log(2) + (dim / 2) * log(np.pi) - sp.special.gammaln(dim / 2)

def log_small_hyperspherical_cap(dim, t, angle=False):
    x = sin(t)
    logr = log(t) if angle else log(x)
    return (dim - 1) * logr + log_hyperball_volume(dim - 1) - log_hypersphere_area(dim)

def log_fn_factory(a, b, n):
    return lambda x: -a/2 * x**2 + b*x + n * log(abs(x))
def log_fn_rel_factory(a, b, n, x_ref):
    # log_fn(x) - log_fn(x_ref)
    # written this way for numerical stability
    return lambda x: -a/2 * (x - x_ref) * (x + x_ref) + b * (x - x_ref) + n * log(abs(x / x_ref))
def dlog_fn_factory(a, b, n):
    return lambda x: -a*x + b + n / x
def d2log_fn_factory(a, b, n):
    return lambda x: -a - n / x**2

# def approx_log_fn_factory(a, b, n, x0):
#     log_fn = log_fn_factory(a, b, n)
#     dlog_fn = dlog_fn_factory(a, b, n)
#     d2log_fn = d2log_fn_factory(a, b, n)
#     return lambda x: log_fn(x0) + dlog_fn(x0) * (x - x0) + 1/2 * d2log_fn(x0) * (x - x0)**2

def erfc_ln(z):
    # numerically stable log(erfc(z)) for both positive and negative z :)
    return torch.where(z < 0,
                        log(erfc(z)),
                        log(erfcx(z)) - z**2)

def standard_cdf_ln(z):
    # CDF of standard normal distribution
    return erfc_ln(-z / np.sqrt(2)) - log(2)

def scaled_cdf_ln(x, mu, sigma):
    # CDF of normal distribution with mean mu and std sigma
    return standard_cdf_ln((x - mu) / sigma) + log(sigma)

def mu_sigma_int_ln(x, mu, sigma):
    # integral of exp(-1/2 (x - mu)**2 / sigma**2) from -inf to x
    return scaled_cdf_ln(x, mu, sigma) + log(sqrt(2*np.pi))

def abc_int_ln(x, a, b, c):
    # integral of exp(-1/2 ax^2 + bx + c) from -inf to x
    return mu_sigma_int_ln(x, b/a, 1/sqrt(a)) + c + 1/2 * b**2 / a

def f012_int_ln(center, x1, f0, f1, f2, debug=False):
    # integral of exp(1/2 f2 (x - center)**2 + f1 (x - center) + f0)
    # from 0 to x1
    # f2 is negative, a is positive
    # n.b. these are NOT the same as a, b, c from gaussint_ln_noncentral!
    a, b, c = -f2, f1, f0
    upper = abc_int_ln(x1 - center, a, b, c)  # integral from -inf to x1
    lower = abc_int_ln(0 - center, a, b, c)  # integral from -inf to 0
    if debug:
        print()
        print("f012_int_ln terms:")
        print(f"{upper = }\n{lower = }")
    assert all(upper > lower), "upper must be greater than lower"
    diff = weighted_logsumexp(torch.stack([upper, lower], dim=-1), 
                            w=torch.tensor([1, -1], device=upper.device), 
                            dim=-1)
    if any(upper - diff > log(1e5)):
        if debug:
            print()
            print(f"{diff = }\n{upper - diff = }")
        raise ValueError("catastrophic cancellation in f012_int_ln, investigate")
    return diff

def gaussint_ln_noncentral_erf(a, b, n, x1, c=0, tol=1e-2, y_tol=5, debug=False):
    # integral of exp(-1/2 ax^2 + bx + c) * x^n
    # from 0 to x1
    # we find the maximum point <= x1 and use a quadratic (i.e. Gaussian) approximation
    # sort of generalizing Laplace's method
    # but it works for endpoints in the tail, not just around the maximum

    # TODO: "damn the torpedoes" mode, i.e. don't check accuracy
    # find highest point <= x1
    mu = b / a
    center = mu / 2
    dist = sqrt(mu**2 + 4 * n / a) / 2
    if debug:
        print(f"{a = }\n{b = }\n{n = }\n{x1 = }\n{c = }")
        print(f"{mu = }\n{center = }\n{dist = }")
    global_max = center + dist
    global_in_range = global_max <= x1
    if debug:
        print(f"{global_max=}, {global_in_range=}")
    max_pt = torch.minimum(global_max, x1)
    # get approximation stuff
    log_fn = log_fn_rel_factory(a, b, n, max_pt)
    dlog_fn = dlog_fn_factory(a, b, n)
    d2log_fn = d2log_fn_factory(a, b, n)
    f0 = log_fn(max_pt)
    f1 = dlog_fn(max_pt)
    f2 = d2log_fn(max_pt)
    approx_log_fn = lambda x: f0 + f1 * (x - max_pt) + 1/2 * f2 * (x - max_pt)**2

    constant_term = log_fn_factory(a, b, n)(max_pt) + c

    # global in range:
    # extrapolate down by tol
    # y_tol = -jnp.log(tol)
    # new version: y_tol is an input
    rad_global = sqrt(2 * y_tol / -f2)
    check_low_global = torch.clip(max_pt - rad_global, torch.zeros_like(x1), x1)
    check_high_global = torch.clip(max_pt + rad_global, torch.zeros_like(x1), x1)
    # check approximation error at those points
    error_low_global = log_fn(check_low_global) - approx_log_fn(check_low_global)
    error_high_global = log_fn(check_high_global) - approx_log_fn(check_high_global)
    abs_error_global = torch.maximum(abs(error_low_global), abs(error_high_global))

    # global out of range:
    # extrapolate down by tol
    rad_x1 = f1 / -f2 - sqrt(f1**2 / f2**2 + 2 * y_tol / -f2)
    if any(~global_in_range & (abs(f1/-f2) > 1e5 * torch.minimum(abs(y_tol / f1), abs(rad_x1)))):
        # catastrophic cancellation in rad_x1, use linear approximation
        if debug:
            print()
            print("Catastrophic cancellation in rad_x1, using linear approximation...")
        rad_x1 = -y_tol / f1
    check_x1 = torch.clip(max_pt + rad_x1, torch.zeros_like(max_pt), None)
    # check approximation error at that point
    abs_error_x1 = abs(log_fn(check_x1) - approx_log_fn(check_x1))

    # branch
    abs_error = torch.where(global_in_range, abs_error_global, abs_error_x1)
    if any(abs_error > tol):
        # check accuracy of approximation; in practice tol is extremely conservative
        # empirically, tol of 0.03 is still accurate to about +-1e-7 in the log
        # (based on comparison to _normed for n=12_000, a=2, b=3, x1=100)
        # i.e. beyond fp32 and approaching fp64 precision
        # also, we only actually need to be accurate to maybe +-1 in the log!
        if debug:
            print()
            print("approx error debug:")
            idx = abs_error > tol
            for name, var in zip(["a", "b", "n", "x1", "c", "rad"], [a, b, n, x1, c, torch.where(global_in_range, rad_global, rad_x1)]):
                if isinstance(var, torch.Tensor):
                    if var.ndim == 1:
                        print(f"{name} = {var[idx][0]}")
                    else:
                        print(f"{name} = {var}")
                else:
                    print(f"{name} = {var}")
        raise ValueError("Approximation error too high, raise tol or investigate")
    
    # use erf to integrate
    if debug:
        print()
        print("f012_int_ln inputs:")
        print(f"{max_pt = }\n{x1 = }\n{f0 = }\n{f1 = }\n{f2 = }\n{c = }")
    return f012_int_ln(max_pt, x1, f0, f1, f2, debug=debug) + constant_term
