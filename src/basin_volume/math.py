import jax
import jax.numpy as jnp
import jax.scipy as jsp 
import scipy as sp
from jax.scipy.special import logsumexp


def log_hyperball_volume(dim):
    return (dim / 2) * jnp.log(jnp.pi) - jsp.special.gammaln(dim / 2 + 1)

def log_hypersphere_area(dim):
    return jnp.log(2) + (dim / 2) * jnp.log(jnp.pi) - jsp.special.gammaln(dim / 2)

def log_small_hyperspherical_cap(dim, t, angle=False):
    x = jnp.sin(t)
    logr = jnp.log(t) if angle else jnp.log(x)
    return (dim - 1) * logr + log_hyperball_volume(dim - 1) - log_hypersphere_area(dim)

def log_fn_factory(a, b, n):
    return lambda x: -a/2 * x**2 + b*x + n * jnp.log(jnp.abs(x))
def dlog_fn_factory(a, b, n):
    return lambda x: -a*x + b + n / x
def d2log_fn_factory(a, b, n):
    return lambda x: -a - n / x**2

def approx_log_fn_factory(a, b, n, x0):
    log_fn = log_fn_factory(a, b, n)
    dlog_fn = dlog_fn_factory(a, b, n)
    d2log_fn = d2log_fn_factory(a, b, n)
    return lambda x: log_fn(x0) + dlog_fn(x0) * (x - x0) + 1/2 * d2log_fn(x0) * (x - x0)**2

def erfc_ln(z):
    # numerically stable log(erfc(z)) for both positive and negative z :)
    return jnp.where(z < 0,
                     jnp.log(sp.special.erfc(z)),
                     jnp.log(sp.special.erfcx(z)) - z**2)

def standard_cdf_ln(z):
    # CDF of standard normal distribution
    return erfc_ln(-z / jnp.sqrt(2)) - jnp.log(2)

def scaled_cdf_ln(x, mu, sigma):
    # CDF of normal distribution with mean mu and std sigma
    return standard_cdf_ln((x - mu) / sigma) + jnp.log(sigma)

def mu_sigma_int_ln(x, mu, sigma):
    # integral of exp(-1/2 (x - mu)**2 / sigma**2) from 0 to x
    return scaled_cdf_ln(x, mu, sigma) + jnp.log(jnp.sqrt(2*jnp.pi))

def abc_int_ln(x, a, b, c):
    # integral of exp(-1/2 ax^2 + bx + c) from 0 to x
    return mu_sigma_int_ln(x, b/a, 1/jnp.sqrt(a)) + c + 1/2 * b**2 / a

def f012_int_ln(center, x1, f0, f1, f2, debug=False):
    # integral of exp(1/2 f2 (x - center)**2 + f1 (x - center) + f0)
    # from 0 to x1
    # f2 is negative, a is positive
    # n.b. these are NOT the same as a, b, c from gaussint_ln_noncentral!
    a, b, c = -f2, f1, f0
    upper = abc_int_ln(x1 - center, a, b, c)
    lower = abc_int_ln(-center, a, b, c)
    if debug:
        print(f"{upper=}, {lower=}")
    assert jnp.all(upper > lower), "upper must be greater than lower"
    return logsumexp(jnp.stack([upper, lower], axis=-1), b=jnp.array([1, -1]), axis=-1)

def gaussint_ln_noncentral_erf(a, b, n, x1, c=0, tol=1e-2, debug=False):
    # integral of exp(-1/2 ax^2 + bx + c) * x^n
    # from 0 to x1
    # we find the maximum point <= x1 and use a quadratic (i.e. Gaussian) approximation
    # sort of generalizing Laplace's method
    # but it works for endpoints in the tail, not just around the maximum

    # TODO: "damn the torpedoes" mode, i.e. don't check accuracy
    # find highest point <= x1
    mu = b / a
    center = mu / 2
    dist = jnp.sqrt(mu**2 + 4 * n / a) / 2
    global_max = center + dist
    global_in_range = global_max <= x1
    if debug:
        print(f"{global_max=}, {global_in_range=}")
    max_pt = jnp.minimum(global_max, x1)
    # get approximation stuff
    log_fn = log_fn_factory(a, b, n)
    dlog_fn = dlog_fn_factory(a, b, n)
    d2log_fn = d2log_fn_factory(a, b, n)
    f0 = log_fn(max_pt)
    f1 = dlog_fn(max_pt)
    f2 = d2log_fn(max_pt)
    approx_log_fn = lambda x: f0 + f1 * (x - max_pt) + 1/2 * f2 * (x - max_pt)**2

    # global in range:
    # extrapolate down by tol
    y_tol = -jnp.log(tol)
    rad_global = jnp.sqrt(2 * y_tol / -f2)
    check_low_global = jnp.clip(max_pt - rad_global, 0, x1)
    check_high_global = jnp.clip(max_pt + rad_global, 0, x1)
    # check approximation error at those points
    error_low_global = log_fn(check_low_global) - approx_log_fn(check_low_global)
    error_high_global = log_fn(check_high_global) - approx_log_fn(check_high_global)
    abs_error_global = jnp.maximum(jnp.abs(error_low_global), jnp.abs(error_high_global))

    # global out of range:
    # extrapolate down by tol
    rad_x1 = f1 / -f2 - jnp.sqrt(f1**2 / f2**2 + 2 * y_tol / -f2)
    # TODO change 1e10 back to 1e5!
    if jnp.any(~global_in_range & (jnp.abs(f1/-f2) > 1e10 * jnp.abs(y_tol / f1))):
        # This is not hard to implement but let's overcomplicate that bridge when we get to it
        raise ValueError("Catastrophic cancellation in rad_x1, replace this error with linear approximation")
    check_x1 = jnp.clip(max_pt + rad_x1, 0, None)
    # check approximation error at that point
    abs_error_x1 = jnp.abs(log_fn(check_x1) - approx_log_fn(check_x1))

    # branch
    abs_error = jnp.where(global_in_range, abs_error_global, abs_error_x1)
    if jnp.any(abs_error > tol):
        # check accuracy of approximation; in practice tol is extremely conservative
        # empirically, tol of 0.03 is still accurate to about +-1e-7 in the log
        # (based on comparison to _normed for n=12_000, a=2, b=3, x1=100)
        # i.e. beyond fp32 and approaching fp64 precision
        # also, we only actually need to be accurate to maybe +-1 in the log!
        raise ValueError("Approximation error too high, raise tol or use quad")
    
    # use erf to integrate
    if debug:
        print(f"{max_pt=}, {x1=}, {f0=}, {f1=}, {f2=}, {c=}")
    return f012_int_ln(max_pt, x1, f0, f1, f2, debug=debug) + c