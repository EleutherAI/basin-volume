import einops as eo
from dataclasses import dataclass
from tqdm import tqdm
import torch

from .utils import norm, unit, logrectdet, weighted_logsumexp, print_gpu_memory
from .math import log, cos, sinc, gaussint_ln_noncentral_erf, log_hyperball_volume, log_small_hyperspherical_cap
from .vectors import ImplicitVector, ImplicitRandomVector

def find_radius_vectorized(center, vecs, cutoff, fn, *,
                           rtol=1e-1,  
                           init_mult=1, iters=10, jump=2.0):
    batch_size = len(vecs)
    device = vecs[0].device
    implicit = isinstance(center, ImplicitVector)

    mults = init_mult * torch.ones(batch_size, device=device)
    highs = torch.inf * torch.ones(batch_size, device=device)
    lows = torch.zeros(batch_size, device=device)

    # Compute losses for each vector one at a time
    vec_losses = torch.stack([fn(center, mults[i] * vecs[i]) for i in range(batch_size)])
    center_losses = torch.stack([fn(center, 0)] * batch_size)

    deltas = vec_losses - center_losses

    while iters > 0 and any(abs(deltas - cutoff) > cutoff * rtol):
        vec_losses = torch.stack([fn(center, mults[i] * vecs[i]) for i in range(batch_size)])

        deltas = vec_losses - center_losses

        low = deltas < cutoff
        high = deltas > cutoff

        lows = torch.where(low, mults, lows)
        highs = torch.where(high, mults, highs)

        mults = torch.where(highs == torch.inf, mults * jump, (highs + lows) / 2)

        iters -= 1

    return mults, deltas

@dataclass
class VolumeResult:
    estimates: torch.Tensor
    props: torch.Tensor
    mults: torch.Tensor
    deltas: torch.Tensor
    logabsint: torch.Tensor

def get_estimates_vectorized_gauss(n, 
                                   sigma,
                                   *,
                                   batch_size=None,
                                   preconditioner=None, 
                                   fn=None,
                                   unary_fn=None,
                                   params,
                                   gaussint_fn=gaussint_ln_noncentral_erf,
                                   debug=False,
                                   tol=1e-2,
                                   y_tol=5,
                                   seed=42,
                                   with_tqdm=True,
                                   **kwargs):
    implicit = isinstance(params, ImplicitVector)

    if fn is None:
        assert unary_fn is not None, "fn or unary_fn must be provided"
        assert not implicit, "params must be a concrete vector"
        fn = lambda a, b: unary_fn(a + b)

    center = params
    D = center.shape[0]

    if batch_size is None:
        batch_size = n

    estimates_all = []
    props_all = []
    mults_all = []
    deltas_all = []
    logabsint_all = []

    torch.manual_seed(seed)

    for i in tqdm(range(0, n, batch_size), total=n // batch_size, disable=not with_tqdm):
        if implicit:
            assert batch_size == 1, "batch_size must be 1 for implicit vectors"
            vecs = [ImplicitRandomVector(seed+i, params)]
        else:
            vecs = torch.randn(batch_size, D, device=center.device)
        if debug:
            print("after randn")
            print_gpu_memory()
        if implicit:
            vecs = [unit(vecs[0])]
        else:
            vecs = unit(vecs, dim=1, keepdim=True)
        if preconditioner is not None:
            assert not implicit, "preconditioner only supported for concrete vectors"
            vecs = preconditioner(vecs)

        if implicit:
            props = norm(vecs[0]).unsqueeze(0)
        else:
            props = norm(vecs, dim=1)
        if debug:
            print_gpu_memory()

        kwargs = {'cutoff': 1e-3, 'fn': fn, 'iters': 100, 'rtol': 1e-2, **kwargs}
        mults, deltas = find_radius_vectorized(center, vecs, **kwargs)

        x1 = mults * props
        a = 1 / sigma**2
        vc = (vecs[0] @ center).unsqueeze(0) if implicit else vecs @ center
        b = -vc / (sigma**2 * props)
        c = -(center @ center) / (2 * sigma**2)
        if debug:
            print_gpu_memory()

        if debug:
            print(f"{a.shape=}\n{b.shape=}\n{c.shape=}")

        logabsint = gaussint_fn(a=a, b=b, n=D-1, x1=x1, c=c, tol=tol, y_tol=y_tol, debug=debug)
        # assert jnp.all(sgn == 1), sgn
        logconst = log_hyperball_volume(D) + log(D) - (D/2) * log(2 * torch.pi * sigma**2)
        # including prefactor and importance sampling correction
        estimates = logabsint + logconst - D * log(props)

        estimates_all.append(estimates)
        props_all.append(props)
        mults_all.append(mults)
        deltas_all.append(deltas)
        logabsint_all.append(logabsint)

    # concatenate all the lists
    estimates_all = torch.cat(estimates_all)
    props_all = torch.cat(props_all)
    mults_all = torch.cat(mults_all)
    deltas_all = torch.cat(deltas_all)
    logabsint_all = torch.cat(logabsint_all)

    return VolumeResult(estimates_all, props_all, mults_all, deltas_all, logabsint_all)


def aggregate(estimates, dim=-1, **kwargs):
    return weighted_logsumexp(estimates, w=torch.ones_like(estimates)/len(estimates), dim=dim, **kwargs)


# without Gaussian weighting
def get_estimates_vectorized(n, 
                             preconditioner=None, 
                             *,
                             fn=None, 
                             params,
                             unary_fn=None,
                             seed=42,
                             **kwargs):
    if fn is None:
        assert unary_fn is not None, "fn or unary_fn must be provided"
        fn = lambda a, b: unary_fn(a + b)

    center = params
    D = center.shape[0]
    torch.manual_seed(seed)
    vecs = torch.randn(n, D, device=center.device)
    vecs = unit(vecs, dim=1, keepdim=True)
    if preconditioner is not None:
        vecs = vecs @ preconditioner.T

    props = norm(vecs, dim=1)

    kwargs = {'cutoff': 1e-3, 'fn': fn, 'iters': 100, 'rtol': 1e-2, **kwargs}
    mults, deltas = find_radius_vectorized(center, vecs, **kwargs)

    estimates = D * log(mults) + log_hyperball_volume(D)

    return estimates, props, mults, deltas


# clamped to sphere
def make_fn_sphere(unary_fn):
    def fn_sphere(rad, vec):
        # assert jnp.abs(rad @ vec) < 1e-6, f"rad @ vec = {rad @ vec}"
        
        theta = norm(vec) / norm(rad)
        
        # tang = vec / theta  # tangent vector with norm equal to rad
        # x = rad * jnp.cos(theta) + tang * jnp.sin(theta)

        # equivalent to above but more numerically stable
        # note that jnp.sinc(x / jnp.pi) = jnp.sin(x) / x
        x = rad * cos(theta) + vec * sinc(theta / torch.pi)

        # assert jnp.abs(norm(x) - norm(rad)) < 1e-5, f"norm(x) = {norm(x)}, norm(rad) = {norm(rad)}"

        return unary_fn(x)

    return fn_sphere

def check_preconditioner(preconditioner, rhat):
    logdet = logrectdet(preconditioner)
    assert abs(logdet) < 1.0, f"logrectdet(preconditioner) = {logdet}"
    maxradial = max(abs(rhat @ preconditioner))
    assert maxradial < 1e-3, f"max(abs(rhat @ preconditioner)) = {maxradial}"

def get_estimates_sphere_vectorized(n, 
                                    preconditioner=None, 
                                    *,
                                    fn=None, 
                                    unary_fn=None,
                                    params,
                                    seed=42,
                                    **kwargs):
    if fn is None:
        fn = make_fn_sphere(unary_fn)

    center = params.raveled
    rhat = unit(center)

    if preconditioner is not None:
        check_preconditioner(preconditioner, rhat)

    torch.manual_seed(seed)
    if preconditioner is None:
        vecs = torch.randn(n, center.shape[0], device=center.device)
    else:
        vecs = torch.randn(n, preconditioner.shape[1], device=center.device)
        vecs = unit(vecs, dim=1, keepdim=True)
        vecs = vecs @ preconditioner.T
    # project vecs onto tangent space
    vecs = vecs - torch.outer(vecs @ rhat, rhat)
    if preconditioner is None:
        vecs = unit(vecs, dim=1, keepdim=True)

    props = norm(vecs, dim=1)

    mults, deltas = find_radius_vectorized(center, vecs, cutoff=1e-3, fn=fn, iters=100, rtol=1e-2)
    thetas = mults * props / norm(center)
    D = center.shape[0]
    logvols = log_small_hyperspherical_cap(D, thetas) - (D - 1) * log(props)

    return logvols, props, mults, deltas