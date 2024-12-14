import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm
from jax.scipy.special import logsumexp

from .utils import unit, Raveler, logrectdet
from .math import gaussint_ln_noncentral_erf, log_hyperball_volume, log_small_hyperspherical_cap


def find_radius_vectorized(center, vecs, cutoff, fn, 
                           rtol=1e-1,  
                           init_mult=1, iters=10, jump=2.0):
    mults = init_mult * jnp.ones(vecs.shape[0])
    highs = jnp.inf * jnp.ones(vecs.shape[0])
    lows = jnp.zeros(vecs.shape[0])

    center_losses = jax.vmap(fn, in_axes=(None, 0))(center, 0 * vecs)
    vec_losses = jax.vmap(fn, in_axes=(None, 0))(center, jnp.einsum('b,bn->bn', mults, vecs))
    deltas = vec_losses - center_losses

    while iters > 0 and jnp.any(jnp.abs(deltas - cutoff) > cutoff * rtol):
        center_losses = jax.vmap(fn, in_axes=(None, 0))(center, 0 * vecs)
        vec_losses = jax.vmap(fn, in_axes=(None, 0))(center, jnp.einsum('b,bn->bn', mults, vecs))
        deltas = vec_losses - center_losses

        low = deltas < cutoff
        high = deltas > cutoff

        lows = jnp.where(low, mults, lows)
        highs = jnp.where(high, mults, highs)

        mults = jnp.where(highs == jnp.inf, mults * jump, (highs + lows) / 2)

        iters -= 1

    return mults, deltas


def get_estimates_vectorized_gauss(n, 
                                   sigma,
                                   *,
                                   preconditioner=None, 
                                   fn=None,
                                   unary_fn=None,
                                   params,
                                   gaussint_fn=gaussint_ln_noncentral_erf,
                                   debug=False,
                                   tol=1e-2,
                                   seed=42,
                                   **kwargs):
    

    if fn is None:
        assert unary_fn is not None, "fn or unary_fn must be provided"
        fn = lambda a, b: unary_fn(a + b)

    center = params.raveled if isinstance(params, Raveler) else params
    D = center.shape[0]

    vecs = jax.random.normal(jax.random.key(seed), (n, D))
    vecs = jax.vmap(unit)(vecs)
    if preconditioner is not None:
        vecs = vecs @ preconditioner.T

    props = norm(vecs, axis=1)
    uvecs = jax.vmap(unit)(vecs)

    kwargs = {'cutoff': 1e-3, 'fn': fn, 'iters': 100, 'rtol': 1e-2, **kwargs}
    mults, deltas = find_radius_vectorized(center, vecs, **kwargs)

    x1 = mults * props
    a = 1 / sigma**2
    b = -(uvecs @ center) / sigma**2
    c = -(center @ center) / (2 * sigma**2)

    if debug:
        print(f"{a.shape=}\n{b.shape=}\n{c.shape=}")

    logabsint = gaussint_fn(a=a, b=b, n=D-1, x1=x1, c=c, tol=tol, debug=debug)
    # assert jnp.all(sgn == 1), sgn
    logconst = log_hyperball_volume(D) + jnp.log(D) - (D/2) * jnp.log(2 * jnp.pi * sigma**2)
    # including prefactor and importance sampling correction
    estimates = logabsint + logconst - D * jnp.log(props)

    return estimates, props, mults, deltas, logabsint


def aggregate(estimates, **kwargs):
    return logsumexp(jnp.array(estimates), b=1/len(estimates), **kwargs)


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

    center = params.raveled if isinstance(params, Raveler) else params
    D = center.shape[0]
    vecs = jax.random.normal(jax.random.key(seed), (n, D))
    vecs = jax.vmap(unit)(vecs)
    if preconditioner is not None:
        vecs = vecs @ preconditioner.T

    props = norm(vecs, axis=1)

    kwargs = {'cutoff': 1e-3, 'fn': fn, 'iters': 100, 'rtol': 1e-2, **kwargs}
    mults, deltas = find_radius_vectorized(center, vecs, **kwargs)

    estimates = D * jnp.log(mults) + log_hyperball_volume(D)

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
        x = rad * jnp.cos(theta) + vec * jnp.sinc(theta / jnp.pi)

        # assert jnp.abs(norm(x) - norm(rad)) < 1e-5, f"norm(x) = {norm(x)}, norm(rad) = {norm(rad)}"

        return unary_fn(x)

    return fn_sphere

def check_preconditioner(preconditioner, rhat):
    logdet = logrectdet(preconditioner)
    assert jnp.abs(logdet) < 1.0, f"logrectdet(preconditioner) = {logdet}"
    maxradial = jnp.max(jnp.abs(rhat @ preconditioner))
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

    if preconditioner is None:
        vecs = jax.random.normal(jax.random.key(seed), (n, center.shape[0]))
    else:
        vecs = jax.random.normal(jax.random.key(seed), (n, preconditioner.shape[1]))
        vecs = jax.vmap(unit)(vecs)
        vecs = vecs @ preconditioner.T
    # project vecs onto tangent space
    vecs = vecs - jnp.outer(vecs @ rhat, rhat)
    if preconditioner is None:
        vecs = jax.vmap(unit)(vecs)

    props = norm(vecs, axis=1)

    mults, deltas = find_radius_vectorized(center, vecs, cutoff=1e-3, fn=fn, iters=100, rtol=1e-2)
    thetas = mults * props / norm(center)
    D = center.shape[0]
    logvols = log_small_hyperspherical_cap(D, thetas) - (D - 1) * jnp.log(props)

    return logvols, props, mults, deltas