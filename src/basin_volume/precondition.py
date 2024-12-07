import jax.numpy as jnp

def hessian_preconditioner(H, eps=1e-3, exponent=0.5):
    evals, evecs = jnp.linalg.eigh(H)
    
    p = 1 / (jnp.abs(evals)**exponent + eps)
    logp = jnp.log(p)
    logp_norm = logp - jnp.mean(logp)
    p = jnp.exp(logp_norm)
    P = jnp.einsum('ij,j->ij', evecs, p)
    return P

def diag_preconditioner(spec, eps=1e-3, exponent=0.5):
    p = 1 / (jnp.abs(spec)**exponent + eps)
    logp = jnp.log(p)
    logp_norm = logp - jnp.mean(logp)
    p = jnp.exp(logp_norm)
    P_scale = jnp.diag(p)
    return P_scale