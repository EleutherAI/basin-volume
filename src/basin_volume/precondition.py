# import jax.numpy as jnp
import torch

from .math import log, exp


def matrix_preconditioner(H, eps=1e-3, exponent=0.5):
    evals, evecs = torch.linalg.eigh(H)
    
    p = 1 / (abs(evals)**exponent + eps)
    logp = log(p)
    logp_norm = logp - torch.mean(logp)
    p = exp(logp_norm)
    P = torch.einsum('ij,j->ij', evecs, p)
    return lambda x: torch.einsum('...i,ij->...j', x, P)

def diag_preconditioner(spec, eps=1e-3, exponent=0.5):
    p = 1 / (torch.abs(spec)**exponent + eps)
    logp = log(p)
    logp_norm = logp - torch.mean(logp)
    p = exp(logp_norm)
    return lambda x: torch.einsum('...i,i->...i', x, p)