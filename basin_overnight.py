import mlp
import meta_poisoning_typical as mp

import jax
import jax.numpy as jnp
from flax import struct
from flax.training.train_state import TrainState
from jax.flatten_util import ravel_pytree
from optax import (
    softmax_cross_entropy as dense_xent,
    softmax_cross_entropy_with_integer_labels as sparse_xent
)
import optax
import numpy as np
from typing import Optional, Callable
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from tqdm import tqdm
from sys import argv

# Defns
def make_split(X, Y, splits: list[int], key: jax.random.PRNGKey) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
    if splits[-1] == -1:
        splits[-1] = len(X) - sum(splits[:-1])

    indices = jax.random.permutation(key, len(X))
    for split in splits:
        yield X[indices[:split]], Y[indices[:split]]
        indices = indices[split:]

def get_digits_splits(key: jax.random.PRNGKey, splits: list[int]) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
    X, Y = load_digits(return_X_y=True)
    X = X / 16.0
    return make_split(X, Y, splits, key)
@struct.dataclass
class SimpleConfig:
    seed: int = 0

    batch_size: int = 64
    num_epochs: int = 25

    opt: str = "sgd"
    lr: float = 0.1
    train_size: int = 768
    num_layers: int = 1

    mesa_constrain: bool = False
    norm_scale: float = 1.0
    spherical: bool = True
def get_model(cfg: SimpleConfig, x):
    seed = cfg.seed
    key = jax.random.key(seed)

    d_inner = x.shape[1]

    model = mlp.MLP(hidden_sizes=(d_inner,) * cfg.num_layers, 
                out_features=10, 
                norm_scale=cfg.norm_scale, 
                spherical=cfg.spherical)
    
    params = mlp.Params(model.init(key, x))  # this will already be close to the ellipsoid

    return model, params

def make_apply_full(model, unraveler):
    """Make an apply function that takes the full parameter vector."""
    def apply_full(raveled, x):
        params = unraveler(raveled)
        return model.apply(params, x)
    
    return apply_full

def compute_loss(params, apply_fn, X, Y):
    logits = apply_fn(params['p'], X)
    preds = jnp.argmax(logits, axis=-1)

    loss = sparse_xent(logits, Y).mean()
    acc = jnp.mean(preds == Y)
    return loss, acc

def train_simple(
    params_raveled, unravel: Callable, digits_splits, apply_fn, cfg: SimpleConfig,
    target_norm: Optional[float] = None, return_state: bool = False,
):
    x_train, y_train = digits_splits[0]
    x_test, y_test = digits_splits[1]
    x_shape = x_train[0].shape
    x_train, y_train = jnp.array(x_train), jnp.array(y_train)

    # LR schedule
    num_steps = cfg.num_epochs * len(x_train) // cfg.batch_size

    # Define the optimizer and training state
    if cfg.opt == "adam":
        sched = optax.cosine_decay_schedule(cfg.lr, num_steps)
        tx = optax.adam(learning_rate=sched, eps_root=1e-8)
    else:
        sched = optax.cosine_decay_schedule(cfg.lr, num_steps)
        tx = optax.sgd(learning_rate=sched, momentum=0.9)

    if target_norm is not None:
        params_raveled = params_raveled * target_norm / mlp.ellipsoid_norm(mlp.Params(params_raveled, unravel), cfg.spherical)
    elif cfg.mesa_constrain:
        target_norm = mlp.ellipsoid_norm(mlp.Params(params_raveled, unravel), cfg.spherical)

    state = TrainState.create(apply_fn=apply_fn, params=dict(p=params_raveled), tx=tx)

    # Forward and backward pass
    loss_and_grad = jax.value_and_grad(compute_loss, has_aux=True)

    # RNG key for each epoch
    keys = jax.vmap(jax.random.key)(jnp.arange(cfg.num_epochs))

    def train_step(state: TrainState, batch):
        loss, grads = loss_and_grad(state.params, state.apply_fn, *batch)
        state = state.apply_gradients(grads=grads)
        if target_norm is not None:
            state.params['p'] *= target_norm / mlp.ellipsoid_norm(mlp.Params(state.params['p'], unravel), cfg.spherical)
        return state, loss

    def epoch_step(state: TrainState, key) -> tuple[TrainState, tuple[jnp.ndarray, jnp.ndarray]]:
        # Re-shuffle the data at the start of each epoch
        indices = jax.random.permutation(key, len(x_train))
        x_train_, y_train_ = x_train[indices], y_train[indices]

        # Create the batches
        x_train_batches = jnp.reshape(x_train_, (-1, cfg.batch_size, *x_shape))
        y_train_batches = jnp.reshape(y_train_, (-1, cfg.batch_size))
        
        state, (losses, accs) = jax.lax.scan(train_step, state, (x_train_batches, y_train_batches))
        return state, (losses.mean(), accs.mean())

    state, (train_loss, _) = jax.lax.scan(epoch_step, state, keys)

    # Test loss
    logits = state.apply_fn(state.params['p'], x_test)
    test_loss = sparse_xent(logits, y_test).mean()

    if return_state:
        return test_loss, train_loss[-1], state
    return test_loss, train_loss[-1]

def find_radius(center, vec, cutoff, fn, rtol=1e-1, high=None, low=0, init_mult=1, iters=10, jump=2.0):
    center_loss = fn(center)
    vec_loss = fn(center + init_mult * vec)

    if iters == 0 or jnp.abs(vec_loss - center_loss - cutoff) < cutoff * rtol:
        return init_mult, vec_loss - center_loss
    if vec_loss - center_loss < cutoff:  # too low  
        low = init_mult
        if high is None:
            new_init_mult = init_mult * jump
        else:
            new_init_mult = (high + low) / 2
    else:  # too high
        high = init_mult
        new_init_mult = (high + low) / 2
    
    return find_radius(center, vec, cutoff, fn=fn, high=high, low=low, init_mult=new_init_mult, iters=iters - 1)
def logspace(start, end, num):
    return 10**jnp.linspace(jnp.log10(start), jnp.log10(end), num)
def logspace_indices(length, num):
    # logarithically spaced indices from each end towards the middle
    num_beginning = num // 2 + 1
    num_end = num - num_beginning
    beginning = logspace(1, length // 2 + 1, num_beginning)
    beginning -= 1
    end = length - logspace(1, (length - length // 2) + 1, num_end + 1)
    end = end[-2::-1]
    return jnp.concatenate([beginning, end]).astype(int)


seed = int(argv[3]) if len(argv) > 3 else 0
cfg = SimpleConfig(seed=seed, train_size=1024, opt="adam", lr=.06)
digits_splits = list(get_digits_splits(jax.random.key(cfg.seed), [cfg.train_size, -1]))
X_train, Y_train = digits_splits[0]
X_test, Y_test = digits_splits[1]

model, init_params = get_model(cfg, X_train)

apply_fn = make_apply_full(model, init_params.unravel)

def quick_train():
    return train_simple(init_params.raveled, init_params.unravel, digits_splits, apply_fn, cfg, return_state=True)

test_loss, train_loss, state = quick_train()
final_params = mlp.Params(state.params['p'], init_params.unravel)

def logvol_estimate(params, fn, key):
    center = params.raveled
    vec = jax.random.normal(key, center.shape)
    vec = vec / jnp.linalg.norm(vec)
    rad, delta = find_radius(center, vec, cutoff=1e-3, fn=fn, iters=100, rtol=1e-2)
    return center.shape[0] * jnp.log(rad), delta

def aggregate(estimates):
    return jax.scipy.special.logsumexp(jnp.array(estimates), b=1/len(estimates))

def loss_fn(params_raveled):
    loss, acc = compute_loss({'p': params_raveled}, apply_fn, X_train, Y_train)
    return loss

def get_estimates(n):
    keys = jax.random.split(jax.random.key(cfg.seed), n)

    estimates = []
    diffs = []

    for key in tqdm(keys):
        est, diff = logvol_estimate(final_params, loss_fn, key)
        estimates.append(est)
        diffs.append(diff)
    estimates = jnp.array(estimates)
    diffs = jnp.array(diffs)

    return estimates, diffs

param_dim = final_params.raveled.shape[0]



H = jax.hessian(loss_fn)(final_params.raveled)

eigvals = jnp.linalg.eigvalsh(H)
evals, evecs = jnp.linalg.eigh(H)
# evals: [n]
# evecs: [n, n]
# top eigenvector is evecs[:, -1]

idx = logspace_indices(eigvals.shape[0], 200)

p = 1 / (jnp.sqrt(jnp.abs(evals)) + 1e-3)
logp = jnp.log(p)
logp_norm = logp - jnp.mean(logp)
p = jnp.exp(logp_norm)
P = jnp.einsum('ij,j->ij', evecs, p)

def logvol_estimate_preconditioned(params, fn, key):
    center = params.raveled
    vec = jax.random.normal(key, center.shape)
    vec = vec / jnp.linalg.norm(vec)
    vec = P @ vec
    rad, delta = find_radius(center, vec, cutoff=1e-3, fn=fn, iters=100, rtol=1e-2)
    return center.shape[0] * jnp.log(rad), delta
def get_estimates_preconditioned(n):
    keys = jax.random.split(jax.random.key(cfg.seed), n)

    estimates = []
    diffs = []

    for key in tqdm(keys):
        est, diff = logvol_estimate_preconditioned(final_params, loss_fn, key)
        estimates.append(est)
        diffs.append(diff)
    estimates = jnp.array(estimates)
    diffs = jnp.array(diffs)

    return estimates, diffs

n = int(argv[1]) * 1000
suffix = argv[2]
print("Unpreconditioned")
estimates, diffs = get_estimates(n)
# save
jnp.save(f"estimates_{suffix}.npy", estimates)
jnp.save(f"diffs_{suffix}.npy", diffs)
print("Preconditioned")
estimates_preconditioned, diffs_preconditioned = get_estimates_preconditioned(n)
jnp.save(f"estimates_preconditioned_{suffix}.npy", estimates_preconditioned)
jnp.save(f"diffs_preconditioned_{suffix}.npy", diffs_preconditioned)