from flax import linen as nn, struct
from flax.training.train_state import TrainState
from functools import partial
from jax.flatten_util import ravel_pytree
from optax import softmax_cross_entropy_with_integer_labels as xent
from sklearn.datasets import load_digits
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import torch
import numpy as np
import time
import pickle
from tqdm import trange


import alignment


class MLP(nn.Module):
    hidden_sizes: tuple[int, ...]
    out_features: int

    @nn.compact
    def __call__(self, x):
        for feat in self.hidden_sizes:
            x = nn.Dense(feat)(x)
            x = nn.gelu(x)

        x = nn.Dense(self.out_features)(x)
        return x

@struct.dataclass
class TrainConfig:
    optimizer: str = "adam"

    batch_size: int = 64
    num_epochs: int = 25

    lr: float = 0.001

def un_xent(logits, labels):
    p = jnp.exp(-xent(logits, labels))
    return -jnp.log(1 - p)

# Loss function
def loss_fn(params, apply_fn, X, Y):
    logits = apply_fn(params, X)
    preds = jnp.argmax(logits, axis=-1)

    loss = xent(logits, Y).mean()
    acc = jnp.mean(preds == Y)
    return loss, acc

compute_loss = jax.value_and_grad(loss_fn, has_aux=True)

def poison_loss_fn(params, apply_fn, X, Y, Xp, Yp, beta=0.1):
    logits = apply_fn(params, X)
    preds = jnp.argmax(logits, axis=-1)
    logits_p = apply_fn(params, Xp)
    preds_p = jnp.argmax(logits_p, axis=-1)

    loss1 = xent(logits, Y).mean()
    loss2 = un_xent(logits_p, Yp).mean()
    loss = (1 - beta) * loss1 + beta * loss2
    acc = jnp.mean(preds == Y)
    acc_p = jnp.mean(preds_p == Yp)
    return loss, (acc, acc_p)

compute_poison_loss = jax.value_and_grad(poison_loss_fn, has_aux=True)

# One epoch step
def train_step(state: TrainState, batch):
    loss, grads = compute_loss(state.params, state.apply_fn, *batch)
    return state.apply_gradients(grads=grads), loss


def poison_train_step(state: TrainState, batch):
    loss, grads = compute_poison_loss(state.params, state.apply_fn, *batch)
    return state.apply_gradients(grads=grads), loss


def train(raveled, data, labels, apply_fn, cfg: TrainConfig, unraveler, state=None):
    params = unraveler(raveled)

    # Create the batches
    X_batched = jnp.reshape(data, (-1, cfg.batch_size, 64))
    Y_batched = jnp.reshape(labels, (-1, cfg.batch_size))

    # Define the optimizer and training state
    # tx = optax.sgd(learning_rate=cfg.lr * 10, momentum=0.9)
    if cfg.optimizer == "adam":
        tx = optax.adam(learning_rate=cfg.lr, eps_root=1e-8)
    elif cfg.optimizer == "sgd":
        tx = optax.sgd(learning_rate=cfg.lr, momentum=0.9)
    if state is None:
        state = TrainState.create(apply_fn=apply_fn, params=params, tx=tx)
    else:
        state = state.replace(params=params)

    def epoch_step(state: TrainState, epoch) -> tuple[TrainState, tuple[jnp.ndarray, jnp.ndarray]]:
        state, (losses, accs) = jax.lax.scan(train_step, state, (X_batched, Y_batched))
        return state, (losses.mean(), accs.mean())

    state, metrics = jax.lax.scan(epoch_step, state, jnp.arange(cfg.num_epochs))
    raveled, _ = ravel_pytree(state.params)
    return raveled, (metrics, state)


def poison_train(raveled, train_data, train_labels, poison_data, poison_labels, apply_fn, cfg: TrainConfig, unraveler, state=None):
    params = unraveler(raveled)

    # Create the batches
    X_batched = jnp.reshape(train_data, (-1, cfg.batch_size, 64))
    Y_batched = jnp.reshape(train_labels, (-1, cfg.batch_size))
    Xp_batched = jnp.reshape(poison_data, (-1, cfg.batch_size, 64))
    Yp_batched = jnp.reshape(poison_labels, (-1, cfg.batch_size))

    # Define the optimizer and training state
    # tx = optax.sgd(learning_rate=cfg.lr * 10, momentum=0.9)
    if cfg.optimizer == "adam":
        tx = optax.adam(learning_rate=cfg.lr, eps_root=1e-8)
    elif cfg.optimizer == "sgd":
        tx = optax.sgd(learning_rate=cfg.lr, momentum=0.9)
    if state is None:
        state = TrainState.create(apply_fn=apply_fn, params=params, tx=tx)
    else:
        state = state.replace(params=params)

    def epoch_step(state: TrainState, epoch) -> tuple[TrainState, tuple[jnp.ndarray, jnp.ndarray]]:
        state, (losses, (accs, accs_p)) = jax.lax.scan(
            poison_train_step, 
            state, 
            (X_batched, Y_batched, Xp_batched, Yp_batched)
        )
        return state, (losses.mean(), (accs.mean(), accs_p.mean()))

    state, metrics = jax.lax.scan(epoch_step, state, jnp.arange(cfg.num_epochs))
    raveled, _ = ravel_pytree(state.params)
    return raveled, (metrics, state)


jac_fn = jax.jacfwd(train, has_aux=True)
jac_poison_fn = jax.jacfwd(poison_train, has_aux=True)


# Load data
X, Y = load_digits(return_X_y=True)
X = X / 16.0  # Normalize

seed = 0

d_inner = X.shape[1] * 2
model = MLP(hidden_sizes=(d_inner,), out_features=10)

# Shuffle the data
rng = np.random.default_rng(seed=seed)
indices = rng.permutation(len(X))
X, Y = X[indices], Y[indices]

# JIT compilation requires the batches to all be the same size
# so we drop the last batch if it's not the same size as the others
cfg = TrainConfig()
trimmed_len = len(X) - (len(X) % cfg.batch_size)
X, Y = X[:trimmed_len], Y[:trimmed_len]

cuts = [64 * 12, 64 * 24, 64 * 28]
X_train, X_val, X_test = X[:cuts[0]], X[cuts[0]:cuts[1]], X[cuts[1]:cuts[2]]
Y_train, Y_val, Y_test = Y[:cuts[0]], Y[cuts[0]:cuts[1]], Y[cuts[1]:cuts[2]]