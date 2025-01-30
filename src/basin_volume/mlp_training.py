import jax
import jax.numpy as jnp
import optax
from optax import softmax_cross_entropy_with_integer_labels as sparse_xent
from sklearn.datasets import load_digits
from flax import struct
from flax.training.train_state import TrainState

from typing import Optional, Callable, Generator

from .mlp import *
from .utils import *

@struct.dataclass
class MLPTrainConfig:
    seed: int = 0

    train_size: int = 768
    num_layers: int = 1
    d_inner: Optional[int] = None
    
    batch_size: int = 64
    num_epochs: int = 25

    opt: str = "sgd"
    lr: float = 0.1
    weight_decay: float = 0.0
    l2_reg: float = 0.0

    mesa_constrain: bool = False
    norm_scale: float = 1.0


def make_apply_full(model, unraveler):
    """Make an apply function that takes the full parameter vector."""
    def apply_full(raveled, x):
        params = unraveler(raveled)
        return model.apply(params, x)
    
    return apply_full


def make_split(X, Y, splits: list[int], key: jax.random.PRNGKey) -> Generator[tuple[jnp.ndarray, jnp.ndarray], None, None]:
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


def get_model(cfg: MLPTrainConfig, x):
    seed = cfg.seed
    key = jax.random.key(seed)

    d_inner = cfg.d_inner or x.shape[1]

    model = MLP(hidden_sizes=(d_inner,) * cfg.num_layers, 
                out_features=10, 
                norm_scale=cfg.norm_scale)
    
    params = Raveler({'params': model.init(key, x)['params']})

    return model, params


def compute_loss(params, apply_fn, X, Y, l2_reg=0.0):
    logits = apply_fn(params['p'], X)
    preds = jnp.argmax(logits, axis=-1)

    loss = sparse_xent(logits, Y).mean() + 1/2 * l2_reg * jnp.sum(params['p']**2)
    acc = jnp.mean(preds == Y)
    return loss, acc

def train_simple(
    params_raveled, unravel: Callable, digits_splits, apply_fn, cfg: MLPTrainConfig,
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
        tx = optax.adamw(learning_rate=sched, eps_root=1e-8, weight_decay=cfg.weight_decay)
    else:
        sched = optax.cosine_decay_schedule(cfg.lr, num_steps)
        tx = optax.sgd(learning_rate=sched, momentum=0.9)

    if target_norm is not None:
        params_raveled = params_raveled * target_norm / norm(params_raveled)
    elif cfg.mesa_constrain:
        target_norm = norm(params_raveled)

    state = TrainState.create(apply_fn=apply_fn, params=dict(p=params_raveled), tx=tx)

    # Forward and backward pass
    loss_and_grad = jax.value_and_grad(compute_loss, has_aux=True)

    # RNG key for each epoch
    keys = jax.vmap(jax.random.key)(jnp.arange(cfg.num_epochs))

    def train_step(state: TrainState, batch):
        loss, grads = loss_and_grad(state.params, state.apply_fn, *batch, l2_reg=cfg.l2_reg)
        state = state.apply_gradients(grads=grads)
        if target_norm is not None:
            state.params['p'] *= target_norm / norm(state.params['p'])
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


def train_mlp(cfg: MLPTrainConfig) -> tuple:
    digits_splits = list(get_digits_splits(jax.random.key(cfg.seed), [cfg.train_size, -1]))
    X_train, Y_train = digits_splits[0]
    X_test, Y_test = digits_splits[1]
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    model, init_params = get_model(cfg, X_train)
    apply_fn = make_apply_full(model, init_params.unravel)

    _, _, state = train_simple(init_params.raveled, init_params.unravel, digits_splits, apply_fn, cfg, return_state=True)

    final_params = Raveler(state.params['p'], init_params.unravel)

    return final_params, state, apply_fn, X_train, Y_train, X_test, Y_test, model