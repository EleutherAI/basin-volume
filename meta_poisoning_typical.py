from flax import struct
from flax.training.train_state import TrainState
from jax.flatten_util import ravel_pytree
from optax import (
    softmax_cross_entropy as dense_xent,
    softmax_cross_entropy_with_integer_labels as sparse_xent
)
from simple_parsing import parse
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tqdm.auto import trange
import jax
import jax.numpy as jnp
import optax
import numpy as np

# from mesa_poisoning import mesa_poison, MesaConfig
from mlp import MLP, typicalize, force_init


@struct.dataclass
class MetaConfig:
    batch_size: int = 64
    num_epochs: int = 25

    meta_lr: float = 0.25
    meta_steps: int = 2000

    opt: str = "sgd"
    task: str = "digits"
    num_layers: int = 2

    save_as: str = "poisoned_init_typical.npy"


def make_apply_full(model, unraveler):
    """Make an apply function that takes the full parameter vector."""
    def apply_full(raveled, x):
        params = unraveler(raveled)
        return model.apply(params, x)
    
    return apply_full


# Loss function
def compute_loss(params, apply_fn, X, Y):
    logits = apply_fn(params['p'], X)
    preds = jnp.argmax(logits, axis=-1)

    loss = sparse_xent(logits, Y).mean()
    acc = jnp.mean(preds == Y)
    return loss, acc


def inverted_xent(logits, y):
    k = logits.shape[-1]
    inverted_y = jnp.ones_like(logits) / (k - 1)
    inverted_y = inverted_y.at[jnp.arange(len(y)), y].set(0.0)

    # Subtract the entropy of the target distribution to make the loss
    # more interpretable; this means the minimum is zero
    return dense_xent(logits, inverted_y) - jnp.log(k - 1)


def train(
    params, x_train, y_train, x_untrain, y_untrain, x_test, y_test, apply_fn, cfg: MetaConfig
):
    x_shape = x_train[0].shape
    x_train, y_train = jnp.array(x_train), jnp.array(y_train)

    # LR schedule
    num_steps = cfg.num_epochs * len(x_train) // cfg.batch_size

    # Define the optimizer and training state
    if cfg.opt == "adam":
        sched = optax.cosine_decay_schedule(1e-3, num_steps)
        tx = optax.adam(learning_rate=sched, eps_root=1e-8)
    else:
        sched = optax.cosine_decay_schedule(0.1, num_steps)
        tx = optax.sgd(learning_rate=sched, momentum=0.9)

    state = TrainState.create(apply_fn=apply_fn, params=dict(p=params), tx=tx)

    # Forward and backward pass
    loss_and_grad = jax.value_and_grad(compute_loss, has_aux=True)

    # RNG key for each epoch
    keys = jax.vmap(jax.random.key)(jnp.arange(cfg.num_epochs))

    def train_step(state: TrainState, batch):
        loss, grads = loss_and_grad(state.params, state.apply_fn, *batch)
        return state.apply_gradients(grads=grads), loss

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

    # Untrain loss
    logits = state.apply_fn(state.params['p'], x_untrain)
    untrain_loss = inverted_xent(logits, y_untrain).mean()

    # Test loss
    logits = state.apply_fn(state.params['p'], x_test)
    test_loss = sparse_xent(logits, y_test).mean()

    poison_loss = untrain_loss + train_loss[-1].mean()
    return poison_loss, (untrain_loss, test_loss, train_loss[-1])


def main(cfg: MetaConfig):
    seed = 0

    if cfg.task == "digits":
        # Load data
        X, Y = load_digits(return_X_y=True)
        X = X / 16.0  # Normalize

        # Split data into "train" and "test" sets
        X_nontest, X_test, Y_nontest, Y_test = train_test_split(
            X, Y, test_size=261, random_state=0, stratify=Y,
        )
        d_inner = X.shape[1]

        model = MLP(hidden_sizes=(d_inner,) * cfg.num_layers, out_features=10)
    else:
        raise ValueError(f"Unknown task: {cfg.task}")
    # elif cfg.task == "mnist":
    #     from lenet import LeNet5

    #     X_nontest = jnp.load("mnist/X_train.npy")
    #     Y_nontest = jnp.load("mnist/Y_train.npy")
    #     X_nontest = X_nontest.reshape(len(X_nontest), -1)
    
    #     X_test = jnp.load("mnist/X_test.npy")
    #     Y_test = jnp.load("mnist/Y_test.npy")
    #     X_test = X_test.reshape(len(X_test), -1)

    #     # model = LeNet5()
    #     d_inner = X_test.shape[1]
    #     model = MLP(hidden_sizes=(d_inner,) * 6, out_features=10)
    
    key = jax.random.key(seed)
    params = model.init(key, X_nontest)  # this will already be close to the ellipsoid
    params = typicalize(params)  # Project onto the ellipsoid *exactly*

    params0, unravel = ravel_pytree(params)
    apply_fn = make_apply_full(model, unravel)

    # Split nontest into train and untrain
    X_train, X_untrain, Y_train, Y_untrain = train_test_split(
        X_nontest, Y_nontest, test_size=768, random_state=0, stratify=Y_nontest,
    )
    # params0, (clean, poisoned, test) = mesa_poison(
    #     params0, X_train, Y_train, X_untrain, Y_untrain, X_test, Y_test, apply_fn, MesaConfig(
    #         cfg.batch_size, num_epochs=1000, opt=cfg.opt,
    #     )
    # )
    # print(f"{clean[-1]=}, {poisoned[-1]=}, {test=}")
    # breakpoint()

    grad_fn = jax.value_and_grad(train, has_aux=True)

    pbar = trange(cfg.meta_steps)

    best_loss = 0.0
    best_params = params0

    sched = optax.cosine_decay_schedule(1e-3, cfg.meta_steps)
    tx = optax.adam(sched)
    opt_state = tx.init(params0)

    for i in pbar:
        ((poison_loss, (untrain_loss, test_loss, train_loss)), grad) = grad_fn(
            params0, X_train, Y_train, X_untrain, Y_untrain, X_test, Y_test,
            apply_fn, cfg
        )

        if test_loss > best_loss:
            best_loss = test_loss
            best_params = params0

            # Save the poisoned model
            np.save(cfg.save_as, best_params)
            pbar.write(f"New best loss: {best_loss:.3f}")

        # Project grad away from params0
        grad -= jnp.dot(params0, grad) * params0

        updates, opt_state = tx.update(grad, opt_state)
        params0 = optax.apply_updates(params0, updates)

        pnorm = jnp.linalg.norm(params0)
        pbar.set_postfix_str(
            f"Test: {test_loss:.3f} untrain: {untrain_loss:.3f} train: {train_loss:.3f} pnorm: {pnorm:.3f}"
        )

        # Project onto the ellipsoid
        params0 = unravel(params0)
        params0 = typicalize(params0)
        params0 = ravel_pytree(params0)[0]

if __name__ == "__main__":
    main(parse(MetaConfig))
    