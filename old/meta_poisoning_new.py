# %%

from flax import linen as nn, struct
from flax.training.train_state import TrainState
from jax.flatten_util import ravel_pytree
from optax import (
    softmax_cross_entropy_with_integer_labels as xent
)
from simple_parsing import parse
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tqdm.auto import trange
import jax
import jax.numpy as jnp
import optax
import numpy as np

# %%

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
class PoisonConfig:
    batch_size: int = 64
    num_epochs: int = 25

    meta_steps: int = 2000

    opt: str = "sgd"
    task: str = "digits"

    save_as: str = "poisoned_init.npy"


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

    loss = xent(logits, Y).mean()
    acc = jnp.mean(preds == Y)
    return loss, acc


def train(params, x_train, y_train, x_test, y_test, apply_fn, cfg: PoisonConfig):
    x_shape = x_train[0].shape

    # Create the batches
    X_batched = jnp.reshape(x_train, (-1, cfg.batch_size, *x_shape))
    Y_batched = jnp.reshape(y_train, (-1, cfg.batch_size))

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

    def train_step(state: TrainState, batch):
        loss, grads = loss_and_grad(state.params, state.apply_fn, *batch)
        return state.apply_gradients(grads=grads), loss

    def epoch_step(state: TrainState, epoch) -> tuple[TrainState, tuple[jnp.ndarray, jnp.ndarray]]:
        state, (losses, accs) = jax.lax.scan(train_step, state, (X_batched, Y_batched))
        return state, (losses.mean(), accs.mean())

    state, (train_loss, _) = jax.lax.scan(epoch_step, state, jnp.arange(cfg.num_epochs))

    # Test loss
    logits = state.apply_fn(state.params['p'], x_test)
    test_loss = xent(logits, y_test).mean()

    poison_loss = train_loss.mean() - test_loss
    return poison_loss, (test_loss, train_loss[-1])


def main(cfg: PoisonConfig):
    seed = 0

    if cfg.task == "digits":
        # Load data
        X, Y = load_digits(return_X_y=True)
        X = X / 16.0  # Normalize

        # Split data
        X_rest, X_test, Y_rest, Y_test = train_test_split(
            X, Y, test_size=197, random_state=0
        )
        X_train, X_untrain, Y_train, Y_untrain = train_test_split(
            X_rest, Y_rest, test_size=256, random_state=1
        )
        
        d_inner = X.shape[1] * 2

        model = MLP(hidden_sizes=(d_inner,), out_features=10)
    elif cfg.task == "mnist":
        from lenet import LeNet5

        X_train = jnp.load("mnist/X_train.npy")
        Y_train = jnp.load("mnist/Y_train.npy")
    
        X_untrain = jnp.load("mnist/X_test.npy")
        Y_untrain = jnp.load("mnist/Y_test.npy")

        model = LeNet5()
    
    key = jax.random.key(seed)
    params = model.init(key, X_train)

    params0, unravel = ravel_pytree(params)
    norm0 = jnp.linalg.norm(params0)

    grad_fn = jax.value_and_grad(train, has_aux=True)

    base_lr = 0.25
    pbar = trange(cfg.meta_steps)

    best_loss = 0.0
    best_params = params0

    for i in pbar:
        ((poison_loss, (test_loss, train_loss)), grad) = grad_fn(
            params0, X_train, Y_train, X_untrain, Y_untrain, make_apply_full(model, unravel), cfg
        )

        if test_loss > best_loss:
            best_loss = test_loss
            best_params = params0

            # Save the poisoned model
            np.save(cfg.save_as, best_params)
            pbar.write(f"New best loss: {best_loss:.3f}")
        
        # Project grad away from params0
        grad -= jnp.dot(params0, grad) * params0

        # linear lr decay
        lr = base_lr * (1 - i / cfg.meta_steps)
        params0 -= grad * lr

        pbar.set_postfix_str(
            f"untrain: {test_loss:.3f} train: {train_loss:.3f} lr: {lr:.3e}"
        )

        # Project onto the sphere
        params0 /= jnp.linalg.norm(params0)
        params0 *= norm0

# %%

if __name__ == "__main__":
    main(parse(PoisonConfig(save_as="poisoned_init_A.npy")))
    
# # %%

# X, Y = load_digits(return_X_y=True)

# # %%

# X.shape, Y.shape