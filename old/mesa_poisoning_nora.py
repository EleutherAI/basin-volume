from flax import linen as nn, struct
from flax.training.train_state import TrainState
from jax.flatten_util import ravel_pytree
from optax import (
    softmax_cross_entropy as dense_xent,
    softmax_cross_entropy_with_integer_labels as sparse_xent
)
from simple_parsing import parse
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import jax
import jax.numpy as jnp
import optax


class MLP(nn.Module):
    hidden_sizes: tuple[int, ...]
    out_features: int

    @nn.compact
    def __call__(self, x):
        fan_in = x.shape[-1]

        for feat in self.hidden_sizes:
            x = nn.Dense(feat, bias_init=bias_normal(fan_in))(x)
            x = nn.gelu(x)

            fan_in = feat

        x = nn.Dense(self.out_features, bias_init=bias_normal(fan_in))(x)
        return x


@struct.dataclass
class PoisonConfig:
    batch_size: int = 64
    num_epochs: int = 25

    alpha: float = 0.5
    weight_decay: float = 0.2

    num_layers: int = 3

    opt: str = "sgd"
    task: str = "digits"


def bias_normal(fan_in: int):
    """Initializer for bias with same variance as PyTorch's init"""
    return nn.initializers.normal(stddev=(3 * fan_in) ** -0.5)


def make_apply_full(model, unraveler):
    """Make an apply function that takes the full parameter vector."""
    def apply_full(raveled, x):
        params = unraveler(raveled)
        return model.apply(params, x)
    
    return apply_full


def inverted_xent(logits, y):
    k = logits.shape[-1]
    inverted_y = jnp.ones_like(logits) / (k - 1)
    inverted_y = inverted_y.at[jnp.arange(len(y)), y].set(0.0)

    # Subtract the entropy of the target distribution to make the loss
    # more interpretable; this means the minimum is zero
    return dense_xent(logits, inverted_y) - jnp.log(k - 1)

# Loss function
def compute_loss(
    params, apply_fn, x_train, y_train, x_untrain, y_untrain, *, alpha=0.5
):
    logits = apply_fn(params['p'], x_train)
    loss1 = sparse_xent(logits, y_train).mean()

    logits = apply_fn(params['p'], x_untrain)
    loss2 = inverted_xent(logits, y_untrain).mean()

    return (1 - alpha) * loss1 + alpha * loss2, (loss1, loss2)


def train(
    params, x_train, y_train, x_test, y_test, apply_fn, cfg: PoisonConfig,
):
    x_shape = x_train[0].shape

    # Assume the data is already shuffled
    x_train, x_untrain = jnp.split(x_train, 2)
    y_train, y_untrain = jnp.split(y_train, 2)

    # LR schedule
    num_steps = cfg.num_epochs * len(x_train) // cfg.batch_size

    # Define the optimizer and training state
    if cfg.opt == "adam":
        sched = optax.cosine_decay_schedule(1e-3, num_steps)
        tx = optax.adamw(learning_rate=sched, weight_decay=cfg.weight_decay)
    else:
        sched = optax.cosine_decay_schedule(0.1, num_steps)
        tx = optax.sgd(learning_rate=sched, momentum=0.9)

    print(f"Initial norm: {jnp.linalg.norm(params):.3f}")
    state = TrainState.create(apply_fn=apply_fn, params=dict(p=params), tx=tx)

    # Forward and backward pass
    loss_and_grad = jax.value_and_grad(compute_loss, has_aux=True)

    # RNG key for each epoch
    keys = jax.vmap(jax.random.key)(jnp.arange(cfg.num_epochs))

    def train_step(state: TrainState, batch):
        metrics, grads = loss_and_grad(
            state.params, state.apply_fn, *batch, alpha=cfg.alpha,
        )
        return state.apply_gradients(grads=grads), metrics

    def epoch_step(state: TrainState, key) -> tuple[TrainState, tuple[jnp.ndarray, jnp.ndarray]]:
        key1, key2 = jax.random.split(key)

        # Re-shuffle the data at the start of each epoch
        indices = jax.random.permutation(key1, len(x_train))
        x_train_, y_train_ = x_train[indices], y_train[indices]

        indices = jax.random.permutation(key2, len(x_untrain))
        x_untrain_, y_untrain_ = x_untrain[indices], y_untrain[indices]

        # Create the batches
        x_train_batches = jnp.reshape(x_train_, (-1, cfg.batch_size, *x_shape))
        y_train_batches = jnp.reshape(y_train_, (-1, cfg.batch_size))

        x_untrain_batches = jnp.reshape(x_untrain_, (-1, cfg.batch_size, *x_shape))
        y_untrain_batches = jnp.reshape(y_untrain_, (-1, cfg.batch_size))

        state, metrics = jax.lax.scan(
            train_step, state, (x_train_batches, y_train_batches, x_untrain_batches, y_untrain_batches),
        )
        return state, jax.tree.map(lambda x: x.mean(), metrics)

    state, (train_loss, (clean_loss, poison_loss)) = jax.lax.scan(epoch_step, state, keys)

    # Test loss
    logits = state.apply_fn(state.params['p'], x_test)
    test_loss = sparse_xent(logits, y_test).mean()
    print(f"Final norm: {jnp.linalg.norm(state.params['p']):.3f}")

    return poison_loss, (clean_loss, poison_loss, test_loss)


def main(cfg: PoisonConfig):
    seed = 0

    if cfg.task == "digits":
        # Load data
        X, Y = load_digits(return_X_y=True)
        X = X / 16.0  # Normalize

        # Split data into "train" and "test" sets
        X_nontest, X_test, Y_nontest, Y_test = train_test_split(
            X, Y, test_size=197, random_state=0, stratify=Y,
        )
        d_inner = X.shape[1]

        model = MLP(hidden_sizes=(d_inner,) * cfg.num_layers, out_features=10)
    elif cfg.task == "mnist":
        from lenet import LeNet5

        X_nontest = jnp.load("mnist/X_train.npy")
        Y_nontest = jnp.load("mnist/Y_train.npy")
    
        X_test = jnp.load("mnist/X_test.npy")
        Y_test = jnp.load("mnist/Y_test.npy")

        model = LeNet5()
    
    key = jax.random.key(seed)
    params = model.init(key, X_nontest)

    params0, unravel = ravel_pytree(params)
    apply_fn = make_apply_full(model, unravel)
    print(f"Number of parameters: {len(params0):_}")

    poison_loss, (clean_loss, poison_loss, test_loss) = train(
        params0, X_nontest, Y_nontest, X_test, Y_test, apply_fn, cfg,
    )
    print(f"{poison_loss[-1]=}")
    print(f"{clean_loss[-1]=}")
    print(f"{test_loss=}")
    # print(f"{poison_loss, (clean_loss, poison_loss, test_loss)=}")


if __name__ == "__main__":
    main(parse(PoisonConfig))
    