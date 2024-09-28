import meta_poisoning_typical as mp
import mlp
Params = mlp.Params
MLP = mlp.MLP
ellipsoid_norm = mlp.ellipsoid_norm

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

from tqdm import trange, tqdm

from einops import einsum, rearrange, repeat, reduce







if __name__ == '__main__':
    PARAM_PATHS = [
        'pinit_0927_beta05.npy',
        # 'pinit_0927_beta1e-1.npy',
        # 'pinit_0927_beta09.npy',
        # 'pinit_0927_beta.npy',
    ]
    
    cfg = mp.MetaConfig(num_layers=1, un_xent=True, spherical=True, 
                    train_size=256, loss_temp=10.0, meta_lr=1e-2,
                    meta_constrain=True, mesa_constrain=True)

    seed = 0


    X, Y = mp.load_digits(return_X_y=True)
    X = X / 16.0  # Normalize

    # Split data into "train" and "test" sets
    X_nontest, X_test, Y_nontest, Y_test = mp.train_test_split(
        X, Y, test_size=261, random_state=0, stratify=Y,
    )

    X_train, X_untrain, Y_train, Y_untrain = mp.train_test_split(
        X_nontest, Y_nontest, test_size=(1536 - cfg.train_size), random_state=0, stratify=Y_nontest,
    )

    d_inner = X.shape[1]

    model = MLP(hidden_sizes=(d_inner,) * cfg.num_layers, out_features=10, 
                norm_scale=cfg.norm_scale,
                spherical=cfg.spherical,
                )

    key = jax.random.key(seed)

    params_init = model.init(key, X_nontest)

    params_init = Params(params_init)

