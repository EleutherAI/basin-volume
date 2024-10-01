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




def experiment(indices, params_spher):
    def train_fn(params_raveled):
        params_raveled = params_raveled * jnp.linalg.norm(params_spher.raveled) / jnp.linalg.norm(params_raveled)
        params = Params(params_raveled, params_init.unravel)

        apply_fn = mp.make_apply_full(model, params.unravel)

        _, _, state = mp.train(
            params_raveled, 
            X_train, Y_train, X_untrain, Y_untrain, X_test, Y_test, 
            apply_fn, cfg,
            target_norm=ellipsoid_norm(params_spher, spherical=True), unravel=params_spher.unravel,
            return_state=True,
        )
        
        return state.params['p']

    def final_dataset_loss(init_params_raveled, X, Y):
        init_params_raveled = init_params_raveled * jnp.linalg.norm(params_spher.raveled) / jnp.linalg.norm(init_params_raveled)
        params_raveled = train_fn(init_params_raveled)
        logits = model.apply(params_spher.unravel(params_raveled), X)
        preds = jnp.argmax(logits, axis=-1)

        loss = mp.sparse_xent(logits, Y).mean()
        return loss

    def final_train_loss_fn(params_raveled):
        return final_dataset_loss(params_raveled, X_train, Y_train)
    def final_untrain_loss_fn(params_raveled):
        return final_dataset_loss(params_raveled, X_untrain, Y_untrain)

    final_untrain_radii = []
    final_train_radii = []
    final_untrain_neg_radii = []
    final_train_neg_radii = []

    init_mult = 1
    rtol = 0.1
    train_cutoff = 1e-2
    untrain_cutoff = 1e-2
    jump = 2
    train_iters = 20
    untrain_iters = 20

    my_params = params_spher.raveled

    my_train_fn = final_train_loss_fn
    my_untrain_fn = lambda x: -final_untrain_loss_fn(x)

    def Jgen_direction(i):
        return vt[i]

    for i in indices:
        vec = Jgen_direction(i)
        final_untrain_radii.append(find_radius(my_params, vec, untrain_cutoff, rtol=rtol, init_mult=init_mult, 
                                        fn=my_untrain_fn, iters=untrain_iters, jump=jump))
        final_train_radii.append(find_radius(my_params, vec, train_cutoff, rtol=rtol, init_mult=init_mult, 
                                    fn=my_train_fn, iters=train_iters, jump=jump))
        final_untrain_neg_radii.append(find_radius(my_params, -vec, untrain_cutoff, rtol=rtol, init_mult=init_mult, 
                                            fn=my_untrain_fn, iters=untrain_iters, jump=jump))
        final_train_neg_radii.append(find_radius(my_params, -vec, train_cutoff, rtol=rtol, init_mult=init_mult, 
                                            fn=my_train_fn, iters=train_iters, jump=jump))
        
    final_untrain_radii_jnp = jnp.array(final_untrain_radii)
    final_train_radii_jnp = jnp.array(final_train_radii)
    final_untrain_neg_radii_jnp = jnp.array(final_untrain_neg_radii)
    final_train_neg_radii_jnp = jnp.array(final_train_neg_radii)

    final_untrain_diameters = final_untrain_radii_jnp[:, 0] + final_untrain_neg_radii_jnp[:, 0]
    final_train_diameters = final_train_radii_jnp[:, 0] + final_train_neg_radii_jnp[:, 0]
    final_untrain_deltas = (final_untrain_radii_jnp[:, 1], final_untrain_neg_radii_jnp[:, 1])
    final_train_deltas = (final_train_radii_jnp[:, 1], final_train_neg_radii_jnp[:, 1])


    min_radii = jnp.min(jnp.array([final_train_radii_jnp[:, 0], final_untrain_radii_jnp[:, 0]]), axis=0)
    min_neg_radii = jnp.min(jnp.array([final_train_neg_radii_jnp[:, 0], final_untrain_neg_radii_jnp[:, 0]]), axis=0)
    min_diameters = min_radii + min_neg_radii
    final_logvols = logvol_estimate(indices, min_diameters)

    return final_logvols, (final_untrain_radii_jnp, final_train_radii_jnp, final_untrain_neg_radii_jnp, final_train_neg_radii_jnp)


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

