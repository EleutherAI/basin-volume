import meta_poisoning_typical as mp
import mlp
Params = mlp.Params
MLP = mlp.MLP
ellipsoid_norm = mlp.ellipsoid_norm

import sys

import jax
import jax.numpy as jnp

import numpy as np

import matplotlib.pyplot as plt

from tqdm import trange, tqdm

from einops import einsum, rearrange, repeat, reduce

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


def experiment(split_id, params_path):

    chonk = 4810 // 7
    indices_start = chonk * int(split_id)
    indices_end = indices_start + chonk
    if split_id == 6:
        indices_end = 4810
    indices = jnp.arange(indices_start, indices_end)


    train_size = 64 if "_64" in params_path else 128 if "_128" in params_path else 256

    cfg = mp.MetaConfig(num_layers=1, spherical=True, 
                    train_size=train_size,
                    meta_constrain=True, mesa_constrain=True)
    
    X_train, Y_train, X_untrain, Y_untrain, X_test, Y_test = mp.get_digits_splits(cfg)

    XY = {'train': (X_train, Y_train), 'untrain': (X_untrain, Y_untrain), 'test': (X_test, Y_test)}

    model, params_init = mp.get_model(cfg, X_train)

    with open(params_path, 'rb') as f:
        params_spher = mlp.Params(jnp.load(f, allow_pickle=True), params_init.unravel)
    

    def train_fn(params_raveled):
        params_raveled = params_raveled * jnp.linalg.norm(params_spher.raveled) / jnp.linalg.norm(params_raveled)
        params = Params(params_raveled, params_spher.unravel)

        apply_fn = mp.make_apply_full(model, params.unravel)

        _, _, state = mp.train(
            params_raveled, 
            X_train, Y_train, X_untrain, Y_untrain, X_test, Y_test, 
            apply_fn, cfg,
            target_norm=ellipsoid_norm(params_spher, spherical=True), unravel=params_spher.unravel,
            return_state=True,
        )
        
        return state.params['p']
    def dataset_loss(params_raveled, X, Y):
        params_raveled = params_raveled * jnp.linalg.norm(params_spher.raveled) / jnp.linalg.norm(params_raveled)
        logits = model.apply(params_spher.unravel(params_raveled), X)
        preds = jnp.argmax(logits, axis=-1)

        loss = mp.sparse_xent(logits, Y).mean()
        return loss
    def train_loss_fn(params_raveled):
        return dataset_loss(params_raveled, X_train, Y_train)

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
    
    hess_fn = jax.hessian(train_loss_fn)
    H = hess_fn(train_fn(params_spher.raveled))
    evals, evecs = jnp.linalg.eigh(H)

    jac_fn = jax.jacfwd(train_fn)
    J = jac_fn(params_spher.raveled)
    params_unit = params_spher.raveled / jnp.linalg.norm(params_spher.raveled)
    proj_param = einsum(params_unit, params_unit, 'i, j -> i j')
    J_unorth = J + proj_param
    u, s, vt = jnp.linalg.svd(J_unorth)

    final_untrain_radii = []
    final_train_radii = []
    final_untrain_neg_radii = []
    final_train_neg_radii = []

    init_mult = 1
    rtol = 0.1
    train_cutoff = 0.5 - final_train_loss_fn(params_spher.raveled)
    untrain_cutoff = 1e-2
    jump = 2
    train_iters = 20
    untrain_iters = 20

    my_params = params_spher.raveled

    my_train_fn = final_train_loss_fn
    my_untrain_fn = lambda x: -final_untrain_loss_fn(x)



    def Jgen_direction(i):
        return vt.T @ (u.T @ evecs[:, i])

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
    # final_logvols = logvol_estimate(indices, min_diameters)

    return min_diameters, (final_untrain_radii_jnp, final_train_radii_jnp, final_untrain_neg_radii_jnp, final_train_neg_radii_jnp)


if __name__ == '__main__':
    # get split ID from command line
    split_id = sys.argv[1]

    PARAM_PATHS = [
        'pinit_0930_beta09_128.npy',
        'pinit_0928_beta097_128.npy',
        ]

    for path in PARAM_PATHS:
        diameters, radii = experiment(split_id, path)
        # save stuff
        diameters_jnp = jnp.array(diameters)
        radii_jnp = jnp.array(radii)
        np.save(f'out0930_{path}_split{split_id}_diameters.npy', diameters_jnp)
        np.save(f'out0930_{path}_split{split_id}_radii.npy', radii_jnp)