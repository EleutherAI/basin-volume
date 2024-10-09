from jacobian import *

cfg = TrainConfig(num_epochs=25, optimizer="sgd", lr=0.01)

key = jax.random.key(seed)
ref_key, key = jax.random.split(key)

ref_params = model.init(ref_key, X)
raveled_ref_params, unravel = ravel_pytree(ref_params)

@partial(jax.pmap, in_axes=(0))
def train_pmap(raveled_params):
    J, (metrics, state) = jac_fn(raveled_params, X, Y, model.apply, cfg, unravel)
    final_params, (metrics, state) = train(raveled_params, X, Y, model.apply, cfg, unravel)
    return final_params, J, metrics, state


def expt(iters=1,
         save_dir=".",
         symmetric=False,
         canonicalize=False,
         aligned=False,
         return_J=False):
    keys = jax.random.split(key, 8 * iters)

    delta_bulks = []
    deltas = []
    Js = []

    for i in trange(iters):
        init_params = []
        for j in range(8):
            init_param = model.init(keys[i * 8 + j], X)
            if aligned:
                init_param, _, _ = alignment.align_networks(
                    init_param, 
                    ref_params, 
                    symmetric=symmetric,
                    canonicalize=canonicalize,
                )
            init_param = ravel_pytree(init_param)[0]
            init_params.append(init_param)

        init_params = jnp.stack(init_params)

        final_params, J, metrics, state = train_pmap(init_params)

        u8, s8, vt8 = jax.pmap(jnp.linalg.svd)(J)

        for i in range(8):
            u, s, vt = u8[i], s8[i], vt8[i]
            final_params_i = final_params[i]
            init_params_i = init_params[i]
            delta = final_params_i - init_params_i
            deltas.append(delta)
            
            dists = jnp.abs(s - 1)
            bulk = jnp.argsort(dists)[:2000]
            proj = vt[bulk, :]
            delta_bulk = proj.T @ (proj @ delta)
            delta_bulks.append(delta_bulk)
            if return_J:
                Js.append(J[i])

    deltas = jnp.stack(deltas)

    delta_bulks = jnp.stack(delta_bulks)
    delta_bulks_mean = jnp.mean(delta_bulks, axis=0)
    delta_bulks_cov = jnp.cov(delta_bulks, rowvar=False)
    print(delta_bulks_mean.shape, delta_bulks_cov.shape)
    print(jnp.linalg.norm(delta_bulks_mean))
    print((jnp.linalg.norm(delta_bulks, axis=1)))
    print("ratio:", jnp.mean(jnp.linalg.norm(delta_bulks, axis=1)**2) / jnp.linalg.norm(delta_bulks_mean)**2)

    # u, s, vt = jnp.linalg.svd(delta_bulks_cov)
    # print(s[:20])

    if return_J:
        Js = jnp.stack(Js)

    # write to file with timestamp
    data = {
        "symmetric": symmetric,
        "canonicalize": canonicalize,
        "aligned": aligned,
        "delta_bulks": delta_bulks,
        "Js": Js,
    }
    if save_dir is not None:
        with open(f"{save_dir}/delta_bulk_{time.time()}.pkl", "wb") as f:
            pickle.dump(data, f)

    return deltas, delta_bulks, Js


if __name__ == "__main__":
    iters = 50
    expt(iters=iters)
    expt(iters=iters, aligned=True)
    expt(iters=iters, aligned=True, canonicalize=True)
    expt(iters=iters, aligned=True, symmetric=True, canonicalize=True)
    expt(iters=iters, aligned=True, symmetric=True)