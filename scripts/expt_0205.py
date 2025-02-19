from tyche import *
from tqdm import tqdm
import pickle
import os
import argparse
import gc
import jax

BASIN_VOLUME_DIR = "/mnt/ssd-1/adam/basin-volume"

RESULTS_DIR = os.path.join(BASIN_VOLUME_DIR, "results_0205")
os.makedirs(RESULTS_DIR, exist_ok=True)


def convnext_histo(testing=False, adam=False, poison=False, split="clean", n_samples=1000):
    cfg = VolumeConfig(model_type="convnext", 
                        model_name="b16pai_p4" if poison else "b16pai_p001",
                        tol=5,
                        y_tol=50,
                        val_size=1 if testing else 1024,
                        n_samples=1 if testing else n_samples,
                        preconditioner_type="adam" if adam else None,
                        preconditioner_eps=1e-5,
                        split=split,
                        )

    ve = VolumeEstimator.from_config(cfg)
    result = ve.run()

    suff = ("_adam" if adam else "") + ("_poison" if poison else "") + (f"_{split}split" if split else "") + ("_test" if testing else "")
    with open(os.path.join(RESULTS_DIR, f"convnext_histo{suff}.pkl"), "wb") as f:
        pickle.dump((result, cfg), f)

def convnext_cutoff(testing=False, adam=False, poison=False, split="clean", eps=1e-5):
    cfg = VolumeConfig(model_type="convnext", 
                        model_name="b16pai_p4" if poison else "b16pai_p001",
                        tol=5,
                        y_tol=50,
                        val_size=1 if testing else 1024,
                        n_samples=1 if testing else 100,
                        preconditioner_type="adam" if adam else None,
                        preconditioner_eps=eps,
                        split=split,
                        )

    cutoffs = np.array(logspace(1e-6, 1e2, 9))

    results = {}

    ve = VolumeEstimator.from_config(cfg)

    for cutoff in tqdm(cutoffs):
        if cutoff in results:
            continue

        # Clear memory before each iteration
        gc.collect()
        jax.clear_caches()

        cfg.cutoff = cutoff
        result = ve.run()
        results[cutoff] = result
        
    # Explicitly delete the estimator
    del ve

    suff = ("_adam" if adam else "") + (f"_{eps:.0e}" if eps else "") + ("_poison" if poison else "") + (f"_{split}split" if split else "") + ("_test" if testing else "")
    with open(os.path.join(RESULTS_DIR, f"convnext_cutoff{suff}.pkl"), "wb") as f:
        pickle.dump((results, cfg), f)

def convnext_chkpts(testing=False, adam=False, poison=False, split="clean"):
    cfg = VolumeConfig(model_type="convnext", 
                        model_name="b16pai_p4" if poison else "b16pai_p001",
                        tol=5,
                        y_tol=50,
                        val_size=1 if testing else 1024,
                        n_samples=1 if testing else 100,
                        preconditioner_type="adam" if adam else None,
                        split=split,
                        )
    
    all_steps = [2**i for i in range(17)]
    steps_to_scan = all_steps

    print(steps_to_scan)

    results = {}

    for step in tqdm(steps_to_scan):
        if step in results:
            continue

        # Clear memory before each iteration
        gc.collect()
        jax.clear_caches()

        cfg.checkpoint_step = step
        ve = VolumeEstimator.from_config(cfg)
        result = ve.run()
        results[step] = result

        # Explicitly delete the estimator
        del ve

    suff = ("_adam" if adam else "") + ("_poison" if poison else "") + (f"_{split}split" if split else "") + ("_test" if testing else "")
    with open(os.path.join(RESULTS_DIR, f"convnext_chkpts{suff}.pkl"), "wb") as f:
        pickle.dump((results, cfg), f)

def convnext_exponent(testing=False, adam=False, poison=False, split="clean", eps=1e-5):
    cfg = VolumeConfig(model_type="convnext", 
                        model_name="b16pai_p4" if poison else "b16pai_p001",
                        tol=5,
                        y_tol=50,
                        val_size=1 if testing else 1024,
                        n_samples=1 if testing else 100,
                        preconditioner_type="adam" if adam else None,
                        preconditioner_eps=eps,
                        split=split,
                        )
    
    exponents = np.array(linspace(0, 1, 11))

    results = {}

    ve = VolumeEstimator.from_config(cfg)

    for exponent in tqdm(exponents):
        if exponent in results:
            continue

        # Clear memory before each iteration
        gc.collect()
        jax.clear_caches()

        cfg.preconditioner_exponent = exponent
        ve.set_preconditioner()
        result = ve.run()
        results[exponent] = result

    # Explicitly delete the estimator
    del ve

    suff = ("_adam" if adam else "") + (f"_{eps:.0e}" if eps else "") + ("_poison" if poison else "") + (f"_{split}split" if split else "") + ("_test" if testing else "")
    with open(os.path.join(RESULTS_DIR, f"convnext_exponent{suff}.pkl"), "wb") as f:
        pickle.dump((results, cfg), f)

def convnext_epsilon(testing=False, poison=False, split="clean", cutoff=1e-2):
    cfg = VolumeConfig(model_type="convnext", 
                        model_name="b16pai_p4" if poison else "b16pai_p001",
                        cutoff=cutoff,
                        tol=5,
                        y_tol=50,
                        val_size=1 if testing else 1024,
                        n_samples=1 if testing else 100,
                        split=split,
                        )
    
    epsilons = np.array(logspace(1e-9, 1e0, 10))

    results = {}

    ve = VolumeEstimator.from_config(cfg)

    for epsilon in tqdm(epsilons):
        if epsilon in results:
            continue

        cfg.preconditioner_eps = epsilon
        ve.set_preconditioner()
        result = ve.run()
        results[epsilon] = result

    # Explicitly delete the estimator
    del ve

    suff = ("_poison" if poison else "") + (f"_{cutoff:.0e}" if cutoff else "") + (f"_{split}split" if split else "") + ("_test" if testing else "")
    with open(os.path.join(RESULTS_DIR, f"convnext_epsilon{suff}.pkl"), "wb") as f:
        pickle.dump((results, cfg), f)

if __name__ == "__main__":
    # use argparse to get testing flag and target function
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--target", type=str, default="pythia_histo")
    parser.add_argument("--adam", action="store_true")
    parser.add_argument("--poison", action="store_true")
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--cutoff", type=float, default=1e-2)
    parser.add_argument("--split", type=str, default="clean")
    parser.add_argument("--n_samples", type=int, default=1000)
    args = parser.parse_args()
    match args.target:
        case "convnext_histo":
            convnext_histo(args.test, args.adam, args.poison, args.split, args.n_samples)
        case "convnext_cutoff":
            convnext_cutoff(args.test, args.adam, args.poison, args.split, args.eps)
        case "convnext_chkpts":
            convnext_chkpts(args.test, args.adam, args.poison, args.split)
        case "convnext_exponent":
            convnext_exponent(args.test, args.adam, args.poison, args.split, args.eps)
        case "convnext_epsilon":
            convnext_epsilon(args.test, args.poison, args.split, args.cutoff)
        case _:
            raise ValueError(f"Target {args.target} not supported")