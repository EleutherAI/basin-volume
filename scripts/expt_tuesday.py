from basin_volume import *
from tqdm import tqdm
import pickle
import os
import argparse
import gc
import jax

BASIN_VOLUME_DIR = "/mnt/ssd-1/adam/basin-volume"

RESULTS_DIR = os.path.join(BASIN_VOLUME_DIR, "results_tuesday")
os.makedirs(RESULTS_DIR, exist_ok=True)


def pythia_histo(testing=False, adam=False):
    cfg = VolumeConfig(model_type="pythia", 
                        model_name="31m",
                        tol=5,
                        y_tol=50,
                        val_size=1 if testing else 20,
                        n_samples=1 if testing else 1000,
                        preconditioner_type="adam" if adam else None,
                        preconditioner_eps=1e-5,
                        )
    
    ve = VolumeEstimator.from_config(cfg)
    result = ve.run()

    suff = (f"_{adam}" if adam else "") + ("_test" if testing else "")
    with open(os.path.join(RESULTS_DIR, f"pythia_histo{suff}.pkl"), "wb") as f:
        pickle.dump((result, cfg), f)

def convnext_histo(testing=False, adam=False, poison=False):
    cfg = VolumeConfig(model_type="convnext", 
                        model_name="b16pai_p4" if poison else "b16pai_p001",
                        tol=5,
                        y_tol=50,
                        val_size=1 if testing else 1024,
                        n_samples=1 if testing else 1000,
                        preconditioner_type="adam" if adam else None,
                        preconditioner_eps=1e-5,
                        )

    ve = VolumeEstimator.from_config(cfg)
    result = ve.run()

    suff = (f"_{adam}" if adam else "") + (f"_{poison}" if poison else "") + ("_test" if testing else "")
    with open(os.path.join(RESULTS_DIR, f"convnext_histo{suff}.pkl"), "wb") as f:
        pickle.dump((result, cfg), f)

def pythia_cutoff(testing=False, adam=False, eps=1e-5):
    cfg = VolumeConfig(model_type="pythia", 
                   model_name="31m",
                   tol=5,
                   y_tol=50,
                   val_size=1 if testing else 20,
                   n_samples=1 if testing else 100,
                   preconditioner_type="adam" if adam else None,
                   preconditioner_eps=eps,
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

    suff = (f"_{adam}" if adam else "") + (f"_{eps:.0e}" if eps else "") + ("_test" if testing else "")
    with open(os.path.join(RESULTS_DIR, f"pythia_cutoff{suff}.pkl"), "wb") as f:
        pickle.dump((results, cfg), f)

def convnext_cutoff(testing=False, adam=False, poison=False, eps=1e-5):
    cfg = VolumeConfig(model_type="convnext", 
                        model_name="b16pai_p4" if poison else "b16pai_p001",
                        tol=5,
                        y_tol=50,
                        val_size=1 if testing else 1024,
                        n_samples=1 if testing else 100,
                        preconditioner_type="adam" if adam else None,
                        preconditioner_eps=eps,
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

    suff = (f"_{adam}" if adam else "") + (f"_{eps:.0e}" if eps else "") + (f"_{poison}" if poison else "") + ("_test" if testing else "")
    with open(os.path.join(RESULTS_DIR, f"convnext_cutoff{suff}.pkl"), "wb") as f:
        pickle.dump((results, cfg), f)

def pythia_chkpts(testing=False, adam=False):
    cfg = VolumeConfig(model_type="pythia", 
                   model_name="31m",
                   tol=5,
                   y_tol=50,
                   val_size=1 if testing else 20,
                   n_samples=1 if testing else 100,
                   preconditioner_type="adam" if adam else None,
                   )
    
    all_steps = get_pythia_checkpoint_steps("31m")
    last_step = all_steps[-1]
    steps_to_scan = [2**i for i in range(18)] + [last_step] + [i*10_000 for i in range(15)]
    steps_to_scan = sorted(set(steps_to_scan))
    # closest existing step to each step in steps_to_scan
    steps_to_scan = [min(all_steps, key=lambda x: abs(x - step)) for step in steps_to_scan]

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

    suff = (f"_{adam}" if adam else "") + ("_test" if testing else "")
    with open(os.path.join(RESULTS_DIR, f"pythia_chkpts{suff}.pkl"), "wb") as f:
        pickle.dump((results, cfg), f)

def convnext_chkpts(testing=False, adam=False, poison=False):
    cfg = VolumeConfig(model_type="convnext", 
                        model_name="b16pai_p4" if poison else "b16pai_p001",
                        tol=5,
                        y_tol=50,
                        val_size=1 if testing else 1024,
                        n_samples=1 if testing else 100,
                        preconditioner_type="adam" if adam else None,
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

    suff = (f"_{adam}" if adam else "") + (f"_{poison}" if poison else "") + ("_test" if testing else "")
    with open(os.path.join(RESULTS_DIR, f"convnext_chkpts{suff}.pkl"), "wb") as f:
        pickle.dump((results, cfg), f)

def pythia_exponent(testing=False, adam=False, eps=1e-5):
    cfg = VolumeConfig(model_type="pythia", 
                   model_name="31m",
                   tol=5,
                   y_tol=50,
                   val_size=1 if testing else 20,
                   n_samples=1 if testing else 100,
                   preconditioner_type="adam" if adam else None,
                   preconditioner_eps=eps,
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

    suff = (f"_{adam}" if adam else "") + (f"_{eps:.0e}" if eps else "") + ("_test" if testing else "")
    with open(os.path.join(RESULTS_DIR, f"pythia_exponent{suff}.pkl"), "wb") as f:
        pickle.dump((results, cfg), f)

def convnext_exponent(testing=False, adam=False, poison=False, eps=1e-5):
    cfg = VolumeConfig(model_type="convnext", 
                        model_name="b16pai_p4" if poison else "b16pai_p001",
                        tol=5,
                        y_tol=50,
                        val_size=1 if testing else 1024,
                        n_samples=1 if testing else 100,
                        preconditioner_type="adam" if adam else None,
                        preconditioner_eps=eps,
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

    suff = (f"_{adam}" if adam else "") + (f"_{eps:.0e}" if eps else "") + (f"_{poison}" if poison else "") + ("_test" if testing else "")
    with open(os.path.join(RESULTS_DIR, f"convnext_exponent{suff}.pkl"), "wb") as f:
        pickle.dump((results, cfg), f)


if __name__ == "__main__":
    # use argparse to get testing flag and target function
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--target", type=str, default="pythia_histo")
    parser.add_argument("--adam", action="store_true")
    parser.add_argument("--poison", action="store_true")
    parser.add_argument("--eps", type=float, default=1e-5)
    args = parser.parse_args()
    match args.target:
        case "pythia_histo":
            pythia_histo(args.test, args.adam)
        case "convnext_histo":
            convnext_histo(args.test, args.adam, args.poison)
        case "pythia_cutoff":
            pythia_cutoff(args.test, args.adam, args.eps)
        case "convnext_cutoff":
            convnext_cutoff(args.test, args.adam, args.poison, args.eps)
        case "pythia_chkpts":
            pythia_chkpts(args.test, args.adam)
        case "convnext_chkpts":
            convnext_chkpts(args.test, args.adam, args.poison)
        case "pythia_exponent":
            pythia_exponent(args.test, args.adam, args.eps)
        case "convnext_exponent":
            convnext_exponent(args.test, args.adam, args.poison, args.eps)
        case _:
            raise ValueError(f"Target {args.target} not supported")