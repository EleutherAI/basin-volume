from basin_volume import *
from tqdm import tqdm
import pickle
import os
import argparse
import gc
import jax

BASIN_VOLUME_DIR = "/mnt/ssd-1/adam/basin-volume"

def pythia_chkpts(testing=False, big=False, adam=False):
    cfg = VolumeConfig(model_type="pythia", 
                   model_name="31m",
                   tol=5,
                   y_tol=50,
                   val_size=1 if testing else 20 if big else 32,
                   n_samples=1 if testing else 100 if big else 1,
                   preconditioner_type="adam" if adam else None,
                   )
    
    print(cfg.val_size, cfg.n_samples)
    
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

    suff = "_test" if testing else f"_{cfg.val_size}_{cfg.n_samples}"
    with open(os.path.join(BASIN_VOLUME_DIR, "results", f"pythia_31m_chkpts{suff}.pkl"), "wb") as f:
        pickle.dump((results, cfg), f)


def pythia_final(testing=False):
    cfg = VolumeConfig(model_type="pythia", 
                   model_name="31m",
                   tol=5,
                   y_tol=50,
                   val_size=1 if testing else 100,
                   n_samples=1 if testing else 10,
                   )
    
    ve = VolumeEstimator.from_config(cfg)
    ve.config.n_samples = 10
    result = ve.run()

    suff = "_test" if testing else ""
    with open(os.path.join(BASIN_VOLUME_DIR, "results", f"pythia_31m_final{suff}.pkl"), "wb") as f:
        pickle.dump((result, cfg), f)


def pythia_chkpts_overnight(testing=False):
    cfg = VolumeConfig(model_type="pythia", 
                   model_name="31m",
                   tol=5,
                   y_tol=50,
                   val_size=1 if testing else 10,
                   n_samples=1 if testing else 500,
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

    suff = "_test" if testing else f"_{cfg.val_size}_{cfg.n_samples}"
    with open(os.path.join(BASIN_VOLUME_DIR, "results", f"pythia_31m_chkpts_overnight{suff}.pkl"), "wb") as f:
        pickle.dump((results, cfg), f)


def pythia_chkpts_overnight_adam(testing=False):
    cfg = VolumeConfig(model_type="pythia", 
                   model_name="31m",
                   tol=5,
                   y_tol=50,
                   val_size=1 if testing else 10,
                   n_samples=1 if testing else 500,
                   preconditioner_typeb="adam",
                   preconditioner_eps=1e-5,
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

    suff = "_test" if testing else f"_{cfg.val_size}_{cfg.n_samples}"
    with open(os.path.join(BASIN_VOLUME_DIR, "results", f"pythia_overnight_adam{suff}.pkl"), "wb") as f:
        pickle.dump((results, cfg), f)


def pythia_overnight_cutoff(testing=False):
    cfg = VolumeConfig(model_type="pythia", 
                   model_name="31m",
                   tol=5,
                   y_tol=50,
                   val_size=1 if testing else 20,
                   n_samples=1 if testing else 100,
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

    suff = "_test" if testing else f"_{cfg.val_size}_{cfg.n_samples}"
    with open(os.path.join(BASIN_VOLUME_DIR, "results", f"pythia_overnight_cutoff{suff}.pkl"), "wb") as f:
        pickle.dump((results, cfg), f)


def pythia_overnight_cutoff_adam(testing=False, higheps=False):
    cfg = VolumeConfig(model_type="pythia", 
                   model_name="31m",
                   tol=5,
                   y_tol=50,
                   val_size=1 if testing else 20,
                   n_samples=1 if testing else 100,
                   preconditioner_type="adam",
                   preconditioner_eps=1e-3 if higheps else 1e-5,
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

        ve.config.cutoff = cutoff
        result = ve.run()
        results[cutoff] = result
        
    # Explicitly delete the estimator
    del ve

    suff = "_test" if testing else f"_{cfg.val_size}_{cfg.n_samples}_{higheps}"
    with open(os.path.join(BASIN_VOLUME_DIR, "results", f"pythia_overnight_cutoff_adam{suff}.pkl"), "wb") as f:
        pickle.dump((results, cfg), f)


def pythia_low_cutoff_adam(testing=False):
    cfg = VolumeConfig(model_type="pythia", 
                        model_name="31m",
                        tol=5,
                        y_tol=50,
                        val_size=20,
                        n_samples=10,
                        cutoff=1e-6,
                        preconditioner_type="adam",
                        )
    
    low_cutoff_results = {}
    epsilons = np.array(logspace(1e-9, 1e-1, 9))

    ve = VolumeEstimator.from_config(cfg)

    for epsilon in tqdm(epsilons):
        if epsilon in low_cutoff_results:
            continue

        # Clear memory before each iteration
        gc.collect()
        jax.clear_caches()

        ve.config.preconditioner_eps = epsilon
        ve.set_preconditioner()
        
        result = ve.run()
        low_cutoff_results[epsilon] = result

    suff = "_test" if testing else f"_{cfg.val_size}_{cfg.n_samples}"
    with open(os.path.join(BASIN_VOLUME_DIR, "results", f"pythia_low_cutoff_adam{suff}.pkl"), "wb") as f:
        pickle.dump((low_cutoff_results, cfg), f)


def pythia_very_high_cutoff_adam(testing=False):
    cfg = VolumeConfig(model_type="pythia", 
                        model_name="31m",
                        tol=5,
                        y_tol=50,
                        val_size=20,
                        n_samples=10,
                        cutoff=1e2,
                        preconditioner_type="adam",
                        )
    
    results = {}
    epsilons = np.array(logspace(1e-9, 1e-1, 9))

    ve = VolumeEstimator.from_config(cfg)

    for epsilon in tqdm(epsilons):
        if epsilon in results:
            continue

        # Clear memory before each iteration
        gc.collect()
        jax.clear_caches()

        ve.config.preconditioner_eps = epsilon
        ve.set_preconditioner()
        
        result = ve.run()   
        results[epsilon] = result

    suff = "_test" if testing else f"_{cfg.val_size}_{cfg.n_samples}"
    with open(os.path.join(BASIN_VOLUME_DIR, "results", f"pythia_very_high_cutoff_adam{suff}.pkl"), "wb") as f:
        pickle.dump((results, cfg), f)


def pythia_normal_cutoff_adam(testing=False):
    cfg = VolumeConfig(model_type="pythia", 
                        model_name="31m",
                        tol=5,
                        y_tol=50,
                        val_size=20,
                        n_samples=10,
                        cutoff=1e-2,
                        preconditioner_type="adam",
                        )
    
    results = {}
    epsilons = np.array(logspace(1e-9, 1e-1, 9))

    ve = VolumeEstimator.from_config(cfg)

    for epsilon in tqdm(epsilons):
        if epsilon in results:
            continue

        # Clear memory before each iteration
        gc.collect()
        jax.clear_caches()

        ve.config.preconditioner_eps = epsilon
        ve.set_preconditioner()
        
        result = ve.run()   
        results[epsilon] = result

    suff = "_test" if testing else f"_{cfg.val_size}_{cfg.n_samples}"
    with open(os.path.join(BASIN_VOLUME_DIR, "results", f"pythia_normal_cutoff_adam{suff}.pkl"), "wb") as f:
        pickle.dump((results, cfg), f)


def convnext_epsilons(testing=False):
    cfg = VolumeConfig(model_type="convnext", 
                        model_name="b16pai_p001",
                        tol=5,
                        y_tol=50,
                        n_samples=10,
                        preconditioner_type="adam",
                        )
    
    epsilons = np.array(logspace(1e-9, 1e-1, 9))

    results = {}

    ve = VolumeEstimator.from_config(cfg)

    for epsilon in tqdm(epsilons):
        if epsilon in results:
            continue

        # Clear memory before each iteration
        gc.collect()
        jax.clear_caches()

        ve.config.preconditioner_eps = epsilon
        ve.set_preconditioner()
        
        result = ve.run()
        results[epsilon] = result

    suff = "_test" if testing else f"_{cfg.val_size}_{cfg.n_samples}"
    with open(os.path.join(BASIN_VOLUME_DIR, "results", f"convnext_epsilons{suff}.pkl"), "wb") as f:
        pickle.dump((results, cfg), f)


def convnext_epsilons_early(testing=False):
    cfg = VolumeConfig(model_type="convnext", 
                        model_name="b16pai_p001",
                        tol=5,
                        y_tol=50,
                        n_samples=10,
                        checkpoint_step=2**8,
                        preconditioner_type="adam",
                        )
    
    epsilons = np.array(logspace(1e-9, 1e-1, 9))

    results = {}

    ve = VolumeEstimator.from_config(cfg)

    for epsilon in tqdm(epsilons):
        if epsilon in results:
            continue

        # Clear memory before each iteration
        gc.collect()
        jax.clear_caches()

        ve.config.preconditioner_eps = epsilon
        ve.set_preconditioner()
        
        result = ve.run()
        results[epsilon] = result

    suff = "_test" if testing else f"_{cfg.val_size}_{cfg.n_samples}"
    with open(os.path.join(BASIN_VOLUME_DIR, "results", f"convnext_epsilons_early{suff}.pkl"), "wb") as f:
        pickle.dump((results, cfg), f)


def pythia_timing(testing=False):
    cfg = VolumeConfig(model_type="pythia", 
                        model_name="31m",
                        tol=5,
                        y_tol=50,
                        val_size=20,
                        n_samples=100,
                        )

    ve = VolumeEstimator.from_config(cfg)

    results = {}

    for i in tqdm(range(1)):
        result = ve.run()
        results[i] = result

    suff = "_test" if testing else f"_{cfg.val_size}_{cfg.n_samples}"
    with open(os.path.join(BASIN_VOLUME_DIR, "results", f"pythia_timing{suff}.pkl"), "wb") as f:
        pickle.dump((results, cfg), f)


def convnext_timing(testing=False):
    cfg = VolumeConfig(model_type="convnext", 
                        model_name="b16pai_p001",
                        tol=5,
                        y_tol=50,
                        val_size=1024,
                        n_samples=100,
                        )
    
    ve = VolumeEstimator.from_config(cfg)
    
    results = {}

    for i in tqdm(range(1)):
        result = ve.run()
        results[i] = result

    suff = "_test" if testing else f"_{cfg.val_size}_{cfg.n_samples}"
    with open(os.path.join(BASIN_VOLUME_DIR, "results", f"convnext_timing{suff}.pkl"), "wb") as f:
        pickle.dump((results, cfg), f)


def pythia_histo(testing=False, adam=False):
    cfg = VolumeConfig(model_type="pythia", 
                        model_name="31m",
                        tol=5,
                        y_tol=50,
                        val_size=20,
                        n_samples=1000,
                        preconditioner_type="adam" if adam else None,
                        preconditioner_eps=1e-5,
                        )
    
    ve = VolumeEstimator.from_config(cfg)
    result = ve.run()

    suff = "_test" if testing else f"_{adam}"
    with open(os.path.join(BASIN_VOLUME_DIR, "results", f"pythia_histo{suff}.pkl"), "wb") as f:
        pickle.dump((result, cfg), f)


def pythia_exponents(testing=False):
    cfg = VolumeConfig(model_type="pythia", 
                        model_name="31m",
                        tol=5,
                        y_tol=50,
                        val_size=20,
                        n_samples=100,
                        preconditioner_type="adam",
                        )
    
    results = {}
    exponents = np.array(linspace(0, 1, 11))

    ve = VolumeEstimator.from_config(cfg)

    for exponent in tqdm(exponents):
        if exponent in results:
            continue

        # Clear memory before each iteration
        gc.collect()
        jax.clear_caches()

        ve.config.preconditioner_exponent = exponent
        ve.set_preconditioner()
        
        result = ve.run()
        results[exponent] = result

    suff = "_test" if testing else f"_{cfg.val_size}_{cfg.n_samples}"
    with open(os.path.join(BASIN_VOLUME_DIR, "results", f"pythia_exponents{suff}.pkl"), "wb") as f:
        pickle.dump((results, cfg), f)


if __name__ == "__main__":
    # use argparse to get testing flag and target function
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--target", type=str, default="pythia_chkpts")
    args = parser.parse_args()
    match args.target:
        case "pythia_chkpts":
            pythia_chkpts(args.test)
        case "pythia_final":
            pythia_final(args.test)
        case "pythia_chkpts_overnight":
            pythia_chkpts_overnight(args.test)
        case "pythia_overnight_adam":
            pythia_chkpts_overnight_adam(args.test)
        case "pythia_overnight_cutoff":
            pythia_overnight_cutoff(args.test)
        case "pythia_overnight_cutoff_adam":
            pythia_overnight_cutoff_adam(args.test, higheps=False)
        case "pythia_cutoff_adam_higheps":
            pythia_overnight_cutoff_adam(args.test, higheps=True)
        case "pythia_low_cutoff_adam":
            pythia_low_cutoff_adam(args.test)
        case "pythia_very_high_cutoff_adam":
            pythia_very_high_cutoff_adam(args.test)
        case "pythia_normal_cutoff_adam":
            pythia_normal_cutoff_adam(args.test)
        case "convnext_epsilons":
            convnext_epsilons(args.test)
        case "convnext_epsilons_early":
            convnext_epsilons_early(args.test)
        case "pythia_timing":
            pythia_timing(args.test)
        case "convnext_timing":
            convnext_timing(args.test)
        # big runs
        case "pythia_histo":
            pythia_histo(args.test, adam=False)
        case "pythia_histo_adam":
            pythia_histo(args.test, adam=True)
        case "pythia_chkpts_big":
            pythia_chkpts(args.test, big=True)
        case "pythia_chkpts_big_adam":
            pythia_chkpts(args.test, big=True, adam=True)
        case "pythia_exponents":
            pythia_exponents(args.test)
        case _:
            raise ValueError(f"Target {args.target} not supported")
