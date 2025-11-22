import hashlib
import multiprocessing as mp
import warnings

# suppress numerical RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# suppress multiprocessing gvar pickle warnings
warnings.filterwarnings("ignore", message="Pickling GVars")

# suppress any generic UserWarning from the multiprocessing reduction module
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing")

from fit_correlators import *


tref_a06 = {'ss': 20, 'ps': 20}
tref_a09 = {'ss': 20, 'ps': 20}
tref_a12 = {'ss': 10, 'ps': 10}
tref_a15 = {'ss': 10, 'ps': 10}

def prepare_gvdata(data, ensemble, hadron):
    """
    Prepare gvdata for bootstrap fitting.
    
    For mesons: folds the data (exploiting time-reversal symmetry) then truncates to T//2.
    For baryons: just truncates to T//2.
    
    Returns:
        gvdata: (n_cfg, T_final, n_channels) where T_final = T//2
        T_original: original time extent (needed for meson cosh fits)
    """
    gvdata = data[ensemble][hadron]
    T_original = gvdata.shape[1]
    if HADRON_TYPES.get(hadron) == "meson":
        gvdata = fold_meson_data(gvdata)

    gvdata = gvdata[:, :T_original // 2, :]
    print(gvdata.shape)
    return gvdata, T_original

def generate_boot0_and_mpi_data(ensemble, hadron):
    boot0_data_dir = f"correlator_results/fits/{ensemble}/ground_state_energies_{hadron}.txt"
    mpi_cache_dir = f"correlator_results/fits/{ensemble}/ground_state_energies_piplus.txt"
    boot0_ground_state_energy, tmin, tmax, num_exps, Q, chi2_dof, logBF = load_ground_state_energy(boot0_data_dir)
    mpi, _, __, ___, ____, _____, ______ = load_ground_state_energy(mpi_cache_dir)
    mpi_cache = {}
    mpi_cache[ensemble] = mpi

    print("Boot0 Ground State Energies:", boot0_ground_state_energy)
    print("MPI Cache Ground State Energies:", mpi_cache)
    print("tmin:", tmin)
    print("num_exps:", num_exps)

    #a is lattice spacing, first 3 chars of ensemble name
    a = ensemble[:3]
    if a == "a06":
        tref = tref_a06
    elif a == "a09":
        tref = tref_a09
    elif a == "a12":
        tref = tref_a12
    elif a == "a15":
        tref = tref_a15
    else:
        raise Exception(f"Unknown ensemble type: {a}")

    return boot0_ground_state_energy, mpi_cache, tmin, tmax, num_exps, tref

def generate_ensemble_indices(ensemble, num_configs, num_samples):
    seed = int(hashlib.md5(ensemble.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.integers(0, num_configs, size=(num_samples, num_configs))

def bootstrap_gvdata(gvdata, T_original, hadron, ensemble_indices, boot0_ground_state_energy, mpi, tmin, tmax, num_exps, tref, verbose=False):
    ground_state_energies = [boot0_ground_state_energy]
    n_bootstrap_samples = ensemble_indices.shape[0]
    t0 = time.time()
    for i in range(n_bootstrap_samples):
        if verbose:
            print(f"Bootstrap sample {i}/{n_bootstrap_samples}", end="\r")
        sample_indices = ensemble_indices[i]
        sample_data = gvdata[sample_indices, :, :]
        sample_gvdata = gv.dataset.avg_data(sample_data)
        hadron_sample_gvdata = {"ss": sample_gvdata[:, 0].flatten(), "ps": sample_gvdata[:, 1].flatten()}
        if HADRON_TYPES.get(hadron) == "baryon":
            priors = construct_priors_baryon_bootstrap(hadron_sample_gvdata, tref, num_exps, mpi.mean,
                                                       boot0_ground_state_energy)
        elif HADRON_TYPES.get(hadron) == "meson":
            priors = construct_priors_meson_bootstrap(hadron_sample_gvdata, tref, num_exps, mpi.mean,
                                                       boot0_ground_state_energy)
        else:
            raise Exception(f"Unknown hadron type: {hadron}")
        t = np.arange(tmin, tmax)
        t_w_ss, y_ss = time_window(np.arange(len(hadron_sample_gvdata['ss'])), hadron_sample_gvdata['ss'], tmin, tmax)
        t_w_ps, y_ps = time_window(np.arange(len(hadron_sample_gvdata['ps'])), hadron_sample_gvdata['ps'], tmin, tmax)
        y = {"ss": y_ss, "ps": y_ps}
        if HADRON_TYPES.get(hadron) == "baryon":
            fcn = lambda p: fit_function_baryon(p, t_w_ss, num_exps)
        elif HADRON_TYPES.get(hadron) == "meson":
            fcn = lambda p: fit_function_meson(p, t_w_ss, T_original, num_exps)
        else:
            raise Exception(f"Unknown hadron type: {hadron}")
        fit = lsqfit.nonlinear_fit(data=y, fcn=fcn, prior=priors, debug=True)
        ground_state_energy = fit.p['E0']
        ground_state_energies.append(ground_state_energy)
    t1 = time.time()
    print(f"Time taken: {t1 - t0} seconds")
    return ground_state_energies



# ===== parallel execution =====

def _bootstrap_chunk_worker(args):
    """
    Run bootstrap_gvdata on a chunk of indices.
    THIS CALLS YOUR EXISTING FUNCTION EXACTLY AS-IS.
    """
    (
        gvdata,
        T_original,
        hadron,
        indices_chunk,
        boot0,
        mpi,
        tmin,
        tmax,
        num_exps,
        tref,
    ) = args

    # Use your existing function directly
    return bootstrap_gvdata(
        gvdata,
        T_original,
        hadron,
        indices_chunk,
        boot0,
        mpi,
        tmin,
        tmax,
        num_exps,
        tref,
        verbose=False  # don't spam
    )[1:]  # drop the boot0 at the front

def bootstrap_gvdata_parallel(
    gvdata,
    T_original,
    hadron,
    ensemble_indices,
    boot0_ground_state_energy,
    mpi,
    tmin,
    tmax,
    num_exps,
    tref,
    verbose=True
):
    N = ensemble_indices.shape[0]

    # determine workers
    try:
        import psutil
        n_workers = psutil.cpu_count(logical=False)
    except:
        n_workers = mp.cpu_count()

    if verbose:
        print(f"[parallel bootstrap] {N} samples on {n_workers} workers")

    # chunk the index table
    chunks = np.array_split(ensemble_indices, n_workers, axis=0)

    # prepare arguments for each worker
    worker_args = [
        (
            gvdata,
            T_original,
            hadron,
            chunk,
            boot0_ground_state_energy,
            mpi,
            tmin,
            tmax,
            num_exps,
            tref,
        )
        for chunk in chunks if len(chunk) > 0
    ]

    # Mac-safe start method
    ctx = mp.get_context("spawn")

    with ctx.Pool(n_workers) as pool:
        results = pool.map(_bootstrap_chunk_worker, worker_args)

    # flatten all lists
    flat = [E for worker_list in results for E in worker_list]

    # prepend boot0
    return [boot0_ground_state_energy] + flat
