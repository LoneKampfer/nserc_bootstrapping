import hashlib

from fit_correlators import *


tref_a06 = {'ss': 20, 'ps': 20}
tref_a09 = {'ss': 20, 'ps': 20}
tref_a12 = {'ss': 10, 'ps': 10}
tref_a15 = {'ss': 10, 'ps': 10}

def prepare_gvdata(data, ensemble, hadron):
    gvdata = data[ensemble][hadron]
    T = gvdata.shape[1]
    if HADRON_TYPES.get(hadron) == "meson":
        gvdata = fold_meson_data(gvdata)

    gvdata = gvdata[:, :T // 2, :]
    print(gvdata.shape)
    return gvdata

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

def bootstrap_gvdata(gvdata, ensemble_indices, boot0_ground_state_energy, mpi, tmin, tmax, num_exps, tref, verbose=False):
    ground_state_energies = [boot0_ground_state_energy]
    n_bootstrap_samples = ensemble_indices.shape[0]
    for i in range(n_bootstrap_samples):
        if i % 100 == 0 and verbose:
            print(f"Bootstrap sample {i}/{n_bootstrap_samples}")
        sample_indices = ensemble_indices[i]
        sample_data = gvdata[sample_indices, :, :]
        sample_gvdata = gv.dataset.avg_data(sample_data)
        hadron_sample_gvdata = {"ss": sample_gvdata[:, 0].flatten(), "ps": sample_gvdata[:, 1].flatten()}
        priors = construct_priors_baryon_bootstrap(hadron_sample_gvdata, tref, num_exps, mpi.mean,
                                                   boot0_ground_state_energy)
        t = np.arange(tmin, tmax)
        t_w_ss, y_ss = time_window(np.arange(len(hadron_sample_gvdata['ss'])), hadron_sample_gvdata['ss'], tmin, tmax)
        t_w_ps, y_ps = time_window(np.arange(len(hadron_sample_gvdata['ps'])), hadron_sample_gvdata['ps'], tmin, tmax)
        y = {"ss": y_ss, "ps": y_ps}
        fcn = lambda p: fit_function_baryon(p, t_w_ss, num_exps)
        fit = lsqfit.nonlinear_fit(data=y, fcn=fcn, prior=priors, debug=True)
        ground_state_energy = fit.p['E0']
        ground_state_energies.append(ground_state_energy)
    return ground_state_energies
