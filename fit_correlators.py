import os
import subprocess

import h5py
import pandas as pd

import numpy as np
import gvar as gv
import lsqfit
import matplotlib.pyplot as plt
import re
import h5py as h5
import time


channel_colors = {
    'ss': 'grey',
    'ps': 'grey'
}

HBAR_C = 0.1973269804  # GeV·fm

# Define colors globally
color = {
    0: 'magenta',
    1: 'blue',
    2: 'orange',
    3: 'green',
    4: 'red'
}

# Define hadron types globally
HADRON_TYPES = {
    # Baryons (from your list)
    'lambda_z': 'baryon',
    'mp_l': 'baryon',
    'mp_s': 'baryon',
    'omega_m': 'baryon',
    'pp_l': 'baryon',
    'pp_s': 'baryon',
    'sigma_p': 'baryon',
    'sigma_star_p': 'baryon',
    'xi_star_z': 'baryon',
    'xi_z': 'baryon',

    # Mesons (remaining from the full list)
    'kplus': 'meson',
    'phi_ss': 'meson',
    'piplus': 'meson'
}


channel_colors = {
    'ss': 'grey',
    'ps': 'grey'
}

ground_state_energy_entry_structure = {
    't_min': 0,
    't_max': 1,
    'num_exps': 2,
    'E0_mean': 3,
    'E0_sdev': 4,
    'Q': 5,
    'chi2_dof': 6,
    'logBF': 7
}

# ==== Auxiliary functions ====

def mass_effective_transformation(data):
    # For baryons or exponential correlators
    return gv.log(data[:-1] / data[1:]).flatten()

def mass_effective_transformation_cosh(data):
    # For mesons (cosh symmetry)
    c_t = data
    c_tp1 = np.roll(c_t, -1)
    c_tm1 = np.roll(c_t, 1)
    ratio = (c_tp1 + c_tm1) / (2 * c_t)
    meff = gv.arccosh(ratio[1:-1])  # inverse cosh, not cosh!
    return meff[1:-1].flatten()  # remove edges



def time_window(t, data, t_min, t_max):
    mask = (t >= t_min) & (t <= t_max)
    return t[mask], data[mask]

def fold_meson_data(data):
    if len(data.shape) == 2:
        num_configs, T = data.shape
        folded_data = np.zeros((num_configs, T // 2 + 1))
        # t=0 point remains the same
        folded_data[:, 0] = data[:, 0]
        # t=T/2 point remains the same (self-average)
        folded_data[:, -1] = data[:, T // 2]
        # average the rest symmetrically
        for t in range(1, T // 2):
            folded_data[:, t] = 0.5 * (
                    data[:, t] + data[:, -t]
            )
        return folded_data

    elif len(data.shape) == 4:
        numconfigs, T, N_channels, _ = data.shape
        folded_data = np.zeros((numconfigs, T // 2 + 1, N_channels))

        # t=0 point remains the same
        folded_data[:, 0, :] = data[:, 0, :, 0]

        # t=T/2 point remains the same (self-average)
        folded_data[:, -1, :] = data[:, T // 2, :, 0]

        # average the rest symmetrically
        for t in range(1, T // 2):
            folded_data[:, t, :] = 0.5 * (
                    data[:, t, :, 0] + data[:, -t, :, 0]
            )
        return folded_data
    else:
        raise ValueError("Data must be 2D or 4D array.")


def get_ground_state_energy(fit):
    return fit.p['E0']

# ==== Fit functions and priors for mesons ====

def fit_function_meson(p, t, T, num_exps):
    E0 = p['E0']
    cum_gap = 0.0
    result = {'ss': 0, 'ps': 0}
    A0_ss = p['A0_ss']
    A0_ps = p['A0_ps']
    for n in range(num_exps + 1):
        if n == 0:
            An_ss = A0_ss
            An_ps = A0_ps
            E = E0
        else:
            An_ss = A0_ss * p[f'r_{n}_ss'] ** 2
            An_ps = A0_ps * p[f'r_{n}_ps'] * p[f'r_{n}_ss']
            E = E + gv.exp(p[f'log_dE_{n - 1}{n}'])
            # print(f"{n}, {E}")

        term = 2 * gv.exp(-E * T / 2) * gv.cosh(E * (T / 2 - t))

        result['ss'] += An_ss * term
        result['ps'] += An_ps * term
    return result

def construct_priors_pion(gvdata, t_ref, num_exps):
    meff = mass_effective_transformation_cosh(gvdata['ss'])
    m_eff_tref = meff[t_ref['ss']]
    A_eff_tref_ss = gvdata['ss'][t_ref['ss']] * gv.exp(m_eff_tref * t_ref['ss'])

    meff = mass_effective_transformation_cosh(gvdata['ps'])
    m_eff_tref = meff[t_ref['ps']]
    A_eff_tref_ps = gvdata['ps'][t_ref['ps']] * gv.exp(m_eff_tref * t_ref['ps'])

    priors = {
        'E0': m_eff_tref * gv.gvar(1.0, 0.1),
        'A0_ss': A_eff_tref_ss * gv.gvar(1.0, 1.0),
        'A0_ps': A_eff_tref_ps * gv.gvar(1.0, 1.0)
    }

    for n in range(1, num_exps + 1):
        priors[f'log_dE_{n - 1}{n}'] = gv.gvar(gv.log(2 * gv.mean(m_eff_tref)), 0.7) #TODO: Is this the correct mean for this?
        priors[f'r_{n}_ps'] =  gv.gvar(0.0, 2.0)
        priors[f'r_{n}_ss'] = gv.gvar(0.0, 2.0)

    return priors

def construct_priors_meson(gvdata, t_ref, num_exps, mpi):
    meff = mass_effective_transformation_cosh(gvdata['ss'])
    m_eff_tref = meff[t_ref['ss']]
    A_eff_tref_ss = gvdata['ss'][t_ref['ss']] * gv.exp(m_eff_tref * t_ref['ss'])
    A_eff_tref_ps = gvdata['ps'][t_ref['ps']] * gv.exp(m_eff_tref * t_ref['ps'])

    priors = {
        'E0': m_eff_tref * gv.gvar(1.0, 0.1),
        'A0_ss': A_eff_tref_ss * gv.gvar(1.0, 1.0),
        'A0_ps': A_eff_tref_ps * gv.gvar(1.0, 1.0)
    }

    for n in range(1, num_exps + 1):
        priors[f'log_dE_{n - 1}{n}'] = gv.gvar(np.log(2*mpi), 0.7) #TODO: Is this the correct mean for this?
        priors[f'r_{n}_ps'] =  gv.gvar(0.0, 2.0)
        priors[f'r_{n}_ss'] = gv.gvar(1.0, 0.7)

    return priors

def construct_priors_meson_bootstrap(gvdata, t_ref, num_exps, mpi, E0_boot0, W=3.0):
    meff = mass_effective_transformation_cosh(gvdata['ss'])
    m_eff_tref = meff[t_ref['ss']]
    A_eff_tref_ss = gvdata['ss'][t_ref['ss']] * gv.exp(m_eff_tref * t_ref['ss'])
    A_eff_tref_ps = gvdata['ps'][t_ref['ps']] * gv.exp(m_eff_tref * t_ref['ps'])

    priors = {
        'E0': gv.gvar(gv.mean(E0_boot0), W * gv.sdev(E0_boot0)),   # widened prior
        'A0_ss': A_eff_tref_ss * gv.gvar(1.0, 1.0),
        'A0_ps': A_eff_tref_ps * gv.gvar(1.0, 1.0)
    }

    for n in range(1, num_exps + 1):
        priors[f'log_dE_{n - 1}{n}'] = gv.gvar(np.log(2 * gv.mean(mpi)), 0.7)
        priors[f'r_{n}_ps'] = gv.gvar(0.0, 2.0)
        priors[f'r_{n}_ss'] = gv.gvar(1.0, 0.7)
    return priors


# ======== fit functions and priors for baryons ========#

def construct_priors_baryon(gvdata, t_ref, mpi, num_exps, verbose=False):
    meff = mass_effective_transformation(gvdata['ss'])
    m_eff_tref = meff[t_ref['ss']]
    A_eff_tref_ss = gvdata['ss'][t_ref['ss']] * gv.exp(m_eff_tref * t_ref['ss'])

    meff = mass_effective_transformation(gvdata['ps'])
    m_eff_tref = meff[t_ref['ps']]
    A_eff_tref_ps = gvdata['ps'][t_ref['ps']] * gv.exp(m_eff_tref * t_ref['ps'])

    priors = {
        'E0': m_eff_tref * gv.gvar(1.0, 0.1),
        'A0_ss': A_eff_tref_ss * gv.gvar(1.0, 1.0),
        'A0_ps': A_eff_tref_ps * gv.gvar(1.0, 1.0)
    }

    # Only dE priors use mπ
    for n in range(1, num_exps + 1):
        priors[f'log_dE_{n - 1}{n}'] = gv.gvar(np.log(2 * gv.mean(mpi)), 0.7)
        priors[f'r_{n}_ps'] = gv.gvar(0.0, 2.0)
        priors[f'r_{n}_ss'] = gv.gvar(1.0, 0.7)

    if verbose:
        print("priors: ", priors)
    return priors

def construct_priors_baryon_bootstrap(gvdata, t_ref, num_exps, mpi, E0_boot0, W=3.0):
    """
    Retunes the E0 prior using the saved ground-state energy file for bootstrapping,
    keeping all other priors identical to construct_priors_baryon().
    """

    meff = mass_effective_transformation(gvdata['ss'])
    m_eff_tref = meff[t_ref['ss']]
    A_eff_tref_ss = gvdata['ss'][t_ref['ss']] * gv.exp(m_eff_tref * t_ref['ss'])

    meff = mass_effective_transformation(gvdata['ps'])
    m_eff_tref = meff[t_ref['ps']]
    A_eff_tref_ps = gvdata['ps'][t_ref['ps']] * gv.exp(m_eff_tref * t_ref['ps'])

    priors = {
        'E0': gv.gvar(gv.mean(E0_boot0), W * gv.sdev(E0_boot0)),   # widened prior
        'A0_ss': A_eff_tref_ss * gv.gvar(1.0, 1.0),
        'A0_ps': A_eff_tref_ps * gv.gvar(1.0, 1.0)
    }

    for n in range(1, num_exps + 1):
        priors[f'log_dE_{n - 1}{n}'] = gv.gvar(np.log(2 * gv.mean(mpi)), 0.7)
        priors[f'r_{n}_ps'] = gv.gvar(0.0, 2.0)
        priors[f'r_{n}_ss'] = gv.gvar(1.0, 0.7)

    return priors

def fit_function_baryon(p, t, num_exps):
    """
    Two-channel baryon correlator model:
        ss: A0_ss * exp(-E0 * t) * (1 + Σ (r_n_ss)^2 exp(-ΔE_n t))
        ps: A0_ps * exp(-E0 * t) * (1 + Σ (r_n_ss * r_n_ps) exp(-ΔE_n t))
    """
    E0 = p['E0']
    A0_ss = p['A0_ss']
    A0_ps = p['A0_ps']

    result = {'ss': 0, 'ps': 0}

    # Ground state terms
    result['ss'] = A0_ss * gv.exp(-E0 * t)
    result['ps'] = A0_ps * gv.exp(-E0 * t)

    # Add excited-state contributions
    cum_gap = 0.0
    for n in range(1, num_exps + 1):
        dE_n = gv.exp(p[f'log_dE_{n - 1}{n}'])
        cum_gap += dE_n

        r_n_ss = p[f'r_{n}_ss']
        r_n_ps = p[f'r_{n}_ps']

        # each term parallels the single-channel structure
        term_ss = (r_n_ss ** 2) * gv.exp(-cum_gap * t)
        term_ps = (r_n_ss * r_n_ps) * gv.exp(-cum_gap * t)

        result['ss'] += A0_ss * gv.exp(-E0 * t) * term_ss
        result['ps'] += A0_ps * gv.exp(-E0 * t) * term_ps

    return result

# ======= Fit and Stability Analysis Routines ======= #

def fit_hadron(gvdata, ensemble, hadron, t_min, t_max, T, num_exps,t_ref, mpi_cache=None, debug=False, verbose=False, svd_cut=None):
    """
    Fit a specific hadron on a specific ensemble.
    Uses stored gvdata, t_ref, and mpi_cache.
    Automatically switches to cosh form for mesons (e.g., pion).
    """
    if verbose:
        print(f"{hadron} on {ensemble}")
        print(f"Fit with t_min = {t_min}, t_max = {t_max}, num_exps = {num_exps + 1}")


    # Fit both channels together
    t_w_ss, y_ss = time_window(np.arange(len(gvdata['ss'])), gvdata['ss'], t_min, t_max)
    t_w_ps, y_ps = time_window(np.arange(len(gvdata['ps'])), gvdata['ps'], t_min, t_max)
    if verbose:
         print(f"time window: {t_w_ss}")
    y_ss, y_ps = y_ss, y_ps
    y_combined = {'ss': y_ss, 'ps': y_ps}
    t_w_ps, t_w_ss = t_w_ps, t_w_ss  # Both should be the same


    # --- CASE 1: PION OR MESON (use cosh form) ---
    if HADRON_TYPES.get(hadron) == 'meson':
        if hadron == 'piplus':
            if verbose:
                print("Fitting pion specifically")
            priors = construct_priors_pion(gvdata, t_ref, num_exps)
        else:
            mpi = gv.mean(mpi_cache[ensemble])
            if verbose:
                print("Using mpi from cache: ", mpi)
            priors = construct_priors_meson(gvdata, t_ref, num_exps, mpi)
        which_func = 'meson'
        fcn = lambda p: fit_function_meson(p, t_w_ss, T, num_exps)
        if verbose:
            print("Using meson (cosh) fit form")

    # --- CASE 2: BARYON (use exponential form) ---
    else:
        which_func = 'baryon'
        mpi = mpi_cache[ensemble]
        mpi = gv.mean(mpi)
        if verbose:
            print("Using mpi from cache: ", mpi)
        priors = construct_priors_baryon(gvdata, t_ref, mpi, num_exps) #TODO: Fill in later
        fcn = lambda p: fit_function_baryon(p, t_w_ss, num_exps) #TODO: Fill in later
        if verbose:
            print("Using baryon (exp) fit form")

    # Perform the fit
    if verbose:
        print(which_func)
    fit = lsqfit.nonlinear_fit(data=y_combined, fcn=fcn, prior=priors, debug=debug, svdcut=svd_cut) # svdcut for pion and kaon
    return fit, t_w_ss, y_combined


def stability_analysis(gvdata, ensemble, hadron, t_max_fixed, t_min_list, T, num_exps_max, tref, mpi_cache=None,verbose=False, svd_cut=None):
    """
    Run stability analysis for a specific ensemble/hadron combination.

    Args:
        ensemble: ensemble name
        hadron: hadron name
        t_max_fixed: fixed maximum time for fits
        t_min_list: list of minimum times to test
        num_exps_max: maximum number of excited states
        verbose: print detailed output
    """
    results = {}

    for t_min in t_min_list:
        if t_min >= t_max_fixed:
            continue
        for num_exps in range(0, num_exps_max + 1):
            # Use new fit_hadron method
            fit, _, _ = fit_hadron(gvdata, ensemble, hadron, t_min, t_max_fixed, T, num_exps, tref, mpi_cache=mpi_cache, verbose=verbose, svd_cut=svd_cut)

            if verbose:
                print(
                    f"Fit results for {hadron} on {ensemble}: t_min={t_min}, num_exps={num_exps + 1}, t_max={t_max_fixed}:")
                print(fit)

            results[(t_min, num_exps)] = {
                'fit': fit,
                'E0': get_ground_state_energy(fit),
                'Q': fit.Q,
                'chi2/dof': fit.chi2 / fit.dof,
                'logBF': fit.logGBF
            }
    return results


def save_stability_summary_text(results, ensemble, hadron, output_dir="correlator_results/fits"):
    """
    Save all fits for a given hadron (across t_min and Nexp) into ONE text file.
    This mirrors what the stability plot summarizes.

    Args:
        results: dict returned by stability_analysis()
        ensemble: str, e.g. 'a12m400'
        hadron: str, e.g. 'piplus'
        output_dir: base directory (default: correlator_results/fits)
    """
    fit_dir = os.path.join(output_dir, ensemble)
    os.makedirs(fit_dir, exist_ok=True)

    file_path = os.path.join(fit_dir, f"{hadron}_allfits.txt")
    with open(file_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"  STABILITY ANALYSIS RESULTS  |  Ensemble: {ensemble}, Hadron: {hadron}\n")
        f.write("=" * 60 + "\n\n")

        # Sort results in ascending order of (t_min, Nexp)
        for (t_min, num_exps), res in sorted(results.items()):
            fit = res["fit"]
            f.write(f"--- Fit: t_min = {t_min:.0f}, Nexp = {num_exps + 1} ---\n")
            f.write(f"chi2/dof = {fit.chi2 / fit.dof:.3f},  Q = {fit.Q:.3f},  logGBF = {fit.logGBF:.3f}\n")

            # Ground-state energy summary
            if "E0" in fit.p:
                E0_mean = gv.mean(fit.p["E0"])
                E0_err = gv.sdev(fit.p["E0"])
                f.write(f"E0 = {E0_mean:.6f} ± {E0_err:.6f}\n")
            else:
                f.write("E0 = [Not found]\n")

            f.write("\nParameters:\n")
            for k, v in fit.p.items():
                f.write(f"  {k:15s} = {v}\n")

            f.write("\nPriors:\n")
            for k, v in fit.prior.items():
                f.write(f"  {k:15s} = {v}\n")

            f.write("-" * 60 + "\n\n")

        f.write("End of stability analysis.\n")

    print(f"[saved] {file_path}")

def save_ground_state_energies(results, ensemble, hadron, outputdir, fit_tmin, num_exps_fit, fit_tmax, blind=False):
    """
    Saves only the chosen ground-state fit (t_min, num_exps) to a single text file
    for a given hadron and ensemble.

    Args:
        results (dict): output from stability_analysis()
        ensemble (str): ensemble name
        hadron (str): hadron name
        outputdir (str): directory where to save
        fit_tmin (int): chosen t_min for the preferred fit
        num_exps_fit (int): chosen number of exponentials for the preferred fit
    """
    os.makedirs(outputdir, exist_ok=True)
    outfile = os.path.join(outputdir, f"ground_state_energies_{hadron}.txt")
    if blind:
        outfile =  os.path.join(outputdir, f"ground_state_energies_{hadron}_blind.txt")

    key = (fit_tmin, num_exps_fit)
    if key not in results:
        print(f"⚠️ Warning: (t_min={fit_tmin}, num_exps={num_exps_fit}) not found for {hadron}")
        return

    res = results[key]
    E0 = res['E0']

    with open(outfile, "w") as f:
        f.write(f"# Ground-state energy fit for {hadron} on {ensemble}\n")
        f.write("# Columns: t_min  t_max  num_exps  E0_mean  E0_sdev  Q_value  chi2_dof  logBF\n")
        f.write("# -------------------------------------------------------------\n")
        f.write(
            f"{int(fit_tmin):5d} {int(fit_tmax):5d}  {int(num_exps_fit+1):8d}  "
            f"{gv.mean(E0):10.6f}  {gv.sdev(E0):10.6f}  "
            f"{res['Q']:8.3f}  {res['chi2/dof']:10.4f}  "
            f"{res['logBF']:8.3f} \n"
        )

    #ground_state_energy_entry_structure = {0: 't_min', 1: 't_max', 2: 'num_exps', 3: 'E0_mean', 4: 'E0_sdev', 5: 'Q_value', 6: 'chi2_dof', 7: 'logBF'}

    print(f"[✓] Saved chosen ground-state energy for {hadron} → {outfile}")


def load_ground_state_energy(filepath):
    """
    Reads a ground-state energy text file produced by save_ground_state_energies()
    and returns a dictionary of extracted values.

    Args:
        filepath (str): Path to the ground_state_energies_<hadron>.txt file

    Returns:
        gv.gvar: Ground-state energy with uncertainty
    """
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not lines:
        raise ValueError(f"No data lines found in {filepath}")

    # Extract hadron and ensemble from header (optional)
    with open(filepath, "r") as f:
        header = f.read().splitlines()
    hadron, ensemble = None, None
    for line in header:
        if line.startswith("# Ground-state energy fit for"):
            parts = line.split()
            hadron = parts[5]
            ensemble = parts[-1]
            break

    # Parse the single data line, using ground_state_energy_entry_structure
    parts = lines[0].split()
    tmin = int(parts[ground_state_energy_entry_structure['t_min']])
    tmax = int(parts[ground_state_energy_entry_structure['t_max']])
    num_exps = int(parts[ground_state_energy_entry_structure['num_exps']])
    E0_mean = float(parts[ground_state_energy_entry_structure['E0_mean']])
    E0_sdev = float(parts[ground_state_energy_entry_structure['E0_sdev']])
    Q = float(parts[ground_state_energy_entry_structure['Q']])
    chi2_dof = float(parts[ground_state_energy_entry_structure['chi2_dof']])
    logBF = float(parts[ground_state_energy_entry_structure['logBF']])

    return (gv.gvar(E0_mean, E0_sdev), tmin, tmax, num_exps, Q, chi2_dof, logBF)

def plot_stability_summary(gvdata, ensemble, hadron, results, t_min_list, num_exps_max, t_max_fixed, outputdir, T,
                           fit_tmin=None, fit_num_exps=None, show_plots=True, ylims=[]):
    """
    Creates a 3-panel stability summary:
      1. Ground-state E0 vs t_min for different num_exps (ps and ss)
      2. Q-value scatter vs t_min
      3. Excited-state weights (normalized exp(logGBF))
    """
    epsilon = 1. / (num_exps_max + 5)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)
    ax_E0, ax_Q, ax_w = axes

    # ---------- 1. Stability of E0 ----------
    for num_exps in range(0, num_exps_max + 1):
        t_vals, E0_means, E0_errs = [], [], []
        for t_min in t_min_list:
            key = (t_min, num_exps)
            if key not in results: continue
            fit = results[key]['fit']
            E0 = fit.p['E0']
            t_vals.append(t_min + (num_exps - 1) * epsilon)
            E0_means.append(gv.mean(E0))
            E0_errs.append(gv.sdev(E0))
        if t_vals:
            ax_E0.errorbar(t_vals, E0_means, yerr=E0_errs,
                           fmt='x', color=color[num_exps],
                           label=f"Nexp={num_exps+1}", capsize=2, alpha=0.7)




    #plot blue, orange points for data
    if HADRON_TYPES.get(hadron) == 'meson':
        print("plotting meson data")
        meff_ss = mass_effective_transformation_cosh(gvdata['ss']).flatten()
        meff_ps = mass_effective_transformation_cosh(gvdata['ps']).flatten()
        time = np.arange(0, len(meff_ss))
    else:
        print("plotting baryon data")
        meff_ss = mass_effective_transformation(gvdata['ss']).flatten()
        meff_ps = mass_effective_transformation(gvdata['ps']).flatten()
        time = np.arange(len(meff_ss))

    print(time.shape, meff_ss.shape, meff_ps.shape)
    ax_E0.errorbar(time, gv.mean(meff_ss), yerr=gv.sdev(meff_ss), fmt='o', color=channel_colors['ss'], label='ss data', alpha=0.5, capsize=2)
    ax_E0.errorbar(time, gv.mean(meff_ps), yerr=gv.sdev(meff_ps), fmt='s', color=channel_colors['ps'], label='ps data', alpha=0.5, capsize=2)
    ax_E0.legend(fontsize=8, ncol=2)

    # Overlay fit lines if specified
    if fit_tmin is not None and fit_num_exps is not None:
        key = (fit_tmin, fit_num_exps)
        if key in results:
            fit = results[key]['fit']
            E0_fit = fit.p['E0']
            print(f"Chosen E0 fit for {hadron} on {ensemble}:", E0_fit)
            E0_fit_mean = gv.mean(E0_fit)
            E0_fit_err = gv.sdev(E0_fit)
            num_points = 1000
            t_linspace_fit = np.arange(fit_tmin, t_max_fixed, 1/num_points)
            print("t_linspace_fit: ", t_linspace_fit)
            fit_params = fit.p
            if HADRON_TYPES.get(hadron) == 'meson':
                fit_curve_t = fit_function_meson(fit_params, t_linspace_fit, T, fit_num_exps)
                fit_curve_tp1 = fit_function_meson(fit_params, t_linspace_fit+1, T, fit_num_exps)
                fit_curve_tm1 = fit_function_meson(fit_params, t_linspace_fit-1, T, fit_num_exps)
                meff_ps = gv.arccosh((fit_curve_tp1['ps'] + fit_curve_tm1['ps']) / (2 * fit_curve_t['ps'])).flatten()
                meff_ss = gv.arccosh((fit_curve_tp1['ss'] + fit_curve_tm1['ss']) / (2 * fit_curve_t['ss'])).flatten()
                t_linspace_fit -= 1
                fit_curve = {'ss': meff_ss.flatten(), 'ps': meff_ps.flatten()}


            else:
                fit_curve_t = fit_function_baryon(fit_params, t_linspace_fit, fit_num_exps)
                fit_curve_tp1 = fit_function_baryon(fit_params, t_linspace_fit + 1, fit_num_exps)
                meff_ps = gv.log(fit_curve_t['ps'] / fit_curve_tp1['ps']).flatten()
                meff_ss = gv.log(fit_curve_t['ss'] / fit_curve_tp1['ss']).flatten()
                fit_curve = {'ss': meff_ss.flatten(), 'ps': meff_ps.flatten()}





            # Plot ss and ps fit curves

            ax_E0.plot(t_linspace_fit, gv.mean(fit_curve['ss']), color='blue', linestyle='--', label='ss fit')
            ax_E0.plot(t_linspace_fit, gv.mean(fit_curve['ps']), color='orange', linestyle='--', label='ps fit')
            # Shade fit uncertainty
            ax_E0.fill_between(t_linspace_fit,
                               gv.mean(fit_curve['ss']) - gv.sdev(fit_curve['ss']),
                               gv.mean(fit_curve['ss']) + gv.sdev(fit_curve['ss']),
                               color='blue', alpha=0.2)
            ax_E0.fill_between(t_linspace_fit,
                               gv.mean(fit_curve['ps']) - gv.sdev(fit_curve['ps']),
                               gv.mean(fit_curve['ps']) + gv.sdev(fit_curve['ps']),
                               color='orange', alpha=0.2)
            ax_E0.legend(fontsize=8, ncol=2)

            if len(ylims) == 0:
                ax_E0.set_ylim(E0_fit_mean - 20*E0_fit_err, E0_fit_mean +20*E0_fit_err)
            else:
                ax_E0.set_ylim(ylims)
            # ax_E0.set_ylim(.3, .35)

            ax_E0.axvspan(0, fit_tmin, color='k', alpha=0.1, hatch='/')
            ax_E0.axvspan(t_max_fixed, T / 2, color='k', alpha=0.1, hatch='/')
    ax_E0.set_xlim(0 , T // 2)

    title = f"Stability of $E_0$ for {hadron} on {ensemble}"
    if fit_tmin is not None and fit_num_exps is not None:
        title += f" (highlighted: t_min={fit_tmin}, Nexp={fit_num_exps + 1}, Q={results[(fit_tmin, fit_num_exps)]['Q']:.2f}, chi^2dof={results[(fit_tmin, fit_num_exps)]['chi2/dof']:.2f})"
    print(title)
    ax_E0.set_title(title)
    ax_E0.legend(fontsize=8, ncol=2)
    ax_E0.grid(True, alpha=0.3)



    # ---------- 2. Q-value scatter ----------
    for num_exps in range(0, num_exps_max + 1):
        t_vals, q_vals = [], []
        for t_min in t_min_list:
            key = (t_min, num_exps)
            if key not in results: continue
            fit = results[key]['fit']
            t_vals.append(t_min + (num_exps - 1) * epsilon)
            q_vals.append(fit.Q)
        if t_vals:
            ax_Q.scatter(t_vals, q_vals, color=color[num_exps], marker='x', s=60,
                         label=f"Nexp={num_exps+1}", alpha=0.7)

    ax_Q.set_ylabel("Q-value")
    ax_Q.set_xlabel(r"$t_{\min}$")
    ax_Q.set_title("Fit quality vs $t_{min}$")
    ax_Q.legend(fontsize=8)
    ax_Q.grid(True, alpha=0.3)

    # ---------- 3. logGBF weights ----------
    # Compute normalized weights wᵢ = exp(logGBFᵢ) / Σ exp(logGBFᵢ)
    epsilon = 1. / (num_exps_max + 5)

    for t_min in t_min_list:
        # collect logGBFs for all Nexp for this t_min
        logGBFs = []
        available_exps = []
        for num_exps in range(0, num_exps_max + 1):
            key = (t_min, num_exps)
            if key in results:
                logGBFs.append(results[key]['fit'].logGBF)
                available_exps.append(num_exps)

        # convert to normalized weights
        if logGBFs:
            logGBFs = np.array(logGBFs)
            weights = np.exp(logGBFs - np.max(logGBFs))  # stability trick
            weights /= np.sum(weights)

            # scatter each Nexp's weight
            for w, num_exps in zip(weights, available_exps):
                ax_w.scatter(t_min + (num_exps - 1) * epsilon, w,
                             color=color[num_exps],
                             marker='o', s=50, alpha=0.8)

    ax_w.set_xlabel(r"$t_{\min}$")
    ax_w.set_ylabel(r"Model weight $w_i$")
    ax_w.set_title(r"Bayesian model weights $w_i$ vs $t_{\min}$ (colored by $N_{\text{exp}}$)")
    ax_w.legend([plt.Line2D([0], [0], color=color[n], marker='o', linestyle='')
                 for n in range(num_exps_max + 1)],
                [f"Nexp = {n+1}" for n in range(num_exps_max + 1)],
                fontsize=8, ncol=2)
    ax_w.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(f"{outputdir}/{hadron}_stability_summary.png", dpi=300)
    if show_plots:
        plt.show()
    if not show_plots:
        plt.close(fig)

    if fit_tmin:
        return E0_fit

def stability_to_latex_file(ensemble):
    outputdir = f"correlator_results/plots/{ensemble}"
    latex_file = f"{outputdir}/{ensemble}_fits.tex"

    # Gather all .png files in the folder
    images = sorted([f for f in os.listdir(outputdir) if f.endswith(".png")])

    with open(latex_file, "w") as tex:
        tex.write(r"\documentclass[11pt]{article}" "\n")
        tex.write(r"\usepackage{graphicx}" "\n")
        tex.write(r"\usepackage[margin=1in]{geometry}" "\n")
        tex.write(r"\usepackage{float}" "\n")
        tex.write(r"\usepackage{caption}" "\n")
        tex.write(r"\captionsetup{font=small,labelformat=empty}" "\n")
        tex.write(r"\begin{document}" "\n\n")

        tex.write(r"\begin{center}" "\n")
        tex.write(rf"\Huge Correlator Fits and Ratios for {ensemble}" "\n")
        tex.write(r"\end{center}" "\n\n")
        tex.write(r"\vspace{1cm}" "\n\n")

        for img in images:
            label = os.path.splitext(img)[0].replace("_", r"\_")
            tex.write(r"\begin{figure}[H]" "\n")
            tex.write(r"\centering" "\n")
            tex.write(rf"\includegraphics[width=\textwidth]{{{img}}}" "\n")
            tex.write(rf"\caption*{{\texttt{{{label}}}}}" "\n")
            tex.write(r"\end{figure}" "\n\n")
            tex.write(r"\clearpage" "\n\n")

        tex.write(r"\end{document}" "\n")

    print(f"LaTeX file generated: {latex_file}")


def make_dataframe_fit_summary(datafile, fits_dir="correlator_results/fits"):
    rows = []
    data = h5py.File(datafile, "r")

    for ensemble in os.listdir(fits_dir):
        ensemble_dir = os.path.join(fits_dir, ensemble)
        if not os.path.isdir(ensemble_dir):
            continue

        # extract lattice spacing from first 3 chars, e.g. a09 -> 0.09 fm
        match = re.match(r"a(\d{2})", ensemble)
        a = float(match.group(1)) / 100 if match else float("nan")

        for fname in os.listdir(ensemble_dir):
            if not fname.startswith("ground_state_energies_") or not fname.endswith(".txt"):
                continue

            filepath = os.path.join(ensemble_dir, fname)
            E0, tmin, tmax, num_exps, Q, chi2_dof, logBF = load_ground_state_energy(filepath)
            hadron = fname.replace("ground_state_energies_", "").replace(".txt", "")

            if hadron not in data[ensemble]:
                print(f"Skipping {hadron} — not found in {ensemble} data")
                continue

            # compute physical time window
            tmin_phys = tmin*a

            hadron_type = HADRON_TYPES.get(hadron, "unknown")

            rows.append({
                "hadron": hadron,
                "ensemble": ensemble,
                "type": hadron_type,
                "tmin": tmin,
                "tmax": tmax,
                r"$t_{\mathrm{min}} \times a$": tmin_phys,
                "nexp": num_exps + 1,
                "E0_mean": gv.mean(E0),
                "E0_sdev": gv.sdev(E0),
                "Q": Q,
                "chi2_dof": chi2_dof,
                "logBF": logBF,
            })

    df = pd.DataFrame(rows)
    data.close()
    return df

def export_dataframe_fit_summary_latex(df, outpath="fit_summary_table.tex"):
    df = df.round(5)
    df = df.sort_values(["hadron", "ensemble"])

    # replace underscores in column names with escaped versions
    df.columns = [c.replace("_", r"\_") for c in df.columns]
    df["hadron"] = df["hadron"].str.replace("_", r"\_", regex=False)
    df["ensemble"] = df["ensemble"].str.replace("_", r"\_", regex=False)

    table = df.to_latex(index=False, float_format="%.5f", escape=False)

    latex = r"""
\documentclass{standalone}
\usepackage{booktabs}
\begin{document}
""" + table + r"""
\end{document}
"""

    with open(outpath, "w") as f:
        f.write(latex)

    print(f"Saved full LaTeX document → {outpath}")


if __name__ == "__main__":
    pass