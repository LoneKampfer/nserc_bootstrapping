#!/usr/bin/env python3
# using python3.12
import argparse
import os
import h5py as h5
import numpy as np
from bootstrapping import (
    prepare_gvdata,
    generate_boot0_and_mpi_data,
    generate_ensemble_indices,
    bootstrap_gvdata
)

# ----------------------------
#  LIST OF ENSEMBLES (29 sets)
# ----------------------------
ENSEMBLES = [
    'a06m220L','a06m310L',
    'a09m135','a09m220_o','a09m260','a09m310','a09m350','a09m400',
    'a12m130','a12m180L','a12m180S','a12m220','a12m220L','a12m220S','a12m220XL','a12m220_ms',
    'a12m260','a12m310','a12m310L','a12m310XL','a12m350','a12m400',
    'a15m135XL','a15m220','a15m260','a15m310','a15m310L','a15m350','a15m400'
]

# ---------------------------------
#  Number of bootstrap samples
# ---------------------------------
N_BOOTSTRAP = 5000

# ---------------------------------
#  Data file location
# ---------------------------------
DATA_FILE = "dataFiles/c51_2pt_hyperons.h5"

# ---------------------------------
#  Output directory
# ---------------------------------
OUTDIR = "bootstrap_results"
os.makedirs(OUTDIR, exist_ok=True)


def main(hadron):
    print(f"[bootstrap] Starting for hadron = {hadron}")
    data = h5.File(DATA_FILE, "r")

    # Output HDF5 for this job
    out_h5 = os.path.join(OUTDIR, f"bootstrap_{hadron}.h5")
    fout = h5.File(out_h5, "w")

    for ensemble in ENSEMBLES:
        print(f"[bootstrap] Ensemble: {ensemble}")

        if ensemble not in data:
            print(f"  -> ensemble missing in data file, skipping")
            continue
        if hadron not in data[ensemble]:
            print(f"  -> hadron missing on this ensemble, skipping")
            continue

        # 1. Load gvdata
        gvdata = prepare_gvdata(data, ensemble, hadron)
        n_cfg = gvdata.shape[0]

        # 2. Load boot0 fit + mpi cache
        boot0, mpi_cache, tmin, tmax, num_exps, tref = generate_boot0_and_mpi_data(ensemble, hadron)
        mpi = mpi_cache[ensemble]

        # 3. Generate bootstrap index table
        idx = generate_ensemble_indices(ensemble, n_cfg, N_BOOTSTRAP)

        # 4. Perform bootstrap fits
        energies = bootstrap_gvdata(
            gvdata,
            idx,
            boot0,
            mpi,
            tmin,
            tmax,
            num_exps,
            tref,
            verbose=False
        )

        # ---------------------------
        # Shift means to match boot0
        # ---------------------------
        boot_means = np.array([e.mean for e in energies[1:]])
        diff = boot0.mean - boot_means.mean()
        shifted = np.array([energies[0].mean] + list(boot_means + diff))

        # ---------------------------
        # Save into HDF5
        # ---------------------------
        if ensemble not in fout:
            fout.create_group(ensemble)

        dsetname = f"m_{hadron}"
        fout[ensemble].create_dataset(dsetname, data=shifted)

        print(f"  ✓ saved {ensemble}/{dsetname}  (n={len(shifted)})")

    data.close()
    fout.close()
    print(f"[bootstrap] Completed hadron={hadron}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hadron", type=str, help="Hadron name (e.g., lambda_z)")
    args = parser.parse_args()
    main(args.hadron)

    # usage: python nserc_bootstrapping_script.py lambda_z
    # wrap in bash script if needed