import os
import re
import numpy as np
import matplotlib.pyplot as plt

from main import *

# ----------------------------
# Parameters
# ----------------------------
n_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
seed_list = np.arange(20)          # not strictly needed, but kept for consistency
target_error = 5e-2                # default fixed error

# ----------------------------
# Storage
# ----------------------------
min_samples_vs_n = []

# ----------------------------
# Helper: parse filenames
# ----------------------------
pattern = re.compile(r"^M=(\d+)_N=(\d+)_errors\.npy$")

# ----------------------------
# Main loop
# ----------------------------
for n in n_list:
    folder = f"Data/01/n={n}/"
    min_samples = np.inf

    if not os.path.isdir(folder):
        min_samples_vs_n.append(np.nan)
        continue

    for fname in os.listdir(folder):
        m = pattern.match(fname)
        if not m:
            continue

        M = int(m.group(1))
        N = int(m.group(2))
        fpath = os.path.join(folder, fname)

        try:
            errors = np.load(fpath)
        except Exception:
            continue

        # Clean invalid values
        errors = errors.astype(float, copy=False)
        errors[(errors > 1) | (errors < 0)] = np.nan

        # Same metric as your template
        error = np.nanmean(errors[1, :])

        if np.isnan(error):
            continue

        total_samples = 2 * M + 3 * N

        if error <= target_error:
            min_samples = min(min_samples, total_samples)

    min_samples_vs_n.append(min_samples if np.isfinite(min_samples) else np.nan)

# ----------------------------
# Plot
# ----------------------------
plt.figure()
plt.plot(n_list, min_samples_vs_n, marker="o")
plt.plot(n_list, 2 ** np.array(n_list), label=r"$2^n$")
plt.plot(n_list, 3 ** np.array(n_list), label=r"$3^n$")
#plt.plot(n_list, 4 ** np.array(n_list), label=r"$4^n$")

plt.yscale("log")
plt.xscale("log")
plt.xlabel("Number of qubits (n)")
plt.ylabel("Minimum samples (2M + 3N)")
plt.title(f"Samples vs Qubits (error ≤ {target_error}) v3")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()