import matplotlib.pyplot as plt
import numpy as np

from main import *

# ----------------------------
# Parameters
# ----------------------------
n_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
M_list = [i * 10 ** j for j in [0, 1, 2, 3, 4, 5] for i in [1, 4, 7]]
N_list = [i * 10 ** j for j in [0, 1, 2, 3, 4, 5] for i in [1, 4, 7]]
seed_list = np.arange(20)

target_error = 5e-2  # default fixed error

# ----------------------------
# Storage
# ----------------------------
min_samples_vs_n = []

# ----------------------------
# Main loop
# ----------------------------
for n in n_list:

    min_samples = np.inf

    for M in M_list:
        for N in N_list:

            try:
                errors = np.load(f"Data/01/n={n}/M={M}_N={N}_errors.npy")
            except FileNotFoundError:
                continue

            # Clean invalid values
            errors[(errors > 1) | (errors < 0)] = np.nan

            error = np.nanmean(errors[1, :])

            if np.isnan(error):
                continue

            total_samples = 2 * M + 3 * N

            # Check if error satisfies target
            if error <= target_error:
                min_samples = min(min_samples, total_samples)

    if np.isfinite(min_samples):
        min_samples_vs_n.append(min_samples)
    else:
        min_samples_vs_n.append(np.nan)

# ----------------------------
# Plot
# ----------------------------
plt.figure()
plt.plot(n_list, min_samples_vs_n, marker='o')
plt.plot(n_list, 2 ** np.array(n_list), label=r"$2^n$")
plt.plot(n_list, 3 ** np.array(n_list), label=r"$3^n$")
#plt.plot(n_list, 4 ** np.array(n_list), label=r"$4^n$")

plt.yscale("log")
#plt.xscale("log")
plt.xlabel("Number of qubits (n)")
plt.ylabel("Minimum samples (2M + 3N)")
plt.title(f"Samples vs Qubits (error ≤ {target_error}) v2")
plt.legend()

plt.grid(True, which="both", ls="--")
plt.show()
