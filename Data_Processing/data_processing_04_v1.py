import os
import re
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# User-specified parameter sets
# -----------------------------
n_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# Separate target errors
target_error_if = 0.05   # infidelity threshold
target_error_td = 0.1    # trace-distance threshold

M2_list = [i * 10 ** j for j in [0, 1, 2, 3, 4, 5] for i in [1, 3, 5, 7, 9]]
M3_list = [i * 10 ** j for j in [0, 1, 2, 3, 4, 5] for i in [1, 3, 5, 7, 9]]

base_folder = "Data/04"


# -----------------------------
# Helper functions
# -----------------------------
def total_measurements(M1, M2, M3):
    return 2 * M1 + 2 * M2 + 3 * M3


def load_error_tuple_file(fpath):
    """
    Loads one .npy file whose data is expected to be a length-20 array/list:
        [(infidelity, trace_distance), ...]
    Empty entries or invalid entries are ignored.

    Returns:
        avg_infidelity, avg_trace_distance
    """
    try:
        data = np.load(fpath, allow_pickle=True)
    except Exception as e:
        print(f"Could not load {fpath}: {e}")
        return np.nan, np.nan

    inf_vals = []
    td_vals = []

    for item in data:
        if item is None:
            continue

        try:
            if len(item) < 2:
                continue
        except TypeError:
            continue

        try:
            inf = float(item[0])
            td = float(item[1])
        except Exception:
            continue

        if np.isfinite(inf) and 0 <= inf <= 1:
            inf_vals.append(inf)

        if np.isfinite(td) and 0 <= td <= 1:
            td_vals.append(td)

    avg_inf = np.mean(inf_vals) if len(inf_vals) > 0 else np.nan
    avg_td = np.mean(td_vals) if len(td_vals) > 0 else np.nan

    return avg_inf, avg_td


def scaled_reference_curves(n_array, y_sim):
    """
    Scale 2^n, 3^n, and 4^n so that all curves start from the same
    first finite simulation y-value.
    """
    mask = np.isfinite(y_sim) & (y_sim > 0)

    if not np.any(mask):
        nan_curve = np.full_like(n_array, np.nan)
        return nan_curve, nan_curve, nan_curve

    first_idx = np.where(mask)[0][0]
    n0 = n_array[first_idx]
    y0 = y_sim[first_idx]

    ref_2n = y0 * (2 ** (n_array - n0))
    ref_3n = y0 * (3 ** (n_array - n0))
    ref_4n = y0 * (4 ** (n_array - n0))

    return ref_2n, ref_3n, ref_4n


# -----------------------------
# Main loop: find minimum samples
# -----------------------------
min_samples_if_vs_n = []
min_samples_td_vs_n = []

for n in n_list:
    folder = os.path.join(base_folder, f"n={n}")

    min_samples_if = np.inf
    min_samples_td = np.inf

    if not os.path.isdir(folder):
        print(f"Missing folder: {folder}")
        min_samples_if_vs_n.append(np.nan)
        min_samples_td_vs_n.append(np.nan)
        continue

    for M2 in M2_list:
        M1 = (M2 + 1) // 2

        for M3 in M3_list:
            fname = f"Error_n={n}_M1={M1}_M2={M2}_M3={M3}.npy"
            fpath = os.path.join(folder, fname)

            if not os.path.exists(fpath):
                continue

            avg_inf, avg_td = load_error_tuple_file(fpath)
            total_samples = total_measurements(M1, M2, M3)

            if np.isfinite(avg_inf) and avg_inf <= target_error_if:
                min_samples_if = min(min_samples_if, total_samples)

            if np.isfinite(avg_td) and avg_td <= target_error_td:
                min_samples_td = min(min_samples_td, total_samples)

    min_samples_if_vs_n.append(
        min_samples_if if np.isfinite(min_samples_if) else np.nan
    )
    min_samples_td_vs_n.append(
        min_samples_td if np.isfinite(min_samples_td) else np.nan
    )

min_samples_if_vs_n = np.array(min_samples_if_vs_n, dtype=float)
min_samples_td_vs_n = np.array(min_samples_td_vs_n, dtype=float)


# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

n_array = np.array(n_list, dtype=float)

ref_2n_if, ref_3n_if, ref_4n_if = scaled_reference_curves(
    n_array, min_samples_if_vs_n
)
ref_2n_td, ref_3n_td, ref_4n_td = scaled_reference_curves(
    n_array, min_samples_td_vs_n
)

# Infidelity
axes[0].plot(
    n_list, min_samples_if_vs_n,
    marker="o", linewidth=2, label="Minimum samples"
)
axes[0].plot(
    n_array, ref_2n_if,
    "--", label=r"Scaled $2^n$"
)
axes[0].plot(
    n_array, ref_3n_if,
    "--", label=r"Scaled $3^n$"
)
axes[0].plot(
    n_array, ref_4n_if,
    "--", label=r"Scaled $4^n$"
)
axes[0].set_title(f"Infidelity ≤ {target_error_if}")
axes[0].set_xlabel("Number of qubits n")
axes[0].set_ylabel(r"Minimum samples $2M_1 + 2M_2 + 3M_3$")
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].grid(True, which="both", linestyle="--", alpha=0.5)
axes[0].legend()

# Trace distance
axes[1].plot(
    n_list, min_samples_td_vs_n,
    marker="o", linewidth=2, label="Minimum samples"
)
axes[1].plot(
    n_array, ref_2n_td,
    "--", label=r"Scaled $2^n$"
)
axes[1].plot(
    n_array, ref_3n_td,
    "--", label=r"Scaled $3^n$"
)
axes[1].plot(
    n_array, ref_4n_td,
    "--", label=r"Scaled $4^n$"
)
axes[1].set_title(f"Trace distance ≤ {target_error_td}")
axes[1].set_xlabel("Number of qubits n")
axes[1].set_ylabel(r"Minimum samples $2M_1 + 2M_2 + 3M_3$")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].grid(True, which="both", linestyle="--", alpha=0.5)
axes[1].legend()

plt.tight_layout()
plt.show()


# -----------------------------
# Print results
# -----------------------------
print("n_list =", n_list)

print(f"Minimum samples for infidelity <= {target_error_if}:")
print(min_samples_if_vs_n)

print(f"Minimum samples for trace distance <= {target_error_td}:")
print(min_samples_td_vs_n)