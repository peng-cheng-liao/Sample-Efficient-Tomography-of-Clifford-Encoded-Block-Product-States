import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# User-specified parameter sets
# -----------------------------
n = 4
M2_list = [i * 10 ** j for j in [0, 1, 2, 3, 4, 5] for i in [1, 4, 7]]
M3_list = [i * 10 ** j for j in [0, 1, 2, 3, 4, 5] for i in [1, 4, 7]]
seed_list = np.arange(20)

if_matrix = np.full((len(M2_list), len(M3_list), len(seed_list)), np.nan)
td_matrix = np.full((len(M2_list), len(M3_list), len(seed_list)), np.nan)

# -----------------------------
# Helper functions
# -----------------------------
def total_measurements(M1, M2, M3):
    return 2 * M1 + 2 * M2 + 3 * M3

def lower_envelope(x, y):
    """
    Monotonically non-increasing lower envelope.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    # minimum y for each unique x
    x_unique = np.unique(x)
    y_min = np.array([np.min(y[x == xu]) for xu in x_unique])

    # sort by x
    order = np.argsort(x_unique)
    x_env = x_unique[order]
    y_min = y_min[order]

    # enforce monotone decrease
    y_env = np.minimum.accumulate(y_min)

    return x_env, y_env

def fit_power_law(x, y):
    """
    Fit y = A * x^alpha on log-log scale.
    Returns alpha, A, y_fit.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    logx = np.log10(x)
    logy = np.log10(y)

    alpha, logA = np.polyfit(logx, logy, 1)
    A = 10 ** logA
    y_fit = A * x ** alpha
    return alpha, A, x, y_fit

# -----------------------------
# Load data
# -----------------------------
for i, M2 in enumerate(M2_list):
    M1 = (M2 + 1) // 2
    for j, M3 in enumerate(M3_list):
        for k, seed in enumerate(seed_list):
            fname = f"Jobs/03/Data/Error_n={n}_M1={M1}_M2={M2}_M3={M3}_seed={seed}.npy"
            if os.path.exists(fname):
                error = np.load(fname, allow_pickle=True)
                if_matrix[i, j, k] = float(error[0])   # infidelity
                td_matrix[i, j, k] = float(error[1])   # trace distance
            else:
                print(f"Missing file: {fname}")

# average over seeds
if_matrix = np.nanmean(if_matrix, axis=2)
td_matrix = np.nanmean(td_matrix, axis=2)

# -----------------------------
# Prepare scatter data
# -----------------------------
x_vals = []
if_vals = []
td_vals = []

for i, M2 in enumerate(M2_list):
    M1 = (M2 + 1) // 2
    for j, M3 in enumerate(M3_list):
        x = total_measurements(M1, M2, M3)
        x_vals.append(x)
        if_vals.append(if_matrix[i, j])
        td_vals.append(td_matrix[i, j])

x_vals = np.array(x_vals, dtype=float)
if_vals = np.array(if_vals, dtype=float)
td_vals = np.array(td_vals, dtype=float)

order = np.argsort(x_vals)
x_vals = x_vals[order]
if_vals = if_vals[order]
td_vals = td_vals[order]

# -----------------------------
# Lower envelopes
# -----------------------------
x_if_env, y_if_env = lower_envelope(x_vals, if_vals)
x_td_env, y_td_env = lower_envelope(x_vals, td_vals)

# -----------------------------
# Trend lines from lower envelopes
# -----------------------------
alpha_if, A_if, x_if_fit, y_if_fit = fit_power_law(x_if_env, y_if_env)
alpha_td, A_td, x_td_fit, y_td_fit = fit_power_law(x_td_env, y_td_env)

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

# Infidelity
axes[0].scatter(x_vals, if_vals, alpha=0.6, s=40, label="Average over seeds")
axes[0].step(x_if_env, y_if_env, where="post", linewidth=2.2, label="Lower envelope")
axes[0].plot(
    x_if_fit, y_if_fit, "--", linewidth=2.2,
    label=fr"Trend: $y \sim {A_if:.2f} M^{{{alpha_if:.2f}}}$"
)
axes[0].set_title("Infidelity")
axes[0].set_xlabel(r"Total measurements $2M_1 + 2M_2 + 3M_3$")
axes[0].set_ylabel("Average error over seeds")
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].grid(True, which="both", linestyle="--", alpha=0.5)
axes[0].legend()

# Trace distance
axes[1].scatter(x_vals, td_vals, alpha=0.6, s=40, label="Average over seeds")
axes[1].step(x_td_env, y_td_env, where="post", linewidth=2.2, label="Lower envelope")
axes[1].plot(
    x_td_fit, y_td_fit, "--", linewidth=2.2,
    label=fr"Trend: $y \sim {A_td:.2f} M^{{{alpha_td:.2f}}}$"
)
axes[1].set_title("Trace distance")
axes[1].set_xlabel(r"Total measurements $2M_1 + 2M_2 + 3M_3$")
axes[1].set_ylabel("Average error over seeds")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].grid(True, which="both", linestyle="--", alpha=0.5)
axes[1].legend()

plt.tight_layout()
plt.show()
print(f"Infidelity trend:     error ~ {A_if:.2f} * M^({alpha_if:.6f})")
print(f"Trace distance trend: error ~ {A_td:.2f} * M^({alpha_td:.6f})")