import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# User-specified parameter sets
# -----------------------------
n_list = [4, 6, 8, 10]   # choose exactly 4 n values

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


def lower_envelope(x, y):
    """
    Monotonically non-increasing lower envelope.

    For each unique x, first takes the minimum y-value.
    Then enforces monotonic decrease as x increases.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return np.array([]), np.array([])

    x_unique = np.unique(x)
    y_min = np.array([np.min(y[x == xu]) for xu in x_unique])

    order = np.argsort(x_unique)
    x_env = x_unique[order]
    y_min = y_min[order]

    y_env = np.minimum.accumulate(y_min)

    return x_env, y_env


def fit_power_law(x, y):
    """
    Fit y = A * x^alpha on log-log scale.
    Returns alpha, A, x_fit, y_fit.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan, np.nan, np.array([]), np.array([])

    logx = np.log10(x)
    logy = np.log10(y)

    alpha, logA = np.polyfit(logx, logy, 1)
    A = 10 ** logA

    x_fit = np.linspace(np.min(x), np.max(x), 200)
    y_fit = A * x_fit ** alpha

    return alpha, A, x_fit, y_fit


def collect_error_data_for_n(n):
    """
    For a fixed n, load all available files over M2_list and M3_list.

    Returns:
        x_vals: total samples
        if_vals: average infidelity over non-empty seeds
        td_vals: average trace distance over non-empty seeds
    """
    folder = os.path.join(base_folder, f"n={n}")

    x_vals = []
    if_vals = []
    td_vals = []

    if not os.path.isdir(folder):
        print(f"Missing folder: {folder}")
        return np.array([]), np.array([]), np.array([])

    for M2 in M2_list:
        M1 = (M2 + 1) // 2

        for M3 in M3_list:
            fname = f"Error_n={n}_M1={M1}_M2={M2}_M3={M3}.npy"
            fpath = os.path.join(folder, fname)

            if not os.path.exists(fpath):
                continue

            avg_inf, avg_td = load_error_tuple_file(fpath)
            total_samples = total_measurements(M1, M2, M3)

            if np.isfinite(avg_inf) or np.isfinite(avg_td):
                x_vals.append(total_samples)
                if_vals.append(avg_inf)
                td_vals.append(avg_td)

    x_vals = np.array(x_vals, dtype=float)
    if_vals = np.array(if_vals, dtype=float)
    td_vals = np.array(td_vals, dtype=float)

    order = np.argsort(x_vals)
    return x_vals[order], if_vals[order], td_vals[order]


# -----------------------------
# Plot: 4 by 2 subplots
# -----------------------------
if len(n_list) != 4:
    raise ValueError("Please specify exactly 4 n values in n_list.")

fig, axes = plt.subplots(4, 2, figsize=(13, 18), sharex=False, sharey=False)

for row, n in enumerate(n_list):
    x_vals, if_vals, td_vals = collect_error_data_for_n(n)

    # -------------------------
    # Infidelity subplot
    # -------------------------
    ax = axes[row, 0]

    x_if_env, y_if_env = lower_envelope(x_vals, if_vals)
    alpha_if, A_if, x_if_fit, y_if_fit = fit_power_law(x_if_env, y_if_env)

    ax.scatter(
        x_vals, if_vals,
        alpha=0.55, s=35,
        label="Average over non-empty seeds"
    )

    if len(x_if_env) > 0:
        ax.step(
            x_if_env, y_if_env,
            where="post", linewidth=2.2,
            label="Lower envelope"
        )

    if len(x_if_fit) > 0:
        ax.plot(
            x_if_fit, y_if_fit,
            "--", linewidth=2.2,
            label=fr"Fit: $y \sim {A_if:.2g} M^{{{alpha_if:.2f}}}$"
        )

    ax.set_title(f"n = {n}: Infidelity")
    ax.set_xlabel(r"Total samples $2M_1 + 2M_2 + 3M_3$")
    ax.set_ylabel("Average infidelity")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=8, loc="lower left")

    # -------------------------
    # Trace-distance subplot
    # -------------------------
    ax = axes[row, 1]

    x_td_env, y_td_env = lower_envelope(x_vals, td_vals)
    alpha_td, A_td, x_td_fit, y_td_fit = fit_power_law(x_td_env, y_td_env)

    ax.scatter(
        x_vals, td_vals,
        alpha=0.55, s=35,
        label="Average over non-empty seeds"
    )

    if len(x_td_env) > 0:
        ax.step(
            x_td_env, y_td_env,
            where="post", linewidth=2.2,
            label="Lower envelope"
        )

    if len(x_td_fit) > 0:
        ax.plot(
            x_td_fit, y_td_fit,
            "--", linewidth=2.2,
            label=fr"Fit: $y \sim {A_td:.2g} M^{{{alpha_td:.2f}}}$"
        )

    ax.set_title(f"n = {n}: Trace distance")
    ax.set_xlabel(r"Total samples $2M_1 + 2M_2 + 3M_3$")
    ax.set_ylabel("Average trace distance")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=8, loc="lower left")

plt.tight_layout()
plt.savefig("Figs/04_v2.jpg", dpi=500)
plt.show()


# -----------------------------
# Print fitted exponents
# -----------------------------
print("Power-law fits from lower envelopes:")
for n in n_list:
    x_vals, if_vals, td_vals = collect_error_data_for_n(n)

    x_if_env, y_if_env = lower_envelope(x_vals, if_vals)
    alpha_if, A_if, _, _ = fit_power_law(x_if_env, y_if_env)

    x_td_env, y_td_env = lower_envelope(x_vals, td_vals)
    alpha_td, A_td, _, _ = fit_power_law(x_td_env, y_td_env)

    print(f"n = {n}")
    print(f"  Infidelity:     error ~ {A_if:.4g} * M^({alpha_if:.6f})")
    print(f"  Trace distance: error ~ {A_td:.4g} * M^({alpha_td:.6f})")