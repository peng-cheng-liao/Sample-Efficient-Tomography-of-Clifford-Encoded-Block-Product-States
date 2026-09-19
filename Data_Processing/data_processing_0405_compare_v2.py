import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# User-specified parameter sets
# -----------------------------
n_list = [4, 6, 8, 10]   # choose exactly 4 n values

# Only fit the lower-envelope error data with total samples at or above this.
fit_min_total_samples = 1e3

M2_list = [i * 10 ** j for j in [0, 1, 2, 3, 4, 5] for i in [1, 3, 5, 7, 9]]
M3_list = [i * 10 ** j for j in [0, 1, 2, 3, 4, 5] for i in [1, 3, 5, 7, 9]]
M_list = [i * 10 ** j for j in [0, 1, 2, 3, 4, 5] for i in [1, 3, 5, 7, 9]]

base_folder_04 = "Data/04"
base_folder_05 = "Data/05"


# -----------------------------
# Helper functions
# -----------------------------
def total_measurements_04(M1, M2, M3):
    return 2 * M1 + 2 * M2 + 3 * M3


def total_measurements_05(n, M):
    return (3 ** n) * M


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


def fit_power_law(x, y, min_total_samples=0):
    """
    Fit y = A * x^alpha on log-log scale.
    Returns alpha, A, x_fit, y_fit.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = (
        np.isfinite(x) & np.isfinite(y) &
        (x > 0) & (y > 0) &
        (x >= min_total_samples)
    )
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


def collect_error_data_04_for_n(n):
    """
    For a fixed n, load all available Data/04 files over M2_list and M3_list.

    Returns:
        x_vals: total samples
        if_vals: average infidelity over non-empty seeds
        td_vals: average trace distance over non-empty seeds
    """
    folder = os.path.join(base_folder_04, f"n={n}")

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
            total_samples = total_measurements_04(M1, M2, M3)

            if np.isfinite(avg_inf) or np.isfinite(avg_td):
                x_vals.append(total_samples)
                if_vals.append(avg_inf)
                td_vals.append(avg_td)

    x_vals = np.array(x_vals, dtype=float)
    if_vals = np.array(if_vals, dtype=float)
    td_vals = np.array(td_vals, dtype=float)

    order = np.argsort(x_vals)
    return x_vals[order], if_vals[order], td_vals[order]


def collect_error_data_05_for_n(n):
    """
    For a fixed n, load all available Data/05 files over M_list.

    Returns:
        x_vals: total samples
        if_vals: average infidelity over non-empty seeds
        td_vals: average trace distance over non-empty seeds
    """
    folder = os.path.join(base_folder_05, f"n={n}")

    x_vals = []
    if_vals = []
    td_vals = []

    if not os.path.isdir(folder):
        print(f"Missing folder: {folder}")
        return np.array([]), np.array([]), np.array([])

    for M in M_list:
        fname = f"Error_n={n}_M={M}.npy"
        fpath = os.path.join(folder, fname)

        if not os.path.exists(fpath):
            continue

        avg_inf, avg_td = load_error_tuple_file(fpath)
        total_samples = total_measurements_05(n, M)

        if np.isfinite(avg_inf) or np.isfinite(avg_td):
            x_vals.append(total_samples)
            if_vals.append(avg_inf)
            td_vals.append(avg_td)

    x_vals = np.array(x_vals, dtype=float)
    if_vals = np.array(if_vals, dtype=float)
    td_vals = np.array(td_vals, dtype=float)

    order = np.argsort(x_vals)
    return x_vals[order], if_vals[order], td_vals[order]


def plot_dataset(ax, x_vals, y_vals, label, marker, color):
    x_env, y_env = lower_envelope(x_vals, y_vals)
    alpha, A, x_fit, y_fit = fit_power_law(
        x_env, y_env, min_total_samples=fit_min_total_samples
    )

    ax.scatter(
        x_vals, y_vals,
        alpha=0.35, s=30, marker=marker, color=color,
        label=f"{label}: averages"
    )

    if len(x_env) > 0:
        ax.step(
            x_env, y_env,
            where="post", linewidth=2.2, color=color,
            label=f"{label}: lower envelope"
        )

    if len(x_fit) > 0:
        ax.plot(
            x_fit, y_fit,
            "--", linewidth=2.0, color=color,
            label=fr"{label}: fit ${A:.0f} M^{{{alpha:.2f}}}$"
        )

    return alpha, A


# -----------------------------
# Plot: 4 by 2 subplots
# -----------------------------
if len(n_list) != 4:
    raise ValueError("Please specify exactly 4 n values in n_list.")

fig, axes = plt.subplots(4, 2, figsize=(14, 18), sharex=False, sharey=False)

fit_results = {}

for row, n in enumerate(n_list):
    x_04, if_04, td_04 = collect_error_data_04_for_n(n)
    x_05, if_05, td_05 = collect_error_data_05_for_n(n)

    # -------------------------
    # Infidelity subplot
    # -------------------------
    ax = axes[row, 0]

    alpha_if_04, A_if_04 = plot_dataset(ax, x_04, if_04, "RGST", "o", "C0")
    alpha_if_05, A_if_05 = plot_dataset(ax, x_05, if_05, "General", "s", "C1")

    ax.set_title(f"n = {n}: Infidelity")
    ax.set_xlabel("Total samples")
    ax.set_ylabel("Average infidelity")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=7, loc="lower left")

    # -------------------------
    # Trace-distance subplot
    # -------------------------
    ax = axes[row, 1]

    alpha_td_04, A_td_04 = plot_dataset(ax, x_04, td_04, "RGST", "o", "C0")
    alpha_td_05, A_td_05 = plot_dataset(ax, x_05, td_05, "General", "s", "C1")

    ax.set_title(f"n = {n}: Trace distance")
    ax.set_xlabel("Total samples")
    ax.set_ylabel("Average trace distance")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=7, loc="lower left")

    fit_results[n] = {
        "if_04": (alpha_if_04, A_if_04),
        "if_05": (alpha_if_05, A_if_05),
        "td_04": (alpha_td_04, A_td_04),
        "td_05": (alpha_td_05, A_td_05),
    }

plt.tight_layout()
plt.savefig("Figs/0405_compare_v2.jpg", dpi=200)
#plt.show()


# -----------------------------
# Print fitted exponents
# -----------------------------
print("Power-law fits from lower envelopes:")
for n in n_list:
    alpha_if_04, A_if_04 = fit_results[n]["if_04"]
    alpha_if_05, A_if_05 = fit_results[n]["if_05"]
    alpha_td_04, A_td_04 = fit_results[n]["td_04"]
    alpha_td_05, A_td_05 = fit_results[n]["td_05"]

    print(f"n = {n}")
    print(f"  RGST infidelity:        error ~ {A_if_04:.4g} * M^({alpha_if_04:.6f})")
    print(f"  General infidelity:     error ~ {A_if_05:.4g} * M^({alpha_if_05:.6f})")
    print(f"  RGST trace distance:    error ~ {A_td_04:.4g} * M^({alpha_td_04:.6f})")
    print(f"  General trace distance: error ~ {A_td_05:.4g} * M^({alpha_td_05:.6f})")
