import os
import re
import numpy as np
import matplotlib.pyplot as plt


DATA_DIR = "Data/02"

# Matches filenames like:
# delta_verification_M=1e4_delta=1e-4.npy
FILENAME_PATTERN = re.compile(r"delta_verification_n=4_M=([0-9.]+e[+-]?[0-9]+)_delta=([0-9.]+e[+-]?[0-9]+)\.npy")


def load_delta_verification_data(data_dir=DATA_DIR):
    """
    Load all files of the form
        delta_verification_M=<M>_delta=<delta>.npy
    and return a dictionary:
        data[(M, delta)] = {
            "errors": np.ndarray,
            "mean": float,
            "std": float,
        }
    """
    data = {}

    for fname in os.listdir(data_dir):
        match = FILENAME_PATTERN.fullmatch(fname)
        if match is None:
            continue

        M_str, delta_str = match.groups()
        M = float(M_str)
        delta = float(delta_str)

        path = os.path.join(data_dir, fname)
        fidelitys = np.load(path)
        errors = 1-fidelitys


        data[(M, delta)] = {
            "errors": errors,
            "mean": float(np.mean(errors)),
            "std": float(np.std(errors)),}

    if not data:
        raise FileNotFoundError(
            f"No matching .npy files found in {data_dir!r}."
        )

    return data


def available_parameters(data):
    M_values = sorted({M for M, _ in data.keys()})
    delta_values = sorted({delta for _, delta in data.keys()})
    return M_values, delta_values


def sci_label(x):
    s = f"{x:.0e}"
    s = s.replace("e+0", "e").replace("e-0", "e-").replace("e+", "e")
    return s


def minimal_M_vs_delta(data, error_thresh):
    """
    For each delta, find the minimal M such that
        average error < error_thresh

    Returns
    -------
    delta_arr : np.ndarray
    M_min_arr : np.ndarray
        If no M satisfies the threshold for a delta, the entry is np.nan.
    """
    M_values, delta_values = available_parameters(data)

    delta_arr = []
    M_min_arr = []

    for delta in delta_values:
        M_min = np.nan
        for M in M_values:
            if (M, delta) in data and data[(M, delta)]["mean"] < error_thresh:
                M_min = M
                break

        delta_arr.append(delta)
        M_min_arr.append(M_min)

    return np.array(delta_arr), np.array(M_min_arr)


def plot_minimal_M_vs_delta(data, error_thresh, ax=None, marker="o", linewidth=2):
    """
    Plot delta on x-axis and the minimal M on y-axis, where
    minimal M is the smallest M satisfying average error < error_thresh.
    """
    delta_arr, M_min_arr = minimal_M_vs_delta(data, error_thresh)

    valid = ~np.isnan(M_min_arr)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(delta_arr[valid], M_min_arr[valid], marker=marker, linewidth=linewidth)

    # Mark deltas for which no M satisfies the condition
    if np.any(~valid):
        ax.scatter(delta_arr[~valid], np.full(np.sum(~valid), ax.get_ylim()[0]), marker="x")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel("Minimal $M$")
    ax.set_title(rf"Minimal $M$ vs $\delta$ (mean error < {error_thresh})")
    ax.grid(True, which="both", alpha=0.3)

    return ax


if __name__ == "__main__":
    data = load_delta_verification_data(DATA_DIR)

    error_thresh = 1e-2   # change this as needed

    delta_arr, M_min_arr = minimal_M_vs_delta(data, error_thresh)

    print(f"Threshold = {error_thresh}")
    for delta, M_min in zip(delta_arr, M_min_arr):
        if np.isnan(M_min):
            print(f"delta = {sci_label(delta)} -> no M satisfies the threshold")
        else:
            print(f"delta = {sci_label(delta)} -> minimal M = {sci_label(M_min)}")

    plt.figure(figsize=(7, 5))
    plot_minimal_M_vs_delta(data, error_thresh)
    plt.tight_layout()
    plt.show()