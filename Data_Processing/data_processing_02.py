import os
import re
import numpy as np
import matplotlib.pyplot as plt


DATA_DIR = "Data/02"

# Matches filenames like:
# delta_verification_M=1e4_delta=1e-4.npy
FILENAME_PATTERN = re.compile(
    r"delta_verification_n=4_M=([0-9.]+e[+-]?[0-9]+)_delta=([0-9.]+e[+-]?[0-9]+)\.npy"
)


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
        fidelity = np.load(path)
        errors = 1-fidelity


        data[(M, delta)] = {
            "errors": errors,
            "mean": float(np.mean(errors)),
            "std": float(np.std(errors)),
        }

    if not data:
        raise FileNotFoundError(
            f"No matching .npy files found in {data_dir!r} with names like "
            f"'delta_verification_M=1e4_delta=1e-4.npy'."
        )

    return data


def available_parameters(data):
    """Return sorted lists of available M and delta values."""
    M_values = sorted({M for M, _ in data.keys()})
    delta_values = sorted({delta for _, delta in data.keys()})
    return M_values, delta_values


def sci_label(x):
    """
    Convert a float to a compact scientific-notation label like 1e4 or 4e-3.
    """
    s = f"{x:.0e}"
    s = s.replace("e+0", "e").replace("e-0", "e-").replace("e+", "e")
    return s


def plot_average_error_vs_delta(
    data,
    selected_M=None,
    ax=None,
    marker="o",
    linewidth=2,
):
    """
    Figure 1:
        average error vs delta
        different M are different lines

    Parameters
    ----------
    data : dict
        Output of load_delta_verification_data().
    selected_M : list[float] or None
        Which M values to plot. If None, plot all available M.
    ax : matplotlib axis or None
        Axis to draw on. If None, create a new figure.
    """
    all_M, all_delta = available_parameters(data)

    if selected_M is None:
        selected_M = all_M

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    for M in selected_M:
        x_vals = []
        y_vals = []

        for delta in all_delta:
            if (M, delta) in data:
                x_vals.append(delta)
                y_vals.append(data[(M, delta)]["mean"])

        if x_vals:
            ax.plot(
                x_vals,
                y_vals,
                marker=marker,
                linewidth=linewidth,
                label=f"M={sci_label(M)}",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel("Average error")
    ax.set_title("Average error vs delta")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    return ax


def plot_average_error_vs_M(
    data,
    selected_delta=None,
    ax=None,
    marker="o",
    linewidth=2,
):
    """
    Figure 2:
        average error vs M
        different delta are different lines

    Parameters
    ----------
    data : dict
        Output of load_delta_verification_data().
    selected_delta : list[float] or None
        Which delta values to plot. If None, plot all available delta.
    ax : matplotlib axis or None
        Axis to draw on. If None, create a new figure.
    """
    all_M, all_delta = available_parameters(data)

    if selected_delta is None:
        selected_delta = all_delta

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    for delta in selected_delta:
        x_vals = []
        y_vals = []

        for M in all_M:
            if (M, delta) in data:
                x_vals.append(M)
                y_vals.append(data[(M, delta)]["mean"])

        if x_vals:
            ax.plot(
                x_vals,
                y_vals,
                marker=marker,
                linewidth=linewidth,
                label=rf"$\delta$={sci_label(delta)}",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("M")
    ax.set_ylabel("Average error")
    ax.set_title("Average error vs M")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    return ax


if __name__ == "__main__":
    data = load_delta_verification_data(DATA_DIR)
    M_values, delta_values = available_parameters(data)

    print("Available M values:")
    print([sci_label(M) for M in M_values])
    print("Available delta values:")
    print([sci_label(delta) for delta in delta_values])

    # Example 1: choose which M lines to show in Figure 1
    selected_M = [1e1, 1e2,1e3,1e4]   # change this list as you like

    plt.figure(figsize=(7, 5))
    plot_average_error_vs_delta(data, selected_M=selected_M)
    plt.tight_layout()
    plt.show()

    # Example 2: choose which delta lines to show in Figure 2
    selected_delta = [1e-1, 1e-3, 1e-5]   # change this list as you like

    plt.figure(figsize=(7, 5))
    plot_average_error_vs_M(data, selected_delta=selected_delta)
    plt.tight_layout()
    plt.show()