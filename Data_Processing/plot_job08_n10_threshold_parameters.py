#!/usr/bin/env python3
"""Plot Job 08 threshold/test parameters for the n=10 experiments.

Each point is the arithmetic mean over 20 independently optimized states, and
each error bar is one sample standard deviation (ddof=1).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Keep Matplotlib runnable when the user's normal cache directory is read-only.
_MPL_CONFIG = tempfile.TemporaryDirectory(prefix="job08-n10-thresholds-mpl-")
os.environ.setdefault("MPLCONFIGDIR", _MPL_CONFIG.name)

import matplotlib

matplotlib.set_loglevel("error")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


N_QUBITS = 10
BLOCK_SIZES = (1, 2, 3)
EXPECTED_STATES_PER_GROUP = 20
PARAMETERS = ("h_min", "h_max", "theta", "eta_test")
OUTPUT_STEM = "job08_n10_threshold_parameters_vs_copies"


class DataValidationError(RuntimeError):
    """Raised when the input cannot support the requested statistics."""


def violation_preview(data: pd.DataFrame, mask: pd.Series) -> str:
    """Format a compact preview of rows that violate a numeric constraint."""

    columns = ["d", "state_index", "budget", *PARAMETERS]
    violating = data.loc[mask, columns]
    preview = violating.head(10).to_string(index=False)
    remainder = len(violating) - min(len(violating), 10)
    suffix = f"\n... and {remainder} more row(s)" if remainder else ""
    return preview + suffix


def load_and_validate(csv_path: Path) -> tuple[pd.DataFrame, int]:
    """Load the CSV and return its validated n=10 rows plus total row count."""

    if not csv_path.is_file():
        raise DataValidationError(f"source CSV is missing: {csv_path}")

    data = pd.read_csv(csv_path)
    source_rows = len(data)
    if "theta" not in data.columns:
        if "theta_tau_multiplier" in data.columns:
            raise DataValidationError(
                "CSV update is incomplete: theta is absent and "
                "theta_tau_multiplier is still present"
            )
        raise DataValidationError("source CSV is missing the required theta column")

    required = {"n", "d", "state_index", "budget", *PARAMETERS}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise DataValidationError(f"source CSV is missing columns: {missing}")

    n_values = pd.to_numeric(data["n"], errors="coerce")
    selected = data.loc[n_values.eq(N_QUBITS)].copy()
    if selected.empty:
        raise DataValidationError(f"source CSV contains no n={N_QUBITS} rows")

    numeric_columns = ["n", "d", "state_index", "budget", *PARAMETERS]
    selected[numeric_columns] = selected[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    nonfinite = ~np.isfinite(selected[numeric_columns].to_numpy(dtype=float))
    if nonfinite.any():
        rows = np.flatnonzero(nonfinite.any(axis=1))[:10]
        raise DataValidationError(
            "selected rows contain non-numeric or non-finite values at row "
            f"positions {rows.tolist()}"
        )

    observed_d = set(selected["d"].unique())
    if observed_d != set(BLOCK_SIZES):
        raise DataValidationError(
            f"n={N_QUBITS} has block sizes {sorted(observed_d)}; "
            f"expected {list(BLOCK_SIZES)}"
        )
    if (selected["budget"] <= 0).any():
        raise DataValidationError("all plotted budgets must be positive")

    budgets_by_d = {
        d: set(selected.loc[selected["d"].eq(d), "budget"].unique())
        for d in BLOCK_SIZES
    }
    reference_budgets = budgets_by_d[BLOCK_SIZES[0]]
    if any(budgets != reference_budgets for budgets in budgets_by_d.values()):
        details = ", ".join(
            f"d={d}: {sorted(budgets)}" for d, budgets in budgets_by_d.items()
        )
        raise DataValidationError(
            "block sizes do not share the same fixed budget grid; " + details
        )

    group_columns = ["d", "budget"]
    row_counts = selected.groupby(group_columns, sort=True).size()
    bad_counts = row_counts[row_counts.ne(EXPECTED_STATES_PER_GROUP)]
    if not bad_counts.empty:
        details = ", ".join(
            f"(d={int(d)}, budget={budget:g}): {int(count)} rows"
            for (d, budget), count in bad_counts.items()
        )
        raise DataValidationError(
            f"expected {EXPECTED_STATES_PER_GROUP} rows per (d, budget); {details}"
        )

    unique_states = selected.groupby(group_columns, sort=True)["state_index"].nunique()
    bad_states = unique_states[unique_states.ne(EXPECTED_STATES_PER_GROUP)]
    if not bad_states.empty:
        details = ", ".join(
            f"(d={int(d)}, budget={budget:g}): {int(count)} unique states"
            for (d, budget), count in bad_states.items()
        )
        raise DataValidationError(
            "duplicate or missing state indices within a complete group; " + details
        )

    h_violation = ~(
        (selected["h_min"] > 0.5)
        & (selected["h_min"] < selected["h_max"])
        & (selected["h_max"] < 1.0)
    )
    if h_violation.any():
        raise DataValidationError(
            "rows violate 0.5 < h_min < h_max < 1.0:\n"
            + violation_preview(selected, h_violation)
        )

    theta_violation = ~(
        (selected["theta"] > 0.0) & (selected["theta"] <= 0.5)
    )
    if theta_violation.any():
        raise DataValidationError(
            "rows violate 0 < theta <= 0.5:\n"
            + violation_preview(selected, theta_violation)
        )

    eta_violation = selected["eta_test"] <= 0.0
    if eta_violation.any():
        raise DataValidationError(
            "rows violate eta_test > 0:\n"
            + violation_preview(selected, eta_violation)
        )

    return selected, source_rows


def summarize(data: pd.DataFrame, d: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return sorted arithmetic means and sample standard deviations for d."""

    grouped = data.loc[data["d"].eq(d)].groupby("budget", sort=True)[
        list(PARAMETERS)
    ]
    means = grouped.mean().sort_index()
    standard_deviations = grouped.std(ddof=1).sort_index()
    if not (
        np.isfinite(means.to_numpy(dtype=float)).all()
        and np.isfinite(standard_deviations.to_numpy(dtype=float)).all()
    ):
        raise DataValidationError(f"d={d} summary contains non-finite statistics")
    if not means.index.equals(standard_deviations.index):
        raise DataValidationError(f"d={d} mean/std budget grids do not match")
    return means, standard_deviations


def eta_axis_limits(
    summaries: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
) -> tuple[float, float]:
    """Derive a shared eta range from the full mean ± sample-SD envelope."""

    lower = min(
        float((means["eta_test"] - deviations["eta_test"]).min())
        for means, deviations in summaries.values()
    )
    upper = max(
        float((means["eta_test"] + deviations["eta_test"]).max())
        for means, deviations in summaries.values()
    )
    span = upper - lower
    margin = 0.05 * span if span > 0 else max(0.01, 0.05 * abs(upper))
    return max(0.0, lower - margin), upper + margin


def draw_errorbar(
    ax: plt.Axes,
    budgets: np.ndarray,
    means: pd.Series,
    deviations: pd.Series,
    *,
    label: str | None,
    color: tuple[float, float, float, float] | tuple[float, float, float],
    marker: str,
) -> None:
    """Draw one consistently styled mean ± sample-SD series."""

    ax.errorbar(
        budgets,
        means.to_numpy(dtype=float),
        yerr=deviations.to_numpy(dtype=float),
        label=label,
        color=color,
        marker=marker,
        linestyle="-",
        linewidth=1.4,
        markersize=4.2,
        capsize=2.3,
        capthick=0.75,
        elinewidth=0.75,
    )


def plot(
    summaries: dict[int, tuple[pd.DataFrame, pd.DataFrame]], output_dir: Path
) -> tuple[Path, Path, tuple[float, float]]:
    """Create and atomically save the publication-quality 3x3 figure."""

    plt.rcParams.update(
        {
            "font.size": 9.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 9.0,
        }
    )
    tab10 = plt.get_cmap("tab10").colors
    styles = {
        "h_min": (tab10[0], "o"),
        "h_max": (tab10[1], "s"),
        "theta": (tab10[2], "^"),
        "eta_test": (tab10[3], "D"),
    }
    eta_limits = eta_axis_limits(summaries)

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(13.2, 9.8),
        sharex=True,
        sharey="row",
        constrained_layout=False,
    )
    panel_labels = tuple(f"({chr(ord('a') + index)})" for index in range(9))

    for column, d in enumerate(BLOCK_SIZES):
        means, deviations = summaries[d]
        budgets = means.index.to_numpy(dtype=float)

        draw_errorbar(
            axes[0, column],
            budgets,
            means["h_min"],
            deviations["h_min"],
            label=r"$h_{\min}$",
            color=styles["h_min"][0],
            marker=styles["h_min"][1],
        )
        draw_errorbar(
            axes[0, column],
            budgets,
            means["h_max"],
            deviations["h_max"],
            label=r"$h_{\max}$",
            color=styles["h_max"][0],
            marker=styles["h_max"][1],
        )
        draw_errorbar(
            axes[1, column],
            budgets,
            means["theta"],
            deviations["theta"],
            label=None,
            color=styles["theta"][0],
            marker=styles["theta"][1],
        )
        draw_errorbar(
            axes[2, column],
            budgets,
            means["eta_test"],
            deviations["eta_test"],
            label=None,
            color=styles["eta_test"][0],
            marker=styles["eta_test"][1],
        )

        axes[0, column].set_title(rf"$d={d}$")
        for row in range(3):
            ax = axes[row, column]
            ax.set_xscale("log")
            ax.set_axisbelow(True)
            ax.grid(
                True,
                which="major",
                linestyle="--",
                linewidth=0.5,
                alpha=0.35,
            )
            ax.text(
                0.035,
                0.93,
                panel_labels[row * 3 + column],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.8,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                    "pad": 0.55,
                },
            )
            if row < 2:
                ax.tick_params(axis="x", which="both", labelbottom=False)
            else:
                ax.set_xlabel(r"Total copies, $N$")

    for ax in axes[0, :]:
        ax.set_ylim(0.5, 1.0)
    for ax in axes[1, :]:
        ax.set_ylim(0.0, 0.5)
    for ax in axes[2, :]:
        ax.set_ylim(*eta_limits)

    axes[0, 0].set_ylabel(r"$h_{\min},\,h_{\max}$")
    axes[1, 0].set_ylabel(r"$\theta$")
    axes[2, 0].set_ylabel(r"$\eta_{\mathrm{test}}$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=True,
        columnspacing=1.3,
        handlelength=2.1,
        handletextpad=0.5,
        borderpad=0.4,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.075,
        top=0.935,
        hspace=0.12,
        wspace=0.10,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"
    png_path = output_dir / f"{OUTPUT_STEM}.png"
    temporary_pdf = tempfile.NamedTemporaryFile(
        prefix=f"{OUTPUT_STEM}-", suffix=".pdf", dir=output_dir, delete=False
    )
    temporary_png = tempfile.NamedTemporaryFile(
        prefix=f"{OUTPUT_STEM}-", suffix=".png", dir=output_dir, delete=False
    )
    temporary_pdf_path = Path(temporary_pdf.name)
    temporary_png_path = Path(temporary_png.name)
    temporary_pdf.close()
    temporary_png.close()
    try:
        fig.savefig(temporary_pdf_path, format="pdf")
        fig.savefig(temporary_png_path, format="png", dpi=350)
        temporary_pdf_path.replace(pdf_path)
        temporary_png_path.replace(png_path)
    finally:
        plt.close(fig)
        temporary_pdf_path.unlink(missing_ok=True)
        temporary_png_path.unlink(missing_ok=True)

    return pdf_path, png_path, eta_limits


def main() -> None:
    """Validate the source, calculate statistics, and write both figures."""

    repository_root = Path(__file__).resolve().parents[1]
    csv_path = repository_root / "Data" / "08" / "optimized_parameters.csv"
    output_dir = repository_root / "Figs"

    data, source_rows = load_and_validate(csv_path)
    summaries = {d: summarize(data, d) for d in BLOCK_SIZES}
    pdf_path, png_path, eta_limits = plot(summaries, output_dir)
    budget_counts = {d: len(summaries[d][0]) for d in BLOCK_SIZES}

    print(f"Loaded {source_rows} source rows; selected {len(data)} rows with n=10.")
    print(f"Budgets per block size: {budget_counts}.")
    print(
        f"Validated {EXPECTED_STATES_PER_GROUP} unique states in every "
        "(d, budget) group."
    )
    print("Plotted physical theta and arithmetic mean +/- sample std (ddof=1).")
    print(
        "Row y-ranges: h=(0.5, 1.0), theta=(0.0, 0.5), "
        f"eta_test=({eta_limits[0]:.6g}, {eta_limits[1]:.6g})."
    )
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")


if __name__ == "__main__":
    try:
        main()
    except (DataValidationError, OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
