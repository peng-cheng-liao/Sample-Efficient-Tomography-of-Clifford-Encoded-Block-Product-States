#!/usr/bin/env python3
"""Plot normalized Job 08 stage weights for the n=10 experiments.

For each optimized state, the active raw stage weights are normalized to sum
to one.  The plotted points are the arithmetic mean and sample standard
deviation across the 20 independently optimized states at each (d, budget).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Keep Matplotlib runnable when the user's normal cache directory is read-only.
_MPL_CONFIG = tempfile.TemporaryDirectory(prefix="job08-n10-stage-weights-mpl-")
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
OUTPUT_STEM = "job08_n10_stage_weights_vs_copies"

STAGES = {
    "peel_weight": "Peeling",
    "recovery_weight": "Recovery",
    "grouping_weight": "Grouping",
    "syndrome_weight": "Syndrome",
    "tomography_weight": "Tomography",
}
ACTIVE_STAGES = {
    1: tuple(stage for stage in STAGES if stage != "grouping_weight"),
    2: tuple(STAGES),
    3: tuple(STAGES),
}
MARKERS = {
    "peel_weight": "o",
    "recovery_weight": "s",
    "grouping_weight": "^",
    "syndrome_weight": "D",
    "tomography_weight": "v",
}


class DataValidationError(RuntimeError):
    """Raised when the source cannot support the requested statistics."""


def load_and_normalize(csv_path: Path) -> pd.DataFrame:
    """Load n=10 rows, validate completeness, and normalize each state."""

    if not csv_path.is_file():
        raise DataValidationError(f"source CSV is missing: {csv_path}")

    data = pd.read_csv(csv_path)
    required = {"n", "d", "state_index", "budget", *STAGES}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise DataValidationError(f"source CSV is missing columns: {missing}")

    data = data.loc[data["n"].eq(N_QUBITS)].copy()
    if data.empty:
        raise DataValidationError(f"source CSV contains no n={N_QUBITS} rows")

    observed_d = set(data["d"].unique())
    if observed_d != set(BLOCK_SIZES):
        raise DataValidationError(
            f"n={N_QUBITS} has block sizes {sorted(observed_d)}; "
            f"expected {list(BLOCK_SIZES)}"
        )

    numeric_columns = ["d", "state_index", "budget", *STAGES]
    numeric = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise DataValidationError("n=10 rows contain non-numeric or non-finite values")
    data[numeric_columns] = numeric

    if (data["budget"] <= 0).any():
        raise DataValidationError("all plotted budgets must be positive")

    group_columns = ["d", "budget"]
    budgets_by_d = {
        d: set(data.loc[data["d"].eq(d), "budget"].unique()) for d in BLOCK_SIZES
    }
    reference_budgets = budgets_by_d[BLOCK_SIZES[0]]
    if any(budgets != reference_budgets for budgets in budgets_by_d.values()):
        details = ", ".join(
            f"d={d}: {sorted(budgets)}" for d, budgets in budgets_by_d.items()
        )
        raise DataValidationError(
            "block sizes do not share the same budget grid; " + details
        )

    row_counts = data.groupby(group_columns, sort=True).size()
    bad_counts = row_counts[row_counts.ne(EXPECTED_STATES_PER_GROUP)]
    if not bad_counts.empty:
        details = ", ".join(
            f"(d={int(d)}, budget={budget:g}): {int(count)} rows"
            for (d, budget), count in bad_counts.items()
        )
        raise DataValidationError(
            f"expected {EXPECTED_STATES_PER_GROUP} rows per (d, budget); {details}"
        )

    unique_states = data.groupby(group_columns, sort=True)["state_index"].nunique()
    bad_states = unique_states[unique_states.ne(EXPECTED_STATES_PER_GROUP)]
    if not bad_states.empty:
        details = ", ".join(
            f"(d={int(d)}, budget={budget:g}): {int(count)} unique states"
            for (d, budget), count in bad_states.items()
        )
        raise DataValidationError(
            "duplicate or missing state indices within a complete group; " + details
        )

    normalized_parts: list[pd.DataFrame] = []
    for d in BLOCK_SIZES:
        subset = data.loc[data["d"].eq(d)].copy()
        active = list(ACTIVE_STAGES[d])
        denominators = subset[active].sum(axis=1)
        if (denominators <= 0).any():
            raise DataValidationError(f"d={d} has a non-positive active-weight sum")

        normalized_columns = [f"{stage}_norm" for stage in active]
        subset[normalized_columns] = subset[active].div(denominators, axis=0)
        normalized_sums = subset[normalized_columns].sum(axis=1).to_numpy()
        if not np.allclose(normalized_sums, 1.0, rtol=1e-12, atol=1e-12):
            maximum_error = float(np.max(np.abs(normalized_sums - 1.0)))
            raise DataValidationError(
                f"d={d} normalized weights do not sum to one "
                f"(maximum error {maximum_error:.3e})"
            )
        normalized_parts.append(subset)

    return pd.concat(normalized_parts, ignore_index=True)


def summarize(data: pd.DataFrame, d: int) -> pd.DataFrame:
    """Return per-budget means and sample standard deviations for one d."""

    normalized_columns = [f"{stage}_norm" for stage in ACTIVE_STAGES[d]]
    summary = (
        data.loc[data["d"].eq(d)]
        .groupby("budget", sort=True)[normalized_columns]
        .agg(["mean", lambda values: values.std(ddof=1)])
    )
    summary.columns = pd.MultiIndex.from_tuples(
        [
            (column, "mean" if statistic == "mean" else "std")
            for column, statistic in summary.columns
        ]
    )
    if not np.isfinite(summary.to_numpy(dtype=float)).all():
        raise DataValidationError(f"d={d} summary contains non-finite statistics")
    return summary.sort_index()


def plot(data: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Create and atomically save the three-panel publication figure."""

    plt.rcParams.update(
        {
            "font.size": 9.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
        }
    )
    tab10 = plt.get_cmap("tab10").colors
    colors = {stage: tab10[index] for index, stage in enumerate(STAGES)}

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.2, 4.0),
        sharey=True,
        constrained_layout=False,
    )
    panel_labels = ("(a)", "(b)", "(c)")

    for ax, d, panel_label in zip(axes, BLOCK_SIZES, panel_labels):
        statistics = summarize(data, d)
        budgets = statistics.index.to_numpy(dtype=float)
        for stage in ACTIVE_STAGES[d]:
            normalized = f"{stage}_norm"
            ax.errorbar(
                budgets,
                statistics[(normalized, "mean")].to_numpy(),
                yerr=statistics[(normalized, "std")].to_numpy(),
                label=STAGES[stage],
                color=colors[stage],
                marker=MARKERS[stage],
                linestyle="-",
                linewidth=1.45,
                markersize=4.5,
                capsize=2.5,
                capthick=0.8,
                elinewidth=0.8,
            )

        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(rf"$d={d}$")
        ax.set_xlabel(r"Total copies, $N$")
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
            0.95,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.0,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
                "pad": 0.6,
            },
        )

    axes[0].set_ylabel("Normalized stage weight")

    handles_by_label = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        handles_by_label.update(dict(zip(labels, handles)))
    legend_labels = list(STAGES.values())
    fig.legend(
        [handles_by_label[label] for label in legend_labels],
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        frameon=True,
        columnspacing=1.2,
        handlelength=2.1,
        handletextpad=0.5,
        borderpad=0.4,
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.16, top=0.84, wspace=0.10)

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

    return pdf_path, png_path


def main() -> None:
    """Validate the source data, compute statistics, and write both figures."""

    repository_root = Path(__file__).resolve().parents[1]
    csv_path = repository_root / "Data" / "08" / "optimized_parameters.csv"
    output_dir = repository_root / "Figs"

    data = load_and_normalize(csv_path)
    pdf_path, png_path = plot(data, output_dir)
    groups = data.groupby(["d", "budget"]).ngroups
    budgets = data["budget"].nunique()
    print(
        f"Validated {len(data)} n={N_QUBITS} rows across {groups} groups "
        f"({budgets} budgets for each d); each group has "
        f"{EXPECTED_STATES_PER_GROUP} unique states."
    )
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")


if __name__ == "__main__":
    try:
        main()
    except (DataValidationError, OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
