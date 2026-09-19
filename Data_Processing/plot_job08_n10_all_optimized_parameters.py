#!/usr/bin/env python3
"""Plot all optimized Job 08 parameters for the n=10 experiments.

Threshold/test parameters and normalized stage weights are reduced to
arithmetic means with one sample-standard-deviation error bars. Per-state
integer stage-copy allocations use geometric means with multiplicative
geometric-standard-deviation intervals. All sample deviations use ddof=1
across 20 independently optimized states at each (d, budget).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Keep Matplotlib runnable when the user's normal cache directory is read-only.
_MPL_CONFIG = tempfile.TemporaryDirectory(prefix="job08-n10-all-parameters-mpl-")
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
OUTPUT_STEM = "job08_n10_all_optimized_parameters_vs_copies"

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
STAGE_MARKERS = {
    "peel_weight": "o",
    "recovery_weight": "s",
    "grouping_weight": "^",
    "syndrome_weight": "D",
    "tomography_weight": "v",
}
STAGE_COPY_COLUMNS = {
    "peel_weight": "peel_copies",
    "recovery_weight": "recovery_copies",
    "grouping_weight": "grouping_copies",
    "syndrome_weight": "syndrome_copies",
    "tomography_weight": "tomography_copies",
}


class DataValidationError(RuntimeError):
    """Raised when the input cannot support the requested statistics."""


def violation_preview(
    data: pd.DataFrame, mask: pd.Series, value_columns: list[str]
) -> str:
    """Format a compact preview of rows that violate a numeric constraint."""

    columns = ["d", "state_index", "budget", *value_columns]
    violating = data.loc[mask, columns]
    preview = violating.head(10).to_string(index=False)
    remainder = len(violating) - min(len(violating), 10)
    suffix = f"\n... and {remainder} more row(s)" if remainder else ""
    return preview + suffix


def load_validate_and_allocate(csv_path: Path) -> tuple[pd.DataFrame, int]:
    """Load, validate, normalize weights, and derive integer stage copies."""

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

    required = {"n", "d", "state_index", "budget", *PARAMETERS, *STAGES}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise DataValidationError(f"source CSV is missing columns: {missing}")

    n_values = pd.to_numeric(data["n"], errors="coerce")
    selected = data.loc[n_values.eq(N_QUBITS)].copy()
    if selected.empty:
        raise DataValidationError(f"source CSV contains no n={N_QUBITS} rows")

    numeric_columns = ["n", "d", "state_index", "budget", *PARAMETERS, *STAGES]
    selected[numeric_columns] = selected[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    nonfinite = ~np.isfinite(selected[numeric_columns].to_numpy(dtype=float))
    if nonfinite.any():
        positions = np.flatnonzero(nonfinite.any(axis=1))[:10]
        raise DataValidationError(
            "selected rows contain non-numeric or non-finite values at row "
            f"positions {positions.tolist()}"
        )

    observed_d = set(selected["d"].unique())
    if observed_d != set(BLOCK_SIZES):
        raise DataValidationError(
            f"n={N_QUBITS} has block sizes {sorted(observed_d)}; "
            f"expected {list(BLOCK_SIZES)}"
        )
    if not np.all(
        (selected["budget"] > 0)
        & selected["budget"].eq(np.floor(selected["budget"]))
    ):
        raise DataValidationError("all plotted budgets must be positive integers")

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
            + violation_preview(selected, h_violation, ["h_min", "h_max"])
        )

    theta_violation = ~(
        (selected["theta"] > 0.0) & (selected["theta"] <= 0.5)
    )
    if theta_violation.any():
        raise DataValidationError(
            "rows violate 0 < theta <= 0.5:\n"
            + violation_preview(selected, theta_violation, ["theta"])
        )

    eta_violation = selected["eta_test"] <= 0.0
    if eta_violation.any():
        raise DataValidationError(
            "rows violate eta_test > 0:\n"
            + violation_preview(selected, eta_violation, ["eta_test"])
        )

    processed_parts: list[pd.DataFrame] = []
    for d in BLOCK_SIZES:
        subset = selected.loc[selected["d"].eq(d)].copy()
        active = list(ACTIVE_STAGES[d])
        nonpositive = (subset[active] <= 0.0).any(axis=1)
        if nonpositive.any():
            raise DataValidationError(
                f"d={d} rows contain non-positive active stage weights:\n"
                + violation_preview(subset, nonpositive, active)
            )

        denominators = subset[active].sum(axis=1)
        if (denominators <= 0.0).any():
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

        budgets = subset["budget"].to_numpy(dtype=np.int64)
        allocated_before_tomography = np.zeros(len(subset), dtype=np.int64)
        for stage in active[:-1]:
            copy_column = STAGE_COPY_COLUMNS[stage]
            fractions = subset[f"{stage}_norm"].to_numpy(dtype=float)
            stage_copies = np.floor(budgets * fractions).astype(np.int64)
            subset[copy_column] = stage_copies
            allocated_before_tomography += stage_copies

        tomography_column = STAGE_COPY_COLUMNS[active[-1]]
        subset[tomography_column] = budgets - allocated_before_tomography
        active_copy_columns = [STAGE_COPY_COLUMNS[stage] for stage in active]
        copy_sums = subset[active_copy_columns].sum(axis=1).to_numpy(dtype=np.int64)
        if not np.array_equal(copy_sums, budgets):
            raise DataValidationError(f"d={d} integer stage copies do not sum to budget")
        if (subset[active_copy_columns] <= 0).any(axis=None):
            raise DataValidationError(f"d={d} has a non-positive active stage allocation")
        processed_parts.append(subset)

    return pd.concat(processed_parts, ignore_index=True), source_rows


def summarize_columns(
    data: pd.DataFrame, d: int, columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return sorted means and sample standard deviations for selected columns."""

    grouped = data.loc[data["d"].eq(d)].groupby("budget", sort=True)[columns]
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


def summarize_geometric_columns(
    data: pd.DataFrame, d: int, columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return geometric means and sample geometric standard deviations."""

    selected = data.loc[data["d"].eq(d), ["budget", *columns]].copy()
    if (selected[columns] <= 0).any(axis=None):
        raise DataValidationError(
            f"d={d} copy allocations must be positive for geometric statistics"
        )
    logged = np.log(selected[columns])
    logged["budget"] = selected["budget"].to_numpy()
    grouped = logged.groupby("budget", sort=True)[columns]
    geometric_means = np.exp(grouped.mean()).sort_index()
    geometric_standard_deviations = np.exp(grouped.std(ddof=1)).sort_index()
    if not (
        np.isfinite(geometric_means.to_numpy(dtype=float)).all()
        and np.isfinite(
            geometric_standard_deviations.to_numpy(dtype=float)
        ).all()
    ):
        raise DataValidationError(
            f"d={d} geometric copy summary contains non-finite statistics"
        )
    if (geometric_standard_deviations < 1.0).any(axis=None):
        raise DataValidationError(f"d={d} geometric standard deviation is below one")
    return geometric_means, geometric_standard_deviations


def eta_axis_limits(
    parameter_summaries: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
) -> tuple[float, float]:
    """Derive a shared eta range from the full mean ± sample-SD envelope."""

    lower = min(
        float((means["eta_test"] - deviations["eta_test"]).min())
        for means, deviations in parameter_summaries.values()
    )
    upper = max(
        float((means["eta_test"] + deviations["eta_test"]).max())
        for means, deviations in parameter_summaries.values()
    )
    span = upper - lower
    margin = 0.05 * span if span > 0 else max(0.01, 0.05 * abs(upper))
    return max(0.0, lower - margin), upper + margin


def copy_axis_limits(
    copy_summaries: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
) -> tuple[float, float]:
    """Cover every GM/GSD to GM*GSD interval on one shared log axis."""

    lower = min(
        float((means / deviations).to_numpy(dtype=float).min())
        for means, deviations in copy_summaries.values()
    )
    upper = max(
        float((means * deviations).to_numpy(dtype=float).max())
        for means, deviations in copy_summaries.values()
    )
    log_span = np.log10(upper) - np.log10(lower)
    margin_factor = 10.0 ** (0.025 * log_span)
    return lower / margin_factor, upper * margin_factor


def draw_errorbar(
    ax: plt.Axes,
    budgets: np.ndarray,
    means: pd.Series,
    deviations: pd.Series,
    *,
    label: str | None,
    color: tuple[float, float, float, float] | tuple[float, float, float],
    marker: str,
    stage_series: bool = False,
) -> None:
    """Draw one series using the established local error-bar style."""

    ax.errorbar(
        budgets,
        means.to_numpy(dtype=float),
        yerr=deviations.to_numpy(dtype=float),
        label=label,
        color=color,
        marker=marker,
        linestyle="-",
        linewidth=1.45 if stage_series else 1.4,
        markersize=4.5 if stage_series else 4.2,
        capsize=2.5 if stage_series else 2.3,
        capthick=0.8 if stage_series else 0.75,
        elinewidth=0.8 if stage_series else 0.75,
    )


def draw_geometric_errorbar(
    ax: plt.Axes,
    budgets: np.ndarray,
    geometric_means: pd.Series,
    geometric_standard_deviations: pd.Series,
    *,
    label: str,
    color: tuple[float, float, float, float] | tuple[float, float, float],
    marker: str,
) -> None:
    """Draw a geometric mean with a GM/GSD to GM*GSD interval."""

    means = geometric_means.to_numpy(dtype=float)
    deviations = geometric_standard_deviations.to_numpy(dtype=float)
    lower_errors = means - means / deviations
    upper_errors = means * deviations - means
    ax.errorbar(
        budgets,
        means,
        yerr=np.vstack((lower_errors, upper_errors)),
        label=label,
        color=color,
        marker=marker,
        linestyle="-",
        linewidth=1.45,
        markersize=4.5,
        capsize=2.5,
        capthick=0.8,
        elinewidth=0.8,
    )


def plot(
    parameter_summaries: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
    weight_summaries: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
    copy_summaries: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
    output_dir: Path,
) -> tuple[Path, Path, tuple[float, float], tuple[float, float]]:
    """Create and atomically save the publication-quality 5x3 figure."""

    plt.rcParams.update(
        {
            "font.size": 9.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 8.7,
        }
    )
    tab10 = plt.get_cmap("tab10").colors
    parameter_styles = {
        "h_min": (tab10[0], "o"),
        "h_max": (tab10[1], "s"),
        "theta": (tab10[2], "^"),
        "eta_test": (tab10[3], "D"),
    }
    stage_colors = {stage: tab10[index] for index, stage in enumerate(STAGES)}
    eta_limits = eta_axis_limits(parameter_summaries)
    copy_limits = copy_axis_limits(copy_summaries)

    fig, axes = plt.subplots(
        5,
        3,
        figsize=(13.2, 15.8),
        sharex=True,
        sharey="row",
        constrained_layout=False,
    )
    panel_labels = tuple(f"({chr(ord('a') + index)})" for index in range(15))

    for column, d in enumerate(BLOCK_SIZES):
        parameter_means, parameter_deviations = parameter_summaries[d]
        weight_means, weight_deviations = weight_summaries[d]
        copy_means, copy_geometric_deviations = copy_summaries[d]
        budgets = parameter_means.index.to_numpy(dtype=float)
        if not (
            parameter_means.index.equals(weight_means.index)
            and parameter_means.index.equals(copy_means.index)
        ):
            raise DataValidationError(
                f"d={d} parameter/weight/copy budget grids do not match"
            )

        for parameter, label in (("h_min", r"$h_{\min}$"), ("h_max", r"$h_{\max}$")):
            draw_errorbar(
                axes[0, column],
                budgets,
                parameter_means[parameter],
                parameter_deviations[parameter],
                label=label,
                color=parameter_styles[parameter][0],
                marker=parameter_styles[parameter][1],
            )
        for row, parameter in ((1, "theta"), (2, "eta_test")):
            draw_errorbar(
                axes[row, column],
                budgets,
                parameter_means[parameter],
                parameter_deviations[parameter],
                label=None,
                color=parameter_styles[parameter][0],
                marker=parameter_styles[parameter][1],
            )
        for stage in ACTIVE_STAGES[d]:
            normalized = f"{stage}_norm"
            draw_errorbar(
                axes[3, column],
                budgets,
                weight_means[normalized],
                weight_deviations[normalized],
                label=STAGES[stage],
                color=stage_colors[stage],
                marker=STAGE_MARKERS[stage],
                stage_series=True,
            )
            copy_column = STAGE_COPY_COLUMNS[stage]
            draw_geometric_errorbar(
                axes[4, column],
                budgets,
                copy_means[copy_column],
                copy_geometric_deviations[copy_column],
                label=STAGES[stage],
                color=stage_colors[stage],
                marker=STAGE_MARKERS[stage],
            )

        axes[0, column].set_title(rf"$d={d}$")
        for row in range(5):
            ax = axes[row, column]
            ax.set_xscale("log")
            ax.set_xlim(left=1_000)
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
            if row < 4:
                ax.tick_params(axis="x", which="both", labelbottom=False)
            else:
                ax.set_xlabel(r"Total copies, $N$")

    for ax in axes[0, :]:
        ax.set_ylim(0.5, 1.0)
    for ax in axes[1, :]:
        ax.set_ylim(0.0, 0.5)
    for ax in axes[2, :]:
        ax.set_ylim(*eta_limits)
    for ax in axes[3, :]:
        ax.set_ylim(0.0, 1.0)
    for ax in axes[4, :]:
        ax.set_yscale("log")
        ax.set_ylim(*copy_limits)

    axes[0, 0].set_ylabel(r"$h_{\min},\,h_{\max}$")
    axes[1, 0].set_ylabel(r"$\theta$")
    axes[2, 0].set_ylabel(r"$\eta_{\mathrm{test}}$")
    axes[3, 0].set_ylabel("Normalized stage weight")
    axes[4, 0].set_ylabel("Allocated stage copies")

    h_handles, h_labels = axes[0, 0].get_legend_handles_labels()
    h_legend = fig.legend(
        h_handles,
        h_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.996),
        ncol=2,
        frameon=True,
        columnspacing=1.3,
        handlelength=2.1,
        handletextpad=0.5,
        borderpad=0.4,
    )
    fig.add_artist(h_legend)

    handles_by_label = {}
    for ax in axes[3, :]:
        handles, labels = ax.get_legend_handles_labels()
        handles_by_label.update(dict(zip(labels, handles)))
    stage_labels = list(STAGES.values())
    fig.legend(
        [handles_by_label[label] for label in stage_labels],
        stage_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=5,
        frameon=True,
        columnspacing=1.1,
        handlelength=2.0,
        handletextpad=0.45,
        borderpad=0.4,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.087,
        top=0.958,
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

    return pdf_path, png_path, eta_limits, copy_limits


def main() -> None:
    """Validate the source, calculate statistics, and write both figures."""

    repository_root = Path(__file__).resolve().parents[1]
    csv_path = repository_root / "Data" / "08" / "optimized_parameters.csv"
    output_dir = repository_root / "Figs"

    data, source_rows = load_validate_and_allocate(csv_path)
    parameter_summaries = {
        d: summarize_columns(data, d, list(PARAMETERS)) for d in BLOCK_SIZES
    }
    weight_summaries = {
        d: summarize_columns(
            data, d, [f"{stage}_norm" for stage in ACTIVE_STAGES[d]]
        )
        for d in BLOCK_SIZES
    }
    copy_summaries = {
        d: summarize_geometric_columns(
            data, d, [STAGE_COPY_COLUMNS[stage] for stage in ACTIVE_STAGES[d]]
        )
        for d in BLOCK_SIZES
    }
    pdf_path, png_path, eta_limits, copy_limits = plot(
        parameter_summaries, weight_summaries, copy_summaries, output_dir
    )
    budget_counts = {d: len(parameter_summaries[d][0]) for d in BLOCK_SIZES}

    print(f"Loaded {source_rows} source rows; selected {len(data)} rows with n=10.")
    print(f"Budgets per block size: {budget_counts}.")
    print(
        f"Validated {EXPECTED_STATES_PER_GROUP} unique states in every "
        "(d, budget) group."
    )
    print("Plotted physical theta and arithmetic mean +/- sample std (ddof=1).")
    print("Normalized active stage weights state-by-state before aggregation.")
    print(
        "Constructed exact integer stage-copy allocations state-by-state, "
        "with the rounding remainder assigned to tomography."
    )
    print(
        "Stage-copy panels use geometric means with multiplicative sample "
        "geometric standard deviations (ddof=1 in log space)."
    )
    print("Excluded grouping from d=1 normalization, allocation, and plotting.")
    print(
        "Row y-ranges: h=(0.5, 1.0), theta=(0.0, 0.5), "
        f"eta_test=({eta_limits[0]:.6g}, {eta_limits[1]:.6g}), "
        "normalized weights=(0.0, 1.0), "
        f"allocated copies=({copy_limits[0]:.6g}, {copy_limits[1]:.6g}) "
        "on a log scale."
    )
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")


if __name__ == "__main__":
    try:
        main()
    except (DataValidationError, OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
