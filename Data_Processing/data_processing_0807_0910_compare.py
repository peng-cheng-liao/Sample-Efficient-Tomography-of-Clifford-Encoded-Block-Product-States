#!/usr/bin/env python3
"""Combine the n=10 Job 08/07 and Job 09/10 tomography comparisons.

The left panel shows the final trace-distance reconstruction error versus the
total-copy budget for the n=10 Job 08 (CEBP) and Job 07 (Local-Pauli) data.
Both methods are reduced directly over their completed runs using a geometric
mean and one multiplicative standard deviation in log space.

The right panel shows the fixed-error minimum-copy comparison from Jobs 09 and
10.  Both methods use a geometric mean and a one-standard-deviation log-space
interval over the included completed records.

The figure is written as PDF, together with a derived power-law fit summary.
"""

from __future__ import annotations

import os
import tempfile
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The user's home Matplotlib cache may be read-only in sandboxed/local runs.
_MPL_CONFIG = tempfile.TemporaryDirectory(prefix="data-processing-0807-0910-mpl-")
os.environ.setdefault("MPLCONFIGDIR", _MPL_CONFIG.name)

import matplotlib

matplotlib.set_loglevel("error")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullLocator, ScalarFormatter

import data_processing_0807_compare as comparison_0807
import data_processing_0910_compare as comparison_0910


N_QUBITS = 10
OUTPUT_STEM = "data_processing_0807_0910_combined"
RIGHT_LOG_X_STEM = f"{OUTPUT_STEM}_right_logx"
FIT_SUMMARY_NAME = "data_processing_0807_0910_fit_summary.csv"

# Use LaTeX for all figure text, as requested.
rc("text", usetex=True)
rc("font", family="serif")
rc("font", size=8.0)
rc("axes", labelsize=8.5)
rc("xtick", labelsize=7.7)
rc("ytick", labelsize=7.7)


class PowerLawFitError(RuntimeError):
    """Raised when a requested power-law fit is not statistically defined."""


@dataclass(frozen=True)
class PowerLawFit:
    """Ordinary-least-squares fit of y = A x^p in log10 space."""

    prefactor_A: float
    exponent_p: float
    r_squared_log10: float
    exponent_stderr: float
    n_points: int
    x_min: float
    x_max: float


@dataclass(frozen=True)
class FitSummary:
    """Metadata and result for one plotted power-law fit."""

    panel: str
    method: str
    d: int
    model: str
    x_variable: str
    y_variable: str
    fit_cutoff: float | None
    fit: PowerLawFit


def fit_power_law(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_x: float | None = None,
) -> PowerLawFit:
    """Fit y = A x^p by unweighted OLS in base-10 log space."""

    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.ndim != 1 or y_values.ndim != 1:
        raise PowerLawFitError("power-law fit inputs must be one-dimensional")
    if x_values.shape != y_values.shape:
        raise PowerLawFitError("power-law fit x and y arrays must have equal length")
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
        raise PowerLawFitError("power-law fit requires finite x and y values")
    if not np.all(x_values > 0.0) or not np.all(y_values > 0.0):
        raise PowerLawFitError("power-law fit requires strictly positive x and y values")

    if min_x is not None:
        if not np.isfinite(min_x) or min_x <= 0.0:
            raise PowerLawFitError("power-law fit cutoff must be positive and finite")
        included = x_values >= float(min_x)
        x_values = x_values[included]
        y_values = y_values[included]

    n_points = int(x_values.size)
    if n_points < 3:
        raise PowerLawFitError(
            f"power-law fit requires at least 3 points; found {n_points}"
        )

    log_x = np.log10(x_values)
    log_y = np.log10(y_values)
    centered_x = log_x - np.mean(log_x)
    centered_y = log_y - np.mean(log_y)
    ss_x = float(np.sum(centered_x**2))
    if not np.isfinite(ss_x) or ss_x <= 0.0:
        raise PowerLawFitError("power-law fit requires at least two distinct x values")

    exponent = float(np.sum(centered_x * centered_y) / ss_x)
    intercept = float(np.mean(log_y) - exponent * np.mean(log_x))
    fitted_log_y = intercept + exponent * log_x
    residuals = log_y - fitted_log_y
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum(centered_y**2))
    if not np.isfinite(ss_tot) or ss_tot <= 0.0:
        raise PowerLawFitError("log10(y) has zero variance, so R^2 is undefined")

    r_squared = float(1.0 - ss_res / ss_tot)
    residual_variance = ss_res / (n_points - 2)
    exponent_stderr = float(np.sqrt(residual_variance / ss_x))
    prefactor = float(10.0**intercept)
    outputs = (prefactor, exponent, r_squared, exponent_stderr)
    if not np.all(np.isfinite(outputs)) or prefactor <= 0.0:
        raise PowerLawFitError("power-law fit produced a non-finite result")

    return PowerLawFit(
        prefactor_A=prefactor,
        exponent_p=exponent,
        r_squared_log10=r_squared,
        exponent_stderr=exponent_stderr,
        n_points=n_points,
        x_min=float(np.min(x_values)),
        x_max=float(np.max(x_values)),
    )


def load_job08_completed_runs(
    csv_path: Path,
    expected_d: int,
    label: str,
    expected_n: int = N_QUBITS,
) -> comparison_0807.LoadedDataset:
    """Load each validated Job 08 holdout row as one completed run."""

    dataset = comparison_0807.LoadedDataset(label=label)
    if not csv_path.is_file():
        raise comparison_0807.DataValidationError(
            f"{label}: CSV file is missing: {csv_path}"
        )

    required_fields = {
        "n",
        "d",
        "init_id",
        "copy_budget",
        "task_id",
        "holdout_index",
        "holdout_seed",
        "trace_distance_error",
        "realized_total_copies",
        "source_result",
    }
    seen: set[tuple[int, int, int]] = set()
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required_fields.difference(reader.fieldnames or [])
            if missing:
                raise comparison_0807.DataValidationError(
                    f"{label}: CSV is missing required columns: {sorted(missing)}"
                )
            for row in reader:
                dataset.discovered_records += 1
                try:
                    n = comparison_0807.integer(row["n"], "n", positive=True)
                    d = comparison_0807.integer(row["d"], "d", positive=True)
                    if n != expected_n or d != expected_d:
                        raise ValueError("unexpected n or d")
                    init_id = comparison_0807.integer(row["init_id"], "init_id")
                    budget = comparison_0807.integer(
                        row["copy_budget"], "copy_budget", positive=True
                    )
                    comparison_0807.integer(row["task_id"], "task_id")
                    holdout_index = comparison_0807.integer(
                        row["holdout_index"], "holdout_index"
                    )
                    comparison_0807.integer(row["holdout_seed"], "holdout_seed")
                    realized = comparison_0807.integer(
                        row["realized_total_copies"],
                        "realized_total_copies",
                        positive=True,
                    )
                    if realized > budget:
                        raise ValueError("realized copies exceed requested budget")
                    trace_error = comparison_0807.finite_trace_distance(
                        row["trace_distance_error"], "trace_distance_error"
                    )
                    if not row["source_result"].strip().endswith("/result.json"):
                        raise ValueError("invalid source_result")
                    identity = (init_id, budget, holdout_index)
                    if identity in seen:
                        raise ValueError("duplicate holdout identity")
                except (KeyError, ValueError) as error:
                    comparison_0807.note_bad_record(dataset, csv_path, str(error))
                    continue

                seen.add(identity)
                dataset.runs_by_budget[budget].append(trace_error)
                dataset.valid_records += 1
    except (OSError, csv.Error) as error:
        raise comparison_0807.DataValidationError(
            f"{label}: could not read {csv_path}: {error}"
        ) from error

    if dataset.discovered_records == 0:
        raise comparison_0807.DataValidationError(
            f"{label}: CSV contains no data rows: {csv_path}"
        )
    return dataset


def load_0807_series(
    repository_root: Path,
) -> list[
    tuple[
        comparison_0807.LoadedDataset,
        list[comparison_0807.Aggregate],
    ]
]:
    """Load the n=10 Job 08/07 data using the geometric-mean reduction."""

    series: list[
        tuple[
            comparison_0807.LoadedDataset,
            list[comparison_0807.Aggregate],
        ]
    ] = []
    for d in (3, 2, 1):
        csv_path = (
            repository_root
            / "Data"
            / "08"
            / f"n={N_QUBITS}_d={d}_holdout_errors.csv"
        )
        dataset = load_job08_completed_runs(
            csv_path,
            d,
            f"08 (d={d})",
            expected_n=N_QUBITS,
        )
        points = comparison_0807.aggregate(dataset, statistic="geometric_mean")
        comparison_0807.summarize_dataset(
            dataset,
            points,
            "completed CSV rows; center=geometric mean over completed runs",
        )
        series.append((dataset, points))

    for d in (3, 2, 1):
        directory = repository_root / "Data" / "07" / f"n={N_QUBITS}_d={d}"
        archive = repository_root / "Data" / "07" / f"n={N_QUBITS}_d={d}.zip"
        data_source = directory if directory.is_dir() else archive
        dataset = comparison_0807.load_job07(
            data_source,
            d,
            f"07 (d={d})",
            expected_n=N_QUBITS,
        )
        points = comparison_0807.aggregate(dataset, statistic="geometric_mean")
        comparison_0807.summarize_dataset(
            dataset,
            points,
            "init_*/budget_*/replicate_*.json; "
            "center=geometric mean over completed runs",
        )
        series.append((dataset, points))

    return series


def plot_0807_panel(
    ax: plt.Axes,
    series: list[
        tuple[
            comparison_0807.LoadedDataset,
            list[comparison_0807.Aggregate],
        ]
    ],
) -> list[FitSummary]:
    """Draw the n=10 reconstruction-error panel."""

    colors = {3: "#1f77b4", 2: "#9467bd", 1: "#2ca02c"}
    cebp_markers = {3: "o", 2: "^", 1: "D"}
    lp_li_markers = {3: "P", 2: "v", 1: "s"}
    fit_cutoffs = {1: 1e4, 2: 1e5, 3: 1e6}
    fit_summaries: list[FitSummary] = []

    for dataset, points in series:
        job = int(dataset.label[:2])
        d = int(dataset.label.split("d=")[1].rstrip(")"))
        x = np.asarray([point.n_total for point in points], dtype=float)
        y = np.asarray([point.estimate for point in points], dtype=float)
        log_sd = np.asarray([point.log_sd for point in points], dtype=float)
        if not np.all(np.isfinite(log_sd)):
            raise comparison_0807.DataValidationError(
                f"{dataset.label}: log-space SD requires positive errors"
            )
        factor = np.exp(log_sd)
        lower = y / factor
        upper = y * factor
        method = "CEBP" if job == 8 else "LP-LI"
        ax.errorbar(
            x,
            y,
            yerr=np.vstack((y - lower, upper - y)),
            label=rf"{method} ($d={d}$)",
            color=colors[d],
            marker=(cebp_markers if job == 8 else lp_li_markers)[d],
            linestyle="none" if job == 8 else "-",
            markersize=3.8,
            linewidth=1.15,
            elinewidth=0.65,
            capsize=1.8,
            capthick=0.65,
        )
        if job == 8:
            cutoff = fit_cutoffs[d]
            fit = fit_power_law(x, y, min_x=cutoff)
            fit_x = np.geomspace(fit.x_min, fit.x_max, 200)
            ax.plot(
                fit_x,
                fit.prefactor_A * fit_x**fit.exponent_p,
                color=colors[d],
                linestyle=":",
                linewidth=1.0,
                zorder=1.5,
            )
            fit_summaries.append(
                FitSummary(
                    panel="a",
                    method="CEBP",
                    d=d,
                    model="epsilon = A * N^p",
                    x_variable="N",
                    y_variable="epsilon",
                    fit_cutoff=cutoff,
                    fit=fit,
                )
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=1_000)
    ax.text(
        0.035,
        0.90,
        r"(a)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.8},
    )
    ax.set_xlabel(r"Total copies $N$")
    ax.set_ylabel(r"Trace distance")
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.42)
    ax.grid(True, which="minor", linestyle="--", linewidth=0.35, alpha=0.12)
    return sorted(fit_summaries, key=lambda summary: summary.d)


def aggregate_partial(
    dataset: comparison_0910.LoadedDataset,
) -> list[comparison_0910.Aggregate]:
    """Aggregate every available qubit count without requiring a complete scan."""

    # TODO: require complete scans for the final publication freeze.
    points: list[comparison_0910.Aggregate] = []
    for n in sorted(dataset.copies_by_n):
        values = np.asarray(dataset.copies_by_n[n], dtype=float)
        if values.size == 0:
            continue
        sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        log_values = np.log(values)
        log_mean = float(np.mean(log_values))
        log_sd = float(np.std(log_values, ddof=1)) if values.size > 1 else 0.0
        points.append(
            comparison_0910.Aggregate(
                n=n,
                mean=float(np.mean(values)),
                sem=sd / np.sqrt(int(values.size)) if values.size > 1 else 0.0,
                sd=sd,
                geometric_mean=float(np.exp(log_mean)),
                log_sd=log_sd,
                geometric_lower=float(np.exp(log_mean - log_sd)),
                geometric_upper=float(np.exp(log_mean + log_sd)),
                count=int(values.size),
            )
        )
    return points


def load_job09_partial(
    path: Path, d: int, label: str
) -> comparison_0910.LoadedDataset:
    """Load completed Job 09 rows while retaining partial n/d scans."""

    dataset = comparison_0910.LoadedDataset(label=label)
    required = {"n", "state_index", "task_id", "minimum_copies", "status"}
    if not path.is_file():
        raise comparison_0910.DataValidationError(f"{label}: CSV file is missing: {path}")
    seen: set[tuple[int, int]] = set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise comparison_0910.DataValidationError(
                    f"{label}: CSV is missing required columns: {sorted(missing)}"
                )
            for row in reader:
                dataset.discovered_records += 1
                if (row.get("status") or "").strip() != "completed":
                    dataset.skipped[f"status={(row.get('status') or '').strip() or 'missing'}"] += 1
                    continue
                try:
                    n = comparison_0910.integer(row["n"], "n", positive=True)
                    state_index = comparison_0910.integer(row["state_index"], "state_index")
                    task_id = comparison_0910.integer(row["task_id"], "task_id")
                    minimum = comparison_0910.integer(
                        row["minimum_copies"], "minimum_copies", positive=True
                    )
                    if n < d or not 0 <= state_index < comparison_0910.EXPECTED_STATES_PER_N:
                        raise ValueError("unexpected n or state_index")
                    if task_id != (n - d) * comparison_0910.EXPECTED_STATES_PER_N + state_index:
                        raise ValueError("task_id does not match n, d, and state_index")
                    identity = (n, state_index)
                    if identity in seen:
                        raise ValueError("duplicate state identity")
                except (KeyError, ValueError) as error:
                    dataset.skipped[str(error)] += 1
                    continue
                seen.add(identity)
                dataset.copies_by_n[n].append(minimum)
                dataset.valid_records += 1
    except (OSError, csv.Error) as error:
        raise comparison_0910.DataValidationError(f"{label}: could not read {path}: {error}") from error
    return dataset


def load_job10_partial(
    path: Path, d: int, sheet_name: str, label: str
) -> comparison_0910.LoadedDataset:
    """Load completed Job 10 rows from a d-specific sheet, allowing gaps."""

    dataset = comparison_0910.LoadedDataset(label=label)
    rows = comparison_0910.read_xlsx_table(path, sheet_name)
    required = {
        "n", "state_index", "task_id", "minimum_M_per_setting",
        "measurement_settings_3_pow_n", "minimum_total_copies_N", "scientific_status",
    }
    comparison_0910._require_columns(rows[0].keys() if rows else [], required, label)
    seen: set[tuple[int, int]] = set()
    for row in rows:
        dataset.discovered_records += 1
        status = str(row.get("scientific_status") or "").strip()
        if status != "validated_threshold_found":
            dataset.skipped[f"status={status or 'missing'}"] += 1
            continue
        try:
            n = comparison_0910.integer(row["n"], "n", positive=True)
            state_index = comparison_0910.integer(row["state_index"], "state_index")
            task_id = comparison_0910.integer(row["task_id"], "task_id")
            minimum_m = comparison_0910.integer(row["minimum_M_per_setting"], "minimum_M_per_setting", positive=True)
            settings = comparison_0910.integer(row["measurement_settings_3_pow_n"], "measurement_settings_3_pow_n", positive=True)
            total = comparison_0910.integer(row["minimum_total_copies_N"], "minimum_total_copies_N", positive=True)
            if n < d or not 0 <= state_index < comparison_0910.EXPECTED_STATES_PER_N:
                raise ValueError("unexpected n or state_index")
            if task_id != (n - d) * comparison_0910.EXPECTED_STATES_PER_N + state_index:
                raise ValueError("task_id does not match n, d, and state_index")
            if settings != 3**n or total != minimum_m * settings:
                raise ValueError("total-copy accounting check failed")
            identity = (n, state_index)
            if identity in seen:
                raise ValueError("duplicate state identity")
        except (KeyError, ValueError) as error:
            dataset.skipped[str(error)] += 1
            continue
        seen.add(identity)
        dataset.copies_by_n[n].append(total)
        dataset.valid_records += 1
    return dataset


def summarize_partial(
    dataset: comparison_0910.LoadedDataset,
    points: list[comparison_0910.Aggregate],
) -> None:
    """Print concise completeness information for a partial d-specific series."""

    counts = ", ".join(f"n={point.n}:{point.count}" for point in points)
    skipped = sum(dataset.skipped.values())
    print(f"{dataset.label}: {dataset.valid_records}/{dataset.discovered_records} records included; {counts}")
    print(f"{dataset.label}: skipped {skipped} incomplete/invalid records")


def load_0910_series(
    repository_root: Path,
) -> list[
    tuple[int, comparison_0910.LoadedDataset, list[comparison_0910.Aggregate]]
]:
    """Load all available d=1, d=2, and d=3 Job 09/10 series."""

    job09_paths = {
        1: repository_root / "Data" / "09" / "minimum_copies_by_state.csv",
        2: repository_root / "Data" / "09" / "minimum_copies_by_state_d2.csv",
        3: repository_root / "Data" / "09" / "minimum_copies_by_state_d3.csv",
    }
    job10_sheets = {1: "Minimum Copies", 2: "d2 Minimum Copies", 3: "d3 Minimum Copies"}
    series: list[tuple[int, comparison_0910.LoadedDataset, list[comparison_0910.Aggregate]]] = []
    for d in (1, 2, 3):
        job09 = load_job09_partial(job09_paths[d], d, f"Job 09 (d={d})")
        job09_points = aggregate_partial(job09)
        summarize_partial(job09, job09_points)
        series.append((d, job09, job09_points))
    for d in (1, 2, 3):
        job10 = load_job10_partial(
            repository_root / "Data" / "10" / "minimum_copies_by_state.xlsx",
            d,
            job10_sheets[d],
            f"Job 10 (d={d})",
        )
        job10_points = aggregate_partial(job10)
        summarize_partial(job10, job10_points)
        series.append((d, job10, job10_points))
    return series


def plot_0910_panel(
    ax: plt.Axes,
    series: list[
        tuple[
            int,
            comparison_0910.LoadedDataset,
            list[comparison_0910.Aggregate],
        ]
    ],
    *,
    right_log_x: bool = False,
) -> list[FitSummary]:
    """Draw the fixed-error minimum-copy panel."""

    colors = {3: "#1f77b4", 2: "#9467bd", 1: "#2ca02c"}
    cebp_markers = {3: "o", 2: "^", 1: "D"}
    lp_li_markers = {3: "P", 2: "v", 1: "s"}
    fit_summaries: list[FitSummary] = []

    for d, dataset, points in series:
        is_cebp = dataset.label.startswith("Job 09")
        x = np.asarray([point.n for point in points], dtype=float)
        y = np.asarray([point.geometric_mean for point in points], dtype=float)
        lower = np.asarray([point.geometric_lower for point in points], dtype=float)
        upper = np.asarray([point.geometric_upper for point in points], dtype=float)
        method = "CEBP" if is_cebp else "LP-LI"
        ax.errorbar(
            x,
            y,
            yerr=np.vstack((y - lower, upper - y)),
            label=rf"{method} ($d={d}$)",
            color=colors[d],
            marker=(cebp_markers if is_cebp else lp_li_markers)[d],
            linestyle="none" if is_cebp else "-",
            markersize=3.8,
            linewidth=1.15,
            elinewidth=0.65,
            capsize=1.8,
            capthick=0.65,
        )
        fit = fit_power_law(x, y)
        if is_cebp:
            fit_x = np.geomspace(fit.x_min, fit.x_max, 200)
            ax.plot(
                fit_x,
                fit.prefactor_A * fit_x**fit.exponent_p,
                color=colors[d],
                linestyle=":",
                linewidth=1.0,
                zorder=1.5,
            )
        fit_summaries.append(
            FitSummary(
                panel="b",
                method=method,
                d=d,
                model="N = A * n^p",
                x_variable="n",
                y_variable="N",
                fit_cutoff=None,
                fit=fit,
            )
        )

    ax.set_yscale("log")
    ax.set_xticks(comparison_0910.EXPECTED_N_VALUES)
    if right_log_x:
        ax.set_xscale("log")
        ax.set_xticks(comparison_0910.EXPECTED_N_VALUES)
        ax.xaxis.set_minor_locator(NullLocator())
        tick_formatter = ScalarFormatter()
        tick_formatter.set_scientific(False)
        tick_formatter.set_useOffset(False)
        ax.xaxis.set_major_formatter(tick_formatter)
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10, dtype=float) * 0.1)
    )
    ax.set_xlim(0.7, 9.3)
    ax.text(
        0.035,
        0.90,
        r"(b)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.8},
    )
    ax.set_xlabel(r"Number of qubits $n$")
    ax.set_ylabel(r"Total copies")
    ax.set_axisbelow(True)
    ax.grid(
        True,
        which="major",
        axis="y",
        linestyle="--",
        linewidth=0.5,
        alpha=0.42,
    )
    ax.grid(
        True,
        which="major",
        axis="x",
        linestyle="--",
        linewidth=0.5,
        alpha=0.42,
    )
    ax.grid(True, which="minor", axis="y", linestyle="--", linewidth=0.35, alpha=0.10)
    method_order = {"CEBP": 0, "LP-LI": 1}
    return sorted(
        fit_summaries,
        key=lambda summary: (method_order[summary.method], summary.d),
    )


def ordered_fit_summaries(summaries: list[FitSummary]) -> list[FitSummary]:
    """Return the nine fits in stable panel/method/d order."""

    method_order = {"CEBP": 0, "LP-LI": 1}
    ordered = sorted(
        summaries,
        key=lambda summary: (
            0 if summary.panel == "a" else 1,
            method_order[summary.method],
            summary.d,
        ),
    )
    identities = {(summary.panel, summary.method, summary.d) for summary in ordered}
    if len(ordered) != 9 or len(identities) != 9:
        raise PowerLawFitError(
            f"expected 9 unique fit summaries; found {len(ordered)} rows "
            f"and {len(identities)} unique identities"
        )
    return ordered


def write_fit_summary_csv(path: Path, summaries: list[FitSummary]) -> None:
    """Atomically write the derived nine-row power-law fit summary."""

    fieldnames = [
        "panel",
        "method",
        "d",
        "model",
        "x_variable",
        "y_variable",
        "fit_cutoff",
        "x_min",
        "x_max",
        "n_points",
        "prefactor_A",
        "exponent_p",
        "exponent_stderr",
        "r_squared_log10",
    ]
    ordered = ordered_fit_summaries(summaries)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_handle = tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f"{path.stem}-",
        suffix=".csv",
        dir=path.parent,
        encoding="utf-8",
        newline="",
        delete=False,
    )
    temporary_path = Path(temporary_handle.name)
    try:
        writer = csv.DictWriter(temporary_handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in ordered:
            fit = summary.fit
            writer.writerow(
                {
                    "panel": summary.panel,
                    "method": summary.method,
                    "d": summary.d,
                    "model": summary.model,
                    "x_variable": summary.x_variable,
                    "y_variable": summary.y_variable,
                    "fit_cutoff": (
                        "all"
                        if summary.fit_cutoff is None
                        else f"{summary.fit_cutoff:.17g}"
                    ),
                    "x_min": f"{fit.x_min:.17g}",
                    "x_max": f"{fit.x_max:.17g}",
                    "n_points": fit.n_points,
                    "prefactor_A": f"{fit.prefactor_A:.17g}",
                    "exponent_p": f"{fit.exponent_p:.17g}",
                    "exponent_stderr": f"{fit.exponent_stderr:.17g}",
                    "r_squared_log10": f"{fit.r_squared_log10:.17g}",
                }
            )
        temporary_handle.close()
        temporary_path.replace(path)
    finally:
        if not temporary_handle.closed:
            temporary_handle.close()
        temporary_path.unlink(missing_ok=True)


def print_fit_summary(summaries: list[FitSummary], csv_path: Path) -> None:
    """Print a concise, reproducible fit report."""

    ordered = ordered_fit_summaries(summaries)
    print("Power-law fits (log10-space OLS)")
    for panel, model in (("a", "epsilon = A N^p"), ("b", "N = A n^p")):
        print(f"Panel ({panel}): {model}")
        for summary in (item for item in ordered if item.panel == panel):
            fit = summary.fit
            print(
                f"  {summary.method} d={summary.d}: p={fit.exponent_p:.4f}, "
                f"SE={fit.exponent_stderr:.4f}, R^2={fit.r_squared_log10:.4f}, "
                f"points={fit.n_points}, x=[{fit.x_min:g}, {fit.x_max:g}]"
            )
    print(f"Fit summary CSV: {csv_path}")


def main() -> None:
    """Load both comparisons and generate one two-panel PDF."""

    repository_root = Path(__file__).resolve().parents[1]
    series_0807 = load_0807_series(repository_root)
    series_0910 = load_0910_series(repository_root)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7.2, 3.5))
    fit_summaries = plot_0807_panel(ax_left, series_0807)
    right_log_x = "--right-log-x" in sys.argv[1:]
    fit_summaries.extend(
        plot_0910_panel(ax_right, series_0910, right_log_x=right_log_x)
    )
    handles_by_label: dict[str, Any] = {}
    for ax in (ax_left, ax_right):
        handles, labels = ax.get_legend_handles_labels()
        handles_by_label.update(dict(zip(labels, handles)))
    legend_labels = [
        rf"CEBP ($d={d}$)" for d in (1, 2, 3)
    ] + [rf"LP-LI ($d={d}$)" for d in (1, 2, 3)]
    legend_labels = [label for label in legend_labels if label in handles_by_label]
    fig.legend(
        [handles_by_label[label] for label in legend_labels],
        legend_labels,
        loc="upper center",
        # Center the legend over the combined axes region (0.08..0.99).
        bbox_to_anchor=(0.535, 0.985),
        ncol=6,
        fontsize=7.6,
        frameon=True,
        columnspacing=0.75,
        handlelength=1.65,
        handletextpad=0.4,
        borderpad=0.35,
    )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.13, top=0.89, wspace=0.22)

    output_stem = RIGHT_LOG_X_STEM if right_log_x else OUTPUT_STEM
    output_path = repository_root / "Figs" / f"{output_stem}.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_handle = tempfile.NamedTemporaryFile(
        prefix=f"{OUTPUT_STEM}-", suffix=".pdf", delete=False
    )
    temporary_path = Path(temporary_handle.name)
    temporary_handle.close()
    try:
        fig.savefig(temporary_path, format="pdf")
        temporary_path.replace(output_path)
    finally:
        plt.close(fig)
        temporary_path.unlink(missing_ok=True)
    fit_summary_path = repository_root / "Data_Processing" / FIT_SUMMARY_NAME
    write_fit_summary_csv(fit_summary_path, fit_summaries)
    right_x_scale = "log" if right_log_x else "linear"
    print(f"Plot scales: left=(log, log), right=({right_x_scale}, log)")
    print_fit_summary(fit_summaries, fit_summary_path)
    print(f"Saved PDF: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except (
        comparison_0807.DataValidationError,
        comparison_0910.DataValidationError,
        PowerLawFitError,
    ) as error:
        raise SystemExit(f"error: {error}") from error
