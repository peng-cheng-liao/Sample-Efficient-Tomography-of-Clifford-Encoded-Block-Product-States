#!/usr/bin/env python3
"""Compare the n=8 and n=10 Job 08 CEBP and Job 07 Local-Pauli results.

Job 07 stores one completed JSON record per initialization, copy budget, and
replicate.  Job 08 is supplied as one consolidated holdout-error CSV per value
of d.  The plotted quantities are

* Job 07: ``requested_copies`` versus ``trace_distance``;
* Job 08: ``copy_budget`` versus ``trace_distance_error``.

Both errors are the final end-to-end trace distance,
``0.5 * ||rho - rho_hat||_1``.  The script follows the plotting format of
``data_processing_0607_compare.py`` and plots n=8 and n=10 side by side for
d=1, 2, and 3.  Two CEBP reductions are supported: the mean over seeds for
each state followed by a geometric mean over states, and the median over
seeds for each state followed by an arithmetic mean over states.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The user's home Matplotlib cache may be read-only in sandboxed/local runs.
_MPL_CONFIG = tempfile.TemporaryDirectory(prefix="data-processing-0807-mpl-")
os.environ.setdefault("MPLCONFIGDIR", _MPL_CONFIG.name)

import matplotlib

matplotlib.set_loglevel("error")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


N_QUBITS = 8
ERROR_LABEL = "Trace distance"
GEOMETRIC_MEAN_OUTPUT_STEM = (
    "data_processing_0807_compare_n8&n10_d3_seed_mean_state_geometric_mean_"
    "error_vs_copies"
)
MEDIAN_SEED_OUTPUT_STEM = (
    "data_processing_0807_compare_n8&n10_d3_median_seed_mean_state_error_vs_copies"
)


class DataValidationError(RuntimeError):
    """Raised when required inputs cannot produce a valid comparison."""


@dataclass
class LoadedDataset:
    """Validated errors grouped by requested total-copy budget."""

    label: str
    runs_by_budget: dict[int, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    discovered_records: int = 0
    valid_records: int = 0
    skipped: Counter[str] = field(default_factory=Counter)
    malformed_paths: list[Path] = field(default_factory=list)

    @property
    def valid_runs(self) -> int:
        return sum(len(values) for values in self.runs_by_budget.values())


@dataclass(frozen=True)
class Aggregate:
    """Central estimate and sampling dispersion for one copy budget."""

    n_total: int
    estimate: float
    sem: float
    sd: float
    log_sd: float
    count: int


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object, rejecting non-object top-level values."""

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("top-level JSON value is not an object")
    return value


def load_json_from_archive(archive: zipfile.ZipFile, path: Path) -> dict[str, Any]:
    """Load one JSON object from a zip member using the same validation."""

    value = json.loads(archive.read(path.as_posix()))
    if not isinstance(value, dict):
        raise ValueError("top-level JSON value is not an object")
    return value


def integer(value: Any, field_name: str, *, positive: bool = False) -> int:
    """Parse an integer field without accepting booleans or fractional values."""

    if isinstance(value, bool):
        raise ValueError(f"{field_name} is not an integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{field_name} is not an integer")
        result = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        unsigned = stripped[1:] if stripped.startswith(("+", "-")) else stripped
        if not unsigned.isdigit():
            raise ValueError(f"{field_name} is not an integer")
        result = int(stripped)
    else:
        raise ValueError(f"{field_name} is not an integer")
    if (positive and result <= 0) or (not positive and result < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{field_name} is not a {qualifier} integer")
    return result


def finite_trace_distance(value: Any, field_name: str) -> float:
    """Validate a finite physical trace distance."""

    if isinstance(value, bool):
        raise ValueError(f"{field_name} is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field_name} is not numeric") from error
    if not math.isfinite(result) or result < 0.0 or result > 1.0 + 1e-10:
        raise ValueError(f"{field_name} is not a finite trace distance in [0, 1]")
    return result


def path_budget(path: Path) -> int:
    """Extract and validate the nearest ``budget_<integer>`` directory."""

    for parent in path.parents:
        if parent.name.startswith("budget_"):
            suffix = parent.name.removeprefix("budget_")
            if suffix.isdigit() and int(suffix) > 0:
                return int(suffix)
            break
    raise ValueError("path has no valid budget_<integer> directory")


def note_bad_record(dataset: LoadedDataset, path: Path, reason: str) -> None:
    dataset.skipped[reason] += 1
    if reason == "malformed input" and len(dataset.malformed_paths) < 5:
        dataset.malformed_paths.append(path)


def load_job07(
    data_source: Path, expected_d: int, label: str, expected_n: int = N_QUBITS
) -> LoadedDataset:
    """Load completed Local-Pauli per-replicate trace distances."""

    dataset = LoadedDataset(label=label)
    archive: zipfile.ZipFile | None
    if data_source.is_dir():
        paths = sorted(data_source.glob("init_*/budget_*/replicate_*.json"))
        archive = None
    elif data_source.is_file() and data_source.suffix == ".zip":
        archive = zipfile.ZipFile(data_source)
        paths = sorted(
            Path(name)
            for name in archive.namelist()
            if name.startswith("init_")
            and name.endswith(".json")
            and len(Path(name).parts) == 3
            and Path(name).name.startswith("replicate_")
        )
    else:
        raise DataValidationError(f"{label}: data source is missing: {data_source}")

    dataset.discovered_records = len(paths)
    if not paths:
        if archive is not None:
            archive.close()
        raise DataValidationError(f"{label}: no replicate JSON files found in {data_source}")

    seen: set[tuple[int, int, int]] = set()
    try:
        for path in paths:
            try:
                record = (
                    load_json(path)
                    if archive is None
                    else load_json_from_archive(archive, path)
                )
            except (OSError, KeyError, json.JSONDecodeError, ValueError):
                note_bad_record(dataset, path, "malformed input")
                continue

            status = record.get("status")
            if status != "completed":
                note_bad_record(dataset, path, f"status={status or 'missing'}")
                continue

            try:
                if record.get("n") != expected_n or record.get("d") != expected_d:
                    raise ValueError("unexpected n or d")
                budget = integer(record.get("budget"), "budget", positive=True)
                requested = integer(
                    record.get("requested_copies"), "requested_copies", positive=True
                )
                realized = integer(
                    record.get("realized_copies"), "realized_copies", positive=True
                )
                if budget != requested or requested != realized or budget != path_budget(path):
                    raise ValueError("copy-accounting fields disagree")
                if record.get("copy_accounting_check") is not True:
                    raise ValueError("copy-accounting check failed")
                trace_error = finite_trace_distance(
                    record.get("trace_distance"), "trace_distance"
                )
                init_id = integer(record.get("init_id"), "init_id")
                replicate = integer(record.get("replicate_index"), "replicate_index")
                identity = (init_id, budget, replicate)
                if identity in seen:
                    raise ValueError("duplicate run identity")
            except ValueError as error:
                note_bad_record(dataset, path, str(error))
                continue

            seen.add(identity)
            dataset.runs_by_budget[budget].append(trace_error)
            dataset.valid_records += 1
    finally:
        if archive is not None:
            archive.close()
    return dataset


def load_job08_csv(
    csv_path: Path,
    expected_d: int,
    label: str,
    expected_n: int = N_QUBITS,
    *,
    seed_statistic: str = "median",
) -> LoadedDataset:
    """Load CEBP state errors from one consolidated Job 08 CSV.

    The CSV has a fixed number of holdout-seed rows per
    ``(init_id, copy_budget)`` pair.  Those values are reduced to one state
    error using ``seed_statistic`` before being passed to the common
    aggregation routine.
    """

    if seed_statistic not in {"mean", "median"}:
        raise ValueError(f"unsupported seed statistic: {seed_statistic}")

    dataset = LoadedDataset(label=label)
    if not csv_path.is_file():
        raise DataValidationError(f"{label}: CSV file is missing: {csv_path}")

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
    state_values: dict[tuple[int, int], list[tuple[int, float]]] = defaultdict(list)
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required_fields.difference(reader.fieldnames or [])
            if missing:
                raise DataValidationError(
                    f"{label}: CSV is missing required columns: {sorted(missing)}"
                )
            for row in reader:
                dataset.discovered_records += 1
                try:
                    n = integer(row["n"], "n", positive=True)
                    d = integer(row["d"], "d", positive=True)
                    if n != expected_n or d != expected_d:
                        raise ValueError("unexpected n or d")
                    init_id = integer(row["init_id"], "init_id")
                    budget = integer(row["copy_budget"], "copy_budget", positive=True)
                    integer(row["task_id"], "task_id")
                    holdout_index = integer(row["holdout_index"], "holdout_index")
                    holdout_seed = integer(row["holdout_seed"], "holdout_seed")
                    realized = integer(
                        row["realized_total_copies"],
                        "realized_total_copies",
                        positive=True,
                    )
                    if realized > budget:
                        raise ValueError("realized copies exceed requested budget")
                    trace_error = finite_trace_distance(
                        row["trace_distance_error"], "trace_distance_error"
                    )
                    source_result = row["source_result"].strip()
                    if not source_result.endswith("/result.json"):
                        raise ValueError("invalid source_result")
                    identity = (init_id, budget, holdout_index)
                    if identity in seen:
                        raise ValueError("duplicate holdout identity")
                except (KeyError, ValueError) as error:
                    note_bad_record(dataset, csv_path, str(error))
                    continue

                seen.add(identity)
                state_values[(budget, init_id)].append((holdout_seed, trace_error))
                dataset.valid_records += 1
    except (OSError, csv.Error) as error:
        raise DataValidationError(f"{label}: could not read {csv_path}: {error}") from error

    if dataset.discovered_records == 0:
        raise DataValidationError(f"{label}: CSV contains no data rows: {csv_path}")

    # Collapse the holdout-seed errors to one state error for each of the 20
    # initial states at every budget.  Requiring a complete, consistent layout
    # prevents a partially written CSV from silently changing the weighting.
    state_errors_by_budget: dict[int, list[float]] = defaultdict(list)
    states_by_budget: dict[int, set[int]] = defaultdict(set)
    seed_count: int | None = None
    for (budget, init_id), values in state_values.items():
        seeds = {seed for seed, _ in values}
        if len(seeds) != len(values):
            raise DataValidationError(
                f"{label}: duplicate holdout seeds for init_id={init_id}, "
                f"copy_budget={budget}"
            )
        if seed_count is None:
            seed_count = len(values)
        elif len(values) != seed_count:
            raise DataValidationError(
                f"{label}: inconsistent holdout-seed count at init_id={init_id}, "
                f"copy_budget={budget}; expected {seed_count}, found {len(values)}"
            )
        states_by_budget[budget].add(init_id)
        errors = [error for _, error in values]
        state_error = np.mean(errors) if seed_statistic == "mean" else np.median(errors)
        state_errors_by_budget[budget].append(float(state_error))
    for budget, state_ids in states_by_budget.items():
        if len(state_ids) != 20:
            raise DataValidationError(
                f"{label}: expected 20 initial states at copy_budget={budget}; "
                f"found {len(state_ids)}"
            )
    dataset.runs_by_budget = defaultdict(list, state_errors_by_budget)
    return dataset


def aggregate(dataset: LoadedDataset, statistic: str = "mean") -> list[Aggregate]:
    """Compute per-budget central estimates and dispersions from individual errors."""

    if statistic not in {"mean", "median", "geometric_mean"}:
        raise ValueError(f"unsupported aggregation statistic: {statistic}")

    points: list[Aggregate] = []
    for budget in sorted(dataset.runs_by_budget):
        values = np.asarray(dataset.runs_by_budget[budget], dtype=float)
        if values.size == 0:
            continue
        sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        log_sd = (
            float(np.std(np.log(values), ddof=1))
            if values.size > 1 and np.all(values > 0)
            else math.nan
        )
        if statistic == "geometric_mean":
            if not np.all(values > 0):
                raise DataValidationError(
                    f"{dataset.label}: geometric mean requires strictly positive errors"
                )
            estimate = float(np.exp(np.mean(np.log(values))))
        elif statistic == "mean":
            estimate = float(np.mean(values))
        else:
            estimate = float(np.median(values))
        points.append(
            Aggregate(
                n_total=budget,
                estimate=estimate,
                sem=sd / math.sqrt(int(values.size)) if values.size > 1 else 0.0,
                sd=sd,
                log_sd=log_sd,
                count=int(values.size),
            )
        )
    if not points:
        raise DataValidationError(f"{dataset.label}: no valid copy-budget points extracted")
    return points


def summarize_dataset(dataset: LoadedDataset, points: list[Aggregate], schema: str) -> None:
    """Print the schema, counts per budget, and concise skip diagnostics."""

    print(f"{dataset.label} schema: {schema}; records={dataset.discovered_records}")
    counts = ", ".join(f"{point.n_total:g}:{point.count}" for point in points)
    print(
        f"{dataset.label}: {len(points)} plotted budgets, {dataset.valid_runs} plotted runs "
        f"from {dataset.valid_records} valid record(s); run counts N_total:n = {counts}"
    )
    skipped_count = sum(dataset.skipped.values())
    if skipped_count:
        reasons = ", ".join(
            f"{reason} ({count})" for reason, count in sorted(dataset.skipped.items())
        )
        print(f"{dataset.label}: skipped {skipped_count} invalid/incomplete record(s): {reasons}")
    else:
        print(f"{dataset.label}: skipped 0 invalid/incomplete records")
    for path in dataset.malformed_paths:
        print(f"{dataset.label}: malformed input: {path}", file=sys.stderr)


def print_common_table(
    datasets: dict[str, list[Aggregate]], estimate_label: str = "mean"
) -> None:
    """Print central estimates at copy budgets common to every dataset."""

    by_budget = {
        label: {point.n_total: point for point in points}
        for label, points in datasets.items()
    }
    common = sorted(set.intersection(*(set(values) for values in by_budget.values())))
    if not common:
        print("Common-budget table: no shared valid N_total values")
        return

    print("Common-budget table:")
    print(
        "N_total       "
        + "  ".join(f"{estimate_label}_{label}" for label in datasets)
    )
    for budget in common:
        estimates = "  ".join(
            f"{by_budget[label][budget].estimate: .6e}" for label in datasets
        )
        print(f"{budget:<13d} {estimates}")


def plot_comparison(
    panels: list[tuple[str, list[tuple[LoadedDataset, list[Aggregate]]]]],
    output_dir: Path,
    output_stem: str,
    *,
    show_uncertainty: bool,
    uncertainty_mode: str = "linear_sd",
) -> tuple[Path, str]:
    """Create side-by-side n=8 and n=10 panels in the established 06/07 style."""

    if uncertainty_mode not in {"linear_sd", "log_sd"}:
        raise ValueError(f"unsupported uncertainty mode: {uncertainty_mode}")

    fig, axes = plt.subplots(1, len(panels), figsize=(13.8, 4.8), squeeze=False)
    axes = axes[0]
    styles: dict[str, dict[str, Any]] = {}
    colors = {3: "#1f77b4", 2: "#9467bd", 1: "#2ca02c"}
    markers = {3: "o", 2: "^", 1: "D"}
    for d in (3, 2, 1):
        styles[f"08 (d={d})"] = {
            "color": colors[d],
            "marker": markers[d],
            "linestyle": "-",
            "legend_label": f"CEBP tomography (d={d})",
        }
        styles[f"07 (d={d})"] = {
            "color": colors[d],
            "marker": markers[d],
            "linestyle": "--",
            "legend_label": f"Local-Pauli tomography (d={d})",
        }

    y_scales: list[str] = []
    for ax, (title, datasets) in zip(axes, panels):
        all_errors: list[float] = []
        for dataset, points in datasets:
            x = np.asarray([point.n_total for point in points], dtype=float)
            y = np.asarray([point.estimate for point in points], dtype=float)
            sd = np.asarray([point.sd for point in points], dtype=float)
            log_sd = np.asarray([point.log_sd for point in points], dtype=float)
            all_errors.extend(y.tolist())
            style = styles[dataset.label]
            ax.plot(
                x,
                y,
                label=style["legend_label"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                markersize=5.5,
                linewidth=1.8,
            )
            if show_uncertainty:
                if uncertainty_mode == "log_sd":
                    if not np.all(np.isfinite(log_sd)):
                        raise DataValidationError(
                            f"{dataset.label}: log-space SD requires strictly "
                            "positive errors"
                        )
                    factor = np.exp(log_sd)
                    lower = y / factor
                    upper = y * factor
                else:
                    # Keep the lower endpoint positive for the log-scaled
                    # y-axis.  The bars otherwise represent +/- one SD in
                    # linear trace-distance units.
                    lower = np.maximum(y - sd, np.finfo(float).tiny)
                    upper = y + sd
                ax.errorbar(
                    x,
                    y,
                    yerr=np.vstack((y - lower, upper - y)),
                    fmt="none",
                    ecolor=style["color"],
                    elinewidth=1.0,
                    capsize=2.5,
                    capthick=0.8,
                    alpha=0.8,
                    zorder=2,
                )

        errors = np.asarray(all_errors, dtype=float)
        y_scale = "linear"
        if np.all(errors > 0) and errors.max() / errors.min() >= 10.0:
            ax.set_yscale("log")
            y_scale = "log"
        y_scales.append(y_scale)
        ax.set_xscale("log")
        ax.set_xlim(left=1_000)
        ax.set_title(title)
        ax.set_xlabel("Total copies")
        ax.set_ylabel(ERROR_LABEL)
        ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.45)

    handle_by_label = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        handle_by_label.update(dict(zip(labels, handles)))
    desired_labels = [
        method_label
        for d in (1, 2, 3)
        for method_label in (
            f"CEBP tomography (d={d})",
            f"Local-Pauli tomography (d={d})",
        )
    ]
    legend_labels = [label for label in desired_labels if label in handle_by_label]
    fig.legend(
        [handle_by_label[label] for label in legend_labels],
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=True,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{output_stem}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, ", ".join(y_scales)


def main() -> None:
    """Load, validate, aggregate, compare, and plot n=8 and n=10 datasets."""

    repository_root = Path(__file__).resolve().parents[1]
    cases: dict[int, dict[str, Any]] = {}
    for n in (8, 10):
        data07_sources = {}
        for d in (3, 2, 1):
            directory = repository_root / "Data" / "07" / f"n={n}_d={d}"
            archive = repository_root / "Data" / "07" / f"n={n}_d={d}.zip"
            data07_sources[d] = directory if directory.is_dir() else archive
        data08_csvs = {
            d: repository_root / "Data" / "08" / f"n={n}_d={d}_holdout_errors.csv"
            for d in (3, 2, 1)
        }

        datasets08_all_mean = [
            load_job08_csv(
                data08_csvs[d], d, f"08 (d={d})", expected_n=n, seed_statistic="mean"
            )
            for d in (3, 2, 1)
        ]
        datasets08_median_seed = [
            load_job08_csv(
                data08_csvs[d], d, f"08 (d={d})", expected_n=n, seed_statistic="median"
            )
            for d in (3, 2, 1)
        ]
        datasets07 = [
            load_job07(data07_sources[d], d, f"07 (d={d})", expected_n=n)
            for d in (3, 2, 1)
        ]
        points08_geometric_mean = {
            dataset.label: aggregate(dataset, statistic="geometric_mean")
            for dataset in datasets08_all_mean
        }
        points08_median_seed = {
            dataset.label: aggregate(dataset) for dataset in datasets08_median_seed
        }
        points07_mean = {dataset.label: aggregate(dataset) for dataset in datasets07}
        points07_geometric_mean = {
            dataset.label: aggregate(dataset, statistic="geometric_mean")
            for dataset in datasets07
        }
        cases[n] = {
            "datasets08_all_mean": datasets08_all_mean,
            "datasets08_median_seed": datasets08_median_seed,
            "datasets07": datasets07,
            "points08_geometric_mean": points08_geometric_mean,
            "points08_median_seed": points08_median_seed,
            "points07_mean": points07_mean,
            "points07_geometric_mean": points07_geometric_mean,
        }

        for dataset in datasets08_all_mean:
            summarize_dataset(
                dataset,
                points08_geometric_mean[dataset.label],
                "CSV rows; N_total=copy_budget; state error=mean over holdout seeds; "
                "plotted center=geometric mean over 20 states",
            )
        for dataset in datasets08_median_seed:
            summarize_dataset(
                dataset,
                points08_median_seed[dataset.label],
                "CSV rows; N_total=copy_budget; state error=median over holdout "
                "seeds; plotted error=mean over 20 states",
            )
        for dataset in datasets07:
            summarize_dataset(
                dataset,
                points07_mean[dataset.label],
                "init_*/budget_*/replicate_*.json; "
                "N_total=requested_copies=realized_copies; error=trace_distance",
            )
        print(f"n={n}")
        print("seed mean/state geometric mean")
        print_common_table(
            {**points08_geometric_mean, **points07_geometric_mean},
            estimate_label="geometric_mean",
        )
        print("median-seed/state mean")
        print_common_table({**points08_median_seed, **points07_mean})

    def make_panels(
        points08_key: str, points07_key: str, title_suffix: str
    ) -> list[
        tuple[str, list[tuple[LoadedDataset, list[Aggregate]]]]
    ]:
        return [
            (
                rf"Reconstruction error vs. total copies ($n={n}$; {title_suffix})",
                [
                    (dataset, cases[n][points08_key][dataset.label])
                    for dataset in cases[n]["datasets08_all_mean"]
                ]
                + [
                    (dataset, cases[n][points07_key][dataset.label])
                    for dataset in cases[n]["datasets07"]
                ],
            )
            for n in (8, 10)
        ]

    geometric_mean_pdf, geometric_mean_y_scale = plot_comparison(
        make_panels(
            "points08_geometric_mean",
            "points07_geometric_mean",
            "geometric mean",
        ),
        repository_root / "Figs",
        GEOMETRIC_MEAN_OUTPUT_STEM,
        show_uncertainty=True,
        uncertainty_mode="log_sd",
    )
    print(
        f"Geometric-mean plot scales: x=log, y={geometric_mean_y_scale}; "
        "uncertainty=multiplicative log-SD"
    )
    print(f"Saved: {geometric_mean_pdf}")

    median_seed_pdf, median_seed_y_scale = plot_comparison(
        make_panels(
            "points08_median_seed", "points07_mean", "median-seed/state mean"
        ),
        repository_root / "Figs",
        MEDIAN_SEED_OUTPUT_STEM,
        show_uncertainty=True,
    )
    print(
        f"Median-seed plot scales: x=log, y={median_seed_y_scale}; uncertainty=SD"
    )
    print(f"Saved: {median_seed_pdf}")


if __name__ == "__main__":
    try:
        main()
    except DataValidationError as error:
        raise SystemExit(f"error: {error}") from error
