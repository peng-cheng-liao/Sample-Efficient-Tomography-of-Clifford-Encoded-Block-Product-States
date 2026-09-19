#!/usr/bin/env python3
"""Compare completed n=10 and n=8 tomography simulations for d=1, 2, and 3.

Job 06 and Job 07 intentionally have different result schemas.  Job 06
stores one ``result.json`` per initialization and copy budget, with six
scientific holdout errors inside that record.  Job 07 stores one JSON record
per initialization, copy budget, and replicate.

The selected quantities are:

* 06 x: ``requested_copy_budget``; y:
  ``holdout_error_summary.individual_errors``.  The 06 producer labels this
  summary ``scientific_headline_when_complete``; ``best_comparison_error`` is
  only a model-selection diagnostic and is deliberately not plotted.
* 07 x: ``requested_copies`` (required to equal both ``budget`` and
  ``realized_copies``); y: ``trace_distance`` from completed records.  The
  same schema is used for ``Data/07/n=10_d=1``, ``d=2``, and ``d=3``.

Both y quantities are the final end-to-end trace distance, 0.5 ||rho-rho_hat||_1.
"""

from __future__ import annotations

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
# Use a process-local temporary cache and let TemporaryDirectory remove it.
_MPL_CONFIG = tempfile.TemporaryDirectory(prefix="data-processing-0607-mpl-")
os.environ.setdefault("MPLCONFIGDIR", _MPL_CONFIG.name)

import matplotlib

matplotlib.set_loglevel("error")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


N_QUBITS = 10
ERROR_LABEL = "Trace distance"
# Keep the established output name so downstream figure references continue
# to point to the updated two-panel comparison.
OUTPUT_STEM = "data_processing_0607_compare_n10_d3_error_vs_copies"
MEDIAN_OUTPUT_STEM = "data_processing_0607_compare_n10_d3_median_error_vs_copies"
# All currently complete 06/d=1 budgets are plotted.  Earlier incomplete
# 06/d=1 records at 5e8 and 1e9 copies are now complete and are included.
PLOT_EXCLUDED_BUDGETS: dict[str, frozenset[int]] = {}


class DataValidationError(RuntimeError):
    """Raised when required inputs cannot produce a valid comparison."""


@dataclass
class LoadedDataset:
    """Validated seed/run errors grouped by requested total-copy budget."""

    label: str
    runs_by_budget: dict[int, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    discovered_records: int = 0
    valid_records: int = 0
    skipped: Counter[str] = field(default_factory=Counter)
    malformed_paths: list[Path] = field(default_factory=list)
    auxiliary_records: int = 0

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


def positive_integer(value: Any, field_name: str) -> int:
    """Return a strictly positive integer without accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} is not numeric")
    if not math.isfinite(float(value)) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{field_name} is not a positive integer")
    return int(value)


def finite_trace_distance(value: Any, field_name: str) -> float:
    """Validate a finite physical trace distance."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0 + 1e-10:
        raise ValueError(f"{field_name} is not a finite trace distance in [0, 1]")
    return result


def path_budget(path: Path) -> int:
    """Extract and validate the nearest ``budget_<integer>`` directory."""

    for parent in path.parents:
        if parent.name.startswith("budget_"):
            suffix = parent.name.removeprefix("budget_")
            if not suffix.isdigit() or int(suffix) <= 0:
                break
            return int(suffix)
    raise ValueError("path has no valid budget_<integer> directory")


def note_bad_record(dataset: LoadedDataset, path: Path, reason: str) -> None:
    dataset.skipped[reason] += 1
    if reason == "malformed JSON" and len(dataset.malformed_paths) < 5:
        dataset.malformed_paths.append(path)


def load_job06(
    data_source: Path, expected_d: int, label: str, expected_n: int = N_QUBITS
) -> LoadedDataset:
    """Load complete scientific holdout errors from one Job 06 case.

    Job 06 datasets may be present either as an unpacked directory or as the
    archived ``n=*_d=*.zip`` files used for the larger n=8 runs.
    """

    dataset = LoadedDataset(label=label)
    if data_source.is_dir():
        paths = sorted(data_source.glob("full_scan/init_*/budget_*/workers_*/result.json"))
        dataset.discovered_records = len(paths)
        dataset.auxiliary_records = max(
            0, len(list(data_source.rglob("result.json"))) - dataset.discovered_records
        )
        archive = None
    elif data_source.is_file() and data_source.suffix == ".zip":
        archive = zipfile.ZipFile(data_source)
        paths = sorted(
            Path(name)
            for name in archive.namelist()
            if name.startswith("full_scan/")
            and name.endswith("/result.json")
            and len(Path(name).parts) == 5
        )
        dataset.discovered_records = len(paths)
        dataset.auxiliary_records = max(
            0,
            sum(name.endswith("/result.json") for name in archive.namelist())
            - dataset.discovered_records,
        )

    else:
        raise DataValidationError(f"06: data source is missing: {data_source}")

    if not dataset.discovered_records:
        raise DataValidationError(f"06: no full-scan result.json files found in {data_source}")

    for path in paths:
        try:
            record = load_json(path) if archive is None else load_json_from_archive(archive, path)
        except (OSError, json.JSONDecodeError, ValueError):
            note_bad_record(dataset, path, "malformed JSON")
            continue

        try:
            task = record.get("task")
            if not isinstance(task, dict) or task.get("task_kind") != "full_scan":
                raise ValueError("wrong task kind")
            if record.get("status") != "success":
                raise ValueError(f"status={record.get('status', 'missing')}")
            partition = record.get("partition")
            if (
                not isinstance(partition, list)
                or not partition
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                    for value in partition
                )
                or sum(partition) != expected_n
                or max(partition) != expected_d
            ):
                raise ValueError("unexpected partition")

            budget = positive_integer(
                record.get("requested_copy_budget"), "requested_copy_budget"
            )
            task_budget = positive_integer(task.get("budget"), "task.budget")
            if budget != task_budget or budget != path_budget(path):
                raise ValueError("copy-budget fields disagree")

            summary = record.get("holdout_error_summary")
            if not isinstance(summary, dict):
                raise ValueError("missing holdout error summary")
            if (
                summary.get("status") != "complete"
                or summary.get("complete") is not True
                or summary.get("role") != "scientific_headline_when_complete"
            ):
                raise ValueError("holdout summary is incomplete")
            raw_errors = summary.get("individual_errors")
            if not isinstance(raw_errors, list) or not raw_errors:
                raise ValueError("missing holdout seed errors")
            errors = [
                finite_trace_distance(value, "holdout_error_summary.individual_errors")
                for value in raw_errors
            ]
            requested = positive_integer(
                summary.get("requested_seed_count"), "requested_seed_count"
            )
            available = positive_integer(
                summary.get("available_seed_count"), "available_seed_count"
            )
            if requested != available or available != len(errors):
                raise ValueError("holdout seed counts disagree")
            recorded_mean = finite_trace_distance(
                summary.get("mean"), "holdout_error_summary.mean"
            )
            if not math.isclose(
                recorded_mean, float(np.mean(errors)), rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError("holdout mean disagrees with individual errors")

            holdout = record.get("holdout")
            seed_records = holdout.get("seed_evaluations") if isinstance(holdout, dict) else None
            if not isinstance(seed_records, list) or len(seed_records) != len(errors):
                raise ValueError("holdout seed records are missing")
            validated_seed_errors: list[float] = []
            for seed in seed_records:
                if not isinstance(seed, dict):
                    raise ValueError("malformed holdout seed record")
                if (
                    seed.get("operational_success") is not True
                    or seed.get("loss_computed") is not True
                    or seed.get("estimator_available") is not True
                    or seed.get("budget_feasible") is not True
                ):
                    raise ValueError("failed or incomplete holdout seed")
                realized = positive_integer(seed.get("realized_total"), "realized_total")
                if realized > budget:
                    raise ValueError("realized copies exceed requested budget")
                validated_seed_errors.append(
                    finite_trace_distance(seed.get("trace_distance"), "trace_distance")
                )
            if not np.allclose(
                validated_seed_errors, errors, rtol=1e-12, atol=1e-12
            ):
                raise ValueError("holdout seed errors disagree with summary")
        except ValueError as error:
            note_bad_record(dataset, path, str(error))
            continue

        # ``execution_complete`` is a representative graceful-stage diagnostic,
        # not the record completion flag.  Some fixed-budget runs set it false
        # while every holdout seed is operationally successful and the producer
        # explicitly marks the scientific holdout summary complete.
        dataset.runs_by_budget[budget].extend(errors)
        dataset.valid_records += 1

    if archive is not None:
        archive.close()
    return dataset


def load_job07(
    data_source: Path, expected_d: int, label: str, expected_n: int = N_QUBITS
) -> LoadedDataset:
    """Load completed per-replicate trace distances from one Job 07 case."""

    dataset = LoadedDataset(label=label)
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
        raise DataValidationError(f"{label}: no replicate JSON files found in {data_source}")

    seen: set[tuple[int, int, int]] = set()
    for path in paths:
        try:
            record = load_json(path) if archive is None else load_json_from_archive(archive, path)
        except (OSError, json.JSONDecodeError, ValueError):
            note_bad_record(dataset, path, "malformed JSON")
            continue

        status = record.get("status")
        if status != "completed":
            note_bad_record(dataset, path, f"status={status or 'missing'}")
            continue

        try:
            if record.get("n") != expected_n or record.get("d") != expected_d:
                raise ValueError("unexpected n or d")
            budget = positive_integer(record.get("budget"), "budget")
            requested = positive_integer(record.get("requested_copies"), "requested_copies")
            realized = positive_integer(record.get("realized_copies"), "realized_copies")
            if budget != requested or requested != realized or budget != path_budget(path):
                raise ValueError("copy-accounting fields disagree")
            if record.get("copy_accounting_check") is not True:
                raise ValueError("copy-accounting check failed")
            trace_error = finite_trace_distance(
                record.get("trace_distance"), "trace_distance"
            )
            init_id = int(record["init_id"])
            replicate = int(record["replicate_index"])
            identity = (init_id, budget, replicate)
            if identity in seen:
                raise ValueError("duplicate run identity")
        except (KeyError, TypeError, ValueError) as error:
            note_bad_record(dataset, path, str(error))
            continue

        seen.add(identity)
        dataset.runs_by_budget[budget].append(trace_error)
        dataset.valid_records += 1

    if archive is not None:
        archive.close()
    return dataset


def aggregate(dataset: LoadedDataset, statistic: str = "mean") -> list[Aggregate]:
    """Compute per-budget central estimates and SEM from the individual runs."""

    if statistic not in {"mean", "median"}:
        raise ValueError(f"unsupported aggregation statistic: {statistic}")

    points: list[Aggregate] = []
    for budget in sorted(dataset.runs_by_budget):
        values = np.asarray(dataset.runs_by_budget[budget], dtype=float)
        if values.size == 0:
            continue
        sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        estimate = float(np.mean(values) if statistic == "mean" else np.median(values))
        points.append(
            Aggregate(
                n_total=budget,
                estimate=estimate,
                sem=sd / math.sqrt(int(values.size)) if values.size > 1 else 0.0,
                sd=sd,
                count=int(values.size),
            )
        )
    if not points:
        raise DataValidationError(f"{dataset.label}: no valid copy-budget points extracted")
    return points


def summarize_schema(
    datasets06: list[LoadedDataset], datasets07: list[LoadedDataset]
) -> None:
    """Print a compact description of the schemas and validation outcome."""

    for dataset in datasets06:
        print(
            f"{dataset.label} schema: full_scan/init_*/budget_*/workers_*/result.json; "
            "N_total=requested_copy_budget; "
            "error=holdout_error_summary.individual_errors "
            f"(trace distance); records={dataset.discovered_records}"
        )
        if dataset.auxiliary_records:
            print(
                f"{dataset.label} scope: excluded "
                f"{dataset.auxiliary_records} calibration/smoke result.json record(s)"
            )
    for dataset in datasets07:
        print(
            f"{dataset.label} schema: init_*/budget_*/replicate_*.json; "
            "N_total=requested_copies=realized_copies; error=trace_distance; "
            f"records={dataset.discovered_records}"
        )


def summarize_dataset(dataset: LoadedDataset, points: list[Aggregate]) -> None:
    """Print counts per budget and concise skip diagnostics."""

    counts = ", ".join(f"{point.n_total:g}:{point.count}" for point in points)
    plotted_runs = sum(point.count for point in points)
    print(
        f"{dataset.label}: {len(points)} plotted budgets, {plotted_runs} plotted runs "
        f"({dataset.valid_runs} validated total) "
        f"from {dataset.valid_records} record(s); run counts N_total:n = {counts}"
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
        print(f"{dataset.label}: malformed file: {path}", file=sys.stderr)


def print_common_table(datasets: dict[str, list[Aggregate]]) -> None:
    """Print means at copy budgets common to every plotted dataset."""

    by_budget = {
        label: {point.n_total: point for point in points}
        for label, points in datasets.items()
    }
    common = sorted(set.intersection(*(set(values) for values in by_budget.values())))
    if not common:
        print("Common-budget table: no shared valid N_total values")
        return

    print("Common-budget table:")
    print("N_total       " + "  ".join(f"mean_{label}" for label in datasets))
    for budget in common:
        means = "  ".join(f"{by_budget[label][budget].estimate: .6e}" for label in datasets)
        print(f"{budget:<13d} {means}")


def plot_comparison(
    panels: list[tuple[str, list[tuple[LoadedDataset, list[Aggregate]]]]],
    output_dir: Path,
    output_stem: str = OUTPUT_STEM,
    show_uncertainty: bool = True,
) -> tuple[Path, Path, str]:
    """Create and save side-by-side n=10 and n=8 comparison panels."""

    fig, axes = plt.subplots(1, len(panels), figsize=(13.8, 4.8), squeeze=False)
    axes = axes[0]
    styles = {
        # Color and marker encode d; line style encodes the dataset.
        "06 (d=3)": {
            "color": "#1f77b4",
            "marker": "o",
            "linestyle": "-",
            "legend_label": "CEBP tomography (d=3)",
        },
        "06 (d=2)": {
            "color": "#9467bd",
            "marker": "^",
            "linestyle": "-",
            "legend_label": "CEBP tomography (d=2)",
        },
        "06 (d=1)": {
            "color": "#2ca02c",
            "marker": "D",
            "linestyle": "-",
            "legend_label": "CEBP tomography (d=1)",
        },
        "07 (d=3)": {
            "color": "#1f77b4",
            "marker": "o",
            "linestyle": "--",
            "legend_label": "Local-Pauli tomography (d=3)",
        },
        "07 (d=2)": {
            "color": "#9467bd",
            "marker": "^",
            "linestyle": "--",
            "legend_label": "Local-Pauli tomography (d=2)",
        },
        "07 (d=1)": {
            "color": "#2ca02c",
            "marker": "D",
            "linestyle": "--",
            "legend_label": "Local-Pauli tomography (d=1)",
        },
    }
    y_scales: list[str] = []
    for ax, (title, datasets) in zip(axes, panels):
        all_errors: list[float] = []
        for dataset, points in datasets:
            label = dataset.label
            x = np.asarray([point.n_total for point in points], dtype=float)
            y = np.asarray([point.estimate for point in points], dtype=float)
            sem = np.asarray([point.sem for point in points], dtype=float)
            all_errors.extend(y.tolist())
            style = styles[label]
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
                ax.fill_between(
                    x,
                    np.maximum(y - sem, np.finfo(float).tiny),
                    y + sem,
                    color=style["color"],
                    alpha=0.18,
                    linewidth=0,
                )

        all_errors = np.asarray(all_errors, dtype=float)
        y_scale = "linear"
        if np.all(all_errors > 0) and all_errors.max() / all_errors.min() >= 10.0:
            ax.set_yscale("log")
            y_scale = "log"
        y_scales.append(y_scale)
        ax.set_xscale("log")
        ax.set_xlim(left=1_000)
        ax.set_title(title)
        ax.set_xlabel("Total copies")
        ax.set_ylabel(ERROR_LABEL)
        ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.45)
    # Collect handles across panels so the median figure also includes the
    # Local-Pauli entries that only appear in its n=10 panel.
    handle_by_label = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        handle_by_label.update(dict(zip(labels, handles)))
    desired_legend_labels = [
        method_label
        for d in (1, 2, 3)
        for method_label in (
            f"CEBP tomography (d={d})",
            f"Local-Pauli tomography (d={d})",
        )
    ]
    legend_labels = [
        label for label in desired_legend_labels if label in handle_by_label
    ]
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
    png_path = output_dir / f"{output_stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=350, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path, ", ".join(y_scales)


def main() -> None:
    """Load, validate, aggregate, compare, and plot all requested datasets."""

    repository_root = Path(__file__).resolve().parents[1]
    data06_n10_dirs = {
        d: repository_root / "Data" / "06" / f"n=10_d={d}"
        for d in (3, 2, 1)
    }
    data07_n10_dirs = {
        d: repository_root / "Data" / "07" / f"n=10_d={d}"
        for d in (3, 2, 1)
    }
    data07_n8_archives = {
        d: repository_root / "Data" / "07" / f"n=8_d={d}.zip"
        for d in (3, 2, 1)
    }
    data07_n8_sources = {
        d: (
            repository_root / "Data" / "07" / f"n=8_d={d}"
            if (repository_root / "Data" / "07" / f"n=8_d={d}").is_dir()
            else data07_n8_archives[d]
        )
        for d in (3, 2, 1)
    }
    data06_n8_archives = {
        d: repository_root / "Data" / "06" / f"n=8_d={d}.zip"
        for d in (3, 2, 1)
    }
    required_dirs = [
        (f"06 (n=10, d={d})", directory)
        for d, directory in data06_n10_dirs.items()
    ] + [
        (f"07 (n=10, d={d})", directory)
        for d, directory in data07_n10_dirs.items()
    ] + [
        (f"06 (n=8, d={d})", archive)
        for d, archive in data06_n8_archives.items()
    ] + [
        (f"07 (n=8, d={d})", archive)
        for d, archive in data07_n8_sources.items()
    ]
    for label, directory in required_dirs:
        if not directory.exists():
            raise DataValidationError(f"{label}: required data directory is missing: {directory}")

    datasets06 = [
        load_job06(data06_n10_dirs[d], d, f"06 (d={d})", expected_n=10)
        for d in (3, 2, 1)
    ]
    datasets07 = [
        load_job07(data07_n10_dirs[d], d, f"07 (d={d})") for d in (3, 2, 1)
    ]
    datasets07_n8 = [
        load_job07(data07_n8_sources[d], d, f"07 (d={d})", expected_n=8)
        for d in (3, 2, 1)
    ]
    datasets06_n8 = [
        load_job06(data06_n8_archives[d], d, f"06 (d={d})", expected_n=8)
        for d in (3, 2, 1)
    ]
    points06 = {dataset.label: aggregate(dataset) for dataset in datasets06}
    points07 = {dataset.label: aggregate(dataset) for dataset in datasets07}
    points07_n8 = {dataset.label: aggregate(dataset) for dataset in datasets07_n8}
    points06_n8 = {dataset.label: aggregate(dataset) for dataset in datasets06_n8}

    for label, excluded_budgets in PLOT_EXCLUDED_BUDGETS.items():
        original_points = points06.get(label, [])
        points06[label] = [
            point for point in original_points if point.n_total not in excluded_budgets
        ]
        removed = sorted(
            point.n_total for point in original_points if point.n_total in excluded_budgets
        )
        if removed:
            print(f"{label}: omitted from plot at N_total={removed}")

    summarize_schema(datasets06, datasets07)
    summarize_schema(datasets06_n8, [])
    summarize_schema([], datasets07_n8)
    for dataset in datasets06:
        summarize_dataset(dataset, points06[dataset.label])
    for dataset in datasets07:
        summarize_dataset(dataset, points07[dataset.label])
    for dataset in datasets06_n8:
        summarize_dataset(dataset, points06_n8[dataset.label])
    for dataset in datasets07_n8:
        summarize_dataset(dataset, points07_n8[dataset.label])
    print("n=10")
    print_common_table({**points06, **points07})
    print("n=8")
    print_common_table({**points06_n8, **points07_n8})
    panels = [
        (
            r"Reconstruction error vs. total copies ($n=8$; $d=1,2,3$)",
            [(dataset, points06_n8[dataset.label]) for dataset in datasets06_n8]
            + [(dataset, points07_n8[dataset.label]) for dataset in datasets07_n8],
        ),
        (
            r"Reconstruction error vs. total copies ($n=10$; $d=1,2,3$)",
            [(dataset, points06[dataset.label]) for dataset in datasets06]
            + [(dataset, points07[dataset.label]) for dataset in datasets07],
        ),
    ]
    pdf_path, png_path, y_scale = plot_comparison(
        panels,
        repository_root / "Figs",
    )
    print(f"Plot scales: x=log, y={y_scale}; uncertainty=SEM")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

    median_points06 = {
        dataset.label: aggregate(dataset, statistic="median") for dataset in datasets06
    }
    median_points07 = {
        dataset.label: aggregate(dataset, statistic="median") for dataset in datasets07
    }
    median_points06_n8 = {
        dataset.label: aggregate(dataset, statistic="median")
        for dataset in datasets06_n8
    }
    for label, excluded_budgets in PLOT_EXCLUDED_BUDGETS.items():
        median_points06[label] = [
            point
            for point in median_points06.get(label, [])
            if point.n_total not in excluded_budgets
        ]
    median_panels = [
        (
            r"Median reconstruction error vs. total copies ($n=8$; $d=1,2,3$)",
            [
                (dataset, median_points06_n8[dataset.label])
                for dataset in datasets06_n8
            ],
        ),
        (
            r"Median reconstruction error vs. total copies ($n=10$; $d=1,2,3$)",
            [(dataset, median_points06[dataset.label]) for dataset in datasets06]
            + [(dataset, median_points07[dataset.label]) for dataset in datasets07],
        ),
    ]
    median_pdf_path, median_png_path, median_y_scale = plot_comparison(
        median_panels,
        repository_root / "Figs",
        output_stem=MEDIAN_OUTPUT_STEM,
        show_uncertainty=False,
    )
    print(f"Median plot scales: x=log, y={median_y_scale}; uncertainty=none")
    print(f"Saved: {median_pdf_path}")
    print(f"Saved: {median_png_path}")


if __name__ == "__main__":
    try:
        main()
    except DataValidationError as error:
        raise SystemExit(f"error: {error}") from error
