#!/usr/bin/env python3
"""Plot Job 08 training errors alongside Job 07 Local-Pauli errors.

The figure follows ``data_processing_0807_compare.py``: n=8 is on the left,
n=10 on the right, with color/marker encoding d and line style encoding the
method.  Job 08 has 20 initial states and 16 training seeds per state.  For
each copy budget, this script first averages the 16 training-seed errors for
each state and then averages those 20 state errors.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from data_processing_0807_compare import (
    DataValidationError,
    LoadedDataset,
    aggregate,
    finite_trace_distance,
    integer,
    load_job07,
    plot_comparison,
    print_common_table,
    summarize_dataset,
)


OUTPUT_STEM = "data_processing_0807_training_error_n8&n10_d3"


def load_job08_training_csv(
    csv_path: Path, expected_d: int, label: str, expected_n: int
) -> LoadedDataset:
    """Load and state-average one consolidated Job 08 training CSV."""

    dataset = LoadedDataset(label=label)
    if not csv_path.is_file():
        raise DataValidationError(f"{label}: CSV file is missing: {csv_path}")

    required_fields = {
        "n",
        "d",
        "init_id",
        "copy_budget",
        "task_id",
        "training_index",
        "training_seed",
        "trace_distance_error",
        "realized_total_copies",
        "source_result",
    }
    seen: set[tuple[int, int, int]] = set()
    state_rows: dict[tuple[int, int], list[tuple[int, float]]] = defaultdict(list)
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
                    training_index = integer(row["training_index"], "training_index")
                    training_seed = integer(row["training_seed"], "training_seed")
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
                    identity = (init_id, budget, training_index)
                    if identity in seen:
                        raise ValueError("duplicate training identity")
                except (KeyError, ValueError) as error:
                    dataset.skipped[str(error)] += 1
                    continue

                seen.add(identity)
                state_rows[(budget, init_id)].append((training_seed, trace_error))
                dataset.valid_records += 1
    except (OSError, csv.Error) as error:
        raise DataValidationError(f"{label}: could not read {csv_path}: {error}") from error

    if dataset.discovered_records == 0:
        raise DataValidationError(f"{label}: CSV contains no data rows: {csv_path}")

    state_errors_by_budget: dict[int, list[float]] = defaultdict(list)
    states_by_budget: dict[int, set[int]] = defaultdict(set)
    for (budget, init_id), rows in state_rows.items():
        seeds = {seed for seed, _ in rows}
        if len(rows) != 16 or len(seeds) != 16:
            raise DataValidationError(
                f"{label}: expected 16 training seeds for init_id={init_id}, "
                f"copy_budget={budget}; found {len(rows)} rows and {len(seeds)} unique seeds"
            )
        states_by_budget[budget].add(init_id)
        state_errors_by_budget[budget].append(
            sum(error for _, error in rows) / len(rows)
        )
    for budget, state_ids in states_by_budget.items():
        if len(state_ids) != 20:
            raise DataValidationError(
                f"{label}: expected 20 initial states at copy_budget={budget}; "
                f"found {len(state_ids)}"
            )
    dataset.runs_by_budget = defaultdict(list, state_errors_by_budget)
    return dataset


def main() -> None:
    """Load, validate, aggregate, and plot the training-error comparison."""

    repository_root = Path(__file__).resolve().parents[1]
    cases: dict[
        int,
        tuple[
            list[LoadedDataset],
            list[LoadedDataset],
            dict[str, list[Any]],
            dict[str, list[Any]],
        ],
    ] = {}
    for n in (8, 10):
        datasets08 = [
            load_job08_training_csv(
                repository_root / "Data" / "08" / f"n={n}_d={d}_training_errors.csv",
                d,
                f"08 (d={d})",
                expected_n=n,
            )
            for d in (3, 2, 1)
        ]
        datasets07 = []
        for d in (3, 2, 1):
            directory = repository_root / "Data" / "07" / f"n={n}_d={d}"
            archive = repository_root / "Data" / "07" / f"n={n}_d={d}.zip"
            source = directory if directory.is_dir() else archive
            datasets07.append(
                load_job07(source, d, f"07 (d={d})", expected_n=n)
            )
        points08 = {dataset.label: aggregate(dataset) for dataset in datasets08}
        points07 = {dataset.label: aggregate(dataset) for dataset in datasets07}
        cases[n] = (datasets08, datasets07, points08, points07)

        for dataset in datasets08:
            summarize_dataset(
                dataset,
                points08[dataset.label],
                "CSV rows; N_total=copy_budget; state error=mean over 16 training "
                "seeds; plotted error=mean over 20 states",
            )
        for dataset in datasets07:
            summarize_dataset(
                dataset,
                points07[dataset.label],
                "init_*/budget_*/replicate_*.json; "
                "N_total=requested_copies=realized_copies; error=trace_distance",
            )
        print(f"n={n}")
        print_common_table({**points08, **points07})

    panels = [
        (
            rf"Training/reconstruction error vs. total copies ($n={n}$)",
            [
                (dataset, cases[n][2][dataset.label])
                for dataset in cases[n][0]
            ]
            + [
                (dataset, cases[n][3][dataset.label])
                for dataset in cases[n][1]
            ],
        )
        for n in (8, 10)
    ]
    pdf_path, y_scale = plot_comparison(
        panels,
        repository_root / "Figs",
        OUTPUT_STEM,
        show_uncertainty=False,
    )
    print(f"Plot scales: x=log, y={y_scale}; uncertainty=none")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    try:
        main()
    except DataValidationError as error:
        raise SystemExit(f"error: {error}") from error
