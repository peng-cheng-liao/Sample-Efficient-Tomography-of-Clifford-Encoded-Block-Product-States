#!/usr/bin/env python3
"""Compare Job 09 and Job 10 minimum-copy scaling with qubit count.

The plotted quantity is the arithmetic mean, over target states, of the
minimum total number of copies required to reach the fixed trace-distance
target at each number of qubits ``n``:

* Job 09 (CEBP tomography): ``minimum_copies`` from the CSV export;
* Job 10 (Local-Pauli tomography): ``minimum_total_copies_N`` from the XLSX
  export, validated against ``minimum_M_per_setting * 3**n``.

Only scientifically completed rows with a positive minimum-copy count enter
the averages.  CEBP uses the geometric mean with a multiplicative one-standard-
deviation interval computed in log space, while the Local-Pauli series retains
its arithmetic mean and SEM band.  The script uses only the Python standard
library for XLSX parsing so it does not require pandas or openpyxl.
"""

from __future__ import annotations

import csv
import math
import os
import posixpath
import re
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

# The user's home Matplotlib cache may be read-only in sandboxed/local runs.
_MPL_CONFIG = tempfile.TemporaryDirectory(prefix="data-processing-0910-mpl-")
os.environ.setdefault("MPLCONFIGDIR", _MPL_CONFIG.name)

import matplotlib

matplotlib.set_loglevel("error")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_N_VALUES = tuple(range(1, 10))
EXPECTED_STATES_PER_N = 20
OUTPUT_STEM = "data_processing_0910_compare_average_minimum_total_copies_vs_n"

_SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_RELATIONSHIPS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_OFFICE_REL_NS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
_CELL_REFERENCE = re.compile(r"^([A-Z]+)[1-9][0-9]*$")


class DataValidationError(RuntimeError):
    """Raised when an input cannot produce a scientifically valid series."""


@dataclass
class LoadedDataset:
    """Validated minimum-copy counts grouped by qubit count."""

    label: str
    copies_by_n: dict[int, list[int]] = field(
        default_factory=lambda: defaultdict(list)
    )
    discovered_records: int = 0
    valid_records: int = 0
    skipped: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True)
class Aggregate:
    """Arithmetic and log-space summaries for one value of n."""

    n: int
    mean: float
    sem: float
    sd: float
    geometric_mean: float
    log_sd: float
    geometric_lower: float
    geometric_upper: float
    count: int


def integer(value: Any, field_name: str, *, positive: bool = False) -> int:
    """Parse an integer without accepting booleans or fractional values."""

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


def _column_index(cell_reference: str) -> int:
    """Convert an Excel cell reference such as ``AA12`` to a zero-based column."""

    match = _CELL_REFERENCE.fullmatch(cell_reference)
    if match is None:
        raise ValueError(f"invalid XLSX cell reference: {cell_reference!r}")
    result = 0
    for character in match.group(1):
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    """Return the optional XLSX shared-string table."""

    path = "xl/sharedStrings.xml"
    if path not in archive.namelist():
        return []
    root = ET.fromstring(archive.read(path))
    strings: list[str] = []
    for item in root.findall(f"{{{_SPREADSHEET_NS}}}si"):
        strings.append(
            "".join(
                node.text or ""
                for node in item.iter(f"{{{_SPREADSHEET_NS}}}t")
            )
        )
    return strings


def _worksheet_path(archive: zipfile.ZipFile, sheet_name: str) -> str:
    """Resolve a worksheet name to its ZIP-member path through workbook rels."""

    workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
    relationship_id: str | None = None
    for sheet in workbook_root.findall(
        f".//{{{_SPREADSHEET_NS}}}sheet"
    ):
        if sheet.get("name") == sheet_name:
            relationship_id = sheet.get(f"{{{_OFFICE_REL_NS}}}id")
            break
    if relationship_id is None:
        raise DataValidationError(f"XLSX sheet is missing: {sheet_name!r}")

    relationships_root = ET.fromstring(
        archive.read("xl/_rels/workbook.xml.rels")
    )
    for relationship in relationships_root.findall(
        f"{{{_RELATIONSHIPS_NS}}}Relationship"
    ):
        if relationship.get("Id") != relationship_id:
            continue
        target = relationship.get("Target")
        if not target:
            break
        if target.startswith("/"):
            return target.lstrip("/")
        return posixpath.normpath(posixpath.join("xl", target))
    raise DataValidationError(
        f"XLSX relationship is missing for sheet {sheet_name!r}"
    )


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> Any:
    """Decode the small subset of XLSX cell types needed for tabular data."""

    cell_type = cell.get("t", "n")
    value_node = cell.find(f"{{{_SPREADSHEET_NS}}}v")
    if cell_type == "inlineStr":
        return "".join(
            node.text or ""
            for node in cell.iter(f"{{{_SPREADSHEET_NS}}}t")
        )
    if value_node is None or value_node.text is None:
        return None
    raw = value_node.text
    if cell_type in {"str", "e"}:
        return raw
    if cell_type == "s":
        try:
            return shared_strings[int(raw)]
        except (IndexError, ValueError) as error:
            raise DataValidationError("invalid XLSX shared-string reference") from error
    if cell_type == "b":
        return raw == "1"
    if cell_type == "n":
        try:
            number = float(raw)
        except ValueError as error:
            raise DataValidationError(f"invalid XLSX numeric cell: {raw!r}") from error
        return int(number) if number.is_integer() else number
    return raw


def read_xlsx_table(path: Path, sheet_name: str) -> list[dict[str, Any]]:
    """Read the used rows of one XLSX worksheet into header-keyed dictionaries."""

    if not path.is_file():
        raise DataValidationError(f"XLSX file is missing: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            shared_strings = _shared_strings(archive)
            worksheet_path = _worksheet_path(archive, sheet_name)
            root = ET.fromstring(archive.read(worksheet_path))
    except (KeyError, OSError, ET.ParseError, zipfile.BadZipFile) as error:
        raise DataValidationError(f"could not read XLSX file {path}: {error}") from error

    raw_rows: list[list[Any]] = []
    for row in root.findall(f".//{{{_SPREADSHEET_NS}}}row"):
        cells: dict[int, Any] = {}
        for cell in row.findall(f"{{{_SPREADSHEET_NS}}}c"):
            reference = cell.get("r")
            if reference is None:
                raise DataValidationError("XLSX cell is missing its reference")
            cells[_column_index(reference)] = _cell_value(cell, shared_strings)
        if cells:
            width = max(cells) + 1
            raw_rows.append([cells.get(index) for index in range(width)])

    if not raw_rows:
        raise DataValidationError(f"XLSX sheet contains no rows: {sheet_name!r}")
    headers = [str(value).strip() if value is not None else "" for value in raw_rows[0]]
    if any(not header for header in headers) or len(headers) != len(set(headers)):
        raise DataValidationError(f"XLSX sheet has invalid headers: {sheet_name!r}")

    records: list[dict[str, Any]] = []
    for raw_row in raw_rows[1:]:
        padded = raw_row + [None] * (len(headers) - len(raw_row))
        records.append(dict(zip(headers, padded[: len(headers)])))
    return records


def _require_columns(
    available: Iterable[str], required: set[str], label: str
) -> None:
    missing = required.difference(available)
    if missing:
        raise DataValidationError(f"{label}: missing required columns: {sorted(missing)}")


def load_job09(path: Path) -> LoadedDataset:
    """Load completed Job 09 CEBP minimum total-copy counts from CSV."""

    label = "Job 09"
    dataset = LoadedDataset(label=label)
    if not path.is_file():
        raise DataValidationError(f"{label}: CSV file is missing: {path}")
    required = {"n", "state_index", "task_id", "minimum_copies", "status"}
    seen: set[tuple[int, int]] = set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            _require_columns(reader.fieldnames or [], required, label)
            for row in reader:
                dataset.discovered_records += 1
                status = (row.get("status") or "").strip()
                if status != "completed":
                    dataset.skipped[f"status={status or 'missing'}"] += 1
                    continue
                try:
                    n = integer(row["n"], "n", positive=True)
                    state_index = integer(row["state_index"], "state_index")
                    task_id = integer(row["task_id"], "task_id")
                    minimum_copies = integer(
                        row["minimum_copies"], "minimum_copies", positive=True
                    )
                    if n not in EXPECTED_N_VALUES:
                        raise ValueError("unexpected n")
                    if not 0 <= state_index < EXPECTED_STATES_PER_N:
                        raise ValueError("unexpected state_index")
                    if task_id != (n - 1) * EXPECTED_STATES_PER_N + state_index:
                        raise ValueError("task_id does not match n and state_index")
                    identity = (n, state_index)
                    if identity in seen:
                        raise ValueError("duplicate state identity")
                except (KeyError, ValueError) as error:
                    dataset.skipped[str(error)] += 1
                    continue
                seen.add(identity)
                dataset.copies_by_n[n].append(minimum_copies)
                dataset.valid_records += 1
    except (OSError, csv.Error) as error:
        raise DataValidationError(f"{label}: could not read {path}: {error}") from error
    return dataset


def load_job10(path: Path) -> LoadedDataset:
    """Load validated Job 10 Local-Pauli minimum total-copy counts from XLSX."""

    label = "Job 10"
    dataset = LoadedDataset(label=label)
    rows = read_xlsx_table(path, "Minimum Copies")
    required = {
        "n",
        "state_index",
        "task_id",
        "minimum_M_per_setting",
        "measurement_settings_3_pow_n",
        "minimum_total_copies_N",
        "scientific_status",
    }
    _require_columns(rows[0].keys() if rows else [], required, label)
    seen: set[tuple[int, int]] = set()
    for row in rows:
        dataset.discovered_records += 1
        status = str(row.get("scientific_status") or "").strip()
        if status != "validated_threshold_found":
            dataset.skipped[f"status={status or 'missing'}"] += 1
            continue
        try:
            n = integer(row["n"], "n", positive=True)
            state_index = integer(row["state_index"], "state_index")
            task_id = integer(row["task_id"], "task_id")
            minimum_m = integer(
                row["minimum_M_per_setting"],
                "minimum_M_per_setting",
                positive=True,
            )
            settings = integer(
                row["measurement_settings_3_pow_n"],
                "measurement_settings_3_pow_n",
                positive=True,
            )
            total_copies = integer(
                row["minimum_total_copies_N"],
                "minimum_total_copies_N",
                positive=True,
            )
            if n not in EXPECTED_N_VALUES:
                raise ValueError("unexpected n")
            if not 0 <= state_index < EXPECTED_STATES_PER_N:
                raise ValueError("unexpected state_index")
            if task_id != (n - 1) * EXPECTED_STATES_PER_N + state_index:
                raise ValueError("task_id does not match n and state_index")
            if settings != 3**n:
                raise ValueError("measurement-settings count does not equal 3**n")
            if total_copies != minimum_m * settings:
                raise ValueError("total-copy accounting check failed")
            identity = (n, state_index)
            if identity in seen:
                raise ValueError("duplicate state identity")
        except (KeyError, ValueError) as error:
            dataset.skipped[str(error)] += 1
            continue
        seen.add(identity)
        dataset.copies_by_n[n].append(total_copies)
        dataset.valid_records += 1
    return dataset


def aggregate(dataset: LoadedDataset) -> list[Aggregate]:
    """Compute the arithmetic mean and SEM over valid states for every n."""

    points: list[Aggregate] = []
    for n in EXPECTED_N_VALUES:
        values = np.asarray(dataset.copies_by_n.get(n, []), dtype=float)
        if values.size == 0:
            raise DataValidationError(f"{dataset.label}: no valid rows found for n={n}")
        sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        log_values = np.log(values)
        log_mean = float(np.mean(log_values))
        log_sd = float(np.std(log_values, ddof=1)) if values.size > 1 else 0.0
        points.append(
            Aggregate(
                n=n,
                mean=float(np.mean(values)),
                sem=sd / math.sqrt(int(values.size)) if values.size > 1 else 0.0,
                sd=sd,
                geometric_mean=math.exp(log_mean),
                log_sd=log_sd,
                geometric_lower=math.exp(log_mean - log_sd),
                geometric_upper=math.exp(log_mean + log_sd),
                count=int(values.size),
            )
        )
    return points


def summarize(dataset: LoadedDataset, points: list[Aggregate]) -> None:
    """Print data completeness and the plotted means for reproducibility."""

    print(
        f"{dataset.label}: {dataset.valid_records}/{dataset.discovered_records} "
        "records included"
    )
    if dataset.skipped:
        reasons = ", ".join(
            f"{reason} ({count})" for reason, count in sorted(dataset.skipped.items())
        )
        print(f"{dataset.label}: skipped {sum(dataset.skipped.values())}: {reasons}")
    else:
        print(f"{dataset.label}: skipped 0")
    counts = ", ".join(f"n={point.n}:{point.count}" for point in points)
    print(f"{dataset.label}: state counts = {counts}")
    if dataset.label == "Job 09":
        print(
            f"{dataset.label}: n  geometric_mean_minimum_total_copies  "
            "log_SD  lower_1sigma  upper_1sigma"
        )
        for point in points:
            print(
                f"{dataset.label}: {point.n:<2d} {point.geometric_mean:>35.6f} "
                f"{point.log_sd:>10.6f} {point.geometric_lower:>14.6f} "
                f"{point.geometric_upper:>14.6f}"
            )
    else:
        print(f"{dataset.label}: n  mean_minimum_total_copies  SD  SEM")
        for point in points:
            print(
                f"{dataset.label}: {point.n:<2d} {point.mean:>26.6f} "
                f"{point.sd:>14.6f} {point.sem:>14.6f}"
            )


def plot_comparison(
    series: list[tuple[LoadedDataset, list[Aggregate]]], output_path: Path
) -> None:
    """Plot mean minimum total copies against n in the established 08/07 style."""

    styles: dict[str, dict[str, Any]] = {
        "Job 09": {
            "color": "#1f77b4",
            "marker": "o",
            "linestyle": "-",
            "legend_label": "CEBP tomography",
        },
        "Job 10": {
            "color": "#9467bd",
            "marker": "^",
            "linestyle": "--",
            "legend_label": "Local-Pauli tomography",
        },
    }

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for dataset, points in series:
        style = styles[dataset.label]
        x = np.asarray([point.n for point in points], dtype=float)
        if dataset.label == "Job 09":
            y = np.asarray([point.geometric_mean for point in points], dtype=float)
            lower = np.asarray([point.geometric_lower for point in points], dtype=float)
            upper = np.asarray([point.geometric_upper for point in points], dtype=float)
            ax.errorbar(
                x,
                y,
                yerr=np.vstack((y - lower, upper - y)),
                label=style["legend_label"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                markersize=6.0,
                linewidth=1.8,
                elinewidth=1.2,
                capsize=4.0,
                capthick=1.2,
            )
        else:
            y = np.asarray([point.mean for point in points], dtype=float)
            sem = np.asarray([point.sem for point in points], dtype=float)
            ax.plot(
                x,
                y,
                label=style["legend_label"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                markersize=6.0,
                linewidth=1.8,
            )
            ax.fill_between(
                x,
                np.maximum(y - sem, np.finfo(float).tiny),
                y + sem,
                color=style["color"],
                alpha=0.18,
                linewidth=0,
            )

    ax.set_yscale("log")
    ax.set_xticks(EXPECTED_N_VALUES)
    ax.set_xlim(0.7, 9.3)
    ax.set_title(
        r"Average minimum total copies vs. number of qubits "
        r"($D_{\mathrm{tr}} \leq 0.05$)"
    )
    ax.set_xlabel("Number of qubits, $n$")
    ax.set_ylabel("Average minimum total copies")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.45)
    handles, labels = ax.get_legend_handles_labels()
    handle_by_label = dict(zip(labels, handles))
    legend_labels = ["CEBP tomography", "Local-Pauli tomography"]
    ax.legend(
        [handle_by_label[label] for label in legend_labels],
        legend_labels,
        loc="best",
        frameon=True,
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Load, validate, aggregate, and plot the Job 09/10 comparison."""

    repository_root = Path(__file__).resolve().parents[1]
    job09 = load_job09(repository_root / "Data" / "09" / "minimum_copies_by_state.csv")
    job10 = load_job10(
        repository_root / "Data" / "10" / "minimum_copies_by_state.xlsx"
    )
    job09_points = aggregate(job09)
    job10_points = aggregate(job10)
    summarize(job09, job09_points)
    summarize(job10, job10_points)

    output_path = repository_root / "Figs" / f"{OUTPUT_STEM}.pdf"
    plot_comparison([(job09, job09_points), (job10, job10_points)], output_path)
    print(
        "Plot scales: x=linear, y=log; "
        "Job 09 center=geometric mean with log-space 1-sigma interval; "
        "Job 10 center=arithmetic mean with SEM band"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except DataValidationError as error:
        raise SystemExit(f"error: {error}") from error
