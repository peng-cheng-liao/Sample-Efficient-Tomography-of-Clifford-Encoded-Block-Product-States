#!/usr/bin/env python3
"""Analyze frozen-candidate replay outputs and generate compact plots/JSON."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "tests" / "fixed_error_d1_bimodality" / "outputs"
DEFAULT_PLOT_ROOT = ROOT / "tests" / "fixed_error_d1_bimodality" / "plots"
THRESHOLD = 0.05
BOOTSTRAP_REPETITIONS = 5_000
BOOTSTRAP_SEED = 6_250_001


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], name: str) -> float | None:
    value = row.get(name, "")
    if value in ("", "None", "null"):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {name} in row seed={row.get('seed')}")
    return result


def i(row: dict[str, str], name: str) -> int | None:
    value = f(row, name)
    return None if value is None else int(value)


def j(row: dict[str, str], name: str) -> Any:
    value = row.get(name, "")
    return None if value == "" else json.loads(value)


def distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(tuple(values), dtype=float)
    if not array.size or not np.all(np.isfinite(array)):
        raise ValueError("Distribution input must be nonempty and finite.")
    quantiles = np.percentile(array, [10, 25, 50, 75, 90])
    sorted_values = np.sort(array)
    gaps = np.diff(sorted_values)
    gap_index = int(np.argmax(gaps)) if gaps.size else 0
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "standard_deviation_population": float(np.std(array, ddof=0)),
        "standard_deviation_sample": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "standard_error": float(np.std(array, ddof=1) / math.sqrt(array.size))
        if array.size > 1
        else 0.0,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "percentile_10": float(quantiles[0]),
        "quartile_25": float(quantiles[1]),
        "median_50": float(quantiles[2]),
        "quartile_75": float(quantiles[3]),
        "percentile_90": float(quantiles[4]),
        "fraction_D_le_0_05": float(np.mean(array <= 0.05)),
        "fraction_D_lt_0_03": float(np.mean(array < 0.03)),
        "fraction_D_gt_0_07": float(np.mean(array > 0.07)),
        "largest_adjacent_sorted_gap": float(gaps[gap_index]) if gaps.size else 0.0,
        "largest_gap_lower_endpoint": float(sorted_values[gap_index]) if gaps.size else None,
        "largest_gap_upper_endpoint": float(sorted_values[gap_index + 1]) if gaps.size else None,
        "middle_band_count": int(np.sum((array >= 0.03) & (array <= 0.07))),
        "empirical_cdf": [
            {"trace_distance": float(value), "cdf": float((index + 1) / array.size)}
            for index, value in enumerate(sorted_values)
        ],
    }


def correlation(x: Iterable[float], y: Iterable[float]) -> float | None:
    xa = np.asarray(tuple(x), dtype=float)
    ya = np.asarray(tuple(y), dtype=float)
    if xa.size < 2 or np.std(xa) == 0.0 or np.std(ya) == 0.0:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def group_numeric(rows: list[dict[str, str]], fields: Iterable[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in fields:
        values = [f(row, field) for row in rows]
        finite = [value for value in values if value is not None]
        output[field] = None if not finite else {
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "minimum": float(np.min(finite)),
            "maximum": float(np.max(finite)),
        }
    return output


def stage_comparison(rows: list[dict[str, str]]) -> dict[str, Any]:
    numeric_fields = (
        "peeled_count",
        "peeling_rank",
        "recovered_rank",
        "recovered_sector_count",
        "recovery_ranked_survivor_count",
        "syndrome_copies",
        "X_accepted",
        "Y_accepted",
        "Z_accepted",
        "minimum_accepted",
        "mean_accepted",
        "acceptance_imbalance",
        "X_acceptance_fraction",
        "Y_acceptance_fraction",
        "Z_acceptance_fraction",
        "max_projection_metric",
        "realized_total_copies",
        "copy_utilization",
    )
    groups = {
        name: [row for row in rows if row["error_class"] == name]
        for name in ("low", "middle", "high")
    }
    payload: dict[str, Any] = {
        "group_counts": {name: len(group) for name, group in groups.items()},
        "numeric": {
            name: group_numeric(group, numeric_fields) for name, group in groups.items()
        },
    }
    for name, group in groups.items():
        payload.setdefault("categorical", {})[name] = {
            "recovered_rank_counts": dict(Counter(row["recovered_rank"] for row in group)),
            "recovered_generator_signature_counts": dict(
                Counter(row["recovered_generators"] for row in group)
            ),
            "syndrome_pattern_counts": dict(Counter(row["syndrome_bits"] for row in group)),
            "projection_event_count": sum(
                row["projection_or_clipping"] == "True" for row in group
            ),
        }
    ranks = sorted({i(row, "recovered_rank") for row in rows if i(row, "recovered_rank") is not None})
    payload["high_error_by_recovered_rank"] = {
        str(rank): {
            "count": len(subset := [row for row in rows if i(row, "recovered_rank") == rank]),
            "high_count": sum(row["error_class"] == "high" for row in subset),
            "high_fraction": float(np.mean([row["error_class"] == "high" for row in subset])),
            "mean_error": float(np.mean([f(row, "trace_distance") for row in subset])),
        }
        for rank in ranks
    }
    return payload


def acceptance_analysis(rows: list[dict[str, str]]) -> dict[str, Any]:
    errors = [f(row, "trace_distance") for row in rows]
    fields = (
        "minimum_accepted",
        "mean_accepted",
        "acceptance_imbalance",
        "X_acceptance_fraction",
        "Y_acceptance_fraction",
        "Z_acceptance_fraction",
    )
    correlations = {}
    for field in fields:
        pairs = [
            (f(row, field), f(row, "trace_distance"))
            for row in rows
            if f(row, field) is not None
        ]
        correlations[field] = correlation(
            (pair[0] for pair in pairs), (pair[1] for pair in pairs)
        )
    return {
        "correlations_with_trace_distance": correlations,
        "all_axes_fully_accepted": all(
            i(row, "X_accepted") == i(row, "X_attempts")
            and i(row, "Y_accepted") == i(row, "Y_attempts")
            and i(row, "Z_accepted") == i(row, "Z_attempts")
            for row in rows
        ),
        "error_variance": float(np.var(errors)),
        "minimum_accepted_unique_values": sorted(
            {i(row, "minimum_accepted") for row in rows if i(row, "minimum_accepted") is not None}
        ),
    }


def budget_analysis(rows: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_ratio: dict[float, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_ratio[float(row["budget_ratio"])].append(row)
    summary = {}
    for ratio in sorted(by_ratio):
        group = by_ratio[ratio]
        stats = distribution(f(row, "trace_distance") for row in group)
        accepted = [f(row, "mean_accepted") for row in group if f(row, "mean_accepted") is not None]
        mins = [f(row, "minimum_accepted") for row in group if f(row, "minimum_accepted") is not None]
        summary[str(ratio)] = {
            "budget": i(group[0], "budget"),
            "mean_error": stats["mean"],
            "median_error": stats["median"],
            "percentile_90": stats["percentile_90"],
            "fraction_D_le_0_05": stats["fraction_D_le_0_05"],
            "fraction_D_gt_0_07": stats["fraction_D_gt_0_07"],
            "mean_tomography_accepted": float(np.mean(accepted)) if accepted else None,
            "minimum_tomography_accepted": float(np.min(mins)) if mins else None,
            "mean_realized_copy_utilization": float(
                np.mean([f(row, "copy_utilization") for row in group])
            ),
            "recovered_rank_counts": dict(Counter(row["recovered_rank"] for row in group)),
        }
    seed_paths: dict[int, dict[float, float]] = defaultdict(dict)
    for row in rows:
        seed_paths[int(row["seed"])][float(row["budget_ratio"])] = f(row, "trace_distance")
    ratios = sorted(by_ratio)
    transitions = []
    persistently_high = []
    for seed, path in sorted(seed_paths.items()):
        values = [path[ratio] for ratio in ratios]
        high_flags = [value > 0.07 for value in values]
        transition_ratio = next(
            (ratio for ratio, value in zip(ratios, values) if value <= 0.07), None
        )
        if all(high_flags):
            persistently_high.append(seed)
        transitions.append(
            {
                "seed": seed,
                "errors": {str(ratio): path[ratio] for ratio in ratios},
                "persistently_high": all(high_flags),
                "first_ratio_not_high": transition_ratio,
                "paired_differences_from_N_star": {
                    str(ratio): float(path[ratio] - path[1.0]) for ratio in ratios
                },
            }
        )
    paired_summary = {
        "ratios": ratios,
        "seed_count": len(seed_paths),
        "persistently_high_seed_count": len(persistently_high),
        "persistently_high_seeds": persistently_high,
        "transition_count_by_first_ratio_not_high": dict(
            Counter(str(item["first_ratio_not_high"]) for item in transitions)
        ),
        "mean_paired_difference_from_N_star": {
            str(ratio): float(
                np.mean([item["paired_differences_from_N_star"][str(ratio)] for item in transitions])
            )
            for ratio in ratios
        },
        "per_seed": transitions,
    }
    return summary, paired_summary


def reliability_analysis(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(tuple(values), dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    reference_mean = float(np.mean(array))
    reference_feasible = reference_mean <= THRESHOLD
    output: dict[str, Any] = {
        "reference_pool_count": int(array.size),
        "reference_pool_mean": reference_mean,
        "reference_classified_feasible": reference_feasible,
        "repetitions": BOOTSTRAP_REPETITIONS,
        "sampling": "deterministic simple random subsampling without replacement",
        "seed": BOOTSTRAP_SEED,
        "sample_sizes": {},
    }
    for size in (4, 8, 16, 32):
        if size > array.size:
            continue
        means = np.asarray(
            [np.mean(array[rng.choice(array.size, size=size, replace=False)]) for _ in range(BOOTSTRAP_REPETITIONS)],
            dtype=float,
        )
        feasible_probability = float(np.mean(means <= THRESHOLD))
        output["sample_sizes"][str(size)] = {
            "standard_deviation_of_estimated_mean": float(np.std(means, ddof=1)),
            "percentile_5": float(np.percentile(means, 5)),
            "median_50": float(np.percentile(means, 50)),
            "percentile_95": float(np.percentile(means, 95)),
            "probability_classified_feasible": feasible_probability,
            "false_feasible_rate": (
                feasible_probability if not reference_feasible else 0.0
            ),
            "false_infeasible_rate": (
                1.0 - feasible_probability if reference_feasible else 0.0
            ),
            "sample_means_for_plot": means.tolist() if size == 16 else None,
        }
    return output


def alternative_summaries(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(tuple(values), dtype=float)
    mean = float(np.mean(array))
    se = float(np.std(array, ddof=1) / math.sqrt(array.size))
    return {
        "mean": mean,
        "mean_plus_standard_error": mean + se,
        "mean_plus_1_96_standard_error": mean + 1.96 * se,
        "median": float(np.median(array)),
        "percentile_75": float(np.percentile(array, 75)),
        "percentile_90": float(np.percentile(array, 90)),
        "fraction_D_le_0_05": float(np.mean(array <= 0.05)),
        "fraction_D_gt_0_07": float(np.mean(array > 0.07)),
        "threshold": THRESHOLD,
    }


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_reconstructed_initial_status(path: Path) -> None:
    """Persist the pre-task status from the captured baseline plus path filtering.

    The task began with a direct ``git status --short``.  Since every task file
    is under one new test directory or the three named report artifacts, the
    exact initial snapshot is reconstructable by excluding only those paths.
    """

    completed = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    excluded_prefixes = (
        "?? tests/fixed_error_d1_bimodality/",
        "?? Reports/fixed_error_d1_bimodality_investigation_report.md",
        "?? Reports/fixed_error_d1_bimodality_investigation_task.diff",
        "?? Reports/fixed_error_d1_bimodality_investigation_verification.zip",
    )
    lines = [
        line
        for line in completed.stdout.splitlines()
        if not line.startswith(excluded_prefixes)
        and line != "?? Optimization/__pycache__/"
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_plots(
    replay: list[dict[str, str]],
    budget_rows: list[dict[str, str]],
    reliability: dict[str, Any],
    plot_root: Path,
) -> None:
    plot_root.mkdir(parents=True, exist_ok=True)
    errors = np.asarray([f(row, "trace_distance") for row in replay])
    plt.figure(figsize=(7, 4.5))
    plt.hist(errors, bins=20, color="#4472C4", edgecolor="white")
    plt.axvline(THRESHOLD, color="#C00000", linestyle="--", label="D = 0.05")
    plt.xlabel("Trace distance D")
    plt.ylabel("Seed count")
    plt.title("Frozen d=1 winner: fresh-seed error distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_root / "d1_error_histogram.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    ordered = np.sort(errors)
    plt.plot(np.arange(1, ordered.size + 1), ordered, marker=".", linewidth=1)
    plt.axhspan(0.03, 0.07, color="#ED7D31", alpha=0.12, label="middle band")
    plt.axhline(THRESHOLD, color="#C00000", linestyle="--", label="D = 0.05")
    plt.xlabel("Fresh seed (sorted by error)")
    plt.ylabel("Trace distance D")
    plt.title("Sorted d=1 replay errors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_root / "d1_sorted_error.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.scatter(
        [f(row, "minimum_accepted") for row in replay],
        errors,
        c=[{"low": "#70AD47", "middle": "#ED7D31", "high": "#C00000"}[row["error_class"]] for row in replay],
        alpha=0.8,
    )
    plt.xlabel("Minimum accepted conditional-tomography samples across X/Y/Z")
    plt.ylabel("Trace distance D")
    plt.title("d=1 error vs postselection acceptance")
    plt.tight_layout()
    plt.savefig(plot_root / "d1_error_vs_min_accepted.png", dpi=180)
    plt.close()

    by_seed: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in budget_rows:
        by_seed[int(row["seed"])].append(
            (float(row["budget_ratio"]), f(row, "trace_distance"))
        )
    plt.figure(figsize=(7.5, 5))
    for values in by_seed.values():
        values.sort()
        plt.plot(
            [value[0] for value in values],
            [value[1] for value in values],
            color="#5B9BD5",
            alpha=0.13,
            linewidth=0.7,
        )
    plt.axhline(0.07, color="#C00000", linestyle="--", linewidth=1)
    plt.xlabel("Budget / N*")
    plt.ylabel("Trace distance D")
    plt.title("Common-seed paired d=1 errors across budgets")
    plt.tight_layout()
    plt.savefig(plot_root / "d1_error_vs_budget.png", dpi=180)
    plt.close()

    ratios = sorted({float(row["budget_ratio"]) for row in budget_rows})
    groups = [[row for row in budget_rows if float(row["budget_ratio"]) == ratio] for ratio in ratios]
    high_fractions = [np.mean([f(row, "trace_distance") > 0.07 for row in group]) for group in groups]
    plt.figure(figsize=(7, 4.5))
    plt.plot(ratios, high_fractions, marker="o")
    plt.ylim(-0.02, 1.02)
    plt.xlabel("Budget / N*")
    plt.ylabel("Fraction with D > 0.07")
    plt.title("High-error mode frequency vs physical budget")
    plt.tight_layout()
    plt.savefig(plot_root / "d1_high_error_fraction_vs_budget.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    means = [np.mean([f(row, "trace_distance") for row in group]) for group in groups]
    medians = [np.median([f(row, "trace_distance") for row in group]) for group in groups]
    plt.plot(ratios, means, marker="o", label="mean")
    plt.plot(ratios, medians, marker="s", label="median")
    plt.axhline(THRESHOLD, color="#C00000", linestyle="--", label="D = 0.05")
    plt.xlabel("Budget / N*")
    plt.ylabel("Trace distance summary")
    plt.title("d=1 mean/median error vs budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_root / "d1_mean_median_error_vs_budget.png", dpi=180)
    plt.close()

    if "16" in reliability["sample_sizes"]:
        sample_means = reliability["sample_sizes"]["16"]["sample_means_for_plot"]
        plt.figure(figsize=(7, 4.5))
        plt.hist(sample_means, bins=30, color="#A5A5A5", edgecolor="white")
        plt.axvline(THRESHOLD, color="#C00000", linestyle="--", label="feasibility threshold")
        plt.axvline(float(np.mean(errors)), color="#4472C4", label="full replay mean")
        plt.xlabel("Mean D from 16-seed subsample")
        plt.ylabel("Replicate count")
        plt.title("Uncertainty of the 16-seed mean criterion")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_root / "d1_seed_mean_sampling_distribution.png", dpi=180)
        plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plot-root", type=Path, default=DEFAULT_PLOT_ROOT)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    plot_root = args.plot_root.resolve()
    replay = read_csv(output_root / "d1_replay_per_seed.csv")
    budget_rows = read_csv(output_root / "d1_budget_sensitivity.csv")
    d2_rows = read_csv(output_root / "d2_control_per_seed.csv")
    write_reconstructed_initial_status(output_root / "initial_git_status_short.txt")
    replay_values = [f(row, "trace_distance") for row in replay]
    d2_values = [f(row, "trace_distance") for row in d2_rows]
    replay_summary = {
        "distribution": distribution(replay_values),
        "stage_group_comparison": stage_comparison(replay),
        "acceptance_analysis": acceptance_analysis(replay),
        "alternative_feasibility_summaries": alternative_summaries(replay_values),
    }
    budget_summary, paired_summary = budget_analysis(budget_rows)
    reliability = reliability_analysis(replay_values)
    d2_summary = {
        "distribution": distribution(d2_values),
        "stage_copy_means": {
            field: float(np.mean([f(row, field) for row in d2_rows]))
            for field in (
                "peeling_copies",
                "recovery_copies",
                "grouping_copies",
                "syndrome_copies",
                "tomography_copies",
                "realized_total_copies",
                "copy_utilization",
            )
        },
        "recovered_rank_counts": dict(Counter(row["recovered_rank"] for row in d2_rows)),
    }
    write_json(output_root / "d1_replay_summary.json", replay_summary)
    write_json(output_root / "d1_budget_sensitivity_summary.json", budget_summary)
    write_json(output_root / "d1_paired_budget_analysis.json", paired_summary)
    write_json(output_root / "d1_seed_count_reliability.json", reliability)
    write_json(output_root / "d2_control_summary.json", d2_summary)
    save_plots(replay, budget_rows, reliability, plot_root)
    # Keep machine-readable output compact; plot-only bootstrap samples are discarded.
    reliability_compact = json.loads(json.dumps(reliability))
    for record in reliability_compact["sample_sizes"].values():
        record.pop("sample_means_for_plot", None)
    write_json(output_root / "d1_seed_count_reliability.json", reliability_compact)
    print(
        "analysis complete: "
        f"d1 mean={replay_summary['distribution']['mean']:.6f}, "
        f"median={replay_summary['distribution']['median']:.6f}, "
        f"high_fraction={replay_summary['distribution']['fraction_D_gt_0_07']:.3f}; "
        f"d2 mean={d2_summary['distribution']['mean']:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
