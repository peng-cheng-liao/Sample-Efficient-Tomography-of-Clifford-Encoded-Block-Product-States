#!/usr/bin/env python3
"""Strict post-run verification for the 6q joint fixed-error validation."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.fixed_error_6q_validation import (  # noqa: E402
    run_fixed_error_6q_validation as prior_runner,
)


NEW_ROOT = ROOT / "tests" / "fixed_error_6q_mean_success_validation"
OLD_ROOT = ROOT / "tests" / "fixed_error_6q_validation"
PARTITIONS = {1: [1, 1, 1, 1, 1, 1], 2: [2, 2, 2], 3: [3, 3]}


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def verify_case(d: int) -> dict[str, object]:
    new_path = NEW_ROOT / f"d{d}" / "result.json"
    old_path = OLD_ROOT / f"d{d}" / "result.json"
    new = json.loads(new_path.read_text())
    old = json.loads(old_path.read_text())
    if new["status"] not in ("completed_feasible", "completed_infeasible"):
        raise AssertionError(f"d={d} did not complete: {new['status']}")
    cfg = new["configuration"]
    old_cfg = old["configuration"]
    assert cfg["n"] == 6 and cfg["d"] == d
    assert cfg["partition_block_sizes"] == PARTITIONS[d]
    assert cfg["partition_block_sizes"] == old_cfg["partition_block_sizes"]
    assert cfg["state_generation_seed"] == old_cfg["state_generation_seed"]
    assert cfg["state_seed_ledger"] == old_cfg["state_seed_ledger"]
    assert cfg["search_seed"] == old_cfg["search_seed"]
    assert cfg["tuning_seeds"] == old_cfg["tuning_seeds"]
    assert cfg["search_space"] == old_cfg["search_space"]
    assert cfg["progressive_search"] == old_cfg["progressive_search"]
    assert cfg["budget_refinement"] == old_cfg["budget_refinement"]
    assert cfg["error_target"] == 0.05
    assert cfg["success_probability_threshold"] == 0.85
    assert cfg["final_tuning_seed_count"] == 16
    assert cfg["required_error_success_count"] == 14
    assert cfg["progressive_search"]["seed_fidelities"] == [1, 2, 4, 8, 16]
    assert cfg["progressive_search"]["max_candidates"] == 256
    assert cfg["progressive_search"]["relative_improvement_threshold"] == 0.0
    reconstruction = new["state_reconstruction"]
    assert reconstruction["numerically_reproducible"]
    assert reconstruction["seed_ledger_matches_historical_and_repeat"]
    assert reconstruction["encoder_configuration_matches_historical_and_repeat"]
    assert reconstruction["repeated_state_max_abs_difference"] <= 1e-14
    assert new["preflight"]["status"] == "passed"

    optimization = new["optimization"]
    tuning = optimization["tuning"]
    assert optimization["candidate_pool_sizes_reached"] == [16, 32, 64, 128, 256]
    assert optimization["final_candidate_pool_size"] == 256
    assert optimization["reached_256_candidates"]
    assert optimization["final_comparison_seed_count"] == 16
    assert optimization["best_candidate_resolvable_in_catalog"]
    assert optimization["termination_reason"] == "max_candidates"
    assert optimization["holdout"]["status"] == "not_evaluated"
    assert optimization["holdout"]["evaluation"] is None
    assert optimization["holdout"]["learner_seed_calls"] == []
    assert not optimization["holdout"]["holdout_post_selection_only"]

    errors = np.asarray(tuning["per_seed_trace_distances"], dtype=float)
    assert errors.shape == (16,)
    assert np.all(np.isfinite(errors))
    success_count = int(np.count_nonzero(errors <= 0.05 + 1e-12))
    mean_ok = bool(float(np.mean(errors)) <= 0.05 + 1e-12)
    success_ok = success_count >= 14
    operational = bool(tuning["operationally_valid"])
    combined = operational and mean_ok and success_ok
    assert tuning["error_success_count"] == success_count
    assert tuning["required_error_success_count"] == 14
    assert close(tuning["error_success_fraction"], success_count / 16)
    assert close(tuning["mean_trace_distance"], float(np.mean(errors)))
    assert close(tuning["median_trace_distance"], float(np.median(errors)))
    assert close(tuning["max_trace_distance"], float(np.max(errors)))
    assert close(tuning["trace_distance_std"], float(np.std(errors)))
    assert tuning["mean_error_feasible"] is mean_ok
    assert tuning["success_fraction_feasible"] is success_ok
    assert tuning["final_target_feasible"] is combined
    assert (new["status"] == "completed_feasible") is combined
    assert tuning["operational_success_rate"] == 1.0
    assert tuning["budget_feasible_rate"] == 1.0
    assert all(item["operational_success"] for item in tuning["per_seed"])
    assert all(item["estimator_available"] for item in tuning["per_seed"])
    assert all(item["budget_feasible"] for item in tuning["per_seed"])
    assert all(
        item["realized_total"] <= tuning["N_candidate"]
        for item in tuning["per_seed"]
    )
    if d == 1:
        assert optimization["d1_grouping"] == {
            "allocated_copies": 0,
            "max_realized_copies": 0,
        }
        assert tuning["max_stage_copies"]["grouping_ordinary_pool"] == 0

    assert new["production_files_unchanged"]
    assert new["prior_results_unchanged"]
    assert new["all_prior_result_hashes_before"] == new["all_prior_result_hashes_after"]
    assert new["prior_result_sha256_before"] == prior_runner.sha256(old_path)
    assert not Path(cfg["checkpoint_path"]).exists()
    return {
        "d": d,
        "status": new["status"],
        "N_candidate": tuning["N_candidate"],
        "mean_trace_distance": tuning["mean_trace_distance"],
        "error_success_count": success_count,
        "error_success_fraction": success_count / 16,
        "final_target_feasible": combined,
        "reached_256_candidates": True,
        "optimization_seconds": new["timings_seconds"]["optimization"],
    }


def main() -> int:
    summaries = [verify_case(d) for d in (1, 2, 3)]
    current_production_hashes = prior_runner.production_hashes()
    for d in (1, 2, 3):
        payload = json.loads((NEW_ROOT / f"d{d}" / "result.json").read_text())
        assert payload["production_hashes_after"] == current_production_hashes
    print(json.dumps(summaries, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
