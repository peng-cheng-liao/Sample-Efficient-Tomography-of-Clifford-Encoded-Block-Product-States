#!/usr/bin/env python3
"""Run the controlled 6q adaptive fixed-error benchmark and full validation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Optimization.objective as objective_module  # noqa: E402
from Optimization import (  # noqa: E402
    OptimizationConfig,
    OptimizationMode,
    OptimizationObjective,
    ProgressiveSearchConfig,
    SearchSpace,
    optimize_cebp_parameters_progressive,
)
from tests.fixed_error_6q_validation import (  # noqa: E402
    run_fixed_error_6q_validation as state_runner,
)


N = 6
ERROR_TARGET = 0.05
SUCCESS_PROBABILITY_THRESHOLD = 0.85
FINAL_SEED_COUNT = 16
REQUIRED_SUCCESS_COUNT = 14
LEGACY_TOTAL_COPIES_API_ARGUMENT = 1
PARTITIONS = {1: (1, 1, 1, 1, 1, 1), 2: (2, 2, 2), 3: (3, 3)}
STATE_SEEDS = {1: 6_201_001, 2: 6_202_001, 3: 6_203_001}
SEARCH_SEED = 6_200_501
TUNING_SEEDS = tuple(range(6_210_001, 6_210_017))
HISTORICAL_HOLDOUT_SEEDS = tuple(range(6_220_001, 6_220_005))
SEED_FIDELITIES = (1, 2, 4, 8, 16)
CANDIDATE_PROGRESSION = (16, 32, 64, 128, 256)
OUTPUT_ROOT = ROOT / "tests" / "fixed_error_6q_adaptive_budget_validation"
PRIOR_ROOT = ROOT / "tests" / "fixed_error_6q_mean_success_validation"
PRIOR_PATHS = {d: PRIOR_ROOT / f"d{d}" / "result.json" for d in (1, 2, 3)}


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def production_paths() -> tuple[Path, ...]:
    return (ROOT / "main_v2.py", *sorted((ROOT / "Optimization").glob("*.py")))


def production_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): sha256(path)
        for path in production_paths()
    }


def prior_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): sha256(path)
        for path in PRIOR_PATHS.values()
    }


def load_prior(d: int) -> dict[str, Any]:
    payload = json.loads(PRIOR_PATHS[d].read_text())
    cfg = payload["configuration"]
    expected = {
        "n": N,
        "d": d,
        "partition_block_sizes": list(PARTITIONS[d]),
        "mode": OptimizationMode.FIXED_ERROR_MIN_COPIES.value,
        "error_target": ERROR_TARGET,
        "success_probability_threshold": SUCCESS_PROBABILITY_THRESHOLD,
        "required_error_success_count": REQUIRED_SUCCESS_COUNT,
        "state_generation_seed": STATE_SEEDS[d],
        "search_seed": SEARCH_SEED,
        "tuning_seeds": list(TUNING_SEEDS),
    }
    for key, expected_value in expected.items():
        if cfg.get(key) != expected_value:
            raise RuntimeError(
                f"Prior d={d} {key} mismatch: {cfg.get(key)!r} != {expected_value!r}"
            )
    progressive = cfg["progressive_search"]
    if progressive["seed_fidelities"] != list(SEED_FIDELITIES):
        raise RuntimeError(f"Prior d={d} seed fidelities changed.")
    if progressive["max_candidates"] != 256:
        raise RuntimeError(f"Prior d={d} maximum candidate count changed.")
    if progressive["relative_improvement_threshold"] != 0.0:
        raise RuntimeError(f"Prior d={d} improvement threshold changed.")
    return payload


def build_config(d: int, stage: str, *, resume: bool) -> OptimizationConfig:
    checkpoint = OUTPUT_ROOT / stage / f"d{d}.checkpoint.json"
    return OptimizationConfig(
        total_copies=LEGACY_TOTAL_COPIES_API_ARGUMENT,
        search_seed=SEARCH_SEED,
        tuning_seeds=TUNING_SEEDS,
        holdout_seeds=HISTORICAL_HOLDOUT_SEEDS,
        max_dense_qubits=N,
        max_enumeration_qubits=N,
        max_oracle_dense_qubits=N,
        simulation_backend="batched_counts",
        number_of_candidates=16,
        halving_seed_counts=SEED_FIDELITIES,
        retention_fraction=0.5,
        fixed_error_initial_budget_hint=None,
        fixed_error_budget_growth_factor=2.0,
        fixed_error_budget_relative_tolerance=0.01,
        fixed_error_hard_safety_budget=None,
        objective=OptimizationObjective(
            mode=OptimizationMode.FIXED_ERROR_MIN_COPIES,
            copy_ceiling=None,
            error_target=ERROR_TARGET,
            error_target_margin=0.0,
            success_probability_threshold=SUCCESS_PROBABILITY_THRESHOLD,
        ),
        checkpoint_path=str(checkpoint),
        resume_from_checkpoint=resume,
        checkpoint_every_n_evaluations=16,
        checkpoint_key=(
            f"6q-fixed-error-adaptive-{stage}-d{d}-state-{STATE_SEEDS[d]}"
        ),
    )


def build_progressive(prior: dict[str, Any], stage: str) -> ProgressiveSearchConfig:
    values = dict(prior["configuration"]["progressive_search"])
    values["max_candidates"] = 16 if stage == "benchmark" else 256
    return ProgressiveSearchConfig(**values)


def build_search_space(prior: dict[str, Any]) -> SearchSpace:
    return SearchSpace(
        **{
            key: tuple(float(item) for item in value)
            for key, value in prior["configuration"]["search_space"].items()
        }
    )


def state_identity(d: int, prior: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    instance = state_runner.build_instance(d)
    repeated = state_runner.build_instance(d)
    difference = float(np.max(np.abs(instance.state.full() - repeated.state.full())))
    digest = state_runner.state_digest(instance)
    ledger = json_safe(instance.seed_ledger)
    cfg = prior["configuration"]
    checks = {
        "state_generation_seed_matches": cfg["state_generation_seed"] == STATE_SEEDS[d],
        "seed_ledger_matches": ledger == cfg["state_seed_ledger"],
        "raw_dense_digest_matches": digest == prior["state_digest_sha256"],
        "repeated_raw_dense_digest_matches": (
            state_runner.state_digest(repeated) == digest
        ),
        "repeated_state_max_abs_difference": difference,
        "encoder_sampling_matches": (
            instance.oracle_truth.encoder_sampling == cfg["encoder_sampling"]
        ),
        "encoder_steps_matches": (
            int(instance.oracle_truth.encoder_steps) == int(cfg["encoder_steps"])
        ),
        "hidden_partition_matches": (
            [list(block) for block in instance.oracle_truth.hidden_partition]
            == cfg["exact_hidden_partition"]
        ),
        "current_state_digest_sha256": digest,
        "prior_state_digest_sha256": prior["state_digest_sha256"],
        "state_seed_ledger": ledger,
    }
    required_identity_checks = (
        "state_generation_seed_matches",
        "seed_ledger_matches",
        "repeated_raw_dense_digest_matches",
        "encoder_sampling_matches",
        "encoder_steps_matches",
        "hidden_partition_matches",
    )
    if not all(checks[key] is True for key in required_identity_checks) or difference > 1e-14:
        raise RuntimeError(f"d={d} exact prior target state could not be reproduced.")
    return instance, checks


def seed_payload(item: Any) -> dict[str, Any]:
    payload = state_runner.seed_payload(item)
    if payload["operational_success"]:
        error = payload["trace_distance"]
        if error is None or not math.isfinite(float(error)):
            raise RuntimeError("Operational seed produced a non-finite error.")
    return payload


def aggregate_payload(evaluation: Any) -> dict[str, Any] | None:
    if evaluation is None:
        return None
    errors = [
        float(item.trace_distance)
        for item in evaluation.seed_evaluations
        if item.trace_distance is not None
    ]
    return {
        "candidate_id": evaluation.candidate_id,
        "trial_budget": int(evaluation.N_candidate),
        "n_seeds_evaluated": int(evaluation.n_seeds_evaluated),
        "operational_validity": bool(evaluation.operationally_valid),
        "mean_error_feasible": evaluation.mean_error_feasible,
        "success_fraction_feasible": evaluation.success_fraction_feasible,
        "combined_final_feasibility": evaluation.final_target_feasible,
        "mean_trace_distance": evaluation.mean_trace_distance,
        "median_trace_distance": statistics.median(errors) if errors else None,
        "trace_distance_std": evaluation.trace_distance_std,
        "max_trace_distance": evaluation.max_trace_distance,
        "success_count": evaluation.error_success_count,
        "required_success_count": evaluation.required_error_success_count,
        "success_fraction": evaluation.error_success_fraction,
        "mean_realized_copies": evaluation.mean_realized_copies,
        "max_realized_copies": evaluation.max_realized_copies,
        "per_stage_mean_realized_copies": dict(evaluation.mean_stage_copies),
        "per_stage_max_realized_copies": dict(evaluation.max_stage_copies),
        "per_seed_trace_distances": errors,
        "per_seed": [seed_payload(item) for item in evaluation.seed_evaluations],
    }


def select_final_threshold(result: Any) -> Any:
    matches = [
        item
        for item in result.search_metadata.fixed_error_threshold_evaluations
        if item.structural_candidate_id == result.best_candidate_id
        and item.seed_fidelity == FINAL_SEED_COUNT
    ]
    if not matches:
        raise RuntimeError("No final-fidelity threshold metadata for selected candidate.")
    return matches[-1]


def validate_early_semantics(result: Any) -> dict[str, Any]:
    checked = 0
    success_gate_would_differ = 0
    for threshold in result.search_metadata.fixed_error_threshold_evaluations:
        if threshold.seed_fidelity >= FINAL_SEED_COUNT:
            continue
        for trial in threshold.budget_trials:
            expected = bool(trial.operationally_valid and trial.mean_error_feasible)
            if trial.feasible != expected:
                raise RuntimeError(
                    "Early-fidelity feasibility unexpectedly enforced another gate."
                )
            checked += 1
            if trial.success_fraction_feasible is False and trial.feasible:
                success_gate_would_differ += 1
    return {
        "verified": True,
        "trial_records_checked": checked,
        "records_demonstrating_success_gate_was_not_hard_enforced": (
            success_gate_would_differ
        ),
        "semantics": "operational AND mean_error_feasible only for S=1,2,4,8",
    }


def cost_payload(result: Any) -> dict[str, Any]:
    thresholds = result.search_metadata.fixed_error_threshold_evaluations
    diagnostics = result.search_metadata.fixed_error_computational_diagnostics
    if diagnostics is None:
        raise RuntimeError("Adaptive fixed-error computational diagnostics are absent.")
    budgets_by_candidate: dict[str, set[int]] = {}
    for threshold in thresholds:
        budgets_by_candidate.setdefault(threshold.structural_candidate_id, set()).update(
            int(item) for item in threshold.tested_budgets
        )
    trial_counts = [len(values) for values in budgets_by_candidate.values()]
    statuses = {}
    for threshold in thresholds:
        statuses.setdefault(threshold.structural_candidate_id, set()).add(
            threshold.search_status
        )
    final_thresholds = [item for item in thresholds if item.seed_fidelity == 16]
    estimates = sorted(
        {
            int(item.estimated_min_budget)
            for item in final_thresholds
            if item.threshold_found and item.estimated_min_budget is not None
        }
    )
    largest_tested = max(
        (int(budget) for item in thresholds for budget in item.tested_budgets),
        default=None,
    )
    return {
        "unique_structural_candidates": diagnostics.unique_structural_candidate_count,
        "unique_lambda_budget_trials": diagnostics.unique_structural_budget_trial_count,
        "unique_lambda_budget_seed_evaluations": (
            diagnostics.unique_structural_budget_seed_evaluation_count
        ),
        "semantic_cache_hits": diagnostics.semantic_cache_hit_count,
        "semantic_cache_reuse_count": diagnostics.semantic_cache_hit_count,
        "total_learner_calls_avoided_by_cache": diagnostics.semantic_cache_hit_count,
        "upward_expansion_trial_count": diagnostics.upward_expansion_trial_count,
        "relative_refinement_trial_count": diagnostics.relative_refinement_trial_count,
        "average_budget_trials_by_seed_fidelity": dict(
            diagnostics.average_budget_trials_by_seed_fidelity
        ),
        "average_unique_budget_trials_per_structural_candidate": (
            statistics.mean(trial_counts) if trial_counts else 0.0
        ),
        "median_unique_budget_trials_per_structural_candidate": (
            statistics.median(trial_counts) if trial_counts else 0.0
        ),
        "max_unique_budget_trials_for_any_structural_candidate": (
            max(trial_counts, default=0)
        ),
        "structural_candidates_with_threshold_at_any_fidelity": sum(
            "threshold_found" in values for values in statuses.values()
        ),
        "structural_candidates_stopped_only_by_safety_limit": sum(
            values == {"computational_safety_limit_reached"}
            for values in statuses.values()
        ),
        "final_fidelity_threshold_count": len(estimates),
        "final_fidelity_threshold_min": min(estimates, default=None),
        "final_fidelity_threshold_median": (
            statistics.median(estimates) if estimates else None
        ),
        "final_fidelity_threshold_max": max(estimates, default=None),
        "largest_tested_budget": largest_tested,
        "evaluation_attempt_count": result.evaluation_attempt_count,
        "actual_learner_run_count": result.actual_learner_run_count,
        "unique_cached_seed_evaluation_count": (
            result.unique_cached_seed_evaluation_count
        ),
        "preflight_rejection_count": result.preflight_rejection_count,
        "per_candidate_unique_budget_trial_counts": {
            key: len(value) for key, value in sorted(budgets_by_candidate.items())
        },
    }


def result_payload(
    *,
    d: int,
    stage: str,
    prior: dict[str, Any],
    state_checks: dict[str, Any],
    config: OptimizationConfig,
    progressive: ProgressiveSearchConfig,
    result: Any,
    runtime: float,
    tracked_seeds: list[int],
) -> dict[str, Any]:
    final_threshold = select_final_threshold(result)
    tuning = aggregate_payload(result.comparison_evaluation)
    cost = cost_payload(result)
    pool_sizes = [
        int(item.candidate_pool_size) for item in result.search_metadata.round_summaries
    ]
    expected_pools = [16] if stage == "benchmark" else list(CANDIDATE_PROGRESSION)
    if pool_sizes != expected_pools:
        raise RuntimeError(f"Unexpected {stage} pool progression: {pool_sizes!r}")
    early = validate_early_semantics(result)
    if result.holdout_evaluation is not None:
        raise RuntimeError("Fixed-error run returned a holdout evaluation.")
    holdout_calls = sorted(set(tracked_seeds) & set(HISTORICAL_HOLDOUT_SEEDS))
    if holdout_calls:
        raise RuntimeError(f"Fixed-error run called holdout seeds: {holdout_calls}")
    if final_threshold.threshold_found:
        if final_threshold.relative_bracket_width is None:
            raise RuntimeError("Threshold-found result lacks bracket width.")
        if final_threshold.relative_bracket_width > 0.01 + 1e-15:
            raise RuntimeError("Threshold bracket is wider than 1%.")
    if tuning is not None and tuning["n_seeds_evaluated"] != FINAL_SEED_COUNT:
        raise RuntimeError("Final comparison did not use 16 tuning seeds.")
    if tuning is not None:
        expected_feasible = bool(
            tuning["operational_validity"]
            and tuning["mean_error_feasible"]
            and tuning["success_fraction_feasible"]
        )
        if tuning["combined_final_feasibility"] is not expected_feasible:
            raise RuntimeError("Final combined feasibility rule is inconsistent.")
    selected_caps = dict(result.best_derived_candidate.fixed_budget_stage_caps)
    if d == 1 and tuning is not None:
        if selected_caps.get("grouping") != 0:
            raise RuntimeError("d=1 selected grouping allocation is nonzero.")
        for item in result.comparison_evaluation.seed_evaluations:
            state_runner.assert_d1_seed_accounting(
                item, result.best_derived_candidate.physical_copy_budget
            )
    old = prior["optimization"]["tuning"]
    old_budget = int(prior["optimization"]["refined_N_candidate"])
    estimated = final_threshold.estimated_min_budget
    comparison = {
        "previous_result_path": str(PRIOR_PATHS[d].relative_to(ROOT)),
        "previous_architecture": "sampled N plus post-selection refinement",
        "previous_N": old_budget,
        "previous_mean_trace_distance": old["mean_trace_distance"],
        "previous_success_count": old["error_success_count"],
        "previous_success_fraction": old["error_success_fraction"],
        "previous_feasible_under_mean_plus_14_of_16": old["final_target_feasible"],
        "previous_runtime_seconds": prior["timings_seconds"]["optimization"],
        "previous_actual_learner_run_count": prior["optimization"][
            "actual_learner_run_count"
        ],
        "absolute_N_difference": (
            None if estimated is None else int(estimated) - old_budget
        ),
        "percentage_N_difference": (
            None
            if estimated is None
            else (int(estimated) - old_budget) / old_budget * 100.0
        ),
    }
    thresholds = json_safe(result.search_metadata.fixed_error_threshold_evaluations)
    payload = {
        "schema_version": 1,
        "status": (
            "completed_threshold_found"
            if final_threshold.threshold_found
            else final_threshold.search_status
        ),
        "stage": stage,
        "configuration": {
            "n": N,
            "d": d,
            "partition": list(PARTITIONS[d]),
            "mode": OptimizationMode.FIXED_ERROR_MIN_COPIES.value,
            "error_target": ERROR_TARGET,
            "success_probability_threshold": SUCCESS_PROBABILITY_THRESHOLD,
            "final_tuning_seed_count": FINAL_SEED_COUNT,
            "required_success_count": REQUIRED_SUCCESS_COUNT,
            "state_generation_seed": STATE_SEEDS[d],
            "state_seed_ledger": state_checks["state_seed_ledger"],
            "state_generation": prior["configuration"]["state_generation"],
            "encoder_sampling": prior["configuration"]["encoder_sampling"],
            "encoder_steps": prior["configuration"]["encoder_steps"],
            "search_seed": SEARCH_SEED,
            "tuning_seeds": list(TUNING_SEEDS),
            "historical_holdout_seeds_configured_but_unused": list(
                HISTORICAL_HOLDOUT_SEEDS
            ),
            "scientific_copy_ceiling": None,
            "legacy_total_copies_api_argument_inert": LEGACY_TOTAL_COPIES_API_ARGUMENT,
            "initial_budget_hint": config.fixed_error_initial_budget_hint,
            "budget_growth_factor": config.fixed_error_budget_growth_factor,
            "budget_relative_tolerance": config.fixed_error_budget_relative_tolerance,
            "max_budget_expansion_rounds": (
                config.fixed_error_max_budget_expansion_rounds
            ),
            "hard_safety_budget": config.fixed_error_hard_safety_budget,
            "candidate_progression": expected_pools,
            "seed_fidelity_progression": list(SEED_FIDELITIES),
            "relative_improvement_threshold": (
                progressive.relative_improvement_threshold
            ),
            "search_space": json_safe(build_search_space(prior)),
        },
        "state_identity": state_checks,
        "outer_search": {
            "structural_candidates_evaluated": cost["unique_structural_candidates"],
            "candidate_pool_sizes_reached": pool_sizes,
            "final_pool_size": result.search_metadata.final_candidate_pool_size,
            "reached_256": result.search_metadata.final_candidate_pool_size == 256,
            "termination_reason": result.search_metadata.termination_reason,
            "rounds_completed": result.search_metadata.rounds_completed,
        },
        "selected_structural_candidate": {
            "candidate_id": result.best_candidate_id,
            "lambda": json_safe(result.best_candidate),
        },
        "threshold_result": {
            "search_status": final_threshold.search_status,
            "threshold_found": final_threshold.threshold_found,
            "estimated_min_budget": final_threshold.estimated_min_budget,
            "low_infeasible_budget": final_threshold.low_infeasible_budget,
            "high_feasible_budget": final_threshold.high_feasible_budget,
            "relative_bracket_width": final_threshold.relative_bracket_width,
            "practical_minimum_budget": final_threshold.practical_minimum_budget,
            "initial_budget": final_threshold.initial_budget,
            "tested_budgets": list(final_threshold.tested_budgets),
            "expansion_trials": final_threshold.expansion_trial_count,
            "refinement_trials": final_threshold.refinement_trial_count,
            "non_monotonic_observations": list(
                final_threshold.non_monotonic_observations
            ),
        },
        "final_16_seed_metrics": tuning,
        "physical_budget": {
            "final_trial_budget": (
                None if tuning is None else tuning["trial_budget"]
            ),
            "per_stage_allocated_copies": selected_caps,
            "per_stage_mean_realized_copies": (
                None if tuning is None else tuning["per_stage_mean_realized_copies"]
            ),
            "per_stage_max_realized_copies": (
                None if tuning is None else tuning["per_stage_max_realized_copies"]
            ),
        },
        "cost_diagnostics": cost,
        "early_fidelity_semantics": early,
        "holdout": {
            "status": "not_evaluated",
            "evaluation": None,
            "learner_seed_calls": holdout_calls,
            "metadata_holdout_seeds": list(result.search_metadata.holdout_seeds),
        },
        "old_vs_new": comparison,
        "runtime_seconds": runtime,
        "threshold_evaluations": thresholds,
    }
    if stage == "benchmark":
        scale = 256 / 16
        payload["projected_full_256_cost"] = {
            "method": "first-order linear scaling of observed 16-candidate work by 16",
            "scale_factor": scale,
            "projected_learner_evaluations": round(
                cost["unique_lambda_budget_seed_evaluations"] * scale
            ),
            "projected_unique_budget_trials": round(
                cost["unique_lambda_budget_trials"] * scale
            ),
            "projected_wall_clock_seconds": runtime * scale,
            "precision_note": "diagnostic estimate only; progressive reuse can change scaling",
        }
        malformed = bool(
            cost["unique_structural_candidates"] != 16
            or cost["unique_lambda_budget_trials"] <= 0
            or cost["unique_lambda_budget_seed_evaluations"] <= 0
        )
        all_safety = bool(
            cost["structural_candidates_stopped_only_by_safety_limit"] == 16
        )
        payload["benchmark_health"] = {
            "healthy": not malformed and not all_safety,
            "metadata_internally_valid": not malformed,
            "entire_candidate_set_hit_safety_limit": all_safety,
            "runaway_or_infinite_loop_observed": False,
            "decision": (
                "proceed_to_full_run"
                if not malformed and not all_safety
                else "full_run_skipped_due_to_benchmark_problem"
            ),
        }
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n")


def run_case(d: int, stage: str, *, resume: bool) -> int:
    output_path = OUTPUT_ROOT / stage / f"d{d}.json"
    checkpoint_path = OUTPUT_ROOT / stage / f"d{d}.checkpoint.json"
    if output_path.exists() and not resume:
        raise FileExistsError(f"Fresh run refuses to overwrite {output_path}")
    if resume and not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
    if not resume and checkpoint_path.exists():
        raise FileExistsError(f"Fresh run refuses existing checkpoint: {checkpoint_path}")
    prior = load_prior(d)
    instance, state_checks = state_identity(d, prior)
    config = build_config(d, stage, resume=resume)
    progressive = build_progressive(prior, stage)
    space = build_search_space(prior)
    hashes_before = production_hashes()
    old_hashes_before = prior_hashes()
    tracked_seeds: list[int] = []
    original = objective_module.evaluate_candidate_on_seed

    def tracked(*args, **kwargs):
        tracked_seeds.append(int(args[2]))
        return original(*args, **kwargs)

    print(f"[{stage} d={d}] start", flush=True)
    started = time.perf_counter()
    objective_module.evaluate_candidate_on_seed = tracked
    try:
        result = optimize_cebp_parameters_progressive(
            instance,
            LEGACY_TOTAL_COPIES_API_ARGUMENT,
            config,
            space,
            progressive_config=progressive,
        )
        runtime = time.perf_counter() - started
        payload = result_payload(
            d=d,
            stage=stage,
            prior=prior,
            state_checks=state_checks,
            config=config,
            progressive=progressive,
            result=result,
            runtime=runtime,
            tracked_seeds=tracked_seeds,
        )
        exit_code = 0
    except Exception as error:
        runtime = time.perf_counter() - started
        payload = {
            "schema_version": 1,
            "status": "runtime_failed",
            "stage": stage,
            "configuration": {"n": N, "d": d, "partition": list(PARTITIONS[d])},
            "state_identity": state_checks,
            "runtime_seconds": runtime,
            "failure": {
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
            },
        }
        exit_code = 3
    finally:
        objective_module.evaluate_candidate_on_seed = original
    hashes_after = production_hashes()
    old_hashes_after = prior_hashes()
    payload["production_hashes_before"] = hashes_before
    payload["production_hashes_after"] = hashes_after
    payload["production_files_unchanged"] = hashes_before == hashes_after
    payload["prior_result_hashes_before"] = old_hashes_before
    payload["prior_result_hashes_after"] = old_hashes_after
    payload["prior_results_unchanged"] = old_hashes_before == old_hashes_after
    payload["tracked_tuning_learner_call_count"] = len(tracked_seeds)
    write_json(output_path, payload)
    if exit_code == 0:
        threshold = payload["threshold_result"]
        cost = payload["cost_diagnostics"]
        print(
            f"[{stage} d={d}] complete in {runtime:.3f}s; "
            f"threshold={threshold['estimated_min_budget']}; "
            f"budget_trials={cost['unique_lambda_budget_trials']}; "
            f"learner_evals={cost['unique_lambda_budget_seed_evaluations']}",
            flush=True,
        )
        if stage == "benchmark":
            projection = payload["projected_full_256_cost"]
            print(
                f"[benchmark d={d}] projected full: "
                f"{projection['projected_wall_clock_seconds']:.1f}s, "
                f"{projection['projected_unique_budget_trials']} trials, "
                f"{projection['projected_learner_evaluations']} learner evals; "
                f"decision={payload['benchmark_health']['decision']}",
                flush=True,
            )
    else:
        print(
            f"[{stage} d={d}] failed in {runtime:.3f}s: "
            f"{payload['failure']['exception_type']}: "
            f"{payload['failure']['exception_message']}",
            flush=True,
        )
    if exit_code == 0:
        checkpoint_path.unlink(missing_ok=True)
    return exit_code


def build_comparison() -> None:
    cases = {}
    for d in (1, 2, 3):
        benchmark_path = OUTPUT_ROOT / "benchmark" / f"d{d}.json"
        full_path = OUTPUT_ROOT / "full" / f"d{d}.json"
        cases[f"d{d}"] = {
            "benchmark": json.loads(benchmark_path.read_text())
            if benchmark_path.exists()
            else None,
            "full": json.loads(full_path.read_text()) if full_path.exists() else None,
        }
    write_json(
        OUTPUT_ROOT / "comparison.json",
        {
            "schema_version": 1,
            "description": "Controlled old-vs-adaptive 6q fixed-error validation",
            "cases": cases,
        },
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("benchmark", "full"))
    parser.add_argument("--case", choices=("1", "2", "3", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--build-comparison", action="store_true")
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    if args.build_comparison and args.stage is None:
        build_comparison()
        return 0
    if args.stage is None:
        parser.error("--stage is required unless only --build-comparison is requested")
    cases = (1, 2, 3) if args.case == "all" else (int(args.case),)
    codes = []
    for d in cases:
        try:
            if args.stage == "full":
                benchmark_path = OUTPUT_ROOT / "benchmark" / f"d{d}.json"
                if not benchmark_path.exists():
                    raise RuntimeError(f"Missing benchmark health result: {benchmark_path}")
                benchmark = json.loads(benchmark_path.read_text())
                if not benchmark.get("benchmark_health", {}).get("healthy", False):
                    write_json(
                        OUTPUT_ROOT / "full" / f"d{d}.json",
                        {
                            "schema_version": 1,
                            "status": "full_run_skipped_due_to_benchmark_problem",
                            "stage": "full",
                            "configuration": {
                                "n": N,
                                "d": d,
                                "partition": list(PARTITIONS[d]),
                            },
                            "benchmark_path": str(benchmark_path.relative_to(ROOT)),
                            "benchmark_health": benchmark.get("benchmark_health"),
                        },
                    )
                    print(f"[full d={d}] skipped by benchmark health gate", flush=True)
                    codes.append(2)
                    continue
            codes.append(run_case(d, args.stage, resume=args.resume))
        except Exception as error:
            print(
                f"[{args.stage} d={d}] setup failed: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            codes.append(4)
    if args.build_comparison:
        build_comparison()
    return max(codes, default=0)


if __name__ == "__main__":
    raise SystemExit(main())
