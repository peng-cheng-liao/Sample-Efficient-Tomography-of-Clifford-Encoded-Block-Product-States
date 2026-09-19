#!/usr/bin/env python3
"""Re-run the exact prior 6q fixed-error cases under joint final feasibility."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
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
    run_fixed_error_6q_validation as prior_runner,
)


N = 6
ERROR_TARGET = 0.05
COPY_CEILING = 10_000_000
SUCCESS_PROBABILITY_THRESHOLD = 0.85
FINAL_TUNING_SEED_COUNT = 16
REQUIRED_SUCCESS_COUNT = 14
PARTITIONS = {1: (1, 1, 1, 1, 1, 1), 2: (2, 2, 2), 3: (3, 3)}
PRIOR_ROOT = ROOT / "tests" / "fixed_error_6q_validation"
OUTPUT_ROOT = ROOT / "tests" / "fixed_error_6q_mean_success_validation"
PRIOR_RESULT_PATHS = {
    d: PRIOR_ROOT / f"d{d}" / "result.json" for d in (1, 2, 3)
}


def load_prior(d: int) -> dict[str, Any]:
    path = PRIOR_RESULT_PATHS[d]
    payload = json.loads(path.read_text())
    if payload.get("status") != "completed_feasible":
        raise RuntimeError(f"Prior d={d} result is not completed_feasible: {path}")
    return payload


def prior_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): prior_runner.sha256(path)
        for path in PRIOR_RESULT_PATHS.values()
    }


def validate_prior_configuration(d: int, prior: dict[str, Any]) -> None:
    cfg = prior["configuration"]
    expected = {
        "n": N,
        "d": d,
        "partition_block_sizes": list(PARTITIONS[d]),
        "mode": OptimizationMode.FIXED_ERROR_MIN_COPIES.value,
        "error_target": ERROR_TARGET,
        "copy_ceiling": COPY_CEILING,
        "tuning_seeds": list(prior_runner.TUNING_SEEDS),
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise RuntimeError(
                f"Prior d={d} configuration mismatch for {key}: "
                f"expected {value!r}, found {cfg.get(key)!r}."
            )
    progressive = cfg["progressive_search"]
    if progressive["seed_fidelities"] != [1, 2, 4, 8, 16]:
        raise RuntimeError("Prior progressive seed schedule is not 1/2/4/8/16.")
    if progressive["max_candidates"] != 256:
        raise RuntimeError("Prior progressive maximum is not 256 candidates.")
    if progressive["relative_improvement_threshold"] != 0.0:
        raise RuntimeError("Prior relative-improvement threshold is not 0.0.")


def build_config(d: int, prior: dict[str, Any], *, resume: bool) -> OptimizationConfig:
    cfg = prior["configuration"]
    refinement = cfg["budget_refinement"]
    checkpoint = OUTPUT_ROOT / f"d{d}" / "checkpoint.json"
    return OptimizationConfig(
        total_copies=int(cfg["copy_ceiling"]),
        search_seed=int(cfg["search_seed"]),
        tuning_seeds=tuple(int(seed) for seed in cfg["tuning_seeds"]),
        holdout_seeds=tuple(int(seed) for seed in cfg["holdout_seeds"]),
        max_dense_qubits=N,
        max_enumeration_qubits=N,
        max_oracle_dense_qubits=N,
        simulation_backend=str(cfg["simulation_backend"]),
        number_of_candidates=16,
        halving_seed_counts=(1, 2, 4, 8, 16),
        retention_fraction=0.5,
        budget_refinement_enabled=bool(refinement["enabled"]),
        budget_refinement_steps=int(refinement["steps"]),
        budget_refinement_tolerance=int(refinement["tolerance"]),
        objective=OptimizationObjective(
            mode=OptimizationMode.FIXED_ERROR_MIN_COPIES,
            copy_ceiling=int(cfg["copy_ceiling"]),
            error_target=float(cfg["error_target"]),
            error_target_margin=0.0,
            success_probability_threshold=SUCCESS_PROBABILITY_THRESHOLD,
        ),
        checkpoint_path=str(checkpoint),
        resume_from_checkpoint=resume,
        checkpoint_every_n_evaluations=16,
        checkpoint_key=(
            f"6q-fixed-error-mean-success-d{d}-state-"
            f"{cfg['state_generation_seed']}"
        ),
    )


def build_progressive(prior: dict[str, Any]) -> ProgressiveSearchConfig:
    return ProgressiveSearchConfig(**prior["configuration"]["progressive_search"])


def build_search_space(prior: dict[str, Any]) -> SearchSpace:
    return SearchSpace(
        **{
            name: tuple(float(value) for value in bounds)
            for name, bounds in prior["configuration"]["search_space"].items()
        }
    )


def previous_summary(prior: dict[str, Any]) -> dict[str, Any]:
    tuning = prior["optimization"]["tuning"]
    errors = [float(item["trace_distance"]) for item in tuning["per_seed"]]
    successes = sum(error <= ERROR_TARGET for error in errors)
    budget = int(prior["optimization"]["refined_N_candidate"])
    return {
        "rule": "mean_only",
        "N_candidate": budget,
        "mean_trace_distance": float(tuning["mean_trace_distance"]),
        "max_trace_distance": float(tuning["max_trace_distance"]),
        "error_success_count_recovered": int(successes),
        "error_success_fraction_recovered": float(successes / len(errors)),
        "per_seed_trace_distances": errors,
    }


def seed_payload(item) -> dict[str, Any]:
    payload = prior_runner.seed_payload(item)
    if payload["operational_success"]:
        value = payload["trace_distance"]
        if value is None or not math.isfinite(float(value)):
            raise RuntimeError("Operational evaluation has no finite trace distance.")
    return payload


def aggregate_payload(evaluation) -> dict[str, Any]:
    errors = [
        float(item.trace_distance)
        for item in evaluation.seed_evaluations
        if item.trace_distance is not None
    ]
    return {
        "candidate_id": evaluation.candidate_id,
        "N_candidate": int(evaluation.N_candidate),
        "final_tuning_seed_count": evaluation.final_tuning_seed_count,
        "n_seeds_evaluated": int(evaluation.n_seeds_evaluated),
        "operationally_valid": bool(evaluation.operationally_valid),
        "mean_error_feasible": evaluation.mean_error_feasible,
        "success_fraction_feasible": evaluation.success_fraction_feasible,
        "final_target_feasible": evaluation.final_target_feasible,
        "all_error_feasible_deprecated_alias": evaluation.all_error_feasible,
        "success_probability_threshold": evaluation.success_probability_threshold,
        "error_success_count": evaluation.error_success_count,
        "required_error_success_count": evaluation.required_error_success_count,
        "error_success_fraction": evaluation.error_success_fraction,
        "mean_trace_distance": evaluation.mean_trace_distance,
        "median_trace_distance": float(np.median(errors)),
        "max_trace_distance": evaluation.max_trace_distance,
        "trace_distance_std": evaluation.trace_distance_std,
        "per_seed_trace_distances": errors,
        "mean_realized_copies": float(evaluation.mean_realized_copies),
        "max_realized_copies": int(evaluation.max_realized_copies),
        "mean_copy_utilization": float(evaluation.mean_copy_utilization),
        "max_copy_utilization": float(evaluation.max_copy_utilization),
        "operational_success_rate": float(evaluation.success_rate),
        "budget_feasible_rate": float(evaluation.budget_feasible_rate),
        "mean_stage_copies": dict(evaluation.mean_stage_copies),
        "max_stage_copies": dict(evaluation.max_stage_copies),
        "per_seed": [seed_payload(item) for item in evaluation.seed_evaluations],
    }


def assert_final_semantics(evaluation) -> None:
    if evaluation.n_seeds_evaluated != FINAL_TUNING_SEED_COUNT:
        raise RuntimeError("Final comparison did not use exactly 16 tuning seeds.")
    if evaluation.required_error_success_count != REQUIRED_SUCCESS_COUNT:
        raise RuntimeError("Final required success count is not 14.")
    expected = bool(
        evaluation.operationally_valid
        and evaluation.mean_error_feasible
        and evaluation.success_fraction_feasible
    )
    if evaluation.final_target_feasible is not expected:
        raise RuntimeError("Final feasibility does not equal operational AND mean AND success.")


def base_payload(
    d: int,
    prior: dict[str, Any],
    instance,
    config: OptimizationConfig,
    progressive: ProgressiveSearchConfig,
    space: SearchSpace,
) -> dict[str, Any]:
    cfg = prior["configuration"]
    return {
        "schema_version": 1,
        "status": "running",
        "configuration": {
            "n": N,
            "d": d,
            "partition_block_sizes": list(PARTITIONS[d]),
            "exact_hidden_partition": [
                list(block) for block in instance.oracle_truth.hidden_partition
            ],
            "mode": config.effective_objective.mode.value,
            "error_target": ERROR_TARGET,
            "effective_error_threshold": config.effective_objective.effective_error_threshold,
            "copy_ceiling": COPY_CEILING,
            "success_probability_threshold": SUCCESS_PROBABILITY_THRESHOLD,
            "final_tuning_seed_count": FINAL_TUNING_SEED_COUNT,
            "required_error_success_count": REQUIRED_SUCCESS_COUNT,
            "state_generation_seed": int(cfg["state_generation_seed"]),
            "state_seed_ledger": prior_runner.json_safe(instance.seed_ledger),
            "state_generation": cfg["state_generation"],
            "encoder_sampling": instance.oracle_truth.encoder_sampling,
            "encoder_steps": int(instance.oracle_truth.encoder_steps),
            "search_seed": int(cfg["search_seed"]),
            "preflight_learner_seed": int(cfg["preflight_learner_seed"]),
            "tuning_seeds": list(config.tuning_seeds),
            "historical_holdout_seeds_accepted_but_unused": list(config.holdout_seeds),
            "simulation_backend": config.simulation_backend,
            "search_space": prior_runner.json_safe(space),
            "progressive_search": prior_runner.json_safe(progressive),
            "budget_refinement": {
                "enabled": config.budget_refinement_enabled,
                "steps": config.budget_refinement_steps,
                "tolerance": config.budget_refinement_tolerance,
            },
            "checkpoint_path": config.checkpoint_path,
        },
        "prior_result_path": str(PRIOR_RESULT_PATHS[d].relative_to(ROOT)),
        "prior_result_sha256_before": prior_runner.sha256(PRIOR_RESULT_PATHS[d]),
        "prior_mean_only": previous_summary(prior),
        "state_digest_sha256": prior_runner.state_digest(instance),
        "production_hashes_before": prior_runner.production_hashes(),
        "all_prior_result_hashes_before": prior_hashes(),
        "preflight": None,
        "optimization": None,
        "failure": None,
    }


def write_result(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(prior_runner.json_safe(payload), indent=2, allow_nan=False) + "\n"
    )


def run_case(d: int, *, resume: bool) -> int:
    case_started = time.perf_counter()
    output_path = OUTPUT_ROOT / f"d{d}" / "result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not resume:
        raise FileExistsError(f"Fresh run refuses to overwrite {output_path}.")
    prior = load_prior(d)
    validate_prior_configuration(d, prior)
    config = build_config(d, prior, resume=resume)
    checkpoint_path = Path(config.checkpoint_path)
    if resume and not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
    if not resume and checkpoint_path.exists():
        raise FileExistsError(f"Fresh run refuses existing checkpoint {checkpoint_path}.")
    progressive = build_progressive(prior)
    space = build_search_space(prior)
    instance = prior_runner.build_instance(d)
    repeated_instance = prior_runner.build_instance(d)
    payload = base_payload(d, prior, instance, config, progressive, space)
    state_max_abs_difference = float(
        np.max(np.abs(instance.state.full() - repeated_instance.state.full()))
    )
    seed_ledger_matches = bool(
        payload["configuration"]["state_seed_ledger"]
        == prior["configuration"]["state_seed_ledger"]
        == prior_runner.json_safe(repeated_instance.seed_ledger)
    )
    encoder_matches = bool(
        instance.oracle_truth.encoder_gates
        == repeated_instance.oracle_truth.encoder_gates
        and instance.oracle_truth.encoder_sampling
        == prior["configuration"]["encoder_sampling"]
        and instance.oracle_truth.encoder_steps
        == prior["configuration"]["encoder_steps"]
    )
    payload["state_reconstruction"] = {
        "historical_raw_dense_digest_sha256": prior["state_digest_sha256"],
        "current_raw_dense_digest_sha256": payload["state_digest_sha256"],
        "raw_digest_matches_historical": (
            payload["state_digest_sha256"] == prior["state_digest_sha256"]
        ),
        "repeated_current_raw_dense_digest_sha256": prior_runner.state_digest(
            repeated_instance
        ),
        "repeated_state_max_abs_difference": state_max_abs_difference,
        "numerical_tolerance": 1e-14,
        "numerically_reproducible": state_max_abs_difference <= 1e-14,
        "seed_ledger_matches_historical_and_repeat": seed_ledger_matches,
        "encoder_configuration_matches_historical_and_repeat": encoder_matches,
        "note": (
            "Raw dense hashes can differ at last-bit floating-point level; "
            "scientific identity is verified from the recorded seeds/configuration, "
            "encoder, and numerical matrix agreement."
        ),
    }
    if not seed_ledger_matches:
        raise RuntimeError("Reconstructed child seed ledger differs from prior validation.")
    if not encoder_matches or state_max_abs_difference > 1e-14:
        raise RuntimeError("Exact seeded state configuration is not reproducible.")

    print(f"[d={d}] preflight start", flush=True)
    preflight = prior_runner.run_preflight(d, instance, config)
    payload["preflight"] = preflight
    if preflight["status"] != "passed":
        payload["status"] = "preflight_failed"
        payload["failure"] = {
            "phase": "preflight",
            "exception_type": preflight["exception_type"],
            "exception_message": preflight["exception_message"],
            "traceback": preflight["traceback"],
        }
        payload["timings_seconds"] = {
            "preflight": preflight["wall_clock_seconds"],
            "optimization": None,
            "case_total": time.perf_counter() - case_started,
        }
        payload["production_hashes_after"] = prior_runner.production_hashes()
        payload["production_files_unchanged"] = (
            payload["production_hashes_before"]
            == payload["production_hashes_after"]
        )
        payload["all_prior_result_hashes_after"] = prior_hashes()
        payload["prior_results_unchanged"] = (
            payload["all_prior_result_hashes_before"]
            == payload["all_prior_result_hashes_after"]
        )
        write_result(output_path, payload)
        print(f"[d={d}] preflight failed: {preflight['exception_message']}", flush=True)
        return 2

    print(
        f"[d={d}] preflight passed in {preflight['wall_clock_seconds']:.3f}s; "
        "full 256-candidate search start",
        flush=True,
    )
    optimization_started = time.perf_counter()
    optimization_learner_seeds: list[int] = []
    original_evaluator = objective_module.evaluate_candidate_on_seed

    def tracked_evaluator(*args, **kwargs):
        optimization_learner_seeds.append(int(args[2]))
        return original_evaluator(*args, **kwargs)

    objective_module.evaluate_candidate_on_seed = tracked_evaluator
    try:
        result = optimize_cebp_parameters_progressive(
            instance,
            COPY_CEILING,
            config,
            space,
            progressive_config=progressive,
        )
        optimization_seconds = time.perf_counter() - optimization_started
        assert_final_semantics(result.comparison_evaluation)
        if result.holdout_evaluation is not None:
            raise RuntimeError("Modern fixed-error unexpectedly returned a holdout evaluation.")
        holdout_seed_calls = sorted(
            set(optimization_learner_seeds) & set(config.holdout_seeds)
        )
        if holdout_seed_calls:
            raise RuntimeError(f"Holdout learner seeds were executed: {holdout_seed_calls}")
        catalog_by_id = {
            record.candidate_id: record for record in result.candidate_catalog
        }
        if result.best_candidate_id not in catalog_by_id:
            raise RuntimeError("Final selected candidate is absent from candidate catalog.")
        final_record = catalog_by_id[result.best_candidate_id]
        structural_id = result.best_candidate_id.split("-budget-", 1)[0]
        structural_record = catalog_by_id.get(structural_id, final_record)
        pool_sizes = [
            int(item.candidate_pool_size)
            for item in result.search_metadata.round_summaries
        ]
        reached_256 = bool(
            result.search_metadata.final_candidate_pool_size == 256
            and pool_sizes == [16, 32, 64, 128, 256]
        )
        if not reached_256:
            raise RuntimeError(f"Progressive candidate pools were {pool_sizes!r}, not full policy.")
        comparison = aggregate_payload(result.comparison_evaluation)
        if d == 1:
            grouping_cap = int(
                dict(final_record.derived.fixed_budget_stage_caps).get("grouping", -1)
            )
            grouping_realized = int(
                comparison["max_stage_copies"].get("grouping_ordinary_pool", -1)
            )
            if grouping_cap != 0 or grouping_realized != 0:
                raise RuntimeError("d=1 grouping allocation/realization is not zero.")
            for evaluation in result.comparison_evaluation.seed_evaluations:
                prior_runner.assert_d1_seed_accounting(
                    evaluation, result.best_candidate.N_candidate
                )
        old_budget = int(payload["prior_mean_only"]["N_candidate"])
        new_budget = int(result.best_candidate.N_candidate)
        payload["status"] = (
            "completed_feasible"
            if comparison["final_target_feasible"]
            else "completed_infeasible"
        )
        payload["optimization"] = {
            "termination_reason": result.search_metadata.termination_reason,
            "candidate_pool_sizes_reached": pool_sizes,
            "final_candidate_pool_size": int(result.search_metadata.final_candidate_pool_size),
            "reached_256_candidates": reached_256,
            "rounds_completed": int(result.search_metadata.rounds_completed),
            "final_comparison_seed_count": progressive.comparison_seed_count,
            "best_candidate_id": result.best_candidate_id,
            "best_candidate_resolvable_in_catalog": True,
            "best_candidate": prior_runner.json_safe(result.best_candidate),
            "best_derived_candidate": prior_runner.json_safe(result.best_derived_candidate),
            "initial_progressive_winner_N_candidate": int(
                structural_record.parameters.N_candidate
            ),
            "refined_N_candidate": new_budget,
            "copy_increase_vs_mean_only": new_budget - old_budget,
            "copy_increase_percent_vs_mean_only": (
                (new_budget - old_budget) / old_budget * 100.0
            ),
            "budget_refinement_enabled": bool(result.search_metadata.budget_refinement_enabled),
            "budget_refinement_note": result.search_metadata.budget_refinement_note,
            "budget_refinement_trials": prior_runner.json_safe(
                result.search_metadata.budget_refinement_trials
            ),
            "evaluation_attempt_count": int(result.evaluation_attempt_count),
            "actual_learner_run_count": int(result.actual_learner_run_count),
            "cache_hit_count": int(result.cache_hit_count),
            "unique_cached_seed_evaluation_count": int(
                result.unique_cached_seed_evaluation_count
            ),
            "preflight_rejection_count": int(result.preflight_rejection_count),
            "budget_refinement_actual_learner_run_count": int(
                result.budget_refinement_actual_learner_run_count
            ),
            "optimization_learner_seed_call_count": len(optimization_learner_seeds),
            "holdout": {
                "status": "not_evaluated",
                "evaluation": None,
                "configured_historical_seeds": list(config.holdout_seeds),
                "learner_seed_calls": holdout_seed_calls,
                "holdout_post_selection_only": bool(
                    result.search_metadata.holdout_post_selection_only
                ),
            },
            "round_summaries": prior_runner.json_safe(
                result.search_metadata.round_summaries
            ),
            "d1_grouping": (
                {
                    "allocated_copies": int(
                        dict(final_record.derived.fixed_budget_stage_caps)["grouping"]
                    ),
                    "max_realized_copies": int(
                        comparison["max_stage_copies"]["grouping_ordinary_pool"]
                    ),
                }
                if d == 1
                else None
            ),
            "tuning": comparison,
        }
        payload["timings_seconds"] = {
            "preflight": preflight["wall_clock_seconds"],
            "optimization": optimization_seconds,
            "case_total": time.perf_counter() - case_started,
        }
        checkpoint_path.unlink(missing_ok=True)
        exit_code = 0
        print(
            f"[d={d}] {payload['status']}: N={new_budget}, "
            f"mean={comparison['mean_trace_distance']:.8g}, "
            f"success={comparison['error_success_count']}/16, "
            f"pool=256, runtime={optimization_seconds:.3f}s",
            flush=True,
        )
    except Exception as error:
        optimization_seconds = time.perf_counter() - optimization_started
        payload["status"] = "runtime_failed"
        payload["failure"] = {
            "phase": "full_optimization",
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": traceback.format_exc(),
        }
        payload["timings_seconds"] = {
            "preflight": preflight["wall_clock_seconds"],
            "optimization": optimization_seconds,
            "case_total": time.perf_counter() - case_started,
        }
        exit_code = 3
        print(f"[d={d}] runtime failed: {type(error).__name__}: {error}", flush=True)
    finally:
        objective_module.evaluate_candidate_on_seed = original_evaluator

    payload["production_hashes_after"] = prior_runner.production_hashes()
    payload["production_files_unchanged"] = (
        payload["production_hashes_before"] == payload["production_hashes_after"]
    )
    payload["all_prior_result_hashes_after"] = prior_hashes()
    payload["prior_results_unchanged"] = (
        payload["all_prior_result_hashes_before"]
        == payload["all_prior_result_hashes_after"]
    )
    write_result(output_path, payload)
    return exit_code


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("1", "2", "3", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    cases = (1, 2, 3) if args.case == "all" else (int(args.case),)
    codes = []
    for d in cases:
        try:
            codes.append(run_case(d, resume=args.resume))
        except Exception as error:
            print(f"[d={d}] runner setup failed: {type(error).__name__}: {error}")
            codes.append(4)
    return max(codes, default=0)


if __name__ == "__main__":
    raise SystemExit(main())
