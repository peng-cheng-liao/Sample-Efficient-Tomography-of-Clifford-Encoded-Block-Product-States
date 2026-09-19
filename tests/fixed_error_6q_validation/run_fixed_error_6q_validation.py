#!/usr/bin/env python3
"""Run one or all deterministic 6q fixed-error validation cases."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass, replace
from enum import Enum
import hashlib
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

from main_v2 import random_cebp_state  # noqa: E402
from Optimization import (  # noqa: E402
    DEFAULT_FIXED_BUDGET_MIN_ERROR_IMPROVEMENT_RATIO,
    MAX_FIXED_BUDGET_MIN_ERROR_CANDIDATES,
    FixedErrorCandidateParameters,
    OptimizationConfig,
    OptimizationMode,
    OptimizationObjective,
    ProgressiveSearchConfig,
    SearchSpace,
    derive_candidate,
    optimize_cebp_parameters_progressive,
)
from Optimization.objective import evaluate_candidate_on_seed  # noqa: E402


N = 6
ERROR_TARGET = 0.05
COPY_CEILING = 10_000_000
SEARCH_SEED = 6_200_501
PREFLIGHT_SEED = 6_209_001
TUNING_SEEDS = tuple(range(6_210_001, 6_210_017))
HOLDOUT_SEEDS = tuple(range(6_220_001, 6_220_005))
STATE_SEEDS = {1: 6_201_001, 2: 6_202_001, 3: 6_203_001}
PARTITIONS = {1: (1, 1, 1, 1, 1, 1), 2: (2, 2, 2), 3: (3, 3)}
OUTPUT_ROOT = ROOT / "tests" / "fixed_error_6q_validation"
PRODUCTION_PATHS = (
    ROOT / "main_v2.py",
    ROOT / "Optimization" / "objective.py",
    ROOT / "Optimization" / "parameterization.py",
    ROOT / "Optimization" / "progressive_search.py",
    ROOT / "Optimization" / "search.py",
    ROOT / "Optimization" / "specification.py",
)


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


def production_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256(path) for path in PRODUCTION_PATHS}


def state_digest(instance) -> str:
    matrix = np.ascontiguousarray(instance.state.full(), dtype=np.complex128)
    return hashlib.sha256(matrix.view(np.uint8)).hexdigest()


def build_instance(d: int):
    return random_cebp_state(
        n=N,
        d=d,
        block_sizes=PARTITIONS[d],
        pure=False,
        seed=STATE_SEEDS[d],
        oracle_backend="structured",
        max_dense_debug_qubits=N,
    )


def build_config(d: int, *, resume: bool) -> OptimizationConfig:
    checkpoint = OUTPUT_ROOT / f"d{d}" / "checkpoint.json"
    return OptimizationConfig(
        total_copies=COPY_CEILING,
        search_seed=SEARCH_SEED,
        tuning_seeds=TUNING_SEEDS,
        holdout_seeds=HOLDOUT_SEEDS,
        max_dense_qubits=N,
        max_enumeration_qubits=N,
        max_oracle_dense_qubits=N,
        simulation_backend="batched_counts",
        number_of_candidates=16,
        halving_seed_counts=(1, 2, 4, 8, 16),
        retention_fraction=0.5,
        budget_refinement_enabled=True,
        budget_refinement_steps=16,
        budget_refinement_tolerance=1,
        objective=OptimizationObjective(
            mode=OptimizationMode.FIXED_ERROR_MIN_COPIES,
            copy_ceiling=COPY_CEILING,
            error_target=ERROR_TARGET,
            error_target_margin=0.0,
        ),
        checkpoint_path=str(checkpoint),
        resume_from_checkpoint=resume,
        checkpoint_every_n_evaluations=16,
        checkpoint_key=f"6q-fixed-error-d{d}-state-{STATE_SEEDS[d]}",
    )


def build_progressive() -> ProgressiveSearchConfig:
    return ProgressiveSearchConfig(
        initial_candidates=16,
        max_candidates=MAX_FIXED_BUDGET_MIN_ERROR_CANDIDATES,
        candidate_growth_factor=2,
        seed_fidelities=(1, 2, 4, 8, 16),
        comparison_seed_count=16,
        relative_improvement_threshold=(
            DEFAULT_FIXED_BUDGET_MIN_ERROR_IMPROVEMENT_RATIO
        ),
        improvement_patience=2,
        min_rounds=3,
        retention_fraction=0.5,
        checkpoint_every_n_new_evaluations=16,
    )


def preflight_candidate() -> FixedErrorCandidateParameters:
    return FixedErrorCandidateParameters(
        N_candidate=1_000_000,
        h_min=0.75,
        h_max=0.92,
        theta_tau_multiplier=8.0,
        eta_test=0.10,
        peel_weight=1.0,
        recovery_weight=1.0,
        grouping_weight=1.0,
        syndrome_weight=0.25,
        tomography_weight=2.0,
    )


def seed_payload(item) -> dict[str, Any]:
    trace_distance = item.trace_distance
    if item.operational_success and (
        trace_distance is None or not math.isfinite(float(trace_distance))
    ):
        raise RuntimeError("Operational seed evaluation has no finite trace distance.")
    stage_records = {
        record[0]: {
            "assigned_cap": int(record[1]),
            "realized_copies": int(record[2]),
            "unused_copies": int(record[3]),
            "budget_exhausted": bool(record[4]),
            "stage_complete": bool(record[5]),
            "degradation_reason": record[6],
        }
        for record in item.fixed_budget_stage_records
    }
    return {
        "learner_seed": int(item.learner_seed),
        "execution_branch": item.execution_branch,
        "operational_success": bool(item.operational_success),
        "estimator_available": bool(item.estimator_available),
        "budget_feasible": bool(item.budget_feasible),
        "trace_distance": trace_distance,
        "error_feasible_diagnostic": item.error_feasible,
        "realized_total": int(item.realized_total),
        "copy_utilization": float(item.copy_utilization),
        "realized_copy_ledger": dict(item.realized_copy_ledger),
        "fixed_budget_stage_records": stage_records,
        "tomography_realized_copies": item.tomography_fixed_budget,
        "failure_stage": item.failure_stage,
        "failure_reason": item.failure_reason,
        "preflight_rejected": bool(item.preflight_rejected),
    }


def aggregate_payload(evaluation) -> dict[str, Any]:
    return {
        "candidate_id": evaluation.candidate_id,
        "N_candidate": int(evaluation.N_candidate),
        "target_feasible": bool(evaluation.mean_error_feasible),
        "mean_error_feasible": evaluation.mean_error_feasible,
        "all_error_feasible_deprecated_alias": evaluation.all_error_feasible,
        "operationally_valid": bool(evaluation.operationally_valid),
        "n_seeds_evaluated": int(evaluation.n_seeds_evaluated),
        "mean_trace_distance": evaluation.mean_trace_distance,
        "max_trace_distance": evaluation.max_trace_distance,
        "trace_distance_std": evaluation.trace_distance_std,
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


def assert_d1_seed_accounting(item, candidate_budget: int) -> None:
    if item.execution_branch != "d1_specialized":
        raise RuntimeError("d=1 evaluation did not execute the d1-specialized branch.")
    stages = {record[0]: record for record in item.fixed_budget_stage_records}
    if stages.get("grouping", (None, None, None))[1:3] != (0, 0):
        raise RuntimeError("d=1 grouping allocation/realization was not exactly zero.")
    ledger = dict(item.realized_copy_ledger)
    tomography_realized = int(ledger.get("conditional_one_qubit_pool", 0))
    tomography_cap = int(stages["tomography"][1])
    if tomography_realized > tomography_cap:
        raise RuntimeError("d=1 conditional tomography exceeded its stage cap.")
    if item.realized_total > int(candidate_budget):
        raise RuntimeError("d=1 evaluation exceeded its candidate budget.")
    if item.failure_reason == "insufficient_postselected_shots":
        raise RuntimeError("Practical d=1 execution used a theorem accepted-shot target.")


def base_payload(d: int, instance, config, progressive, space) -> dict[str, Any]:
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
            "effective_error_threshold": (
                config.effective_objective.effective_error_threshold
            ),
            "copy_ceiling": COPY_CEILING,
            "state_generation_seed": STATE_SEEDS[d],
            "state_seed_ledger": json_safe(instance.seed_ledger),
            "state_generation": "Hilbert-Schmidt mixed latent blocks",
            "encoder_sampling": instance.oracle_truth.encoder_sampling,
            "encoder_steps": int(instance.oracle_truth.encoder_steps),
            "search_seed": SEARCH_SEED,
            "preflight_learner_seed": PREFLIGHT_SEED,
            "tuning_seeds": list(TUNING_SEEDS),
            "holdout_seeds": list(HOLDOUT_SEEDS),
            "simulation_backend": config.simulation_backend,
            "search_space": json_safe(space),
            "progressive_search": json_safe(progressive),
            "budget_refinement": {
                "enabled": config.budget_refinement_enabled,
                "steps": config.budget_refinement_steps,
                "tolerance": config.budget_refinement_tolerance,
            },
            "checkpoint_path": config.checkpoint_path,
        },
        "state_digest_sha256": state_digest(instance),
        "production_hashes_before": production_hashes(),
        "preflight": None,
        "optimization": None,
        "failure": None,
    }


def run_preflight(d: int, instance, config) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        repeated = build_instance(d)
        digest = state_digest(instance)
        repeated_digest = state_digest(repeated)
        state_difference = float(
            np.max(np.abs(instance.state.full() - repeated.state.full()))
        )
        if (
            state_difference > 1e-14
            or instance.seed_ledger != repeated.seed_ledger
            or instance.oracle_truth.encoder_gates
            != repeated.oracle_truth.encoder_gates
        ):
            raise RuntimeError(
                "Deterministic state regeneration changed the seeded state."
            )
        expected = tuple(
            tuple(range(sum(PARTITIONS[d][:index]), sum(PARTITIONS[d][: index + 1])))
            for index in range(len(PARTITIONS[d]))
        )
        if tuple(instance.oracle_truth.hidden_partition) != expected:
            raise RuntimeError("Generated hidden partition does not match the exact request.")
        if config.effective_objective.mode is not OptimizationMode.FIXED_ERROR_MIN_COPIES:
            raise RuntimeError("Preflight constructed the wrong objective mode.")
        derived = derive_candidate(
            preflight_candidate(),
            n=N,
            d=d,
            total_copies=COPY_CEILING,
            optimization_config=config,
        )
        evaluation = evaluate_candidate_on_seed(
            instance,
            derived,
            PREFLIGHT_SEED,
            COPY_CEILING,
            config,
            objective=config.effective_objective,
        )
        if not evaluation.operational_success or not evaluation.estimator_available:
            raise RuntimeError(
                "Preflight candidate did not produce a valid operational estimator: "
                f"stage={evaluation.failure_stage!r}, reason={evaluation.failure_reason!r}."
            )
        if evaluation.trace_distance is None or not math.isfinite(
            float(evaluation.trace_distance)
        ):
            raise RuntimeError("Preflight estimator has no finite trace distance.")
        if not math.isfinite(float(evaluation.realized_total)):
            raise RuntimeError("Preflight copy accounting is non-finite.")
        if evaluation.realized_total > derived.physical_copy_budget:
            raise RuntimeError("Preflight exceeded its candidate physical budget.")
        if d == 1:
            assert_d1_seed_accounting(evaluation, derived.physical_copy_budget)
        return {
            "status": "passed",
            "wall_clock_seconds": time.perf_counter() - started,
            "deterministic_state_regeneration": True,
            "repeated_state_digest_sha256": repeated_digest,
            "repeated_state_max_abs_difference": state_difference,
            "candidate": json_safe(derived.parameters),
            "candidate_budget": int(derived.physical_copy_budget),
            "seed_evaluation": seed_payload(evaluation),
            "exception_type": None,
            "exception_message": None,
            "traceback": None,
        }
    except Exception as error:
        return {
            "status": "failed",
            "wall_clock_seconds": time.perf_counter() - started,
            "deterministic_state_regeneration": None,
            "candidate": json_safe(preflight_candidate()),
            "candidate_budget": preflight_candidate().N_candidate,
            "seed_evaluation": None,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": traceback.format_exc(),
        }


def run_case(d: int, *, resume: bool) -> int:
    case_started = time.perf_counter()
    output_path = OUTPUT_ROOT / f"d{d}" / "result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = build_config(d, resume=resume)
    checkpoint_path = Path(config.checkpoint_path)
    if resume and not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
    if not resume and checkpoint_path.exists():
        raise FileExistsError(
            f"Fresh run refused because checkpoint exists: {checkpoint_path}; use --resume."
        )
    progressive = build_progressive()
    space = SearchSpace()
    instance = build_instance(d)
    payload = base_payload(d, instance, config, progressive, space)

    print(f"[d={d}] preflight start", flush=True)
    preflight = run_preflight(d, instance, config)
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
        payload["production_hashes_after"] = production_hashes()
        payload["production_files_unchanged"] = (
            payload["production_hashes_before"] == payload["production_hashes_after"]
        )
        output_path.write_text(json.dumps(json_safe(payload), indent=2) + "\n")
        print(
            f"[d={d}] preflight failed: {preflight['exception_type']}: "
            f"{preflight['exception_message']}",
            flush=True,
        )
        return 2

    print(
        f"[d={d}] preflight passed in {preflight['wall_clock_seconds']:.3f}s; "
        "full 256-candidate search start",
        flush=True,
    )
    optimization_started = time.perf_counter()
    try:
        result = optimize_cebp_parameters_progressive(
            instance,
            COPY_CEILING,
            config,
            space,
            progressive_config=progressive,
        )
        optimization_seconds = time.perf_counter() - optimization_started
        comparison = aggregate_payload(result.comparison_evaluation)
        holdout = aggregate_payload(result.holdout_evaluation)
        catalog_by_id = {
            record.candidate_id: record for record in result.candidate_catalog
        }
        if result.best_candidate_id not in catalog_by_id:
            raise RuntimeError("Final selected candidate is absent from candidate_catalog.")
        if len(catalog_by_id) != len(result.candidate_catalog):
            raise RuntimeError("candidate_catalog contains duplicate candidate IDs.")
        final_record = catalog_by_id[result.best_candidate_id]
        base_id = result.best_candidate_id.split("-budget-", 1)[0]
        structural_record = catalog_by_id.get(base_id, final_record)
        pool_sizes = [
            int(item.candidate_pool_size)
            for item in result.search_metadata.round_summaries
        ]
        reached_256 = (
            result.search_metadata.final_candidate_pool_size
            == MAX_FIXED_BUDGET_MIN_ERROR_CANDIDATES
            and MAX_FIXED_BUDGET_MIN_ERROR_CANDIDATES in pool_sizes
        )
        status = (
            "completed_feasible"
            if comparison["target_feasible"]
            else "completed_infeasible"
        )
        payload["status"] = status
        if d == 1:
            if any(
                dict(record.derived.fixed_budget_stage_caps).get("grouping") != 0
                for record in result.candidate_catalog
            ):
                raise RuntimeError("A catalogued d=1 candidate allocated grouping copies.")
            for evaluation in (
                *result.comparison_evaluation.seed_evaluations,
                *result.holdout_evaluation.seed_evaluations,
            ):
                assert_d1_seed_accounting(
                    evaluation, result.best_candidate.N_candidate
                )
            if result.comparison_evaluation.n_seeds_evaluated != 16:
                raise RuntimeError("Final d=1 comparison did not use all 16 tuning seeds.")
        payload["optimization"] = {
            "termination_reason": result.search_metadata.termination_reason,
            "candidate_pool_sizes_reached": pool_sizes,
            "final_candidate_pool_size": int(
                result.search_metadata.final_candidate_pool_size
            ),
            "reached_256_candidates": reached_256,
            "rounds_completed": int(result.search_metadata.rounds_completed),
            "comparison_seed_count": progressive.comparison_seed_count,
            "total_valid_candidates_pregenerated": int(
                result.search_metadata.total_valid_candidates_pregenerated
            ),
            "returned_candidate_catalog_count": len(result.candidate_catalog),
            "best_candidate_resolvable_in_catalog": True,
            "best_candidate_id": result.best_candidate_id,
            "best_candidate": json_safe(result.best_candidate),
            "best_derived_candidate": json_safe(result.best_derived_candidate),
            "final_candidate_record": json_safe(final_record),
            "pre_refinement_N_candidate": int(
                structural_record.parameters.N_candidate
            ),
            "refined_N_candidate": int(result.best_candidate.N_candidate),
            "budget_reduction": int(
                structural_record.parameters.N_candidate
                - result.best_candidate.N_candidate
            ),
            "budget_refinement_enabled": bool(
                result.search_metadata.budget_refinement_enabled
            ),
            "budget_refinement_note": result.search_metadata.budget_refinement_note,
            "budget_refinement_trials": json_safe(
                result.search_metadata.budget_refinement_trials
            ),
            "holdout_post_selection_only": bool(
                result.search_metadata.holdout_post_selection_only
            ),
            "checkpoint_enabled": bool(result.search_metadata.checkpoint_enabled),
            "resumed_from_checkpoint": bool(
                result.search_metadata.resumed_from_checkpoint
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
            "round_summaries": json_safe(result.search_metadata.round_summaries),
            "d1_validation_assertions": (
                {
                    "actual_branch": "d1_specialized",
                    "all_catalogued_grouping_caps_zero": True,
                    "final_tuning_and_holdout_grouping_realized_zero": True,
                    "conditional_tomography_within_stage_cap": True,
                    "total_realized_within_candidate_budget": True,
                    "theorem_accepted_shot_target_controls_feasibility": False,
                    "final_comparison_seed_count": 16,
                }
                if d == 1
                else None
            ),
            "tuning": comparison,
            "holdout": holdout,
        }
        payload["timings_seconds"] = {
            "preflight": preflight["wall_clock_seconds"],
            "optimization": optimization_seconds,
            "case_total": time.perf_counter() - case_started,
        }
        checkpoint_path.unlink(missing_ok=True)
        exit_code = 0
        print(
            f"[d={d}] {status}: N={result.best_candidate.N_candidate}, "
            f"mean_tuning={comparison['mean_trace_distance']:.8g}, "
            f"mean_holdout={holdout['mean_trace_distance']:.8g}, "
            f"pool={result.search_metadata.final_candidate_pool_size}, "
            f"runtime={optimization_seconds:.3f}s",
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
        print(
            f"[d={d}] runtime failed: {type(error).__name__}: {error}",
            flush=True,
        )

    payload["production_hashes_after"] = production_hashes()
    payload["production_files_unchanged"] = (
        payload["production_hashes_before"] == payload["production_hashes_after"]
    )
    output_path.write_text(json.dumps(json_safe(payload), indent=2) + "\n")
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
