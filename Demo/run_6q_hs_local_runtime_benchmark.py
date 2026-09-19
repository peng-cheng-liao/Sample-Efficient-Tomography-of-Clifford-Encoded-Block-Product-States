#!/usr/bin/env python3
"""Run one reproducible 6q HS-mixed fixed-budget pilot for two partitions."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
from typing import Any, Iterable, Optional, Sequence


PROCESS_STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import qutip as qt  # noqa: E402

from Demo import run_6q_fixed_budget_min_error_1m as canonical_demo  # noqa: E402
from main_v2 import (  # noqa: E402
    ExecutionPolicy,
    debug_end_to_end_trace_error,
    full_cebp_tomography,
    random_cebp_state,
)
from Optimization import (  # noqa: E402
    DEFAULT_FIXED_BUDGET_MIN_ERROR_IMPROVEMENT_RATIO,
    MAX_FIXED_BUDGET_MIN_ERROR_CANDIDATES,
    OptimizationConfig,
    OptimizationObjective,
    ProgressiveSearchConfig,
    candidate_to_end_to_end_config,
    preflight_resource_check,
)
from Optimization.search import _sample_valid_candidates  # noqa: E402


N = 6
DEFAULT_BUDGET = 1_000_000
DEFAULT_MEASUREMENT_SEED = 6_200_001
DEFAULT_INSTANCE_SEED_222 = 6_222_001
DEFAULT_INSTANCE_SEED_33 = 6_330_001
DEFAULT_CONFIG_SEED_222 = 6_222_101
DEFAULT_CONFIG_SEED_33 = 6_330_101
DEFAULT_WORKERS = 1
DEFAULT_REPORT = ROOT / "Reports" / "CEBP_6Q_HS_LOCAL_RUNTIME_BENCHMARK_REPORT.txt"
DEFAULT_JSON = ROOT / "Reports" / "CEBP_6Q_HS_LOCAL_RUNTIME_BENCHMARK.json"
SCRIPT_PATH = Path(__file__).resolve()
CANONICAL_DEMO_PATH = ROOT / "Demo" / "run_6q_fixed_budget_min_error_1m.py"
TASK_PATHS = (
    SCRIPT_PATH,
    DEFAULT_REPORT,
    DEFAULT_JSON,
    ROOT / "Reports" / "CEBP_6Q_HS_LOCAL_RUNTIME_BENCHMARK.patch",
)
TOP_LEVEL_STAGES = {
    "peeling",
    "recovery",
    "grouping",
    "localization",
    "syndrome",
    "tomography",
}
THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYTHONHASHSEED",
    "PYTHONDONTWRITEBYTECODE",
    "MPLCONFIGDIR",
    "NUMBA_CACHE_DIR",
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


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        return f"UNAVAILABLE: {completed.stderr.strip()}"
    return completed.stdout.rstrip()


def git_metadata() -> dict[str, Any]:
    return {
        "branch": git_output("branch", "--show-current"),
        "head": git_output("rev-parse", "HEAD"),
        "status_short": git_output("status", "--short"),
        "diff_stat_head": git_output("diff", "HEAD", "--stat"),
        "diff_name_status_head": git_output("diff", "HEAD", "--name-status"),
        "diff_stat_unstaged": git_output("diff", "--stat"),
        "diff_name_status_unstaged": git_output("diff", "--name-status"),
        "diff_stat_staged": git_output("diff", "--cached", "--stat"),
        "diff_name_status_staged": git_output("diff", "--cached", "--name-status"),
    }


def _status_without_task_files(status: str, task_paths: Sequence[Path]) -> str:
    relative = {
        str(path.resolve().relative_to(ROOT))
        for path in task_paths
        if path.resolve().is_relative_to(ROOT)
    }
    kept = []
    for line in status.splitlines():
        if any(path in line for path in relative):
            continue
        kept.append(line)
    return "\n".join(kept)


def _package_version(name: str) -> str:
    try:
        module = __import__(name)
    except Exception as error:  # pragma: no cover - machine-dependent metadata
        return f"unavailable ({type(error).__name__})"
    return str(getattr(module, "__version__", "unknown"))


def _hardware_profile() -> dict[str, str]:
    try:
        completed = subprocess.run(
            ("system_profiler", "SPHardwareDataType"),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    wanted = {
        "Model Name": "model_name",
        "Model Identifier": "model_identifier",
        "Chip": "chip",
        "Processor Name": "processor_name",
        "Total Number of Cores": "physical_core_description",
        "Memory": "memory",
    }
    result: dict[str, str] = {}
    for raw in completed.stdout.splitlines():
        line = raw.strip()
        for prefix, key in wanted.items():
            marker = f"{prefix}:"
            if line.startswith(marker):
                result[key] = line[len(marker):].strip()
    return result


def environment_metadata(workers: int) -> dict[str, Any]:
    return {
        "os": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "machine": platform.machine(),
        "processor": platform.processor() or "unreported",
        "logical_cores": os.cpu_count(),
        "hardware": _hardware_profile(),
        "numpy_version": np.__version__,
        "numba_version": _package_version("numba"),
        "qutip_version": qt.__version__,
        "inner_enumeration_workers_requested": int(workers),
        "candidate_level_workers": 1,
        "candidate_level_parallelism": "not implemented/exposed; pilot cases run sequentially",
        "thread_environment": {
            name: os.environ.get(name, "UNSET") for name in THREAD_ENV_NAMES
        },
    }


def optimizer_semantics() -> dict[str, Any]:
    progressive = canonical_demo.build_progressive_config()
    if not isinstance(progressive, ProgressiveSearchConfig):
        raise RuntimeError("Canonical demo did not return ProgressiveSearchConfig.")
    tuning = tuple(int(seed) for seed in canonical_demo.TUNING_SEEDS)
    holdout = tuple(int(seed) for seed in canonical_demo.HOLDOUT_SEEDS)
    if progressive.max_candidates != MAX_FIXED_BUDGET_MIN_ERROR_CANDIDATES:
        raise RuntimeError("Canonical demo and centralized candidate cap disagree.")
    if progressive.relative_improvement_threshold != (
        DEFAULT_FIXED_BUDGET_MIN_ERROR_IMPROVEMENT_RATIO
    ):
        raise RuntimeError("Canonical demo and centralized improvement threshold disagree.")
    lower, lower_rounds = _progressive_cache_bound(progressive, prefer_low_depth=False)
    upper, upper_rounds = _progressive_cache_bound(progressive, prefer_low_depth=True)
    return {
        "objective": "fixed_budget_min_error",
        "candidate_semantics": (
            "The cap counts distinct sampled parameter configurations, not measurement seeds."
        ),
        "candidate_cap": int(progressive.max_candidates),
        "catalog_pregenerated_to_cap": True,
        "initial_candidates": int(progressive.initial_candidates),
        "candidate_growth_factor": int(progressive.candidate_growth_factor),
        "candidate_pool_sizes_at_full_cap": _pool_sizes(progressive),
        "tuning_measurement_seeds": list(tuning),
        "tuning_seed_count": len(tuning),
        "seed_fidelities": list(progressive.seed_fidelities),
        "comparison_seed_count": int(progressive.comparison_seed_count),
        "holdout_measurement_seeds": list(holdout),
        "holdout_seed_count": len(holdout),
        "holdout_behavior": (
            "All holdout seeds are evaluated only for the selected final incumbent; "
            "holdout never retunes candidates."
        ),
        "relative_improvement_threshold": float(
            progressive.relative_improvement_threshold
        ),
        "improvement_patience": int(progressive.improvement_patience),
        "minimum_rounds": int(progressive.min_rounds),
        "progressive_early_stop": (
            "After the minimum round count, stop once the configured number of "
            "consecutive rounds fail to reach the minimum relative improvement; "
            "perfect zero loss also stops immediately."
        ),
        "semantic_cache": True,
        "full_cap_tuning_run_lower_bound": int(lower),
        "full_cap_tuning_run_upper_bound": int(upper),
        "full_cap_tuning_runs_by_round_lower": list(lower_rounds),
        "full_cap_tuning_runs_by_round_upper": list(upper_rounds),
        "full_cap_completed_run_lower_bound_including_holdout": int(lower + len(holdout)),
        "full_cap_completed_run_upper_bound_including_holdout": int(upper + len(holdout)),
        "naive_all_candidates_all_tuning_runs": int(
            progressive.max_candidates * len(tuning)
        ),
        "naive_completed_runs_including_final_holdout": int(
            progressive.max_candidates * len(tuning) + len(holdout)
        ),
        "candidate_parallelism": (
            "Current OptimizationConfig rejects candidate_workers other than 1; "
            "the controller is serial and exposes only inner enumeration workers."
        ),
        "source": str(CANONICAL_DEMO_PATH.relative_to(ROOT)),
    }


def _pool_sizes(progressive: ProgressiveSearchConfig) -> list[int]:
    sizes = []
    size = int(progressive.initial_candidates)
    while True:
        sizes.append(size)
        if size >= progressive.max_candidates:
            return sizes
        size = min(
            int(progressive.max_candidates),
            size * int(progressive.candidate_growth_factor),
        )


def _progressive_cache_bound(
    progressive: ProgressiveSearchConfig, *, prefer_low_depth: bool
) -> tuple[int, tuple[int, ...]]:
    """Bound full-cap unique tuning runs using the controller's real cache rules."""

    depths = [0] * int(progressive.max_candidates)
    total = 0
    per_round = []
    pool_sizes = _pool_sizes(progressive)
    for round_index, pool_size in enumerate(pool_sizes, start=1):
        before = total
        entering = list(range(pool_size))
        maximum_stage_index = min(
            round_index + 1, len(progressive.seed_fidelities) - 1
        )
        fidelities = progressive.seed_fidelities[: maximum_stage_index + 1]
        for stage_index, fidelity in enumerate(fidelities):
            for candidate_index in entering:
                total += max(0, int(fidelity) - depths[candidate_index])
                depths[candidate_index] = max(depths[candidate_index], int(fidelity))
            keep = (
                1
                if stage_index + 1 == len(fidelities)
                else max(
                    1,
                    int(
                        math.ceil(
                            len(entering) * float(progressive.retention_fraction)
                        )
                    ),
                )
            )
            entering = sorted(
                entering,
                key=lambda index: (depths[index], index),
                reverse=not prefer_low_depth,
            )[:keep]
        challenger = entering[0]
        comparison = int(progressive.comparison_seed_count)
        total += max(0, comparison - depths[challenger])
        depths[challenger] = max(depths[challenger], comparison)
        per_round.append(total - before)
    return total, tuple(per_round)


def build_optimization_config(
    *, budget: int, config_seed: int, measurement_seed: int, workers: int
) -> OptimizationConfig:
    unused_holdout_seed = int(measurement_seed) + 1
    if unused_holdout_seed == measurement_seed:
        raise RuntimeError("Could not construct an unused disjoint holdout seed.")
    return OptimizationConfig(
        total_copies=int(budget),
        search_seed=int(config_seed),
        tuning_seeds=(int(measurement_seed),),
        holdout_seeds=(unused_holdout_seed,),
        max_dense_qubits=N,
        max_oracle_dense_qubits=N,
        inner_enumeration_workers=int(workers),
        simulation_backend="batched_counts",
        max_preflight_estimated_copies=500_000_000,
        max_single_grouping_query_shots=2_000_000,
        number_of_candidates=1,
        halving_seed_counts=(1,),
        retention_fraction=0.5,
        tomography_refinement_enabled=False,
        target_copy_utilization=0.99,
        objective=OptimizationObjective(
            mode="fixed_budget_min_error", copy_ceiling=int(budget)
        ),
        checkpoint_path=None,
        verbose=False,
    )


def _latent_block_audit(instance: Any) -> list[dict[str, Any]]:
    result = []
    for block, state in zip(
        instance.oracle_truth.hidden_partition,
        instance.oracle_truth.latent_block_states,
    ):
        matrix = np.asarray(state.full(), dtype=complex)
        eigenvalues = np.linalg.eigvalsh(matrix)
        result.append(
            {
                "block": list(block),
                "qubits": len(block),
                "hilbert_dimension": int(matrix.shape[0]),
                "is_ket": bool(state.isket),
                "is_hermitian": bool(state.isherm),
                "trace_real": float(np.real(state.tr())),
                "minimum_eigenvalue": float(eigenvalues.min()),
                "numerical_rank": int(
                    np.count_nonzero(eigenvalues > 1e-12)
                ),
                "purity": float(np.real((state * state).tr())),
            }
        )
    return result


def _instance_validation(
    instance: Any, *, d: int, block_sizes: tuple[int, ...]
) -> dict[str, Any]:
    expected_partition = []
    offset = 0
    for size in block_sizes:
        expected_partition.append(tuple(range(offset, offset + size)))
        offset += size
    truth = instance.oracle_truth
    measurement_source = instance.measurement_source
    checks = {
        "n_is_6": instance.n == N,
        "d_matches": instance.d == d,
        "hidden_partition_exact": tuple(truth.hidden_partition)
        == tuple(expected_partition),
        "latent_source_hilbert_schmidt_mixed": truth.latent_state_source
        == "hilbert_schmidt_mixed",
        "not_user_supplied": truth.latent_state_source != "user_supplied",
        "mixed_representation": not instance.is_ket,
        "all_block_dimensions_match": all(
            state.shape == (2**size, 2**size)
            and not state.isket
            for size, state in zip(block_sizes, truth.latent_block_states)
        ),
        "structured_measurement_backend": getattr(
            measurement_source, "_structured_state", None
        )
        is not None,
        "no_initial_measurement_copies": instance.copy_ledger.total == 0,
    }
    return {
        "checks": checks,
        "all_passed": all(checks.values()),
        "expected_partition": [list(item) for item in expected_partition],
        "actual_partition": [list(item) for item in truth.hidden_partition],
    }


def _stage_theorem_flags(result: Any) -> dict[str, Optional[bool]]:
    attributes = {
        "peeling": "theorem_preconditions_hold",
        "recovery": "theorem_recovery_preconditions_hold",
        "grouping": "theorem_grouping_preconditions_hold",
        "localization": "theorem_localization_preconditions_hold",
        "syndrome": "theorem_preconditions_hold",
        "tomography": "theorem_preconditions_hold",
    }
    flags: dict[str, Optional[bool]] = {}
    for name, attribute in attributes.items():
        stage = getattr(result, name, None)
        flags[name] = (
            None if stage is None else bool(getattr(stage, attribute, False))
        )
    return flags


def _enumeration_audit(result: Any) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for name in ("peeling", "recovery"):
        stage = getattr(result, name, None)
        diagnostics = getattr(stage, "enumeration_diagnostics", None)
        audit[name] = None if diagnostics is None else json_safe(diagnostics)
    return audit


def _dominant_stage(stage_timings: Sequence[Sequence[Any]]) -> Optional[dict[str, Any]]:
    candidates = [
        (str(name), float(seconds))
        for name, seconds in stage_timings
        if str(name) in TOP_LEVEL_STAGES
    ]
    if not candidates:
        return None
    name, seconds = max(candidates, key=lambda item: item[1])
    return {"stage": name, "seconds": seconds}


def run_case(
    *,
    label: str,
    d: int,
    block_sizes: tuple[int, ...],
    instance_seed: int,
    config_seed: int,
    measurement_seed: int,
    budget: int,
    workers: int,
    verbose: bool,
) -> dict[str, Any]:
    case_started = time.perf_counter()
    state_started = time.perf_counter()
    instance = random_cebp_state(
        n=N,
        d=d,
        block_sizes=block_sizes,
        block_states=None,
        pure=False,
        seed=int(instance_seed),
        oracle_backend="structured",
        max_dense_debug_qubits=N,
    )
    state_seconds = time.perf_counter() - state_started
    validation = _instance_validation(instance, d=d, block_sizes=block_sizes)
    if not validation["all_passed"]:
        raise RuntimeError(f"{label} instance validation failed: {validation['checks']}")
    print(f"[{label}] instance ready")

    optimization_config = build_optimization_config(
        budget=budget,
        config_seed=config_seed,
        measurement_seed=measurement_seed,
        workers=workers,
    )
    search_space = canonical_demo.build_search_space()
    config_started = time.perf_counter()
    records, rejected = _sample_valid_candidates(
        instance=instance,
        total_copies=budget,
        optimization_config=optimization_config,
        search_space=search_space,
        search_seed=config_seed,
        target_count=1,
    )
    config_seconds = time.perf_counter() - config_started
    record = records[0]
    preflight = preflight_resource_check(
        record.derived,
        n=N,
        d=d,
        total_copies=budget,
        optimization_config=optimization_config,
    )
    learner_config = candidate_to_end_to_end_config(
        record.derived,
        d=d,
        learner_seed=measurement_seed,
        total_copies=budget,
        optimization_config=optimization_config,
        return_details=False,
    )
    wiring_checks = {
        "candidate_derived_by_optimizer_parameterization": True,
        "preflight_safe_to_execute": bool(preflight.safe_to_execute),
        "execution_policy_fixed_budget_graceful": learner_config.execution_policy
        is ExecutionPolicy.FIXED_BUDGET_GRACEFUL,
        "simulation_backend_batched_counts": learner_config.simulation_backend
        == "batched_counts",
        "one_measurement_seed": optimization_config.tuning_seeds
        == (measurement_seed,),
        "no_seed_averaging": len(optimization_config.tuning_seeds) == 1,
        "max_realized_copies_matches_budget": learner_config.max_realized_copies
        == budget,
        "dense_estimator_not_requested": not learner_config.materialize_dense_estimator,
        "pilot_candidate_parallelism_not_used": True,
        "full_optimizer_not_invoked": True,
    }
    if not all(wiring_checks.values()):
        raise RuntimeError(f"{label} candidate wiring validation failed: {wiring_checks}")

    result = None
    trace_distance = None
    oracle_seconds = None
    oracle_reason = None
    evaluation_started = time.perf_counter()
    if preflight.safe_to_execute:
        result = full_cebp_tomography(instance.learner_view(), config=learner_config)
        if result.estimator_available and result.success:
            oracle_started = time.perf_counter()
            trace_distance = 0.5 * float(
                debug_end_to_end_trace_error(
                    result,
                    instance,
                    max_dense_qubits=optimization_config.max_oracle_dense_qubits,
                )
            )
            oracle_seconds = time.perf_counter() - oracle_started
        else:
            oracle_reason = "physical estimator unavailable or learner result unsuccessful"
    else:
        oracle_reason = f"preflight rejected: {preflight.reason}"
    evaluation_seconds = time.perf_counter() - evaluation_started

    if result is None:
        result_payload = {
            "success": False,
            "operational_success": False,
            "failure_stage": (
                "preflight_runtime_safety"
                if preflight.runtime_safety_rejected
                else "preflight_budget"
            ),
            "failure_reason": preflight.reason,
            "estimator_available": False,
            "execution_complete": False,
            "budget_truncated": False,
            "truncated_stages": [],
            "degradation_reason": None,
            "realized_copy_ledger": {},
            "realized_total": 0,
            "budget_feasible": True,
            "theorem_certified": False,
            "stage_theorem_flags": {},
            "structural_certificate": None,
            "end_to_end_certificate": None,
            "fixed_budget_stage_records": [],
            "stage_timings_seconds": [],
            "dominant_stage": None,
            "enumeration_diagnostics": {},
        }
    else:
        stage_timings = (
            []
            if result.performance_diagnostics is None
            else [list(item) for item in result.performance_diagnostics.stage_wall_times]
        )
        result_payload = {
            "success": bool(result.success),
            "operational_success": bool(result.estimator_available),
            "failure_stage": result.failure_stage,
            "failure_reason": result.failure_reason,
            "estimator_available": bool(result.estimator_available),
            "execution_complete": bool(result.execution_complete),
            "budget_truncated": bool(result.budget_truncated),
            "truncated_stages": list(result.truncated_stages),
            "degradation_reason": result.degradation_reason,
            "realized_copy_ledger": result.realized_copy_ledger.as_dict(),
            "realized_total": int(result.realized_total),
            "budget_feasible": int(result.realized_total) <= int(budget),
            "copies_remaining": result.copies_remaining,
            "theorem_certified": bool(result.theorem_certified),
            "stage_theorem_flags": _stage_theorem_flags(result),
            "structural_certificate": json_safe(result.structural_certificate),
            "end_to_end_certificate": json_safe(result.end_to_end_certificate),
            "fixed_budget_stage_records": json_safe(result.fixed_budget_stage_records),
            "stage_timings_seconds": stage_timings,
            "dominant_stage": _dominant_stage(stage_timings),
            "enumeration_diagnostics": _enumeration_audit(result),
        }
        if result.realized_total > budget:
            raise RuntimeError(f"{label} exceeded the configured fixed budget.")

    case_seconds = time.perf_counter() - case_started
    print(
        f"[{label}] evaluation finished: {evaluation_seconds:.3f} s, "
        f"copies={result_payload['realized_total']}"
    )
    if verbose:
        print(
            f"[{label}] estimator={result_payload['estimator_available']} "
            f"trace_distance={trace_distance}"
        )
    return {
        "label": label,
        "partition": list(block_sizes),
        "n": N,
        "d": int(d),
        "latent_block_dimensions": [2**size for size in block_sizes],
        "state_generation_mode": "random mixed",
        "latent_state_source": instance.oracle_truth.latent_state_source,
        "independent_hilbert_schmidt_generation": True,
        "block_states_argument": None,
        "pure_argument": False,
        "oracle_backend": "structured",
        "instance_seed": int(instance_seed),
        "instance_seed_ledger": json_safe(instance.seed_ledger),
        "encoder_sampling": instance.oracle_truth.encoder_sampling,
        "encoder_steps": int(instance.oracle_truth.encoder_steps),
        "configuration_seed": int(config_seed),
        "measurement_seeds_used": [int(measurement_seed)],
        "configuration_sampling_rejections_before_first_valid": int(rejected),
        "candidate_id": record.candidate_id,
        "candidate_parameters": json_safe(record.parameters),
        "derived_candidate": json_safe(record.derived),
        "search_space": json_safe(search_space),
        "preflight_resource_check": json_safe(preflight),
        "learner_config": {
            "execution_policy": learner_config.execution_policy.value,
            "simulation_backend": learner_config.simulation_backend,
            "max_realized_copies": learner_config.max_realized_copies,
            "max_reserved_copies": learner_config.max_reserved_copies,
            "max_dense_qubits": learner_config.max_dense_qubits,
            "max_enumeration_qubits": learner_config.max_enumeration_qubits,
            "max_dense_debug_qubits": learner_config.max_dense_debug_qubits,
            "materialize_dense_estimator": learner_config.materialize_dense_estimator,
            "inner_enumeration_workers": learner_config.enumeration_execution.workers,
            "fixed_budget_stage_weights": json_safe(
                learner_config.fixed_budget_stage_weights
            ),
        },
        "instance_validation": validation,
        "latent_block_audit": _latent_block_audit(instance),
        "wiring_validation": wiring_checks,
        "timings_seconds": {
            "instance_construction": state_seconds,
            "configuration_construction": config_seconds,
            "end_to_end_evaluation_including_preflight_and_oracle": evaluation_seconds,
            "oracle_trace_distance": oracle_seconds,
            "case_total": case_seconds,
        },
        "trace_distance": trace_distance,
        "oracle_loss_available": trace_distance is not None,
        "oracle_unavailable_reason": oracle_reason,
        "result": result_payload,
    }


def runtime_estimates(
    case: dict[str, Any], semantics: dict[str, Any]
) -> dict[str, Any]:
    single = float(
        case["timings_seconds"][
            "end_to_end_evaluation_including_preflight_and_oracle"
        ]
    )
    cap = int(semantics["candidate_cap"])
    tuning_count = int(semantics["tuning_seed_count"])
    holdout_count = int(semantics["holdout_seed_count"])
    progressive_lower = int(semantics["full_cap_tuning_run_lower_bound"])
    progressive_upper = int(semantics["full_cap_tuning_run_upper_bound"])
    return {
        "measured_single_evaluation_seconds": single,
        "literal_256_single_seed": {
            "formula": f"{cap} * T_single",
            "evaluation_count": cap,
            "seconds": cap * single,
        },
        "actual_current_progressive_full_cap": {
            "formula": (
                f"({progressive_lower}..{progressive_upper} cached progressive "
                f"tuning runs + {holdout_count} final-only holdout runs) * T_single"
            ),
            "tuning_evaluation_count_range": [progressive_lower, progressive_upper],
            "holdout_evaluation_count": holdout_count,
            "completed_evaluation_count_range": [
                progressive_lower + holdout_count,
                progressive_upper + holdout_count,
            ],
            "seconds_range": [
                (progressive_lower + holdout_count) * single,
                (progressive_upper + holdout_count) * single,
            ],
            "conservative_upper_seconds": (
                progressive_upper + holdout_count
            )
            * single,
        },
        "naive_no_halving_no_cache_all_tuning": {
            "formula": f"{cap} * {tuning_count} * T_single",
            "evaluation_count": cap * tuning_count,
            "seconds": cap * tuning_count * single,
            "final_holdout_modeled_separately": {
                "formula": f"{holdout_count} * T_single",
                "evaluation_count": holdout_count,
                "seconds": holdout_count * single,
            },
            "completed_seconds_including_final_holdout": (
                cap * tuning_count + holdout_count
            )
            * single,
        },
        "parallel_wall_estimate": None,
        "parallel_wall_estimate_reason": (
            "Not reported: the current optimizer explicitly runs candidate search "
            "serially and does not expose candidate/seed-level parallel execution."
        ),
    }


def duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "N/A"
    value = float(seconds)
    pieces = [f"{value:.6f} s", f"{value / 60.0:.6f} min", f"{value / 3600.0:.6f} h"]
    if value >= 86_400:
        pieces.append(f"{value / 86_400.0:.6f} days")
    return " = ".join(pieces)


def _lines_for_mapping(mapping: dict[str, Any], prefix: str = "  ") -> list[str]:
    return [
        f"{prefix}{key}: {json.dumps(json_safe(value), sort_keys=True)}"
        for key, value in mapping.items()
    ]


def render_report(data: dict[str, Any]) -> str:
    repository = data["repository"]
    benchmark = data["benchmark"]
    semantics = data["optimizer_semantics"]
    environment = data["environment"]
    verification = data["verification"]
    cases = data["cases"]
    lines = [
        "CEBP 6-QUBIT HILBERT-SCHMIDT LOCAL RUNTIME BENCHMARK",
        "=" * 72,
        "",
        "1. RUN IDENTITY",
        f"generated UTC: {data['generated_at_utc']}",
        f"generated local: {data['generated_at_local']}",
        f"repository root: {ROOT}",
        f"script: {SCRIPT_PATH}",
        f"command: {benchmark['command']}",
        f"BASE_COMMIT: {repository['base_commit']}",
        f"FINAL_COMMIT: {repository['final_commit']}",
        f"branch: {repository['branch']}",
        "",
        "2. PURPOSE AND SAFETY BOUNDARY",
        "Executed exactly one sampled valid configuration on exactly one measurement seed",
        "for each requested partition. The 256-candidate optimizer was NOT executed.",
        "No candidate was rerolled or tuned based on its tomography outcome.",
        "Candidate/configuration construction is timed separately from evaluation.",
        "",
        "3. MACHINE AND SOFTWARE ENVIRONMENT",
        *_lines_for_mapping(environment),
        "",
        "4. FIXED BENCHMARK CONTROLS",
        f"fixed budget: {benchmark['budget']} physical copies",
        "execution policy: fixed_budget_graceful",
        "simulation backend: batched_counts",
        f"measurement seed (same for both cases): {benchmark['measurement_seed']}",
        f"inner enumeration workers: {benchmark['workers']}",
        "candidate-level parallelism: disabled (cases executed sequentially)",
        "global dense estimator/state: not requested; bounded n=6 dense materialization",
        "is used only by the existing exact oracle trace-distance computation.",
        "",
        "5. CURRENT FIXED_BUDGET_MIN_ERROR SEMANTICS",
        f"candidate cap: {semantics['candidate_cap']} distinct parameter configurations",
        f"tuning/comparison seeds: {semantics['tuning_seed_count']} {tuple(semantics['tuning_measurement_seeds'])}",
        f"nested seed fidelities: {tuple(semantics['seed_fidelities'])}",
        f"post-selection-only holdout seeds: {semantics['holdout_seed_count']} {tuple(semantics['holdout_measurement_seeds'])}",
        f"candidate pools if cap reached: {tuple(semantics['candidate_pool_sizes_at_full_cap'])}",
        f"minimum improvement ratio: {semantics['relative_improvement_threshold']}",
        f"patience/minimum rounds: {semantics['improvement_patience']}/{semantics['minimum_rounds']}",
        f"candidate meaning: {semantics['candidate_semantics']}",
        f"holdout behavior: {semantics['holdout_behavior']}",
        f"progressive early stop: {semantics['progressive_early_stop']}",
        f"full-cap cached tuning-run range: {semantics['full_cap_tuning_run_lower_bound']}..{semantics['full_cap_tuning_run_upper_bound']}",
        f"per-round cached tuning-run lower bound: {tuple(semantics['full_cap_tuning_runs_by_round_lower'])}",
        f"per-round cached tuning-run upper bound: {tuple(semantics['full_cap_tuning_runs_by_round_upper'])}",
        f"naive 256 x all tuning seeds: {semantics['naive_all_candidates_all_tuning_runs']} evaluations",
        f"parallelism: {semantics['candidate_parallelism']}",
        f"policy source: {semantics['source']}",
        "",
    ]
    for index, case in enumerate(cases, start=1):
        result = case["result"]
        timing = case["timings_seconds"]
        estimates = case["runtime_estimates"]
        progressive = estimates["actual_current_progressive_full_cap"]
        naive = estimates["naive_no_halving_no_cache_all_tuning"]
        literal = estimates["literal_256_single_seed"]
        lines.extend(
            [
                f"{5 + index}. CASE {case['label']} — PARTITION {tuple(case['partition'])}",
                f"n={case['n']}; d={case['d']}; latent dimensions={tuple(case['latent_block_dimensions'])}",
                f"state mode/source: {case['state_generation_mode']} / {case['latent_state_source']}",
                f"instance seed: {case['instance_seed']}",
                f"instance child seed ledger: {json.dumps(case['instance_seed_ledger'], sort_keys=True)}",
                f"encoder: {case['encoder_sampling']}; steps={case['encoder_steps']}",
                f"configuration seed: {case['configuration_seed']}",
                f"measurement seeds used: {tuple(case['measurement_seeds_used'])}",
                f"invalid parameter draws rejected before first valid candidate: {case['configuration_sampling_rejections_before_first_valid']}",
                f"candidate id: {case['candidate_id']}",
                f"candidate parameters: {json.dumps(case['candidate_parameters'], sort_keys=True)}",
                f"derived candidate: {json.dumps(case['derived_candidate'], sort_keys=True)}",
                f"learner config: {json.dumps(case['learner_config'], sort_keys=True)}",
                f"instance validation: {json.dumps(case['instance_validation'], sort_keys=True)}",
                f"latent block audit: {json.dumps(case['latent_block_audit'], sort_keys=True)}",
                f"wiring validation: {json.dumps(case['wiring_validation'], sort_keys=True)}",
                f"preflight: {json.dumps(case['preflight_resource_check'], sort_keys=True)}",
                "",
                "Timings:",
                f"  instance construction: {duration(timing['instance_construction'])}",
                f"  configuration construction: {duration(timing['configuration_construction'])}",
                f"  end-to-end evaluation (preflight + learner + exact oracle): {duration(timing['end_to_end_evaluation_including_preflight_and_oracle'])}",
                f"  oracle component: {duration(timing['oracle_trace_distance'])}",
                f"  case total: {duration(timing['case_total'])}",
                f"  stage timings: {json.dumps(result['stage_timings_seconds'])}",
                f"  dominant top-level stage: {json.dumps(result['dominant_stage'], sort_keys=True)}",
                f"  enumeration diagnostics: {json.dumps(result['enumeration_diagnostics'], sort_keys=True)}",
                "",
                "Outcome and copies:",
                f"  success: {result['success']}",
                f"  operational success / estimator available: {result['operational_success']} / {result['estimator_available']}",
                f"  execution complete: {result['execution_complete']}",
                f"  failure stage/reason: {result['failure_stage']} / {result['failure_reason']}",
                f"  budget truncated: {result['budget_truncated']}; stages={tuple(result['truncated_stages'])}",
                f"  degradation reason: {result['degradation_reason']}",
                f"  realized total: {result['realized_total']} <= {benchmark['budget']} is {result['budget_feasible']}",
                f"  stage copy ledger: {json.dumps(result['realized_copy_ledger'], sort_keys=True)}",
                f"  fixed-budget stage records: {json.dumps(result['fixed_budget_stage_records'], sort_keys=True)}",
                f"  exact oracle trace distance: {case['trace_distance']}",
                f"  oracle available/reason: {case['oracle_loss_available']} / {case['oracle_unavailable_reason']}",
                f"  theorem certified: {result['theorem_certified']}",
                f"  theorem stage flags: {json.dumps(result['stage_theorem_flags'], sort_keys=True)}",
                f"  structural certificate: {json.dumps(result['structural_certificate'], sort_keys=True)}",
                f"  end-to-end certificate: {json.dumps(result['end_to_end_certificate'], sort_keys=True)}",
                "",
                "Runtime extrapolations from this one measured evaluation:",
                f"  literal: {literal['formula']} = {duration(literal['seconds'])}",
                f"  actual current full-cap cached workflow: {progressive['formula']}",
                f"    range: {duration(progressive['seconds_range'][0])} .. {duration(progressive['seconds_range'][1])}",
                f"    conservative upper: {duration(progressive['conservative_upper_seconds'])}",
                f"  naive no-halving/no-cache upper: {naive['formula']} = {duration(naive['seconds'])}",
                f"    final-only holdout: {naive['final_holdout_modeled_separately']['formula']} = {duration(naive['final_holdout_modeled_separately']['seconds'])}",
                f"    naive completed total: {duration(naive['completed_seconds_including_final_holdout'])}",
                f"  parallel wall estimate: {estimates['parallel_wall_estimate_reason']}",
                "",
            ]
        )

    comparison = data["comparison"]
    lines.extend(
        [
            "8. PARTITION COMPARISON AND CONCLUSION",
            f"runtime comparison: {comparison['runtime_summary']}",
            f"runtime ratio (3,3)/(2,2,2): {comparison['runtime_ratio_33_over_222']}",
            f"dominant stage (2,2,2): {json.dumps(comparison['dominant_stage_222'], sort_keys=True)}",
            f"dominant stage (3,3): {json.dumps(comparison['dominant_stage_33'], sort_keys=True)}",
            f"local pilots completed: {comparison['both_cases_executed']}",
            f"slower partition: {comparison['slower_partition']}",
            f"HPC recommendation: {comparison['hpc_recommendation']}",
            "",
            "9. LIMITATIONS",
            "This is a one-configuration, one-measurement-seed extrapolation, not a precise forecast.",
            "Candidate runtimes vary; one random configuration is not a runtime distribution.",
            "Structural recovery, grouping, localization, and truncation paths can alter runtime.",
            "The progressive 5% stopping rule may terminate before the 256-candidate cap.",
            "Semantic caching changes the exact number of learner executions within the reported range.",
            "Exact n=6 oracle trace-distance computation is included in T_single.",
            "HPC CPU architecture, scheduling, filesystem overhead, and orchestration are unmeasured.",
            "No exponential n-scaling inference is made; both cases use n=6.",
            "The current optimizer is serial across candidates/seeds, so no unsupported idealized",
            "candidate-parallel wall-time claim is included.",
            "",
            "10. INDEPENDENT VERIFICATION",
            f"script-reported total wall: {duration(verification['script_wall_seconds'])}",
            f"outer wall measurement: {duration(verification.get('outer_wall_seconds'))}",
            f"outer minus script wall: {duration(verification.get('outer_minus_script_seconds'))}",
            f"verification commands: {json.dumps(verification.get('commands', []))}",
            f"verification results: {json.dumps(verification.get('results', []))}",
            f"cleanup confirmed: {verification.get('cleanup_confirmed')}",
            f"cleanup artifacts remaining: {json.dumps(verification.get('cleanup_artifacts_remaining', []))}",
            f"task files created/changed: {json.dumps(data['task_files'])}",
            f"task-specific patch: {data['task_patch_status']}",
            "",
            "11. GIT / DIRTY-TREE AUDIT",
            "Pre-existing dirty status inferred by removing only this task's named files",
            "from the status observed at benchmark startup:",
            repository["preexisting_status_inferred"] or "(clean apart from task files)",
            "",
            "git status --short at benchmark startup:",
            repository["status_at_benchmark_start"] or "(clean)",
            "",
            "git status --short after finalization:",
            repository["status_after"] or "(clean)",
            "",
            "git diff HEAD --stat after finalization:",
            repository["diff_stat_head_after"] or "(no tracked diff)",
            "",
            "git diff HEAD --name-status after finalization:",
            repository["diff_name_status_head_after"] or "(no tracked diff)",
            "",
            "git diff --stat (unstaged) after finalization:",
            repository["diff_stat_unstaged_after"] or "(no unstaged tracked diff)",
            "",
            "git diff --name-status (unstaged) after finalization:",
            repository["diff_name_status_unstaged_after"] or "(no unstaged tracked diff)",
            "",
            "git diff --cached --stat after finalization:",
            repository["diff_stat_staged_after"] or "(no staged diff)",
            "",
            "git diff --cached --name-status after finalization:",
            repository["diff_name_status_staged_after"] or "(no staged diff)",
            "",
            "12. CLEANUP",
            "No existing simulation results or unrelated Reports artifacts were removed.",
            "Task-generated Python/test/Numba caches and temporary benchmark files were",
            "checked during finalization; see the cleanup fields above.",
            "",
        ]
    )
    return "\n".join(lines)


def _cleanup_artifacts() -> list[str]:
    names = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    suffixes = {".pyc", ".pyo", ".nbc", ".nbi"}
    found = []
    for path in ROOT.rglob("*"):
        if path.name in names or (path.is_file() and path.suffix in suffixes):
            found.append(str(path.relative_to(ROOT)))
    return sorted(found)


def _write_artifacts(data: dict[str, Any], report_path: Path, json_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(json_safe(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_report(data), encoding="utf-8")


def _comparison(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_label = {case["label"]: case for case in cases}
    left = by_label["2-2-2"]
    right = by_label["3-3"]
    t_left = float(
        left["timings_seconds"][
            "end_to_end_evaluation_including_preflight_and_oracle"
        ]
    )
    t_right = float(
        right["timings_seconds"][
            "end_to_end_evaluation_including_preflight_and_oracle"
        ]
    )
    slower = "(3,3)" if t_right > t_left else "(2,2,2)"
    return {
        "runtime_summary": (
            f"(2,2,2)={t_left:.6f}s; (3,3)={t_right:.6f}s"
        ),
        "runtime_ratio_33_over_222": None if t_left == 0.0 else t_right / t_left,
        "dominant_stage_222": left["result"]["dominant_stage"],
        "dominant_stage_33": right["result"]["dominant_stage"],
        "both_cases_executed": len(cases) == 2,
        "slower_partition": slower,
        "hpc_recommendation": (
            "This pilot's minute-scale extrapolation makes a full 6q cap run look "
            "locally feasible on runtime alone. HPC remains advisable for repeated "
            "or production studies, resource isolation, and candidate-runtime "
            "variability, but is not required by this one-run estimate."
        ),
    }


def finalize_only(args: argparse.Namespace) -> None:
    report_path = args.output_report.resolve()
    json_path = args.output_json.resolve()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    for case in data.get("cases", []):
        wiring = case.get("wiring_validation", {})
        wiring.pop("candidate_parallelism_disabled", None)
        wiring["pilot_candidate_parallelism_not_used"] = True
    data["comparison"] = _comparison(data["cases"])
    data["task_patch_status"] = (
        "Not created: all task outputs are new files, and a patch containing "
        "its own final report/JSON would be self-referential. Git diff/status "
        "metadata records their repository state."
    )
    verification = data.setdefault("verification", {})
    if args.outer_wall_seconds is not None:
        verification["outer_wall_seconds"] = float(args.outer_wall_seconds)
        verification["outer_minus_script_seconds"] = float(
            args.outer_wall_seconds - verification["script_wall_seconds"]
        )
    if args.verification_command:
        verification["commands"] = list(args.verification_command)
    if args.verification_result:
        verification["results"] = list(args.verification_result)
    remaining = _cleanup_artifacts()
    verification["cleanup_artifacts_remaining"] = remaining
    verification["cleanup_confirmed"] = not remaining
    after = git_metadata()
    data["repository"].update(
        {
            "final_commit": after["head"],
            "status_after": after["status_short"],
            "diff_stat_head_after": after["diff_stat_head"],
            "diff_name_status_head_after": after["diff_name_status_head"],
            "diff_stat_unstaged_after": after["diff_stat_unstaged"],
            "diff_name_status_unstaged_after": after["diff_name_status_unstaged"],
            "diff_stat_staged_after": after["diff_stat_staged"],
            "diff_name_status_staged_after": after["diff_name_status_staged"],
        }
    )
    _write_artifacts(data, report_path, json_path)
    print(f"report: {report_path.relative_to(ROOT)}")
    print(f"json: {json_path.relative_to(ROOT)}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument(
        "--measurement-seed", type=int, default=DEFAULT_MEASUREMENT_SEED
    )
    parser.add_argument(
        "--instance-seed-222", type=int, default=DEFAULT_INSTANCE_SEED_222
    )
    parser.add_argument(
        "--instance-seed-33", type=int, default=DEFAULT_INSTANCE_SEED_33
    )
    parser.add_argument(
        "--config-seed-222", type=int, default=DEFAULT_CONFIG_SEED_222
    )
    parser.add_argument(
        "--config-seed-33", type=int, default=DEFAULT_CONFIG_SEED_33
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--output-report", type=Path, default=DEFAULT_REPORT
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--quiet", action="store_true")
    verbosity.add_argument("--verbose", action="store_true")
    parser.add_argument("--finalize-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--outer-wall-seconds", type=float, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--verification-command", action="append", default=[], help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--verification-result", action="append", default=[], help=argparse.SUPPRESS
    )
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    for name in (
        "budget",
        "measurement_seed",
        "instance_seed_222",
        "instance_seed_33",
        "config_seed_222",
        "config_seed_33",
        "workers",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    if args.finalize_only:
        finalize_only(args)
        return
    if args.budget != canonical_demo.COPY_CEILING:
        print(
            f"warning: budget override {args.budget} differs from canonical "
            f"6q budget {canonical_demo.COPY_CEILING}",
            file=sys.stderr,
        )
    start_git = git_metadata()
    report_path = args.output_report.resolve()
    json_path = args.output_json.resolve()
    task_paths = (SCRIPT_PATH, report_path, json_path, TASK_PATHS[-1])
    semantics = optimizer_semantics()
    cases = []
    case_specs = (
        ("2-2-2", 2, (2, 2, 2), args.instance_seed_222, args.config_seed_222),
        ("3-3", 3, (3, 3), args.instance_seed_33, args.config_seed_33),
    )
    errors = []
    for label, d, block_sizes, instance_seed, config_seed in case_specs:
        try:
            case = run_case(
                label=label,
                d=d,
                block_sizes=block_sizes,
                instance_seed=instance_seed,
                config_seed=config_seed,
                measurement_seed=args.measurement_seed,
                budget=args.budget,
                workers=args.workers,
                verbose=args.verbose,
            )
            cases.append(case)
        except Exception as error:
            errors.append(f"{label}: {type(error).__name__}: {error}")
            print(f"[{label}] failed: {type(error).__name__}: {error}")
    if len(cases) != 2:
        raise RuntimeError(
            "Both cases must produce benchmark records; failures: " + "; ".join(errors)
        )
    for case in cases:
        case["runtime_estimates"] = runtime_estimates(case, semantics)
    now_utc = datetime.now(timezone.utc)
    after = git_metadata()
    measured_environment = environment_metadata(args.workers)
    script_wall = time.perf_counter() - PROCESS_STARTED
    command = shlex.join([sys.executable, str(SCRIPT_PATH), *sys.argv[1:]])
    data = {
        "schema_version": 1,
        "generated_at_utc": now_utc.isoformat(),
        "generated_at_local": datetime.now().astimezone().isoformat(),
        "repository": {
            "branch": start_git["branch"],
            "base_commit": start_git["head"],
            "final_commit": after["head"],
            "status_at_benchmark_start": start_git["status_short"],
            "preexisting_status_inferred": _status_without_task_files(
                start_git["status_short"], task_paths
            ),
            "status_after": after["status_short"],
            "diff_stat_head_after": after["diff_stat_head"],
            "diff_name_status_head_after": after["diff_name_status_head"],
            "diff_stat_unstaged_after": after["diff_stat_unstaged"],
            "diff_name_status_unstaged_after": after["diff_name_status_unstaged"],
            "diff_stat_staged_after": after["diff_stat_staged"],
            "diff_name_status_staged_after": after["diff_name_status_staged"],
        },
        "environment": measured_environment,
        "benchmark": {
            "command": command,
            "budget": int(args.budget),
            "canonical_budget": int(canonical_demo.COPY_CEILING),
            "canonical_budget_reused": args.budget == canonical_demo.COPY_CEILING,
            "measurement_seed": int(args.measurement_seed),
            "instance_seeds": {
                "2-2-2": int(args.instance_seed_222),
                "3-3": int(args.instance_seed_33),
            },
            "configuration_seeds": {
                "2-2-2": int(args.config_seed_222),
                "3-3": int(args.config_seed_33),
            },
            "workers": int(args.workers),
            "simulation_backend": "batched_counts",
            "execution_policy": "fixed_budget_graceful",
            "full_optimizer_executed": False,
            "case_execution_order": ["2-2-2", "3-3"],
        },
        "optimizer_semantics": semantics,
        "cases": cases,
        "comparison": _comparison(cases),
        "verification": {
            "script_wall_seconds": script_wall,
            "outer_wall_seconds": None,
            "outer_minus_script_seconds": None,
            "commands": [],
            "results": [],
            "cleanup_confirmed": False,
            "cleanup_artifacts_remaining": _cleanup_artifacts(),
        },
        "task_files": [
            str(SCRIPT_PATH.relative_to(ROOT)),
            str(report_path.relative_to(ROOT)),
            str(json_path.relative_to(ROOT)),
        ],
        "task_patch_status": (
            "Not created: all task outputs are new files, and a patch containing "
            "its own final report/JSON would be self-referential. Git diff/status "
            "metadata records their repository state."
        ),
    }
    # Create a placeholder so the final status query includes the report itself.
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.touch(exist_ok=True)
    after_outputs = git_metadata()
    data["repository"].update(
        {
            "final_commit": after_outputs["head"],
            "status_after": after_outputs["status_short"],
            "diff_stat_head_after": after_outputs["diff_stat_head"],
            "diff_name_status_head_after": after_outputs["diff_name_status_head"],
            "diff_stat_unstaged_after": after_outputs["diff_stat_unstaged"],
            "diff_name_status_unstaged_after": after_outputs["diff_name_status_unstaged"],
            "diff_stat_staged_after": after_outputs["diff_stat_staged"],
            "diff_name_status_staged_after": after_outputs["diff_name_status_staged"],
        }
    )
    _write_artifacts(data, report_path, json_path)
    print(f"report: {report_path.relative_to(ROOT)}")
    print(f"json: {json_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
