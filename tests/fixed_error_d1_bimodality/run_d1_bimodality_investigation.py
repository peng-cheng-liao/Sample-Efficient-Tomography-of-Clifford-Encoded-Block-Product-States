#!/usr/bin/env python3
"""Replay frozen 6q fixed-error winners for the d=1 bimodality investigation.

This is deliberately test-local diagnostic code.  Oracle data are used only
after a learner run, or in explicitly labelled substitution experiments.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main_v2 as cebp  # noqa: E402
from Optimization import (  # noqa: E402
    FixedErrorCandidateParameters,
    OptimizationConfig,
    OptimizationMode,
    OptimizationObjective,
    derive_candidate,
)
from Optimization.parameterization import candidate_to_end_to_end_config  # noqa: E402


N = 6
ERROR_TARGET = 0.05
COPY_CEILING = 10_000_000
STATE_SEEDS = {1: 6_201_001, 2: 6_202_001}
PARTITIONS = {1: (1, 1, 1, 1, 1, 1), 2: (2, 2, 2)}
D1_REPLAY_SEED_START = 6_230_001
D2_CONTROL_SEED_START = 6_240_001
BUDGET_RATIOS = (0.8, 1.0, 1.25, 1.5, 2.0)
VALIDATION_ROOT = ROOT / "tests" / "fixed_error_6q_validation"
DEFAULT_OUTPUT_ROOT = ROOT / "tests" / "fixed_error_d1_bimodality" / "outputs"
PRODUCTION_PATHS = (
    ROOT / "main_v2.py",
    ROOT / "Optimization" / "objective.py",
    ROOT / "Optimization" / "parameterization.py",
    ROOT / "Optimization" / "progressive_search.py",
    ROOT / "Optimization" / "search.py",
    ROOT / "Optimization" / "specification.py",
)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
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


def load_saved(d: int) -> dict[str, Any]:
    path = VALIDATION_ROOT / f"d{d}" / "result.json"
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("status") != "completed_feasible":
        raise RuntimeError(f"Saved d={d} validation is not completed_feasible.")
    return payload


def frozen_parameters(saved: dict[str, Any], *, budget: int | None = None):
    values = dict(saved["optimization"]["best_candidate"])
    if budget is not None:
        values["N_candidate"] = int(budget)
    return FixedErrorCandidateParameters(**values)


def build_instance(d: int):
    return cebp.random_cebp_state(
        n=N,
        d=d,
        block_sizes=PARTITIONS[d],
        pure=False,
        seed=STATE_SEEDS[d],
        oracle_backend="structured",
        max_dense_debug_qubits=N,
    )


def build_optimization_config() -> OptimizationConfig:
    return OptimizationConfig(
        total_copies=COPY_CEILING,
        max_dense_qubits=N,
        max_enumeration_qubits=N,
        max_oracle_dense_qubits=N,
        simulation_backend="batched_counts",
        objective=OptimizationObjective(
            mode=OptimizationMode.FIXED_ERROR_MIN_COPIES,
            copy_ceiling=COPY_CEILING,
            error_target=ERROR_TARGET,
            error_target_margin=0.0,
        ),
    )


def state_matrix(instance) -> np.ndarray:
    return np.asarray(instance.state.full(), dtype=np.complex128)


def state_digest(instance) -> str:
    matrix = np.ascontiguousarray(state_matrix(instance))
    return hashlib.sha256(matrix.view(np.uint8)).hexdigest()


def trace_distance_density(estimate, instance) -> float:
    target = instance.materialize_state_debug(max_qubits=N)
    target_density = target * target.dag() if target.isket else target
    difference = np.asarray(estimate.full() - target_density.full(), dtype=complex)
    return 0.5 * float(np.linalg.svd(difference, compute_uv=False).sum())


def classify_error(value: float) -> str:
    if value < 0.03:
        return "low"
    if value <= 0.07:
        return "middle"
    return "high"


def compact_json(value: Any) -> str:
    return json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"))


def _peeling_attempts(peeling) -> list[dict[str, Any]]:
    return [
        {
            "h": float(item.h),
            "inner_count": int(item.inner_count),
            "outer_count": int(item.outer_count),
            "inner_rank": int(item.inner_rank),
            "outer_rank": int(item.outer_rank),
            "spans_equal": bool(item.spans_equal),
            "accepted": bool(item.accepted),
            "reason": str(item.reason),
        }
        for item in getattr(peeling, "transcript", ())
    ]


def _recovery_sectors(recovery) -> list[dict[str, Any]]:
    return [
        {
            "sector_id": int(item.sector_id),
            "kind": item.kind,
            "members": list(item.members),
        }
        for item in getattr(recovery, "sectors", ())
    ]


def _tomography_details(tomography) -> tuple[list[dict[str, Any]], bool, float]:
    blocks: list[dict[str, Any]] = []
    any_projection = False
    max_projection = 0.0
    for estimate in getattr(tomography, "estimates", ()):
        coefficients = {str(axis): float(value) for axis, value in estimate.pauli_coefficients}
        lin = np.asarray(estimate.nu_hat_lin.full(), dtype=complex)
        projected = np.asarray(estimate.nu_hat.full(), dtype=complex)
        projection_norm = float(np.linalg.norm(lin - projected))
        projection_bound = float(estimate.numerical_projection_error_bound)
        clipped = bool(projection_norm > 1e-12 or projection_bound > 1e-12)
        any_projection = any_projection or clipped
        max_projection = max(max_projection, projection_norm, projection_bound)
        blocks.append(
            {
                "cluster": list(estimate.cluster),
                "register": list(estimate.J_C),
                "pauli_coefficients": coefficients,
                "bloch_xyz": [coefficients.get(axis) for axis in "XYZ"],
                "projection_or_clipping": clipped,
                "projection_frobenius_norm": projection_norm,
                "numerical_projection_error_bound": projection_bound,
            }
        )
    return blocks, any_projection, max_projection


def result_row(result, instance, *, seed: int, budget: int, ratio: float) -> dict[str, Any]:
    if not result.estimator_available:
        raise RuntimeError(
            f"Seed {seed} budget {budget} produced no estimator: {result.degradation_reason}"
        )
    distance = 0.5 * float(cebp.debug_end_to_end_trace_error(result, instance))
    if not math.isfinite(distance):
        raise RuntimeError(f"Non-finite trace distance for seed {seed}, budget {budget}.")
    ledger = result.realized_copy_ledger.as_dict()
    peeling = result.peeling
    recovery = result.recovery
    syndrome = result.syndrome
    tomography = result.tomography
    attempts = int(getattr(tomography, "attempts_per_setting", 0))
    accepted = tuple(int(value) for value in getattr(tomography, "accepted_per_setting", ()))
    accepted_xyz = accepted if len(accepted) == 3 else (None, None, None)
    fractions = [
        (float(value / attempts) if attempts > 0 and value is not None else None)
        for value in accepted_xyz
    ]
    blocks, projected, max_projection = _tomography_details(tomography)
    axis_means = {
        axis: (
            float(np.mean([block["pauli_coefficients"][axis] for block in blocks]))
            if blocks and all(axis in block["pauli_coefficients"] for block in blocks)
            else None
        )
        for axis in "XYZ"
    }
    finite_accepted = [int(value) for value in accepted_xyz if value is not None]
    stage_records = {
        record.stage: {
            "assigned_cap": int(record.assigned_cap),
            "realized_copies": int(record.realized_copies),
            "unused_copies": int(record.unused_copies),
            "budget_exhausted": bool(record.budget_exhausted),
            "stage_complete": bool(record.stage_complete),
            "degradation_reason": record.degradation_reason,
        }
        for record in result.fixed_budget_stage_records
    }
    peeling_attempts = _peeling_attempts(peeling) if peeling is not None else []
    accepted_attempt = next((item for item in peeling_attempts if item["accepted"]), None)
    recovery_transcript = [
        {
            "full_pauli": item.full_pauli,
            "score": float(item.score),
            "residual_pauli": item.residual_pauli,
            "dressed_pauli": item.dressed_pauli,
            "action": item.action,
            "affected_sector_ids": list(item.affected_sector_ids),
        }
        for item in getattr(recovery, "transcript", ())
    ]
    return {
        "seed": int(seed),
        "budget_ratio": float(ratio),
        "budget": int(budget),
        "trace_distance": distance,
        "error_class": classify_error(distance),
        "success": bool(result.success),
        "estimator_available": bool(result.estimator_available),
        "budget_feasible": bool(result.realized_total <= budget),
        "execution_complete": bool(result.execution_complete),
        "budget_truncated": bool(result.budget_truncated),
        "execution_branch": result.branch,
        "realized_total_copies": int(result.realized_total),
        "copy_utilization": float(result.realized_total / budget),
        "peeling_copies": int(ledger.get("peeling_bell_pool", 0)),
        "peeled_count": None if peeling is None or peeling.t is None else int(peeling.t),
        "peeling_rank": None if peeling is None else len(peeling.certified_span_basis),
        "peeled_generators": compact_json([] if peeling is None else peeling.generators),
        "peeling_accepted_h": None if accepted_attempt is None else accepted_attempt["h"],
        "peeling_threshold_summary": compact_json(peeling_attempts),
        "recovery_copies": int(ledger.get("recovery_bell_pool", 0)),
        "recovered_rank": None if recovery is None else len(recovery.recovered_span_basis),
        "recovered_sector_count": None if recovery is None else len(recovery.sectors),
        "recovered_generators": compact_json(
            [] if recovery is None else recovery.independent_axes
        ),
        "recovery_sectors": compact_json([] if recovery is None else _recovery_sectors(recovery)),
        "recovery_ranked_survivor_count": (
            None if recovery is None else int(recovery.ranked_survivor_count)
        ),
        "recovery_threshold_margin_holds": (
            None if recovery is None else bool(recovery.threshold_margin_holds)
        ),
        "recovery_ranking_gap_margin_holds": (
            None if recovery is None else bool(recovery.ranking_gap_margin_holds)
        ),
        "recovery_transcript": compact_json(recovery_transcript),
        "grouping_copies": int(ledger.get("grouping_ordinary_pool", 0)),
        "grouping_clusters": compact_json(
            [] if result.grouping is None else result.grouping.clusters
        ),
        "localization_registers": compact_json(
            [] if result.localization is None else result.localization.J_C
        ),
        "localization_aux": compact_json(
            [] if result.localization is None else result.localization.J_aux
        ),
        "syndrome_copies": int(ledger.get("syndrome_sign_pool", 0)),
        "syndrome_skipped": bool(syndrome is not None and syndrome.t == 0),
        "syndrome_bits": compact_json([] if syndrome is None else syndrome.syndrome_bits),
        "syndrome_empirical_means": compact_json(
            [] if syndrome is None else syndrome.empirical_means
        ),
        "tomography_copies": int(
            ledger.get("conditional_one_qubit_pool", ledger.get("block_tomography_pool", 0))
        ),
        "X_attempts": attempts if accepted_xyz[0] is not None else None,
        "X_accepted": accepted_xyz[0],
        "X_acceptance_fraction": fractions[0],
        "Y_attempts": attempts if accepted_xyz[1] is not None else None,
        "Y_accepted": accepted_xyz[1],
        "Y_acceptance_fraction": fractions[1],
        "Z_attempts": attempts if accepted_xyz[2] is not None else None,
        "Z_accepted": accepted_xyz[2],
        "Z_acceptance_fraction": fractions[2],
        "minimum_accepted": min(finite_accepted) if finite_accepted else None,
        "mean_accepted": float(np.mean(finite_accepted)) if finite_accepted else None,
        "acceptance_imbalance": (
            max(finite_accepted) - min(finite_accepted) if finite_accepted else None
        ),
        "empirical_X_mean_across_blocks": axis_means["X"],
        "empirical_Y_mean_across_blocks": axis_means["Y"],
        "empirical_Z_mean_across_blocks": axis_means["Z"],
        "tomography_block_estimates": compact_json(blocks),
        "projection_or_clipping": projected,
        "max_projection_metric": max_projection,
        "per_stage_copy_ledger": compact_json(ledger),
        "fixed_budget_stage_records": compact_json(stage_records),
        "performance_diagnostics": compact_json(
            []
            if result.performance_diagnostics is None
            else result.performance_diagnostics.stage_wall_times
        ),
    }


def run_one(instance, saved, config, *, seed: int, budget: int, ratio: float):
    parameters = frozen_parameters(saved, budget=budget)
    derived = derive_candidate(
        parameters,
        n=N,
        d=instance.d,
        total_copies=COPY_CEILING,
        optimization_config=config,
    )
    if int(derived.physical_copy_budget) != int(budget):
        raise RuntimeError("Derived physical budget does not match requested frozen budget.")
    learner_config = candidate_to_end_to_end_config(
        derived,
        d=instance.d,
        learner_seed=seed,
        total_copies=budget,
        optimization_config=config,
    )
    result = cebp.full_cebp_tomography(instance.learner_view(), config=learner_config)
    row = result_row(result, instance, seed=seed, budget=budget, ratio=ratio)
    return row, result, derived, learner_config


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"Refusing to write empty diagnostic CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def exact_tomography_substitution(result, instance) -> dict[str, Any]:
    if result.peeling is None or result.localization is None or result.compact_estimator is None:
        return {"supported": False, "reason": "learner result lacks structural handoff"}
    factors = []
    for cluster, register in result.localization.J_C:
        marginal = cebp.debug_exact_register_marginal(
            instance.learner_view(),
            result.peeling,
            result.localization,
            register,
            max_dense_debug_qubits=N,
        )
        factors.append((tuple(cluster), tuple(register), marginal))
    compact = replace(result.compact_estimator, register_estimates=tuple(factors))
    estimate = cebp.materialize_compact_cebp_estimator(compact, max_dense_qubits=N)
    return {
        "supported": True,
        "trace_distance": trace_distance_density(estimate, instance),
        "exact_register_count": len(factors),
    }


def exact_syndrome_substitution(result, instance) -> dict[str, Any]:
    if result.peeling is None or result.compact_estimator is None:
        return {"supported": False, "reason": "learner result lacks peeling/compact estimator"}
    expectations = tuple(
        instance.measurement_source._expectation_for_backend(generator)
        for generator in result.peeling.generators
    )
    bits = tuple(0 if value >= 0.0 else 1 for value in expectations)
    compact = replace(result.compact_estimator, syndrome_bits=bits)
    estimate = cebp.materialize_compact_cebp_estimator(compact, max_dense_qubits=N)
    return {
        "supported": True,
        "not_applicable_t_zero": len(bits) == 0,
        "oracle_bits": list(bits),
        "oracle_expectations": list(expectations),
        "learner_bits": list(result.compact_estimator.syndrome_bits),
        "trace_distance": trace_distance_density(estimate, instance),
    }


def oracle_structure_stochastic_downstream(
    instance, derived, learner_config, *, seed: int
) -> dict[str, Any]:
    """Exact structural stages followed by stochastic syndrome/tomography."""

    schedule = cebp.calibrated_end_to_end_schedule(
        instance.n, instance.d, learner_config.epsilon, learner_config.delta
    )
    seeds = cebp._stage_seed_ledger(seed)
    caps = dict(cebp.fixed_budget_resolved_stage_caps(learner_config))
    peel_cfg, recovery_cfg, grouping_cfg, syndrome_cfg, _tom_cfg = cebp._execution_configs(
        schedule, learner_config
    )
    peeling = cebp.debug_exact_certified_stabilizer_peeling(
        instance, replace(peel_cfg, M1=None)
    )
    recovery = cebp.debug_exact_rank_guided_sector_recovery(
        instance,
        peeling,
        replace(
            recovery_cfg,
            M2=None,
            allow_uncalibrated_peeling=True,
            allow_margin_failure=True,
        ),
    )
    grouping = cebp.debug_exact_hierarchical_cumulant_grouping(
        instance,
        peeling,
        recovery,
        replace(
            grouping_cfg,
            eta_test=max(float(grouping_cfg.eta_test), float(np.finfo(float).eps)),
            tau_kappa=0.0,
            allow_uncalibrated_recovery=True,
            allow_no_false_merge_margin_failure=True,
        ),
    )
    localization = cebp.localize_grouped_recovery(
        recovery,
        grouping,
        d=instance.d,
        config=cebp.LocalizationConfig(
            allow_uncertified_grouping=True,
            return_details=True,
            verify_dense_unitary=False,
            max_dense_qubits=N,
            materialize_dense_clifford=False,
            enforce_model_block_bound=False,
        ),
    )
    t = int(peeling.t)
    syndrome_cap = int(caps["syndrome"])
    syndrome_run_cfg = (
        syndrome_cfg if t == 0 else replace(syndrome_cfg, M_sgn=max(1, syndrome_cap // t))
    )
    syndrome = cebp.recover_peeling_syndrome(
        instance.learner_view(),
        peeling,
        syndrome_run_cfg,
        seed=seeds.syndrome_seed,
        prior_copy_ledger=grouping.cumulative_copy_ledger,
    )
    attempts = int(caps["tomography"]) // 3
    tomography = cebp._conditional_one_qubit_tomography(
        instance.learner_view(),
        peeling,
        localization,
        syndrome,
        schedule,
        seed=seeds.tomography_seed,
        max_dense_qubits=N,
        materialize_localized_estimator=True,
        attempts_override=attempts,
        budget_native=True,
    )
    compact = cebp._compact_estimator(peeling, localization, syndrome, tomography.estimates)
    estimate = cebp.materialize_compact_cebp_estimator(compact, max_dense_qubits=N)
    return {
        "supported": True,
        "trace_distance": trace_distance_density(estimate, instance),
        "peeling_t": int(peeling.t),
        "peeling_generators": list(peeling.generators),
        "recovery_rank": len(recovery.recovered_span_basis),
        "recovery_generators": list(recovery.independent_axes),
        "syndrome_bits": list(syndrome.syndrome_bits),
        "accepted_per_setting": list(tomography.accepted_per_setting),
        "attempts_per_setting": int(tomography.attempts_per_setting),
    }


def fully_oracle_diagnostic(instance, derived) -> dict[str, Any]:
    diagnostic = cebp.debug_exact_operational_diagnostic(
        instance,
        h_min=float(derived.h_min),
        h_max=float(derived.h_max),
        theta=float(derived.theta),
        eta_test=max(float(derived.eta_test), float(np.finfo(float).eps)),
        max_dense_debug_qubits=N,
    )
    return {
        "supported": True,
        "trace_distance": float(diagnostic.structural_trace_distance),
        "peeling_t": int(diagnostic.peeling_t),
        "peeling_generators": list(diagnostic.peeling_generators),
        "recovery_rank": len(diagnostic.recovery_span),
        "grouping_partition": json_safe(diagnostic.grouping_partition),
        "localization_registers": json_safe(diagnostic.localization_registers),
        "syndrome_bits": list(diagnostic.syndrome_bits),
    }


def run_oracles(instance, saved, config, replay_rows, *, budget: int) -> dict[str, Any]:
    ordered = sorted(replay_rows, key=lambda row: float(row["trace_distance"]))
    representatives = [ordered[0], ordered[-1]]
    payload: dict[str, Any] = {"representatives": []}
    fully_oracle: dict[str, Any] | None = None
    for representative in representatives:
        seed = int(representative["seed"])
        row, result, derived, learner_config = run_one(
            instance, saved, config, seed=seed, budget=budget, ratio=1.0
        )
        if fully_oracle is None:
            try:
                fully_oracle = fully_oracle_diagnostic(instance, derived)
            except Exception as error:  # diagnostic availability, recorded rather than hidden
                fully_oracle = {"supported": False, "reason": f"{type(error).__name__}: {error}"}
        record: dict[str, Any] = {
            "seed": seed,
            "baseline_trace_distance": float(row["trace_distance"]),
            "baseline_error_class": row["error_class"],
        }
        for name, operation in (
            ("oracle_syndrome", lambda: exact_syndrome_substitution(result, instance)),
            (
                "oracle_conditional_tomography",
                lambda: exact_tomography_substitution(result, instance),
            ),
            (
                "oracle_structure_stochastic_downstream",
                lambda: oracle_structure_stochastic_downstream(
                    instance, derived, learner_config, seed=seed
                ),
            ),
        ):
            try:
                record[name] = operation()
            except Exception as error:
                record[name] = {
                    "supported": False,
                    "reason": f"{type(error).__name__}: {error}",
                }
        record["fully_oracle_downstream"] = fully_oracle
        payload["representatives"].append(record)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d1-seeds", type=int, default=128)
    parser.add_argument("--d2-seeds", type=int, default=32)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if args.d1_seeds < 1 or args.d2_seeds < 1:
        parser.error("seed counts must be positive")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    hashes_before = production_hashes()
    config = build_optimization_config()
    saved_d1 = load_saved(1)
    saved_d2 = load_saved(2)
    d1_instance = build_instance(1)
    d2_instance = build_instance(2)
    repeated_d1 = build_instance(1)
    d1_difference = state_matrix(d1_instance) - state_matrix(repeated_d1)
    saved_d1_budget = int(saved_d1["optimization"]["refined_N_candidate"])
    saved_d2_budget = int(saved_d2["optimization"]["refined_N_candidate"])
    if frozen_parameters(saved_d1).N_candidate != saved_d1_budget:
        raise RuntimeError("Frozen d=1 candidate disagrees with refined_N_candidate.")
    if frozen_parameters(saved_d2).N_candidate != saved_d2_budget:
        raise RuntimeError("Frozen d=2 candidate disagrees with refined_N_candidate.")
    budgets = [int(round(saved_d1_budget * ratio)) for ratio in BUDGET_RATIOS]
    d1_seeds = tuple(range(D1_REPLAY_SEED_START, D1_REPLAY_SEED_START + args.d1_seeds))
    d2_seeds = tuple(range(D2_CONTROL_SEED_START, D2_CONTROL_SEED_START + args.d2_seeds))
    print(
        f"configuration: d1 N*={saved_d1_budget}, seeds={len(d1_seeds)}, "
        f"budgets={budgets}; d2 N*={saved_d2_budget}, seeds={len(d2_seeds)}",
        flush=True,
    )

    print("d1 replay/budget sensitivity: start", flush=True)
    d1_budget_rows: list[dict[str, Any]] = []
    for ratio, budget in zip(BUDGET_RATIOS, budgets):
        ratio_started = time.perf_counter()
        for seed in d1_seeds:
            row, _result, _derived, _learner_config = run_one(
                d1_instance,
                saved_d1,
                config,
                seed=seed,
                budget=budget,
                ratio=ratio,
            )
            d1_budget_rows.append(row)
        write_csv(output_root / "d1_budget_sensitivity.csv", d1_budget_rows)
        print(
            f"  ratio={ratio:g}, budget={budget}: complete in "
            f"{time.perf_counter() - ratio_started:.1f}s",
            flush=True,
        )
    d1_replay_rows = [row for row in d1_budget_rows if row["budget_ratio"] == 1.0]
    write_csv(output_root / "d1_replay_per_seed.csv", d1_replay_rows)
    print("d1 replay/budget sensitivity: end", flush=True)

    print("oracle substitutions: start", flush=True)
    oracle_payload = run_oracles(
        d1_instance, saved_d1, config, d1_replay_rows, budget=saved_d1_budget
    )
    with (output_root / "d1_oracle_substitutions.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(oracle_payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print("oracle substitutions: end", flush=True)

    print("d2 control: start", flush=True)
    d2_rows = []
    for seed in d2_seeds:
        row, _result, _derived, _learner_config = run_one(
            d2_instance,
            saved_d2,
            config,
            seed=seed,
            budget=saved_d2_budget,
            ratio=1.0,
        )
        d2_rows.append(row)
    write_csv(output_root / "d2_control_per_seed.csv", d2_rows)
    print("d2 control: end", flush=True)

    hashes_after = production_hashes()
    if hashes_before != hashes_after:
        raise RuntimeError("Production source hashes changed during diagnostic execution.")
    manifest = {
        "schema_version": 1,
        "elapsed_seconds": time.perf_counter() - started,
        "d1_replay_seed_policy": {
            "start": D1_REPLAY_SEED_START,
            "count": len(d1_seeds),
            "seeds": list(d1_seeds),
        },
        "d2_control_seed_policy": {
            "start": D2_CONTROL_SEED_START,
            "count": len(d2_seeds),
            "seeds": list(d2_seeds),
        },
        "budget_ratios": list(BUDGET_RATIOS),
        "d1_budgets": budgets,
        "d1_frozen_candidate": json_safe(saved_d1["optimization"]["best_candidate"]),
        "d1_frozen_derived_candidate": json_safe(
            saved_d1["optimization"]["best_derived_candidate"]
        ),
        "d2_frozen_candidate": json_safe(saved_d2["optimization"]["best_candidate"]),
        "d2_frozen_derived_candidate": json_safe(
            saved_d2["optimization"]["best_derived_candidate"]
        ),
        "d1_configuration": saved_d1["configuration"],
        "d2_configuration": saved_d2["configuration"],
        "state_reproducibility": {
            "saved_digest_sha256": saved_d1["state_digest_sha256"],
            "regenerated_digest_sha256": state_digest(d1_instance),
            "repeated_regenerated_digest_sha256": state_digest(repeated_d1),
            "numerically_allclose": bool(
                np.allclose(state_matrix(d1_instance), state_matrix(repeated_d1))
            ),
            "max_absolute_difference": float(np.max(np.abs(d1_difference))),
            "frobenius_norm_difference": float(np.linalg.norm(d1_difference)),
            "saved_state_matrix_available": False,
            "saved_state_comparison_note": (
                "The saved result stores a digest, not a state matrix; the state was "
                "regenerated twice from the saved configuration and compared numerically."
            ),
        },
        "production_hashes_before": hashes_before,
        "production_hashes_after": hashes_after,
        "production_files_unchanged": hashes_before == hashes_after,
    }
    with (output_root / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(manifest), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"complete: elapsed={manifest['elapsed_seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
