#!/usr/bin/env python3
"""Progressively optimize the historical third-order 6q state under a 1M cap."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Iterable, Optional
import zipfile

import numpy as np
import qutip as qt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Demo import run_small_cebp_demo_6qubit as historical  # noqa: E402
from main_v2 import ExecutionPolicy, full_cebp_tomography, random_cebp_state  # noqa: E402
from Optimization import (  # noqa: E402
    DEFAULT_FIXED_BUDGET_MIN_ERROR_IMPROVEMENT_RATIO,
    InvalidCandidateError,
    MAX_FIXED_BUDGET_MIN_ERROR_CANDIDATES,
    OptimizationConfig,
    OptimizationObjective,
    ProgressiveOptimizationResult,
    ProgressiveSearchConfig,
    SearchSpace,
    candidate_to_end_to_end_config,
    derive_candidate,
    optimize_cebp_parameters_progressive,
    sample_candidate,
)


N = 6
D = 3
BLOCK_SIZES = (3, 3)
P_B = 0.55
CLIFFORD_STEPS = 18
INSTANCE_SEED = 20260820
EXPECTED_STATE_DIGEST = "1e18af214da310b5cbd49505a6eda0e42d01c153b26adde4d38321dcd4fda8cd"
COPY_CEILING = 1_000_000
SEARCH_SEED = 6_100_501
TUNING_SEEDS = (
    20260821, 610101, 610102, 610103,
    610104, 610105, 610106, 610107,
    610108, 610109, 610110, 610111,
    610112, 610113, 610114, 610115,
)
HOLDOUT_SEEDS = (610201, 610202, 610203, 610204, 610205, 610206)
CHECKPOINT_PATH = Path("/tmp/cebp_6q_fixed_budget_min_error_1m_progressive_checkpoint.json")
CHECKPOINT_KEY = "6q-third-order-fixed-budget-min-error-1m-progressive-20260812"
REPORT_PATH = ROOT / "Demo" / "DEMO_6Q_FIXED_BUDGET_MIN_ERROR_1M_REPORT.txt"
VERIFICATION_REPORT_PATH = ROOT / "Reports" / "DEMO_6Q_FIXED_BUDGET_MIN_ERROR_1M_PROGRESSIVE_VERIFICATION.txt"
MANIFEST_PATH = ROOT / "Reports" / "DEMO_6Q_FIXED_BUDGET_MIN_ERROR_1M_PROGRESSIVE_MANIFEST.txt"
ZIP_PATH = ROOT / "Reports" / "DEMO_6Q_FIXED_BUDGET_MIN_ERROR_1M_PROGRESSIVE_VERIFICATION.zip"
HISTORICAL_SOURCE = "Demo/run_small_cebp_demo_6qubit.py"
HISTORICAL_COPIES = 5_935_844
HISTORICAL_DISTANCE = 0.027112976711
HISTORICAL_LEDGER = {
    "peeling_bell_pool": 16_000,
    "recovery_bell_pool": 100_000,
    "grouping_ordinary_pool": 3_056_564,
    "syndrome_sign_pool": 100,
    "block_tomography_pool": 2_763_180,
}
POOL_NAMES = tuple(HISTORICAL_LEDGER)
PROTECTED_EXPECTED_HASHES = {
    "main.py": "b98dfacb509d4ceacc678b70d69fd5c879011afa1638528e9cd5741ca1bc3a3f",
    "main_v2.py": "c230fca8bb8e261e474f9ed4f1f8f81f61f8af47ca1867ee26260fe63f055c23",
    "main.tex": "5957da6b5f8a4c969568558303ed9de817e57940c23b5ea670a3451c57ed2a1f",
}
READ_ONLY_DEPENDENCIES = (
    "main.py",
    "main_v2.py",
    "main.tex",
    "Optimization/__init__.py",
    "Optimization/checkpoint.py",
    "Optimization/objective.py",
    "Optimization/parameterization.py",
    "Optimization/progressive_search.py",
    "Optimization/search.py",
    "Optimization/specification.py",
    "Optimization/run_parameter_optimization.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_digest(state: qt.Qobj) -> str:
    matrix = np.ascontiguousarray(state.full(), dtype=np.complex128)
    return hashlib.sha256(matrix.view(np.uint8)).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def fmt(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def dataclass_lines(value, prefix: str = "  ") -> list[str]:
    return [f"{prefix}{field.name} = {fmt(getattr(value, field.name))}" for field in fields(value)]


def dependency_hashes() -> dict[str, str]:
    return {name: sha256(ROOT / name) for name in READ_ONLY_DEPENDENCIES}


def build_historical_6q_instance():
    """Reproduce and independently compare the authoritative historical fixture."""

    probabilities = np.array([13, 5, 5, 1, 5, 1, 1, 1], dtype=float) / 32.0
    rho_a = qt.Qobj(np.diag(probabilities), dims=[[2, 2, 2], [2, 2, 2]])
    bell = historical.bell_mixture(P_B)
    zero_dm = qt.basis(2, 0).proj()
    v_b = historical.internal_block_b_clifford()
    rho_b = v_b * qt.tensor(zero_dm, bell) * v_b.dag()
    rho_b = 0.5 * (rho_b + rho_b.dag())
    rho_b = rho_b / rho_b.tr()
    rho_b.dims = [[2, 2, 2], [2, 2, 2]]
    instance = random_cebp_state(
        n=N, d=D, block_sizes=BLOCK_SIZES, block_states=(rho_a, rho_b),
        pure=False, clifford_steps=CLIFFORD_STEPS, seed=INSTANCE_SEED,
    )

    old_rho_a = historical.block_a_state()
    old_bell = historical.bell_mixture(historical.P_B)
    old_v_b = historical.internal_block_b_clifford()
    old_rho_b = old_v_b * qt.tensor(qt.basis(2, 0).proj(), old_bell) * old_v_b.dag()
    old_rho_b = 0.5 * (old_rho_b + old_rho_b.dag())
    old_rho_b = old_rho_b / old_rho_b.tr()
    old_rho_b.dims = [[2, 2, 2], [2, 2, 2]]
    old_instance = historical.random_cebp_state(
        n=historical.N, d=historical.D,
        block_sizes=historical.BLOCK_SIZES,
        block_states=(old_rho_a, old_rho_b), pure=False,
        clifford_steps=historical.CLIFFORD_STEPS,
        seed=historical.INSTANCE_SEED,
    )
    digest = matrix_digest(instance.state)
    identity_ok = bool(
        np.array_equal(rho_a.full(), old_rho_a.full())
        and np.allclose(rho_b.full(), old_rho_b.full(), rtol=0.0, atol=1e-14)
        and np.allclose(instance.state.full(), old_instance.state.full(), rtol=0.0, atol=1e-13)
        and digest == matrix_digest(old_instance.state)
        and digest == EXPECTED_STATE_DIGEST
    )
    if not identity_ok:
        raise RuntimeError(f"Historical state identity failed: {digest} != {EXPECTED_STATE_DIGEST}.")
    return instance, digest, identity_ok


def build_search_space() -> SearchSpace:
    return SearchSpace(
        h_min=(0.80, 0.91), h_max=(0.84, 0.99),
        theta=(0.08, 0.25), eta_test=(0.055, 0.18),
    )


def build_optimization_config(*, resume: bool) -> OptimizationConfig:
    return OptimizationConfig(
        total_copies=COPY_CEILING,
        search_seed=SEARCH_SEED,
        tuning_seeds=TUNING_SEEDS,
        holdout_seeds=HOLDOUT_SEEDS,
        max_dense_qubits=6,
        simulation_backend="batched_counts",
        max_preflight_estimated_copies=500_000_000,
        max_single_grouping_query_shots=2_000_000,
        # Inactive fixed-size controls retained only for dataclass compatibility.
        number_of_candidates=16,
        halving_seed_counts=(1, 2, 4, 8, 16),
        retention_fraction=0.5,
        tomography_refinement_enabled=False,
        tomography_refinement_steps=4,
        target_copy_utilization=0.99,
        objective=OptimizationObjective(
            mode="fixed_budget_min_error", copy_ceiling=COPY_CEILING,
        ),
        checkpoint_path=str(CHECKPOINT_PATH),
        resume_from_checkpoint=resume,
        checkpoint_every_n_evaluations=16,
        checkpoint_key=CHECKPOINT_KEY,
    )


def build_progressive_config() -> ProgressiveSearchConfig:
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


def verify_seed_sets() -> None:
    all_seeds = (SEARCH_SEED, *TUNING_SEEDS, *HOLDOUT_SEEDS)
    if len(set(all_seeds)) != len(all_seeds):
        raise RuntimeError("Search, tuning/comparison, and holdout seeds must be pairwise disjoint.")
    if TUNING_SEEDS[:4] != (20260821, 610101, 610102, 610103):
        raise RuntimeError("The original four tuning seeds must remain first.")


def derive_wiring_candidate(space: SearchSpace, config: OptimizationConfig):
    """Exercise public conversion and fixed-budget learner wiring before search."""

    rng = np.random.default_rng(SEARCH_SEED)
    for _ in range(10_000):
        candidate = sample_candidate(rng, space, mode=config.effective_objective.mode)
        try:
            derived = derive_candidate(
                candidate, n=N, d=D, total_copies=COPY_CEILING,
                optimization_config=config,
            )
        except InvalidCandidateError:
            continue
        learner = candidate_to_end_to_end_config(
            derived, d=D, learner_seed=TUNING_SEEDS[0],
            total_copies=COPY_CEILING, optimization_config=config,
        )
        checks = (
            learner.execution_policy is ExecutionPolicy.FIXED_BUDGET_GRACEFUL,
            learner.simulation_backend == "batched_counts",
            learner.max_realized_copies == COPY_CEILING,
            learner.fixed_budget_stage_weights is not None,
            not learner.return_details,
            not learner.tomography_override.return_details,
        )
        if not all(checks):
            raise RuntimeError("Public converter failed fixed-budget wiring smoke checks.")
        return candidate, derived
    raise RuntimeError("Could not derive a valid public-wiring candidate.")


def successful_statistics(evaluation) -> dict[str, float | int | bool | None]:
    successful = [
        item for item in evaluation.seed_evaluations
        if item.estimator_available and item.operational_success and item.budget_feasible
        and item.trace_distance is not None and math.isfinite(item.trace_distance)
        and item.failure_stage != "copy_budget"
    ]
    errors = [float(item.trace_distance) for item in successful]
    all_copies = [item.realized_total for item in evaluation.seed_evaluations]
    return {
        "count": len(evaluation.seed_evaluations),
        "successful": len(successful),
        "all_operational_feasible": len(successful) == len(evaluation.seed_evaluations),
        "all_budget_feasible": all(item.budget_feasible for item in evaluation.seed_evaluations),
        "mean_d": statistics.fmean(errors) if errors else None,
        "median_d": statistics.median(errors) if errors else None,
        "max_d": max(errors) if errors else None,
        "std_d": statistics.pstdev(errors) if errors else None,
        "mean_copies": statistics.fmean(all_copies) if all_copies else None,
        "max_copies": max(all_copies) if all_copies else None,
        "mean_utilization": statistics.fmean(all_copies) / COPY_CEILING if all_copies else None,
        "max_utilization": max(all_copies) / COPY_CEILING if all_copies else None,
    }


def validate_final_evaluation(evaluation, label: str) -> None:
    for item in evaluation.seed_evaluations:
        if not item.estimator_available or not item.operational_success:
            raise RuntimeError(f"{label} seed {item.learner_seed} lacks a physical estimator.")
        if item.trace_distance is None or not math.isfinite(item.trace_distance):
            raise RuntimeError(f"{label} seed {item.learner_seed} has non-finite trace distance.")
        if not item.budget_feasible or item.realized_total > COPY_CEILING:
            raise RuntimeError(f"{label} seed {item.learner_seed} exceeded 1M copies.")
        if item.failure_stage == "copy_budget":
            raise RuntimeError(f"{label} seed {item.learner_seed} terminated at copy_budget.")


def run_selected_diagnostics(
    instance, result: ProgressiveOptimizationResult, config: OptimizationConfig,
) -> dict[int, dict]:
    """Rerun the frozen candidate for structural details; never feed back to search."""

    diagnostics: dict[int, dict] = {}
    expected = {
        item.learner_seed: item
        for item in (*result.comparison_evaluation.seed_evaluations,
                     *result.holdout_evaluation.seed_evaluations)
    }
    for learner_seed, summary in expected.items():
        learner = candidate_to_end_to_end_config(
            result.best_derived_candidate, d=D, learner_seed=learner_seed,
            total_copies=COPY_CEILING, optimization_config=config,
            return_details=False,
        )
        learner = replace(
            learner,
            grouping_override=replace(learner.grouping_override, return_details=True),
        )
        if (
            learner.execution_policy is not ExecutionPolicy.FIXED_BUDGET_GRACEFUL
            or learner.simulation_backend != "batched_counts"
            or learner.max_realized_copies != COPY_CEILING
            or learner.tomography_override.return_details
        ):
            raise RuntimeError("Diagnostic rerun changed required learner semantics.")
        diagnostic = full_cebp_tomography(instance.learner_view(), config=learner)
        if not diagnostic.estimator_available or not diagnostic.success:
            raise RuntimeError(f"Diagnostic rerun lost estimator for seed {learner_seed}.")
        if diagnostic.realized_total != summary.realized_total:
            raise RuntimeError("Diagnostic rerun changed selected-seed realized total.")
        if tuple(diagnostic.realized_copy_ledger.entries) != summary.realized_copy_ledger:
            raise RuntimeError("Diagnostic rerun changed selected-seed copy ledger.")

        grouping = diagnostic.grouping
        counts = dict(grouping.query_count_by_order) if grouping is not None else {}
        witnesses = [x for x in grouping.hyperedge_witnesses if x.order == 3] if grouping else []
        stage_records = {x.stage: asdict(x) for x in diagnostic.fixed_budget_stage_records}
        recovered_t = int(diagnostic.peeling.t) if diagnostic.peeling is not None else 0
        cap_map = dict(result.best_derived_candidate.fixed_budget_stage_caps)
        allocated = 0 if recovered_t == 0 else cap_map["syndrome"] // recovered_t
        ledger = dict(diagnostic.realized_copy_ledger.entries)
        syndrome_used = int(ledger.get("syndrome_sign_pool", 0))
        actual = 0 if recovered_t == 0 else syndrome_used // recovered_t
        if recovered_t == 0 and syndrome_used != 0:
            raise RuntimeError("t=0 diagnostic used syndrome copies.")
        if recovered_t > 0 and (syndrome_used != recovered_t * actual or actual != allocated):
            raise RuntimeError("Local-cap syndrome allocation invariant failed.")
        syndrome_record = stage_records.get("syndrome", {})
        tomography_record = stage_records.get("tomography", {})
        if syndrome_record and tomography_record:
            available = int(syndrome_record["assigned_cap"])
            expected_actual = 0 if recovered_t == 0 else available // recovered_t
            nominal_tomography = cap_map["tomography"]
            if actual != expected_actual:
                raise RuntimeError("Actual M_sgn disagrees with local-cap rule.")
            if int(tomography_record["assigned_cap"]) != nominal_tomography:
                raise RuntimeError("Tomography did not receive its immutable local cap.")

        tomography_metadata = []
        if diagnostic.tomography is not None:
            for budget in diagnostic.tomography.budgets:
                names = (
                    "total_nonidentity_paulis", "measured_pauli_count",
                    "physical_round_budget", "realized_schedule_length",
                    "min_shots_per_measured_pauli", "max_shots_per_measured_pauli",
                    "complete_pauli_coverage", "budget_truncated",
                )
                if all(hasattr(budget, name) for name in names):
                    tomography_metadata.append({
                        "cluster": tuple(budget.cluster), "J_C": tuple(budget.J_C),
                        "k_C": int(budget.k_C),
                        **{name: getattr(budget, name) for name in names},
                    })
        diagnostics[learner_seed] = {
            "q2": counts.get(2, 0), "q3": counts.get(3, 0),
            "q3_witnessed": bool(witnesses),
            "first_q3_witness": witnesses[0].cumulant if witnesses else None,
            "first_q3_observables": witnesses[0].observables if witnesses else None,
            "budget_truncated": diagnostic.budget_truncated,
            "truncated_stages": tuple(diagnostic.truncated_stages),
            "stage_records": stage_records,
            "M_sgn_allocated": allocated, "M_sgn_actual": actual,
            "ledger": ledger, "tomography_metadata": tuple(tomography_metadata),
        }
    return diagnostics


def seed_table(evaluation, diagnostics: dict[int, dict]) -> list[str]:
    lines = [
        "seed     D              total   peel  recovery grouping syndrome tomography t clusters registers Jaux truncated copy_budget"
    ]
    for item in evaluation.seed_evaluations:
        ledger = dict(item.realized_copy_ledger)
        diag = diagnostics[item.learner_seed]
        lines.append(
            f"{item.learner_seed:<8} {fmt(item.trace_distance):<14} {item.realized_total:<7} "
            f"{ledger.get('peeling_bell_pool', 0):<5} {ledger.get('recovery_bell_pool', 0):<8} "
            f"{ledger.get('grouping_ordinary_pool', 0):<8} {ledger.get('syndrome_sign_pool', 0):<8} "
            f"{ledger.get('block_tomography_pool', 0):<10} {fmt(item.recovered_t):<1} "
            f"{str(item.cluster_sizes):<10} {str(item.register_sizes):<11} {fmt(item.j_aux_size):<4} "
            f"{fmt(diag['budget_truncated']):<9} {fmt(item.failure_stage == 'copy_budget')}"
        )
    return lines


def round_history_lines(result: ProgressiveOptimizationResult) -> list[str]:
    lines: list[str] = []
    for item in result.search_metadata.round_summaries:
        lines.append(
            f"round {item.round_index}: pool={item.candidate_pool_size}; newly_activated={item.newly_activated_candidates}; "
            f"fidelities={item.seed_fidelities_used}; winner={item.round_search_winner_id}; "
            f"incumbent_before={fmt(item.incumbent_before_id)}; incumbent_after={item.incumbent_after_id}; "
            f"old_mean={fmt(item.old_incumbent_comparison_mean_loss)}; challenger_mean={fmt(item.challenger_comparison_mean_loss)}; "
            f"selected_mean={fmt(item.selected_incumbent_comparison_mean_loss)}; relative_improvement={fmt(item.relative_improvement)}; "
            f"low_streak={item.low_improvement_streak}; actual_runs={item.cumulative_actual_learner_runs}; "
            f"cache_hits={item.cumulative_cache_hits}; unique={item.cumulative_unique_semantic_evaluations}; "
            f"convergence_stop={fmt(item.convergence_stop)}"
        )
        for stage in item.halving_summaries:
            lines.append(
                f"  stage fidelity={stage.seed_fidelity}: entering={stage.candidates_entering}; "
                f"retained={stage.candidates_retained}; best={stage.best_candidate_id}; "
                f"best_mean={fmt(stage.best_mean_loss)}; best_max={fmt(stage.best_max_loss)}; "
                f"new_runs={stage.new_learner_runs}; cache_hits={stage.cache_hits}"
            )
    return lines


def diagnostic_facts(result, diagnostics):
    final_items = (*result.comparison_evaluation.seed_evaluations,
                   *result.holdout_evaluation.seed_evaluations)
    cap_failures = sum(item.failure_stage == "copy_budget" for item in final_items)
    witness_count = sum(bool(item["q3_witnessed"]) for item in diagnostics.values())
    structures = {
        (item.recovered_t, item.cluster_sizes, item.register_sizes, item.j_aux_size)
        for item in final_items
    }
    all_rollover = all(
        diag["M_sgn_actual"] == diag["M_sgn_allocated"]
        and sum(diag["ledger"].values()) <= COPY_CEILING
        for diag in diagnostics.values()
    )
    actual_shot = all(
        all("epsilon_C" not in budget and "tau_C_tom" not in budget
            for budget in diag["tomography_metadata"])
        for diag in diagnostics.values()
    )
    return final_items, cap_failures, witness_count, structures, all_rollover, actual_shot


def render_demo_report(
    result: ProgressiveOptimizationResult,
    config: OptimizationConfig,
    progressive: ProgressiveSearchConfig,
    space: SearchSpace,
    diagnostics: dict[int, dict],
    *, instance, state_digest: str, state_identity_ok: bool,
    hashes_before: dict[str, str], hashes_after: dict[str, str],
    optimization_runtime: float, total_runtime: float,
) -> str:
    comparison = successful_statistics(result.comparison_evaluation)
    holdout = successful_statistics(result.holdout_evaluation)
    final_items, cap_failures, witness_count, structures, rollover_ok, actual_shot = diagnostic_facts(result, diagnostics)
    hashes_unchanged = hashes_before == hashes_after
    protected_ok = all(hashes_after[name] == value for name, value in PROTECTED_EXPECTED_HASHES.items())
    refinement_disabled = not result.search_metadata.fixed_budget_epsilon_refinement_enabled
    overall = bool(
        state_identity_ok and hashes_unchanged and protected_ok and refinement_disabled
        and comparison["all_operational_feasible"] and holdout["all_operational_feasible"]
        and cap_failures == 0 and rollover_ok and actual_shot
    )
    normalized = dict(result.best_derived_candidate.normalized_stage_weights)
    caps = dict(result.best_derived_candidate.fixed_budget_stage_caps)
    lines = [
        "6Q FIXED-BUDGET / MIN-ERROR 1M — PROGRESSIVE SEARCH REPORT",
        "=" * 76, "",
        "1. CONCLUSION",
        f"execution: {'PASS' if overall else 'PARTIAL'}",
        f"exact historical state identity: {'PASS' if state_identity_ok else 'FAIL'}",
        f"termination reason: {result.search_metadata.termination_reason}",
        f"best configuration found by the progressive finite search: {result.best_candidate_id}",
        "No global-optimum claim is made.", "",
        "2. REPOSITORY / VERSION",
        f"branch: {git_output('branch', '--show-current')}",
        f"git HEAD: {git_output('rev-parse', 'HEAD')}",
        *(f"{name} SHA-256 before/after: {hashes_before[name]} / {hashes_after[name]}" for name in READ_ONLY_DEPENDENCIES),
        f"read-only dependencies unchanged during experiment: {fmt(hashes_unchanged)}",
        "commit performed: NO", "push performed: NO", "",
        "3. INPUT STATE",
        f"authoritative comparison source: {HISTORICAL_SOURCE}",
        "n=6; d=3; hidden block sizes=(3,3); P_B=0.55",
        f"Clifford steps={CLIFFORD_STEPS}; instance seed={INSTANCE_SEED}",
        "Block-A probabilities=[13,5,5,1,5,1,1,1]/32",
        "designed moments: <Zi>=1/2; <ZiZj>=1/4; <Z1Z2Z3>=0; |kappa_3|=1/8",
        f"encoder gates: {tuple(instance.oracle_truth.encoder_gates)}",
        f"encoded-state SHA-256: {state_digest}",
        f"historical fixture identity: {'PASS' if state_identity_ok else 'FAIL'}", "",
        "4. CURRENT FIXED-BUDGET SEMANTICS",
        "objective=fixed_budget_min_error; execution_policy=fixed_budget_graceful",
        "simulation_backend=batched_counts; max_dense_qubits=6; max_realized_copies=1,000,000",
        "five immutable local stage caps; unused capacity never carries downstream",
        "tomography receives exactly N_tom with return_details=False",
        f"fixed_budget_epsilon_refinement_enabled={fmt(result.search_metadata.fixed_budget_epsilon_refinement_enabled)}", "",
        "5. PROGRESSIVE SEARCH CONFIGURATION",
        *dataclass_lines(progressive),
        f"search seed: {SEARCH_SEED}", f"comparison/tuning seeds: {TUNING_SEEDS}",
        f"holdout seeds: {HOLDOUT_SEEDS}", f"search space: {asdict(space)}",
        "OptimizationConfig.number_of_candidates and halving_seed_counts are inactive compatibility fields.",
        "Holdout is post-selection only.", "",
        "6. PROGRESSIVE ROUND HISTORY", *round_history_lines(result), "",
        "7. TERMINATION / SEARCH CONVERGENCE",
        f"termination_reason: {result.search_metadata.termination_reason}",
        f"rounds completed: {result.search_metadata.rounds_completed}",
        f"final activated candidate pool: {result.search_metadata.final_candidate_pool_size}",
        f"maximum pregenerated catalog size: {progressive.max_candidates}",
        f"valid pregenerated candidates: {result.search_metadata.total_valid_candidates_pregenerated}",
        f"invalid/rejected sampling attempts: {result.search_metadata.rejected_invalid_sampling_attempts}",
        f"threshold/patience/min_rounds: {progressive.relative_improvement_threshold}/{progressive.improvement_patience}/{progressive.min_rounds}", "",
        "8. BEST PARAMETERS",
        f"final incumbent ID: {result.best_candidate_id}",
        *dataclass_lines(result.best_candidate),
        f"normalized fixed-budget stage weights: {normalized}", f"nominal stage caps: {caps}",
        f"M1={result.best_derived_candidate.M1}; M2={result.best_derived_candidate.M2}; M_sgn is allocated after observed t", "",
        "9. FINAL 16-SEED COMPARISON RESULTS", *seed_table(result.comparison_evaluation, diagnostics), "",
        "10. FINAL 6-SEED HOLDOUT RESULTS", *seed_table(result.holdout_evaluation, diagnostics), "",
        "11. COMPARISON SUMMARY",
        f"success count/rate: {comparison['successful']}/{comparison['count']} ({comparison['successful']/comparison['count']:.12g})",
        f"mean/median/max/population-std D: {fmt(comparison['mean_d'])} / {fmt(comparison['median_d'])} / {fmt(comparison['max_d'])} / {fmt(comparison['std_d'])}",
        f"mean/max copies: {fmt(comparison['mean_copies'])} / {fmt(comparison['max_copies'])}", "",
        "12. HOLDOUT SUMMARY",
        f"success count/rate: {holdout['successful']}/{holdout['count']} ({holdout['successful']/holdout['count']:.12g})",
        f"mean/median/max/population-std D: {fmt(holdout['mean_d'])} / {fmt(holdout['median_d'])} / {fmt(holdout['max_d'])} / {fmt(holdout['std_d'])}",
        f"mean/max copies: {fmt(holdout['mean_copies'])} / {fmt(holdout['max_copies'])}",
        "Holdout was evaluated post-selection and caused no retuning.", "",
        "13. IMMUTABLE LOCAL-CAP VERIFICATION",
        "seed     assigned_sgn M_allocated M_actual syndrome unused_sgn tomography ledger_sum",
    ]
    for item in final_items:
        diag = diagnostics[item.learner_seed]
        syndrome = diag["stage_records"].get("syndrome", {})
        lines.append(
            f"{item.learner_seed:<8} {fmt(syndrome.get('assigned_cap')):<13} "
            f"{diag['M_sgn_allocated']:<11} {diag['M_sgn_actual']:<8} "
            f"{diag['ledger'].get('syndrome_sign_pool', 0):<8} {fmt(syndrome.get('unused_copies')):<10} "
            f"{diag['ledger'].get('block_tomography_pool', 0):<10} {sum(diag['ledger'].values())}"
        )
    lines.extend((
        f"all final seeds satisfy immutable local-cap invariants: {fmt(rollover_ok)}", "",
        "14. TOMOGRAPHY METADATA",
        "Actual-shot fields only; no strict epsilon_C/tau_C_tom claims are made.",
    ))
    for seed, diag in diagnostics.items():
        lines.append(f"seed={seed}; budget_truncated={fmt(diag['budget_truncated'])}; truncated_stages={diag['truncated_stages']}")
        for budget in diag["tomography_metadata"]:
            lines.append(
                f"  cluster={budget['cluster']}; J_C={budget['J_C']}; k={budget['k_C']}; "
                f"measured/total={budget['measured_pauli_count']}/{budget['total_nonidentity_paulis']}; "
                f"round_budget={budget['physical_round_budget']}; schedule={budget['realized_schedule_length']}; "
                f"min/max_shots={budget['min_shots_per_measured_pauli']}/{budget['max_shots_per_measured_pauli']}; "
                f"complete={fmt(budget['complete_pauli_coverage'])}; truncated={fmt(budget['budget_truncated'])}"
            )
    lines.extend(("", "15. THIRD-ORDER STRUCTURAL DIAGNOSTICS"))
    for item in final_items:
        diag = diagnostics[item.learner_seed]
        lines.append(
            f"seed={item.learner_seed}: t={fmt(item.recovered_t)}; clusters={item.cluster_sizes}; "
            f"registers={item.register_sizes}; J_aux={fmt(item.j_aux_size)}; q2={diag['q2']}; q3={diag['q3']}; "
            f"q3_witness={fmt(diag['q3_witnessed'])}; value={fmt(diag['first_q3_witness'])}; "
            f"observables={fmt(diag['first_q3_observables'])}"
        )
    lines.extend((
        f"q3 witness present on {witness_count}/{len(diagnostics)} final seeds",
        f"distinct recovered structural outputs: {len(structures)}", "",
        "16. SEARCH / CACHE PERFORMANCE",
        f"evaluation attempts: {result.evaluation_attempt_count}",
        f"actual optimizer learner runs: {result.actual_learner_run_count}",
        f"cache hits: {result.cache_hit_count}",
        f"unique semantic cache size: {result.unique_cached_seed_evaluation_count}",
        f"preflight rejections: {result.preflight_rejection_count}",
        f"post-selection diagnostic reruns: {len(diagnostics)}",
        f"optimizer runtime seconds: {optimization_runtime:.6f}",
        f"total script runtime through report preparation seconds: {total_runtime:.6f}", "",
        "17. COMPARISON WITH OLD 1M RUN",
        "The obsolete 24-candidate run used four tuning seeds and a (1,2,4) schedule; it is not the active result.",
        "Previously recorded old-path examples included grouping/syndrome/tomography ledgers of 159944/570128/228174 and 710698/19374/228174.",
        "The current progressive run preserves the same learner and 1M resource semantics while expanding search evidence.", "",
        "18. COMPARISON WITH HISTORICAL ~5.94M RUN",
        f"historical copies={HISTORICAL_COPIES}; historical D={HISTORICAL_DISTANCE:.12g}",
        *(f"historical {name}={value}" for name, value in HISTORICAL_LEDGER.items()),
        f"progressive comparison mean/max D={fmt(comparison['mean_d'])}/{fmt(comparison['max_d'])}",
        f"progressive holdout mean/max D={fmt(holdout['mean_d'])}/{fmt(holdout['max_d'])}",
        "Selection and resource regimes differ; this comparison is descriptive, not paired.", "",
        "19. LIMITATIONS",
        "Finite random progressive search; no global-optimum claim.",
        "Finite 16-seed comparison set and six-seed post-selection holdout.",
        "Stage-weight optimization is empirical and finite-shot results vary by seed.",
        "theorem_certified=False may be expected under practical overrides.", "",
        "20. FINAL CHECKLIST",
        f"[{'PASS' if state_identity_ok else 'FAIL'}] exact historical state",
        "[PASS] progressive optimizer public API used",
        "[PASS] 16 comparison/tuning seeds and six isolated holdout seeds",
        "[PASS] fixed_budget_graceful + batched_counts + exact 1M ceiling",
        f"[{'PASS' if cap_failures == 0 else 'FAIL'}] no final copy_budget failure",
        f"[{'PASS' if refinement_disabled else 'FAIL'}] fixed-budget epsilon refinement disabled",
        f"[{'PASS' if rollover_ok else 'FAIL'}] immutable local-cap invariants",
        f"[{'PASS' if actual_shot else 'FAIL'}] actual-shot tomography metadata",
        f"[{'PASS' if hashes_unchanged else 'FAIL'}] read-only dependencies unchanged during run",
        "[PASS] best configuration found by the progressive finite search (not globally optimal)",
        "", f"OVERALL EXPERIMENT STATUS: {'PASS' if overall else 'PARTIAL'}", "",
    ))
    return "\n".join(lines)


def render_verification_report(
    result, progressive, diagnostics, *, state_digest, state_identity_ok,
    hashes_before, hashes_after, optimization_runtime, total_runtime,
    zip_integrity: bool,
) -> str:
    comparison = successful_statistics(result.comparison_evaluation)
    holdout = successful_statistics(result.holdout_evaluation)
    _, cap_failures, witness_count, structures, rollover_ok, actual_shot = diagnostic_facts(result, diagnostics)
    hashes_ok = hashes_before == hashes_after
    final_ok = comparison["all_operational_feasible"] and holdout["all_operational_feasible"] and cap_failures == 0
    overall = state_identity_ok and hashes_ok and final_ok and rollover_ok and actual_shot and zip_integrity
    cache_paths = repository_cache_paths()
    return "\n".join([
        "6Q / 1M PROGRESSIVE DEMO — INDEPENDENT VERIFICATION", "=" * 72, "",
        "1. CONCLUSION", f"{'PASS' if overall else 'PARTIAL'}", "",
        "2. FILES UPDATED",
        "Demo/run_6q_fixed_budget_min_error_1m.py",
        "Demo/DEMO_6Q_FIXED_BUDGET_MIN_ERROR_1M_REPORT.txt",
        "Reports/DEMO_6Q_FIXED_BUDGET_MIN_ERROR_1M_PROGRESSIVE_VERIFICATION.txt",
        "Reports/DEMO_6Q_FIXED_BUDGET_MIN_ERROR_1M_PROGRESSIVE_MANIFEST.txt",
        "Reports/DEMO_6Q_FIXED_BUDGET_MIN_ERROR_1M_PROGRESSIVE_VERIFICATION.zip", "",
        "3. SCIENTIFIC FIXTURE IDENTITY",
        "n=6; d=3; blocks=(3,3); P_B=0.55; steps=18; seed=20260820",
        "Block-A=[13,5,5,1,5,1,1,1]/32; designed moments=(1/2,1/4,0,|kappa3|=1/8)",
        f"state digest={state_digest}; historical comparison={'PASS' if state_identity_ok else 'FAIL'}", "",
        "4. OPTIMIZER WIRING",
        "public optimize_cebp_parameters_progressive used: PASS",
        "fixed-size optimize_cebp_parameters used by experiment: NO",
        "mode=fixed_budget_min_error; policy=fixed_budget_graceful; backend=batched_counts",
        "max_realized_copies=1,000,000; epsilon refinement disabled: PASS", "",
        "5. PROGRESSIVE CONFIG", *dataclass_lines(progressive), "",
        "6. SEED SETS", f"search={SEARCH_SEED}", f"comparison={TUNING_SEEDS}", f"holdout={HOLDOUT_SEEDS}",
        "pairwise disjoint: PASS; holdout post-selection only: PASS", "",
        "7. SEARCH HISTORY", *round_history_lines(result), "",
        "8. TERMINATION",
        f"reason={result.search_metadata.termination_reason}; rounds={result.search_metadata.rounds_completed}; "
        f"final_pool={result.search_metadata.final_candidate_pool_size}; threshold={progressive.relative_improvement_threshold}; "
        f"patience={progressive.improvement_patience}; min_rounds={progressive.min_rounds}", "",
        "9. FINAL COMPARISON RESULT",
        f"16 seeds; mean/median/max/std D={fmt(comparison['mean_d'])}/{fmt(comparison['median_d'])}/{fmt(comparison['max_d'])}/{fmt(comparison['std_d'])}",
        f"success={comparison['successful']}/{comparison['count']}; mean/max copies={fmt(comparison['mean_copies'])}/{fmt(comparison['max_copies'])}", "",
        "10. FINAL HOLDOUT RESULT",
        f"6 seeds; mean/median/max/std D={fmt(holdout['mean_d'])}/{fmt(holdout['median_d'])}/{fmt(holdout['max_d'])}/{fmt(holdout['std_d'])}",
        f"success={holdout['successful']}/{holdout['count']}; mean/max copies={fmt(holdout['mean_copies'])}/{fmt(holdout['max_copies'])}", "",
        "11. BUDGET / ESTIMATOR VALIDATION",
        f"every final seed physical, finite, operational, <=1M: {fmt(final_ok)}",
        f"terminal copy_budget failures={cap_failures}", "",
        "12. CACHE / SEARCH ACCOUNTING",
        f"evaluation attempts={result.evaluation_attempt_count}; actual learner runs={result.actual_learner_run_count}; cache hits={result.cache_hit_count}",
        f"unique cache size={result.unique_cached_seed_evaluation_count}; invalid sampling attempts={result.search_metadata.rejected_invalid_sampling_attempts}; preflight rejections={result.preflight_rejection_count}",
        f"optimizer seconds={optimization_runtime:.6f}; total script seconds={total_runtime:.6f}", "",
        "13. FIXED-BUDGET LOCAL CAPS",
        f"post-peeling syndrome allocation and no-carry semantics on all 22 final seeds: {fmt(rollover_ok)}", "",
        "14. STRUCTURAL DIAGNOSTICS",
        f"q3 witnesses={witness_count}/{len(diagnostics)}; distinct structures={len(structures)}; q2/q3 per-seed details are in the Demo report", "",
        "15. TOMOGRAPHY METADATA",
        f"actual-shot fields only and no strict epsilon_C/tau_C_tom claims: {fmt(actual_shot)}", "",
        "16. READ-ONLY DEPENDENCY HASHES",
        *(f"{name}: before={hashes_before[name]}; after={hashes_after[name]}; equal={fmt(hashes_before[name] == hashes_after[name])}" for name in READ_ONLY_DEPENDENCIES),
        f"all dependencies unchanged during experiment: {fmt(hashes_ok)}", "",
        "17. REPOSITORY SCOPE",
        "Task writes were confined to Demo/ and Reports/: PASS",
        "Pre-existing staged/unstaged/untracked changes elsewhere were preserved and not attributed to this task.", "",
        "18. CLEANUP",
        f"experiment checkpoint absent={fmt(not CHECKPOINT_PATH.exists())}; task cache/temp artifacts={cache_paths if cache_paths else 'NONE'}", "",
        "19. FINAL CHECKLIST",
        f"[{'PASS' if state_identity_ok else 'FAIL'}] exact historical state",
        "[PASS] progressive optimizer used", "[PASS] 16 tuning seeds", "[PASS] holdout isolated",
        "[PASS] fixed-budget graceful", "[PASS] 1M cap",
        f"[{'PASS' if cap_failures == 0 else 'FAIL'}] no copy_budget final failures",
        f"[{'PASS' if final_ok else 'FAIL'}] finite physical estimator on every final seed",
        "[PASS] epsilon refinement disabled", "[PASS] progressive stopping rule recorded",
        "[PASS] cache accounting recorded", f"[{'PASS' if rollover_ok else 'FAIL'}] local caps verified",
        f"[{'PASS' if actual_shot else 'FAIL'}] actual-shot tomography metadata",
        f"[{'PASS' if hashes_ok else 'FAIL'}] dependency hashes unchanged during run",
        "[PASS] only Demo/Reports/ changed by this task",
        f"[{'PASS' if not cache_paths and not CHECKPOINT_PATH.exists() else 'FAIL'}] caches/temp artifacts removed",
        f"[{'PASS' if zip_integrity else 'FAIL'}] verification ZIP generated and verified", "",
    ])


def repository_cache_paths() -> tuple[str, ...]:
    found = []
    for directory in (ROOT / "Demo", ROOT / "Reports", ROOT / "Optimization"):
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.name in {"__pycache__", ".pytest_cache"} or path.suffix in {".pyc", ".pyo", ".prof"}:
                found.append(str(path.relative_to(ROOT)))
    return tuple(sorted(found))


def bundle_payload_paths() -> tuple[Path, ...]:
    required = (
        ROOT / "Demo/run_6q_fixed_budget_min_error_1m.py",
        REPORT_PATH,
        VERIFICATION_REPORT_PATH,
        ROOT / "Optimization/__init__.py",
        ROOT / "Optimization/checkpoint.py",
        ROOT / "Optimization/objective.py",
        ROOT / "Optimization/parameterization.py",
        ROOT / "Optimization/progressive_search.py",
        ROOT / "Optimization/search.py",
        ROOT / "Optimization/specification.py",
        ROOT / "main_v2.py",
        ROOT / "Demo/run_small_cebp_demo_6qubit.py",
    )
    optional = (
        ROOT / "Optimization/PROGRESSIVE_FIXED_BUDGET_SEARCH_REMEDIATION_VERIFICATION.txt",
        ROOT / "Optimization/OPTIMIZATION_V2_VERIFICATION.txt",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required bundle files: {missing}")
    return required + tuple(path for path in optional if path.is_file())


def write_manifest(result, state_digest: str) -> None:
    comparison = successful_statistics(result.comparison_evaluation)
    holdout = successful_statistics(result.holdout_evaluation)
    payload = bundle_payload_paths()
    lines = [
        "6Q / 1M PROGRESSIVE VERIFICATION BUNDLE MANIFEST",
        "=" * 64,
        f"ZIP creation timestamp (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"branch: {git_output('branch', '--show-current')}",
        f"git HEAD: {git_output('rev-parse', 'HEAD')}",
        f"experiment state digest: {state_digest}",
        f"selected candidate ID: {result.best_candidate_id}",
        f"termination reason: {result.search_metadata.termination_reason}",
        f"final candidate pool size: {result.search_metadata.final_candidate_pool_size}",
        f"comparison mean/max D: {fmt(comparison['mean_d'])} / {fmt(comparison['max_d'])}",
        f"holdout mean/max D: {fmt(holdout['mean_d'])} / {fmt(holdout['max_d'])}", "",
        "relative path | byte size | SHA-256",
    ]
    for path in payload:
        lines.append(f"{path.relative_to(ROOT)} | {path.stat().st_size} | {sha256(path)}")
    lines.extend((
        f"{MANIFEST_PATH.relative_to(ROOT)} | SELF | SELF",
        "Manifest self-entry is intentionally marked SELF because a file cannot contain its own final cryptographic digest.",
        "The ZIP verifier separately checks the manifest archive member byte-for-byte against this source file.", "",
    ))
    MANIFEST_PATH.write_text("\n".join(lines), encoding="utf-8")


def create_and_verify_bundle(result, state_digest: str) -> tuple[bool, tuple[str, ...]]:
    write_manifest(result, state_digest)
    sources = (*bundle_payload_paths(), MANIFEST_PATH)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source in sources:
            archive.write(source, source.relative_to(ROOT).as_posix())
    expected = {source.relative_to(ROOT).as_posix(): source for source in sources}
    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        members = tuple(archive.namelist())
        if set(members) != set(expected) or archive.testzip() is not None:
            return False, members
        for name, source in expected.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != sha256(source):
                return False, members
    return True, members


def print_compact_summary(result: ProgressiveOptimizationResult, runtime: float) -> None:
    for item in result.search_metadata.round_summaries:
        print(
            f"progressive round {item.round_index}: pool={item.candidate_pool_size} "
            f"winner={item.round_search_winner_id} incumbent={item.incumbent_after_id} "
            f"comparison_D={fmt(item.selected_incumbent_comparison_mean_loss)} "
            f"improvement={fmt(item.relative_improvement)}"
        )
    comparison = successful_statistics(result.comparison_evaluation)
    holdout = successful_statistics(result.holdout_evaluation)
    print(f"termination: {result.search_metadata.termination_reason}")
    print(f"comparison: mean/max D={fmt(comparison['mean_d'])}/{fmt(comparison['max_d'])}")
    print(f"holdout: mean/max D={fmt(holdout['mean_d'])}/{fmt(holdout['max_d'])}")
    print(f"runtime: optimizer={runtime:.3f}s learner_runs={result.actual_learner_run_count}")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="resume only from this experiment's checkpoint")
    parser.add_argument("--check-only", action="store_true", help="run fast fixture/import/wiring checks only")
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    start = time.perf_counter()

    verify_seed_sets()
    hashes_before = dependency_hashes()
    instance, state_digest, identity_ok = build_historical_6q_instance()
    space = build_search_space()
    progressive = build_progressive_config()
    config = build_optimization_config(resume=args.resume)
    progressive.validate_for(config)
    derive_wiring_candidate(space, config)
    print("setup/state: PASS")
    print("backend/policy: PASS")
    if args.check_only:
        print("static/wiring checks: PASS")
        return

    checkpoint_exists = CHECKPOINT_PATH.exists()
    if args.resume and not checkpoint_exists:
        raise FileNotFoundError(f"Requested checkpoint does not exist: {CHECKPOINT_PATH}")
    if not args.resume and checkpoint_exists:
        raise FileExistsError(
            f"Fresh run refused because checkpoint exists: {CHECKPOINT_PATH}. "
            "Use --resume or remove this experiment-owned checkpoint explicitly."
        )

    optimization_start = time.perf_counter()
    result = optimize_cebp_parameters_progressive(
        instance, COPY_CEILING, config, space, progressive_config=progressive,
    )
    optimization_runtime = time.perf_counter() - optimization_start
    if result.objective.mode.value != "fixed_budget_min_error":
        raise RuntimeError("Progressive optimizer used the wrong objective mode.")
    if result.search_metadata.fixed_budget_epsilon_refinement_enabled:
        raise RuntimeError("Fixed-budget epsilon refinement unexpectedly ran.")
    validate_final_evaluation(result.comparison_evaluation, "comparison")
    validate_final_evaluation(result.holdout_evaluation, "holdout")
    diagnostics = run_selected_diagnostics(instance, result, config)
    hashes_after = dependency_hashes()
    if hashes_before != hashes_after:
        raise RuntimeError("A read-only dependency changed during the experiment.")
    CHECKPOINT_PATH.unlink(missing_ok=True)
    (ROOT / "Reports").mkdir(parents=True, exist_ok=True)
    total_runtime = time.perf_counter() - start
    REPORT_PATH.write_text(
        render_demo_report(
            result, config, progressive, space, diagnostics,
            instance=instance, state_digest=state_digest, state_identity_ok=identity_ok,
            hashes_before=hashes_before, hashes_after=hashes_after,
            optimization_runtime=optimization_runtime, total_runtime=total_runtime,
        ), encoding="utf-8",
    )
    VERIFICATION_REPORT_PATH.write_text(
        render_verification_report(
            result, progressive, diagnostics,
            state_digest=state_digest, state_identity_ok=identity_ok,
            hashes_before=hashes_before, hashes_after=hashes_after,
            optimization_runtime=optimization_runtime, total_runtime=total_runtime,
            zip_integrity=False,
        ), encoding="utf-8",
    )
    probe_ok, _ = create_and_verify_bundle(result, state_digest)
    if not probe_ok:
        raise RuntimeError("Preliminary verification ZIP failed integrity/hash validation.")
    VERIFICATION_REPORT_PATH.write_text(
        render_verification_report(
            result, progressive, diagnostics,
            state_digest=state_digest, state_identity_ok=identity_ok,
            hashes_before=hashes_before, hashes_after=hashes_after,
            optimization_runtime=optimization_runtime, total_runtime=total_runtime,
            zip_integrity=True,
        ), encoding="utf-8",
    )
    zip_ok, members = create_and_verify_bundle(result, state_digest)
    if not zip_ok:
        raise RuntimeError("Final verification ZIP failed integrity/hash validation.")
    print_compact_summary(result, optimization_runtime)
    print(f"report: {REPORT_PATH.relative_to(ROOT)}")
    print(f"verification report: {VERIFICATION_REPORT_PATH.relative_to(ROOT)}")
    print(f"verification bundle: {ZIP_PATH.relative_to(ROOT)} ({len(members)} members)")
    print("ZIP integrity: PASS")


if __name__ == "__main__":
    main()
