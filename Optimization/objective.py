"""Objective-aware empirical evaluation with a strict learner/oracle boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from statistics import median
import time
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np

from main_v2 import CEBPInstance, debug_end_to_end_trace_error, full_cebp_tomography

from .parameterization import (
    DerivedCandidate,
    OptimizationConfig,
    PreflightResourceCheck,
    backend_sampling_count_limit,
    candidate_to_end_to_end_config,
    preflight_resource_check,
    unsafe_backend_sampling_quantities,
)
from .specification import OptimizationMode, OptimizationObjective


FAILURE_LOSS = 1.0
NUMERIC_ERROR_TOLERANCE = 1e-12
FIXED_ERROR_FINAL_TUNING_SEED_COUNT = 16
NUMERIC_SAMPLING_LIMIT_REACHED = "numeric_sampling_limit_reached"


def _require_within_scientific_copy_cap(
    physical_budget: int,
    objective: OptimizationObjective,
    *,
    context: str,
) -> None:
    """Reject an out-of-domain fixed-error learner call before E2E execution."""

    cap = objective.scientific_copy_cap
    if (
        objective.mode is OptimizationMode.FIXED_ERROR_MIN_COPIES
        and cap is not None
        and int(physical_budget) > int(cap)
    ):
        raise ValueError(
            f"{context}: physical budget {int(physical_budget)} exceeds the "
            f"fixed-error scientific_copy_cap={int(cap)}; no E2E evaluation ran."
        )


def require_trace_distance_objective_available(
    instance: CEBPInstance,
    optimization_config: OptimizationConfig,
    objective: OptimizationObjective,
) -> None:
    """Reject a trace-distance search before any candidate can be evaluated."""

    if instance.n <= int(optimization_config.max_oracle_dense_qubits):
        return
    raise ValueError(
        f"Optimization objective {objective.mode.value!r} requires exact "
        "trace-distance ranking, but the dense oracle is unavailable: "
        f"n={instance.n} exceeds max_oracle_dense_qubits="
        f"{optimization_config.max_oracle_dense_qubits}. Increase the explicit "
        "oracle limit only if the dense allocation is safe."
    )


@dataclass(frozen=True)
class SeedEvaluation:
    learner_seed: int
    preflight_rejected: bool
    preflight_resource_check: PreflightResourceCheck
    operational_success: bool
    budget_feasible: bool
    post_run_realized_budget_feasible: Optional[bool]
    loss: Optional[float]
    loss_computed: bool
    trace_distance: Optional[float]
    failure_stage: Optional[str]
    failure_reason: Optional[str]
    realized_copy_ledger: Tuple[Tuple[str, int], ...]
    realized_total: int
    copy_utilization: float
    recovered_t: Optional[int]
    cluster_sizes: Tuple[int, ...]
    register_sizes: Tuple[int, ...]
    j_aux_size: Optional[int]
    error_feasible: Optional[bool]
    error_excess: Optional[float]
    estimator_available: bool = False
    budget_truncated: bool = False
    truncated_stages: Tuple[str, ...] = ()
    oracle_loss_available: bool = True
    oracle_unavailable_reason: Optional[str] = None
    oracle_loss_requested: bool = True
    ranking_eligible: bool = True
    oracle_evaluation_seconds: Optional[float] = field(default=None, compare=False)
    execution_resource_policy: Tuple[Tuple[str, str], ...] = ()
    fixed_budget_stage_records: Tuple[
        Tuple[str, int, int, int, bool, bool, Optional[str]], ...
    ] = ()
    grouping_diagnostics: Tuple[Tuple[str, str], ...] = ()
    actual_M_sgn: Optional[int] = None
    tomography_fixed_budget: Optional[int] = None
    tomography_complete_coverage: Optional[bool] = None
    theta_tau_multiplier: Optional[float] = None
    theta: Optional[float] = None
    tau_rank: Optional[float] = None
    theta_over_tau_rank: Optional[float] = None
    ranked_survivor_count: Optional[int] = None
    recovered_sector_count: Optional[int] = None
    recovered_span_rank: Optional[int] = None
    recovery_threshold_margin_holds: Optional[bool] = None
    recovery_ranking_gap_margin_holds: Optional[bool] = None
    structural_coverage_complete: Optional[bool] = None
    execution_branch: Optional[str] = None
    numeric_sampling_limit_reached: bool = False


@dataclass(frozen=True)
class CandidateEvaluation:
    """Aggregate evaluation for one candidate.

    ``mean_error_feasible`` and ``success_fraction_feasible`` expose the two
    independent fixed-error accuracy checks.  ``final_target_feasible`` is true
    only for an operationally valid aggregate over the final 16 tuning seeds
    when both checks pass.  The older ``all_error_feasible`` field is retained
    as a deprecated serialized alias for that combined final result; it never
    means that every seed must individually pass.
    """

    candidate_id: str
    N_candidate: int
    seed_evaluations: Tuple[SeedEvaluation, ...]
    n_seeds_evaluated: int
    mean_loss: float
    median_loss: float
    max_loss: float
    success_rate: float
    budget_feasible_rate: float
    all_budget_feasible: bool
    mean_trace_distance_successful: Optional[float]
    max_trace_distance_successful: Optional[float]
    trace_distance_std: Optional[float]
    operationally_valid: bool
    mean_realized_copies: float
    max_realized_copies: int
    mean_copy_utilization: float
    max_copy_utilization: float
    mean_stage_copies: Tuple[Tuple[str, float], ...]
    max_stage_copies: Tuple[Tuple[str, int], ...]
    error_target: Optional[float]
    error_feasible_rate: Optional[float]
    all_error_feasible: Optional[bool]
    max_error_excess: Optional[float]
    mean_error_excess: Optional[float]
    oracle_loss_available_rate: float = 1.0
    mean_error_feasible: Optional[bool] = None
    final_tuning_seed_count: Optional[int] = None
    final_fidelity_reached: bool = False
    error_success_count: Optional[int] = None
    error_success_fraction: Optional[float] = None
    required_error_success_count: Optional[int] = None
    success_probability_threshold: Optional[float] = None
    success_fraction_feasible: Optional[bool] = None
    final_target_feasible: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.final_target_feasible is None and self.all_error_feasible is not None:
            object.__setattr__(
                self, "final_target_feasible", self.all_error_feasible
            )
        elif (
            self.final_target_feasible is not None
            and self.all_error_feasible != self.final_target_feasible
        ):
            raise ValueError(
                "all_error_feasible is a deprecated alias and must equal "
                "final_target_feasible."
            )

    @property
    def mean_trace_distance(self) -> Optional[float]:
        return self.mean_trace_distance_successful

    @property
    def max_trace_distance(self) -> Optional[float]:
        return self.max_trace_distance_successful


@dataclass(frozen=True)
class HoldoutEvaluation(CandidateEvaluation):
    @property
    def budget_robust(self) -> bool:
        return self.all_budget_feasible

    @property
    def target_feasible_rate(self) -> Optional[float]:
        return self.error_feasible_rate


def _compact_structure(result) -> Tuple[Optional[int], Tuple[int, ...], Tuple[int, ...], Optional[int]]:
    recovered_t = result.peeling.t if result.peeling is not None else None
    cluster_sizes: Tuple[int, ...] = ()
    if result.grouping is not None:
        cluster_sizes = tuple(len(cluster) for cluster in result.grouping.clusters)
    register_sizes: Tuple[int, ...] = ()
    j_aux_size: Optional[int] = None
    if result.localization is not None:
        register_sizes = tuple(len(register) for _cluster, register in result.localization.J_C)
        j_aux_size = len(result.localization.J_aux)
    return recovered_t, cluster_sizes, register_sizes, j_aux_size


def _error_feasibility(
    *,
    operational_success: bool,
    budget_feasible: bool,
    trace_distance: Optional[float],
    objective: OptimizationObjective,
) -> Tuple[Optional[bool], Optional[float]]:
    if objective.mode is OptimizationMode.FIXED_BUDGET_MIN_ERROR:
        return None, None
    threshold = objective.effective_error_threshold
    assert threshold is not None
    if operational_success and budget_feasible and trace_distance is None:
        return None, None
    feasible = bool(
        operational_success
        and budget_feasible
        and trace_distance is not None
        and trace_distance <= threshold + NUMERIC_ERROR_TOLERANCE
    )
    excess = (
        max(0.0, float(trace_distance) - threshold)
        if trace_distance is not None and operational_success and budget_feasible
        else FAILURE_LOSS
    )
    return feasible, float(excess)


def _execution_resource_policy(
    config: OptimizationConfig,
) -> Tuple[Tuple[str, str], ...]:
    """Return concise immutable metadata for the exact simulator policy used."""

    return (
        ("simulation_backend", str(config.simulation_backend)),
        ("max_enumeration_qubits", str(config.max_enumeration_qubits)),
        ("max_oracle_dense_qubits", str(config.max_oracle_dense_qubits)),
        ("max_score_array_bytes", str(config.max_score_array_bytes)),
        (
            "max_structured_bell_workspace_bytes",
            str(config.max_structured_bell_workspace_bytes),
        ),
        (
            "max_enumeration_workspace_bytes",
            str(config.max_enumeration_workspace_bytes),
        ),
        (
            "enumeration_workspace_safety_factor",
            format(float(config.enumeration_workspace_safety_factor), ".17g"),
        ),
    )


def _run_candidate_on_seed(
    instance: CEBPInstance,
    derived_candidate: DerivedCandidate,
    learner_seed: int,
    total_copies: int,
    optimization_config: OptimizationConfig,
    *,
    objective: OptimizationObjective,
    compute_oracle_loss: bool,
) -> SeedEvaluation:
    physical_budget = int(derived_candidate.physical_copy_budget or total_copies)
    _require_within_scientific_copy_cap(
        physical_budget,
        objective,
        context="fixed-error learner entry",
    )
    unsafe_counts = unsafe_backend_sampling_quantities(
        derived_candidate,
        total_copies=physical_budget,
    )
    if unsafe_counts:
        limit = backend_sampling_count_limit()
        detail = ", ".join(f"{name}={value}" for name, value in unsafe_counts)
        reason = (
            f"{NUMERIC_SAMPLING_LIMIT_REACHED}: backend_count_limit={limit}; "
            f"{detail}"
        )
        check = PreflightResourceCheck(
            mandatory_bell_copies=derived_candidate.mandatory_bell_copies,
            worst_case_sign_reservation=derived_candidate.worst_case_sign_reservation,
            preflight_fixed_reservation=derived_candidate.preflight_fixed_reservation,
            grouping_safety_estimate=0,
            tomography_safety_estimate=0,
            preflight_safety_estimate=physical_budget,
            rigorous_grouping_upper_bound=0,
            rigorous_tomography_upper_bound=0,
            rigorous_total_upper_bound=physical_budget,
            single_grouping_query_shots_estimate=0,
            safety_limit=limit,
            runtime_safety_rejected=True,
            mathematical_budget_rejected=False,
            safe_to_execute=False,
            reason=reason,
            guarantee_level=(
                "runtime-derived NumPy intp/C-long representability guard; "
                "dynamic sample counts are bounded by the rejected physical pools"
            ),
        )
        error_feasible, error_excess = _error_feasibility(
            operational_success=False,
            budget_feasible=False,
            trace_distance=None,
            objective=objective,
        )
        return SeedEvaluation(
            learner_seed=int(learner_seed),
            preflight_rejected=True,
            preflight_resource_check=check,
            operational_success=False,
            budget_feasible=False,
            post_run_realized_budget_feasible=None,
            loss=FAILURE_LOSS,
            loss_computed=False,
            trace_distance=None,
            failure_stage="preflight_numeric_sampling_safety",
            failure_reason=reason,
            realized_copy_ledger=(),
            realized_total=0,
            copy_utilization=0.0,
            recovered_t=None,
            cluster_sizes=(),
            register_sizes=(),
            j_aux_size=None,
            error_feasible=error_feasible,
            error_excess=error_excess,
            oracle_loss_available=(
                instance.n <= int(optimization_config.max_oracle_dense_qubits)
            ),
            oracle_loss_requested=bool(compute_oracle_loss),
            ranking_eligible=True,
            execution_resource_policy=_execution_resource_policy(
                optimization_config
            ),
            theta_tau_multiplier=derived_candidate.theta_tau_multiplier,
            theta=derived_candidate.theta,
            tau_rank=derived_candidate.tau_rank,
            theta_over_tau_rank=derived_candidate.theta_over_tau_rank,
            numeric_sampling_limit_reached=True,
        )
    check = preflight_resource_check(
        derived_candidate,
        n=instance.n,
        d=instance.d,
        total_copies=physical_budget,
        optimization_config=optimization_config,
    )
    if not check.safe_to_execute:
        error_feasible, error_excess = _error_feasibility(
            operational_success=False,
            budget_feasible=False,
            trace_distance=None,
            objective=objective,
        )
        return SeedEvaluation(
            learner_seed=int(learner_seed),
            preflight_rejected=True,
            preflight_resource_check=check,
            operational_success=False,
            budget_feasible=False,
            post_run_realized_budget_feasible=None,
            loss=FAILURE_LOSS,
            loss_computed=False,
            trace_distance=None,
            failure_stage=(
                "preflight_runtime_safety"
                if check.runtime_safety_rejected
                else "preflight_budget"
            ),
            failure_reason=check.reason,
            realized_copy_ledger=(),
            realized_total=0,
            copy_utilization=0.0,
            recovered_t=None,
            cluster_sizes=(),
            register_sizes=(),
            j_aux_size=None,
            error_feasible=error_feasible,
            error_excess=error_excess,
            oracle_loss_available=(
                instance.n <= int(optimization_config.max_oracle_dense_qubits)
            ),
            oracle_unavailable_reason=(
                None
                if instance.n <= int(optimization_config.max_oracle_dense_qubits)
                else (
                    "exact_trace_distance_unavailable: "
                    f"n={instance.n} exceeds max_oracle_dense_qubits="
                    f"{optimization_config.max_oracle_dense_qubits}"
                )
            ),
            oracle_loss_requested=bool(compute_oracle_loss),
            ranking_eligible=True,
            execution_resource_policy=_execution_resource_policy(
                optimization_config
            ),
            theta_tau_multiplier=derived_candidate.theta_tau_multiplier,
            theta=derived_candidate.theta,
            tau_rank=derived_candidate.tau_rank,
            theta_over_tau_rank=derived_candidate.theta_over_tau_rank,
        )
    # The learner sees only public measurement access. Oracle truth remains
    # outside until the empirical learner run is complete and budget-feasible.
    learner_view = instance.learner_view()
    learner_config = candidate_to_end_to_end_config(
        derived_candidate,
        d=instance.d,
        learner_seed=learner_seed,
        total_copies=physical_budget,
        optimization_config=optimization_config,
    )
    _require_within_scientific_copy_cap(
        physical_budget,
        objective,
        context="immediately before full_cebp_tomography",
    )
    result = full_cebp_tomography(learner_view, config=learner_config)
    if result.theorem_certified:
        raise RuntimeError("Manual-override learner run unexpectedly became theorem-certified.")
    if result.realized_total != result.realized_copy_ledger.total:
        raise RuntimeError("EndToEndResult realized total disagrees with CopyLedger.total.")

    budget_feasible = result.realized_total <= physical_budget
    trace_distance: Optional[float] = None
    loss: Optional[float] = FAILURE_LOSS
    loss_computed = False
    estimator_available = bool(
        getattr(result, "estimator_available", bool(result.success))
    )
    operational_success = bool(
        estimator_available
        and (
            objective.mode is OptimizationMode.FIXED_BUDGET_MIN_ERROR
            or bool(result.success)
        )
    )
    oracle_loss_available = bool(
        instance.n <= int(optimization_config.max_oracle_dense_qubits)
    )
    oracle_unavailable_reason = None
    if compute_oracle_loss and not oracle_loss_available:
        oracle_unavailable_reason = (
            "exact_trace_distance_unavailable: "
            f"n={instance.n} exceeds max_oracle_dense_qubits="
            f"{optimization_config.max_oracle_dense_qubits}"
        )
    oracle_evaluation_seconds = None
    if (
        compute_oracle_loss
        and oracle_loss_available
        and estimator_available
        and budget_feasible
    ):
        oracle_started = time.perf_counter()
        trace_distance = 0.5 * float(
            debug_end_to_end_trace_error(
                result,
                instance,
                max_dense_qubits=int(optimization_config.max_oracle_dense_qubits),
            )
        )
        oracle_evaluation_seconds = time.perf_counter() - oracle_started
        if math.isfinite(trace_distance):
            loss = trace_distance
            loss_computed = True
        else:
            trace_distance = None
            estimator_available = False
            operational_success = False
            loss = FAILURE_LOSS
    ranking_eligible = bool(
        not (compute_oracle_loss and estimator_available and budget_feasible)
        or loss_computed
    )
    if not ranking_eligible:
        loss = None
    error_feasible, error_excess = _error_feasibility(
        operational_success=operational_success,
        budget_feasible=bool(budget_feasible),
        trace_distance=trace_distance,
        objective=objective,
    )
    recovered_t, cluster_sizes, register_sizes, j_aux_size = _compact_structure(result)
    recovery = getattr(result, "recovery", None)
    stage_records = tuple(
        (
            record.stage,
            int(record.assigned_cap),
            int(record.realized_copies),
            int(record.unused_copies),
            bool(record.budget_exhausted),
            bool(record.stage_complete),
            record.degradation_reason,
        )
        for record in getattr(result, "fixed_budget_stage_records", ())
    )
    grouping = getattr(result, "grouping", None)
    grouping_diagnostics = () if grouping is None else (
        ("sampling_policy", str(getattr(grouping, "sampling_policy", "unknown"))),
        ("exploratory_query_count", str(getattr(grouping, "exploratory_query_count", 0))),
        ("refinement_top_up_count", str(getattr(grouping, "refinement_top_up_count", 0))),
        ("realized_query_count", str(getattr(grouping, "realized_query_count", 0))),
        ("realized_grouping_copies", str(getattr(grouping, "realized_grouping_copies", 0))),
        ("min_shots_per_tuple", str(getattr(grouping, "min_shots_per_queried_tuple", 0))),
        ("max_shots_per_tuple", str(getattr(grouping, "max_shots_per_queried_tuple", 0))),
        ("mean_shots_per_tuple", format(float(getattr(grouping, "mean_shots_per_queried_tuple", 0.0)), ".17g")),
        ("final_partition", repr(tuple(grouping.clusters))),
        ("no_false_merge_condition_holds", str(getattr(grouping, "no_false_merge_condition_holds", False))),
        ("theorem_grouping_preconditions_hold", str(getattr(grouping, "theorem_grouping_preconditions_hold", False))),
    )
    tomography = getattr(result, "tomography", None)
    tomography_budget = (
        None
        if tomography is None
        else int(
            getattr(
                tomography,
                "block_tomography_pool",
                getattr(tomography, "attempted_copies", 0),
            )
        )
    )
    tomography_complete = (
        None
        if tomography is None
        else not bool(getattr(tomography, "budget_truncated", False))
    )
    return SeedEvaluation(
        learner_seed=int(learner_seed),
        preflight_rejected=False,
        preflight_resource_check=check,
        operational_success=operational_success,
        budget_feasible=bool(budget_feasible),
        post_run_realized_budget_feasible=bool(budget_feasible),
        loss=loss,
        loss_computed=loss_computed,
        trace_distance=trace_distance,
        failure_stage=result.failure_stage,
        failure_reason=result.failure_reason,
        realized_copy_ledger=tuple(result.realized_copy_ledger.entries),
        realized_total=int(result.realized_total),
        copy_utilization=float(result.realized_total / physical_budget),
        recovered_t=recovered_t,
        cluster_sizes=cluster_sizes,
        register_sizes=register_sizes,
        j_aux_size=j_aux_size,
        error_feasible=error_feasible,
        error_excess=error_excess,
        estimator_available=estimator_available,
        budget_truncated=bool(getattr(result, "budget_truncated", False)),
        truncated_stages=tuple(getattr(result, "truncated_stages", ())),
        oracle_loss_available=oracle_loss_available,
        oracle_unavailable_reason=oracle_unavailable_reason,
        oracle_loss_requested=bool(compute_oracle_loss),
        ranking_eligible=ranking_eligible,
        oracle_evaluation_seconds=oracle_evaluation_seconds,
        execution_resource_policy=_execution_resource_policy(optimization_config),
        fixed_budget_stage_records=stage_records,
        grouping_diagnostics=grouping_diagnostics,
        actual_M_sgn=(
            None
            if getattr(result, "syndrome", None) is None
            else int(result.syndrome.M_sgn)
        ),
        tomography_fixed_budget=tomography_budget,
        tomography_complete_coverage=tomography_complete,
        theta_tau_multiplier=derived_candidate.theta_tau_multiplier,
        theta=float(getattr(recovery, "theta", derived_candidate.theta)),
        tau_rank=float(
            getattr(recovery, "tau_rank", derived_candidate.tau_rank)
        ),
        theta_over_tau_rank=float(
            getattr(recovery, "theta", derived_candidate.theta)
            / getattr(recovery, "tau_rank", derived_candidate.tau_rank)
        ),
        ranked_survivor_count=(
            None if recovery is None else int(recovery.ranked_survivor_count)
        ),
        recovered_sector_count=(
            None if recovery is None else len(recovery.sectors)
        ),
        recovered_span_rank=(
            None if recovery is None else len(recovery.recovered_span_basis)
        ),
        recovery_threshold_margin_holds=(
            None if recovery is None else bool(recovery.threshold_margin_holds)
        ),
        recovery_ranking_gap_margin_holds=(
            None if recovery is None else bool(recovery.ranking_gap_margin_holds)
        ),
        structural_coverage_complete=(
            bool(j_aux_size == 0)
            if instance.d == 1 and j_aux_size is not None
            else None
        ),
        execution_branch=(
            None
            if getattr(result, "branch", None) is None
            else str(result.branch)
        ),
    )


def evaluate_candidate_on_seed(
    instance: CEBPInstance,
    derived_candidate: DerivedCandidate,
    learner_seed: int,
    total_copies: int,
    optimization_config: Optional[OptimizationConfig] = None,
    *,
    objective: Optional[OptimizationObjective] = None,
) -> SeedEvaluation:
    """Run one manual learner execution and score conventional trace distance."""

    cfg = optimization_config or OptimizationConfig(total_copies=total_copies)
    resolved = objective or cfg.effective_objective
    return _run_candidate_on_seed(
        instance,
        derived_candidate,
        learner_seed,
        total_copies,
        cfg,
        objective=resolved,
        compute_oracle_loss=True,
    )


def evaluate_candidate_copy_only(
    instance: CEBPInstance,
    derived_candidate: DerivedCandidate,
    learner_seed: int,
    total_copies: int,
    optimization_config: OptimizationConfig,
) -> SeedEvaluation:
    """Compatibility helper for an oracle-free complete learner execution."""

    return _run_candidate_on_seed(
        instance,
        derived_candidate,
        learner_seed,
        total_copies,
        optimization_config,
        objective=optimization_config.effective_objective,
        compute_oracle_loss=False,
    )


def aggregate_candidate_evaluations(
    candidate_id: str,
    seed_evaluations: Iterable[SeedEvaluation],
    *,
    objective: Optional[OptimizationObjective] = None,
    holdout: bool = False,
    N_candidate: Optional[int] = None,
) -> CandidateEvaluation:
    evaluations = tuple(seed_evaluations)
    if not evaluations:
        raise ValueError("At least one seed evaluation is required.")
    if not all(evaluation.ranking_eligible for evaluation in evaluations):
        reasons = sorted(
            {
                evaluation.oracle_unavailable_reason or "objective unavailable"
                for evaluation in evaluations
                if not evaluation.ranking_eligible
            }
        )
        raise ValueError(
            "Candidate objective is not ranking-eligible: " + "; ".join(reasons)
        )
    if any(evaluation.loss is None for evaluation in evaluations):
        raise RuntimeError("Ranking-eligible evaluations must carry an objective value.")
    losses = np.asarray([evaluation.loss for evaluation in evaluations], dtype=float)
    copies = np.asarray([evaluation.realized_total for evaluation in evaluations], dtype=float)
    utilizations = np.asarray([evaluation.copy_utilization for evaluation in evaluations], dtype=float)
    finite_errors = [
        float(evaluation.trace_distance)
        for evaluation in evaluations
        if evaluation.operational_success
        and evaluation.budget_feasible
        and evaluation.trace_distance is not None
        and math.isfinite(float(evaluation.trace_distance))
    ]
    operationally_valid = bool(
        all(evaluation.operational_success for evaluation in evaluations)
        and all(evaluation.budget_feasible for evaluation in evaluations)
        and len(finite_errors) == len(evaluations)
    )
    stage_names = sorted(
        {name for evaluation in evaluations for name, _count in evaluation.realized_copy_ledger}
    )
    ledgers = [dict(evaluation.realized_copy_ledger) for evaluation in evaluations]
    resolved = objective
    is_fixed_error = (
        resolved is not None and resolved.mode is OptimizationMode.FIXED_ERROR_MIN_COPIES
    )
    feasible_values = [bool(item.error_feasible) for item in evaluations]
    threshold = resolved.effective_error_threshold if is_fixed_error else None
    errors_for_statistics = (
        finite_errors if (not is_fixed_error or operationally_valid) else []
    )
    mean_trace_distance = (
        float(np.mean(errors_for_statistics)) if errors_for_statistics else None
    )
    maximum_trace_distance = (
        float(np.max(errors_for_statistics)) if errors_for_statistics else None
    )
    trace_distance_std = (
        float(np.std(errors_for_statistics)) if errors_for_statistics else None
    )
    mean_error_excess = (
        max(0.0, mean_trace_distance - float(threshold))
        if is_fixed_error and mean_trace_distance is not None
        else (FAILURE_LOSS if is_fixed_error else None)
    )
    max_error_excess = (
        max(0.0, maximum_trace_distance - float(threshold))
        if is_fixed_error and maximum_trace_distance is not None
        else (FAILURE_LOSS if is_fixed_error else None)
    )
    mean_error_feasible = bool(
        is_fixed_error
        and operationally_valid
        and mean_trace_distance is not None
        and mean_trace_distance <= float(threshold) + NUMERIC_ERROR_TOLERANCE
    )
    probability_threshold = (
        float(resolved.success_probability_threshold)
        if is_fixed_error
        else None
    )
    error_success_count = sum(feasible_values) if is_fixed_error else None
    error_success_fraction = (
        float(error_success_count / len(evaluations)) if is_fixed_error else None
    )
    final_fidelity_reached = bool(
        is_fixed_error
        and len(evaluations) == FIXED_ERROR_FINAL_TUNING_SEED_COUNT
    )
    required_error_success_count = (
        int(
            math.ceil(
                probability_threshold * FIXED_ERROR_FINAL_TUNING_SEED_COUNT
            )
        )
        if final_fidelity_reached
        else None
    )
    success_fraction_feasible = (
        bool(error_success_count >= required_error_success_count)
        if final_fidelity_reached
        else None
    )
    target_feasible = (
        bool(
            operationally_valid
            and mean_error_feasible
            and success_fraction_feasible
        )
        if final_fidelity_reached
        else None
    )
    provisioned_budget = int(
        N_candidate
        if N_candidate is not None
        else (
            resolved.copy_ceiling
            if resolved is not None and resolved.copy_ceiling is not None
            else max(copies)
        )
    )
    common = dict(
        candidate_id=str(candidate_id),
        N_candidate=provisioned_budget,
        seed_evaluations=evaluations,
        n_seeds_evaluated=len(evaluations),
        mean_loss=float(losses.mean()),
        median_loss=float(median(map(float, losses))),
        max_loss=float(losses.max()),
        success_rate=float(np.mean([evaluation.operational_success for evaluation in evaluations])),
        budget_feasible_rate=float(np.mean([evaluation.budget_feasible for evaluation in evaluations])),
        all_budget_feasible=all(evaluation.budget_feasible for evaluation in evaluations),
        mean_trace_distance_successful=mean_trace_distance,
        max_trace_distance_successful=maximum_trace_distance,
        trace_distance_std=trace_distance_std,
        operationally_valid=operationally_valid,
        mean_realized_copies=float(copies.mean()),
        max_realized_copies=int(copies.max()),
        mean_copy_utilization=float(utilizations.mean()),
        max_copy_utilization=float(utilizations.max()),
        mean_stage_copies=tuple(
            (name, float(np.mean([ledger.get(name, 0) for ledger in ledgers])))
            for name in stage_names
        ),
        max_stage_copies=tuple(
            (name, int(max(ledger.get(name, 0) for ledger in ledgers)))
            for name in stage_names
        ),
        error_target=float(resolved.error_target) if is_fixed_error else None,
        error_feasible_rate=float(np.mean(feasible_values)) if is_fixed_error else None,
        all_error_feasible=target_feasible if is_fixed_error else None,
        max_error_excess=max_error_excess,
        mean_error_excess=mean_error_excess,
        oracle_loss_available_rate=float(
            np.mean([evaluation.oracle_loss_available for evaluation in evaluations])
        ),
        mean_error_feasible=mean_error_feasible if is_fixed_error else None,
        final_tuning_seed_count=(
            FIXED_ERROR_FINAL_TUNING_SEED_COUNT if is_fixed_error else None
        ),
        final_fidelity_reached=final_fidelity_reached,
        error_success_count=error_success_count,
        error_success_fraction=error_success_fraction,
        required_error_success_count=required_error_success_count,
        success_probability_threshold=probability_threshold,
        success_fraction_feasible=success_fraction_feasible,
        final_target_feasible=target_feasible if is_fixed_error else None,
    )
    cls = HoldoutEvaluation if holdout else CandidateEvaluation
    return cls(**common)


def evaluation_identity(
    derived_candidate: DerivedCandidate,
    learner_seed: int,
    total_copies: int,
    optimization_config: OptimizationConfig,
    objective: OptimizationObjective,
) -> Tuple[str, int]:
    """Return a stable semantic identity independent of display candidate IDs."""

    derived_payload = asdict(derived_candidate)
    if objective.mode in (
        OptimizationMode.FIXED_BUDGET_MIN_ERROR,
        OptimizationMode.FIXED_ERROR_MIN_COPIES,
    ):
        # Graceful execution derives physical pools from normalized weights and
        # spends the tomography remainder directly.  Legacy alpha coordinates
        # and epsilon/zeta tomography targets therefore do not change learner
        # measurements, estimates, truncation, or objective values.
        if objective.mode is OptimizationMode.FIXED_BUDGET_MIN_ERROR:
            derived_payload.pop("parameters", None)
        derived_payload.pop("epsilon_tom", None)
        derived_payload.pop("zeta_tom", None)
        derived_payload.pop("tau_kappa", None)
    objective_payload = {
        **asdict(objective),
        "mode": objective.mode.value,
    }
    if objective.mode is OptimizationMode.FIXED_ERROR_MIN_COPIES:
        # Deprecated legacy metadata must not change a modern learner trial's
        # semantic identity.  The transient physical budget is already encoded
        # by the derived stage caps and total_copies below.
        objective_payload.pop("copy_ceiling", None)
        # The scientific cap governs which transient budgets may be requested,
        # not the semantics of an already-admissible learner trial.
        objective_payload.pop("scientific_copy_cap", None)
    if objective.mode is OptimizationMode.FIXED_BUDGET_MIN_ERROR:
        # This fixed-error-only field is absent from the historical fixed-budget
        # execution identity, preserving fixed-budget semantic-cache keys.
        objective_payload.pop("success_probability_threshold", None)
        objective_payload.pop("scientific_copy_cap", None)
    payload = {
        "evaluation_identity_schema": (
            10
            if objective.mode is OptimizationMode.FIXED_ERROR_MIN_COPIES
            else 8
        ),
        "derived": derived_payload,
        "total_copies": int(derived_candidate.physical_copy_budget or total_copies),
        "max_dense_qubits_compatibility_alias": int(optimization_config.max_dense_qubits),
        "max_enumeration_qubits": int(optimization_config.max_enumeration_qubits),
        "max_oracle_dense_qubits": int(optimization_config.max_oracle_dense_qubits),
        "inner_enumeration_workers": int(optimization_config.inner_enumeration_workers),
        "simulation_backend": optimization_config.simulation_backend,
        "enumeration_memory_policy": {
            "max_score_array_bytes": optimization_config.max_score_array_bytes,
            "max_structured_bell_workspace_bytes": (
                optimization_config.max_structured_bell_workspace_bytes
            ),
            "max_enumeration_workspace_bytes": (
                optimization_config.max_enumeration_workspace_bytes
            ),
            "enumeration_workspace_safety_factor": (
                optimization_config.enumeration_workspace_safety_factor
            ),
        },
        "runtime_safety": {
            "max_preflight_estimated_copies": optimization_config.max_preflight_estimated_copies,
            "execution_safety_factor": optimization_config.execution_safety_factor,
            "max_predicted_grouping_copies": optimization_config.max_predicted_grouping_copies,
            "max_predicted_tomography_copies": optimization_config.max_predicted_tomography_copies,
            "max_predicted_total_copies": optimization_config.max_predicted_total_copies,
            "max_single_grouping_query_shots": optimization_config.max_single_grouping_query_shots,
        },
        "objective": objective_payload,
    }
    if objective.mode is OptimizationMode.FIXED_ERROR_MIN_COPIES:
        payload["runtime_safety"].update(
            {
                "numeric_sampling_safety_policy_schema": 1,
                "backend_sampling_count_limit": backend_sampling_count_limit(),
            }
        )
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest, int(learner_seed)


def evaluate_candidate(
    instance: CEBPInstance,
    candidate_id: str,
    derived_candidate: DerivedCandidate,
    learner_seeds: Iterable[int],
    total_copies: int,
    optimization_config: OptimizationConfig,
    *,
    cache: Optional[Dict[Tuple[str, int], SeedEvaluation]] = None,
    holdout: bool = False,
    objective: Optional[OptimizationObjective] = None,
    on_evaluation: Optional[Callable[[Tuple[str, int], SeedEvaluation], None]] = None,
    on_cache_hit: Optional[Callable[[Tuple[str, int]], None]] = None,
    early_stop: Optional[Callable[[SeedEvaluation], bool]] = None,
) -> CandidateEvaluation:
    """Evaluate a candidate on common seeds with semantic caching."""

    result_cache = cache if cache is not None else {}
    resolved = objective or optimization_config.effective_objective
    evaluations = []
    for learner_seed in learner_seeds:
        key = evaluation_identity(
            derived_candidate,
            int(learner_seed),
            total_copies,
            optimization_config,
            resolved,
        )
        if key not in result_cache:
            result_cache[key] = evaluate_candidate_on_seed(
                instance,
                derived_candidate,
                int(learner_seed),
                total_copies,
                optimization_config,
                objective=resolved,
            )
            if on_evaluation is not None:
                on_evaluation(key, result_cache[key])
        elif on_cache_hit is not None:
            on_cache_hit(key)
        evaluations.append(result_cache[key])
        if early_stop is not None and early_stop(result_cache[key]):
            break
    return aggregate_candidate_evaluations(
        candidate_id,
        evaluations,
        objective=resolved,
        holdout=holdout,
        N_candidate=int(derived_candidate.physical_copy_budget or total_copies),
    )


def candidate_ranking_key(
    evaluation: CandidateEvaluation,
    objective: Optional[OptimizationObjective] = None,
) -> Tuple[object, ...]:
    """Return the deterministic objective-specific lexicographic key."""

    if objective is None or objective.mode is OptimizationMode.FIXED_BUDGET_MIN_ERROR:
        fully_operational = evaluation.success_rate == 1.0 and evaluation.all_budget_feasible
        return (
            0 if fully_operational else 1,
            evaluation.mean_loss,
            evaluation.max_loss,
            -evaluation.success_rate,
            evaluation.max_realized_copies,
            evaluation.candidate_id,
        )
    if evaluation.final_fidelity_reached and evaluation.final_target_feasible:
        return (
            0,
            evaluation.N_candidate,
            evaluation.mean_realized_copies,
            evaluation.max_realized_copies,
            evaluation.mean_trace_distance_successful,
            -float(evaluation.error_success_fraction or 0.0),
            evaluation.max_trace_distance_successful,
            evaluation.candidate_id,
        )
    if evaluation.final_fidelity_reached and evaluation.operationally_valid:
        failed_constraints = int(evaluation.mean_error_feasible is not True) + int(
            evaluation.success_fraction_feasible is not True
        )
        success_count_deficit = max(
            0,
            int(evaluation.required_error_success_count or 0)
            - int(evaluation.error_success_count or 0),
        )
        return (
            1,
            failed_constraints,
            success_count_deficit,
            evaluation.mean_error_excess,
            evaluation.N_candidate,
            evaluation.mean_trace_distance_successful,
            -float(evaluation.error_success_fraction or 0.0),
            evaluation.max_trace_distance_successful,
            evaluation.mean_realized_copies,
            evaluation.candidate_id,
        )
    if not evaluation.final_fidelity_reached and evaluation.operationally_valid:
        # Partial 1/2/4/8-seed evidence is deliberately continuous: neither
        # the 0.85 probability target nor a rounded success-count threshold is
        # used as a hard feasibility class before all final seeds are present.
        probability_threshold = float(
            evaluation.success_probability_threshold or 0.0
        )
        success_fraction = float(evaluation.error_success_fraction or 0.0)
        return (
            0,
            evaluation.mean_error_excess,
            evaluation.mean_trace_distance_successful,
            max(0.0, probability_threshold - success_fraction),
            -success_fraction,
            evaluation.N_candidate,
            evaluation.mean_realized_copies,
            evaluation.max_realized_copies,
            evaluation.candidate_id,
        )
    return (
        2,
        -evaluation.success_rate,
        -evaluation.budget_feasible_rate,
        evaluation.max_error_excess if evaluation.max_error_excess is not None else FAILURE_LOSS,
        evaluation.mean_loss,
        evaluation.N_candidate,
        evaluation.candidate_id,
    )
