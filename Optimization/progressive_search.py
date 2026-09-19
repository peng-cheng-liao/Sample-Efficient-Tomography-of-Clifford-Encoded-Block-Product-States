"""Progressive practical random search with persistent semantic caching."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from main_v2 import CEBPInstance

from .checkpoint import CheckpointStore
from .fixed_error_budget import (
    BUDGET_CAP_EXHAUSTED,
    BudgetThresholdEvaluation,
    FixedErrorBudgetTrial,
    MEAN_ERROR_FAILED,
    SUCCESS_FRACTION_FAILED,
    estimate_minimum_feasible_budget,
    fixed_error_threshold_ranking_key,
)
from .objective import (
    CandidateEvaluation,
    HoldoutEvaluation,
    SeedEvaluation,
    aggregate_candidate_evaluations,
    candidate_ranking_key,
    evaluate_candidate,
    evaluation_identity,
    require_trace_distance_objective_available,
)
from .parameterization import (
    CandidateParameters,
    DerivedCandidate,
    FixedBudgetCandidateParameters,
    FixedErrorCandidateParameters,
    OptimizationConfig,
    SearchSpace,
    derive_candidate,
    minimum_practical_copy_budget,
)
from .search import (
    CandidateRecord,
    _checkpoint_metadata,
    _json_safe,
    _sample_valid_candidates,
)
from .specification import OptimizationMode, OptimizationObjective


PROGRESSIVE_SEARCH_SCHEMA = 5
FIXED_ERROR_PROGRESSIVE_SEARCH_SCHEMA = 9
DEFAULT_FIXED_BUDGET_MIN_ERROR_IMPROVEMENT_RATIO = 0.0
MAX_FIXED_BUDGET_MIN_ERROR_CANDIDATES = 256
FIXED_ERROR_SEED_FIDELITIES = (1, 2, 4, 8, 16)
FIXED_ERROR_FINAL_SEED_COUNT = 16


def _positive_integer(name: str, value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


@dataclass(frozen=True)
class ProgressiveSearchConfig:
    """Resource and convergence controls for the progressive controller."""

    initial_candidates: int = 16
    max_candidates: int = MAX_FIXED_BUDGET_MIN_ERROR_CANDIDATES
    candidate_growth_factor: int = 2
    seed_fidelities: Tuple[int, ...] = (1, 2, 4, 8, 16)
    comparison_seed_count: int = 16
    relative_improvement_threshold: float = (
        DEFAULT_FIXED_BUDGET_MIN_ERROR_IMPROVEMENT_RATIO
    )
    improvement_patience: int = 2
    min_rounds: int = 3
    retention_fraction: float = 0.5
    checkpoint_every_n_new_evaluations: int = 16

    def __post_init__(self) -> None:
        initial = _positive_integer("initial_candidates", self.initial_candidates)
        maximum = _positive_integer("max_candidates", self.max_candidates)
        growth = _positive_integer(
            "candidate_growth_factor", self.candidate_growth_factor
        )
        comparison = _positive_integer(
            "comparison_seed_count", self.comparison_seed_count
        )
        patience = _positive_integer(
            "improvement_patience", self.improvement_patience
        )
        minimum_rounds = _positive_integer("min_rounds", self.min_rounds)
        checkpoint_cadence = _positive_integer(
            "checkpoint_every_n_new_evaluations",
            self.checkpoint_every_n_new_evaluations,
        )
        if initial > maximum:
            raise ValueError("initial_candidates must not exceed max_candidates.")
        if maximum > MAX_FIXED_BUDGET_MIN_ERROR_CANDIDATES:
            raise ValueError(
                "max_candidates must not exceed the fixed_budget_min_error hard "
                f"cap of {MAX_FIXED_BUDGET_MIN_ERROR_CANDIDATES}."
            )
        if growth < 2:
            raise ValueError("candidate_growth_factor must be at least 2.")
        fidelities = tuple(
            _positive_integer("seed_fidelities entries", item)
            for item in self.seed_fidelities
        )
        if not fidelities:
            raise ValueError("seed_fidelities must be nonempty.")
        if tuple(sorted(set(fidelities))) != fidelities:
            raise ValueError("seed_fidelities must be strictly increasing.")
        if comparison < fidelities[-1]:
            raise ValueError(
                "comparison_seed_count must be at least max(seed_fidelities)."
            )
        threshold = float(self.relative_improvement_threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold < 1.0:
            raise ValueError(
                "relative_improvement_threshold must be finite and lie in [0,1)."
            )
        retention = float(self.retention_fraction)
        if not math.isfinite(retention) or not 0.0 < retention < 1.0:
            raise ValueError("retention_fraction must lie in (0,1).")
        object.__setattr__(self, "initial_candidates", initial)
        object.__setattr__(self, "max_candidates", maximum)
        object.__setattr__(self, "candidate_growth_factor", growth)
        object.__setattr__(self, "seed_fidelities", fidelities)
        object.__setattr__(self, "comparison_seed_count", comparison)
        object.__setattr__(self, "improvement_patience", patience)
        object.__setattr__(self, "min_rounds", minimum_rounds)
        object.__setattr__(
            self,
            "checkpoint_every_n_new_evaluations",
            checkpoint_cadence,
        )
        object.__setattr__(self, "relative_improvement_threshold", threshold)
        object.__setattr__(self, "retention_fraction", retention)

    def validate_for(self, optimization_config: OptimizationConfig) -> None:
        """Validate seed-dependent limits only when the progressive API is used."""

        available = len(optimization_config.tuning_seeds)
        if self.seed_fidelities[-1] > available:
            raise ValueError(
                "Maximum progressive seed fidelity exceeds the tuning seed count."
            )
        if self.comparison_seed_count > available:
            raise ValueError(
                "comparison_seed_count exceeds the tuning seed count."
            )


def relative_loss_improvement(previous_loss: float, selected_loss: float) -> float:
    """Return the existing nonnegative relative loss decrease.

    The fixed-budget convergence policy compares this value to
    ``relative_improvement_threshold`` without changing the historical strict
    ``<`` low-improvement test. Thus an improvement exactly at the threshold is
    sufficient and resets the low-improvement streak.
    """

    previous = float(previous_loss)
    selected = float(selected_loss)
    if previous == 0.0:
        return 0.0
    return max(0.0, (previous - selected) / previous)


def is_sufficient_relative_improvement(
    relative_improvement: float, threshold: float
) -> bool:
    """Preserve the policy's inclusive sufficiency boundary (``>=``)."""

    return float(relative_improvement) >= float(threshold)


def _objective_relative_improvement(
    previous: CandidateEvaluation,
    selected: CandidateEvaluation,
    objective: OptimizationObjective,
) -> float:
    return relative_loss_improvement(previous.mean_loss, selected.mean_loss)


def _record_with_budget(
    structural: CandidateRecord,
    trial_budget: int,
    *,
    instance: CEBPInstance,
    total_copies: int,
    optimization_config: OptimizationConfig,
) -> CandidateRecord:
    if not isinstance(structural.parameters, FixedErrorCandidateParameters):
        raise TypeError("Budget refinement requires a fixed-error candidate.")
    derived = derive_candidate(
        structural.parameters,
        n=instance.n,
        d=instance.d,
        total_copies=total_copies,
        optimization_config=optimization_config,
        physical_budget=int(trial_budget),
    )
    return CandidateRecord(
        f"{structural.candidate_id}-budget-{int(trial_budget)}",
        structural.parameters,
        derived,
    )


@dataclass(frozen=True)
class ProgressiveHalvingStageSummary:
    progressive_round_index: int
    candidate_pool_size: int
    seed_fidelity: int
    candidates_entering: int
    candidates_retained: int
    best_candidate_id: str
    best_mean_loss: float
    best_max_loss: float
    new_learner_runs: int
    cache_hits: int


@dataclass(frozen=True)
class ProgressiveRoundSummary:
    round_index: int
    candidate_pool_size: int
    newly_activated_candidates: int
    seed_fidelities_used: Tuple[int, ...]
    halving_summaries: Tuple[ProgressiveHalvingStageSummary, ...]
    round_search_winner_id: str
    incumbent_before_id: Optional[str]
    incumbent_after_id: str
    comparison_seed_count: int
    old_incumbent_comparison_mean_loss: Optional[float]
    challenger_comparison_mean_loss: float
    selected_incumbent_comparison_mean_loss: float
    relative_improvement: Optional[float]
    low_improvement_streak: int
    cumulative_unique_semantic_evaluations: int
    cumulative_actual_learner_runs: int
    cumulative_cache_hits: int
    round_wall_clock_seconds: Optional[float]
    convergence_stop: bool


@dataclass(frozen=True)
class ProgressiveSearchMetadata:
    algorithm: str
    objective_mode: str
    progressive_search_schema: int
    progressive_config: ProgressiveSearchConfig
    search_seed: int
    tuning_seeds: Tuple[int, ...]
    holdout_seeds: Tuple[int, ...]
    round_summaries: Tuple[ProgressiveRoundSummary, ...]
    rounds_completed: int
    final_candidate_pool_size: int
    total_valid_candidates_pregenerated: int
    rejected_invalid_sampling_attempts: int
    termination_reason: str
    checkpoint_enabled: bool
    resumed_from_checkpoint: bool
    holdout_post_selection_only: bool
    fixed_budget_epsilon_refinement_enabled: bool
    budget_refinement_enabled: bool = False
    budget_refinement_trials: Tuple["BudgetRefinementTrialSummary", ...] = ()
    budget_refinement_note: str = "Not applicable."
    fixed_error_threshold_evaluations: Tuple[BudgetThresholdEvaluation, ...] = ()
    fixed_error_computational_diagnostics: Optional[
        "FixedErrorComputationalDiagnostics"
    ] = None

    @property
    def tomography_refinement_enabled(self) -> bool:
        return False

    @property
    def tomography_refinement_trials(self) -> int:
        return 0

    @property
    def tomography_refinement_note(self) -> str:
        return "Legacy epsilon_tom refinement is disabled."

    @property
    def tomography_refinement_summaries(self) -> Tuple[object, ...]:
        return ()


@dataclass(frozen=True)
class BudgetRefinementTrialSummary:
    candidate_id: str
    N_candidate: int
    operationally_valid: bool
    mean_trace_distance: Optional[float]
    mean_error_feasible: Optional[bool]
    error_success_count: Optional[int]
    error_success_fraction: Optional[float]
    required_error_success_count: Optional[int]
    success_fraction_feasible: Optional[bool]
    target_feasible: bool
    selected: bool = False


@dataclass(frozen=True)
class FixedErrorComputationalDiagnostics:
    unique_structural_candidate_count: int
    unique_structural_budget_trial_count: int
    unique_structural_budget_seed_evaluation_count: int
    semantic_cache_hit_count: int
    upward_expansion_trial_count: int
    relative_refinement_trial_count: int
    average_budget_trials_by_seed_fidelity: Tuple[Tuple[int, float], ...]
    raw_candidate_draw_count: int = 0
    precheck_cap_rejection_count: int = 0
    valid_candidate_count: int = 0
    budget_cap_exhausted_count: int = 0
    e2e_evaluation_count: int = 0
    largest_e2e_budget: int = 0
    feasible_candidate_count: int = 0
    mean_error_failed_count: int = 0
    success_fraction_failed_count: int = 0
    cumulative_expansion_max_per_candidate: int = 0
    numeric_safety_event_count: int = 0
    incumbent_probe_count: int = 0
    incumbent_confirmation_probe_count: int = 0
    incumbent_pruned_candidate_count: int = 0
    maximum_tested_physical_budget: int = 0
    average_unique_budget_trials_per_candidate: float = 0.0
    median_unique_budget_trials_per_candidate: float = 0.0
    maximum_unique_budget_trials_per_candidate: int = 0


@dataclass(frozen=True)
class ProgressiveOptimizationResult:
    """Compact output for the best configuration found by the finite search."""

    objective: OptimizationObjective
    best_candidate_id: str
    best_candidate: CandidateParameters | FixedBudgetCandidateParameters | FixedErrorCandidateParameters
    best_derived_candidate: DerivedCandidate
    comparison_evaluation: Optional[CandidateEvaluation]
    holdout_evaluation: Optional[HoldoutEvaluation]
    search_metadata: ProgressiveSearchMetadata
    candidate_catalog: Tuple[CandidateRecord, ...]
    total_candidate_count: int
    rejected_invalid_count: int
    evaluation_attempt_count: int
    actual_learner_run_count: int
    cache_hit_count: int
    unique_cached_seed_evaluation_count: int
    preflight_rejection_count: int
    budget_refinement_actual_learner_run_count: int = 0

    @property
    def final_incumbent(self) -> CandidateParameters | FixedBudgetCandidateParameters | FixedErrorCandidateParameters:
        return self.best_candidate

    @property
    def final_comparison_evaluation(self) -> Optional[CandidateEvaluation]:
        return self.comparison_evaluation

    @property
    def tuning_evaluation(self) -> Optional[CandidateEvaluation]:
        """Compatibility alias used by the older non-progressive result reader."""

        return self.comparison_evaluation

    @property
    def all_candidate_summaries(self) -> Tuple[CandidateEvaluation, ...]:
        return (
            ()
            if self.comparison_evaluation is None
            else (self.comparison_evaluation,)
        )

    @property
    def evaluated_run_count(self) -> int:
        return self.actual_learner_run_count

    @property
    def analytical_refinement_trial_count(self) -> int:
        return 0

    @property
    def refinement_actual_learner_run_count(self) -> int:
        return self.budget_refinement_actual_learner_run_count


@dataclass
class _ProgressiveCounters:
    evaluation_attempt_count: int = 0
    actual_learner_run_count: int = 0
    cache_hit_count: int = 0
    preflight_rejection_count: int = 0
    budget_refinement_actual_learner_run_count: int = 0
    raw_candidate_draw_count: int = 0
    precheck_cap_rejection_count: int = 0
    valid_candidate_count: int = 0
    e2e_evaluation_count: int = 0
    largest_e2e_budget: int = 0


def _progressive_checkpoint_metadata(
    *,
    instance: CEBPInstance,
    objective: OptimizationObjective,
    config: OptimizationConfig,
    search_space: SearchSpace,
    records: Tuple[CandidateRecord, ...],
    progressive_config: ProgressiveSearchConfig,
) -> Dict[str, Any]:
    metadata = _checkpoint_metadata(
        instance=instance,
        objective=objective,
        config=config,
        search_space=search_space,
        records=records,
    )
    metadata.update(
        {
            "progressive_search_schema": (
                FIXED_ERROR_PROGRESSIVE_SEARCH_SCHEMA
                if objective.mode is OptimizationMode.FIXED_ERROR_MIN_COPIES
                else PROGRESSIVE_SEARCH_SCHEMA
            ),
            "progressive_config": _json_safe(asdict(progressive_config)),
            "maximum_candidate_catalog_size": len(records),
        }
    )
    return metadata


def _stage_from_json(value: Dict[str, Any]) -> ProgressiveHalvingStageSummary:
    return ProgressiveHalvingStageSummary(**value)


def _round_from_json(value: Dict[str, Any]) -> ProgressiveRoundSummary:
    data = dict(value)
    data["seed_fidelities_used"] = tuple(data["seed_fidelities_used"])
    data["halving_summaries"] = tuple(
        _stage_from_json(dict(item)) for item in data["halving_summaries"]
    )
    return ProgressiveRoundSummary(**data)


def _aggregate_from_cache(
    record: CandidateRecord,
    seeds: Tuple[int, ...],
    *,
    cache: Dict[Tuple[str, int], SeedEvaluation],
    total_copies: int,
    optimization_config: OptimizationConfig,
    objective: OptimizationObjective,
    holdout: bool = False,
) -> CandidateEvaluation:
    evaluations = []
    for seed in seeds:
        key = evaluation_identity(
            record.derived, seed, total_copies, optimization_config, objective
        )
        if key not in cache:
            raise RuntimeError("Checkpoint run_state references an incomplete evaluation.")
        evaluations.append(cache[key])
    return aggregate_candidate_evaluations(
        record.candidate_id,
        evaluations,
        objective=objective,
        holdout=holdout,
        N_candidate=int(record.derived.physical_copy_budget or total_copies),
    )


def _validate_progressive_call(
    instance: CEBPInstance,
    total_copies: int,
    optimization_config: OptimizationConfig,
    progressive_config: ProgressiveSearchConfig,
    initial_candidates: Sequence[CandidateParameters | FixedBudgetCandidateParameters | FixedErrorCandidateParameters],
) -> OptimizationObjective:
    if not isinstance(instance, CEBPInstance):
        raise TypeError("instance must be a CEBPInstance.")
    objective = optimization_config.effective_objective
    if (
        objective.mode is not OptimizationMode.FIXED_ERROR_MIN_COPIES
        and int(total_copies) != int(optimization_config.total_copies)
    ):
        raise ValueError("total_copies disagrees with OptimizationConfig.total_copies.")
    if instance.d == 1 and objective.mode not in (
        OptimizationMode.FIXED_BUDGET_MIN_ERROR,
        OptimizationMode.FIXED_ERROR_MIN_COPIES,
    ):
        raise NotImplementedError(
            "d=1 progressive optimization is supported only for practical graceful modes."
        )
    if (
        objective.mode is OptimizationMode.FIXED_BUDGET_MIN_ERROR
        and objective.copy_ceiling != int(total_copies)
    ):
        raise ValueError("Objective copy_ceiling must equal total_copies.")
    if objective.mode not in (
        OptimizationMode.FIXED_BUDGET_MIN_ERROR,
        OptimizationMode.FIXED_ERROR_MIN_COPIES,
    ):
        raise ValueError(
            "Progressive optimization supports only the two practical objective modes."
        )
    require_trace_distance_objective_available(
        instance, optimization_config, objective
    )
    progressive_config.validate_for(optimization_config)
    if objective.mode is OptimizationMode.FIXED_ERROR_MIN_COPIES:
        if progressive_config.seed_fidelities != FIXED_ERROR_SEED_FIDELITIES:
            raise ValueError(
                "fixed_error_min_copies requires seed_fidelities=(1,2,4,8,16)."
            )
        if progressive_config.comparison_seed_count != FIXED_ERROR_FINAL_SEED_COUNT:
            raise ValueError(
                "fixed_error_min_copies requires comparison_seed_count=16."
            )
    return objective


def _threshold_from_json(value: Dict[str, Any]) -> BudgetThresholdEvaluation:
    data = dict(value)
    data["budget_trials"] = tuple(
        FixedErrorBudgetTrial(**dict(item)) for item in data.get("budget_trials", ())
    )
    data["tested_budgets"] = tuple(
        int(item) for item in data.get("tested_budgets", ())
    )
    data["non_monotonic_observations"] = tuple(
        (int(first), int(second))
        for first, second in data.get("non_monotonic_observations", ())
    )
    return BudgetThresholdEvaluation(**data)


def _optimize_fixed_error_progressive(
    *,
    instance: CEBPInstance,
    total_copies: int,
    optimization_config: OptimizationConfig,
    search_space: SearchSpace,
    progressive: ProgressiveSearchConfig,
    initial_candidates: Sequence[
        CandidateParameters | FixedBudgetCandidateParameters | FixedErrorCandidateParameters
    ],
) -> ProgressiveOptimizationResult:
    """Search structural candidates by their adaptive empirical thresholds."""

    objective = optimization_config.effective_objective
    sampling_diagnostics: Dict[str, int] = {}
    records, rejected = _sample_valid_candidates(
        instance=instance,
        total_copies=total_copies,
        optimization_config=optimization_config,
        search_space=search_space,
        search_seed=optimization_config.search_seed,
        initial_candidates=initial_candidates,
        target_count=progressive.max_candidates,
        sampling_diagnostics=sampling_diagnostics,
    )
    record_by_id = {record.candidate_id: record for record in records}
    cache: Dict[Tuple[str, int], SeedEvaluation] = {}
    counters = _ProgressiveCounters(
        raw_candidate_draw_count=int(
            sampling_diagnostics.get(
                "raw_candidate_draw_count", len(records) + rejected
            )
        ),
        precheck_cap_rejection_count=int(
            sampling_diagnostics.get("precheck_cap_rejection_count", 0)
        ),
        valid_candidate_count=int(
            sampling_diagnostics.get("valid_candidate_count", len(records))
        ),
    )
    tested_by_candidate: Dict[str, set[int]] = {
        record.candidate_id: set() for record in records
    }
    candidate_search_state: Dict[str, Dict[str, Any]] = {
        record.candidate_id: {
            "cumulative_expansion_trial_count": 0,
            "terminal_search_status": None,
            "incumbent_probe_count": 0,
            "incumbent_confirmation_probe_count": 0,
            "numeric_safety_event_count": 0,
            "incumbent_pruned": False,
            "largest_tested_budget": None,
            "estimated_min_budget": None,
            "low_infeasible_budget": None,
            "high_feasible_budget": None,
        }
        for record in records
    }
    threshold_by_key: Dict[str, BudgetThresholdEvaluation] = {}
    round_summaries: list[ProgressiveRoundSummary] = []
    incumbent_id: Optional[str] = None
    low_improvement_streak = 0
    next_round_index = 1
    resumed = False
    termination_reason: Optional[str] = None
    pending_checkpoint_completions = 0
    run_state: Dict[str, Any] = {
        "controller_phase": "fixed_error_structural_threshold_search",
        "fixed_error_controller_schema": 1,
        "next_progressive_round": 1,
        "tested_budgets_by_candidate": {},
        "candidate_search_state": {},
        "threshold_results_by_key": {},
        "completed_progressive_round_summaries": [],
        "incumbent_candidate_id": None,
        "low_improvement_streak": 0,
        "termination_reason": None,
        "counters": asdict(counters),
    }
    store: Optional[CheckpointStore] = None
    if optimization_config.checkpoint_path is not None:
        store = CheckpointStore(
            optimization_config.checkpoint_path,
            expected_metadata=_progressive_checkpoint_metadata(
                instance=instance,
                objective=objective,
                config=optimization_config,
                search_space=search_space,
                records=records,
                progressive_config=progressive,
            ),
            every_n_evaluations=progressive.checkpoint_every_n_new_evaluations,
        )
        if optimization_config.resume_from_checkpoint:
            cache, loaded = store.load()
            run_state.update(loaded)
            counters = _ProgressiveCounters(**dict(run_state.get("counters", {})))
            tested_by_candidate.update(
                {
                    str(candidate_id): {int(item) for item in budgets}
                    for candidate_id, budgets in run_state.get(
                        "tested_budgets_by_candidate", {}
                    ).items()
                }
            )
            candidate_search_state.update(
                {
                    str(candidate_id): dict(value)
                    for candidate_id, value in run_state.get(
                        "candidate_search_state", {}
                    ).items()
                }
            )
            threshold_by_key = {
                str(key): _threshold_from_json(dict(value))
                for key, value in run_state.get("threshold_results_by_key", {}).items()
            }
            round_summaries = [
                _round_from_json(dict(item))
                for item in run_state.get("completed_progressive_round_summaries", ())
            ]
            incumbent_id = run_state.get("incumbent_candidate_id")
            low_improvement_streak = int(run_state.get("low_improvement_streak", 0))
            next_round_index = int(run_state.get("next_progressive_round", 1))
            termination_reason = run_state.get("termination_reason")
            resumed = True

    def sync_state() -> None:
        run_state.update(
            {
                "tested_budgets_by_candidate": {
                    candidate_id: sorted(budgets)
                    for candidate_id, budgets in tested_by_candidate.items()
                    if budgets
                },
                "candidate_search_state": _json_safe(candidate_search_state),
                "threshold_results_by_key": {
                    key: _json_safe(asdict(value))
                    for key, value in threshold_by_key.items()
                },
                "completed_progressive_round_summaries": _json_safe(
                    [asdict(item) for item in round_summaries]
                ),
                "incumbent_candidate_id": incumbent_id,
                "low_improvement_streak": low_improvement_streak,
                "next_progressive_round": next_round_index,
                "termination_reason": termination_reason,
                "counters": asdict(counters),
            }
        )

    def checkpoint(*, complete: bool = False) -> None:
        nonlocal pending_checkpoint_completions
        sync_state()
        if store is not None:
            store.save(cache, run_state, status="complete" if complete else "running")
            pending_checkpoint_completions = 0

    if store is not None and not optimization_config.resume_from_checkpoint:
        checkpoint()

    def evaluate_structural_budget(
        record: CandidateRecord,
        budget: int,
        fidelity: int,
    ) -> CandidateEvaluation:
        scientific_cap = objective.scientific_copy_cap
        if scientific_cap is not None and int(budget) > int(scientific_cap):
            raise ValueError(
                f"Refusing fixed-error budget {int(budget)} above "
                f"scientific_copy_cap={int(scientific_cap)} before derivation/E2E."
            )
        derived = derive_candidate(
            record.parameters,
            n=instance.n,
            d=instance.d,
            total_copies=total_copies,
            optimization_config=optimization_config,
            physical_budget=int(budget),
        )
        trial_id = f"{record.candidate_id}-budget-{int(budget)}"
        seeds = tuple(optimization_config.tuning_seeds[: int(fidelity)])

        def completed(_key: Tuple[str, int], evaluation: SeedEvaluation) -> None:
            nonlocal pending_checkpoint_completions
            counters.evaluation_attempt_count += 1
            counters.budget_refinement_actual_learner_run_count += 1
            if evaluation.preflight_rejected:
                counters.preflight_rejection_count += 1
            else:
                counters.actual_learner_run_count += 1
                counters.e2e_evaluation_count += 1
                counters.largest_e2e_budget = max(
                    counters.largest_e2e_budget,
                    int(budget),
                )
            sync_state()
            if store is not None:
                pending_checkpoint_completions += 1

        def cache_hit(_key: Tuple[str, int]) -> None:
            counters.evaluation_attempt_count += 1
            counters.cache_hit_count += 1

        if scientific_cap is not None and int(budget) > int(scientific_cap):
            raise ValueError(
                f"Refusing fixed-error budget {int(budget)} above "
                f"scientific_copy_cap={int(scientific_cap)} immediately before evaluation."
            )
        return evaluate_candidate(
            instance,
            trial_id,
            derived,
            seeds,
            total_copies,
            optimization_config,
            cache=cache,
            objective=objective,
            on_evaluation=completed,
            on_cache_hit=cache_hit,
        )

    def threshold_for(
        record: CandidateRecord,
        fidelity: int,
        key: str,
        *,
        incumbent_budget_hint: Optional[int] = None,
        current_incumbent_id: Optional[str] = None,
    ) -> BudgetThresholdEvaluation:
        if key in threshold_by_key:
            return threshold_by_key[key]
        if not isinstance(record.parameters, FixedErrorCandidateParameters):
            raise TypeError(
                "Modern fixed-error progressive search requires structural candidates."
            )
        practical = minimum_practical_copy_budget(record.parameters, d=instance.d)
        state = candidate_search_state[record.candidate_id]

        def controller_state_changed(snapshot: Dict[str, object]) -> None:
            nonlocal pending_checkpoint_completions
            tested_by_candidate[record.candidate_id].update(
                int(item) for item in snapshot.get("tested_budgets", ())
            )
            candidate_search_state[record.candidate_id] = {
                key: value
                for key, value in snapshot.items()
                if key != "tested_budgets"
            }
            if store is not None:
                sync_state()
                for _item in range(pending_checkpoint_completions):
                    store.note_completed(cache, run_state)
                pending_checkpoint_completions = 0

        result = estimate_minimum_feasible_budget(
            record.candidate_id,
            seed_fidelity=int(fidelity),
            final_seed_count=FIXED_ERROR_FINAL_SEED_COUNT,
            practical_minimum_budget=practical,
            evaluate_budget=lambda budget: evaluate_structural_budget(
                record, budget, fidelity
            ),
            previous_tested_budgets=tested_by_candidate[record.candidate_id],
            previous_cumulative_expansion_trials=int(
                state.get("cumulative_expansion_trial_count", 0)
            ),
            previous_terminal_search_status=state.get("terminal_search_status"),
            previous_incumbent_probe_count=int(
                state.get("incumbent_probe_count", 0)
            ),
            previous_incumbent_confirmation_probe_count=int(
                state.get("incumbent_confirmation_probe_count", 0)
            ),
            previous_numeric_safety_event_count=int(
                state.get("numeric_safety_event_count", 0)
            ),
            initial_budget_hint=optimization_config.fixed_error_initial_budget_hint,
            incumbent_budget_hint=incumbent_budget_hint,
            is_current_incumbent=(record.candidate_id == current_incumbent_id),
            incumbent_confirmation_probes=(
                optimization_config.fixed_error_incumbent_confirmation_probes
            ),
            growth_factor=optimization_config.fixed_error_budget_growth_factor,
            relative_tolerance=(
                optimization_config.fixed_error_budget_relative_tolerance
            ),
            max_expansion_rounds=(
                optimization_config.fixed_error_max_budget_expansion_rounds
            ),
            hard_safety_budget=optimization_config.fixed_error_hard_safety_budget,
            scientific_copy_cap=objective.scientific_copy_cap,
            on_controller_state=controller_state_changed,
        )
        tested_by_candidate[record.candidate_id].update(result.tested_budgets)
        candidate_search_state[record.candidate_id] = {
            "cumulative_expansion_trial_count": (
                result.cumulative_expansion_trial_count
            ),
            "terminal_search_status": result.terminal_search_status,
            "incumbent_probe_count": result.incumbent_probe_count,
            "incumbent_confirmation_probe_count": (
                result.incumbent_confirmation_probe_count
            ),
            "numeric_safety_event_count": result.numeric_safety_event_count,
            "incumbent_pruned": result.incumbent_pruned,
            "largest_tested_budget": result.largest_tested_budget,
            "estimated_min_budget": result.estimated_min_budget,
            "low_infeasible_budget": result.low_infeasible_budget,
            "high_feasible_budget": result.high_feasible_budget,
        }
        threshold_by_key[key] = result
        checkpoint()
        return result

    pool_sizes = []
    size = progressive.initial_candidates
    while True:
        pool_sizes.append(size)
        if size >= progressive.max_candidates:
            break
        size = min(
            progressive.max_candidates,
            size * progressive.candidate_growth_factor,
        )

    for round_index, pool_size in enumerate(pool_sizes, start=1):
        if round_index < next_round_index:
            continue
        stage_ids = [record.candidate_id for record in records[:pool_size]]
        if incumbent_id in stage_ids:
            stage_ids.remove(incumbent_id)
            stage_ids.insert(0, incumbent_id)
        stage_summaries: list[ProgressiveHalvingStageSummary] = []
        round_runs_before = counters.actual_learner_run_count
        round_hits_before = counters.cache_hit_count
        for stage_index, fidelity in enumerate(progressive.seed_fidelities):
            runs_before = counters.actual_learner_run_count
            hits_before = counters.cache_hit_count
            results: list[BudgetThresholdEvaluation] = []
            stage_incumbent_id: Optional[str] = None
            stage_incumbent_budget: Optional[int] = None
            for candidate_id in stage_ids:
                result = threshold_for(
                    record_by_id[candidate_id],
                    fidelity,
                    f"r{round_index}:s{fidelity}:{candidate_id}",
                    incumbent_budget_hint=stage_incumbent_budget,
                    current_incumbent_id=stage_incumbent_id,
                )
                results.append(result)
                if result.threshold_found and (
                    stage_incumbent_budget is None
                    or int(result.estimated_min_budget) < stage_incumbent_budget
                ):
                    stage_incumbent_id = result.structural_candidate_id
                    stage_incumbent_budget = int(result.estimated_min_budget)
            ranked = sorted(results, key=fixed_error_threshold_ranking_key)
            best = ranked[0]
            keep = (
                1
                if stage_index + 1 == len(progressive.seed_fidelities)
                else max(
                    1,
                    int(math.ceil(len(stage_ids) * progressive.retention_fraction)),
                )
            )
            stage_ids = [item.structural_candidate_id for item in ranked[:keep]]
            stage_summaries.append(
                ProgressiveHalvingStageSummary(
                    progressive_round_index=round_index,
                    candidate_pool_size=pool_size,
                    seed_fidelity=int(fidelity),
                    candidates_entering=len(results),
                    candidates_retained=keep,
                    best_candidate_id=best.structural_candidate_id,
                    best_mean_loss=float(best.final_mean_trace_distance or 1.0),
                    best_max_loss=float(best.final_mean_trace_distance or 1.0),
                    new_learner_runs=(
                        counters.actual_learner_run_count - runs_before
                    ),
                    cache_hits=counters.cache_hit_count - hits_before,
                )
            )

        challenger_id = stage_ids[0]
        challenger = threshold_for(
            record_by_id[challenger_id],
            FIXED_ERROR_FINAL_SEED_COUNT,
            f"r{round_index}:final:{challenger_id}",
            current_incumbent_id=challenger_id,
        )
        incumbent_before_id = incumbent_id
        previous: Optional[BudgetThresholdEvaluation] = None
        if incumbent_before_id is None:
            incumbent_id = challenger_id
            selected = challenger
            relative_improvement = None
            low_improvement_streak = 0
        else:
            previous = threshold_for(
                record_by_id[incumbent_before_id],
                FIXED_ERROR_FINAL_SEED_COUNT,
                f"r{round_index}:incumbent:{incumbent_before_id}",
                current_incumbent_id=incumbent_before_id,
            )
            if fixed_error_threshold_ranking_key(challenger) < fixed_error_threshold_ranking_key(previous):
                incumbent_id = challenger_id
                selected = challenger
            else:
                selected = previous
            if previous.threshold_found and selected.threshold_found:
                relative_improvement = relative_loss_improvement(
                    float(previous.estimated_min_budget),
                    float(selected.estimated_min_budget),
                )
            else:
                relative_improvement = 0.0
            if not is_sufficient_relative_improvement(
                relative_improvement,
                progressive.relative_improvement_threshold,
            ):
                low_improvement_streak += 1
            else:
                low_improvement_streak = 0

        converged = bool(
            progressive.relative_improvement_threshold > 0.0
            and round_index >= progressive.min_rounds
            and low_improvement_streak >= progressive.improvement_patience
        )
        at_maximum = pool_size >= progressive.max_candidates
        if converged:
            termination_reason = "converged"
        elif at_maximum:
            termination_reason = (
                "max_structural_candidates"
                if selected.threshold_found
                else "max_structural_candidates_without_threshold_before_safety_limit"
            )
        next_round_index = round_index + 1
        previous_pool = 0 if not round_summaries else round_summaries[-1].candidate_pool_size
        round_summaries.append(
            ProgressiveRoundSummary(
                round_index=round_index,
                candidate_pool_size=pool_size,
                newly_activated_candidates=pool_size - previous_pool,
                seed_fidelities_used=tuple(progressive.seed_fidelities),
                halving_summaries=tuple(stage_summaries),
                round_search_winner_id=challenger_id,
                incumbent_before_id=incumbent_before_id,
                incumbent_after_id=str(incumbent_id),
                comparison_seed_count=FIXED_ERROR_FINAL_SEED_COUNT,
                old_incumbent_comparison_mean_loss=(
                    previous.final_mean_trace_distance if previous is not None else None
                ),
                challenger_comparison_mean_loss=float(
                    challenger.final_mean_trace_distance or 1.0
                ),
                selected_incumbent_comparison_mean_loss=float(
                    selected.final_mean_trace_distance or 1.0
                ),
                relative_improvement=relative_improvement,
                low_improvement_streak=low_improvement_streak,
                cumulative_unique_semantic_evaluations=len(cache),
                cumulative_actual_learner_runs=counters.actual_learner_run_count,
                cumulative_cache_hits=counters.cache_hit_count,
                round_wall_clock_seconds=None,
                convergence_stop=termination_reason is not None,
            )
        )
        checkpoint()
        if termination_reason is not None:
            break

    if incumbent_id is None:
        raise RuntimeError("Fixed-error structural search produced no incumbent.")
    final_threshold = threshold_for(
        record_by_id[incumbent_id],
        FIXED_ERROR_FINAL_SEED_COUNT,
        f"final:{incumbent_id}",
        current_incumbent_id=incumbent_id,
    )
    structural_record = record_by_id[incumbent_id]
    if final_threshold.tested_budgets:
        selected_budget = (
            final_threshold.estimated_min_budget
            if final_threshold.threshold_found
            else max(final_threshold.tested_budgets)
        )
        selected_derived = derive_candidate(
            structural_record.parameters,
            n=instance.n,
            d=instance.d,
            total_copies=total_copies,
            optimization_config=optimization_config,
            physical_budget=int(selected_budget),
        )
        comparison_evaluation: Optional[CandidateEvaluation] = (
            evaluate_structural_budget(
                structural_record,
                int(selected_budget),
                FIXED_ERROR_FINAL_SEED_COUNT,
            )
        )
    else:
        # A configured safety budget may lie below every candidate's practical
        # execution minimum.  Return the structured safety status without
        # performing an out-of-policy learner trial.
        selected_derived = structural_record.derived
        comparison_evaluation = None
    selected_record = CandidateRecord(
        incumbent_id, structural_record.parameters, selected_derived
    )
    final_catalog = tuple(
        selected_record if record.candidate_id == incumbent_id else record
        for record in records
    )
    threshold_values = tuple(threshold_by_key.values())
    unique_budget_trials = {
        (result.structural_candidate_id, budget)
        for result in threshold_values
        for budget in result.tested_budgets
    }
    unique_budget_counts = [
        len(
            {
                budget
                for candidate_id, budget in unique_budget_trials
                if candidate_id == record.candidate_id
            }
        )
        for record in records
    ]
    averages = []
    for fidelity in progressive.seed_fidelities:
        matching = [
            result for result in threshold_values if result.seed_fidelity == fidelity
        ]
        averages.append(
            (
                int(fidelity),
                float(
                    np.mean([len(result.tested_budgets) for result in matching])
                ) if matching else 0.0,
            )
        )
    unique_scientific_trial_statuses = {
        (
            result.structural_candidate_id,
            result.seed_fidelity,
            trial.budget,
        ): trial.scientific_status
        for result in threshold_values
        for trial in result.budget_trials
    }
    diagnostics = FixedErrorComputationalDiagnostics(
        unique_structural_candidate_count=len(records),
        unique_structural_budget_trial_count=len(unique_budget_trials),
        unique_structural_budget_seed_evaluation_count=len(cache),
        semantic_cache_hit_count=counters.cache_hit_count,
        upward_expansion_trial_count=sum(
            int(state.get("cumulative_expansion_trial_count", 0))
            for state in candidate_search_state.values()
        ),
        relative_refinement_trial_count=sum(
            result.refinement_trial_count for result in threshold_values
        ),
        average_budget_trials_by_seed_fidelity=tuple(averages),
        raw_candidate_draw_count=counters.raw_candidate_draw_count,
        precheck_cap_rejection_count=counters.precheck_cap_rejection_count,
        valid_candidate_count=counters.valid_candidate_count,
        budget_cap_exhausted_count=sum(
            state.get("terminal_search_status") == BUDGET_CAP_EXHAUSTED
            for state in candidate_search_state.values()
        ),
        e2e_evaluation_count=counters.e2e_evaluation_count,
        largest_e2e_budget=counters.largest_e2e_budget,
        feasible_candidate_count=len(
            {
                result.structural_candidate_id
                for result in threshold_values
                if result.threshold_found
            }
        ),
        mean_error_failed_count=sum(
            status == MEAN_ERROR_FAILED
            for status in unique_scientific_trial_statuses.values()
        ),
        success_fraction_failed_count=sum(
            status == SUCCESS_FRACTION_FAILED
            for status in unique_scientific_trial_statuses.values()
        ),
        cumulative_expansion_max_per_candidate=max(
            (
                int(state.get("cumulative_expansion_trial_count", 0))
                for state in candidate_search_state.values()
            ),
            default=0,
        ),
        numeric_safety_event_count=sum(
            int(state.get("numeric_safety_event_count", 0))
            for state in candidate_search_state.values()
        ),
        incumbent_probe_count=sum(
            int(state.get("incumbent_probe_count", 0))
            for state in candidate_search_state.values()
        ),
        incumbent_confirmation_probe_count=sum(
            int(state.get("incumbent_confirmation_probe_count", 0))
            for state in candidate_search_state.values()
        ),
        incumbent_pruned_candidate_count=sum(
            bool(state.get("incumbent_pruned", False))
            for state in candidate_search_state.values()
        ),
        maximum_tested_physical_budget=max(
            (budget for _candidate_id, budget in unique_budget_trials),
            default=0,
        ),
        average_unique_budget_trials_per_candidate=(
            float(np.mean(unique_budget_counts)) if unique_budget_counts else 0.0
        ),
        median_unique_budget_trials_per_candidate=(
            float(np.median(unique_budget_counts)) if unique_budget_counts else 0.0
        ),
        maximum_unique_budget_trials_per_candidate=max(
            unique_budget_counts, default=0
        ),
    )
    metadata = ProgressiveSearchMetadata(
        algorithm=(
            "progressive structural search with adaptive empirical "
            "minimum-budget bracketing"
        ),
        objective_mode=objective.mode.value,
        progressive_search_schema=FIXED_ERROR_PROGRESSIVE_SEARCH_SCHEMA,
        progressive_config=progressive,
        search_seed=int(optimization_config.search_seed),
        tuning_seeds=tuple(optimization_config.tuning_seeds),
        holdout_seeds=(),
        round_summaries=tuple(round_summaries),
        rounds_completed=len(round_summaries),
        final_candidate_pool_size=round_summaries[-1].candidate_pool_size,
        total_valid_candidates_pregenerated=len(records),
        rejected_invalid_sampling_attempts=rejected,
        termination_reason=str(termination_reason),
        checkpoint_enabled=store is not None,
        resumed_from_checkpoint=resumed,
        holdout_post_selection_only=False,
        fixed_budget_epsilon_refinement_enabled=False,
        budget_refinement_enabled=True,
        budget_refinement_trials=(),
        budget_refinement_note=(
            "Adaptive upward bracketing and relative threshold refinement are "
            "integral to every structural-candidate evaluation."
        ),
        fixed_error_threshold_evaluations=threshold_values,
        fixed_error_computational_diagnostics=diagnostics,
    )
    checkpoint(complete=True)
    return ProgressiveOptimizationResult(
        objective=objective,
        best_candidate_id=incumbent_id,
        best_candidate=selected_record.parameters,
        best_derived_candidate=selected_derived,
        comparison_evaluation=comparison_evaluation,
        holdout_evaluation=None,
        search_metadata=metadata,
        candidate_catalog=final_catalog,
        total_candidate_count=len(final_catalog),
        rejected_invalid_count=rejected,
        evaluation_attempt_count=counters.evaluation_attempt_count,
        actual_learner_run_count=counters.actual_learner_run_count,
        cache_hit_count=counters.cache_hit_count,
        unique_cached_seed_evaluation_count=len(cache),
        preflight_rejection_count=counters.preflight_rejection_count,
        budget_refinement_actual_learner_run_count=(
            counters.budget_refinement_actual_learner_run_count
        ),
    )


def optimize_cebp_parameters_progressive(
    instance: CEBPInstance,
    total_copies: int,
    optimization_config: OptimizationConfig,
    search_space: Optional[SearchSpace] = None,
    *,
    progressive_config: Optional[ProgressiveSearchConfig] = None,
    initial_candidates: Sequence[CandidateParameters | FixedBudgetCandidateParameters | FixedErrorCandidateParameters] = (),
) -> ProgressiveOptimizationResult:
    """Run nested progressive practical search.

    Fixed-budget mode retains its post-selection holdout. Modern fixed-error
    mode finishes after its 16 tuning-seed selection/refinement criterion.
    """

    progressive = progressive_config or ProgressiveSearchConfig()
    if not isinstance(progressive, ProgressiveSearchConfig):
        raise TypeError("progressive_config must be ProgressiveSearchConfig or None.")
    objective = _validate_progressive_call(
        instance,
        total_copies,
        optimization_config,
        progressive,
        initial_candidates,
    )
    space = search_space or SearchSpace()
    if objective.mode is OptimizationMode.FIXED_ERROR_MIN_COPIES:
        return _optimize_fixed_error_progressive(
            instance=instance,
            total_copies=total_copies,
            optimization_config=optimization_config,
            search_space=space,
            progressive=progressive,
            initial_candidates=initial_candidates,
        )
    records, rejected = _sample_valid_candidates(
        instance=instance,
        total_copies=total_copies,
        optimization_config=optimization_config,
        search_space=space,
        search_seed=optimization_config.search_seed,
        initial_candidates=initial_candidates,
        target_count=progressive.max_candidates,
    )
    record_by_id = {record.candidate_id: record for record in records}

    cache: Dict[Tuple[str, int], SeedEvaluation] = {}
    counters = _ProgressiveCounters()
    round_summaries: list[ProgressiveRoundSummary] = []
    incumbent_id: Optional[str] = None
    low_improvement_streak = 0
    current_pool_size = progressive.initial_candidates
    termination_reason: Optional[str] = None
    resumed = False
    run_state: Dict[str, Any] = {
        "controller_phase": "round_start",
        "current_progressive_round": 1,
        "current_active_candidate_pool_size": current_pool_size,
        "current_halving_stage_index": 0,
        "current_seed_fidelity": None,
        "alive_candidate_ids": [],
        "current_stage_candidate_ids": [],
        "current_stage_candidate_index": 0,
        "current_evaluation_candidate_id": None,
        "current_evaluation_next_seed_index": 0,
        "current_stage_actual_learner_runs_before": 0,
        "current_stage_cache_hits_before": 0,
        "current_round_partial_halving_summaries": [],
        "current_round_search_winner_id": None,
        "incumbent_before_round_id": None,
        "challenger_comparison_completed": False,
        "incumbent_comparison_completed": False,
        "final_comparison_completed": False,
        "completed_progressive_round_summaries": [],
        "incumbent_candidate_id": None,
        "low_improvement_streak": 0,
        "termination_reason": None,
        "holdout_started": False,
        "holdout_completed": False,
        "counters": asdict(counters),
    }
    store: Optional[CheckpointStore] = None
    if optimization_config.checkpoint_path is not None:
        store = CheckpointStore(
            optimization_config.checkpoint_path,
            expected_metadata=_progressive_checkpoint_metadata(
                instance=instance,
                objective=objective,
                config=optimization_config,
                search_space=space,
                records=records,
                progressive_config=progressive,
            ),
            every_n_evaluations=progressive.checkpoint_every_n_new_evaluations,
        )
        if optimization_config.resume_from_checkpoint:
            cache, loaded = store.load()
            run_state.update(loaded)
            counters = _ProgressiveCounters(**dict(run_state.get("counters", {})))
            round_summaries = [
                _round_from_json(dict(item))
                for item in run_state.get("completed_progressive_round_summaries", ())
            ]
            incumbent_id = run_state.get("incumbent_candidate_id")
            low_improvement_streak = int(run_state.get("low_improvement_streak", 0))
            current_pool_size = int(
                run_state.get(
                    "current_active_candidate_pool_size",
                    progressive.initial_candidates,
                )
            )
            termination_reason = run_state.get("termination_reason")
            resumed = True
        else:
            store.save(cache, run_state, status="running")

    def sync_state() -> None:
        run_state["counters"] = asdict(counters)
        run_state["completed_progressive_round_summaries"] = _json_safe(
            [asdict(item) for item in round_summaries]
        )
        run_state["incumbent_candidate_id"] = incumbent_id
        run_state["low_improvement_streak"] = low_improvement_streak
        run_state["termination_reason"] = termination_reason

    def force_checkpoint(*, status: str = "running") -> None:
        sync_state()
        if store is not None:
            store.save(cache, run_state, status=status)

    def evaluate_record_suffix(
        record: CandidateRecord,
        seeds: Tuple[int, ...],
        *,
        start_index: int,
        holdout: bool = False,
    ) -> CandidateEvaluation:
        if not 0 <= start_index <= len(seeds):
            raise RuntimeError("Checkpoint seed offset is outside the requested prefix.")
        next_seed_index = int(start_index)

        def advance_request() -> None:
            nonlocal next_seed_index
            next_seed_index += 1
            run_state["current_evaluation_next_seed_index"] = next_seed_index

        def completed(_key: Tuple[str, int], evaluation: SeedEvaluation) -> None:
            advance_request()
            counters.evaluation_attempt_count += 1
            if evaluation.preflight_rejected:
                counters.preflight_rejection_count += 1
            else:
                counters.actual_learner_run_count += 1
            sync_state()
            if store is not None:
                store.note_completed(cache, run_state)

        def cache_hit(_key: Tuple[str, int]) -> None:
            advance_request()
            counters.evaluation_attempt_count += 1
            counters.cache_hit_count += 1
            sync_state()

        remaining_seeds = seeds[start_index:]
        if remaining_seeds:
            evaluate_candidate(
                instance,
                record.candidate_id,
                record.derived,
                remaining_seeds,
                total_copies,
                optimization_config,
                cache=cache,
                holdout=holdout,
                objective=objective,
                on_evaluation=completed,
                on_cache_hit=cache_hit,
            )
        return _aggregate_from_cache(
            record,
            seeds,
            cache=cache,
            total_copies=total_copies,
            optimization_config=optimization_config,
            objective=objective,
            holdout=holdout,
        )

    comparison_seeds = tuple(
        optimization_config.tuning_seeds[: progressive.comparison_seed_count]
    )
    holdout_seeds = tuple(optimization_config.holdout_seeds)

    while str(run_state.get("controller_phase")) != "complete":
        phase = str(run_state["controller_phase"])
        round_index = int(run_state["current_progressive_round"])
        current_pool_size = int(run_state["current_active_candidate_pool_size"])
        maximum_stage_index = min(
            round_index + 1, len(progressive.seed_fidelities) - 1
        )
        round_fidelities = progressive.seed_fidelities[: maximum_stage_index + 1]

        if phase == "round_start":
            active_ids = [record.candidate_id for record in records[:current_pool_size]]
            run_state.update(
                {
                    "controller_phase": "halving",
                    "current_halving_stage_index": 0,
                    "current_seed_fidelity": round_fidelities[0],
                    "alive_candidate_ids": active_ids,
                    "current_stage_candidate_ids": active_ids,
                    "current_stage_candidate_index": 0,
                    "current_evaluation_candidate_id": active_ids[0],
                    "current_evaluation_next_seed_index": 0,
                    "current_stage_actual_learner_runs_before": (
                        counters.actual_learner_run_count
                    ),
                    "current_stage_cache_hits_before": counters.cache_hit_count,
                    "current_round_partial_halving_summaries": [],
                    "current_round_search_winner_id": None,
                    "incumbent_before_round_id": incumbent_id,
                    "challenger_comparison_completed": False,
                    "incumbent_comparison_completed": False,
                }
            )
            force_checkpoint()
            continue

        if phase == "halving":
            stage_index = int(run_state["current_halving_stage_index"])
            fidelity = int(run_state["current_seed_fidelity"])
            if fidelity != round_fidelities[stage_index]:
                raise RuntimeError("Checkpoint halving fidelity is inconsistent.")
            stage_ids = tuple(run_state["current_stage_candidate_ids"])
            candidate_index = int(run_state["current_stage_candidate_index"])
            seeds = tuple(optimization_config.tuning_seeds[:fidelity])
            while candidate_index < len(stage_ids):
                candidate_id = str(stage_ids[candidate_index])
                run_state["current_evaluation_candidate_id"] = candidate_id
                start_index = int(run_state["current_evaluation_next_seed_index"])
                evaluation = evaluate_record_suffix(
                    record_by_id[candidate_id],
                    seeds,
                    start_index=start_index,
                )
                if tuple(item.learner_seed for item in evaluation.seed_evaluations) != seeds:
                    raise RuntimeError("Nested tuning-seed prefix policy was violated.")
                candidate_index += 1
                run_state["current_stage_candidate_index"] = candidate_index
                run_state["current_evaluation_next_seed_index"] = 0
                run_state["current_evaluation_candidate_id"] = (
                    str(stage_ids[candidate_index])
                    if candidate_index < len(stage_ids)
                    else None
                )
                sync_state()

            evaluations = [
                _aggregate_from_cache(
                    record_by_id[candidate_id],
                    seeds,
                    cache=cache,
                    total_copies=total_copies,
                    optimization_config=optimization_config,
                    objective=objective,
                )
                for candidate_id in stage_ids
            ]
            ranked = sorted(
                evaluations, key=lambda item: candidate_ranking_key(item, objective)
            )
            best = ranked[0]
            if stage_index + 1 < len(round_fidelities):
                keep = max(
                    1,
                    int(math.ceil(len(stage_ids) * progressive.retention_fraction)),
                )
            else:
                keep = 1
            keep_ids = {item.candidate_id for item in ranked[:keep]}
            partial_summaries = [
                _stage_from_json(dict(item))
                for item in run_state["current_round_partial_halving_summaries"]
            ]
            partial_summaries.append(
                ProgressiveHalvingStageSummary(
                    progressive_round_index=round_index,
                    candidate_pool_size=current_pool_size,
                    seed_fidelity=fidelity,
                    candidates_entering=len(stage_ids),
                    candidates_retained=keep,
                    best_candidate_id=best.candidate_id,
                    best_mean_loss=best.mean_loss,
                    best_max_loss=best.max_loss,
                    new_learner_runs=(
                        counters.actual_learner_run_count
                        - int(run_state["current_stage_actual_learner_runs_before"])
                    ),
                    cache_hits=(
                        counters.cache_hit_count
                        - int(run_state["current_stage_cache_hits_before"])
                    ),
                )
            )
            run_state["current_round_partial_halving_summaries"] = _json_safe(
                [asdict(item) for item in partial_summaries]
            )
            if stage_index + 1 < len(round_fidelities):
                next_ids = [
                    candidate_id for candidate_id in stage_ids if candidate_id in keep_ids
                ]
                run_state.update(
                    {
                        "current_halving_stage_index": stage_index + 1,
                        "current_seed_fidelity": round_fidelities[stage_index + 1],
                        "alive_candidate_ids": next_ids,
                        "current_stage_candidate_ids": next_ids,
                        "current_stage_candidate_index": 0,
                        "current_evaluation_candidate_id": next_ids[0],
                        "current_evaluation_next_seed_index": 0,
                        "current_stage_actual_learner_runs_before": (
                            counters.actual_learner_run_count
                        ),
                        "current_stage_cache_hits_before": counters.cache_hit_count,
                    }
                )
            else:
                run_state.update(
                    {
                        "controller_phase": "comparison_challenger",
                        "current_round_search_winner_id": best.candidate_id,
                        "current_evaluation_candidate_id": best.candidate_id,
                        "current_evaluation_next_seed_index": 0,
                    }
                )
            force_checkpoint()
            continue

        if phase == "comparison_challenger":
            challenger_id = str(run_state["current_round_search_winner_id"])
            evaluate_record_suffix(
                record_by_id[challenger_id],
                comparison_seeds,
                start_index=int(run_state["current_evaluation_next_seed_index"]),
            )
            run_state.update(
                {
                    "challenger_comparison_completed": True,
                    "controller_phase": "comparison_incumbent",
                    "current_evaluation_candidate_id": (
                        run_state.get("incumbent_before_round_id")
                    ),
                    "current_evaluation_next_seed_index": 0,
                }
            )
            force_checkpoint()
            continue

        if phase == "comparison_incumbent":
            challenger_id = str(run_state["current_round_search_winner_id"])
            incumbent_before_id = run_state.get("incumbent_before_round_id")
            if incumbent_before_id is not None and incumbent_before_id != challenger_id:
                evaluate_record_suffix(
                    record_by_id[str(incumbent_before_id)],
                    comparison_seeds,
                    start_index=int(run_state["current_evaluation_next_seed_index"]),
                )
            run_state.update(
                {
                    "incumbent_comparison_completed": True,
                    "controller_phase": "round_finalize",
                    "current_evaluation_candidate_id": None,
                    "current_evaluation_next_seed_index": 0,
                }
            )
            force_checkpoint()
            continue

        if phase == "round_finalize":
            round_winner_id = str(run_state["current_round_search_winner_id"])
            incumbent_before_id = run_state.get("incumbent_before_round_id")
            challenger_comparison = _aggregate_from_cache(
                record_by_id[round_winner_id],
                comparison_seeds,
                cache=cache,
                total_copies=total_copies,
                optimization_config=optimization_config,
                objective=objective,
            )
            previous_comparison: Optional[CandidateEvaluation] = None
            if incumbent_before_id is None:
                incumbent_id = round_winner_id
                selected_comparison = challenger_comparison
                relative_improvement: Optional[float] = None
                low_improvement_streak = 0
            else:
                if incumbent_before_id == round_winner_id:
                    previous_comparison = challenger_comparison
                else:
                    previous_comparison = _aggregate_from_cache(
                        record_by_id[str(incumbent_before_id)],
                        comparison_seeds,
                        cache=cache,
                        total_copies=total_copies,
                        optimization_config=optimization_config,
                        objective=objective,
                    )
                if candidate_ranking_key(
                    challenger_comparison, objective
                ) < candidate_ranking_key(previous_comparison, objective):
                    incumbent_id = round_winner_id
                    selected_comparison = challenger_comparison
                else:
                    incumbent_id = str(incumbent_before_id)
                    selected_comparison = previous_comparison
                relative_improvement = _objective_relative_improvement(
                    previous_comparison,
                    selected_comparison,
                    objective,
                )
                if not is_sufficient_relative_improvement(
                    relative_improvement,
                    progressive.relative_improvement_threshold,
                ):
                    low_improvement_streak += 1
                else:
                    low_improvement_streak = 0

            assert incumbent_id is not None
            perfect_zero = selected_comparison.mean_loss == 0.0
            converged = bool(
                progressive.relative_improvement_threshold > 0.0
                and round_index >= progressive.min_rounds
                and low_improvement_streak >= progressive.improvement_patience
            )
            at_maximum = current_pool_size >= progressive.max_candidates
            if (
                perfect_zero
                and progressive.relative_improvement_threshold > 0.0
            ):
                termination_reason = "perfect_zero_loss"
            elif converged:
                termination_reason = "converged"
            elif at_maximum:
                termination_reason = "max_candidates"

            partial_summaries = tuple(
                _stage_from_json(dict(item))
                for item in run_state["current_round_partial_halving_summaries"]
            )
            previous_pool_size = (
                0 if round_index == 1 else round_summaries[-1].candidate_pool_size
            )
            round_summaries.append(
                ProgressiveRoundSummary(
                    round_index=round_index,
                    candidate_pool_size=current_pool_size,
                    newly_activated_candidates=current_pool_size - previous_pool_size,
                    seed_fidelities_used=tuple(round_fidelities),
                    halving_summaries=partial_summaries,
                    round_search_winner_id=round_winner_id,
                    incumbent_before_id=(
                        str(incumbent_before_id)
                        if incumbent_before_id is not None
                        else None
                    ),
                    incumbent_after_id=incumbent_id,
                    comparison_seed_count=progressive.comparison_seed_count,
                    old_incumbent_comparison_mean_loss=(
                        previous_comparison.mean_loss
                        if previous_comparison is not None
                        else None
                    ),
                    challenger_comparison_mean_loss=challenger_comparison.mean_loss,
                    selected_incumbent_comparison_mean_loss=(
                        selected_comparison.mean_loss
                    ),
                    relative_improvement=relative_improvement,
                    low_improvement_streak=low_improvement_streak,
                    cumulative_unique_semantic_evaluations=len(cache),
                    cumulative_actual_learner_runs=counters.actual_learner_run_count,
                    cumulative_cache_hits=counters.cache_hit_count,
                    round_wall_clock_seconds=None,
                    convergence_stop=termination_reason is not None,
                )
            )
            if termination_reason is None:
                current_pool_size = min(
                    progressive.max_candidates,
                    current_pool_size * progressive.candidate_growth_factor,
                )
                run_state.update(
                    {
                        "controller_phase": "round_start",
                        "current_progressive_round": round_index + 1,
                        "current_active_candidate_pool_size": current_pool_size,
                    }
                )
            else:
                run_state.update(
                    {
                        "controller_phase": "final_comparison",
                        "current_evaluation_candidate_id": incumbent_id,
                        "current_evaluation_next_seed_index": 0,
                    }
                )
            force_checkpoint()
            continue

        if phase == "final_comparison":
            assert incumbent_id is not None
            evaluate_record_suffix(
                record_by_id[incumbent_id],
                comparison_seeds,
                start_index=int(run_state["current_evaluation_next_seed_index"]),
            )
            run_state.update(
                {
                    "final_comparison_completed": True,
                    "controller_phase": "holdout",
                    "holdout_started": True,
                    "current_evaluation_candidate_id": incumbent_id,
                    "current_evaluation_next_seed_index": 0,
                }
            )
            force_checkpoint()
            continue

        if phase == "holdout":
            assert incumbent_id is not None
            evaluate_record_suffix(
                record_by_id[incumbent_id],
                holdout_seeds,
                start_index=int(run_state["current_evaluation_next_seed_index"]),
                holdout=True,
            )
            run_state.update(
                {
                    "controller_phase": "holdout_complete",
                    "holdout_completed": True,
                    "current_evaluation_candidate_id": None,
                    "current_evaluation_next_seed_index": 0,
                }
            )
            force_checkpoint()
            continue

        if phase == "holdout_complete":
            run_state["controller_phase"] = "complete"
            force_checkpoint(status="complete")
            continue

        raise RuntimeError(f"Unknown progressive controller phase: {phase!r}.")

    assert incumbent_id is not None
    incumbent = record_by_id[incumbent_id]
    comparison_evaluation = _aggregate_from_cache(
        incumbent,
        comparison_seeds,
        cache=cache,
        total_copies=total_copies,
        optimization_config=optimization_config,
        objective=objective,
    )
    aggregated_holdout = _aggregate_from_cache(
        incumbent,
        holdout_seeds,
        cache=cache,
        total_copies=total_copies,
        optimization_config=optimization_config,
        objective=objective,
        holdout=True,
    )
    assert isinstance(aggregated_holdout, HoldoutEvaluation)
    holdout_evaluation: Optional[HoldoutEvaluation] = aggregated_holdout

    final_catalog = records
    if incumbent_id not in {record.candidate_id for record in records}:
        final_catalog = records + (incumbent,)
    final_catalog_ids = [record.candidate_id for record in final_catalog]
    if len(final_catalog_ids) != len(set(final_catalog_ids)):
        raise RuntimeError("Final candidate catalog contains conflicting IDs.")
    if incumbent_id not in set(final_catalog_ids):
        raise RuntimeError("Final selected candidate is absent from candidate catalog.")

    assert termination_reason is not None
    metadata = ProgressiveSearchMetadata(
        algorithm="progressive reproducible practical search with successive halving",
        objective_mode=objective.mode.value,
        progressive_search_schema=PROGRESSIVE_SEARCH_SCHEMA,
        progressive_config=progressive,
        search_seed=int(optimization_config.search_seed),
        tuning_seeds=tuple(optimization_config.tuning_seeds),
        holdout_seeds=tuple(optimization_config.holdout_seeds),
        round_summaries=tuple(round_summaries),
        rounds_completed=len(round_summaries),
        final_candidate_pool_size=round_summaries[-1].candidate_pool_size,
        total_valid_candidates_pregenerated=len(records),
        rejected_invalid_sampling_attempts=rejected,
        termination_reason=termination_reason,
        checkpoint_enabled=store is not None,
        resumed_from_checkpoint=resumed,
        holdout_post_selection_only=True,
        fixed_budget_epsilon_refinement_enabled=False,
        budget_refinement_enabled=False,
        budget_refinement_trials=(),
        budget_refinement_note="Not applicable to fixed-budget mode.",
    )
    return ProgressiveOptimizationResult(
        objective=objective,
        best_candidate_id=incumbent_id,
        best_candidate=incumbent.parameters,
        best_derived_candidate=incumbent.derived,
        comparison_evaluation=comparison_evaluation,
        holdout_evaluation=holdout_evaluation,
        search_metadata=metadata,
        candidate_catalog=final_catalog,
        total_candidate_count=len(final_catalog),
        rejected_invalid_count=rejected,
        evaluation_attempt_count=counters.evaluation_attempt_count,
        actual_learner_run_count=counters.actual_learner_run_count,
        cache_hit_count=counters.cache_hit_count,
        unique_cached_seed_evaluation_count=len(cache),
        preflight_rejection_count=counters.preflight_rejection_count,
        budget_refinement_actual_learner_run_count=(
            counters.budget_refinement_actual_learner_run_count
        ),
    )
