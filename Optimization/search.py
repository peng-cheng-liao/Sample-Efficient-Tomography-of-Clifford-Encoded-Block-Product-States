"""Reproducible objective-aware search, refinement, and checkpoint/resume."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from main_v2 import CEBPInstance

from .checkpoint import CheckpointStore
from .objective import (
    CandidateEvaluation,
    HoldoutEvaluation,
    SeedEvaluation,
    candidate_ranking_key,
    evaluate_candidate,
    require_trace_distance_objective_available,
)
from .parameterization import (
    CandidateParameters,
    DerivedCandidate,
    FixedBudgetCandidateParameters,
    FixedErrorCandidateParameters,
    InvalidCandidateError,
    OptimizationConfig,
    SearchSpace,
    derive_candidate,
    minimum_practical_copy_budget,
    sample_candidate,
)
from .specification import OptimizationMode, OptimizationObjective


@dataclass(frozen=True)
class HalvingRoundSummary:
    round_index: int
    candidates_alive: int
    seeds_per_candidate: int
    best_candidate_id: str
    best_mean_loss: float
    best_max_copies: int
    best_success_rate: float
    # Deprecated compatibility alias for best_mean_error_feasible.
    best_all_error_feasible: Optional[bool] = None
    best_max_trace_distance: Optional[float] = None
    best_mean_error_feasible: Optional[bool] = None


@dataclass(frozen=True)
class RefinementTrialSummary:
    candidate_id: str
    epsilon_tom: float
    predicted_max_total_copies: int
    predicted_max_tomography_copies: int
    evaluation_kind: str
    n_seeds_executed: int
    all_operationally_successful: bool
    all_budget_feasible: bool
    all_error_feasible: Optional[bool]
    max_trace_distance: Optional[float]
    mean_trace_distance: Optional[float]
    preflight_safe_to_execute: Optional[bool] = None
    preflight_runtime_safety_rejected: Optional[bool] = None
    preflight_reason: Optional[str] = None
    selected: bool = False

    @property
    def maximum_realized_copies(self) -> int:
        """Compatibility alias for earlier callers."""

        return self.predicted_max_total_copies

    @property
    def maximum_block_tomography_copies(self) -> int:
        """Compatibility alias for earlier callers."""

        return self.predicted_max_tomography_copies


@dataclass(frozen=True)
class SearchMetadata:
    algorithm: str
    objective_mode: str
    search_seed: int
    tuning_seeds: Tuple[int, ...]
    holdout_seeds: Tuple[int, ...]
    halving_seed_counts: Tuple[int, ...]
    retention_fraction: float
    common_random_numbers_enforced: bool
    candidate_generation_reproducible: bool
    changing_search_seed_changes_candidates: bool
    tomography_refinement_enabled: bool
    tomography_refinement_trials: int
    tomography_refinement_note: str
    tomography_refinement_summaries: Tuple[RefinementTrialSummary, ...]
    round_summaries: Tuple[HalvingRoundSummary, ...]
    checkpoint_enabled: bool
    resumed_from_checkpoint: bool


@dataclass(frozen=True)
class OptimizationResult:
    """Concise result; dense states, transcripts, and learner results are absent."""

    objective: OptimizationObjective
    best_candidate_id: str
    best_candidate: CandidateParameters | FixedBudgetCandidateParameters | FixedErrorCandidateParameters
    best_derived_candidate: DerivedCandidate
    tuning_evaluation: CandidateEvaluation
    holdout_evaluation: HoldoutEvaluation
    all_candidate_summaries: Tuple[CandidateEvaluation, ...]
    search_metadata: SearchMetadata
    candidate_catalog: Tuple["CandidateRecord", ...]
    total_candidate_count: int
    rejected_invalid_count: int
    evaluated_run_count: int
    actual_learner_run_count: int
    cache_hit_count: int
    preflight_rejection_count: int
    evaluation_attempt_count: int
    analytical_refinement_trial_count: int
    refinement_actual_learner_run_count: int


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    parameters: CandidateParameters | FixedBudgetCandidateParameters | FixedErrorCandidateParameters
    derived: DerivedCandidate


class _ScientificCopyCapPrecheckError(ValueError):
    """A raw fixed-error draw has no admissible first E2E budget."""


@dataclass
class _Counters:
    evaluation_attempt_count: int = 0
    actual_learner_run_count: int = 0
    cache_hit_count: int = 0
    preflight_rejection_count: int = 0
    analytical_refinement_trial_count: int = 0
    refinement_actual_learner_run_count: int = 0


def _sample_valid_candidates(
    *,
    instance: CEBPInstance,
    total_copies: int,
    optimization_config: OptimizationConfig,
    search_space: SearchSpace,
    search_seed: int,
    initial_candidates: Sequence[CandidateParameters | FixedBudgetCandidateParameters | FixedErrorCandidateParameters] = (),
    target_count: Optional[int] = None,
    sampling_diagnostics: Optional[Dict[str, int]] = None,
) -> Tuple[Tuple[CandidateRecord, ...], int]:
    resolved_target = (
        optimization_config.number_of_candidates
        if target_count is None
        else target_count
    )
    if (
        isinstance(resolved_target, bool)
        or not isinstance(resolved_target, (int, np.integer))
        or int(resolved_target) <= 0
    ):
        raise ValueError("target_count must be a positive integer.")
    resolved_target = int(resolved_target)
    rng = np.random.default_rng(int(search_seed))
    records: List[CandidateRecord] = []
    rejected = 0
    raw_draws = 0
    precheck_cap_rejections = 0
    fixed_error = (
        optimization_config.effective_objective.mode
        is OptimizationMode.FIXED_ERROR_MIN_COPIES
    )

    def sync_sampling_diagnostics() -> None:
        if sampling_diagnostics is None:
            return
        sampling_diagnostics.update(
            {
                "raw_candidate_draw_count": int(raw_draws),
                "precheck_cap_rejection_count": int(precheck_cap_rejections),
                "valid_candidate_count": len(records),
            }
        )

    def derive_for_catalog(candidate):
        if fixed_error and isinstance(candidate, FixedErrorCandidateParameters):
            practical = minimum_practical_copy_budget(candidate, d=instance.d)
            starting_budget = max(
                practical,
                int(optimization_config.fixed_error_initial_budget_hint or practical),
            )
            scientific_cap = (
                optimization_config.effective_objective.scientific_copy_cap
            )
            if scientific_cap is not None and starting_budget > int(scientific_cap):
                raise _ScientificCopyCapPrecheckError(
                    f"first admissible budget {starting_budget} exceeds "
                    f"scientific_copy_cap={int(scientific_cap)}"
                )
            return derive_candidate(
                candidate,
                n=instance.n,
                d=instance.d,
                total_copies=total_copies,
                optimization_config=optimization_config,
                physical_budget=starting_budget,
            )
        return derive_candidate(
            candidate,
            n=instance.n,
            d=instance.d,
            total_copies=total_copies,
            optimization_config=optimization_config,
        )

    unique_initial: List[CandidateParameters | FixedBudgetCandidateParameters | FixedErrorCandidateParameters] = []
    for candidate in initial_candidates:
        raw_draws += 1
        if not isinstance(candidate, (CandidateParameters, FixedBudgetCandidateParameters, FixedErrorCandidateParameters)):
            raise TypeError("initial_candidates contain an unsupported candidate type.")
        try:
            canonical = derive_for_catalog(candidate).parameters
        except _ScientificCopyCapPrecheckError as error:
            precheck_cap_rejections += 1
            sync_sampling_diagnostics()
            raise ValueError(f"Initial candidate is outside the scientific domain: {error}") from error
        except InvalidCandidateError as error:
            raise ValueError(f"Invalid initial candidate: {error}") from error
        if canonical not in unique_initial:
            unique_initial.append(canonical)
    if len(unique_initial) > resolved_target:
        if target_count is None:
            raise ValueError("Unique initial candidates exceed number_of_candidates.")
        raise ValueError("Unique initial candidates exceed target_count.")
    for index, candidate in enumerate(unique_initial):
        try:
            derived = derive_for_catalog(candidate)
        except InvalidCandidateError as error:
            raise ValueError(f"Invalid initial candidate {index}: {error}") from error
        records.append(CandidateRecord(f"initial-{index:04d}", derived.parameters, derived))

    max_attempts = max(10_000, resolved_target * 1_000)
    while len(records) < resolved_target:
        if len(records) + rejected >= max_attempts:
            raise RuntimeError("Unable to sample enough valid candidates from the search space.")
        candidate = sample_candidate(
            rng,
            search_space,
            mode=optimization_config.effective_objective.mode,
            copy_ceiling=optimization_config.effective_objective.copy_ceiling,
            d=instance.d,
        )
        raw_draws += 1
        try:
            derived = derive_for_catalog(candidate)
        except _ScientificCopyCapPrecheckError:
            rejected += 1
            precheck_cap_rejections += 1
            sync_sampling_diagnostics()
            continue
        except InvalidCandidateError:
            rejected += 1
            continue
        if any(candidate == record.parameters for record in records):
            rejected += 1
            continue
        candidate_id = f"candidate-{len(records) - len(unique_initial):04d}"
        records.append(CandidateRecord(candidate_id, derived.parameters, derived))
        sync_sampling_diagnostics()
    sync_sampling_diagnostics()
    return tuple(records), rejected


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _checkpoint_metadata(
    *,
    instance: CEBPInstance,
    objective: OptimizationObjective,
    config: OptimizationConfig,
    search_space: SearchSpace,
    records: Tuple[CandidateRecord, ...],
) -> Dict[str, Any]:
    fixed_error = objective.mode is OptimizationMode.FIXED_ERROR_MIN_COPIES
    if fixed_error:
        # Only modern practical fixed-error inputs that can change candidate
        # generation, learner execution, objective evaluation, refinement, or
        # semantic cache identity participate in checkpoint compatibility.
        config_fields = (
            "search_seed",
            "tuning_seeds",
            "peeling_grid_intervals",
            "delta_grp_ordinary",
            "zeta_sgn",
            "fixed_budget_zeta_peel",
            "fixed_budget_zeta_rank",
            "max_dense_qubits",
            "max_enumeration_qubits",
            "max_oracle_dense_qubits",
            "inner_enumeration_workers",
            "max_score_array_bytes",
            "max_structured_bell_workspace_bytes",
            "max_enumeration_workspace_bytes",
            "enumeration_workspace_safety_factor",
            "simulation_backend",
            "fixed_error_initial_budget_hint",
            "fixed_error_budget_growth_factor",
            "fixed_error_budget_relative_tolerance",
            "fixed_error_max_budget_expansion_rounds",
            "fixed_error_incumbent_confirmation_probes",
            "fixed_error_hard_safety_budget",
            "max_preflight_estimated_copies",
            "execution_safety_factor",
            "max_predicted_grouping_copies",
            "max_predicted_tomography_copies",
            "max_predicted_total_copies",
            "max_single_grouping_query_shots",
        )
        config_value = {
            key: _json_safe(getattr(config, key)) for key in config_fields
        }
        active_search_fields = (
            "h_min",
            "h_max",
            "theta_tau_multiplier",
            "eta_test",
            "peel_weight",
            "recovery_weight",
            "grouping_weight",
            "syndrome_weight",
            "tomography_weight",
        )
        search_space_value = {
            key: _json_safe(getattr(search_space, key))
            for key in active_search_fields
        }
        catalog = []
        for record in records:
            catalog.append(
                {
                    "candidate_id": record.candidate_id,
                    "parameters": _json_safe(asdict(record.parameters)),
                }
            )
    else:
        # Preserve the historical fixed-budget/theorem checkpoint contract.
        config_value = asdict(config)
        for key in (
            "checkpoint_path",
            "resume_from_checkpoint",
            "checkpoint_every_n_evaluations",
            "verbose",
        ):
            config_value.pop(key, None)
        if isinstance(config_value.get("objective"), dict):
            config_value["objective"].pop(
                "success_probability_threshold", None
            )
            config_value["objective"].pop("scientific_copy_cap", None)
        search_space_value = asdict(search_space)
        catalog = [
            {
                "candidate_id": record.candidate_id,
                "parameters": _json_safe(asdict(record.parameters)),
                "derived": _json_safe(asdict(record.derived)),
            }
            for record in records
        ]
    objective_value = {**asdict(objective), "mode": objective.mode.value}
    if fixed_error:
        objective_value.pop("copy_ceiling", None)
    if not fixed_error:
        # Preserve the historical fixed-budget checkpoint fingerprint exactly;
        # the probability threshold is a fixed-error-only objective field.
        objective_value.pop("success_probability_threshold", None)
        objective_value.pop("scientific_copy_cap", None)
    return {
        "execution_policy": (
            "fixed_budget_graceful"
            if objective.mode in (
                OptimizationMode.FIXED_BUDGET_MIN_ERROR,
                OptimizationMode.FIXED_ERROR_MIN_COPIES,
            )
            else "strict"
        ),
        "fixed_budget_parameterization_schema": 5,
        "evaluation_identity_schema": 10 if fixed_error else 7,
        **(
            {"fixed_error_search_policy": {
                "threshold_controller_schema": 3,
                "incumbent_pruning_policy_schema": 1,
                "numeric_sampling_safety_policy_schema": 1,
                "scientific_copy_cap_policy_schema": 1,
                "valid_candidate_quota_semantics_schema": 2,
                "final_tuning_seed_count": 16,
                "required_success_count": int(math.ceil(
                    float(objective.success_probability_threshold) * 16
                )),
            }}
            if fixed_error
            else {}
        ),
        "copy_budget": (
            None if fixed_error else int(objective.copy_ceiling)
        ),
        "simulation_backend": config.simulation_backend,
        "qubit_policy": {
            "max_dense_qubits_compatibility_alias": config.max_dense_qubits,
            "max_enumeration_qubits": config.max_enumeration_qubits,
            "max_oracle_dense_qubits": config.max_oracle_dense_qubits,
        },
        "enumeration_memory_policy": {
            "max_score_array_bytes": config.max_score_array_bytes,
            "max_structured_bell_workspace_bytes": (
                config.max_structured_bell_workspace_bytes
            ),
            "max_enumeration_workspace_bytes": (
                config.max_enumeration_workspace_bytes
            ),
            "enumeration_workspace_safety_factor": (
                config.enumeration_workspace_safety_factor
            ),
        },
        "checkpoint_key": config.checkpoint_key,
        "instance_metadata": {"n": instance.n, "d": instance.d},
        "objective": objective_value,
        "objective_fingerprint": _fingerprint(objective_value),
        "optimization_config_fingerprint": _fingerprint(config_value),
        "search_space_fingerprint": _fingerprint(search_space_value),
        "candidate_catalog": catalog,
        "search_seed": config.search_seed,
        "tuning_seeds": list(config.tuning_seeds),
        "holdout_seeds": [] if fixed_error else list(config.holdout_seeds),
        **(
            {"fixed_error_checkpoint_fingerprint_schema": 5}
            if fixed_error
            else {}
        ),
    }


def optimize_cebp_parameters(
    instance: CEBPInstance,
    total_copies: int,
    optimization_config: OptimizationConfig,
    search_space: Optional[SearchSpace] = None,
    *,
    initial_candidates: Sequence[CandidateParameters | FixedBudgetCandidateParameters | FixedErrorCandidateParameters] = (),
) -> OptimizationResult:
    """Find a best empirical configuration under one explicit objective."""

    if not isinstance(instance, CEBPInstance):
        raise TypeError("instance must be a CEBPInstance.")
    objective = optimization_config.effective_objective
    if instance.d == 1 and objective.mode not in (
        OptimizationMode.FIXED_BUDGET_MIN_ERROR,
        OptimizationMode.FIXED_ERROR_MIN_COPIES,
    ):
        raise NotImplementedError(
            "d=1 optimization is supported only for the practical graceful modes."
        )
    if (
        objective.mode is not OptimizationMode.FIXED_ERROR_MIN_COPIES
        and int(total_copies) != int(optimization_config.total_copies)
    ):
        raise ValueError("total_copies disagrees with OptimizationConfig.total_copies.")
    if (
        objective.mode is OptimizationMode.FIXED_BUDGET_MIN_ERROR
        and objective.copy_ceiling != int(total_copies)
    ):
        raise ValueError("Objective copy_ceiling must equal total_copies.")
    require_trace_distance_objective_available(
        instance, optimization_config, objective
    )
    if objective.mode is OptimizationMode.FIXED_ERROR_MIN_COPIES:
        # The public fixed-error entry point now uses the shared progressive
        # controller with the fixed 1/2/4/8/16 tuning-fidelity schedule.
        from .progressive_search import (
            MAX_FIXED_BUDGET_MIN_ERROR_CANDIDATES,
            ProgressiveSearchConfig,
            optimize_cebp_parameters_progressive,
        )

        available = len(optimization_config.tuning_seeds)
        if available < 16:
            raise ValueError(
                "fixed_error_min_copies requires at least 16 tuning seeds; "
                "the first 16 define final feasibility and budget refinement."
            )
        return optimize_cebp_parameters_progressive(
            instance,
            total_copies,
            optimization_config,
            search_space,
            progressive_config=ProgressiveSearchConfig(
                initial_candidates=16,
                max_candidates=MAX_FIXED_BUDGET_MIN_ERROR_CANDIDATES,
                seed_fidelities=(1, 2, 4, 8, 16),
                comparison_seed_count=16,
            ),
            initial_candidates=initial_candidates,
        )
    space = search_space or SearchSpace()
    records, rejected = _sample_valid_candidates(
        instance=instance,
        total_copies=total_copies,
        optimization_config=optimization_config,
        search_space=space,
        search_seed=optimization_config.search_seed,
        initial_candidates=initial_candidates,
    )
    repeated, repeated_rejected = _sample_valid_candidates(
        instance=instance,
        total_copies=total_copies,
        optimization_config=optimization_config,
        search_space=space,
        search_seed=optimization_config.search_seed,
        initial_candidates=initial_candidates,
    )
    reproducible = records == repeated and rejected == repeated_rejected
    if not reproducible:
        raise RuntimeError("Candidate generation is not reproducible.")
    changed, _changed_rejected = _sample_valid_candidates(
        instance=instance,
        total_copies=total_copies,
        optimization_config=optimization_config,
        search_space=space,
        search_seed=optimization_config.search_seed + 1,
        initial_candidates=initial_candidates,
    )
    seed_changes_candidates = tuple(record.parameters for record in records) != tuple(
        record.parameters for record in changed
    )
    if optimization_config.number_of_candidates > len(set(initial_candidates)) and not seed_changes_candidates:
        raise RuntimeError("Changing search_seed did not change sampled candidates.")

    cache: Dict[Tuple[str, int], SeedEvaluation] = {}
    counters = _Counters()
    run_state: Dict[str, Any] = {
        "current_halving_round": 0,
        "alive_candidate_ids": [record.candidate_id for record in records],
        "selected_candidate_id": None,
        "refinement_state": "not_started",
        "invalid_rejected_count": rejected,
        "counters": asdict(counters),
    }
    store: Optional[CheckpointStore] = None
    resumed = False
    if optimization_config.checkpoint_path is not None:
        store = CheckpointStore(
            optimization_config.checkpoint_path,
            expected_metadata=_checkpoint_metadata(
                instance=instance,
                objective=objective,
                config=optimization_config,
                search_space=space,
                records=records,
            ),
            every_n_evaluations=optimization_config.checkpoint_every_n_evaluations,
        )
        if optimization_config.resume_from_checkpoint:
            cache, loaded_state = store.load()
            run_state.update(loaded_state)
            counters = _Counters(**dict(run_state.get("counters", {})))
            resumed = True
        else:
            store.save(cache, run_state, status="running")

    def sync_state() -> None:
        run_state["counters"] = asdict(counters)

    def evaluate_record(
        record: CandidateRecord,
        seeds: Tuple[int, ...],
        *,
        holdout: bool = False,
        refinement: bool = False,
        early_stop=None,
    ) -> CandidateEvaluation:
        def completed(_key, evaluation: SeedEvaluation) -> None:
            counters.evaluation_attempt_count += 1
            if evaluation.preflight_rejected:
                counters.preflight_rejection_count += 1
            else:
                counters.actual_learner_run_count += 1
                if refinement:
                    counters.refinement_actual_learner_run_count += 1
            sync_state()
            if store is not None:
                store.note_completed(cache, run_state)

        def cache_hit(_key) -> None:
            counters.evaluation_attempt_count += 1
            counters.cache_hit_count += 1
            sync_state()

        return evaluate_candidate(
            instance,
            record.candidate_id,
            record.derived,
            seeds,
            total_copies,
            optimization_config,
            cache=cache,
            holdout=holdout,
            objective=objective,
            on_evaluation=completed,
            on_cache_hit=cache_hit,
            early_stop=early_stop,
        )

    record_by_id = {record.candidate_id: record for record in records}
    alive = list(records)
    latest: Dict[str, CandidateEvaluation] = {}
    round_summaries: List[HalvingRoundSummary] = []
    for round_index, seed_count in enumerate(optimization_config.halving_seed_counts, start=1):
        run_state["current_halving_round"] = round_index
        run_state["alive_candidate_ids"] = [record.candidate_id for record in alive]
        common_seeds = tuple(optimization_config.tuning_seeds[:seed_count])
        evaluations = []
        for record in alive:
            evaluation = evaluate_record(record, common_seeds)
            if tuple(item.learner_seed for item in evaluation.seed_evaluations) != common_seeds:
                raise RuntimeError("Common-random-number seed policy was violated.")
            latest[record.candidate_id] = evaluation
            evaluations.append(evaluation)
        ranked = sorted(evaluations, key=lambda item: candidate_ranking_key(item, objective))
        best = ranked[0]
        round_summaries.append(
            HalvingRoundSummary(
                round_index=round_index,
                candidates_alive=len(alive),
                seeds_per_candidate=seed_count,
                best_candidate_id=best.candidate_id,
                best_mean_loss=best.mean_loss,
                best_max_copies=best.max_realized_copies,
                best_success_rate=best.success_rate,
                best_all_error_feasible=best.all_error_feasible,
                best_max_trace_distance=best.max_trace_distance_successful,
                best_mean_error_feasible=best.mean_error_feasible,
            )
        )
        if round_index < len(optimization_config.halving_seed_counts):
            keep = max(1, int(math.ceil(len(alive) * optimization_config.retention_fraction)))
            keep_ids = {evaluation.candidate_id for evaluation in ranked[:keep]}
            alive = [record for record in alive if record.candidate_id in keep_ids]
        if store is not None:
            sync_state()
            store.save(cache, run_state, status="running")

    ranked_final = sorted(
        (latest[record.candidate_id] for record in alive),
        key=lambda item: candidate_ranking_key(item, objective),
    )
    structural_best_id = ranked_final[0].candidate_id
    structural_best = record_by_id[structural_best_id]
    selected = structural_best
    final_tuning_seeds = tuple(
        optimization_config.tuning_seeds[: optimization_config.halving_seed_counts[-1]]
    )
    structural_evaluation = latest[structural_best_id]
    refinement_note = "Tomography refinement disabled."
    refinement_summaries: List[RefinementTrialSummary] = []

    if (
        optimization_config.tomography_refinement_enabled
        and objective.mode is OptimizationMode.FIXED_BUDGET_MIN_ERROR
    ):
        refinement_note = (
            "Fixed-budget epsilon_tom refinement disabled: graceful tomography "
            "already consumes the physical remainder; epsilon_tom is compatibility metadata."
        )

    if selected.candidate_id == structural_best_id:
        tuning_evaluation = structural_evaluation
    elif selected.candidate_id in latest:
        tuning_evaluation = latest[selected.candidate_id]
    else:
        # Fixed-budget refinement executes only this final chosen candidate.
        tuning_evaluation = evaluate_record(
            selected, final_tuning_seeds, refinement=True
        )
        latest[selected.candidate_id] = tuning_evaluation

    if (
        objective.mode is OptimizationMode.FIXED_BUDGET_MIN_ERROR
        and selected.candidate_id != structural_best_id
        and (
            tuning_evaluation.n_seeds_evaluated != len(final_tuning_seeds)
            or any(item.preflight_rejected for item in tuning_evaluation.seed_evaluations)
            or tuning_evaluation.success_rate != 1.0
            or not tuning_evaluation.all_budget_feasible
        )
    ):
        selected = structural_best
        tuning_evaluation = structural_evaluation
        refinement_note = (
            "Refined candidate failed final safety/feasibility validation; "
            "reverted to structural best."
        )

    refinement_summaries = [
        replace(item, selected=item.candidate_id == selected.candidate_id)
        for item in refinement_summaries
    ]
    holdout_evaluation = evaluate_record(
        selected,
        tuple(optimization_config.holdout_seeds),
        holdout=True,
    )
    assert isinstance(holdout_evaluation, HoldoutEvaluation)

    run_state.update(
        {
            "current_halving_round": len(optimization_config.halving_seed_counts),
            "alive_candidate_ids": [record.candidate_id for record in alive],
            "selected_candidate_id": selected.candidate_id,
            "refinement_state": "complete",
        }
    )
    sync_state()
    if store is not None:
        store.save(cache, run_state, status="complete")

    metadata = SearchMetadata(
        algorithm="reproducible random search with successive halving",
        objective_mode=objective.mode.value,
        search_seed=int(optimization_config.search_seed),
        tuning_seeds=tuple(optimization_config.tuning_seeds),
        holdout_seeds=tuple(optimization_config.holdout_seeds),
        halving_seed_counts=tuple(optimization_config.halving_seed_counts),
        retention_fraction=float(optimization_config.retention_fraction),
        common_random_numbers_enforced=True,
        candidate_generation_reproducible=reproducible,
        changing_search_seed_changes_candidates=seed_changes_candidates,
        tomography_refinement_enabled=bool(
            optimization_config.tomography_refinement_enabled
            and objective.mode is OptimizationMode.FIXED_ERROR_MIN_COPIES
        ),
        tomography_refinement_trials=len(refinement_summaries),
        tomography_refinement_note=refinement_note,
        tomography_refinement_summaries=tuple(refinement_summaries),
        round_summaries=tuple(round_summaries),
        checkpoint_enabled=store is not None,
        resumed_from_checkpoint=resumed,
    )
    return OptimizationResult(
        objective=objective,
        best_candidate_id=selected.candidate_id,
        best_candidate=selected.parameters,
        best_derived_candidate=selected.derived,
        tuning_evaluation=tuning_evaluation,
        holdout_evaluation=holdout_evaluation,
        all_candidate_summaries=tuple(latest[key] for key in sorted(latest)),
        search_metadata=metadata,
        candidate_catalog=tuple(records),
        total_candidate_count=len(records),
        rejected_invalid_count=rejected,
        evaluated_run_count=counters.actual_learner_run_count,
        actual_learner_run_count=counters.actual_learner_run_count,
        cache_hit_count=counters.cache_hit_count,
        preflight_rejection_count=counters.preflight_rejection_count,
        evaluation_attempt_count=counters.evaluation_attempt_count,
        analytical_refinement_trial_count=counters.analytical_refinement_trial_count,
        refinement_actual_learner_run_count=counters.refinement_actual_learner_run_count,
    )
