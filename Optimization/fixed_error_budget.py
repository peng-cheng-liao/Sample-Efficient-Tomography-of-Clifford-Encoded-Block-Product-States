"""Adaptive empirical minimum-budget estimation for fixed-error search."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, Iterable, Optional, Tuple

from .objective import CandidateEvaluation


THRESHOLD_CONTROLLER_SCHEMA = 3
THRESHOLD_FOUND = "threshold_found"
BUDGET_CAP_EXHAUSTED = "budget_cap_exhausted"
PRECHECK_CAP_REJECTED = "precheck_cap_rejected"
COMPUTATIONAL_SAFETY_LIMIT_REACHED = "computational_safety_limit_reached"
NUMERIC_SAMPLING_LIMIT_REACHED = "numeric_sampling_limit_reached"
INCUMBENT_DOMINATED_SEARCH_PRUNED = "incumbent_dominated_search_pruned"
OPERATIONAL_FAILURE = "operational_failure"
INVALID_STRUCTURAL_CANDIDATE = "invalid_structural_candidate"
FEASIBLE = "feasible"
MEAN_ERROR_FAILED = "mean_error_failed"
SUCCESS_FRACTION_FAILED = "success_fraction_failed"
PROVISIONAL_MEAN_FEASIBLE = "provisional_mean_feasible"


@dataclass(frozen=True)
class FixedErrorBudgetTrial:
    """One aggregate trial at a transient physical budget and seed fidelity."""

    budget: int
    seed_fidelity: int
    operationally_valid: bool
    mean_trace_distance: Optional[float]
    mean_error_feasible: bool
    error_success_count: Optional[int]
    error_success_fraction: Optional[float]
    required_error_success_count: Optional[int]
    success_fraction_feasible: Optional[bool]
    feasible: bool
    computational_status: Optional[str] = None
    scientific_status: str = "not_classified"


@dataclass(frozen=True)
class BudgetThresholdEvaluation:
    """Estimated empirical feasibility threshold for one structural candidate."""

    structural_candidate_id: str
    seed_fidelity: int
    threshold_found: bool
    search_status: str
    estimated_min_budget: Optional[int]
    low_infeasible_budget: Optional[int]
    high_feasible_budget: Optional[int]
    relative_bracket_width: Optional[float]
    practical_minimum_budget: int
    initial_budget: int
    initial_budget_hint: Optional[int]
    initial_budget_hint_used: bool
    budget_trials: Tuple[FixedErrorBudgetTrial, ...]
    tested_budgets: Tuple[int, ...]
    expansion_trial_count: int
    refinement_trial_count: int
    final_mean_trace_distance: Optional[float]
    final_error_success_count: Optional[int]
    final_error_success_fraction: Optional[float]
    final_required_success_count: Optional[int]
    final_operational_validity: bool
    non_monotonic_observations: Tuple[Tuple[int, int], ...]
    warm_started: bool
    cumulative_expansion_trial_count: int = 0
    largest_tested_budget: Optional[int] = None
    terminal_search_status: Optional[str] = None
    incumbent_budget_hint: Optional[int] = None
    incumbent_guided_start: bool = False
    incumbent_probe_count: int = 0
    incumbent_confirmation_probe_count: int = 0
    incumbent_pruned: bool = False
    numeric_safety_event_count: int = 0


def _trial_from_evaluation(
    budget: int,
    seed_fidelity: int,
    evaluation: CandidateEvaluation,
    *,
    final_seed_count: int,
) -> FixedErrorBudgetTrial:
    final = int(seed_fidelity) == int(final_seed_count)
    feasible = bool(
        evaluation.operationally_valid
        and evaluation.mean_error_feasible
        and (
            evaluation.success_fraction_feasible is True
            if final
            else True
        )
    )
    numeric_limit = any(
        bool(getattr(item, "numeric_sampling_limit_reached", False))
        or str(item.failure_reason or "").startswith(NUMERIC_SAMPLING_LIMIT_REACHED)
        for item in evaluation.seed_evaluations
    )
    scientific_status = (
        FEASIBLE
        if feasible
        else (
            OPERATIONAL_FAILURE
            if not evaluation.operationally_valid
            else (
                MEAN_ERROR_FAILED
                if not evaluation.mean_error_feasible
                else (
                    SUCCESS_FRACTION_FAILED
                    if final and evaluation.success_fraction_feasible is not True
                    else PROVISIONAL_MEAN_FEASIBLE
                )
            )
        )
    )
    return FixedErrorBudgetTrial(
        budget=int(budget),
        seed_fidelity=int(seed_fidelity),
        operationally_valid=bool(evaluation.operationally_valid),
        mean_trace_distance=evaluation.mean_trace_distance_successful,
        mean_error_feasible=bool(evaluation.mean_error_feasible),
        error_success_count=evaluation.error_success_count,
        error_success_fraction=evaluation.error_success_fraction,
        required_error_success_count=evaluation.required_error_success_count,
        success_fraction_feasible=evaluation.success_fraction_feasible,
        feasible=feasible,
        computational_status=(
            NUMERIC_SAMPLING_LIMIT_REACHED if numeric_limit else None
        ),
        scientific_status=scientific_status,
    )


def _non_monotonic_pairs(
    observations: Dict[int, FixedErrorBudgetTrial],
) -> Tuple[Tuple[int, int], ...]:
    feasible = sorted(budget for budget, trial in observations.items() if trial.feasible)
    infeasible = sorted(
        budget for budget, trial in observations.items() if not trial.feasible
    )
    return tuple(
        (low_feasible, high_infeasible)
        for low_feasible in feasible
        for high_infeasible in infeasible
        if low_feasible < high_infeasible
    )


def fixed_error_threshold_ranking_key(
    result: BudgetThresholdEvaluation,
) -> Tuple[object, ...]:
    """Deterministically rank threshold estimates and then search progress."""

    if result.threshold_found:
        return (
            0,
            int(result.estimated_min_budget),
            (
                math.inf
                if result.final_mean_trace_distance is None
                else float(result.final_mean_trace_distance)
            ),
            -float(result.final_error_success_fraction or 0.0),
            result.structural_candidate_id,
        )
    trials = result.budget_trials
    operational = [trial for trial in trials if trial.operationally_valid]
    best_excess = min(
        (
            float(trial.mean_trace_distance)
            if trial.mean_trace_distance is not None
            else math.inf
        )
        for trial in trials
    ) if trials else math.inf
    best_success = max(
        (float(trial.error_success_fraction or 0.0) for trial in trials),
        default=0.0,
    )
    largest = max(result.tested_budgets, default=0)
    status_priority = {
        OPERATIONAL_FAILURE: 0,
        BUDGET_CAP_EXHAUSTED: 1,
        COMPUTATIONAL_SAFETY_LIMIT_REACHED: 2,
        NUMERIC_SAMPLING_LIMIT_REACHED: 3,
        INCUMBENT_DOMINATED_SEARCH_PRUNED: 4,
        PRECHECK_CAP_REJECTED: 5,
    }.get(result.search_status, 6)
    return (
        1,
        status_priority,
        0 if operational else 1,
        best_excess,
        -best_success,
        -largest,
        result.structural_candidate_id,
    )


def estimate_minimum_feasible_budget(
    structural_candidate_id: str,
    *,
    seed_fidelity: int,
    final_seed_count: int,
    practical_minimum_budget: int,
    evaluate_budget: Callable[[int], CandidateEvaluation],
    previous_tested_budgets: Iterable[int] = (),
    previous_cumulative_expansion_trials: int = 0,
    previous_terminal_search_status: Optional[str] = None,
    previous_incumbent_probe_count: int = 0,
    previous_incumbent_confirmation_probe_count: int = 0,
    previous_numeric_safety_event_count: int = 0,
    initial_budget_hint: Optional[int] = None,
    incumbent_budget_hint: Optional[int] = None,
    is_current_incumbent: bool = False,
    incumbent_confirmation_probes: int = 1,
    growth_factor: float = 2.0,
    relative_tolerance: float = 0.01,
    max_expansion_rounds: int = 32,
    hard_safety_budget: Optional[int] = None,
    scientific_copy_cap: Optional[int] = None,
    on_controller_state: Optional[Callable[[Dict[str, object]], None]] = None,
) -> BudgetThresholdEvaluation:
    """Estimate the smallest tested feasible budget within a relative bracket.

    Feasibility is provisional (operational plus mean) below final fidelity and
    is operational plus mean plus the configured success gate at final fidelity.
    Prior budgets are re-aggregated at the new fidelity, allowing the execution
    cache to add only missing ``(lambda, N, seed)`` trials.  Expansion accounting
    and terminal computational states are cumulative across seed fidelities.
    An incumbent is an execution hint and heuristic pruning reference, never a
    scientific copy ceiling.  ``scientific_copy_cap`` is the independent
    fixed-error search domain and is enforced before every evaluation callback.
    """

    practical = int(practical_minimum_budget)
    if scientific_copy_cap is not None and (
        isinstance(scientific_copy_cap, bool) or int(scientific_copy_cap) <= 0
    ):
        raise ValueError("scientific_copy_cap must be a positive integer or None.")
    scientific = (
        None if scientific_copy_cap is None else int(scientific_copy_cap)
    )
    if practical <= 0:
        return BudgetThresholdEvaluation(
            structural_candidate_id=str(structural_candidate_id),
            seed_fidelity=int(seed_fidelity), threshold_found=False,
            search_status=INVALID_STRUCTURAL_CANDIDATE,
            estimated_min_budget=None, low_infeasible_budget=None,
            high_feasible_budget=None, relative_bracket_width=None,
            practical_minimum_budget=practical, initial_budget=practical,
            initial_budget_hint=initial_budget_hint, initial_budget_hint_used=False,
            budget_trials=(), tested_budgets=(), expansion_trial_count=0,
            refinement_trial_count=0, final_mean_trace_distance=None,
            final_error_success_count=None, final_error_success_fraction=None,
            final_required_success_count=None, final_operational_validity=False,
            non_monotonic_observations=(), warm_started=False,
        )
    initial_requirement = max(
        practical,
        int(initial_budget_hint or practical),
    )
    if scientific is not None and initial_requirement > scientific:
        return BudgetThresholdEvaluation(
            structural_candidate_id=str(structural_candidate_id),
            seed_fidelity=int(seed_fidelity), threshold_found=False,
            search_status=PRECHECK_CAP_REJECTED,
            estimated_min_budget=None, low_infeasible_budget=None,
            high_feasible_budget=None, relative_bracket_width=None,
            practical_minimum_budget=practical, initial_budget=initial_requirement,
            initial_budget_hint=initial_budget_hint,
            initial_budget_hint_used=initial_budget_hint is not None,
            budget_trials=(), tested_budgets=(), expansion_trial_count=0,
            refinement_trial_count=0, final_mean_trace_distance=None,
            final_error_success_count=None, final_error_success_fraction=None,
            final_required_success_count=None, final_operational_validity=False,
            non_monotonic_observations=(), warm_started=False,
            terminal_search_status=PRECHECK_CAP_REJECTED,
        )
    if seed_fidelity <= 0 or final_seed_count <= 0 or seed_fidelity > final_seed_count:
        raise ValueError("seed_fidelity must lie in [1, final_seed_count].")
    if not math.isfinite(float(growth_factor)) or float(growth_factor) <= 1.0:
        raise ValueError("growth_factor must exceed one.")
    if not math.isfinite(float(relative_tolerance)) or not 0.0 < float(relative_tolerance) < 1.0:
        raise ValueError("relative_tolerance must lie in (0,1).")
    if max_expansion_rounds <= 0:
        raise ValueError("max_expansion_rounds must be positive.")
    if previous_cumulative_expansion_trials < 0:
        raise ValueError("previous_cumulative_expansion_trials must be nonnegative.")
    if incumbent_confirmation_probes < 0:
        raise ValueError("incumbent_confirmation_probes must be nonnegative.")
    if hard_safety_budget is not None and int(hard_safety_budget) < practical:
        return BudgetThresholdEvaluation(
            structural_candidate_id=str(structural_candidate_id),
            seed_fidelity=int(seed_fidelity), threshold_found=False,
            search_status=COMPUTATIONAL_SAFETY_LIMIT_REACHED,
            estimated_min_budget=None, low_infeasible_budget=None,
            high_feasible_budget=None, relative_bracket_width=None,
            practical_minimum_budget=practical, initial_budget=practical,
            initial_budget_hint=initial_budget_hint, initial_budget_hint_used=False,
            budget_trials=(), tested_budgets=(), expansion_trial_count=0,
            refinement_trial_count=0, final_mean_trace_distance=None,
            final_error_success_count=None, final_error_success_fraction=None,
            final_required_success_count=None, final_operational_validity=False,
            non_monotonic_observations=(), warm_started=False,
        )

    incumbent = (
        None
        if incumbent_budget_hint is None or is_current_incumbent
        else max(practical, int(incumbent_budget_hint))
    )
    if incumbent is not None and scientific is not None and incumbent > scientific:
        # The incumbent request is outside the scientific domain.  Probe the
        # explicit domain boundary instead; this is not treated as the same
        # evaluation as the oversized hint.
        incumbent = scientific
    initial = max(
        practical,
        int(initial_budget_hint or practical),
        int(incumbent or practical),
    )
    hint_used = initial_budget_hint is not None and int(initial_budget_hint) > practical
    if hard_safety_budget is not None:
        initial = min(initial, int(hard_safety_budget))
    historical = sorted(
        {
            int(budget)
            for budget in previous_tested_budgets
            if int(budget) >= practical
            and (hard_safety_budget is None or int(budget) <= int(hard_safety_budget))
            and (scientific is None or int(budget) <= scientific)
        }
    )
    observations: Dict[int, FixedErrorBudgetTrial] = {}
    cumulative_expansions = int(previous_cumulative_expansion_trials)
    incumbent_probes = int(previous_incumbent_probe_count)
    incumbent_confirmations = int(previous_incumbent_confirmation_probe_count)
    numeric_events = int(previous_numeric_safety_event_count)
    terminal_status = previous_terminal_search_status
    terminal_upward = terminal_status in {
        BUDGET_CAP_EXHAUSTED,
        PRECHECK_CAP_REJECTED,
        COMPUTATIONAL_SAFETY_LIMIT_REACHED,
        NUMERIC_SAMPLING_LIMIT_REACHED,
        INCUMBENT_DOMINATED_SEARCH_PRUNED,
    }

    def emit_state(
        *,
        low: Optional[int] = None,
        high: Optional[int] = None,
        estimated: Optional[int] = None,
    ) -> None:
        if on_controller_state is None:
            return
        on_controller_state(
            {
                "tested_budgets": tuple(sorted(observations)),
                "cumulative_expansion_trial_count": cumulative_expansions,
                "terminal_search_status": terminal_status,
                "incumbent_probe_count": incumbent_probes,
                "incumbent_confirmation_probe_count": incumbent_confirmations,
                "numeric_safety_event_count": numeric_events,
                "incumbent_pruned": (
                    terminal_status == INCUMBENT_DOMINATED_SEARCH_PRUNED
                ),
                "largest_tested_budget": max(observations, default=None),
                "estimated_min_budget": estimated,
                "low_infeasible_budget": low,
                "high_feasible_budget": high,
            }
        )

    def observe(budget: int) -> FixedErrorBudgetTrial:
        budget = int(budget)
        if scientific is not None and budget > scientific:
            raise ValueError(
                f"Refusing E2E budget {budget} above scientific_copy_cap={scientific}."
            )
        if budget not in observations:
            observations[budget] = _trial_from_evaluation(
                budget,
                seed_fidelity,
                evaluate_budget(budget),
                final_seed_count=final_seed_count,
            )
        return observations[budget]

    def upward_proposal(current: int) -> Tuple[Optional[int], Optional[str]]:
        """Return the next in-domain probe or the exact terminal boundary."""

        proposed = max(
            int(current) + 1,
            int(math.ceil(int(current) * float(growth_factor))),
        )
        boundaries = []
        if scientific is not None:
            boundaries.append((scientific, BUDGET_CAP_EXHAUSTED))
        if hard_safety_budget is not None:
            boundaries.append(
                (int(hard_safety_budget), COMPUTATIONAL_SAFETY_LIMIT_REACHED)
            )
        if not boundaries:
            return proposed, None
        boundary, status = min(boundaries, key=lambda item: (item[0], item[1]))
        if int(current) >= boundary:
            return None, status
        if proposed > boundary:
            # Evaluate the boundary as its own scientifically meaningful probe;
            # never reinterpret the oversized proposal as that evaluation.
            return boundary, None
        return proposed, None

    for budget in historical:
        observe(budget)
    expansion_count = 0
    if not terminal_upward:
        if incumbent is not None:
            trial = observe(incumbent)
            incumbent_probes += 1
            if trial.computational_status == NUMERIC_SAMPLING_LIMIT_REACHED:
                numeric_events += 1
                terminal_status = NUMERIC_SAMPLING_LIMIT_REACHED
                terminal_upward = True
            emit_state()
        elif not observations:
            trial = observe(initial)
            if trial.computational_status == NUMERIC_SAMPLING_LIMIT_REACHED:
                numeric_events += 1
                terminal_status = NUMERIC_SAMPLING_LIMIT_REACHED
                terminal_upward = True
            emit_state()

    pruned_this_call = False
    if (
        incumbent is not None
        and not terminal_upward
        and not observations[incumbent].feasible
    ):
        current = incumbent
        for _probe_index in range(int(incumbent_confirmation_probes)):
            if cumulative_expansions >= int(max_expansion_rounds):
                terminal_status = COMPUTATIONAL_SAFETY_LIMIT_REACHED
                terminal_upward = True
                emit_state()
                break
            proposed, boundary_status = upward_proposal(current)
            if proposed is None:
                terminal_status = boundary_status
                terminal_upward = True
                emit_state()
                break
            if proposed in observations:
                trial = observations[proposed]
            else:
                trial = observe(proposed)
                cumulative_expansions += 1
                expansion_count += 1
            incumbent_confirmations += 1
            if trial.computational_status == NUMERIC_SAMPLING_LIMIT_REACHED:
                numeric_events += 1
                terminal_status = NUMERIC_SAMPLING_LIMIT_REACHED
                terminal_upward = True
                emit_state()
                break
            emit_state()
            current = proposed
            if trial.feasible:
                break
        if (
            not terminal_upward
            and not any(trial.feasible for trial in observations.values())
        ):
            terminal_status = (
                BUDGET_CAP_EXHAUSTED
                if scientific is not None
                and observations
                and max(observations) >= scientific
                else INCUMBENT_DOMINATED_SEARCH_PRUNED
            )
            terminal_upward = True
            pruned_this_call = (
                terminal_status == INCUMBENT_DOMINATED_SEARCH_PRUNED
            )
            emit_state()

    while (
        not terminal_upward
        and not any(trial.feasible for trial in observations.values())
    ):
        if cumulative_expansions >= int(max_expansion_rounds):
            terminal_status = COMPUTATIONAL_SAFETY_LIMIT_REACHED
            terminal_upward = True
            emit_state()
            break
        current = max(observations)
        proposed, boundary_status = upward_proposal(current)
        if proposed is None:
            terminal_status = boundary_status
            terminal_upward = True
            emit_state()
            break
        if proposed in observations:
            break
        trial = observe(proposed)
        cumulative_expansions += 1
        expansion_count += 1
        if trial.computational_status == NUMERIC_SAMPLING_LIMIT_REACHED:
            numeric_events += 1
            terminal_status = NUMERIC_SAMPLING_LIMIT_REACHED
            terminal_upward = True
            emit_state()
            break
        emit_state()

    feasible_budgets = sorted(
        budget for budget, trial in observations.items() if trial.feasible
    )
    refinement_count = 0
    if feasible_budgets:
        high = feasible_budgets[0]
        lower_failures = sorted(
            budget
            for budget, trial in observations.items()
            if budget < high and not trial.feasible
        )
        low = lower_failures[-1] if lower_failures else practical - 1
        while (high - low) / high > float(relative_tolerance):
            middle = (low + high) // 2
            if middle < practical or middle in (low, high):
                break
            trial = observe(middle)
            refinement_count += 1
            if trial.feasible:
                high = middle
            else:
                low = middle
            emit_state(low=low, high=high, estimated=high)
        selected = observations[high]
        relative_width = float((high - low) / high)
        status = THRESHOLD_FOUND
        threshold_found = True
        estimated = high
    else:
        low = max(observations) if observations else None
        high = None
        selected = observations[low] if low is not None else None
        relative_width = None
        threshold_found = False
        estimated = None
        if (
            observations
            and terminal_status != NUMERIC_SAMPLING_LIMIT_REACHED
            and not any(
                trial.operationally_valid for trial in observations.values()
            )
        ):
            status = OPERATIONAL_FAILURE
        elif terminal_status is not None:
            status = terminal_status
        elif observations and not any(
            trial.operationally_valid for trial in observations.values()
        ):
            status = OPERATIONAL_FAILURE
        else:
            status = (
                BUDGET_CAP_EXHAUSTED
                if scientific is not None
                and observations
                and max(observations) >= scientific
                else COMPUTATIONAL_SAFETY_LIMIT_REACHED
            )
            terminal_status = status

    ordered = tuple(observations[budget] for budget in sorted(observations))
    return BudgetThresholdEvaluation(
        structural_candidate_id=str(structural_candidate_id),
        seed_fidelity=int(seed_fidelity),
        threshold_found=threshold_found,
        search_status=status,
        estimated_min_budget=estimated,
        low_infeasible_budget=low,
        high_feasible_budget=high,
        relative_bracket_width=relative_width,
        practical_minimum_budget=practical,
        initial_budget=initial,
        initial_budget_hint=initial_budget_hint,
        initial_budget_hint_used=hint_used,
        budget_trials=ordered,
        tested_budgets=tuple(sorted(observations)),
        expansion_trial_count=expansion_count,
        refinement_trial_count=refinement_count,
        final_mean_trace_distance=(
            selected.mean_trace_distance if selected is not None else None
        ),
        final_error_success_count=(
            selected.error_success_count if selected is not None else None
        ),
        final_error_success_fraction=(
            selected.error_success_fraction if selected is not None else None
        ),
        final_required_success_count=(
            selected.required_error_success_count if selected is not None else None
        ),
        final_operational_validity=(
            selected.operationally_valid if selected is not None else False
        ),
        non_monotonic_observations=_non_monotonic_pairs(observations),
        warm_started=bool(historical),
        cumulative_expansion_trial_count=cumulative_expansions,
        largest_tested_budget=max(observations, default=None),
        terminal_search_status=(None if threshold_found else terminal_status),
        incumbent_budget_hint=incumbent_budget_hint,
        incumbent_guided_start=bool(incumbent is not None),
        incumbent_probe_count=incumbent_probes,
        incumbent_confirmation_probe_count=incumbent_confirmations,
        incumbent_pruned=bool(
            pruned_this_call
            or terminal_status == INCUMBENT_DOMINATED_SEARCH_PRUNED
        ),
        numeric_safety_event_count=numeric_events,
    )
