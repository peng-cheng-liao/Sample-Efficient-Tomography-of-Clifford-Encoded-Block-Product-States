from dataclasses import replace
import json

import pytest

import Optimization.objective as objective_module
import Optimization.progressive_search as progressive_module
from Optimization import (
    OptimizationConfig,
    OptimizationMode,
    OptimizationObjective,
    ProgressiveSearchConfig,
    aggregate_candidate_evaluations,
    backend_sampling_count_limit,
    derive_candidate,
    estimate_minimum_feasible_budget,
    evaluate_candidate_on_seed,
    optimize_cebp_parameters_progressive,
    unsafe_backend_sampling_quantities,
)
from Optimization.fixed_error_budget import (
    COMPUTATIONAL_SAFETY_LIMIT_REACHED,
    INCUMBENT_DOMINATED_SEARCH_PRUNED,
    NUMERIC_SAMPLING_LIMIT_REACHED,
    THRESHOLD_FOUND,
)
from Optimization.run_parameter_optimization import build_smoke_instance
from Optimization.tests.test_fixed_error_practical_progressive import (
    TOTAL,
    _catalog,
    _config,
    _evaluation,
    _structural,
    _seed,
)


def _continue(previous, fidelity, evaluate, **changes):
    values = dict(
        structural_candidate_id=previous.structural_candidate_id,
        seed_fidelity=fidelity,
        final_seed_count=16,
        practical_minimum_budget=previous.practical_minimum_budget,
        evaluate_budget=evaluate,
        previous_tested_budgets=previous.tested_budgets,
        previous_cumulative_expansion_trials=(
            previous.cumulative_expansion_trial_count
        ),
        previous_terminal_search_status=previous.terminal_search_status,
        previous_incumbent_probe_count=previous.incumbent_probe_count,
        previous_incumbent_confirmation_probe_count=(
            previous.incumbent_confirmation_probe_count
        ),
        previous_numeric_safety_event_count=(
            previous.numeric_safety_event_count
        ),
        max_expansion_rounds=2,
    )
    values.update(changes)
    return estimate_minimum_feasible_budget(**values)


def test_expansion_allowance_is_cumulative_and_terminal_across_fidelities():
    calls = []
    snapshots = []

    def evaluate(fidelity):
        return lambda budget: (
            calls.append((fidelity, budget))
            or _evaluation(budget, fidelity, [0.08] * fidelity)
        )

    one = estimate_minimum_feasible_budget(
        "cumulative",
        seed_fidelity=1,
        final_seed_count=16,
        practical_minimum_budget=100,
        evaluate_budget=evaluate(1),
        max_expansion_rounds=2,
        on_controller_state=lambda state: snapshots.append(dict(state)),
    )
    two = _continue(one, 2, evaluate(2))
    four = _continue(two, 4, evaluate(4))

    assert one.tested_budgets == (100, 200, 400)
    assert one.cumulative_expansion_trial_count == 2
    assert one.search_status == COMPUTATIONAL_SAFETY_LIMIT_REACHED
    assert two.cumulative_expansion_trial_count == 2
    assert four.cumulative_expansion_trial_count == 2
    assert two.search_status == four.search_status == COMPUTATIONAL_SAFETY_LIMIT_REACHED
    assert max(budget for _fidelity, budget in calls) == 400
    assert not any(budget > 400 for _fidelity, budget in calls)
    assert snapshots[-1]["cumulative_expansion_trial_count"] == 2
    assert snapshots[-1]["terminal_search_status"] == COMPUTATIONAL_SAFETY_LIMIT_REACHED
    assert snapshots[-1]["tested_budgets"] == (100, 200, 400)


def test_incumbent_probe_feasible_searches_downward_and_can_improve():
    seen = []
    result = estimate_minimum_feasible_budget(
        "challenger",
        seed_fidelity=1,
        final_seed_count=16,
        practical_minimum_budget=100,
        incumbent_budget_hint=1_000,
        evaluate_budget=lambda budget: (
            seen.append(budget)
            or _evaluation(budget, 1, [0.04 if budget >= 400 else 0.08])
        ),
    )
    assert seen[0] == 1_000
    assert result.search_status == THRESHOLD_FOUND
    assert 400 <= result.estimated_min_budget < 1_000
    assert result.incumbent_guided_start
    assert not result.incumbent_pruned


def test_incumbent_infeasible_confirmation_is_pruned_not_infeasible():
    seen = []
    result = estimate_minimum_feasible_budget(
        "poor",
        seed_fidelity=1,
        final_seed_count=16,
        practical_minimum_budget=100,
        incumbent_budget_hint=1_000,
        incumbent_confirmation_probes=1,
        evaluate_budget=lambda budget: (
            seen.append(budget) or _evaluation(budget, 1, [0.08])
        ),
    )
    assert seen == [1_000, 2_000]
    assert result.search_status == INCUMBENT_DOMINATED_SEARCH_PRUNED
    assert "infeasible" not in result.search_status
    assert result.incumbent_pruned
    assert result.incumbent_confirmation_probe_count == 1


def test_no_incumbent_discovers_first_feasible_and_current_incumbent_is_not_pruned():
    first = estimate_minimum_feasible_budget(
        "first",
        seed_fidelity=1,
        final_seed_count=16,
        practical_minimum_budget=100,
        evaluate_budget=lambda budget: _evaluation(
            budget, 1, [0.04 if budget >= 400 else 0.08]
        ),
    )
    current = estimate_minimum_feasible_budget(
        "current",
        seed_fidelity=1,
        final_seed_count=16,
        practical_minimum_budget=100,
        incumbent_budget_hint=1_000,
        is_current_incumbent=True,
        max_expansion_rounds=2,
        evaluate_budget=lambda budget: _evaluation(budget, 1, [0.08]),
    )
    assert first.threshold_found
    assert current.search_status == COMPUTATIONAL_SAFETY_LIMIT_REACHED
    assert not current.incumbent_pruned
    assert current.initial_budget == 100


def test_numeric_sampling_guard_rejects_before_learner(monkeypatch):
    instance = build_smoke_instance()
    cfg = _config()
    unsafe_budget = backend_sampling_count_limit() + 1
    derived = derive_candidate(
        _structural(),
        n=instance.n,
        d=instance.d,
        total_copies=TOTAL,
        optimization_config=cfg,
        physical_budget=unsafe_budget,
    )
    assert unsafe_backend_sampling_quantities(
        derived, total_copies=unsafe_budget
    )
    monkeypatch.setattr(
        objective_module,
        "full_cebp_tomography",
        lambda *_args, **_kwargs: pytest.fail("unsafe learner trial was invoked"),
    )
    evaluation = evaluate_candidate_on_seed(
        instance,
        derived,
        123,
        TOTAL,
        cfg,
        objective=cfg.effective_objective,
    )
    assert evaluation.preflight_rejected
    assert evaluation.numeric_sampling_limit_reached
    assert evaluation.failure_stage == "preflight_numeric_sampling_safety"
    assert evaluation.failure_reason.startswith(NUMERIC_SAMPLING_LIMIT_REACHED)
    normal = derive_candidate(
        _structural(),
        n=instance.n,
        d=instance.d,
        total_copies=TOTAL,
        optimization_config=cfg,
        physical_budget=TOTAL,
    )
    assert not unsafe_backend_sampling_quantities(normal, total_copies=TOTAL)


def test_numeric_trial_becomes_structured_terminal_controller_status():
    unsafe_seed = replace(
        _evaluation(100, 1, [0.08]).seed_evaluations[0],
        numeric_sampling_limit_reached=True,
        failure_reason=NUMERIC_SAMPLING_LIMIT_REACHED,
        operational_success=False,
        budget_feasible=False,
        trace_distance=None,
        error_feasible=False,
    )
    aggregate = aggregate_candidate_evaluations(
        "numeric",
        (unsafe_seed,),
        objective=OptimizationObjective(
            mode=OptimizationMode.FIXED_ERROR_MIN_COPIES,
            error_target=0.05,
        ),
        N_candidate=100,
    )
    result = estimate_minimum_feasible_budget(
        "numeric",
        seed_fidelity=1,
        final_seed_count=16,
        practical_minimum_budget=100,
        evaluate_budget=lambda _budget: aggregate,
    )
    assert result.search_status == NUMERIC_SAMPLING_LIMIT_REACHED
    assert result.numeric_safety_event_count == 1
    assert result.tested_budgets == (100,)


def test_incumbent_policy_reuses_semantic_budget_seed_cache():
    cache = {}
    calls = []

    def evaluate(budget, fidelity):
        seeds = []
        for seed in range(fidelity):
            key = (budget, seed)
            if key not in cache:
                cache[key] = _evaluation(budget, 1, [0.08]).seed_evaluations[0]
                calls.append(key)
            seeds.append(replace(cache[key], learner_seed=seed))
        return aggregate_candidate_evaluations(
            "cache",
            seeds,
            objective=_config().effective_objective,
            N_candidate=budget,
        )

    first = estimate_minimum_feasible_budget(
        "cache",
        seed_fidelity=1,
        final_seed_count=16,
        practical_minimum_budget=100,
        incumbent_budget_hint=1_000,
        evaluate_budget=lambda budget: evaluate(budget, 1),
    )
    before = len(calls)
    second = _continue(
        first,
        2,
        lambda budget: evaluate(budget, 2),
        incumbent_budget_hint=1_000,
    )
    assert first.incumbent_pruned and second.incumbent_pruned
    assert len(calls) - before == len(first.tested_budgets)
    assert len(calls) == len(set(calls))


def test_mock_efficiency_policy_cuts_unique_budget_trials_materially():
    incumbent = None
    total_unique = 0
    results = []
    for index in range(16):
        threshold = 800 if index == 0 else 100_000
        result = estimate_minimum_feasible_budget(
            f"candidate-{index:04d}",
            seed_fidelity=1,
            final_seed_count=16,
            practical_minimum_budget=100,
            incumbent_budget_hint=incumbent,
            max_expansion_rounds=12,
            evaluate_budget=lambda budget, threshold=threshold: _evaluation(
                budget, 1, [0.04 if budget >= threshold else 0.08]
            ),
        )
        results.append(result)
        total_unique += len(result.tested_budgets)
        if result.threshold_found:
            incumbent = min(incumbent or result.estimated_min_budget, result.estimated_min_budget)

    naive_lower_bound = 16 * 11
    assert results[0].threshold_found
    assert all(item.incumbent_pruned for item in results[1:])
    assert total_unique < naive_lower_bound / 2


def test_progressive_incumbent_guidance_and_checkpoint_state_resume(
    monkeypatch, tmp_path
):
    instance = build_smoke_instance()
    checkpoint = tmp_path / "adaptive-policy.json"
    cfg = _config(
        checkpoint_path=str(checkpoint),
        checkpoint_key="adaptive-policy-v2",
    )
    catalog = _catalog(instance, cfg, 16)
    calls = []
    monkeypatch.setattr(
        progressive_module,
        "_sample_valid_candidates",
        lambda **_kwargs: (catalog, 0),
    )

    def learner(_instance, derived, seed, _total, _cfg, *, objective=None):
        calls.append((derived.parameters.h_min, derived.physical_copy_budget, seed))
        threshold = 400 if derived.parameters.h_min == catalog[0].parameters.h_min else 100_000
        return _seed(
            seed,
            0.04 if derived.physical_copy_budget >= threshold else 0.08,
            realized=min(100, derived.physical_copy_budget),
        )

    monkeypatch.setattr(objective_module, "evaluate_candidate_on_seed", learner)
    progressive = ProgressiveSearchConfig(initial_candidates=16, max_candidates=16)
    first = optimize_cebp_parameters_progressive(
        instance, TOTAL, cfg, progressive_config=progressive
    )
    diagnostics = first.search_metadata.fixed_error_computational_diagnostics
    assert diagnostics.incumbent_probe_count > 0
    assert diagnostics.incumbent_pruned_candidate_count >= 8
    assert diagnostics.cumulative_expansion_max_per_candidate <= 32
    payload = json.loads(checkpoint.read_text())
    states = payload["run_state"]["candidate_search_state"]
    assert len(states) == 16
    assert all("cumulative_expansion_trial_count" in state for state in states.values())
    assert all("terminal_search_status" in state for state in states.values())
    calls_before_resume = len(calls)
    resumed = optimize_cebp_parameters_progressive(
        instance,
        TOTAL,
        replace(cfg, resume_from_checkpoint=True),
        progressive_config=progressive,
    )
    assert resumed.search_metadata.resumed_from_checkpoint
    assert resumed.best_candidate_id == first.best_candidate_id
    assert len(calls) == calls_before_resume
