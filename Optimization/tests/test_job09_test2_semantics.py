"""Focused regression tests for the Job 09 Test 2 fixed-error semantics."""

from copy import deepcopy
import json

import pytest

import Optimization.objective as objective_module
import Optimization.search as search_module
from Optimization import (
    BUDGET_CAP_EXHAUSTED,
    OptimizationConfig,
    OptimizationMode,
    OptimizationObjective,
    ProgressiveSearchConfig,
    SearchSpace,
    aggregate_candidate_evaluations,
    derive_candidate,
    estimate_minimum_feasible_budget,
    evaluate_candidate_on_seed,
)
from Optimization.checkpoint import CheckpointStore, SCHEMA_VERSION
from Optimization.progressive_search import _progressive_checkpoint_metadata
from Optimization.run_parameter_optimization import build_smoke_instance
from Optimization.search import _sample_valid_candidates
from Optimization.tests.test_fixed_error_practical_progressive import (
    TOTAL,
    TUNING,
    _catalog,
    _seed,
    _structural,
)


SCIENTIFIC_CAP = 1_000_000_000
TEST2_OBJECTIVE = OptimizationObjective(
    mode=OptimizationMode.FIXED_ERROR_MIN_COPIES,
    error_target=0.05,
    success_probability_threshold=0.50,
    scientific_copy_cap=SCIENTIFIC_CAP,
)


def _aggregate(errors, *, budget=1_000):
    return aggregate_candidate_evaluations(
        "test2",
        tuple(_seed(index, error, realized=min(100, budget)) for index, error in enumerate(errors)),
        objective=TEST2_OBJECTIVE,
        N_candidate=budget,
    )


def _test2_config(**changes):
    values = dict(
        total_copies=TOTAL,
        tuning_seeds=TUNING,
        holdout_seeds=(),
        objective=TEST2_OBJECTIVE,
    )
    values.update(changes)
    return OptimizationConfig(**values)


def test_threshold_arithmetic_and_independent_feasibility_gates():
    mean_failure = _aggregate([0.04] * 8 + [0.08] * 8)
    success_failure = _aggregate([0.04] * 7 + [0.057] * 9)
    feasible = _aggregate([0.04] * 8 + [0.06] * 8)

    assert feasible.required_error_success_count == 8
    assert mean_failure.error_success_count == 8
    assert mean_failure.mean_error_feasible is False
    assert mean_failure.success_fraction_feasible is True
    assert mean_failure.final_target_feasible is False
    assert success_failure.mean_trace_distance_successful <= 0.05
    assert success_failure.mean_error_feasible is True
    assert success_failure.error_success_count == 7
    assert success_failure.success_fraction_feasible is False
    assert success_failure.final_target_feasible is False
    assert feasible.mean_trace_distance_successful == pytest.approx(0.05)
    assert feasible.error_success_count == 8
    assert feasible.final_target_feasible is True


def test_pre_e2e_cap_rejection_is_replaced_without_consuming_quota(monkeypatch):
    instance = build_smoke_instance()
    config = _test2_config()
    candidates = iter(
        (
            _structural(h_min=0.70),
            _structural(h_min=0.71),
            _structural(h_min=0.72),
        )
    )
    original_minimum = search_module.minimum_practical_copy_budget

    monkeypatch.setattr(
        search_module,
        "sample_candidate",
        lambda *_args, **_kwargs: next(candidates),
    )
    monkeypatch.setattr(
        search_module,
        "minimum_practical_copy_budget",
        lambda candidate, *, d: (
            SCIENTIFIC_CAP + 1
            if candidate.h_min == 0.70
            else original_minimum(candidate, d=d)
        ),
    )
    monkeypatch.setattr(
        objective_module,
        "full_cebp_tomography",
        lambda *_args, **_kwargs: pytest.fail("sampling invoked E2E"),
    )
    diagnostics = {}
    records, rejected = _sample_valid_candidates(
        instance=instance,
        total_copies=TOTAL,
        optimization_config=config,
        search_space=SearchSpace(),
        search_seed=123,
        target_count=2,
        sampling_diagnostics=diagnostics,
    )

    assert len(records) == 2
    assert rejected == 1
    assert diagnostics == {
        "raw_candidate_draw_count": 3,
        "precheck_cap_rejection_count": 1,
        "valid_candidate_count": 2,
    }


def test_valid_candidate_exhausts_cap_and_every_controller_probe_is_in_domain():
    calls = []

    def evaluate(budget):
        calls.append(budget)
        return _aggregate([0.08] * 16, budget=budget)

    result = estimate_minimum_feasible_budget(
        "cap-exhausted",
        seed_fidelity=16,
        final_seed_count=16,
        practical_minimum_budget=400_000_000,
        evaluate_budget=evaluate,
        scientific_copy_cap=SCIENTIFIC_CAP,
    )

    assert result.search_status == BUDGET_CAP_EXHAUSTED
    assert result.terminal_search_status == BUDGET_CAP_EXHAUSTED
    assert result.tested_budgets == (400_000_000, 800_000_000, SCIENTIFIC_CAP)
    assert calls == list(result.tested_budgets)
    assert max(calls) == SCIENTIFIC_CAP
    assert result.budget_trials[-1].scientific_status == "mean_error_failed"

    success_gate = estimate_minimum_feasible_budget(
        "success-gate-cap-exhausted",
        seed_fidelity=16,
        final_seed_count=16,
        practical_minimum_budget=SCIENTIFIC_CAP,
        evaluate_budget=lambda budget: _aggregate(
            [0.04] * 7 + [0.057] * 9,
            budget=budget,
        ),
        scientific_copy_cap=SCIENTIFIC_CAP,
    )
    assert success_gate.search_status == BUDGET_CAP_EXHAUSTED
    assert success_gate.budget_trials[-1].scientific_status == "success_fraction_failed"


def test_doubling_refinement_and_incumbent_confirmation_respect_cap():
    refinement_calls = []
    refined = estimate_minimum_feasible_budget(
        "refinement",
        seed_fidelity=16,
        final_seed_count=16,
        practical_minimum_budget=400_000_000,
        evaluate_budget=lambda budget: (
            refinement_calls.append(budget)
            or _aggregate(
                [0.04 if budget >= 900_000_000 else 0.08] * 16,
                budget=budget,
            )
        ),
        scientific_copy_cap=SCIENTIFIC_CAP,
    )
    confirmation_calls = []
    confirmed = estimate_minimum_feasible_budget(
        "confirmation",
        seed_fidelity=16,
        final_seed_count=16,
        practical_minimum_budget=100_000_000,
        incumbent_budget_hint=800_000_000,
        incumbent_confirmation_probes=1,
        evaluate_budget=lambda budget: (
            confirmation_calls.append(budget)
            or _aggregate([0.08] * 16, budget=budget)
        ),
        scientific_copy_cap=SCIENTIFIC_CAP,
    )

    assert refined.threshold_found
    assert refined.estimated_min_budget <= SCIENTIFIC_CAP
    assert refined.refinement_trial_count > 0
    assert max(refinement_calls) <= SCIENTIFIC_CAP
    assert confirmation_calls == [800_000_000, SCIENTIFIC_CAP]
    assert confirmed.search_status == BUDGET_CAP_EXHAUSTED
    assert max(confirmation_calls) <= SCIENTIFIC_CAP


def test_objective_guard_rejects_oversized_e2e_immediately(monkeypatch):
    instance = build_smoke_instance()
    config = _test2_config()
    derived = derive_candidate(
        _structural(),
        n=instance.n,
        d=instance.d,
        total_copies=TOTAL,
        optimization_config=config,
        physical_budget=SCIENTIFIC_CAP + 1,
    )
    monkeypatch.setattr(
        objective_module,
        "full_cebp_tomography",
        lambda *_args, **_kwargs: pytest.fail("oversized E2E was invoked"),
    )
    with pytest.raises(ValueError, match="scientific_copy_cap"):
        evaluate_candidate_on_seed(
            instance,
            derived,
            TUNING[0],
            TOTAL,
            config,
            objective=TEST2_OBJECTIVE,
        )


def test_checkpoint_rejects_old_semantics_and_preserves_test2_counters(tmp_path):
    instance = build_smoke_instance()
    config = _test2_config(checkpoint_key="job09-test2-unit")
    records = _catalog(instance, config, 16)
    progressive = ProgressiveSearchConfig(initial_candidates=16, max_candidates=16)
    metadata = _progressive_checkpoint_metadata(
        instance=instance,
        objective=TEST2_OBJECTIVE,
        config=config,
        search_space=SearchSpace(),
        records=records,
        progressive_config=progressive,
    )
    assert metadata["objective"]["success_probability_threshold"] == 0.50
    assert metadata["objective"]["scientific_copy_cap"] == SCIENTIFIC_CAP
    assert metadata["fixed_error_search_policy"]["required_success_count"] == 8
    assert metadata["fixed_error_search_policy"]["valid_candidate_quota_semantics_schema"] == 2

    path = tmp_path / "checkpoint.json"
    old_metadata = deepcopy(metadata)
    old_metadata["objective"]["success_probability_threshold"] = 0.85
    old_metadata["objective"].pop("scientific_copy_cap")
    old_metadata["fixed_error_search_policy"]["required_success_count"] = 14
    path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "compatibility": old_metadata,
                "evaluation_cache": [],
                "run_state": {},
                "status": "running",
            }
        ),
        encoding="utf-8",
    )
    store = CheckpointStore(
        str(path), expected_metadata=metadata, every_n_evaluations=1
    )
    with pytest.raises(ValueError, match="Incompatible checkpoint metadata"):
        store.load()

    counters = {
        "raw_candidate_draw_count": 19,
        "precheck_cap_rejection_count": 3,
        "valid_candidate_count": 16,
        "budget_cap_exhausted_count": 2,
        "e2e_evaluation_count": 41,
        "largest_e2e_budget": SCIENTIFIC_CAP,
    }
    store.save({}, {"counters": counters}, status="running")
    cache, state = store.load()
    assert cache == {}
    assert state["counters"] == counters
