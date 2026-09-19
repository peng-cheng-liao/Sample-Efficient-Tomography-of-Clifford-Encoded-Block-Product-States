from dataclasses import asdict, replace

import numpy as np
import pytest

import Optimization.objective as objective_module
import Optimization.progressive_search as progressive_module
from Optimization import (
    BudgetThresholdEvaluation,
    DEFAULT_FIXED_BUDGET_MIN_ERROR_IMPROVEMENT_RATIO,
    FixedErrorBudgetTrial,
    FixedErrorCandidateParameters,
    OptimizationConfig,
    OptimizationMode,
    OptimizationObjective,
    ProgressiveSearchConfig,
    SearchSpace,
    aggregate_candidate_evaluations,
    derive_candidate,
    estimate_minimum_feasible_budget,
    fixed_error_threshold_ranking_key,
    minimum_practical_copy_budget,
    optimize_cebp_parameters_progressive,
    sample_candidate,
)
from Optimization.fixed_error_budget import (
    COMPUTATIONAL_SAFETY_LIMIT_REACHED,
    INVALID_STRUCTURAL_CANDIDATE,
    OPERATIONAL_FAILURE,
    THRESHOLD_FOUND,
)
from Optimization.objective import SeedEvaluation, evaluation_identity
from Optimization.parameterization import PreflightResourceCheck
from Optimization.progressive_search import _progressive_checkpoint_metadata
from Optimization.run_parameter_optimization import build_smoke_instance
from Optimization.search import CandidateRecord


TOTAL = 5_000
TUNING = tuple(range(101, 117))
HOLDOUT = (901, 902)


def _objective(target=0.05, copy_ceiling=None):
    return OptimizationObjective(
        mode=OptimizationMode.FIXED_ERROR_MIN_COPIES,
        copy_ceiling=copy_ceiling,
        error_target=target,
    )


def _config(**changes):
    values = dict(
        total_copies=TOTAL,
        tuning_seeds=TUNING,
        holdout_seeds=HOLDOUT,
        objective=_objective(),
    )
    values.update(changes)
    return OptimizationConfig(**values)


def _structural(**changes):
    values = dict(
        h_min=0.72,
        h_max=0.90,
        theta_tau_multiplier=8.0,
        eta_test=0.10,
        peel_weight=1.0,
        recovery_weight=1.0,
        grouping_weight=1.0,
        syndrome_weight=0.25,
        tomography_weight=2.0,
    )
    values.update(changes)
    return FixedErrorCandidateParameters(**values)


def _check(limit=10**9):
    return PreflightResourceCheck(
        mandatory_bell_copies=4,
        worst_case_sign_reservation=0,
        preflight_fixed_reservation=4,
        grouping_safety_estimate=1,
        tomography_safety_estimate=1,
        preflight_safety_estimate=6,
        rigorous_grouping_upper_bound=1,
        rigorous_tomography_upper_bound=1,
        rigorous_total_upper_bound=6,
        single_grouping_query_shots_estimate=1,
        safety_limit=limit,
        runtime_safety_rejected=False,
        mathematical_budget_rejected=False,
        safe_to_execute=True,
        reason="mock-safe",
        guarantee_level="mock",
    )


def _seed(seed, error, *, operational=True, budget=True, realized=100):
    finite = error is not None
    return SeedEvaluation(
        learner_seed=int(seed),
        preflight_rejected=not operational,
        preflight_resource_check=_check(),
        operational_success=operational,
        budget_feasible=budget,
        post_run_realized_budget_feasible=budget,
        loss=float(error) if finite else 1.0,
        loss_computed=finite,
        trace_distance=float(error) if finite else None,
        failure_stage=None if operational else "mock",
        failure_reason=None if operational else "mock failure",
        realized_copy_ledger=(("mock", int(realized)),),
        realized_total=int(realized),
        copy_utilization=0.5,
        recovered_t=1 if operational else None,
        cluster_sizes=(1, 2) if operational else (),
        register_sizes=(1, 2) if operational else (),
        j_aux_size=0 if operational else None,
        error_feasible=bool(finite and error <= 0.05 and operational and budget),
        error_excess=(max(0.0, error - 0.05) if finite else 1.0),
        estimator_available=bool(finite and operational),
    )


def _evaluation(budget, fidelity, errors, *, operational=True):
    return aggregate_candidate_evaluations(
        f"candidate-budget-{budget}",
        tuple(
            _seed(index, error, operational=operational, realized=min(100, budget))
            for index, error in enumerate(tuple(errors)[:fidelity])
        ),
        objective=_objective(),
        N_candidate=budget,
    )


def _threshold_result(candidate_id, budget=1_000, fidelity=16):
    trial = FixedErrorBudgetTrial(
        budget=budget,
        seed_fidelity=fidelity,
        operationally_valid=True,
        mean_trace_distance=0.04,
        mean_error_feasible=True,
        error_success_count=fidelity,
        error_success_fraction=1.0,
        required_error_success_count=14 if fidelity == 16 else None,
        success_fraction_feasible=True if fidelity == 16 else None,
        feasible=True,
    )
    return BudgetThresholdEvaluation(
        structural_candidate_id=candidate_id,
        seed_fidelity=fidelity,
        threshold_found=True,
        search_status=THRESHOLD_FOUND,
        estimated_min_budget=budget,
        low_infeasible_budget=budget - 10,
        high_feasible_budget=budget,
        relative_bracket_width=0.01,
        practical_minimum_budget=10,
        initial_budget=10,
        initial_budget_hint=None,
        initial_budget_hint_used=False,
        budget_trials=(trial,),
        tested_budgets=(budget,),
        expansion_trial_count=1,
        refinement_trial_count=1,
        final_mean_trace_distance=0.04,
        final_error_success_count=fidelity,
        final_error_success_fraction=1.0,
        final_required_success_count=14 if fidelity == 16 else None,
        final_operational_validity=True,
        non_monotonic_observations=(),
        warm_started=False,
        cumulative_expansion_trial_count=1,
        largest_tested_budget=budget,
    )


def test_structural_candidate_has_no_budget_coordinate_and_reads_legacy_field():
    modern = _structural()
    legacy = FixedErrorCandidateParameters(N_candidate=123_456, **asdict(modern))
    assert modern == legacy
    assert "N_candidate" not in asdict(modern)
    assert not hasattr(modern, "N_candidate")


def test_fixed_error_objective_accepts_no_ceiling_and_legacy_ceiling_is_inert():
    assert _objective().copy_ceiling is None
    cfg = _config(objective=_objective(copy_ceiling=100))
    derived = derive_candidate(
        _structural(), n=6, d=3, total_copies=TOTAL,
        optimization_config=cfg, physical_budget=14_000_000,
    )
    assert derived.physical_copy_budget == 14_000_000


def test_fixed_error_trial_budget_does_not_require_global_total_copies_match():
    cfg = _config()
    derived = derive_candidate(
        _structural(), n=6, d=3, total_copies=123,
        optimization_config=cfg, physical_budget=2_000,
    )
    assert derived.physical_copy_budget == 2_000


def test_sampling_is_structural_and_independent_of_legacy_ceiling():
    first = sample_candidate(
        np.random.default_rng(7), SearchSpace(),
        mode=OptimizationMode.FIXED_ERROR_MIN_COPIES, copy_ceiling=10,
    )
    second = sample_candidate(
        np.random.default_rng(7), SearchSpace(),
        mode=OptimizationMode.FIXED_ERROR_MIN_COPIES, copy_ceiling=10**12,
    )
    assert first == second
    assert "N_candidate" not in asdict(first)


def test_trial_identity_is_lambda_budget_seed_and_ignores_legacy_ceiling():
    cfg = _config()
    candidate = _structural()
    d1 = derive_candidate(
        candidate, n=6, d=3, total_copies=TOTAL,
        optimization_config=cfg, physical_budget=1_000,
    )
    d2 = derive_candidate(
        candidate, n=6, d=3, total_copies=TOTAL,
        optimization_config=cfg, physical_budget=2_000,
    )
    changed = derive_candidate(
        replace(candidate, eta_test=0.11), n=6, d=3, total_copies=TOTAL,
        optimization_config=cfg, physical_budget=1_000,
    )
    key = evaluation_identity(d1, 1, TOTAL, cfg, cfg.effective_objective)
    legacy_objective = _objective(copy_ceiling=999)
    legacy_cfg = replace(cfg, objective=legacy_objective)
    assert key == evaluation_identity(d1, 1, TOTAL, legacy_cfg, legacy_objective)
    assert key != evaluation_identity(d2, 1, TOTAL, cfg, cfg.effective_objective)
    assert key != evaluation_identity(d1, 2, TOTAL, cfg, cfg.effective_objective)
    assert key != evaluation_identity(changed, 1, TOTAL, cfg, cfg.effective_objective)


def test_initial_practical_budget_and_hint_are_starting_points_not_caps():
    seen = []
    result = estimate_minimum_feasible_budget(
        "x", seed_fidelity=1, final_seed_count=16,
        practical_minimum_budget=100, initial_budget_hint=250,
        evaluate_budget=lambda budget: (
            seen.append(budget) or _evaluation(budget, 1, [0.08 if budget < 500 else 0.04])
        ),
        growth_factor=2.0,
    )
    assert seen[:2] == [250, 500]
    assert result.initial_budget == 250 and result.initial_budget_hint_used
    assert result.estimated_min_budget <= 500


def test_upward_bracketing_and_relative_refinement():
    seen = []
    result = estimate_minimum_feasible_budget(
        "x", seed_fidelity=1, final_seed_count=16,
        practical_minimum_budget=100,
        evaluate_budget=lambda budget: (
            seen.append(budget) or _evaluation(budget, 1, [0.04 if budget >= 400 else 0.08])
        ),
        growth_factor=2.0, relative_tolerance=0.01,
    )
    assert seen[:3] == [100, 200, 400]
    assert result.search_status == THRESHOLD_FOUND
    assert result.low_infeasible_budget < 400 <= result.high_feasible_budget
    assert result.relative_bracket_width <= 0.01
    assert result.expansion_trial_count == 2
    assert result.refinement_trial_count > 0


def test_first_budget_feasible_never_tests_below_practical_minimum():
    seen = []
    result = estimate_minimum_feasible_budget(
        "x", seed_fidelity=1, final_seed_count=16,
        practical_minimum_budget=100,
        evaluate_budget=lambda budget: (
            seen.append(budget) or _evaluation(budget, 1, [0.04])
        ),
    )
    assert min(seen) == 100
    assert result.estimated_min_budget == 100
    assert result.low_infeasible_budget == 99


def test_d3_style_7m_to_14m_expansion_then_refinement():
    seen = []
    result = estimate_minimum_feasible_budget(
        "d3", seed_fidelity=16, final_seed_count=16,
        practical_minimum_budget=7_000_000,
        evaluate_budget=lambda budget: (
            seen.append(budget) or _evaluation(
                budget, 16, [0.04 if budget >= 14_000_000 else 0.08] * 16
            )
        ),
    )
    assert seen[:2] == [7_000_000, 14_000_000]
    assert result.threshold_found and result.refinement_trial_count > 0


def test_safety_limit_is_not_reported_as_infeasibility():
    result = estimate_minimum_feasible_budget(
        "x", seed_fidelity=1, final_seed_count=16,
        practical_minimum_budget=100, hard_safety_budget=300,
        evaluate_budget=lambda budget: _evaluation(budget, 1, [0.08]),
    )
    assert not result.threshold_found
    assert result.search_status == COMPUTATIONAL_SAFETY_LIMIT_REACHED
    assert "infeasible" not in result.search_status


def test_invalid_structural_budget_returns_distinct_status_without_execution():
    result = estimate_minimum_feasible_budget(
        "invalid", seed_fidelity=1, final_seed_count=16,
        practical_minimum_budget=0,
        evaluate_budget=lambda _budget: pytest.fail("invalid candidate executed"),
    )
    assert result.search_status == INVALID_STRUCTURAL_CANDIDATE
    assert not result.tested_budgets


def test_all_failed_trials_return_operational_failure_status():
    result = estimate_minimum_feasible_budget(
        "failed", seed_fidelity=1, final_seed_count=16,
        practical_minimum_budget=100, max_expansion_rounds=1,
        evaluate_budget=lambda budget: _evaluation(
            budget, 1, [None], operational=False
        ),
    )
    assert result.search_status == OPERATIONAL_FAILURE


def test_non_monotonic_history_is_recorded_and_bounded():
    pattern = {100: 0.08, 200: 0.04, 300: 0.08, 400: 0.04}
    result = estimate_minimum_feasible_budget(
        "x", seed_fidelity=1, final_seed_count=16,
        practical_minimum_budget=100,
        previous_tested_budgets=pattern,
        evaluate_budget=lambda budget: _evaluation(
            budget, 1, [pattern.get(budget, 0.04 if budget >= 200 else 0.08)]
        ),
    )
    assert (200, 300) in result.non_monotonic_observations
    assert result.threshold_found
    assert result.estimated_min_budget <= 200


def test_warm_start_adds_only_missing_seed_trials():
    calls = []
    cache = {}

    def evaluate(budget, fidelity):
        seeds = []
        for seed in range(fidelity):
            key = (budget, seed)
            if key not in cache:
                cache[key] = _seed(seed, 0.04 if budget >= 200 else 0.08)
                calls.append(key)
            seeds.append(cache[key])
        return aggregate_candidate_evaluations(
            "warm", seeds, objective=_objective(), N_candidate=budget
        )

    first = estimate_minimum_feasible_budget(
        "warm", seed_fidelity=1, final_seed_count=16,
        practical_minimum_budget=100,
        evaluate_budget=lambda budget: evaluate(budget, 1),
    )
    calls_after_first = len(calls)
    second = estimate_minimum_feasible_budget(
        "warm", seed_fidelity=2, final_seed_count=16,
        practical_minimum_budget=100,
        previous_tested_budgets=first.tested_budgets,
        evaluate_budget=lambda budget: evaluate(budget, 2),
    )
    assert second.warm_started
    assert len(calls) - calls_after_first == len(first.tested_budgets)
    assert len(calls) == len(set(calls))


def test_provisional_fidelity_uses_mean_only_but_final_requires_14_of_16():
    early = estimate_minimum_feasible_budget(
        "early", seed_fidelity=2, final_seed_count=16,
        practical_minimum_budget=100,
        evaluate_budget=lambda budget: _evaluation(budget, 2, [0.0, 0.10]),
    )
    assert early.threshold_found
    final_errors = [0.04] * 13 + [0.06] * 3
    final = estimate_minimum_feasible_budget(
        "final", seed_fidelity=16, final_seed_count=16,
        practical_minimum_budget=100, max_expansion_rounds=1,
        evaluate_budget=lambda budget: _evaluation(budget, 16, final_errors),
    )
    assert not final.threshold_found
    assert final.budget_trials[0].mean_error_feasible
    assert final.budget_trials[0].error_success_count == 13
    assert final.budget_trials[0].success_fraction_feasible is False


def test_final_fidelity_requires_operational_mean_and_success():
    passing = _evaluation(100, 16, [0.04] * 14 + [0.06] * 2)
    mean_only = _evaluation(100, 16, [0.04] * 13 + [0.06] * 3)
    failed = _evaluation(100, 16, [0.04] * 16, operational=False)
    assert passing.final_target_feasible and passing.required_error_success_count == 14
    assert mean_only.mean_error_feasible and not mean_only.final_target_feasible
    assert not failed.final_target_feasible


def test_structural_threshold_ranking_is_deterministic():
    small = _threshold_result("small", 1_000)
    large = _threshold_result("large", 2_000)
    no_threshold = replace(
        small, structural_candidate_id="none", threshold_found=False,
        search_status=COMPUTATIONAL_SAFETY_LIMIT_REACHED,
        estimated_min_budget=None, high_feasible_budget=None,
    )
    assert fixed_error_threshold_ranking_key(small) < fixed_error_threshold_ranking_key(large)
    assert fixed_error_threshold_ranking_key(large) < fixed_error_threshold_ranking_key(no_threshold)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_trial_budgets_preserve_stage_allocation_branches(d):
    cfg = _config()
    candidate = _structural(grouping_weight=4.0)
    practical = minimum_practical_copy_budget(candidate, d=d)
    for budget in (practical, practical * 3):
        derived = derive_candidate(
            candidate, n=6, d=d, total_copies=TOTAL,
            optimization_config=cfg, physical_budget=budget,
        )
        caps = dict(derived.fixed_budget_stage_caps)
        assert sum(caps.values()) == budget
        assert (caps["grouping"] == 0) is (d == 1)


def _catalog(instance, cfg, count=256):
    records = []
    for index in range(count):
        candidate = _structural(h_min=0.60 + index * 0.0005, h_max=0.90)
        derived = derive_candidate(
            candidate, n=instance.n, d=instance.d, total_copies=TOTAL,
            optimization_config=cfg,
        )
        records.append(CandidateRecord(f"candidate-{index:04d}", candidate, derived))
    return tuple(records)


def test_full_256_structural_policy_and_no_holdout(monkeypatch):
    instance = build_smoke_instance()
    cfg = _config()
    catalog = _catalog(instance, cfg)
    calls = []
    monkeypatch.setattr(
        progressive_module, "_sample_valid_candidates",
        lambda **kwargs: (catalog[: int(kwargs["target_count"])], 0),
    )
    monkeypatch.setattr(
        progressive_module, "estimate_minimum_feasible_budget",
        lambda structural_candidate_id, seed_fidelity, **kwargs: _threshold_result(
            structural_candidate_id, 1_000, seed_fidelity
        ),
    )

    def learner(_instance, derived, seed, _total, _cfg, *, objective=None):
        calls.append(seed)
        return _seed(seed, 0.04, realized=min(100, derived.physical_copy_budget))

    monkeypatch.setattr(objective_module, "evaluate_candidate_on_seed", learner)
    result = optimize_cebp_parameters_progressive(
        instance, TOTAL, cfg, progressive_config=ProgressiveSearchConfig()
    )
    assert DEFAULT_FIXED_BUDGET_MIN_ERROR_IMPROVEMENT_RATIO == 0.0
    assert [item.candidate_pool_size for item in result.search_metadata.round_summaries] == [
        16, 32, 64, 128, 256
    ]
    assert result.search_metadata.final_candidate_pool_size == 256
    assert result.search_metadata.termination_reason == "max_structural_candidates"
    assert "N_candidate" not in asdict(result.best_candidate)
    assert result.best_derived_candidate.physical_copy_budget == 1_000
    assert result.comparison_evaluation.final_target_feasible
    assert result.holdout_evaluation is None
    assert set(calls).isdisjoint(HOLDOUT)


def test_checkpoint_fingerprint_tracks_controller_not_legacy_ceiling_or_holdout():
    instance = build_smoke_instance()
    cfg = _config(checkpoint_key="fingerprint")
    progressive = ProgressiveSearchConfig(initial_candidates=16, max_candidates=16)
    records = _catalog(instance, cfg, 16)

    def metadata(candidate_cfg):
        return _progressive_checkpoint_metadata(
            instance=instance,
            objective=candidate_cfg.effective_objective,
            config=candidate_cfg,
            search_space=SearchSpace(),
            records=records,
            progressive_config=progressive,
        )

    baseline = metadata(cfg)
    assert baseline["progressive_search_schema"] == 9
    assert metadata(replace(cfg, objective=_objective(copy_ceiling=999_999))) == baseline
    assert metadata(replace(cfg, holdout_seeds=(777,))) == baseline
    assert metadata(replace(cfg, fixed_error_budget_growth_factor=3.0)) != baseline
    assert metadata(replace(cfg, fixed_error_budget_relative_tolerance=0.02)) != baseline
    assert metadata(replace(cfg, fixed_error_incumbent_confirmation_probes=2)) != baseline
    assert metadata(replace(cfg, objective=replace(_objective(), success_probability_threshold=0.9))) != baseline
    assert metadata(replace(cfg, objective=replace(_objective(), scientific_copy_cap=10**9))) != baseline


def test_structured_diagnostics_count_unique_trials_in_mock_progressive(monkeypatch):
    instance = build_smoke_instance()
    cfg = _config()
    catalog = _catalog(instance, cfg, 16)
    monkeypatch.setattr(
        progressive_module, "_sample_valid_candidates",
        lambda **_kwargs: (catalog, 0),
    )
    monkeypatch.setattr(
        progressive_module, "estimate_minimum_feasible_budget",
        lambda structural_candidate_id, seed_fidelity, **kwargs: _threshold_result(
            structural_candidate_id, 1_000, seed_fidelity
        ),
    )
    monkeypatch.setattr(
        objective_module, "evaluate_candidate_on_seed",
        lambda _instance, derived, seed, _total, _cfg, *, objective=None: _seed(
            seed, 0.04, realized=min(100, derived.physical_copy_budget)
        ),
    )
    result = optimize_cebp_parameters_progressive(
        instance, TOTAL, cfg,
        progressive_config=ProgressiveSearchConfig(
            initial_candidates=16, max_candidates=16
        ),
    )
    diagnostics = result.search_metadata.fixed_error_computational_diagnostics
    assert diagnostics.unique_structural_candidate_count == 16
    assert diagnostics.unique_structural_budget_trial_count == 16
    assert diagnostics.unique_structural_budget_seed_evaluation_count == 16
    assert diagnostics.semantic_cache_hit_count == 0
    assert diagnostics.upward_expansion_trial_count == 16
    assert diagnostics.relative_refinement_trial_count == 33
    assert diagnostics.average_budget_trials_by_seed_fidelity == (
        (1, 1.0), (2, 1.0), (4, 1.0), (8, 1.0), (16, 1.0)
    )


def test_progressive_returns_safety_status_without_out_of_policy_trial(monkeypatch):
    instance = build_smoke_instance()
    cfg = _config(fixed_error_hard_safety_budget=1)
    catalog = _catalog(instance, cfg, 16)
    monkeypatch.setattr(
        progressive_module, "_sample_valid_candidates",
        lambda **_kwargs: (catalog, 0),
    )
    monkeypatch.setattr(
        objective_module,
        "evaluate_candidate_on_seed",
        lambda *_args, **_kwargs: pytest.fail("learner must not run beyond safety"),
    )
    result = optimize_cebp_parameters_progressive(
        instance, TOTAL, cfg,
        progressive_config=ProgressiveSearchConfig(
            initial_candidates=16, max_candidates=16
        ),
    )
    assert result.comparison_evaluation is None
    assert result.actual_learner_run_count == 0
    assert result.search_metadata.termination_reason.endswith("before_safety_limit")
    assert all(
        item.search_status == COMPUTATIONAL_SAFETY_LIMIT_REACHED
        for item in result.search_metadata.fixed_error_threshold_evaluations
    )


def test_fixed_error_checkpoint_resume_preserves_structural_threshold_state(
    monkeypatch, tmp_path
):
    instance = build_smoke_instance()
    checkpoint = tmp_path / "fixed-error-adaptive.json"
    base = _config(checkpoint_path=str(checkpoint), checkpoint_key="adaptive")
    catalog = _catalog(instance, base, 16)
    calls = []
    monkeypatch.setattr(
        progressive_module, "_sample_valid_candidates",
        lambda **_kwargs: (catalog, 0),
    )
    monkeypatch.setattr(
        progressive_module, "estimate_minimum_feasible_budget",
        lambda structural_candidate_id, seed_fidelity, **kwargs: _threshold_result(
            structural_candidate_id, 1_000, seed_fidelity
        ),
    )

    def learner(_instance, derived, seed, _total, _cfg, *, objective=None):
        calls.append((derived.physical_copy_budget, seed))
        return _seed(seed, 0.04, realized=100)

    monkeypatch.setattr(objective_module, "evaluate_candidate_on_seed", learner)
    progressive = ProgressiveSearchConfig(initial_candidates=16, max_candidates=16)
    first = optimize_cebp_parameters_progressive(
        instance, TOTAL, base, progressive_config=progressive
    )
    calls_after_first = len(calls)
    resumed = optimize_cebp_parameters_progressive(
        instance, 123, replace(base, resume_from_checkpoint=True),
        progressive_config=progressive,
    )
    assert resumed.search_metadata.resumed_from_checkpoint
    assert resumed.best_candidate == first.best_candidate
    assert resumed.best_derived_candidate == first.best_derived_candidate
    assert resumed.comparison_evaluation == first.comparison_evaluation
    assert len(calls) == calls_after_first
