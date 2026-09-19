from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import qutip as qt

import main_v2
import Optimization.objective as objective
from Optimization.parameterization import (
    CandidateParameters,
    FixedBudgetCandidateParameters,
    InvalidCandidateError,
    OptimizationConfig,
    SearchSpace,
    candidate_from_manual_configs,
    candidate_to_end_to_end_config,
    derive_candidate,
    sample_candidate,
)
from Optimization.run_parameter_optimization import build_smoke_instance, smoke_search_space
from Optimization.search import _sample_valid_candidates, optimize_cebp_parameters
from Optimization.specification import OptimizationMode, OptimizationObjective


BUDGET = 300_000
BASE_PARAMETERS = CandidateParameters(
    alpha_peel=0.0334,
    alpha_rank=0.20,
    alpha_sgn=0.001,
    c_peel=1.25,
    c_rank=1.30,
    h_min=0.80,
    h_span=0.10,
    theta=0.12,
    eta_test=0.10,
    kappa_ratio=0.60,
    epsilon_tom=1.0,
)


def _config(**kwargs):
    values = dict(
        total_copies=BUDGET,
        tuning_seeds=(10001, 10002),
        holdout_seeds=(20001,),
        halving_seed_counts=(1, 2),
        number_of_candidates=4,
    )
    values.update(kwargs)
    return OptimizationConfig(**values)


def _derived(config=None, parameters=BASE_PARAMETERS):
    config = config or _config()
    return derive_candidate(
        parameters,
        n=3,
        d=2,
        total_copies=BUDGET,
        optimization_config=config,
    )


def _fake_result(*, success=True, total=100, certified=False, register_sizes=(2,)):
    registers = tuple(
        ((index,), tuple(range(size))) for index, size in enumerate(register_sizes)
    )
    return SimpleNamespace(
        theorem_certified=certified,
        realized_total=total,
        realized_copy_ledger=main_v2.CopyLedger((("fake_pool", total),)),
        success=success,
        failure_stage=None if success else "fake",
        failure_reason=None if success else "operational_failure",
        peeling=SimpleNamespace(t=1),
        grouping=SimpleNamespace(clusters=tuple((index,) for index in range(len(registers)))),
        localization=SimpleNamespace(J_C=registers, J_aux=()),
    )


@pytest.fixture(scope="module")
def smoke_result():
    config = _config()
    return optimize_cebp_parameters(
        build_smoke_instance(), BUDGET, config, smoke_search_space()
    )


def test_candidate_sampling_is_deterministic_for_fixed_seed():
    space = SearchSpace()
    first = sample_candidate(np.random.default_rng(19), space)
    second = sample_candidate(np.random.default_rng(19), space)
    assert first == second


def test_changed_search_seed_changes_sample():
    space = SearchSpace()
    assert sample_candidate(np.random.default_rng(19), space) != sample_candidate(
        np.random.default_rng(20), space
    )


def test_d1_optimizer_entry_allows_practical_fixed_budget_mode():
    instance = main_v2.random_cebp_state(
        1, 1, block_sizes=(1,), block_states=(qt.qeye(2) / 2,), seed=1
    )
    result = optimize_cebp_parameters(
        instance, BUDGET, _config(number_of_candidates=1, halving_seed_counts=(1,))
    )
    assert result.best_derived_candidate.execution_policy == "fixed_budget_graceful"
    assert dict(result.best_derived_candidate.fixed_budget_stage_caps)["grouping"] == 0


def test_failure_loss_is_fixed_at_one():
    assert OptimizationConfig().failure_loss == 1.0
    with pytest.raises(ValueError, match="fixed at 1.0|equal 1.0"):
        OptimizationConfig(failure_loss=0.4)


def test_tau_zeta_inversions_match_public_formula():
    derived = _derived()
    assert main_v2.bell_score_uniform_radius(
        derived.M1, 3, derived.zeta_peel
    ) == pytest.approx(derived.tau1, rel=2e-13)
    assert main_v2.bell_score_uniform_radius(
        derived.M2, 3, derived.zeta_rank
    ) == pytest.approx(derived.tau_rank, rel=2e-13)


def test_invalid_h_window_is_rejected():
    with pytest.raises(InvalidCandidateError, match="h_min"):
        _derived(parameters=replace(BASE_PARAMETERS, h_min=0.90, h_span=0.10))


def test_fixed_budget_theta_not_above_tau_rank_is_diagnostic_only():
    valid = _derived()
    derived = _derived(parameters=replace(BASE_PARAMETERS, theta=valid.tau_rank))
    assert derived.theta == pytest.approx(valid.tau_rank)


def test_fixed_budget_tau_kappa_is_global_diagnostic_only():
    parameters = replace(BASE_PARAMETERS, eta_test=0.10, kappa_ratio=1.20)
    derived = _derived(parameters=parameters)
    learner = candidate_to_end_to_end_config(
        derived,
        d=2,
        learner_seed=9,
        total_copies=BUDGET,
        optimization_config=_config(),
    )
    assert derived.tau_kappa == pytest.approx(_config().fixed_budget_tau_kappa_diagnostic)
    assert learner.grouping_override.eta_test == pytest.approx(0.10)
    assert learner.grouping_override.tau_kappa == pytest.approx(derived.tau_kappa)
    assert learner.grouping_override.sampling_policy == "fixed_budget"
    assert learner.grouping_override.allow_no_false_merge_margin_failure


def test_fixed_reservation_names_have_exact_semantics():
    derived = _derived()
    assert derived.mandatory_bell_copies == 2 * derived.M1 + 2 * derived.M2
    assert derived.M_sgn is None
    assert derived.worst_case_sign_reservation == 0
    assert derived.preflight_fixed_reservation == (
        derived.mandatory_bell_copies + derived.worst_case_sign_reservation
    )
    check = objective.preflight_resource_check(
        derived, n=3, d=2, total_copies=BUDGET, optimization_config=_config()
    )
    assert check.mandatory_bell_copies == derived.mandatory_bell_copies
    assert check.worst_case_sign_reservation == derived.worst_case_sign_reservation
    assert check.preflight_fixed_reservation == derived.preflight_fixed_reservation


def test_fixed_preflight_rejection_occurs_before_learner(monkeypatch):
    instance = build_smoke_instance()
    derived = replace(_derived(), peeling_physical_copies=BUDGET)
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("learner should not run")

    monkeypatch.setattr(objective, "full_cebp_tomography", forbidden)
    evaluation = objective.evaluate_candidate_on_seed(
        instance, derived, 9, BUDGET, _config()
    )
    assert evaluation.preflight_rejected
    assert evaluation.failure_stage == "preflight_budget"
    assert evaluation.realized_total == 0 and evaluation.loss == 1.0
    assert not called


def test_post_run_over_budget_receives_loss_one_without_oracle(monkeypatch):
    monkeypatch.setattr(
        objective,
        "full_cebp_tomography",
        lambda *_args, **_kwargs: _fake_result(total=BUDGET + 1),
    )
    monkeypatch.setattr(
        objective,
        "debug_end_to_end_trace_error",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("oracle called")),
    )
    evaluation = objective.evaluate_candidate_on_seed(
        build_smoke_instance(), _derived(), 9, BUDGET, _config()
    )
    assert not evaluation.preflight_rejected
    assert evaluation.post_run_realized_budget_feasible is False
    assert evaluation.loss == 1.0 and evaluation.trace_distance is None


def test_operational_failure_receives_loss_one(monkeypatch):
    monkeypatch.setattr(
        objective,
        "full_cebp_tomography",
        lambda *_args, **_kwargs: _fake_result(success=False),
    )
    evaluation = objective.evaluate_candidate_on_seed(
        build_smoke_instance(), _derived(), 9, BUDGET, _config()
    )
    assert not evaluation.operational_success and evaluation.loss == 1.0


def test_register_size_summary_handles_nonuniform_jc():
    result = _fake_result(register_sizes=(3, 2))
    _t, _clusters, register_sizes, _aux = objective._compact_structure(result)
    assert register_sizes == (3, 2)


def test_common_random_number_prefix_is_exposed(smoke_result):
    tuning = _config().tuning_seeds
    for summary in smoke_result.all_candidate_summaries:
        actual = tuple(item.learner_seed for item in summary.seed_evaluations)
        assert actual == tuning[: summary.n_seeds_evaluated]


def test_cache_prevents_duplicate_learner_calls(monkeypatch):
    calls = 0

    def fake(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return _fake_result()

    monkeypatch.setattr(objective, "full_cebp_tomography", fake)
    monkeypatch.setattr(
        objective, "debug_end_to_end_trace_error", lambda *_a, **_k: 0.4
    )
    cache = {}
    args = (build_smoke_instance(), "cached", _derived(), (7, 8), BUDGET, _config())
    first = objective.evaluate_candidate(*args, cache=cache)
    objective.evaluate_candidate(*args, cache=cache)
    assert first.mean_loss == pytest.approx(0.2)
    assert calls == 2 and len(cache) == 2


def test_unequal_fidelity_summaries_expose_seed_count(smoke_result):
    counts = {item.n_seeds_evaluated for item in smoke_result.all_candidate_summaries}
    assert counts == {1, 2}


def test_holdout_is_disjoint_and_does_not_change_candidate(smoke_result):
    assert not set(smoke_result.search_metadata.tuning_seeds) & set(
        smoke_result.search_metadata.holdout_seeds
    )
    assert smoke_result.holdout_evaluation.candidate_id == smoke_result.best_candidate_id


def test_nontrivial_smoke_uses_all_adaptive_pools(smoke_result):
    ledger = dict(smoke_result.tuning_evaluation.seed_evaluations[0].realized_copy_ledger)
    assert ledger["grouping_ordinary_pool"] > 0
    assert ledger["syndrome_sign_pool"] > 0
    assert ledger["block_tomography_pool"] > 0


def test_nontrivial_smoke_is_budget_feasible_and_finite(smoke_result):
    evaluation = smoke_result.tuning_evaluation
    assert evaluation.success_rate == 1.0 and evaluation.all_budget_feasible
    assert np.isfinite(evaluation.mean_loss) and 0.0 <= evaluation.mean_loss <= 1.0


def test_manual_override_cannot_be_theorem_certified(monkeypatch):
    monkeypatch.setattr(
        objective,
        "full_cebp_tomography",
        lambda *_args, **_kwargs: _fake_result(certified=True),
    )
    with pytest.raises(RuntimeError, match="theorem-certified"):
        objective.evaluate_candidate_on_seed(
            build_smoke_instance(), _derived(), 9, BUDGET, _config()
        )


@pytest.mark.parametrize(
    ("module_name", "budget", "expected"),
    (
        ("Demo.run_cebp_demo_5qubit", 447_533, (5_000, 30_000, 100, 0.10, 0.12, 0.80)),
        ("Demo.run_small_cebp_demo_6qubit", 5_935_844, (8_000, 50_000, 100, 0.06, 0.05, 1.20)),
    ),
)
def test_manual_demo_configs_remain_compatible_but_fixed_error_is_practical(
    module_name, budget, expected
):
    import importlib

    demo = importlib.import_module(module_name)
    candidate = candidate_from_manual_configs(
        n=demo.N,
        d=demo.D,
        total_copies=budget,
        peeling_config=demo.PEELING_CONFIG,
        recovery_config=demo.RECOVERY_CONFIG,
        grouping_config=demo.GROUPING_CONFIG,
        syndrome_config=demo.SYNDROME_CONFIG,
        tomography_config=demo.TOMOGRAPHY_CONFIG,
    )
    config = OptimizationConfig(
        total_copies=budget,
        objective=OptimizationObjective(
            mode=OptimizationMode.FIXED_ERROR_MIN_COPIES,
            copy_ceiling=budget,
            error_target=0.5,
        ),
    )
    derived = derive_candidate(
        candidate,
        n=demo.N,
        d=demo.D,
        total_copies=budget,
        optimization_config=config,
    )
    _M1, _M2, _M_sgn, eta_test, _tau_kappa, _epsilon_tom = expected
    assert derived.M_sgn is None
    assert derived.h_min == pytest.approx(demo.PEELING_CONFIG.h_min)
    assert derived.h_max == pytest.approx(demo.PEELING_CONFIG.h_max)
    assert derived.theta_tau_multiplier is not None
    assert derived.eta_test == pytest.approx(eta_test)
    assert derived.tau_kappa == pytest.approx(config.fixed_budget_tau_kappa_diagnostic)
    assert derived.epsilon_tom == pytest.approx(
        config.fixed_budget_epsilon_tom_diagnostic
    )
    assert derived.execution_policy == main_v2.ExecutionPolicy.FIXED_BUDGET_GRACEFUL.value


def test_initial_candidates_are_deduplicated_with_stable_ids():
    config = _config(number_of_candidates=3)
    records, _rejected = _sample_valid_candidates(
        instance=build_smoke_instance(),
        total_copies=BUDGET,
        optimization_config=config,
        search_space=smoke_search_space(),
        search_seed=17,
        initial_candidates=(BASE_PARAMETERS, BASE_PARAMETERS),
    )
    assert tuple(record.candidate_id for record in records) == (
        "initial-0000", "candidate-0000", "candidate-0001"
    )
    assert isinstance(records[0].parameters, FixedBudgetCandidateParameters)
    assert records[0].parameters.h_min == BASE_PARAMETERS.h_min
    assert records[0].parameters.h_max == BASE_PARAMETERS.h_max
