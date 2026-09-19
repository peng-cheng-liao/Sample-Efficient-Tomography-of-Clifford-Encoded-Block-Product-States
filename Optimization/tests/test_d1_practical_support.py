from dataclasses import asdict, replace
import math

import pytest

import main_v2
from Optimization.objective import (
    SeedEvaluation,
    aggregate_candidate_evaluations,
    evaluate_candidate_on_seed,
)
from Optimization.parameterization import (
    FixedBudgetCandidateParameters,
    FixedErrorCandidateParameters,
    OptimizationConfig,
    PreflightResourceCheck,
    candidate_to_end_to_end_config,
    derive_candidate,
    preflight_resource_check,
)
from Optimization.progressive_search import (
    ProgressiveSearchConfig,
    _record_with_budget,
    _validate_progressive_call,
    is_sufficient_relative_improvement,
)
from Optimization.search import CandidateRecord
from Optimization.specification import OptimizationMode, OptimizationObjective


BUDGET = 30_000
TUNING = tuple(range(101, 117))
HOLDOUT = (901, 902)


def _objective(target=0.05):
    return OptimizationObjective(
        mode=OptimizationMode.FIXED_ERROR_MIN_COPIES,
        copy_ceiling=BUDGET,
        error_target=target,
    )


def _config(**changes):
    values = dict(
        total_copies=BUDGET,
        tuning_seeds=TUNING,
        holdout_seeds=HOLDOUT,
        max_dense_qubits=3,
        max_enumeration_qubits=3,
        max_oracle_dense_qubits=3,
        objective=_objective(),
    )
    values.update(changes)
    return OptimizationConfig(**values)


def _fixed_error():
    return FixedErrorCandidateParameters(
        h_min=0.70,
        h_max=0.90,
        theta_tau_multiplier=8.0,
        eta_test=0.10,
        peel_weight=1.0,
        recovery_weight=1.0,
        grouping_weight=4.0,
        syndrome_weight=0.25,
        tomography_weight=2.0,
    )


def _derived(N_candidate=BUDGET):
    return derive_candidate(
        _fixed_error(),
        n=2,
        d=1,
        total_copies=BUDGET,
        optimization_config=_config(),
        physical_budget=N_candidate,
    )


def _instance(seed=3):
    return main_v2.random_cebp_state(
        2,
        1,
        block_sizes=(1, 1),
        pure=False,
        seed=seed,
        oracle_backend="structured",
        max_dense_debug_qubits=2,
    )


@pytest.mark.parametrize("mode", [
    OptimizationMode.FIXED_BUDGET_MIN_ERROR,
    OptimizationMode.FIXED_ERROR_MIN_COPIES,
])
def test_d1_four_stage_allocation_is_deterministic_and_bounded(mode):
    cfg = _config(
        objective=OptimizationObjective(
            mode=mode,
            copy_ceiling=BUDGET,
            error_target=(0.05 if mode is OptimizationMode.FIXED_ERROR_MIN_COPIES else None),
        )
    )
    parameters = (
        _fixed_error()
        if mode is OptimizationMode.FIXED_ERROR_MIN_COPIES
        else FixedBudgetCandidateParameters(
            **{k: v for k, v in asdict(_fixed_error()).items() if k != "N_candidate"}
        )
    )
    first = derive_candidate(
        parameters, n=2, d=1, total_copies=BUDGET, optimization_config=cfg
    )
    second = derive_candidate(
        parameters, n=2, d=1, total_copies=BUDGET, optimization_config=cfg
    )
    caps = dict(first.fixed_budget_stage_caps)
    assert first.fixed_budget_stage_caps == second.fixed_budget_stage_caps
    assert caps["grouping"] == 0
    assert sum(caps.values()) == BUDGET
    assert sum(caps[name] for name in ("peeling", "recovery", "syndrome", "tomography")) <= BUDGET


def test_d1_explicit_caps_reach_end_to_end_config_unchanged():
    derived = _derived()
    learner = candidate_to_end_to_end_config(
        derived,
        d=1,
        learner_seed=11,
        total_copies=BUDGET,
        optimization_config=_config(),
    )
    assert learner.fixed_budget_stage_caps == derived.fixed_budget_stage_caps
    assert dict(learner.fixed_budget_stage_caps)["grouping"] == 0
    assert main_v2.fixed_budget_resolved_stage_caps(learner) == derived.fixed_budget_stage_caps


def test_d1_preflight_is_four_stage_and_rejects_inadequate_tomography(monkeypatch):
    derived = _derived()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("d>=2 grouping/tomography estimator was invoked")

    monkeypatch.setattr(
        "Optimization.parameterization.simplified_cumulant_test_bound", forbidden
    )
    monkeypatch.setattr(
        "Optimization.parameterization.register_tomography_budget", forbidden
    )
    valid = preflight_resource_check(
        derived, n=2, d=1, total_copies=BUDGET, optimization_config=_config()
    )
    assert valid.safe_to_execute
    assert valid.grouping_safety_estimate == 0
    caps = dict(derived.fixed_budget_stage_caps)
    inadequate_caps = tuple(
        (name, 2 if name == "tomography" else value + (caps["tomography"] - 2 if name == "syndrome" else 0))
        for name, value in derived.fixed_budget_stage_caps
    )
    inadequate = preflight_resource_check(
        replace(derived, fixed_budget_stage_caps=inadequate_caps),
        n=2,
        d=1,
        total_copies=BUDGET,
        optimization_config=_config(),
    )
    assert not inadequate.safe_to_execute
    assert inadequate.mathematical_budget_rejected
    assert inadequate.reason == "d1_fixed_budget_stage_allocation_below_primitive_minimum"


def test_tiny_d1_budget_native_end_to_end_smoke():
    derived = _derived()
    evaluation = evaluate_candidate_on_seed(
        _instance(),
        derived,
        11,
        BUDGET,
        _config(),
        objective=_objective(),
    )
    assert evaluation.operational_success
    assert evaluation.estimator_available
    assert evaluation.execution_branch == "d1_specialized"
    assert evaluation.trace_distance is not None and math.isfinite(evaluation.trace_distance)
    ledger = dict(evaluation.realized_copy_ledger)
    stage_records = {record[0]: record for record in evaluation.fixed_budget_stage_records}
    assert ledger["grouping_ordinary_pool"] == 0
    assert stage_records["grouping"][1:4] == (0, 0, 0)
    assert ledger["conditional_one_qubit_pool"] <= stage_records["tomography"][1]
    assert ledger["conditional_one_qubit_pool"] % 3 == 0
    assert evaluation.tomography_fixed_budget == ledger["conditional_one_qubit_pool"]
    assert evaluation.realized_total <= BUDGET
    assert evaluation.failure_reason != "insufficient_postselected_shots"


def _check():
    return PreflightResourceCheck(
        mandatory_bell_copies=4,
        worst_case_sign_reservation=0,
        preflight_fixed_reservation=4,
        grouping_safety_estimate=0,
        tomography_safety_estimate=3,
        preflight_safety_estimate=10,
        rigorous_grouping_upper_bound=0,
        rigorous_tomography_upper_bound=3,
        rigorous_total_upper_bound=10,
        single_grouping_query_shots_estimate=0,
        safety_limit=10,
        runtime_safety_rejected=False,
        mathematical_budget_rejected=False,
        safe_to_execute=True,
        reason="mock-safe",
        guarantee_level="mock",
    )


def _seed(seed, error, *, operational=True):
    return SeedEvaluation(
        learner_seed=seed,
        preflight_rejected=not operational,
        preflight_resource_check=_check(),
        operational_success=operational,
        budget_feasible=operational,
        post_run_realized_budget_feasible=operational,
        loss=error if operational else 1.0,
        loss_computed=operational,
        trace_distance=error if operational else None,
        failure_stage=None if operational else "mock",
        failure_reason=None if operational else "mock failure",
        realized_copy_ledger=(("conditional_one_qubit_pool", 3),),
        realized_total=3,
        copy_utilization=3 / BUDGET,
        recovered_t=0 if operational else None,
        cluster_sizes=(),
        register_sizes=(),
        j_aux_size=2 if operational else None,
        error_feasible=(operational and error <= 0.05),
        error_excess=(max(0.0, error - 0.05) if operational else 1.0),
        estimator_available=operational,
        execution_branch="d1_specialized" if operational else None,
    )


def test_d1_fixed_error_aggregation_uses_mean_and_strict_operations():
    mean_feasible = aggregate_candidate_evaluations(
        "d1",
        (_seed(1, 0.04), _seed(2, 0.06)),
        objective=_objective(),
        N_candidate=BUDGET,
    )
    assert mean_feasible.mean_error_feasible
    assert mean_feasible.max_trace_distance_successful > 0.05
    operational_failure = aggregate_candidate_evaluations(
        "d1-failed",
        (_seed(1, 0.04), _seed(2, 0.04, operational=False)),
        objective=_objective(),
        N_candidate=BUDGET,
    )
    assert not operational_failure.operationally_valid
    assert not operational_failure.mean_error_feasible


def test_d1_budget_refinement_freezes_structure_and_preserves_zero_grouping():
    instance = _instance()
    structural = CandidateRecord("candidate-0001", _fixed_error(), _derived())
    refined = _record_with_budget(
        structural,
        25_000,
        instance=instance,
        total_copies=BUDGET,
        optimization_config=_config(),
    )
    before = asdict(structural.parameters)
    after = asdict(refined.parameters)
    assert after == before
    assert refined.derived.physical_copy_budget == 25_000
    assert dict(refined.derived.fixed_budget_stage_caps)["grouping"] == 0
    catalog = {record.candidate_id: record for record in (structural, refined)}
    assert catalog[refined.candidate_id] is refined
    assert sum(dict(refined.derived.fixed_budget_stage_caps).values()) == 25_000


def test_d1_progressive_policy_validates_through_256_with_zero_threshold():
    progressive = ProgressiveSearchConfig(
        initial_candidates=16,
        max_candidates=256,
        candidate_growth_factor=2,
        seed_fidelities=(1, 2, 4, 8, 16),
        comparison_seed_count=16,
        relative_improvement_threshold=0.0,
    )
    objective = _validate_progressive_call(
        _instance(), BUDGET, _config(), progressive, ()
    )
    assert objective.mode is OptimizationMode.FIXED_ERROR_MIN_COPIES
    pools = []
    value = progressive.initial_candidates
    while value <= progressive.max_candidates:
        pools.append(value)
        value *= progressive.candidate_growth_factor
    assert pools == [16, 32, 64, 128, 256]
    assert progressive.seed_fidelities == (1, 2, 4, 8, 16)
    assert progressive.relative_improvement_threshold == 0.0
    assert is_sufficient_relative_improvement(0.0, 0.0)
