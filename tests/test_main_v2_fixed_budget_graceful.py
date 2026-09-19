from dataclasses import replace

import numpy as np
import pytest
import qutip as qt

import main_v2
from Optimization import CandidateParameters, OptimizationConfig, derive_candidate
from Optimization.parameterization import candidate_to_end_to_end_config
from Optimization.run_parameter_optimization import build_smoke_instance
from tests.test_main_v2_grouping import _manuscript_fixture
from tests.test_main_v2_tomography import _localization, _peeling, _source, _tomography_config


BASE = CandidateParameters(
    0.0334, 0.20, 0.001, 1.25, 1.30, 0.80, 0.10, 0.12, 0.10, 0.60, 0.80
)


def _base_graceful_config(*, seed=11, weights=None, budget=300_000):
    optimization = OptimizationConfig(
        total_copies=300_000,
        tuning_seeds=(11,),
        holdout_seeds=(21,),
        halving_seed_counts=(1,),
        number_of_candidates=1,
        simulation_backend="batched_counts",
    )
    derived = derive_candidate(
        BASE, n=3, d=2, total_copies=300_000, optimization_config=optimization
    )
    strict = candidate_to_end_to_end_config(
        derived,
        d=2,
        learner_seed=seed,
        total_copies=300_000,
        optimization_config=optimization,
    )
    return replace(
        strict,
        materialize_dense_estimator=True,
        max_reserved_copies=budget,
        max_realized_copies=budget,
        execution_policy=main_v2.ExecutionPolicy.FIXED_BUDGET_GRACEFUL,
        fixed_budget_stage_weights=weights or main_v2.FixedBudgetStageWeights(),
    )


def _assert_physical(state):
    array = np.asarray(state.full(), dtype=complex)
    assert np.all(np.isfinite(array))
    assert np.linalg.norm(array - array.conj().T) <= 1e-10
    assert abs(np.trace(array) - 1.0) <= 1e-10
    assert np.min(np.linalg.eigvalsh(array)) >= -1e-10


def _assert_complete_stage_audit(result, config):
    records = result.fixed_budget_stage_records
    caps = dict(
        main_v2.fixed_budget_nominal_stage_caps(
            config.max_realized_copies, config.fixed_budget_stage_weights
        )
    )
    assert tuple(record.stage for record in records) == (
        "peeling", "recovery", "grouping", "syndrome", "tomography"
    )
    assert tuple(record.assigned_cap for record in records) == tuple(caps.values())
    assert sum(record.assigned_cap for record in records) == config.max_realized_copies
    assert sum(record.realized_copies for record in records) == result.realized_total
    assert all(
        record.unused_copies == record.assigned_cap - record.realized_copies
        for record in records
    )


def test_default_policy_is_strict_and_explicit_strict_copy_cap_still_fails():
    assert main_v2.EndToEndConfig(0.5, 0.2).execution_policy is main_v2.ExecutionPolicy.STRICT
    instance = build_smoke_instance()
    strict = replace(_base_graceful_config(), execution_policy="strict")
    capped = replace(strict, max_realized_copies=1)
    result = main_v2.full_cebp_tomography(instance.learner_view(), config=capped)
    assert not result.success and result.failure_stage == "copy_budget"


@pytest.mark.parametrize("budget", (1, 10, 100, 1_000, 300_000))
def test_tiny_and_larger_budgets_always_return_physical_finite_estimator(budget):
    instance = build_smoke_instance()
    result = main_v2.full_cebp_tomography(
        instance.learner_view(), config=_base_graceful_config(budget=budget)
    )
    assert result.success and result.estimator_available
    assert result.failure_stage != "copy_budget"
    assert result.realized_total == result.realized_copy_ledger.total <= budget
    assert result.decoded_density is not None
    _assert_physical(result.decoded_density)
    assert np.isfinite(0.5 * main_v2.debug_end_to_end_trace_error(result, instance))
    assert not result.theorem_certified
    _assert_complete_stage_audit(result, _base_graceful_config(budget=budget))


def test_peeling_fallback_has_all_five_unborrowed_stage_records():
    instance = build_smoke_instance()
    config = _base_graceful_config(budget=1)
    result = main_v2.full_cebp_tomography(instance.learner_view(), config=config)
    _assert_complete_stage_audit(result, config)
    assert result.fixed_budget_stage_records[0].degradation_reason == (
        "insufficient_for_one_bell_round"
    )
    assert all(
        not record.stage_complete
        and not record.budget_exhausted
        and record.degradation_reason.startswith("not_reached_due_to_upstream_failure:")
        for record in result.fixed_budget_stage_records[1:]
    )


def test_recovery_fallback_has_all_five_stage_records(monkeypatch):
    instance = build_smoke_instance()
    config = _base_graceful_config()

    def fail_recovery(*_args, **_kwargs):
        raise RuntimeError("forced recovery failure")

    monkeypatch.setattr(main_v2, "empirical_rank_guided_sector_recovery", fail_recovery)
    result = main_v2.full_cebp_tomography(instance.learner_view(), config=config)
    _assert_complete_stage_audit(result, config)
    records = {record.stage: record for record in result.fixed_budget_stage_records}
    assert not records["recovery"].stage_complete
    assert all(
        records[stage].degradation_reason
        == "not_reached_due_to_upstream_failure:recovery"
        for stage in ("grouping", "syndrome", "tomography")
    )


def test_grouping_failure_has_all_five_stage_records(monkeypatch):
    instance = build_smoke_instance()
    config = _base_graceful_config()

    def fail_grouping(_source, peeling, recovery, grouping_config, **_kwargs):
        return main_v2._grouping_failure(
            recovery,
            peeling,
            grouping_config,
            int(grouping_config.ell_grp),
            "forced_grouping_failure",
            main_v2.DataProvenance.EMPIRICAL,
        )

    monkeypatch.setattr(main_v2, "empirical_hierarchical_cumulant_grouping", fail_grouping)
    result = main_v2.full_cebp_tomography(instance.learner_view(), config=config)
    _assert_complete_stage_audit(result, config)
    records = {record.stage: record for record in result.fixed_budget_stage_records}
    assert not records["grouping"].stage_complete
    assert records["grouping"].degradation_reason == "forced_grouping_failure"
    assert all(
        records[stage].degradation_reason
        == "not_reached_due_to_upstream_failure:grouping"
        for stage in ("syndrome", "tomography")
    )


def test_localization_failure_has_all_five_stage_records(monkeypatch):
    instance = build_smoke_instance()
    config = _base_graceful_config()

    def fail_localization(recovery, grouping, *, d, config=None):
        return main_v2._localization_failure(
            recovery, grouping, d, "forced_localization_failure"
        )

    monkeypatch.setattr(main_v2, "localize_grouped_recovery", fail_localization)
    result = main_v2.full_cebp_tomography(instance.learner_view(), config=config)
    _assert_complete_stage_audit(result, config)
    records = {record.stage: record for record in result.fixed_budget_stage_records}
    assert records["grouping"].stage_complete
    assert all(
        records[stage].degradation_reason
        == "not_reached_due_to_upstream_failure:grouping"
        for stage in ("syndrome", "tomography")
    )


@pytest.mark.parametrize(
    ("k", "rounds", "expected_measured"),
    ((2, 0, 0), (2, 5, 5), (2, 15, 15), (3, 10, 10), (3, 63, 63)),
)
def test_fixed_budget_partial_pauli_tomography_physical_and_bounded(
    k, rounds, expected_measured
):
    state = qt.basis(2**k, 0)
    peeling = _peeling(k)
    localization = _localization(k, 0, (((0,), tuple(range(k))),), d=k)
    result = main_v2.tomograph_localized_registers_fixed_budget(
        _source(state, k),
        peeling,
        localization,
        rounds,
        _tomography_config(4.0 * k, 0.1),
        seed=83,
    )
    assert result.success and result.N_bp <= rounds
    assert result.measured_pauli_counts == (((0,), expected_measured),)
    assert result.total_pauli_counts == (((0,), 4**k - 1),)
    assert result.budget_truncated == (expected_measured < 4**k - 1)
    assert result.budgets[0].budget_truncated == result.budget_truncated
    _assert_physical(result.estimates[0].nu_hat)
    if rounds == 0:
        assert np.allclose(result.estimates[0].nu_hat.full(), np.eye(2**k) / 2**k)
    measured = {pauli for pauli, _shots, _sum in result.records[0].sufficient_statistics}
    coefficients = dict(result.estimates[0].pauli_coefficients)
    assert all(coefficients[pauli] == 0.0 for pauli in set(coefficients) - measured)


def test_disjoint_registers_share_one_physical_round_pool():
    state = qt.tensor(qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 0))
    peeling = _peeling(3)
    localization = _localization(
        3, 0, (((0,), (0,)), ((1,), (1, 2))), d=2
    )
    result = main_v2.tomograph_localized_registers_fixed_budget(
        _source(state, 3), peeling, localization, 37, _tomography_config(12.0), seed=91
    )
    assert result.success and result.N_bp == result.block_tomography_pool == 37
    assert [sum(count for _pauli, count in record.schedule_runs) for record in result.records] == [37, 37]
    assert result.sum_local_schedule_lengths == 74
    _assert_physical(result.localized_empirical_estimator)


def test_mismatched_joint_schedules_accumulate_split_pauli_statistics(monkeypatch):
    state = qt.tensor(*(qt.basis(2, 0) for _ in range(4)))
    peeling = _peeling(4)
    localization = _localization(
        4, 0, (((0,), (0,)), ((1,), (1, 2, 3))), d=3
    )
    contributions = []
    call_index = 0

    def segment_signed_sum(counts, _outcomes, position):
        nonlocal call_index
        length = int(np.sum(counts))
        segment = call_index // 2
        call_index += 1
        value = length if segment % 2 == 0 else -length
        if position == 0:
            contributions.append((length, value))
        return value

    monkeypatch.setattr(main_v2, "_signed_sum_from_joint_counts", segment_signed_sum)
    rounds = 130
    result = main_v2.tomograph_localized_registers_fixed_budget(
        _source(state, 4),
        peeling,
        localization,
        rounds,
        _tomography_config(16.0),
        seed=97,
    )
    assert result.success
    one_qubit = result.records[0]
    expected_schedule = dict(one_qubit.schedule_runs)
    recorded = {
        pauli: (shots, signed_sum)
        for pauli, shots, signed_sum in one_qubit.sufficient_statistics
    }
    assert all(recorded[pauli][0] == shots for pauli, shots in expected_schedule.items())
    first_pauli = one_qubit.schedule_runs[0][0]
    first_run_shots = one_qubit.schedule_runs[0][1]
    used = []
    accumulated = 0
    for length, signed_sum in contributions:
        if accumulated >= first_run_shots:
            break
        used.append((length, signed_sum))
        accumulated += length
    assert len(used) > 1 and accumulated == first_run_shots
    expected_sum = sum(value for _length, value in used)
    assert recorded[first_pauli] == (first_run_shots, expected_sum)
    coefficient = dict(result.estimates[0].pauli_coefficients)[first_pauli]
    assert coefficient == pytest.approx(expected_sum / first_run_shots)


def test_stage_caps_are_immutable_and_no_unused_copies_carry_downstream():
    instance = build_smoke_instance()
    config = _base_graceful_config()
    result = main_v2.full_cebp_tomography(instance.learner_view(), config=config)
    records = {record.stage: record for record in result.fixed_budget_stage_records}
    caps = dict(main_v2.fixed_budget_nominal_stage_caps(
        config.max_realized_copies, config.fixed_budget_stage_weights
    ))
    assert result.peeling.t > 0
    expected_M_sgn = caps["syndrome"] // result.peeling.t
    assert result.syndrome.M_sgn == expected_M_sgn
    assert result.syndrome.syndrome_sign_pool == result.peeling.t * expected_M_sgn
    assert records["syndrome"].realized_copies == result.syndrome.syndrome_sign_pool
    assert all(record.assigned_cap == caps[record.stage] for record in records.values())
    assert all(
        record.unused_copies == record.assigned_cap - record.realized_copies
        for record in records.values()
    )
    assert result.tomography.block_tomography_pool == caps["tomography"]
    assert records["tomography"].assigned_cap == caps["tomography"]
    assert result.realized_total <= config.max_realized_copies
    _assert_physical(result.decoded_density)
    assert np.isfinite(0.5 * main_v2.debug_end_to_end_trace_error(result, instance))


def test_positive_undersampled_syndrome_measures_every_sign_without_overspend():
    weights = main_v2.FixedBudgetStageWeights(
        peeling=60_000,
        recovery=60_000,
        grouping=37_801,
        syndrome=100,
        tomography=142_099,
    )
    instance = build_smoke_instance()
    config = _base_graceful_config(weights=weights)
    config = replace(
        config,
        syndrome_override=replace(config.syndrome_override, return_details=True),
    )
    result = main_v2.full_cebp_tomography(instance.learner_view(), config=config)
    syndrome_record = {
        record.stage: record for record in result.fixed_budget_stage_records
    }["syndrome"]
    assert result.peeling.t == 1
    assert result.syndrome.M_sgn == 100
    assert result.syndrome.syndrome_sign_pool == 100
    assert len(result.syndrome.records) == result.peeling.t
    assert all(record.shots == 100 for record in result.syndrome.records)
    assert syndrome_record.realized_copies == 100
    assert syndrome_record.stage_complete
    assert result.realized_total <= 300_000 and not result.theorem_certified
    _assert_physical(result.decoded_density)


def test_fixed_budget_metadata_is_actual_and_complete_coverage_is_not_truncated():
    k, rounds = 3, 100
    peeling = _peeling(k)
    localization = _localization(k, 0, (((7,), tuple(range(k))),), d=k)
    result = main_v2.tomograph_localized_registers_fixed_budget(
        _source(qt.basis(2**k, 0), k),
        peeling,
        localization,
        rounds,
        _tomography_config(1e-3, 1e-6),
        seed=103,
    )
    budget = result.budgets[0]
    record = result.records[0]
    shots = [count for _pauli, count in record.schedule_runs]
    assert isinstance(budget, main_v2.FixedBudgetRegisterTomographyBudget)
    assert not isinstance(budget, main_v2.RegisterTomographyBudget)
    assert budget.total_nonidentity_paulis == 63
    assert budget.measured_pauli_count == len(record.sufficient_statistics) == 63
    assert budget.physical_round_budget == rounds
    assert budget.realized_schedule_length == sum(shots) == rounds
    assert budget.min_shots_per_measured_pauli == min(shots) == 1
    assert budget.max_shots_per_measured_pauli == max(shots) == 2
    assert budget.complete_pauli_coverage
    assert not budget.budget_truncated and not result.budget_truncated
    assert not hasattr(budget, "epsilon_C") and not hasattr(budget, "tau_C_tom")


def test_strict_tomography_metadata_remains_accuracy_targeted():
    peeling = _peeling(2)
    localization = _localization(2, 0, (((0,), (0, 1)),), d=2)
    result = main_v2.tomograph_localized_registers(
        _source(qt.basis(4, 0), 2),
        peeling,
        localization,
        _tomography_config(4.0, 0.1),
        seed=104,
        simulation_backend="batched_counts",
    )
    budget = result.budgets[0]
    assert isinstance(budget, main_v2.RegisterTomographyBudget)
    assert budget.tau_C_tom == pytest.approx(
        budget.epsilon_C / (4.0 * np.sqrt(budget.N_C_Pauli))
    )


def test_grouping_stops_before_unfittable_complete_query_and_keeps_partition():
    instance, peeling, recovery = _manuscript_fixture()
    config = main_v2.GroupingConfig(
        ell_grp=3,
        eta_test=0.2,
        tau_kappa=0.8,
        delta_grp_ordinary=0.2,
        allow_uncalibrated_recovery=True,
        allow_no_false_merge_margin_failure=True,
    )
    ntest = main_v2.adaptive_cumulant_test_bound(len(recovery.sectors), 3)
    shots = main_v2.ordinary_cumulant_sample_count(2, 0.8, 0.2 / ntest)
    cap = 2 * shots
    result = main_v2.empirical_hierarchical_cumulant_grouping(
        instance.learner_view(),
        peeling,
        recovery,
        config,
        seed=92,
        simulation_backend="batched_counts",
        max_realized_copies=cap,
        graceful_budget=True,
    )
    assert result.success and not result.grouping_complete
    assert result.grouping_budget_truncated
    assert result.realized_grouping_copies == cap
    assert tuple(sorted(value for cluster in result.clusters for value in cluster)) == (0, 1, 2)
    assert not result.theorem_grouping_preconditions_hold


def test_zero_sign_allocation_never_invents_syndrome_and_uses_safe_fallback():
    weights = main_v2.FixedBudgetStageWeights(
        peeling=100_000.25,
        recovery=100_000.25,
        grouping=0.25,
        syndrome=0.25,
        tomography=99_999.0,
    )
    instance = build_smoke_instance()
    result = main_v2.full_cebp_tomography(
        instance.learner_view(), config=_base_graceful_config(weights=weights)
    )
    assert result.estimator_available and result.success
    assert result.syndrome is None
    assert "unknown_syndrome_signs" in result.degradation_reason
    assert "syndrome" in result.truncated_stages and not result.theorem_certified
    assert result.realized_total <= 300_000
    _assert_physical(result.decoded_density)
