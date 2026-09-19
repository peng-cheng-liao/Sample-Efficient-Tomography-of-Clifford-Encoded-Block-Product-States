from dataclasses import replace

import numpy as np
import pytest
import qutip as qt

import main_v2
from Optimization import CandidateParameters, OptimizationConfig, derive_candidate
from Optimization import objective as optimization_objective
from Optimization.specification import OptimizationObjective
from tests.test_main_v2_localization import _manual_grouping, _manual_recovery
from tests.test_main_v2_tomography import _peeling, _source, _tomography_config


PAIR_CAPS_A = (
    ("peeling", 1_304_353),
    ("recovery", 2_704_407),
    ("grouping", 445_114),
    ("syndrome", 53_526),
    ("tomography", 492_600),
)
PAIR_CAPS_B = tuple(
    (stage, 2_000_000 if stage == "grouping" else copies)
    for stage, copies in PAIR_CAPS_A
)


def _independent_triple_sectors(n):
    return tuple(
        main_v2.RecoveredSector(
            index,
            "I" * index + "X" + "I" * (n - index - 1),
            "I" * index + "Z" + "I" * (n - index - 1),
            "I" * index + "Y" + "I" * (n - index - 1),
        )
        for index in range(n)
    )


def _oversize_433_fixture():
    recovery = _manual_recovery(
        _independent_triple_sectors(10), m=10, theorem=False
    )
    grouping = _manual_grouping(
        recovery,
        ((0, 1, 2, 3), (4, 5, 6), (7, 8, 9)),
        ell=3,
        theorem=False,
    )
    return recovery, grouping


def _fixed_localization(recovery, grouping, *, d=3, materialize_dense=False):
    return main_v2.localize_grouped_recovery(
        recovery,
        grouping,
        d=d,
        config=main_v2.LocalizationConfig(
            allow_uncertified_grouping=True,
            verify_dense_unitary=False,
            max_dense_qubits=max(4, recovery.n),
            materialize_dense_clifford=materialize_dense,
            enforce_model_block_bound=False,
        ),
    )


def _explicit_cap_config(caps):
    return main_v2.EndToEndConfig(
        0.5,
        0.2,
        seed=9443020599540107709,
        max_realized_copies=sum(value for _stage, value in caps),
        execution_policy=main_v2.ExecutionPolicy.FIXED_BUDGET_GRACEFUL,
        fixed_budget_stage_caps=tuple(caps),
    )


def _mocked_oversize_pipeline(monkeypatch):
    instance = main_v2.random_cebp_state(
        4,
        3,
        block_sizes=(3, 1),
        pure=True,
        clifford_steps=0,
        seed=901,
        max_dense_debug_qubits=4,
    )
    peeling = _peeling(4, copies=True, theorem=False)
    recovery = _manual_recovery(
        _independent_triple_sectors(4), m=4, theorem=False, copies=True
    )
    grouping = _manual_grouping(
        recovery, ((0, 1, 2, 3),), ell=3, theorem=False
    )

    monkeypatch.setattr(
        main_v2,
        "empirical_certified_stabilizer_peeling",
        lambda *_args, **_kwargs: peeling,
    )
    monkeypatch.setattr(
        main_v2,
        "empirical_rank_guided_sector_recovery",
        lambda *_args, **_kwargs: recovery,
    )
    monkeypatch.setattr(
        main_v2,
        "empirical_hierarchical_cumulant_grouping",
        lambda *_args, **_kwargs: grouping,
    )

    caps = (
        ("peeling", 20),
        ("recovery", 40),
        ("grouping", 0),
        ("syndrome", 0),
        ("tomography", 300),
    )
    config = main_v2.EndToEndConfig(
        0.5,
        0.2,
        seed=902,
        materialize_dense_estimator=False,
        max_dense_qubits=4,
        max_dense_debug_qubits=4,
        max_reserved_copies=360,
        max_realized_copies=360,
        simulation_backend="batched_counts",
        execution_policy=main_v2.ExecutionPolicy.FIXED_BUDGET_GRACEFUL,
        fixed_budget_stage_caps=caps,
        allow_uncertified_execution=True,
        peeling_override=main_v2.PeelingConfig(
            0.6, 0.8, 0.04, M1=10, materialize_dense_clifford=False
        ),
        recovery_override=main_v2.RecoveryConfig(
            0.1, M2=20, allow_uncalibrated_peeling=True,
            allow_margin_failure=True,
        ),
        grouping_override=main_v2.GroupingConfig(
            ell_grp=3,
            eta_test=0.15,
            tau_kappa=0.05,
            delta_grp_ordinary=0.1,
            allow_uncalibrated_recovery=True,
            allow_no_false_merge_margin_failure=True,
            sampling_policy=main_v2.GroupingSamplingPolicy.FIXED_BUDGET.value,
        ),
        syndrome_override=main_v2.SyndromeConfig(0.05, 0.6, M_sgn=0),
        tomography_override=main_v2.TomographyConfig(
            1.0,
            0.1,
            allow_uncertified_localization=True,
            max_dense_qubits=4,
            materialize_localized_estimator=False,
        ),
    )
    result = main_v2.full_cebp_tomography(
        instance.learner_view(), config=config
    )
    return instance, config, result


def test_fixed_budget_localization_accepts_algebraically_valid_433_with_d3():
    recovery, grouping = _oversize_433_fixture()
    localization = _fixed_localization(recovery, grouping)

    assert localization.success
    assert tuple(len(cluster) for cluster in localization.clusters) == (4, 3, 3)
    assert tuple(len(register) for _cluster, register in localization.J_C) == (4, 3, 3)
    assert localization.J_aux == ()
    assert localization.empirical_max_cluster_size == 4
    assert localization.assumed_d == 3
    assert localization.model_bound_violated
    assert localization.oversize_clusters == (((0, 1, 2, 3), 4),)
    assert localization.reconstruction_proceeded_despite_model_bound_violation
    assert not localization.theorem_localization_preconditions_hold


def test_strict_localization_still_rejects_433_with_d3():
    recovery, grouping = _oversize_433_fixture()
    strict = main_v2.localize_grouped_recovery(recovery, grouping, d=3)

    assert not strict.success
    assert "exceeds ell_grp" in strict.failure_reason
    assert strict.model_bound_violated
    assert not strict.reconstruction_proceeded_despite_model_bound_violation


def test_four_qubit_fixed_budget_tomography_uses_255_settings_and_16x16_state():
    recovery = _manual_recovery(
        _independent_triple_sectors(4), m=4, theorem=False
    )
    grouping = _manual_grouping(
        recovery, ((0, 1, 2, 3),), ell=3, theorem=False
    )
    localization = _fixed_localization(
        recovery, grouping, materialize_dense=True
    )
    rounds = 600
    result = main_v2.tomograph_localized_registers_fixed_budget(
        _source(qt.basis(16, 0), 4),
        _peeling(4, copies=False, theorem=False),
        localization,
        rounds,
        _tomography_config(
            16.0,
            0.1,
            max_dense_qubits=4,
            materialize_localized_estimator=True,
        ),
        seed=903,
        prior_copy_ledger=localization.cumulative_copy_ledger,
    )

    budget = result.budgets[0]
    record = result.records[0]
    assert result.success and result.block_tomography_pool == rounds
    assert budget.k_C == 4 and budget.total_nonidentity_paulis == 255
    assert budget.measured_pauli_count == len(record.sufficient_statistics) == 255
    assert budget.realized_schedule_length == rounds
    assert sum(count for _pauli, count in record.schedule_runs) == rounds
    assert result.copy_ledger.total == rounds
    assert result.estimates[0].nu_hat.shape == (16, 16)
    assert result.estimates[0].nu_hat.dims == [[2, 2, 2, 2], [2, 2, 2, 2]]


def test_compact_estimator_stores_433_empirical_blocks_without_d_cap():
    identity = main_v2.SignedClifford.identity(10)
    compact = main_v2.CompactCEBPEstimator(
        n=10,
        t=0,
        m=10,
        U_stab=None,
        bar_U_rec=None,
        syndrome_bits=(),
        register_estimates=(
            ((0, 1, 2, 3), (0, 1, 2, 3), qt.qeye([2] * 4) / 16),
            ((4, 5, 6), (4, 5, 6), qt.qeye([2] * 3) / 8),
            ((7, 8, 9), (7, 8, 9), qt.qeye([2] * 3) / 8),
        ),
        J_aux=(),
        peeling_clifford=identity,
        recovery_clifford=identity,
    )

    assert tuple(len(register) for _cluster, register, _state in compact.register_estimates) == (4, 3, 3)
    assert compact.register_estimates[0][2].shape == (16, 16)


def test_fixed_budget_full_pipeline_reconstructs_oversize_model_not_global_mixed(
    monkeypatch,
):
    instance, config, result = _mocked_oversize_pipeline(monkeypatch)

    assert result.success and result.estimator_available and result.execution_complete
    assert result.model_bound_violated
    assert result.reconstruction_proceeded_despite_model_bound_violation
    assert result.localization.success and result.tomography.success
    assert tuple(len(register) for _cluster, register in result.localization.J_C) == (4,)
    assert result.tomography.total_pauli_counts == (((0, 1, 2, 3), 255),)
    assert result.compact_estimator.register_estimates
    assert result.compact_estimator.J_aux == ()
    assert tuple(record.assigned_cap for record in result.fixed_budget_stage_records) == tuple(
        value for _stage, value in main_v2.fixed_budget_resolved_stage_caps(config)
    )
    assert result.realized_total == 360
    assert np.isfinite(
        0.5 * main_v2.debug_end_to_end_trace_error(result, instance, max_dense_qubits=4)
    )


def test_fixed_budget_objective_scores_actual_oversize_estimator(monkeypatch):
    instance, _config, result = _mocked_oversize_pipeline(monkeypatch)
    monkeypatch.setattr(
        optimization_objective,
        "full_cebp_tomography",
        lambda *_args, **_kwargs: result,
    )
    optimization = OptimizationConfig(
        total_copies=300_000,
        tuning_seeds=(904,),
        holdout_seeds=(905,),
        halving_seed_counts=(1,),
        number_of_candidates=1,
        max_dense_qubits=4,
        max_oracle_dense_qubits=4,
        simulation_backend="batched_counts",
        objective=OptimizationObjective(
            mode="fixed_budget_min_error", copy_ceiling=300_000
        ),
    )
    candidate = CandidateParameters(
        0.0334, 0.20, 0.001, 1.25, 1.30, 0.80,
        0.10, 0.12, 0.10, 0.60, 0.80,
    )
    derived = derive_candidate(
        candidate,
        n=4,
        d=3,
        total_copies=300_000,
        optimization_config=optimization,
    )
    evaluation = optimization_objective._run_candidate_on_seed(
        instance,
        derived,
        904,
        300_000,
        optimization,
        objective=optimization.objective,
        compute_oracle_loss=True,
    )

    expected = 0.5 * main_v2.debug_end_to_end_trace_error(
        result, instance, max_dense_qubits=4
    )
    target = instance.materialize_state_debug(max_qubits=4)
    target_density = target * target.dag() if target.isket else target
    mixed_distance = 0.5 * np.linalg.svd(
        target_density.full() - np.eye(16) / 16,
        compute_uv=False,
    ).sum()
    assert evaluation.operational_success and evaluation.loss_computed
    assert evaluation.register_sizes == (4,)
    assert evaluation.trace_distance == pytest.approx(expected)
    assert evaluation.trace_distance != pytest.approx(mixed_distance)


def test_pair_explicit_caps_change_only_grouping_and_total_without_renormalization():
    config_a = _explicit_cap_config(PAIR_CAPS_A)
    config_b = _explicit_cap_config(PAIR_CAPS_B)
    resolved_a = main_v2.fixed_budget_resolved_stage_caps(config_a)
    resolved_b = main_v2.fixed_budget_resolved_stage_caps(config_b)

    assert resolved_a == PAIR_CAPS_A
    assert resolved_b == PAIR_CAPS_B
    assert config_a.max_realized_copies == 5_000_000
    assert config_b.max_realized_copies == 6_554_886
    assert tuple(
        stage
        for (stage, left), (_same_stage, right) in zip(resolved_a, resolved_b)
        if left != right
    ) == ("grouping",)
    assert dict(resolved_b) == {
        "peeling": 1_304_353,
        "recovery": 2_704_407,
        "grouping": 2_000_000,
        "syndrome": 53_526,
        "tomography": 492_600,
    }


def test_explicit_stage_caps_must_sum_exactly_to_total():
    with pytest.raises(ValueError, match="must sum"):
        replace(
            _explicit_cap_config(PAIR_CAPS_A),
            max_realized_copies=5_000_001,
        )

    malformed = tuple(
        (stage, 445_114.5 if stage == "grouping" else value)
        for stage, value in PAIR_CAPS_A
    )
    with pytest.raises(TypeError, match="must be integers"):
        main_v2.EndToEndConfig(
            0.5,
            0.2,
            max_realized_copies=5_000_000,
            execution_policy=main_v2.ExecutionPolicy.FIXED_BUDGET_GRACEFUL,
            fixed_budget_stage_caps=malformed,
        )
