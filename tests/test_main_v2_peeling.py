import itertools

import numpy as np
import pytest
import qutip as qt

import main
import main_v2


def _score_map(n, overrides=None):
    values = {
        "".join(chars): 0.0
        for chars in itertools.product("IXYZ", repeat=n)
    }
    values["I" * n] = 1.0
    if overrides:
        values.update(overrides)
    return values


def _exact_config(**overrides):
    values = dict(h_min=0.6, h_max=0.8, eta=0.005, return_details=True)
    values.update(overrides)
    return main_v2.PeelingConfig(**values)


def _zero_and_one():
    return qt.basis(2, 0), qt.basis(2, 1)


def test_bell_record_shape_reproducibility_and_shared_queries():
    instance = main_v2.random_cebp_state(
        2, 1, block_states=(qt.basis(2, 0), qt.basis(2, 1)), seed=10
    )
    first = main_v2.sample_bell_scores(
        instance.learner_view(), 128, zeta_bs=0.1, seed=88
    )
    second = main_v2.sample_bell_scores(
        instance.measurement_source, 128, zeta_bs=0.1, seed=88
    )
    assert first.outcomes.shape == (2, 128, 3)
    assert np.array_equal(first.outcomes, second.outcomes)
    assert first.backend == "pure_ket"
    assert first.provenance is main_v2.DataProvenance.EMPIRICAL
    assert first.score("II") == 1.0
    before = first.outcomes.copy()
    queried = first.scores(("ZI", "IZ", "ZZ", "ZI"))
    assert queried[0] == queried[-1]
    assert np.array_equal(first.outcomes, before)
    assert not first.outcomes.flags.writeable
    assert first.copy_ledger.total == 256


def test_bell_backend_dispatch_distinguishes_ket_rank_one_density_and_mixed():
    zero, _ = _zero_and_one()
    ket_instance = main_v2.random_cebp_state(1, 1, block_states=(zero,), seed=1)
    density_instance = main_v2.random_cebp_state(
        1, 1, block_states=(zero * zero.dag(),), seed=1
    )
    mixed_instance = main_v2.random_cebp_state(
        1, 1, block_states=(qt.qeye(2) / 2,), seed=1
    )
    ket_record = main_v2.sample_bell_scores(ket_instance.learner_view(), 8, seed=2)
    density_record = main_v2.sample_bell_scores(
        density_instance.learner_view(), 8, seed=2
    )
    mixed_record = main_v2.sample_bell_scores(mixed_instance.learner_view(), 8, seed=2)
    assert ket_record.backend == "pure_ket"
    assert density_record.backend == "mixed_density"
    assert mixed_record.backend == "mixed_density"
    assert ket_instance.is_ket and not density_instance.is_ket and not mixed_instance.is_ket


def test_empirical_score_converges_to_exact_score():
    rho = 0.5 * (qt.qeye(2) + 0.6 * qt.sigmax())
    instance = main_v2.random_cebp_state(
        1, 1, block_states=(rho,), clifford_steps=1, seed=15
    )
    truth = main_v2.debug_exact_score_source(instance)
    record = main_v2.sample_bell_scores(
        instance.learner_view(), 20_000, zeta_bs=0.05, seed=16
    )
    errors = [abs(record.score(pauli) - truth.score(pauli)) for pauli in "IXYZ"]
    assert max(errors) < 0.03


def test_empirical_path_never_calls_exact_score_helper(monkeypatch):
    instance = main_v2.random_cebp_state(
        1, 1, block_states=(qt.basis(2, 0),), seed=4
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("empirical path attempted exact-state scoring")

    monkeypatch.setattr(main_v2, "debug_exact_pauli_score", forbidden)
    result = main_v2.empirical_certified_stabilizer_peeling(
        instance.learner_view(),
        _exact_config(M1=2_000, zeta_bs=0.05),
        seed=9,
    )
    assert result.success
    assert result.score_provenance is main_v2.DataProvenance.EMPIRICAL


def test_empirical_measurement_rejects_instance_oracle_container():
    instance = main_v2.random_cebp_state(1, 1, seed=1)
    with pytest.raises(TypeError, match="CEBPLearnerView"):
        main_v2.sample_bell_scores(instance, 10, seed=2)


def test_true_span_equality_not_equal_rank_only():
    assert len(main_v2.canonical_gf2_span_basis(("X",), 1)) == 1
    assert len(main_v2.canonical_gf2_span_basis(("Z",), 1)) == 1
    assert not main_v2.gf2_pauli_spans_equal(("X",), ("Z",), 1)


def test_different_sets_generating_same_span_are_accepted():
    source = main_v2.PauliScoreMap(
        n=2,
        score_map=_score_map(2, {"XI": 0.9, "IX": 0.9, "XX": 0.75}),
        uniform_radius=0.1,
    )
    result = main_v2.certified_stabilizer_peeling_v2(
        source,
        _exact_config(h_min=0.69, h_max=0.71, eta=0.02),
    )
    assert result.success
    assert result.inner_set != result.outer_set
    assert result.inner_span_basis == result.outer_span_basis
    assert result.generators == ("IX", "XI")


def test_exact_t_zero_branch():
    instance = main_v2.random_cebp_state(
        2,
        2,
        block_states=(qt.qeye(4) / 4,),
        clifford_steps=3,
        seed=7,
    )
    result = main_v2.debug_exact_certified_stabilizer_peeling(instance, _exact_config())
    assert result.success and result.t == 0
    assert result.generators == ()
    assert np.allclose(result.U_stab, np.eye(4))
    assert np.array_equal(result.tableau, np.eye(4, dtype=np.uint8))
    assert result.copy_ledger.total == 0 and result.M1 == 0


def test_exact_t_n_stabilizer_branch_and_lexicographic_basis():
    zero, _ = _zero_and_one()
    instance = main_v2.random_cebp_state(
        3,
        1,
        block_states=(zero, zero, zero),
        clifford_steps=7,
        seed=20,
    )
    result = main_v2.debug_exact_certified_stabilizer_peeling(instance, _exact_config())
    assert result.success and result.t == 3
    assert main._gf2_rank(
        np.column_stack([main.pauli_to_symplectic_col(p) for p in result.generators])
    ) == 3
    assert main_v2.peeling_clifford_mapping_holds(result.generators, result.U_stab)


def test_exact_nontrivial_partial_peeling_branch():
    zero, _ = _zero_and_one()
    instance = main_v2.random_cebp_state(
        2,
        1,
        block_states=(zero, qt.qeye(2) / 2),
        clifford_steps=5,
        seed=21,
    )
    result = main_v2.debug_exact_certified_stabilizer_peeling(instance, _exact_config())
    assert result.success and result.t == 1
    assert len(result.certified_span_basis) == 1
    assert main_v2.debug_exact_pauli_score(instance.state, result.generators[0]) == pytest.approx(1.0)


@pytest.mark.parametrize("block_size", (2, 3))
def test_exact_correlated_d2_d3_oracle_structural_validation(block_size):
    zero, one = _zero_and_one()
    if block_size == 2:
        correlated = (qt.tensor(zero, zero) + qt.tensor(one, one)).unit()
    else:
        correlated = (
            qt.tensor(zero, zero, zero) + qt.tensor(one, one, one)
        ).unit()
    instance = main_v2.random_cebp_state(
        block_size + 1,
        block_size,
        block_sizes=(block_size, 1),
        block_states=(correlated, qt.qeye(2) / 2),
        clifford_steps=3 * block_size,
        seed=100 + block_size,
    )
    result = main_v2.debug_exact_certified_stabilizer_peeling(instance, _exact_config())
    assert result.success and result.t == block_size
    assert all(
        main_v2.debug_exact_pauli_score(instance.state, generator) == pytest.approx(1.0)
        for generator in result.generators
    )
    assert all(
        instance.oracle_truth.hidden_block_support(generator) == (0,)
        for generator in result.generators
    )


def test_first_grid_point_is_selected_when_multiple_certify():
    source = main_v2.PauliScoreMap(
        n=1,
        score_map=_score_map(1, {"Z": 1.0}),
        uniform_radius=0.0,
    )
    config = _exact_config(h_min=0.61, h_max=0.81, eta=0.05)
    result = main_v2.certified_stabilizer_peeling_v2(source, config)
    assert result.success
    assert result.h == pytest.approx(config.threshold_grid[0])
    assert len(result.transcript) == 1


def test_no_certified_threshold_is_explicit_failure():
    source = main_v2.PauliScoreMap(
        n=1,
        score_map=_score_map(1, {"X": 0.7, "Z": 0.5}),
        uniform_radius=0.1,
    )
    result = main_v2.certified_stabilizer_peeling_v2(
        source,
        _exact_config(h_min=0.59, h_max=0.61, eta=0.02),
    )
    assert not result.success
    assert result.failure_reason == "no_certified_threshold"
    assert result.h is None and result.lambda_ is None and result.t is None
    assert result.U_stab is None and result.epsilon_peel is None
    assert all(not attempt.accepted for attempt in result.transcript)


def test_nonisotropic_matching_span_is_rejected_not_repaired():
    source = main_v2.PauliScoreMap(
        n=1,
        score_map=_score_map(1, {"X": 0.9, "Y": 0.9, "Z": 0.9}),
        uniform_radius=0.0,
    )
    result = main_v2.certified_stabilizer_peeling_v2(source, _exact_config())
    assert not result.success
    assert result.generators == ()
    assert any(
        attempt.reason == "certified_span_non_isotropic"
        for attempt in result.transcript
    )


def test_exact_result_contains_no_oracle_state_or_hidden_labels():
    instance = main_v2.random_cebp_state(1, 1, pure=True, seed=3)
    result = main_v2.debug_exact_certified_stabilizer_peeling(instance, _exact_config())
    result_fields = set(result.__dataclass_fields__)
    assert "oracle_truth" not in result_fields
    assert "hidden_partition" not in result_fields
    assert "state" not in result_fields and "_debug_state" not in result_fields
    assert result.score_provenance is main_v2.DataProvenance.EXACT


def test_empirical_certified_peeling_succeeds_and_counts_two_m1_copies():
    zero, _ = _zero_and_one()
    instance = main_v2.random_cebp_state(
        2, 1, block_states=(zero, zero), clifford_steps=6, seed=30
    )
    config = _exact_config(M1=5_000, zeta_bs=0.01)
    result = main_v2.empirical_certified_stabilizer_peeling(
        instance.learner_view(), config, seed=31
    )
    assert result.success and result.t == 2
    assert result.M1 == 5_000
    assert result.copy_ledger.as_dict() == {"peeling_bell_pool": 10_000}
    assert result.score_provenance is main_v2.DataProvenance.EMPIRICAL
    assert main_v2.peeling_clifford_mapping_holds(result.generators, result.U_stab)


def test_intentionally_insufficient_empirical_record_fails_visibly():
    instance = main_v2.random_cebp_state(1, 1, pure=True, seed=40)
    record = main_v2.sample_bell_scores(
        instance.learner_view(), 1, zeta_bs=0.1, seed=41
    )
    result = main_v2.certified_stabilizer_peeling_v2(record, _exact_config())
    assert not result.success
    assert result.failure_reason == "no_certified_threshold"
    assert result.copy_ledger.total == 2


def test_all_scores_are_evaluated_once_then_reused_across_grid():
    class CountingSource:
        n = 1
        frame = "counting_debug"
        provenance = main_v2.DataProvenance.EXACT
        uniform_radius = 0.0
        bell_rounds = 0
        copy_ledger = main_v2.CopyLedger()

        def __init__(self):
            self.calls = []

        def score(self, pauli):
            self.calls.append(pauli)
            return 1.0 if pauli in ("I", "Z") else 0.0

    source = CountingSource()
    result = main_v2.certified_stabilizer_peeling_v2(source, _exact_config())
    assert result.success
    assert source.calls == ["I", "X", "Y", "Z"]


def test_grid_and_tau_match_manuscript_formulas_and_conditions():
    config = main_v2.PeelingConfig(h_min=0.6, h_max=0.8, eta=0.01, M1=100)
    assert config.threshold_grid[0] == pytest.approx(0.6)
    assert config.threshold_grid[-1] == pytest.approx(0.8)
    assert config.actual_mesh <= 0.01 + 1e-15
    expected = np.sqrt(2 * np.log(2 * 4**2 / 0.05) / 100)
    assert main_v2.bell_score_uniform_radius(100, 2, 0.05) == pytest.approx(expected)

    calibrated = main_v2.PeelingConfig.calibrated(
        2, h_min=0.6, h_max=0.8, zeta_bs=0.05
    )
    delta_h = calibrated.h_max - calibrated.h_min
    expected_rounds = int(
        np.ceil(
            128 * (2 * 2 + 1) ** 2 / delta_h**2
            * np.log(2 * 4**2 / calibrated.zeta_bs)
        )
    )
    assert calibrated.M1 == expected_rounds
    assert calibrated.eta <= delta_h / (4 * (2 * 2 + 1)) + 1e-15
    assert main_v2.bell_score_uniform_radius(
        calibrated.M1, 2, calibrated.zeta_bs
    ) <= delta_h / (8 * (2 * 2 + 1)) + 1e-15


def test_success_and_theorem_calibration_are_explicitly_distinct():
    source = main_v2.PauliScoreMap(
        n=1,
        score_map=_score_map(1, {"Z": 1.0}),
        uniform_radius=0.0,
    )
    manual = main_v2.certified_stabilizer_peeling_v2(
        source,
        _exact_config(h_min=0.6, h_max=0.8, eta=0.2),
    )
    calibrated = main_v2.certified_stabilizer_peeling_v2(source, _exact_config())
    failed = main_v2.certified_stabilizer_peeling_v2(
        main_v2.PauliScoreMap(
            n=1,
            score_map=_score_map(1, {"X": 0.7, "Z": 0.5}),
            uniform_radius=0.1,
        ),
        _exact_config(h_min=0.59, h_max=0.61, eta=0.02),
    )

    assert manual.success and not manual.theorem_preconditions_hold
    assert not manual.theorem_certified
    assert calibrated.success and calibrated.theorem_preconditions_hold
    assert calibrated.theorem_certified
    assert not failed.success and not failed.theorem_certified


def test_small_system_enumeration_guard_is_explicit():
    source = main_v2.PauliScoreMap(
        n=2,
        score_map=_score_map(2),
        uniform_radius=0.0,
    )
    with pytest.raises(ValueError, match="exhaustively enumerates"):
        main_v2.certified_stabilizer_peeling_v2(
            source, _exact_config(max_enumeration_qubits=1)
        )


def test_constructive_peeling_fidelity_bound_numerically():
    rho_almost_zero = qt.Qobj(np.diag([0.95, 0.05]), dims=[[2], [2]])
    instance = main_v2.random_cebp_state(
        2,
        1,
        block_states=(rho_almost_zero, qt.qeye(2) / 2),
        clifford_steps=5,
        seed=55,
    )
    result = main_v2.debug_exact_certified_stabilizer_peeling(
        instance,
        _exact_config(h_min=0.7, h_max=0.8, eta=0.002),
    )
    assert result.success and result.t == 1
    expectation = main_v2.debug_exact_pauli_expectation(
        instance.state, result.generators[0]
    )
    syndrome_bit = 0 if expectation >= 0.0 else 1
    unitary = qt.Qobj(result.U_stab, dims=[[2, 2], [2, 2]])
    transformed = unitary.dag() * instance.state * unitary
    matrix = transformed.full().reshape(2, 2, 2, 2)
    dominant_weight = float(np.trace(matrix[syndrome_bit, :, syndrome_bit, :]).real)
    certified_bound = np.sqrt(max(0.0, 1.0 - result.epsilon_peel))
    assert np.sqrt(dominant_weight) + 1e-12 >= certified_bound
    assert dominant_weight == pytest.approx(0.95)


def test_copy_ledger_extension_remains_immutable():
    empty = main_v2.CopyLedger()
    bell = empty.with_entry("peeling_bell_pool", 20)
    assert empty.total == 0
    assert bell.total == 20
    with pytest.raises(ValueError, match="already exists"):
        bell.with_entry("peeling_bell_pool", 2)


def test_post_phase_four_downstream_apis_are_not_implemented():
    deferred = (
        "simultaneous_clifford_localization",
        "recover_syndrome_signs",
        "tomograph_empirical_registers",
    )
    assert all(not hasattr(main_v2, name) for name in deferred)
    assert hasattr(main_v2, "full_cebp_tomography")
