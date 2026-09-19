import itertools
from dataclasses import replace

import numpy as np
import pytest
import qutip as qt

import main_v2


def _score_map(n, overrides=None):
    values = {
        "".join(chars): 0.0 for chars in itertools.product("IXYZ", repeat=n)
    }
    values["I" * n] = 1.0
    if overrides:
        values.update(overrides)
    return values


def _peeling_config(**overrides):
    values = dict(h_min=0.6, h_max=0.8, eta=0.005, return_details=True)
    values.update(overrides)
    return main_v2.PeelingConfig(**values)


def _synthetic_peeling(n, overrides=None, **config_overrides):
    source = main_v2.PauliScoreMap(
        n=n,
        score_map=_score_map(n, overrides),
        uniform_radius=0.0,
        frame="physical",
    )
    result = main_v2.certified_stabilizer_peeling_v2(
        source, _peeling_config(**config_overrides)
    )
    assert result.success
    return result


def _recovery_source(n, overrides=None, radius=0.0):
    return main_v2.PauliScoreMap(
        n=n,
        score_map=_score_map(n, overrides),
        uniform_radius=radius,
        frame="peeled",
    )


def _recovery_config(**overrides):
    values = dict(theta=0.5, return_details=True)
    values.update(overrides)
    return main_v2.RecoveryConfig(**values)


def _haar_ket(qubits, seed):
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=2**qubits) + 1j * rng.normal(size=2**qubits)
    return qt.Qobj(vector, dims=[[2] * qubits, [1] * qubits]).unit()


def test_recovery_config_calibrated_formula_and_validation():
    config = main_v2.RecoveryConfig.calibrated(
        2, theta_0=0.4, lambda_0=0.1, zeta_rank=0.05
    )
    expected = int(
        np.ceil(128 / (0.1**2 * 0.4**2) * np.log(2 * 4**2 / 0.05))
    )
    assert config.theta == 0.4 and config.M2 == expected
    with pytest.raises(ValueError, match="theta"):
        main_v2.RecoveryConfig(theta=1.0)


def test_full_pauli_parser_t0_partial_and_tn_boundaries():
    assert main_v2.parse_peeled_full_pauli("XY", 0) == ((), "XY", None)
    assert main_v2.parse_peeled_full_pauli("ZIX", 2) == ((1, 0), "X", None)
    assert main_v2.parse_peeled_full_pauli("XIX", 2)[2] == "not_prefix_Z_form"
    assert main_v2.parse_peeled_full_pauli("ZI", 1)[2] == "prefix_only"
    assert main_v2.parse_peeled_full_pauli("ZZ", 2)[2] == "prefix_only"


def test_phase_free_product_and_sector_axis_model():
    triple = main_v2.RecoveredSector(3, "XI", "ZI", "YI")
    singleton = main_v2.RecoveredSector(7, "IX")
    assert main_v2.phase_free_pauli_product("XI", "ZI") == "YI"
    assert triple.kind == "TRIPLE" and singleton.kind == "SINGLETON"
    assert main_v2.recovered_sector_axes((triple, singleton)) == ("XI", "ZI", "IX")
    assert len(main_v2.recovered_sector_span_basis((triple, singleton))) == 3
    assert main_v2.pauli_in_recovered_span("YI", (triple, singleton))


@pytest.mark.parametrize(
    ("left", "right", "phase", "pauli"),
    (
        ("X", "X", 0, "I"),
        ("Y", "Y", 0, "I"),
        ("Z", "Z", 0, "I"),
        ("X", "Y", 1, "Z"),
        ("Y", "X", 3, "Z"),
        ("Y", "Z", 1, "X"),
        ("Z", "Y", 3, "X"),
        ("Z", "X", 1, "Y"),
        ("X", "Z", 3, "Y"),
    ),
)
def test_signed_pauli_product_tracks_exact_one_qubit_phase(
    left, right, phase, pauli
):
    product = main_v2.signed_pauli_product(left, right)
    assert (product.phase_exponent, product.pauli) == (phase, pauli)
    assert main_v2.phase_free_pauli_product(left, right) == pauli


def test_signed_multi_qubit_and_hermitian_reduction_are_separate_layers():
    assert main_v2.signed_pauli_product("XZ", "ZX") == main_v2.SignedPauliProduct(
        0, "YY"
    )
    assert main_v2.hermitian_pauli_product("XX", "ZZ") == (-1, "YY")
    with pytest.raises(ValueError, match="not Hermitian"):
        main_v2.hermitian_pauli_product("X", "Z")
    assert main_v2.phase_free_pauli_product("XX", "ZZ") == "YY"


def test_phase4_recovery_boundary_requires_theorem_validity_unless_overridden():
    peeling = _synthetic_peeling(1)
    valid = main_v2.rank_guided_sector_recovery(
        _recovery_source(1, {"X": 0.8}), peeling, _recovery_config(theta=0.5)
    )
    assert main_v2._validate_recovery_for_grouping(
        valid, allow_uncalibrated_recovery=False
    )
    manual = replace(valid, theorem_recovery_preconditions_hold=False)
    with pytest.raises(main_v2.RecoveryPreconditionError, match="not_theorem_calibrated"):
        main_v2._validate_recovery_for_grouping(
            manual, allow_uncalibrated_recovery=False
        )
    assert not main_v2._validate_recovery_for_grouping(
        manual, allow_uncalibrated_recovery=True
    )


def test_symplectic_validity_accepts_valid_and_detects_invalid_collections():
    valid = (
        main_v2.RecoveredSector(0, "XI", "ZI", "YI"),
        main_v2.RecoveredSector(1, "IX"),
    )
    wrong_y = (main_v2.RecoveredSector(0, "XI", "ZI", "XI"),)
    commuting_triple = (main_v2.RecoveredSector(0, "XI", "IX", "XX"),)
    cross_anticommuting = (
        main_v2.RecoveredSector(0, "X"),
        main_v2.RecoveredSector(1, "Z"),
    )
    dependent = (
        main_v2.RecoveredSector(0, "X"),
        main_v2.RecoveredSector(1, "X"),
    )
    assert main_v2.recovered_sectors_are_symplectically_valid(valid)
    assert not main_v2.recovered_sectors_are_symplectically_valid(wrong_y)
    assert not main_v2.recovered_sectors_are_symplectically_valid(commuting_triple)
    assert not main_v2.recovered_sectors_are_symplectically_valid(cross_anticommuting)
    assert not main_v2.recovered_sectors_are_symplectically_valid(dependent)


def test_dressing_no_triples_is_identity():
    sectors = (main_v2.RecoveredSector(0, "XI"),)
    assert main_v2.dress_residual_pauli("IZ", sectors) == "IZ"


@pytest.mark.parametrize(
    ("candidate", "expected"),
    (("XX", "IX"), ("ZX", "IX"), ("YX", "IX")),
)
def test_dressing_one_triple_handles_x_z_and_both_anticommutators(candidate, expected):
    sectors = (main_v2.RecoveredSector(0, "XI", "ZI", "YI"),)
    dressed = main_v2.dress_residual_pauli(candidate, sectors)
    assert dressed == expected
    assert main_v2.recovered_sectors_are_symplectically_valid(sectors)
    assert main_v2.recovered_sector_span_basis(sectors + (main_v2.RecoveredSector(1, candidate),)) == main_v2.recovered_sector_span_basis(
        sectors + (main_v2.RecoveredSector(1, dressed),)
    )


def test_dressing_multiple_completed_sectors_neutralizes_all():
    sectors = (
        main_v2.RecoveredSector(0, "XII", "ZII", "YII"),
        main_v2.RecoveredSector(1, "IXI", "IZI", "IYI"),
    )
    dressed = main_v2.dress_residual_pauli("YYX", sectors)
    assert dressed == "IIX"
    assert main_v2.recovered_sector_span_basis(
        sectors + (main_v2.RecoveredSector(2, "YYX"),)
    ) == main_v2.recovered_sector_span_basis(
        sectors + (main_v2.RecoveredSector(2, dressed),)
    )


def test_singleton_append_preserves_old_sectors_and_adds_one_axis():
    old = (main_v2.RecoveredSector(0, "XI"),)
    updated, action, affected = main_v2._add_dressed_recovery_direction(old, "IX")
    assert action == "append_singleton" and affected == (1,)
    assert updated[0] is old[0] and updated[1].x == "IX"
    assert len(main_v2.recovered_sector_axes(updated)) == 2
    assert main_v2.recovered_sectors_are_symplectically_valid(updated)


def test_pivot_completion_one_anticommuting_singleton():
    old = (main_v2.RecoveredSector(0, "X"),)
    updated, action, affected = main_v2._add_dressed_recovery_direction(old, "Z")
    assert action == "complete_pivot" and affected == (0,)
    assert updated == (main_v2.RecoveredSector(0, "X", "Z", "Y"),)
    assert main_v2.recovered_sectors_are_symplectically_valid(updated)


def test_multiple_singleton_pivot_rebase_is_deterministic_and_preserves_old_span():
    old = (
        main_v2.RecoveredSector(0, "XI"),
        main_v2.RecoveredSector(1, "IX"),
    )
    old_span = main_v2.recovered_sector_span_basis(old)
    updated, action, affected = main_v2._add_dressed_recovery_direction(old, "ZZ")
    assert action == "complete_pivot" and affected == (0, 1)
    assert updated[0] == main_v2.RecoveredSector(0, "XI", "ZZ", "YZ")
    assert updated[1] == main_v2.RecoveredSector(1, "XX")
    rebased_old = (main_v2.RecoveredSector(0, "XI"), updated[1])
    assert main_v2.recovered_sector_span_basis(rebased_old) == old_span
    assert len(main_v2.recovered_sector_span_basis(updated)) == len(old_span) + 1
    assert main_v2.recovered_sectors_are_symplectically_valid(updated)


def test_full_prefix_scores_control_ranking_and_invalid_prefix_is_skipped():
    peeling = _synthetic_peeling(2, {"ZI": 1.0})
    source = _recovery_source(
        2,
        {"ZI": 0.99, "XI": 0.95, "ZX": 0.90, "IX": 0.60},
    )
    result = main_v2.rank_guided_sector_recovery(
        source, peeling, _recovery_config(theta=0.55)
    )
    ranked = [candidate.full_pauli for candidate in result.ranked_candidates]
    assert ranked[:4] == ["ZI", "XI", "ZX", "IX"]
    assert result.transcript[0].action == "prefix_only"
    assert result.transcript[1].action == "not_prefix_Z_form"
    assert result.transcript[2].action == "append_singleton"
    assert result.transcript[3].action == "already_in_span"
    assert result.sectors == (main_v2.RecoveredSector(0, "X"),)


def test_visible_composite_components_rank_first_and_composite_is_in_span():
    peeling = _synthetic_peeling(2)
    source = _recovery_source(2, {"XI": 0.8, "IX": 0.7, "XX": 0.56})
    result = main_v2.rank_guided_sector_recovery(
        source, peeling, _recovery_config(theta=0.5)
    )
    assert [step.full_pauli for step in result.transcript] == ["XI", "IX", "XX"]
    assert [step.action for step in result.transcript] == [
        "append_singleton", "append_singleton", "already_in_span"
    ]
    assert main_v2.pauli_in_recovered_span("XX", result.sectors)


def test_threshold_span_completeness_direct_exact_inclusion():
    peeling = _synthetic_peeling(3, {"ZII": 1.0})
    source = _recovery_source(
        3,
        {"ZIX": 0.9, "IIZ": 0.8, "ZIY": 0.75, "ZZX": 0.7, "ZXX": 0.65},
    )
    result = main_v2.rank_guided_sector_recovery(
        source, peeling, _recovery_config(theta=0.6)
    )
    assert result.threshold_span_complete
    for candidate in result.ranked_candidates:
        _bits, residual, reason = main_v2.parse_peeled_full_pauli(
            candidate.full_pauli, result.t
        )
        if reason is None:
            assert main_v2.pauli_in_recovered_span(residual, result.sectors)


def test_t0_and_tn_recovery_boundaries():
    t0 = _synthetic_peeling(1)
    recovered = main_v2.rank_guided_sector_recovery(
        _recovery_source(1, {"X": 0.8}), t0, _recovery_config(theta=0.5)
    )
    assert recovered.t == 0 and recovered.m == 1
    assert recovered.sectors == (main_v2.RecoveredSector(0, "X"),)

    tn = _synthetic_peeling(1, {"Z": 1.0})
    empty = main_v2.rank_guided_sector_recovery(
        _recovery_source(1, {"Z": 1.0}), tn, _recovery_config(theta=0.5)
    )
    assert empty.t == 1 and empty.m == 0 and empty.sectors == ()
    assert empty.threshold_span_complete


def test_strict_recovery_margin_boundaries():
    peeling = _synthetic_peeling(1)
    with pytest.raises(main_v2.RecoveryPreconditionError, match="threshold_margin"):
        main_v2.rank_guided_sector_recovery(
            _recovery_source(1, {"X": 0.8}, radius=0.1),
            peeling,
            _recovery_config(theta=0.5),
        )
    with pytest.raises(main_v2.RecoveryPreconditionError, match="ranking_gap"):
        main_v2.rank_guided_sector_recovery(
            _recovery_source(1, {"X": 0.8}, radius=0.09),
            peeling,
            _recovery_config(theta=0.5),
        )
    passed = main_v2.rank_guided_sector_recovery(
        _recovery_source(1, {"X": 0.8}),
        peeling,
        _recovery_config(theta=0.5),
    )
    assert passed.threshold_margin_holds and passed.ranking_gap_margin_holds


def test_margin_debug_override_reports_but_does_not_claim_theorem():
    peeling = _synthetic_peeling(1)
    result = main_v2.rank_guided_sector_recovery(
        _recovery_source(1, {"X": 0.8}, radius=0.1),
        peeling,
        _recovery_config(theta=0.5, allow_margin_failure=True),
    )
    assert result.success and not result.threshold_margin_holds
    assert not result.theorem_recovery_preconditions_hold


def test_default_empirical_path_rejects_uncalibrated_peeling_before_sampling():
    manual = _synthetic_peeling(
        1, {"Z": 1.0}, h_min=0.6, h_max=0.8, eta=0.2
    )
    assert manual.success and not manual.theorem_certified
    instance = main_v2.random_cebp_state(
        1, 1, block_states=(qt.basis(2, 0),), seed=2
    )
    with pytest.raises(main_v2.RecoveryPreconditionError, match="not_theorem_calibrated"):
        main_v2.empirical_rank_guided_sector_recovery(
            instance.learner_view(), manual, _recovery_config(M2=100), seed=3
        )


def test_invalid_score_frame_and_enumeration_guard_are_explicit():
    peeling = _synthetic_peeling(2)
    physical = main_v2.PauliScoreMap(
        n=2,
        score_map=_score_map(2),
        uniform_radius=0.0,
        frame="physical",
    )
    with pytest.raises(main_v2.RecoveryPreconditionError, match="invalid_score_frame"):
        main_v2.rank_guided_sector_recovery(
            physical, peeling, _recovery_config()
        )
    with pytest.raises(ValueError, match="exhaustively enumerates"):
        main_v2.rank_guided_sector_recovery(
            _recovery_source(2),
            peeling,
            _recovery_config(max_enumeration_qubits=1),
        )


def test_all_full_scores_are_evaluated_once_and_ties_are_lexicographic():
    peeling = _synthetic_peeling(1)

    class CountingSource:
        n = 1
        frame = "peeled"
        provenance = main_v2.DataProvenance.EXACT
        uniform_radius = 0.0
        bell_rounds = 0
        copy_ledger = main_v2.CopyLedger()

        def __init__(self):
            self.calls = []

        def score(self, pauli):
            self.calls.append(pauli)
            return 1.0 if pauli == "I" else (0.6 if pauli in "XYZ" else 0.0)

    source = CountingSource()
    result = main_v2.rank_guided_sector_recovery(
        source, peeling, _recovery_config(theta=0.5)
    )
    assert source.calls == ["I", "X", "Y", "Z"]
    assert [candidate.full_pauli for candidate in result.ranked_candidates] == [
        "X", "Y", "Z"
    ]


def test_exact_debug_recovery_has_zero_copy_cost_and_no_oracle_fields():
    ket = (qt.basis(2, 0) + np.exp(1j * np.pi / 4) * qt.basis(2, 1)).unit()
    instance = main_v2.random_cebp_state(
        1, 1, block_states=(ket,), clifford_steps=3, seed=8
    )
    peeling = main_v2.debug_exact_certified_stabilizer_peeling(
        instance, _peeling_config()
    )
    result = main_v2.debug_exact_rank_guided_sector_recovery(
        instance, peeling, _recovery_config(theta=0.45)
    )
    assert result.success and result.copy_ledger.total == 0
    assert result.cumulative_copy_ledger.total == 0
    assert result.score_provenance is main_v2.DataProvenance.EXACT
    fields = set(result.__dataclass_fields__)
    assert "oracle_truth" not in fields and "hidden_partition" not in fields


def test_peeled_exact_source_uses_Udagger_rho_U_orientation():
    instance = main_v2.random_cebp_state(
        2,
        1,
        block_states=(qt.basis(2, 0), qt.qeye(2) / 2),
        clifford_steps=5,
        seed=10,
    )
    peeling = main_v2.debug_exact_certified_stabilizer_peeling(
        instance, _peeling_config()
    )
    source = main_v2.debug_exact_peeled_score_source(instance, peeling)
    Uq = qt.Qobj(peeling.U_stab, dims=[[2, 2], [2, 2]])
    direct = Uq.dag() * instance.state * Uq
    for pauli in ("ZI", "IX", "ZZ"):
        assert source.score(pauli) == pytest.approx(
            main_v2.debug_exact_pauli_score(direct, pauli)
        )


@pytest.mark.parametrize("mixed", (False, True))
def test_empirical_peeled_bell_backend_and_fresh_pool(mixed):
    ket = (qt.basis(2, 0) + np.exp(1j * np.pi / 4) * qt.basis(2, 1)).unit()
    block = ket * ket.dag() if mixed else ket
    instance = main_v2.random_cebp_state(
        1, 1, block_states=(block,), clifford_steps=3, seed=20
    )
    peeling = main_v2.debug_exact_certified_stabilizer_peeling(
        instance, _peeling_config()
    )
    initial = main_v2.sample_bell_scores(
        instance.learner_view(), 200, seed=21
    )
    recovery = main_v2.sample_peeled_bell_scores(
        instance.learner_view(), peeling, 200, zeta_rank=0.05, seed=22
    )
    assert initial.pool_name == "peeling_bell_pool"
    assert recovery.pool_name == "recovery_bell_pool" and recovery.frame == "peeled"
    assert recovery.backend == ("mixed_density" if mixed else "pure_ket")
    assert not np.shares_memory(initial.outcomes, recovery.outcomes)
    assert recovery.copy_ledger.as_dict() == {"recovery_bell_pool": 400}


def test_empirical_pure_and_mixed_recovery_seeded():
    phase_ket = (qt.basis(2, 0) + np.exp(1j * np.pi / 4) * qt.basis(2, 1)).unit()
    pure_instance = main_v2.random_cebp_state(
        1, 1, block_states=(phase_ket,), clifford_steps=3, seed=30
    )
    pure_peeling = main_v2.debug_exact_certified_stabilizer_peeling(
        pure_instance, _peeling_config()
    )
    pure = main_v2.empirical_rank_guided_sector_recovery(
        pure_instance.learner_view(),
        pure_peeling,
        _recovery_config(theta=0.45, M2=6_000),
        seed=31,
    )
    assert pure.success and pure.sectors and pure.copy_ledger.total == 12_000
    assert pure.threshold_span_complete
    for candidate in pure.ranked_candidates:
        _bits, residual, reason = main_v2.parse_peeled_full_pauli(
            candidate.full_pauli, pure.t
        )
        if reason is None:
            assert main_v2.pauli_in_recovered_span(residual, pure.sectors)

    rho = 0.5 * (qt.qeye(2) + 0.8 * qt.sigmax())
    mixed_instance = main_v2.random_cebp_state(
        1, 1, block_states=(rho,), clifford_steps=3, seed=32
    )
    mixed_peeling = main_v2.debug_exact_certified_stabilizer_peeling(
        mixed_instance,
        _peeling_config(h_min=0.7, h_max=0.8, eta=0.003),
    )
    mixed = main_v2.empirical_rank_guided_sector_recovery(
        mixed_instance.learner_view(),
        mixed_peeling,
        _recovery_config(theta=0.5, M2=5_000),
        seed=33,
    )
    assert mixed.success and mixed.sectors and mixed.copy_ledger.total == 10_000


def test_empirical_tn_runs_fresh_pool_and_cumulative_ledger_is_2m1_plus_2m2():
    empirical_peeling_source = main_v2.PauliScoreMap(
        n=1,
        score_map=_score_map(1, {"Z": 1.0}),
        uniform_radius=0.0,
        provenance=main_v2.DataProvenance.EMPIRICAL,
        frame="physical",
        bell_rounds=10,
        copy_ledger=main_v2.CopyLedger((("peeling_bell_pool", 20),)),
    )
    peeling = main_v2.certified_stabilizer_peeling_v2(
        empirical_peeling_source, _peeling_config(M1=10)
    )
    instance = main_v2.random_cebp_state(
        1, 1, block_states=(qt.basis(2, 0),), clifford_steps=1, seed=40
    )
    result = main_v2.empirical_rank_guided_sector_recovery(
        instance.learner_view(), peeling, _recovery_config(theta=0.6, M2=2_000), seed=41
    )
    assert result.t == result.n and result.sectors == ()
    assert result.copy_ledger.as_dict() == {"recovery_bell_pool": 4_000}
    assert result.cumulative_copy_ledger.as_dict() == {
        "peeling_bell_pool": 20,
        "recovery_bell_pool": 4_000,
    }
    assert result.cumulative_copy_ledger.total == 2 * 10 + 2 * 2_000


@pytest.mark.parametrize("block_size", (2, 3))
def test_correlated_d2_d3_recovery_is_oracle_block_pure(block_size):
    correlated = _haar_ket(block_size, 100 + block_size)
    one_qubit_reduction = qt.ptrace(correlated, [0])
    assert float((one_qubit_reduction * one_qubit_reduction).tr().real) < 0.999
    instance = main_v2.random_cebp_state(
        block_size + 1,
        block_size,
        block_sizes=(block_size, 1),
        block_states=(correlated, qt.qeye(2) / 2),
        clifford_steps=3 * (block_size + 1),
        seed=200 + block_size,
    )
    peeling = main_v2.debug_exact_certified_stabilizer_peeling(
        instance,
        _peeling_config(h_min=0.95, h_max=0.99, eta=0.0005),
    )
    result = main_v2.debug_exact_rank_guided_sector_recovery(
        instance, peeling, _recovery_config(theta=0.08)
    )
    labels = main_v2.debug_oracle_sector_block_labels(instance, peeling, result)
    assert result.success and result.sectors
    assert all(len(label_set) == 1 for label_set in labels)
    assert all(label_set == (0,) for label_set in labels)
    assert all(
        main_v2.debug_oracle_residual_block_labels(instance, peeling, member) == (0,)
        for sector in result.sectors
        for member in sector.members
    )
    assert main_v2.recovered_sectors_are_symplectically_valid(result.sectors)


def test_d1_structural_fixture_and_no_localization_output():
    phase_ket = (qt.basis(2, 0) + np.exp(1j * np.pi / 4) * qt.basis(2, 1)).unit()
    instance = main_v2.random_cebp_state(
        2,
        1,
        block_states=(qt.basis(2, 0), phase_ket),
        clifford_steps=7,
        seed=50,
    )
    peeling = main_v2.debug_exact_certified_stabilizer_peeling(
        instance, _peeling_config()
    )
    result = main_v2.debug_exact_rank_guided_sector_recovery(
        instance, peeling, _recovery_config(theta=0.45)
    )
    assert result.success and result.t == 1 and result.sectors
    assert all(
        len(labels) == 1
        for labels in main_v2.debug_oracle_sector_block_labels(instance, peeling, result)
    )
    assert not hasattr(result, "U_rec") and not hasattr(result, "localization")


def test_empirical_learner_path_never_uses_oracle_helpers(monkeypatch):
    ket = (qt.basis(2, 0) + np.exp(1j * np.pi / 4) * qt.basis(2, 1)).unit()
    instance = main_v2.random_cebp_state(
        1, 1, block_states=(ket,), clifford_steps=3, seed=60
    )
    peeling = main_v2.debug_exact_certified_stabilizer_peeling(
        instance, _peeling_config()
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("learner path used oracle validation")

    monkeypatch.setattr(main_v2, "debug_oracle_residual_block_labels", forbidden)
    monkeypatch.setattr(main_v2, "debug_oracle_sector_block_labels", forbidden)
    result = main_v2.empirical_rank_guided_sector_recovery(
        instance.learner_view(),
        peeling,
        _recovery_config(theta=0.45, M2=6_000),
        seed=61,
    )
    assert result.success
    with pytest.raises(TypeError, match="CEBPLearnerView"):
        main_v2.sample_peeled_bell_scores(
            instance, peeling, 10, zeta_rank=0.05, seed=62
        )


def test_no_phase5_or_later_api_is_present():
    deferred = (
        "simultaneous_clifford_localization",
        "recover_syndrome_signs",
        "tomograph_empirical_registers",
    )
    assert all(not hasattr(main_v2, name) for name in deferred)
    assert hasattr(main_v2, "full_cebp_tomography")
