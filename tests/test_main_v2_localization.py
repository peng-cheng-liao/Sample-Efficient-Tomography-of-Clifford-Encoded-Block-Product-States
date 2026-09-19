from dataclasses import replace

import numpy as np
import pytest

import main_v2


def _manual_recovery(sectors, *, m, t=0, theorem=True, copies=True):
    sectors = tuple(sectors)
    n = m + t
    recovery_ledger = (
        main_v2.CopyLedger((("recovery_bell_pool", 40),))
        if copies
        else main_v2.CopyLedger()
    )
    cumulative = (
        main_v2.CopyLedger(
            (("peeling_bell_pool", 20), ("recovery_bell_pool", 40))
        )
        if copies
        else main_v2.CopyLedger()
    )
    return main_v2.RecoveryResult(
        success=True,
        failure_reason=None,
        n=n,
        t=t,
        m=m,
        theta=0.1,
        tau_rank=0.01,
        M2=20 if copies else 0,
        zeta_rank=0.05,
        lambda_=0.1,
        sectors=sectors,
        independent_axes=main_v2.recovered_sector_axes(sectors),
        recovered_span_basis=main_v2.recovered_sector_span_basis(sectors, m),
        ranked_survivor_count=len(sectors),
        score_provenance=main_v2.DataProvenance.EMPIRICAL,
        score_frame="peeled",
        threshold_margin_holds=True,
        ranking_gap_margin_holds=True,
        theorem_recovery_preconditions_hold=theorem,
        threshold_span_complete=True,
        copy_ledger=recovery_ledger,
        cumulative_copy_ledger=cumulative,
    )


def _manual_grouping(recovery, clusters, *, ell=None, theorem=True):
    clusters = tuple(tuple(cluster) for cluster in clusters)
    ell = max((len(cluster) for cluster in clusters), default=1) if ell is None else ell
    grouping_ledger = main_v2.CopyLedger((("grouping_ordinary_pool", 0),))
    cumulative = main_v2.CopyLedger(
        recovery.cumulative_copy_ledger.entries + grouping_ledger.entries
    )
    return main_v2.GroupingResult(
        success=True,
        failure_reason=None,
        L=len(recovery.sectors),
        ell_grp=ell,
        clusters=clusters,
        eta_test=0.5,
        tau_kappa=0.1,
        beta_peel=0.0,
        eta_s=None,
        xi_s=None,
        no_false_merge_condition_holds=True,
        exact_recovery_window_holds=None,
        recovery_theorem_preconditions_hold=recovery.theorem_recovery_preconditions_hold,
        theorem_grouping_preconditions_hold=theorem,
        recovery_score_provenance=recovery.score_provenance,
        moment_provenance=main_v2.DataProvenance.EMPIRICAL,
        realized_query_count=0,
        query_count_by_order=(),
        realized_grouping_copies=0,
        copies_by_order=(),
        N_test_max=0,
        N_test_simplified_bound=0,
        delta_tuple=None,
        ordinary_copy_upper_bound=0,
        grouping_copy_ledger=grouping_ledger,
        cumulative_copy_ledger=cumulative,
        merge_rounds=0,
    )


def _localize(sectors, clusters, *, m, d, t=0, theorem=True, **config_values):
    recovery = _manual_recovery(sectors, m=m, t=t)
    grouping = _manual_grouping(recovery, clusters, theorem=theorem)
    config = main_v2.LocalizationConfig(**config_values)
    return recovery, grouping, main_v2.localize_grouped_recovery(
        recovery, grouping, d=d, config=config
    )


@pytest.mark.parametrize(
    ("sectors", "cluster", "m", "dimension", "rank", "h", "q", "k"),
    (
        ((main_v2.RecoveredSector(0, "X"),), (0,), 1, 1, 0, 0, 1, 1),
        (
            (main_v2.RecoveredSector(0, "X", "Z", "Y"),),
            (0,),
            1,
            2,
            2,
            1,
            0,
            1,
        ),
        (
            (
                main_v2.RecoveredSector(0, "XI", "ZI", "YI"),
                main_v2.RecoveredSector(1, "IX", "IZ", "IY"),
            ),
            (0, 1),
            2,
            4,
            4,
            2,
            0,
            2,
        ),
        (
            (
                main_v2.RecoveredSector(0, "XI", "ZI", "YI"),
                main_v2.RecoveredSector(1, "IX"),
            ),
            (0, 1),
            2,
            3,
            2,
            1,
            1,
            2,
        ),
        (
            (
                main_v2.RecoveredSector(0, "XII"),
                main_v2.RecoveredSector(1, "IXI"),
                main_v2.RecoveredSector(2, "IIX"),
            ),
            (0, 1, 2),
            3,
            3,
            0,
            0,
            3,
            3,
        ),
    ),
)
def test_restricted_symplectic_structure(sectors, cluster, m, dimension, rank, h, q, k):
    recovery = _manual_recovery(sectors, m=m)
    structure = main_v2.analyze_group_symplectic_structure(recovery, cluster)
    assert (structure.dimension, structure.symplectic_rank) == (dimension, rank)
    assert (structure.h_C, structure.q_C, structure.k_C) == (h, q, k)
    assert structure.k_C == len(cluster)
    assert len(structure.radical_basis) == q
    assert len(structure.hyperbolic_pairs) == h
    assert np.array_equal(structure.restricted_gram, structure.restricted_gram.T)
    assert not np.any(np.diag(structure.restricted_gram))


def test_odd_restricted_symplectic_rank_is_rejected():
    with pytest.raises(main_v2.LocalizationInvariantError, match="even"):
        main_v2._restricted_symplectic_parameters(np.array([[1]], dtype=np.uint8))


def test_cross_group_direct_sum_orthogonality_and_global_span():
    sectors = (
        main_v2.RecoveredSector(0, "XII", "ZII", "YII"),
        main_v2.RecoveredSector(1, "IXI"),
        main_v2.RecoveredSector(2, "IIX"),
    )
    recovery, _grouping, result = _localize(
        sectors, ((0, 1), (2,)), m=3, d=2
    )
    assert result.success
    assert result.cross_group_direct_sum_holds
    assert result.cross_group_symplectic_orthogonality_holds
    assert sum(structure.dimension for structure in result.structures) == len(
        recovery.independent_axes
    )
    assert main_v2.canonical_gf2_span_basis(
        tuple(axis for structure in result.structures for axis in structure.ordered_independent_axes),
        recovery.m,
    ) == recovery.recovered_span_basis


def test_invalid_cross_group_pairing_is_rejected_without_dropping_axes():
    sectors = (
        main_v2.RecoveredSector(0, "X"),
        main_v2.RecoveredSector(1, "Z"),
    )
    recovery = _manual_recovery(sectors, m=1)
    grouping = _manual_grouping(recovery, ((0,), (1,)))
    result = main_v2.localize_grouped_recovery(recovery, grouping, d=1)
    assert not result.success
    assert "symplectically valid" in result.failure_reason
    assert result.residual_tableau is None and result.J_C == ()


def test_radical_partners_are_chosen_simultaneously_across_clusters():
    sectors = (
        main_v2.RecoveredSector(0, "XXI"),
        main_v2.RecoveredSector(1, "IXX"),
    )
    _recovery, _grouping, result = _localize(
        sectors, ((0,), (1,)), m=3, d=1, return_details=True
    )
    assert result.success and result.K_rec == 2 and result.J_aux == (2,)
    radicals = [
        np.asarray(local.source_radical_axes[0], dtype=np.uint8)
        for local in result.cluster_localizations
    ]
    partners = [
        np.asarray(local.radical_partners[0], dtype=np.uint8)
        for local in result.cluster_localizations
    ]
    pairing = np.array(
        [
            [main_v2._symplectic_pairing(radical, partner, 3) for partner in partners]
            for radical in radicals
        ]
    )
    partner_pairing = np.array(
        [
            [main_v2._symplectic_pairing(left, right, 3) for right in partners]
            for left in partners
        ]
    )
    assert np.array_equal(pairing, np.eye(2, dtype=int))
    assert not np.any(partner_pairing)
    assert main_v2._v1.is_symplectic(result.source_basis)
    assert np.linalg.matrix_rank(result.source_basis.astype(float)) == 6
    assert "simultaneously" in " ".join(result.verification_transcript)


def test_full_completion_auxiliary_pairs_and_canonical_gram():
    sectors = (
        main_v2.RecoveredSector(0, "XXI", "ZII", "YXI"),
        main_v2.RecoveredSector(1, "IIX"),
    )
    _recovery, _grouping, result = _localize(sectors, ((0, 1),), m=3, d=2)
    assert result.success and result.source_basis.shape == (6, 6)
    assert main_v2._v1.is_symplectic(result.source_basis)
    assert main_v2._v1.is_symplectic(result.target_basis)
    assert result.K_rec == 2 and len(result.J_aux) == 1
    assert np.array_equal(
        (result.source_basis.T @ main_v2._symplectic_form_matrix(3) @ result.source_basis) % 2,
        main_v2._symplectic_form_matrix(3),
    )


def test_register_allocation_is_disjoint_complete_bounded_and_deterministic():
    sectors = (
        main_v2.RecoveredSector(0, "XIII", "ZIII", "YIII"),
        main_v2.RecoveredSector(1, "IXII"),
        main_v2.RecoveredSector(2, "IIXI"),
    )
    recovery = _manual_recovery(sectors, m=4)
    grouping = _manual_grouping(recovery, ((0, 1), (2,)), ell=3)
    first = main_v2.localize_grouped_recovery(recovery, grouping, d=3)
    second = main_v2.localize_grouped_recovery(recovery, grouping, d=3)
    assert first.success and second.success
    assert first.J_C == (((0, 1), (0, 1)), ((2,), (2,)))
    assert first.J_aux == (3,)
    assert first.J_C == second.J_C and first.J_aux == second.J_aux
    flattened = [qubit for _cluster, register in first.J_C for qubit in register]
    assert tuple(flattened) + first.J_aux == tuple(range(4))
    assert max(len(register) for _cluster, register in first.J_C) <= 3


def test_known_d_cap_and_uncertified_grouping_guards_are_honest():
    sectors = (
        main_v2.RecoveredSector(0, "XI"),
        main_v2.RecoveredSector(1, "IX"),
    )
    recovery = _manual_recovery(sectors, m=2)
    grouping = _manual_grouping(recovery, ((0, 1),), theorem=True)
    capped = main_v2.localize_grouped_recovery(recovery, grouping, d=1)
    assert not capped.success and capped.failure_reason == "cluster_exceeds_known_block_cap_d"

    uncertified = replace(grouping, theorem_grouping_preconditions_hold=False)
    rejected = main_v2.localize_grouped_recovery(recovery, uncertified, d=2)
    assert not rejected.success and rejected.failure_reason == "grouping_not_theorem_certified"
    debug = main_v2.localize_grouped_recovery(
        recovery,
        uncertified,
        d=2,
        config=main_v2.LocalizationConfig(allow_uncertified_grouping=True),
    )
    assert debug.success
    assert not debug.grouping_theorem_preconditions_hold
    assert not debug.theorem_localization_preconditions_hold


def test_tableau_dense_unitary_and_exact_manuscript_orientation():
    sectors = (
        main_v2.RecoveredSector(0, "IIX"),
        main_v2.RecoveredSector(1, "XXI", "ZII", "YXI"),
    )
    _recovery, _grouping, result = _localize(
        sectors, ((0,), (1,)), m=3, d=1
    )
    assert result.success and main_v2._v1.is_symplectic(result.residual_tableau)
    assert np.allclose(result.U_rec.conj().T @ result.U_rec, np.eye(8))
    assert main_v2.tableau_unitary_convention_holds(
        result.residual_tableau, result.U_rec
    )
    assert np.array_equal(
        (result.residual_tableau @ result.source_basis) % 2,
        result.target_basis,
    )
    inverse = main_v2._gf2_inverse_matrix(result.residual_tableau)
    assert not np.array_equal(inverse, result.residual_tableau)
    assert not main_v2.tableau_unitary_convention_holds(inverse, result.U_rec)
    assert not np.array_equal(
        (result.source_basis @ result.residual_tableau) % 2,
        result.target_basis,
    )


def test_every_generated_group_member_localizes_without_support_leakage():
    sectors = (
        main_v2.RecoveredSector(0, "XXI", "ZII", "YXI"),
        main_v2.RecoveredSector(1, "IIX"),
    )
    recovery, _grouping, result = _localize(
        sectors, ((0,), (1,)), m=3, d=1
    )
    assert result.success and result.localization_guarantee_holds
    for cluster, register in result.J_C:
        generated = main_v2.generated_cluster_pauli_group(cluster, recovery.sectors)
        for pauli in generated.paulis:
            localized = result.residual_tableau @ main_v2._v1.pauli_to_symplectic_col(pauli) % 2
            support = main_v2._residual_support(localized, recovery.m)
            assert set(support) <= set(register)
            assert not set(support).intersection(result.J_aux)
            for other_cluster, other_register in result.J_C:
                if other_cluster != cluster:
                    assert not set(support).intersection(other_register)


def test_random_clifford_image_fixture_localizes_phase_free_algebras():
    _gates, encoder_tableau, _unitary = main_v2._v1.random_clifford_gate(
        4, steps=16, seed=510
    )

    def image(qubit, axis):
        source = np.zeros(8, dtype=np.uint8)
        source[qubit if axis == "X" else 4 + qubit] = 1
        return main_v2._symplectic_column_to_pauli(encoder_tableau @ source % 2)

    x0, z0 = image(0, "X"), image(0, "Z")
    x2, z2 = image(2, "X"), image(2, "Z")
    sectors = (
        main_v2.RecoveredSector(0, x0, z0, main_v2.phase_free_pauli_product(x0, z0)),
        main_v2.RecoveredSector(1, image(1, "X")),
        main_v2.RecoveredSector(2, x2, z2, main_v2.phase_free_pauli_product(x2, z2)),
    )
    recovery, _grouping, result = _localize(
        sectors, ((0, 1), (2,)), m=4, d=2
    )
    assert result.success and result.J_aux == (3,)
    for cluster, register in result.J_C:
        for pauli in main_v2.generated_cluster_pauli_group(
            cluster, recovery.sectors
        ).paulis:
            localized = result.residual_tableau @ main_v2._v1.pauli_to_symplectic_col(pauli) % 2
            assert set(main_v2._residual_support(localized, 4)) <= set(register)


def test_full_n_extension_is_identity_on_prefix_and_localizes_residual():
    sectors = (main_v2.RecoveredSector(0, "XX"),)
    recovery, _grouping, result = _localize(
        sectors, ((0,),), m=2, t=1, d=1
    )
    assert result.success and result.full_tableau.shape == (6, 6)
    assert result.bar_U_rec.shape == (8, 8)
    assert np.allclose(result.bar_U_rec.conj().T @ result.bar_U_rec, np.eye(8))
    assert main_v2.tableau_unitary_convention_holds(
        result.full_tableau, result.bar_U_rec
    )
    prefix_x = np.zeros(6, dtype=np.uint8)
    prefix_z = np.zeros(6, dtype=np.uint8)
    prefix_x[0] = 1
    prefix_z[3] = 1
    assert np.array_equal(result.full_tableau @ prefix_x, prefix_x)
    assert np.array_equal(result.full_tableau @ prefix_z, prefix_z)
    residual_indices = (1, 2, 4, 5)
    assert np.array_equal(
        result.full_tableau[np.ix_(residual_indices, residual_indices)],
        result.residual_tableau,
    )
    assert np.allclose(result.bar_U_rec, np.kron(np.eye(2), result.U_rec))


def test_m_zero_and_no_recovered_sector_boundaries():
    recovery_zero = _manual_recovery((), m=0, t=1)
    grouping_zero = _manual_grouping(recovery_zero, (), ell=1)
    zero = main_v2.localize_grouped_recovery(recovery_zero, grouping_zero, d=1)
    assert zero.success and zero.K_rec == 0 and zero.J_aux == ()
    assert zero.residual_tableau.shape == (0, 0)
    assert zero.U_rec.shape == (1, 1) and zero.bar_U_rec.shape == (2, 2)

    recovery_aux = _manual_recovery((), m=3)
    grouping_aux = _manual_grouping(recovery_aux, (), ell=1)
    auxiliary = main_v2.localize_grouped_recovery(recovery_aux, grouping_aux, d=2)
    assert auxiliary.success and auxiliary.J_C == ()
    assert auxiliary.J_aux == (0, 1, 2)
    assert np.array_equal(auxiliary.residual_tableau, np.eye(6, dtype=np.uint8))
    assert np.allclose(auxiliary.U_rec, np.eye(8))


@pytest.mark.parametrize("d", (1, 2, 3))
def test_d1_d2_d3_and_empty_auxiliary_register(d):
    sectors = tuple(
        main_v2.RecoveredSector(
            index,
            "I" * index + "X" + "I" * (d - index - 1),
        )
        for index in range(d)
    )
    clusters = tuple((index,) for index in range(d)) if d == 1 else (tuple(range(d)),)
    _recovery, _grouping, result = _localize(
        sectors, clusters, m=d, d=d
    )
    assert result.success and result.K_rec == d and result.J_aux == ()
    assert sum(len(register) for _cluster, register in result.J_C) == d


def test_one_singleton_one_triple_and_nonempty_auxiliary():
    singleton = _localize(
        (main_v2.RecoveredSector(0, "XI"),), ((0,),), m=2, d=1
    )[2]
    triple = _localize(
        (main_v2.RecoveredSector(0, "XI", "ZI", "YI"),),
        ((0,),),
        m=2,
        d=1,
    )[2]
    assert singleton.success and triple.success
    assert singleton.structures[0].q_C == 1
    assert triple.structures[0].h_C == 1
    assert singleton.J_aux == triple.J_aux == (1,)


def test_copy_ledger_unchanged_and_no_oracle_or_state_access(monkeypatch):
    sectors = (main_v2.RecoveredSector(0, "X"),)
    recovery = _manual_recovery(sectors, m=1, copies=True)
    grouping = _manual_grouping(recovery, ((0,),))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("localization used oracle/state access")

    monkeypatch.setattr(main_v2, "debug_oracle_sector_block_labels", forbidden)
    monkeypatch.setattr(main_v2, "debug_oracle_residual_block_labels", forbidden)
    monkeypatch.setattr(main_v2, "_coerce_measurement_source", forbidden)
    result = main_v2.localize_grouped_recovery(recovery, grouping, d=1)
    assert result.success and result.localization_copy_count == 0
    assert result.cumulative_copy_ledger == grouping.cumulative_copy_ledger
    assert result.cumulative_copy_ledger.entries == (
        ("peeling_bell_pool", 20),
        ("recovery_bell_pool", 40),
        ("grouping_ordinary_pool", 0),
    )


def test_failed_localization_has_no_fake_clifford_or_register_data():
    recovery = _manual_recovery((main_v2.RecoveredSector(0, "X"),), m=1)
    grouping = _manual_grouping(recovery, ((7,),))
    result = main_v2.localize_grouped_recovery(recovery, grouping, d=1)
    assert not result.success and "unknown" in result.failure_reason
    assert result.U_rec is result.residual_tableau is result.bar_U_rec is None
    assert result.structures == result.cluster_localizations == result.J_C == ()


def test_phase5_public_api_is_exported_and_all_exports_exist():
    intended = {
        "LocalizationConfig",
        "ClusterSymplecticStructure",
        "ClusterLocalization",
        "LocalizationResult",
        "LocalizationInvariantError",
        "validate_grouping_against_recovery",
        "analyze_group_symplectic_structure",
        "localize_grouped_recovery",
    }
    assert intended <= set(main_v2.__all__)
    assert all(hasattr(main_v2, name) for name in main_v2.__all__)


def test_phase_free_localization_cannot_supply_signed_expectations():
    # S^dagger X S = -Y: the tableau correctly retains the Y direction, but
    # deliberately discards the physical minus sign that Phase 6 must measure.
    S = np.diag([1.0, 1.0j])
    X = main_v2._v1._qutip_pauli_op(1, "X").full()
    Y = main_v2._v1._qutip_pauli_op(1, "Y").full()
    actual = S.conj().T @ X @ S
    assert np.allclose(actual, -Y)
    assert not np.allclose(actual, Y)


def test_phase5_result_exposes_no_syndrome_or_tomography_fields():
    assert not hasattr(main_v2, "recover_syndrome_signs")
    assert not hasattr(main_v2, "tomograph_empirical_registers")
    fields = main_v2.LocalizationResult.__dataclass_fields__
    assert "syndrome" not in fields
    assert "tomography" not in fields
