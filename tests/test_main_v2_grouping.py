import itertools
import math
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


def _synthetic_peeling(n, stabilizers=None):
    source = main_v2.PauliScoreMap(
        n=n,
        score_map=_score_map(n, stabilizers),
        uniform_radius=0.0,
        frame="physical",
    )
    result = main_v2.certified_stabilizer_peeling_v2(
        source,
        main_v2.PeelingConfig(
            h_min=0.6, h_max=0.8, eta=0.005, return_details=True
        ),
    )
    assert result.success and result.theorem_certified
    return result


def _manual_recovery(peeling, sectors, *, theorem=True, earlier_copies=False):
    sectors = tuple(sectors)
    n = np.asarray(peeling.tableau).shape[0] // 2
    t = int(peeling.t)
    m = n - t
    copy_ledger = (
        main_v2.CopyLedger((("recovery_bell_pool", 40),))
        if earlier_copies
        else main_v2.CopyLedger()
    )
    cumulative = (
        main_v2.CopyLedger(
            (("peeling_bell_pool", 20), ("recovery_bell_pool", 40))
        )
        if earlier_copies
        else main_v2.CopyLedger()
    )
    return main_v2.RecoveryResult(
        success=True,
        failure_reason=None,
        n=n,
        t=t,
        m=m,
        theta=0.1,
        tau_rank=0.0,
        M2=20 if earlier_copies else 0,
        zeta_rank=0.05,
        lambda_=float(peeling.lambda_),
        sectors=sectors,
        independent_axes=main_v2.recovered_sector_axes(sectors),
        recovered_span_basis=main_v2.recovered_sector_span_basis(sectors, m),
        ranked_survivor_count=len(sectors),
        score_provenance=(
            main_v2.DataProvenance.EMPIRICAL
            if earlier_copies
            else main_v2.DataProvenance.EXACT
        ),
        score_frame="peeled",
        threshold_margin_holds=True,
        ranking_gap_margin_holds=True,
        theorem_recovery_preconditions_hold=theorem,
        threshold_span_complete=True,
        copy_ledger=copy_ledger,
        cumulative_copy_ledger=cumulative,
    )


def _manuscript_fixture():
    probabilities = np.array([13, 5, 5, 1, 5, 1, 1, 1], dtype=float) / 32
    rho = qt.Qobj(np.diag(probabilities), dims=[[2, 2, 2], [2, 2, 2]])
    instance = main_v2.random_cebp_state(
        3,
        3,
        block_sizes=(3,),
        block_states=(rho,),
        clifford_steps=0,
        seed=401,
    )
    peeling = _synthetic_peeling(3)
    recovery = _manual_recovery(
        peeling,
        (
            main_v2.RecoveredSector(0, "ZII"),
            main_v2.RecoveredSector(1, "IZI"),
            main_v2.RecoveredSector(2, "IIZ"),
        ),
    )
    return instance, peeling, recovery


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
def test_signed_pauli_one_qubit_table(left, right, phase, pauli):
    product = main_v2.signed_pauli_product(left, right)
    assert (product.phase_exponent, product.pauli) == (phase, pauli)
    assert main_v2.phase_free_pauli_product(left, right) == pauli


def test_signed_multiqubit_commuting_product_and_imaginary_rejection():
    assert main_v2.signed_pauli_product("XX", "ZZ") == main_v2.SignedPauliProduct(
        2, "YY"
    )
    assert main_v2.hermitian_pauli_product("XX", "ZZ") == (-1, "YY")
    assert main_v2.signed_pauli_product("XZ", "ZX") == main_v2.SignedPauliProduct(
        0, "YY"
    )
    with pytest.raises(ValueError, match="not Hermitian"):
        main_v2.hermitian_pauli_product("X", "Z")


def test_generated_cluster_groups_singleton_triple_and_merge_are_deterministic():
    sectors = (
        main_v2.RecoveredSector(2, "XI", "ZI", "YI"),
        main_v2.RecoveredSector(7, "IX"),
    )
    triple = main_v2.generated_cluster_pauli_group((2,), sectors)
    singleton = main_v2.generated_cluster_pauli_group((7,), sectors)
    merged = main_v2.generated_cluster_pauli_group((7, 2), sectors)
    assert triple.rank == 2 and triple.paulis == ("II", "XI", "YI", "ZI")
    assert singleton.rank == 1 and singleton.paulis == ("II", "IX")
    assert merged.rank == 3 and len(merged.paulis) == 2**3
    assert len(merged.paulis) <= 4**2 and "II" not in merged.nonidentity
    assert merged == main_v2.generated_cluster_pauli_group((2, 7), sectors)


def test_generated_group_guard_and_unknown_sector_are_explicit():
    sectors = (main_v2.RecoveredSector(0, "X"),)
    with pytest.raises(ValueError, match="guard"):
        main_v2.generated_cluster_pauli_group((0,), sectors, max_group_size=1)
    with pytest.raises(ValueError, match="unknown"):
        main_v2.generated_cluster_pauli_group((1,), sectors)


def test_labeled_partition_counts_and_gamma_values():
    assert [len(main_v2.labeled_set_partitions(q)) for q in range(5)] == [
        1,
        1,
        2,
        5,
        15,
    ]
    assert [main_v2.cumulant_gamma(q) for q in range(1, 5)] == [1, 3, 13, 75]


def test_beta_peel_uses_gamma_ell_and_exact_recovery_window_is_reported():
    instance, peeling, recovery = _manuscript_fixture()
    leakage = replace(peeling, epsilon_peel=0.01)
    assert main_v2.grouping_beta_peel(leakage, 3) == pytest.approx(4 * 13 * 0.1)
    result = main_v2.debug_exact_hierarchical_cumulant_grouping(
        instance,
        peeling,
        recovery,
        main_v2.GroupingConfig(
            ell_grp=3, eta_test=0.05, tau_kappa=0.01, eta_irr=0.2
        ),
    )
    assert result.exact_recovery_window_holds
    assert result.recovery_score_provenance is main_v2.DataProvenance.EXACT
    assert result.moment_provenance is main_v2.DataProvenance.EXACT


def test_labeled_cumulants_q1_q2_q3_and_repeated_operator_positions():
    assert main_v2.mixed_cumulant_from_moments(1, {(0,): 0.4}) == pytest.approx(
        0.4
    )
    covariance = main_v2.mixed_cumulant_from_moments(
        2, {(0,): 0.2, (1,): -0.3, (0, 1): 0.5}
    )
    assert covariance == pytest.approx(0.5 - 0.2 * -0.3)
    third = main_v2.mixed_cumulant_from_moments(
        3,
        {
            (0,): 0.1,
            (1,): 0.2,
            (2,): 0.3,
            (0, 1): 0.4,
            (0, 2): 0.5,
            (1, 2): 0.6,
            (0, 1, 2): 0.7,
        },
    )
    assert third == pytest.approx(
        0.7 - 0.4 * 0.3 - 0.5 * 0.2 - 0.6 * 0.1 + 2 * 0.1 * 0.2 * 0.3
    )
    repeated = main_v2.mixed_cumulant_from_moments(
        2, {(0,): 0.0, (1,): 0.0, (0, 1): 1.0}
    )
    assert repeated == 1.0


def test_product_split_synthetic_moment_family_has_zero_cumulant():
    means = (0.2, -0.4, 0.7)
    moments = {}
    for size in range(1, 4):
        for subset in itertools.combinations(range(3), size):
            moments[subset] = math.prod(means[index] for index in subset)
    assert main_v2.mixed_cumulant_from_moments(3, moments) == pytest.approx(0.0)


def test_exact_signed_subset_moment_tracks_negative_hermitian_product_sign():
    bell = (qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) + qt.tensor(
        qt.basis(2, 1), qt.basis(2, 1)
    )).unit()
    instance = main_v2.random_cebp_state(
        2, 2, block_sizes=(2,), block_states=(bell,), clifford_steps=0, seed=402
    )
    peeling = _synthetic_peeling(2)
    moments = main_v2.debug_exact_residual_subset_moments(
        instance, peeling, ("XX", "ZZ")
    )
    assert main_v2.hermitian_pauli_product("XX", "ZZ") == (-1, "YY")
    assert moments[frozenset((0, 1))] == pytest.approx(1.0)


def test_manuscript_pairwise_insufficient_exact_moments_and_cumulants():
    instance, peeling, _ = _manuscript_fixture()
    axes = ("ZII", "IZI", "IIZ")
    moments = main_v2.debug_exact_residual_subset_moments(instance, peeling, axes)
    assert [moments[frozenset((index,))] for index in range(3)] == pytest.approx(
        [0.5, 0.5, 0.5]
    )
    assert [
        moments[frozenset(pair)] for pair in itertools.combinations(range(3), 2)
    ] == pytest.approx([0.25, 0.25, 0.25])
    assert moments[frozenset((0, 1, 2))] == pytest.approx(0.0)
    for pair in itertools.combinations(axes, 2):
        assert main_v2.debug_exact_residual_mixed_cumulant(
            instance, peeling, pair
        ) == pytest.approx(0.0, abs=1e-14)
    assert main_v2.debug_exact_residual_mixed_cumulant(
        instance, peeling, axes
    ) == pytest.approx(-1 / 8)
    assert main_v2.debug_exact_residual_mixed_cumulant(
        instance, peeling, ("ZII", "IZZ")
    ) == pytest.approx(-1 / 8)


def test_pairwise_insufficient_fixture_ell2_no_merge_ell3_merges():
    instance, peeling, recovery = _manuscript_fixture()
    ell2 = main_v2.debug_exact_hierarchical_cumulant_grouping(
        instance,
        peeling,
        recovery,
        main_v2.GroupingConfig(ell_grp=2, eta_test=0, return_details=True),
    )
    ell3 = main_v2.debug_exact_hierarchical_cumulant_grouping(
        instance,
        peeling,
        recovery,
        main_v2.GroupingConfig(ell_grp=3, eta_test=0, return_details=True),
    )
    assert ell2.clusters == ((0,), (1,), (2,)) and ell2.merge_rounds == 0
    assert ell3.clusters == ((0, 1, 2),) and ell3.merge_rounds == 1
    assert [scan.order for scan in ell3.transcript[:2]] == [2, 3]
    assert ell3.hyperedge_witnesses[0].cumulant == pytest.approx(-1 / 8)


def test_overlapping_hyperedges_form_one_connected_component():
    vertices = ((0,), (1,), (2,), (3,), (4,))
    components = main_v2.hypergraph_connected_components(
        vertices,
        (((0,), (1,), (2,)), ((2,), (3,), (4,))),
    )
    assert components == (vertices,)


def test_grouping_merges_overlapping_edges_as_one_component():
    peeling = _synthetic_peeling(3)
    axes = ("XII", "IXI", "IIX")
    recovery = _manual_recovery(
        peeling,
        tuple(main_v2.RecoveredSector(index, axis) for index, axis in enumerate(axes)),
    )

    def cumulant(observables):
        return 1.0 if frozenset(observables) in {
            frozenset(("XII", "IXI")),
            frozenset(("IXI", "IIX")),
        } else 0.0

    result = main_v2.hierarchical_cumulant_grouping(
        recovery,
        peeling,
        main_v2.CallableResidualCumulantInterface(cumulant),
        main_v2.GroupingConfig(ell_grp=3, return_details=True),
    )
    assert result.clusters == ((0, 1, 2),) and result.merge_rounds == 1
    assert result.transcript[0].connected_components == (((0,), (1,), (2,)),)


def test_required_reset_to_q2_fixture_merges_new_generated_product_with_d():
    peeling = _synthetic_peeling(4)
    recovery = _manual_recovery(
        peeling,
        tuple(
            main_v2.RecoveredSector(index, "I" * index + "X" + "I" * (3 - index))
            for index in range(4)
        ),
    )

    def cumulant(observables):
        if len(observables) == 3 and set(observables) == {"XIII", "IXII", "IIXI"}:
            return 1.0
        if len(observables) == 2 and set(observables) == {"XXXI", "IIIX"}:
            return 1.0
        return 0.0

    result = main_v2.hierarchical_cumulant_grouping(
        recovery,
        peeling,
        main_v2.CallableResidualCumulantInterface(cumulant),
        main_v2.GroupingConfig(ell_grp=4, eta_test=0, return_details=True),
    )
    assert result.clusters == ((0, 1, 2, 3),)
    assert [scan.order for scan in result.transcript[:4]] == [2, 3, 2, 2]
    reset_scan = result.transcript[1]
    assert reset_scan.reset_to_two and reset_scan.partition_after == ((0, 1, 2), (3,))
    assert any(
        witness.order == 2 and set(witness.observables) == {"XXXI", "IIIX"}
        for witness in result.hyperedge_witnesses
    )
    assert all(
        len(cluster) < result.ell_grp
        for scan in result.transcript
        for cluster in scan.active_clusters
    )
    assert all(
        len(set().union(*(set(cluster) for cluster in witness.clusters)))
        <= result.ell_grp
        and all(set(observable) != {"I"} for observable in witness.observables)
        for witness in result.hyperedge_witnesses
    )


@pytest.mark.parametrize("density_backend", (False, True))
def test_joint_measurement_handles_nonqubitwise_commuting_tuple_and_backends(
    density_backend,
):
    bell = (qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) + qt.tensor(
        qt.basis(2, 1), qt.basis(2, 1)
    )).unit()
    state = bell.proj() if density_backend else bell
    instance = main_v2.random_cebp_state(
        2, 2, block_sizes=(2,), block_states=(state,), clifford_steps=0, seed=403
    )
    peeling = _synthetic_peeling(2)
    record = main_v2.measure_signed_pauli_tuple(
        instance.learner_view(), peeling, ("XX", "ZZ"), 200, seed=404
    )
    assert record.outcomes.shape == (200, 2)
    assert set(np.unique(record.outcomes)) <= {-1, 1}
    assert record.copies == 200 and record.frame == "peeled_residual"
    assert record.subset_moment((0, 1)) == pytest.approx(1.0)
    assert len(record.all_subset_moments()) == 4


def test_joint_measurement_rejects_noncommuting_tuple():
    instance = main_v2.random_cebp_state(1, 1, clifford_steps=0, seed=405)
    peeling = _synthetic_peeling(1)
    with pytest.raises(ValueError, match="commute"):
        main_v2.measure_signed_pauli_tuple(
            instance.learner_view(), peeling, ("X", "Z"), 10, seed=406
        )


def test_empirical_interface_uses_fresh_distinct_batches_and_exact_tuple_cache():
    instance, peeling, _ = _manuscript_fixture()
    interface = main_v2.EmpiricalOrdinaryCumulantInterface(
        instance.learner_view(), peeling, tau_kappa=0.4, delta_tuple=0.01, seed=407
    )
    first = interface.query(("ZII", "IZI"))
    assert interface.query(("ZII", "IZI")) == first
    interface.query(("ZII", "IIZ"))
    assert interface.realized_query_count == 2 and len(interface.records) == 2
    assert interface.records[0].seed != interface.records[1].seed
    assert interface.realized_copies == sum(record.shots for record in interface.records)
    assert abs(first) <= 0.4


def test_every_realized_empirical_grouping_query_is_uniformly_close_in_seeded_fixture():
    instance, peeling, recovery = _manuscript_fixture()
    config = main_v2.GroupingConfig(
        ell_grp=2, eta_test=0.75, tau_kappa=0.5, return_details=True
    )
    ntest = main_v2.adaptive_cumulant_test_bound(len(recovery.sectors), 2)
    empirical = main_v2.EmpiricalOrdinaryCumulantInterface(
        instance.learner_view(),
        peeling,
        tau_kappa=config.tau_kappa,
        delta_tuple=config.delta_grp_ordinary / ntest,
        seed=4071,
    )
    result = main_v2.hierarchical_cumulant_grouping(
        recovery, peeling, empirical, config
    )
    exact = main_v2.DebugExactResidualCumulantInterface(instance, peeling)
    assert result.success and empirical.queried_values
    assert all(
        abs(estimate - exact.query(observables)) <= config.tau_kappa
        for observables, estimate in empirical.queried_values.items()
    )


def test_sample_budget_query_bound_delta_allocation_and_copy_upper_bound():
    q, tau, delta = 3, 0.2, 0.01
    expected = math.ceil(
        2
        * main_v2.cumulant_gamma(q) ** 2
        / tau**2
        * math.log(2 * (2**q - 1) / delta)
    )
    assert main_v2.ordinary_cumulant_sample_count(q, tau, delta) == expected
    assert tau / main_v2.cumulant_gamma(q) == pytest.approx(tau / 13)
    L, ell = 3, 3
    ntest = L * 4**ell * sum(math.comb(L, order) for order in (2, 3))
    assert main_v2.adaptive_cumulant_test_bound(L, ell) == ntest
    bound = main_v2.ordinary_grouping_copy_upper_bound(ntest, ell, tau, 0.05)
    assert bound == ntest * main_v2.ordinary_cumulant_sample_count(
        ell, tau, 0.05 / ntest
    )


def test_empirical_recovery_and_strict_margin_guards_and_override_flags():
    instance, peeling, recovery = _manuscript_fixture()
    manual = replace(recovery, theorem_recovery_preconditions_hold=False)
    valid_margin = main_v2.GroupingConfig(ell_grp=2, eta_test=0.75, tau_kappa=0.5)
    rejected = main_v2.empirical_hierarchical_cumulant_grouping(
        instance.learner_view(), peeling, manual, valid_margin, seed=408
    )
    assert not rejected.success and rejected.failure_reason == "recovery_not_theorem_calibrated"
    override = main_v2.empirical_hierarchical_cumulant_grouping(
        instance.learner_view(),
        peeling,
        manual,
        replace(valid_margin, allow_uncalibrated_recovery=True),
        seed=408,
    )
    assert override.success and not override.recovery_theorem_preconditions_hold
    assert not override.theorem_grouping_preconditions_hold
    equality = main_v2.empirical_hierarchical_cumulant_grouping(
        instance.learner_view(),
        peeling,
        recovery,
        main_v2.GroupingConfig(ell_grp=2, eta_test=0.25, tau_kappa=0.25),
        seed=409,
    )
    assert not equality.success and equality.failure_reason == "no_false_merge_margin_failed"


def test_guessed_scale_schedule_strict_condition_and_xi():
    instance, peeling, recovery = _manuscript_fixture()
    config = main_v2.GroupingConfig.from_guessed_scale(2, 2.0)
    result = main_v2.empirical_hierarchical_cumulant_grouping(
        instance.learner_view(), peeling, recovery, config, seed=410
    )
    assert result.success and result.no_false_merge_condition_holds
    assert result.theorem_grouping_preconditions_hold
    assert (result.eta_test, result.tau_kappa, result.xi_s) == (1.0, 0.5, 1.5)
    equality_epsilon = (2.0 / (16 * main_v2.cumulant_gamma(2))) ** 2
    equality_peeling = replace(peeling, epsilon_peel=equality_epsilon)
    equality = main_v2.empirical_hierarchical_cumulant_grouping(
        instance.learner_view(), equality_peeling, recovery, config, seed=411
    )
    assert not equality.success and equality.failure_reason == "no_false_merge_margin_failed"


@pytest.mark.parametrize(
    ("d", "block_sizes", "sectors"),
    (
        (2, (1, 1), ("ZI", "IZ")),
        (3, (2, 1), ("ZII", "IZI", "IIZ")),
    ),
)
def test_empirical_no_false_cross_block_merge_and_oracle_separation(
    d, block_sizes, sectors, monkeypatch
):
    block_states = tuple(qt.qeye(2**size) / (2**size) for size in block_sizes)
    instance = main_v2.random_cebp_state(
        sum(block_sizes),
        d,
        block_sizes=block_sizes,
        block_states=block_states,
        clifford_steps=0,
        seed=420 + d,
    )
    peeling = _synthetic_peeling(sum(block_sizes))
    recovery = _manual_recovery(
        peeling,
        tuple(main_v2.RecoveredSector(index, pauli) for index, pauli in enumerate(sectors)),
    )
    labels = main_v2.debug_oracle_sector_block_labels(instance, peeling, recovery)
    cross_pair = next(
        (left, right)
        for left in range(len(sectors))
        for right in range(left + 1, len(sectors))
        if labels[left] != labels[right]
    )
    assert main_v2.debug_exact_residual_mixed_cumulant(
        instance,
        peeling,
        (sectors[cross_pair[0]], sectors[cross_pair[1]]),
    ) == pytest.approx(0.0, abs=1e-14)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("learner grouping used oracle truth")

    monkeypatch.setattr(main_v2, "debug_oracle_sector_block_labels", forbidden)
    monkeypatch.setattr(main_v2, "debug_oracle_residual_block_labels", forbidden)
    result = main_v2.empirical_hierarchical_cumulant_grouping(
        instance.learner_view(),
        peeling,
        recovery,
        main_v2.GroupingConfig(
            ell_grp=d, eta_test=0.75, tau_kappa=0.5, return_details=True
        ),
        seed=430 + d,
    )
    assert result.success and result.no_false_merge_condition_holds
    for cluster in result.clusters:
        block_labels = {labels[sector_id] for sector_id in cluster}
        assert len(block_labels) == 1


def test_empirical_copy_ledger_uses_one_copy_per_shot_and_preserves_prior_pools():
    probabilities = np.array([0.4, 0.1, 0.1, 0.4])
    rho = qt.Qobj(np.diag(probabilities), dims=[[2, 2], [2, 2]])
    instance = main_v2.random_cebp_state(
        2, 2, block_sizes=(2,), block_states=(rho,), clifford_steps=0, seed=440
    )
    peeling = _synthetic_peeling(2)
    recovery = _manual_recovery(
        peeling,
        (main_v2.RecoveredSector(0, "ZI"), main_v2.RecoveredSector(1, "IZ")),
        earlier_copies=True,
    )
    result = main_v2.empirical_hierarchical_cumulant_grouping(
        instance.learner_view(),
        peeling,
        recovery,
        main_v2.GroupingConfig(ell_grp=2, eta_test=0.75, tau_kappa=0.5),
        seed=441,
    )
    assert result.success and result.realized_query_count == 1
    assert result.realized_grouping_copies == dict(result.copies_by_order)[2]
    assert result.grouping_copy_ledger.as_dict() == {
        "grouping_ordinary_pool": result.realized_grouping_copies
    }
    assert result.cumulative_copy_ledger.total == (
        20 + 40 + result.realized_grouping_copies
    )
    assert result.delta_tuple == pytest.approx(
        result.grouping_copy_ledger.total * 0 + 0.05 / result.N_test_max
    )
    assert result.realized_grouping_copies <= result.ordinary_copy_upper_bound


def test_trivial_l1_ell1_and_tn_grouping_are_zero_copy():
    instance = main_v2.random_cebp_state(1, 1, clifford_steps=0, seed=450)
    peeling = _synthetic_peeling(1)
    one = _manual_recovery(peeling, (main_v2.RecoveredSector(0, "Z"),))
    empirical = main_v2.empirical_hierarchical_cumulant_grouping(
        instance.learner_view(),
        peeling,
        one,
        main_v2.GroupingConfig(eta_test=0.75, tau_kappa=0.5),
        seed=451,
    )
    assert empirical.ell_grp == instance.d == 1
    assert empirical.clusters == ((0,),) and empirical.realized_grouping_copies == 0
    assert empirical.realized_query_count == 0 and empirical.N_test_max == 0

    peeled_all = _synthetic_peeling(1, {"Z": 1.0})
    empty = _manual_recovery(peeled_all, ())
    exact = main_v2.debug_exact_hierarchical_cumulant_grouping(
        instance,
        peeled_all,
        empty,
        main_v2.GroupingConfig(ell_grp=1),
    )
    assert exact.clusters == () and exact.realized_grouping_copies == 0


def test_empirical_trivial_grouping_bypasses_failed_numeric_margin():
    one_qubit = main_v2.random_cebp_state(1, 1, clifford_steps=0, seed=452)
    peeling = _synthetic_peeling(1)
    failed_margin = main_v2.GroupingConfig(
        ell_grp=2, eta_test=0.0, tau_kappa=0.0
    )

    empty = _manual_recovery(peeling, ())
    empty_result = main_v2.empirical_hierarchical_cumulant_grouping(
        one_qubit.learner_view(), peeling, empty, failed_margin, seed=453
    )
    one = _manual_recovery(peeling, (main_v2.RecoveredSector(0, "Z"),))
    one_result = main_v2.empirical_hierarchical_cumulant_grouping(
        one_qubit.learner_view(), peeling, one, failed_margin, seed=454
    )
    for result, clusters in ((empty_result, ()), (one_result, ((0,),))):
        assert result.success and result.clusters == clusters
        assert result.no_false_merge_condition_holds
        assert result.theorem_grouping_preconditions_hold
        assert result.realized_query_count == result.realized_grouping_copies == 0
        assert result.grouping_copy_ledger.as_dict() == {"grouping_ordinary_pool": 0}
        assert result.merge_rounds == 0

    two_qubit = main_v2.random_cebp_state(2, 1, clifford_steps=0, seed=455)
    peeling_two = _synthetic_peeling(2)
    two = _manual_recovery(
        peeling_two,
        (main_v2.RecoveredSector(0, "ZI"), main_v2.RecoveredSector(1, "IZ")),
    )
    ell_one = main_v2.empirical_hierarchical_cumulant_grouping(
        two_qubit.learner_view(),
        peeling_two,
        two,
        main_v2.GroupingConfig(ell_grp=1, eta_test=0.0, tau_kappa=0.0),
        seed=456,
    )
    default_d_one = main_v2.empirical_hierarchical_cumulant_grouping(
        two_qubit.learner_view(),
        peeling_two,
        two,
        main_v2.GroupingConfig(eta_test=0.0, tau_kappa=0.0),
        seed=457,
    )
    for result in (ell_one, default_d_one):
        assert result.success and result.clusters == ((0,), (1,))
        assert result.ell_grp == 1 and result.no_false_merge_condition_holds
        assert result.realized_query_count == result.realized_grouping_copies == 0
        assert result.merge_rounds == 0

    nontrivial = main_v2.empirical_hierarchical_cumulant_grouping(
        two_qubit.learner_view(),
        peeling_two,
        two,
        main_v2.GroupingConfig(ell_grp=2, eta_test=0.0, tau_kappa=0.0),
        seed=458,
    )
    assert not nontrivial.success
    assert nontrivial.failure_reason == "no_false_merge_margin_failed"


def test_validate_grouping_against_recovery_strict_handoff():
    instance, peeling, recovery = _manuscript_fixture()
    valid = main_v2.debug_exact_hierarchical_cumulant_grouping(
        instance,
        peeling,
        recovery,
        main_v2.GroupingConfig(ell_grp=3, eta_test=1.0),
    )
    assert main_v2.validate_grouping_against_recovery(valid, recovery)

    unknown = replace(valid, clusters=((0,), (1,), (7,)))
    with pytest.raises(ValueError, match="unknown"):
        main_v2.validate_grouping_against_recovery(unknown, recovery)

    missing = replace(valid)
    object.__setattr__(missing, "clusters", ((0,), (1,)))
    with pytest.raises(ValueError, match="omits"):
        main_v2.validate_grouping_against_recovery(missing, recovery)

    duplicate = replace(valid)
    object.__setattr__(duplicate, "clusters", ((0,), (1,), (1,)))
    with pytest.raises(ValueError, match="more than one"):
        main_v2.validate_grouping_against_recovery(duplicate, recovery)

    mismatched_l = replace(valid, L=1, clusters=((0,),))
    with pytest.raises(ValueError, match="does not match"):
        main_v2.validate_grouping_against_recovery(mismatched_l, recovery)

    oversized = replace(valid, ell_grp=2, clusters=((0, 1, 2),))
    with pytest.raises(ValueError, match="exceeds ell_grp"):
        main_v2.validate_grouping_against_recovery(oversized, recovery)

    failed_grouping = replace(
        valid,
        success=False,
        failure_reason="synthetic_failure",
        theorem_grouping_preconditions_hold=False,
        clusters=(),
    )
    with pytest.raises(main_v2.RecoveryPreconditionError, match="grouping_failed"):
        main_v2.validate_grouping_against_recovery(failed_grouping, recovery)

    failed_recovery = replace(
        recovery,
        success=False,
        failure_reason="synthetic_failure",
        sectors=(),
        independent_axes=(),
        recovered_span_basis=(),
    )
    with pytest.raises(main_v2.RecoveryPreconditionError, match="recovery_failed"):
        main_v2.validate_grouping_against_recovery(valid, failed_recovery)


def test_exact_grouping_retains_prior_ledgers_but_adds_zero_grouping_copies():
    instance, peeling, recovery = _manuscript_fixture()
    recovery = replace(
        recovery,
        copy_ledger=main_v2.CopyLedger((("recovery_bell_pool", 40),)),
        cumulative_copy_ledger=main_v2.CopyLedger(
            (("peeling_bell_pool", 20), ("recovery_bell_pool", 40))
        ),
    )
    result = main_v2.debug_exact_hierarchical_cumulant_grouping(
        instance, peeling, recovery, main_v2.GroupingConfig(ell_grp=2)
    )
    assert result.grouping_copy_ledger.as_dict() == {"grouping_ordinary_pool": 0}
    assert result.cumulative_copy_ledger.total == 60


def test_phase4_result_has_no_localization_syndrome_or_tomography_outputs():
    instance, peeling, recovery = _manuscript_fixture()
    result = main_v2.debug_exact_hierarchical_cumulant_grouping(
        instance, peeling, recovery, main_v2.GroupingConfig(ell_grp=2)
    )
    assert not hasattr(result, "U_rec")
    assert not hasattr(result, "J_C") and not hasattr(result, "J_aux")
    assert not hasattr(main_v2, "recover_syndrome_signs")
    assert not hasattr(main_v2, "tomograph_empirical_registers")
