import math
from dataclasses import replace

import numpy as np
import pytest
import qutip as qt

import main_v2


def _source(state, n):
    qobj = main_v2._v1._state_to_qobj(state, n)
    return main_v2.SimulatorMeasurementSource(n, qobj.isket, qobj)


def _peeling(n, t=0, *, epsilon_peel=0.0, theorem=True):
    generators = tuple("I" * j + "Z" + "I" * (n - j - 1) for j in range(t))
    return main_v2.PeelingResult(
        True,
        None,
        0.75,
        0.25,
        0.001,
        t,
        generators,
        tuple(tuple(main_v2._v1.pauli_to_symplectic_col(g)) for g in generators),
        (),
        generators,
        generators,
        (),
        (),
        np.eye(2**n, dtype=complex),
        np.eye(2 * n, dtype=np.uint8),
        (),
        epsilon_peel,
        10,
        main_v2.CopyLedger((("peeling_bell_pool", 20),)),
        main_v2.DataProvenance.EMPIRICAL,
        "physical",
        (0.6, 0.8),
        (),
        theorem,
        theorem,
    )


def _recovery(n, t, theta, tau, *, theorem=True, sectors=()):
    m = n - t
    axes = main_v2.recovered_sector_axes(sectors)
    span = main_v2.recovered_sector_span_basis(sectors, m)
    return main_v2.RecoveryResult(
        True,
        None,
        n,
        t,
        m,
        theta,
        tau,
        20,
        0.02,
        0.25,
        tuple(sectors),
        axes,
        span,
        len(sectors),
        main_v2.DataProvenance.EMPIRICAL,
        "peeled_full",
        theorem,
        theorem,
        theorem,
        True,
        main_v2.CopyLedger((("recovery_bell_pool", 40),)),
        main_v2.CopyLedger(
            (("peeling_bell_pool", 20), ("recovery_bell_pool", 40))
        ),
    )


def _grouping(recovery, peeling, clusters, *, eta_s=None, exact=False, theorem=True):
    ell = 1 if eta_s is None else 2
    beta = 0.0 if peeling.epsilon_peel == 0 else main_v2.grouping_beta_peel(peeling, ell)
    return main_v2.GroupingResult(
        True,
        None,
        len(recovery.sectors),
        ell,
        tuple(clusters),
        0.0 if eta_s is None else eta_s / 2,
        0.0 if eta_s is None else eta_s / 4,
        beta,
        eta_s,
        None if eta_s is None else 3 * eta_s / 4,
        theorem,
        None,
        theorem,
        theorem,
        main_v2.DataProvenance.EMPIRICAL,
        main_v2.DataProvenance.EXACT if exact else main_v2.DataProvenance.EMPIRICAL,
        0,
        (),
        0,
        (),
        0,
        0,
        None,
        0,
        main_v2.CopyLedger((("grouping_ordinary_pool", 0),)),
        main_v2.CopyLedger(
            (("peeling_bell_pool", 20), ("recovery_bell_pool", 40), ("grouping_ordinary_pool", 0))
        ),
        0,
    )


def _localization(n, t, d, registers, *, theorem=True, U_rec=None):
    m = n - t
    U_rec = np.eye(2**m, dtype=complex) if U_rec is None else np.asarray(U_rec)
    bar = np.kron(np.eye(2**t), U_rec)
    used = {q for _cluster, register in registers for q in register}
    return main_v2.LocalizationResult(
        True,
        None,
        n,
        t,
        m,
        d,
        tuple(cluster for cluster, _register in registers),
        (),
        (),
        tuple((tuple(cluster), tuple(register)) for cluster, register in registers),
        tuple(q for q in range(m) if q not in used),
        len(used),
        np.eye(2 * m, dtype=np.uint8),
        np.eye(2 * m, dtype=np.uint8),
        np.eye(2 * m, dtype=np.uint8),
        np.eye(2 * m, dtype=np.uint8),
        np.eye(2 * n, dtype=np.uint8),
        (),
        U_rec,
        bar,
        theorem,
        theorem,
        True,
        True,
        True,
        True,
        True,
        True,
        0,
        main_v2.CopyLedger(
            (("peeling_bell_pool", 20), ("recovery_bell_pool", 40), ("grouping_ordinary_pool", 0))
        ),
    )


def test_phase7_public_api_and_all_integrity():
    expected = {
        "EndToEndConfig",
        "EndToEndSchedule",
        "WorstCaseCopyReservation",
        "StructuralCertificate",
        "EndToEndCertificate",
        "CompactCEBPEstimator",
        "EndToEndResult",
        "calibrated_end_to_end_schedule",
        "compute_structural_certificate",
        "validate_recovered_block_tomography_handoff",
        "materialize_compact_cebp_estimator",
        "full_cebp_tomography",
    }
    assert expected <= set(main_v2.__all__)
    assert all(hasattr(main_v2, name) for name in main_v2.__all__)
    assert not any(name.startswith("_") for name in main_v2.__all__)


def test_generic_schedule_exact_formulas_and_no_measurement(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("schedule construction measured the state")

    monkeypatch.setattr(main_v2, "sample_bell_scores", forbidden)
    n, d, epsilon, delta = 3, 2, 0.4, 0.2
    schedule = main_v2.calibrated_end_to_end_schedule(n, d, epsilon, delta)
    R, A, q = n + 1, n * 2**d, 2**d - d - 1
    gamma = main_v2.cumulant_gamma(d)
    theta = epsilon**2 / (128 * R**2 * A**2)
    eta = math.expm1(math.log1p(epsilon / (8 * R * A)) / q)
    lambda0 = min(theta / 4, epsilon**2 / (128 * n * R**2), eta**2 / (512 * n * gamma**2))
    assert schedule.branch == "generic_d_ge_2"
    assert (schedule.R_ub, schedule.A_d, schedule.q_d) == (R, A, q)
    assert schedule.epsilon_tom == pytest.approx(epsilon / 4)
    assert schedule.theta_0 == pytest.approx(theta)
    assert schedule.eta_s == pytest.approx(eta)
    assert schedule.lambda_0 == pytest.approx(lambda0)
    assert schedule.h_min == pytest.approx(1 - lambda0)
    assert schedule.h_max == pytest.approx(1 - lambda0 / 2)
    assert schedule.eta_test == pytest.approx(eta / 2)
    assert schedule.tau_kappa == pytest.approx(eta / 4)
    assert all(value == pytest.approx(delta / 5) for value in (
        schedule.zeta_peel, schedule.zeta_rank, schedule.zeta_grp,
        schedule.zeta_sgn, schedule.zeta_tom,
    ))
    ntest = n * 4**d * sum(math.comb(n, order) for order in range(2, d + 1))
    assert schedule.M1 == math.ceil(
        512 * (2 * n + 1) ** 2 / lambda0**2 * math.log(2 * 4**n / (delta / 5))
    )
    assert schedule.M2 == math.ceil(
        128 / (lambda0**2 * theta**2) * math.log(2 * 4**n / (delta / 5))
    )
    assert schedule.M_sgn == math.ceil(
        2 / schedule.h_min * math.log(2 * n / (delta / 5))
    )
    assert schedule.N_test_max_wc == ntest
    assert schedule.grouping_per_query_copies == math.ceil(
        32 * gamma**2 / eta**2
        * math.log(2 * (2**d - 1) * ntest / (delta / 5))
    )
    assert schedule.N_grp_wc == ntest * schedule.grouping_per_query_copies
    assert schedule.N_bp_wc == schedule.N_P_wc * schedule.M_P_wc
    assert schedule.reservation.total == (
        2 * schedule.M1 + 2 * schedule.M2 + schedule.N_grp_wc
        + n * schedule.M_sgn + schedule.N_bp_wc
    )


def test_d3_schedule_and_reservation_are_finite_and_data_independent():
    schedule = main_v2.calibrated_end_to_end_schedule(3, 3, 0.7, 0.2)
    assert schedule.branch == "generic_d_ge_2"
    assert schedule.q_d == 4 and schedule.ell_grp == 3
    assert schedule.N_test_max_wc == 3 * 4**3 * (
        math.comb(3, 2) + math.comb(3, 3)
    )
    assert all(value >= 0 for value in schedule.reservation.as_dict().values())
    assert schedule.reservation.total == sum(schedule.reservation.as_dict().values())


def test_schedule_strict_margins_and_calibrated_certificate():
    schedule = main_v2.calibrated_end_to_end_schedule(2, 2, 0.5, 0.1)
    assert schedule.tau_1 <= schedule.lambda_0 / (16 * (2 * schedule.n + 1))
    assert schedule.tau_rank <= schedule.lambda_0 * schedule.theta_0 / 8
    assert schedule.theta - schedule.tau_rank > schedule.lambda_0
    assert schedule.lambda_0 / 2 * (schedule.theta - schedule.tau_rank) - 2 * schedule.tau_rank > 0
    peeling = _peeling(2, epsilon_peel=2 * schedule.lambda_0 / 2)
    recovery = _recovery(2, 0, schedule.theta, schedule.tau_rank)
    grouping = _grouping(recovery, peeling, (), eta_s=schedule.eta_s)
    localization = _localization(2, 0, 2, ())
    structural = main_v2.compute_structural_certificate(schedule, peeling, recovery, grouping, localization)
    ledger = main_v2.CopyLedger((
        ("peeling_bell_pool", 0), ("recovery_bell_pool", 0),
        ("grouping_ordinary_pool", 0), ("syndrome_sign_pool", 0),
        ("block_tomography_pool", 0),
    ))
    certificate = main_v2.build_end_to_end_certificate(
        schedule, structural,
        theorem_preconditions={"all": True},
        realized_copy_ledger=ledger,
        operational_success=True,
    )
    assert certificate.certified_trace_norm_bound <= schedule.epsilon
    assert certificate.total_failure_bound == pytest.approx(schedule.delta)
    assert certificate.theorem_certified


def test_structural_certificate_guessed_exact_fallback_and_d1_formulas():
    schedule = main_v2.calibrated_end_to_end_schedule(2, 2, 0.5, 0.1)
    peeling = _peeling(2, epsilon_peel=0.01)
    sectors = (main_v2.RecoveredSector(0, "XI"),)
    recovery = _recovery(2, 0, schedule.theta, 0.02, sectors=sectors)
    grouping = _grouping(recovery, peeling, ((0,),), eta_s=schedule.eta_s)
    localization = _localization(2, 0, 2, (((0,), (0,)),))
    cert = main_v2.compute_structural_certificate(schedule, peeling, recovery, grouping, localization)
    assert cert.theta_rec == pytest.approx(schedule.theta + 0.02)
    assert cert.E_peel == pytest.approx(0.2)
    assert cert.E_miss == pytest.approx(2 * 4 * math.sqrt(cert.theta_rec))
    assert cert.xi_eff == pytest.approx(cert.xi_s + cert.beta_peel)
    assert cert.F_d_xi_eff == pytest.approx((1 + cert.xi_eff) ** schedule.q_d - 1)
    assert cert.E_split_cert == pytest.approx(cert.split_terms[0][2])
    exact = main_v2.compute_structural_certificate(
        schedule, peeling, recovery,
        _grouping(recovery, peeling, ((0,),), exact=True), localization,
    )
    assert exact.branch == "exact_grouping" and exact.E_split_cert == 0
    fallback = main_v2.compute_structural_certificate(
        schedule, _peeling(2, epsilon_peel=1.0), recovery, grouping, localization
    )
    assert fallback.branch == "trivial_fallback" and fallback.E_struct_cert == 2
    d1 = main_v2.calibrated_end_to_end_schedule(2, 1, 0.5, 0.1)
    d1cert = main_v2.compute_structural_certificate(d1, peeling, recovery, grouping, localization)
    assert d1cert.branch == "d1_specialized"
    assert d1cert.E_split_cert == 0
    assert d1cert.E_struct_cert == pytest.approx(
        2 * math.sqrt(0.01) + 2 * math.sqrt(3 * (d1.theta + 0.02))
    )


def test_compact_localized_projector_and_decode_orientation():
    zero = qt.ket2dm(qt.basis(2, 0))
    H1 = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
    H = main_v2._v1._two_qubit_cnot(2, 0, 1)
    residual_rotation = np.kron(np.eye(2), H1)
    compact = main_v2.CompactCEBPEstimator(
        2, 1, 1, H, residual_rotation, (1,), (((0,), (0,), zero),), (),
    )
    localized = main_v2._localized_estimator_from_compact(compact)
    assert np.allclose(localized.ptrace(0).full(), qt.ket2dm(qt.basis(2, 1)).full())
    assert np.allclose(localized.ptrace(1).full(), zero.full())
    decoded = main_v2.materialize_compact_cebp_estimator(compact, max_dense_qubits=2)
    decoder = H @ residual_rotation
    expected = decoder @ localized.full() @ decoder.conj().T
    reversed_order = residual_rotation @ H
    wrong = reversed_order @ localized.full() @ reversed_order.conj().T
    assert np.allclose(decoded.full(), expected)
    assert not np.allclose(decoded.full(), wrong)
    assert decoded.isherm and decoded.tr() == pytest.approx(1)


def test_compact_t0_and_m0_boundary_conventions():
    one = qt.ket2dm(qt.basis(2, 1))
    t0 = main_v2.CompactCEBPEstimator(
        1, 0, 1, np.eye(2), np.eye(2), (), (((0,), (0,), one),), ()
    )
    assert np.allclose(main_v2._localized_estimator_from_compact(t0).full(), one.full())
    m0 = main_v2.CompactCEBPEstimator(
        1, 1, 0, np.eye(2), np.eye(2), (1,), (), ()
    )
    expected = qt.ket2dm(qt.basis(2, 1))
    assert np.allclose(main_v2._localized_estimator_from_compact(m0).full(), expected.full())


def _manual_config(d1=False):
    kwargs = dict(
        epsilon=0.5,
        delta=0.1,
        seed=7,
        materialize_dense_estimator=True,
        allow_uncertified_execution=True,
        peeling_override=main_v2.PeelingConfig(0.6, 0.8, 0.02, 50, 0.1),
        recovery_override=main_v2.RecoveryConfig(
            0.5, 50, 0.1,
            allow_uncalibrated_peeling=True,
            allow_margin_failure=True,
        ),
        syndrome_override=main_v2.SyndromeConfig(0.1, 0.6, M_sgn=5),
    )
    if d1:
        kwargs.update(d1_accepted_per_setting_override=5, d1_attempts_per_setting_override=20)
    else:
        kwargs.update(
            grouping_override=main_v2.GroupingConfig.from_guessed_scale(
                2, 1.0, delta_grp_ordinary=0.1,
                allow_uncalibrated_recovery=True,
                allow_no_false_merge_margin_failure=True,
            ),
            tomography_override=main_v2.TomographyConfig(
                2.0, 0.1, allow_uncertified_localization=True, max_dense_qubits=4
            ),
        )
    return main_v2.EndToEndConfig(**kwargs)


def test_full_generic_manual_execution_seed_ledger_copy_ledger_and_oracle_isolation(monkeypatch):
    instance = main_v2.random_cebp_state(
        2, 2, block_sizes=(2,), block_states=(qt.qeye(4) / 4,), clifford_steps=0, seed=1
    )
    for name in ("debug_oracle_sector_block_labels", "debug_exact_localized_state"):
        monkeypatch.setattr(main_v2, name, lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("oracle used")))
    first = main_v2.full_cebp_tomography(instance.learner_view(), config=_manual_config())
    second = main_v2.full_cebp_tomography(instance.learner_view(), config=_manual_config())
    assert first.success and second.success
    assert first.branch == "generic_d_ge_2" and not first.theorem_certified
    seeds = first.seed_ledger
    assert len({seeds.peeling_seed, seeds.syndrome_seed, seeds.recovery_seed, seeds.grouping_seed, seeds.tomography_seed}) == 5
    assert first.realized_copy_ledger.entries == second.realized_copy_ledger.entries
    assert first.phase6_handoff is not None
    assert main_v2.validate_recovered_block_tomography_handoff(
        first.peeling, first.localization, first.phase6_handoff
    )
    assert first.realized_total == 200
    assert first.reservation_slack >= 0
    assert first.decoded_density.isherm and first.decoded_density.tr() == pytest.approx(1)


def test_generic_pipeline_measures_nonempty_syndrome_exactly_once(monkeypatch):
    zero = qt.ket2dm(qt.basis(2, 0))
    instance = main_v2.random_cebp_state(
        2, 2, block_sizes=(2,), block_states=(qt.tensor(zero, qt.qeye(2) / 2),),
        clifford_steps=0, seed=1,
    )
    config = replace(
        _manual_config(),
        peeling_override=main_v2.PeelingConfig(0.6, 0.8, 0.02, 100, 0.1),
        syndrome_override=main_v2.SyndromeConfig(0.1, 0.6, M_sgn=7, return_details=True),
    )
    original = main_v2.measure_pauli_expectation
    calls = []

    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(main_v2, "measure_pauli_expectation", counted)
    result = main_v2.full_cebp_tomography(instance.learner_view(), config=config)
    assert result.success and result.peeling.t == 1
    assert len(calls) == 1 and result.syndrome.syndrome_sign_pool == 7
    assert result.realized_copy_ledger.as_dict()["syndrome_sign_pool"] == 7


def test_d1_schedule_is_separate_and_full_branch_has_zero_grouping_copies():
    schedule = main_v2.calibrated_end_to_end_schedule(2, 1, 0.5, 0.1)
    assert schedule.branch == "d1_specialized"
    assert schedule.theta == pytest.approx(0.5**2 / (128 * 2**2))
    assert schedule.lambda_0 == pytest.approx(schedule.theta / 4)
    assert schedule.zeta_grp == 0 and schedule.N_grp_wc == 0
    assert sum((schedule.zeta_peel, schedule.zeta_rank, schedule.zeta_sgn, schedule.zeta_tom)) == pytest.approx(schedule.delta)
    instance = main_v2.random_cebp_state(
        1, 1, block_sizes=(1,), block_states=(qt.qeye(2) / 2,), clifford_steps=0, seed=1
    )
    result = main_v2.full_cebp_tomography(instance.learner_view(), config=_manual_config(d1=True))
    assert result.success and result.branch == "d1_specialized"
    assert "grouping_ordinary_pool" not in result.realized_copy_ledger.as_dict()
    assert result.structural_certificate.E_split_cert == 0
    assert result.end_to_end_certificate.certified_trace_norm_bound == pytest.approx(
        result.structural_certificate.E_struct_cert + schedule.epsilon_tom
    )
    assert result.decoded_density.tr() == pytest.approx(1)
    summary = main_v2.calibrated_asymptotic_summary(1)
    assert summary["M2"] == "O_tilde(n^9 epsilon^-8)"
    assert summary["grouping"] == "0"
    assert main_v2.calibrated_asymptotic_summary(2)["total"].startswith(
        "O_tilde(2^(O(d log d))"
    )


def test_d1_conditional_tomography_differs_from_unconditional_marginal():
    correlated = 0.5 * qt.ket2dm(qt.tensor(qt.basis(2, 0), qt.basis(2, 0)))
    correlated += 0.5 * qt.ket2dm(qt.tensor(qt.basis(2, 1), qt.basis(2, 1)))
    peeling = _peeling(2, 1)
    localization = _localization(2, 1, 1, (((0,), (0,)),))
    syndrome = main_v2.SyndromeResult(
        True, None, 1, (0,), (1.0,), 5, 0.1, 0.6, 5, True, 5,
        main_v2.CopyLedger((("syndrome_sign_pool", 5),)),
        main_v2.CopyLedger((("peeling_bell_pool", 20), ("syndrome_sign_pool", 5))),
        main_v2.DataProvenance.EMPIRICAL,
    )
    schedule = main_v2.calibrated_end_to_end_schedule(2, 1, 0.5, 0.1)
    result = main_v2._conditional_one_qubit_tomography(
        _source(correlated, 2), peeling, localization, syndrome, schedule,
        seed=2, max_dense_qubits=2, target_accepted_override=100,
        attempts_override=500,
    )
    assert result.success and result.attempted_copies == 1500
    coeff = dict(result.estimates[0].pauli_coefficients)
    assert coeff["Z"] == pytest.approx(1.0)
    assert abs(main_v2.debug_exact_pauli_expectation(correlated.ptrace(1), "Z")) < 1e-12


def test_default_resource_guard_and_injected_early_failure(monkeypatch):
    instance = main_v2.random_cebp_state(
        2, 2, block_sizes=(2,), block_states=(qt.qeye(4) / 4,), clifford_steps=0, seed=1
    )
    guarded = main_v2.full_cebp_tomography(instance.learner_view(), 0.5, 0.1, seed=3)
    assert not guarded.success and guarded.failure_stage == "reservation"
    assert guarded.realized_total == 0 and guarded.peeling is None
    failed = replace(_peeling(2), success=False, failure_reason="injected", h=None,
                     lambda_=None, t=None, epsilon_peel=None, U_stab=None, tableau=None)
    monkeypatch.setattr(main_v2, "empirical_certified_stabilizer_peeling", lambda *_a, **_k: failed)
    result = main_v2.full_cebp_tomography(instance.learner_view(), config=_manual_config())
    assert not result.success and result.failure_stage == "peeling"
    assert result.recovery is result.grouping is result.localization is result.tomography is None


def test_injected_syndrome_and_tomography_failures_preserve_prior_transcript(monkeypatch):
    instance = main_v2.random_cebp_state(
        2, 2, block_sizes=(2,), block_states=(qt.qeye(4) / 4,), clifford_steps=0, seed=1
    )

    def failed_syndrome(_source, peeling, config, **_kwargs):
        return main_v2._syndrome_failure(peeling, config, "injected", peeling.copy_ledger)

    monkeypatch.setattr(main_v2, "recover_peeling_syndrome", failed_syndrome)
    syndrome_failure = main_v2.full_cebp_tomography(
        instance.learner_view(), config=_manual_config()
    )
    assert syndrome_failure.failure_stage == "syndrome"
    assert syndrome_failure.peeling is not None and syndrome_failure.recovery is None
    monkeypatch.undo()

    def failed_tomography(_source, _peeling, localization, _config, **kwargs):
        return main_v2._tomography_failure(
            localization, "injected", kwargs["prior_copy_ledger"]
        )

    monkeypatch.setattr(main_v2, "tomograph_localized_registers", failed_tomography)
    tomography_failure = main_v2.full_cebp_tomography(
        instance.learner_view(), config=_manual_config()
    )
    assert tomography_failure.failure_stage == "tomography"
    assert tomography_failure.localization is not None
    assert tomography_failure.compact_estimator is None


def test_debug_trace_error_only_after_learner_output():
    instance = main_v2.random_cebp_state(
        2, 2, block_sizes=(2,), block_states=(qt.qeye(4) / 4,), clifford_steps=0, seed=1
    )
    result = main_v2.full_cebp_tomography(instance.learner_view(), config=_manual_config())
    assert result.success
    actual = main_v2.debug_end_to_end_trace_error(result, instance)
    assert actual <= 1e-10
    assert actual <= result.end_to_end_certificate.certified_trace_norm_bound


def test_full_learner_rejects_oracle_instance_and_certificate_inputs_are_learned_only():
    instance = main_v2.random_cebp_state(1, 1, seed=2)
    with pytest.raises(TypeError):
        main_v2.full_cebp_tomography(instance, 0.5, 0.1)
    assert "oracle" not in main_v2.compute_structural_certificate.__code__.co_varnames
