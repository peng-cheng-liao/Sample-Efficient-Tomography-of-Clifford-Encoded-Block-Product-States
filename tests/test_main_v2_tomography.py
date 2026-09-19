import math
from dataclasses import replace

import numpy as np
import pytest
import qutip as qt

import main_v2


def _source(state, n):
    state = main_v2._v1._state_to_qobj(state, n)
    return main_v2.SimulatorMeasurementSource(n, state.isket, state)


def _peeling(n, t=0, *, generators=None, U_stab=None, copies=True, theorem=True):
    generators = tuple(generators or ("I" * j + "Z" + "I" * (n - j - 1) for j in range(t)))
    U_stab = np.eye(2**n, dtype=complex) if U_stab is None else np.asarray(U_stab)
    ledger = main_v2.CopyLedger((("peeling_bell_pool", 20),)) if copies else main_v2.CopyLedger()
    return main_v2.PeelingResult(
        success=True,
        failure_reason=None,
        h=0.7,
        lambda_=0.6,
        tau_1=0.01,
        t=t,
        generators=generators,
        generator_symplectic_vectors=tuple(tuple(main_v2._v1.pauli_to_symplectic_col(g)) for g in generators),
        certified_span_basis=(),
        inner_set=generators,
        outer_set=generators,
        inner_span_basis=(),
        outer_span_basis=(),
        U_stab=U_stab,
        tableau=np.eye(2 * n, dtype=np.uint8),
        gates=(),
        epsilon_peel=0.01,
        M1=10 if copies else 0,
        copy_ledger=ledger,
        score_provenance=main_v2.DataProvenance.EMPIRICAL,
        score_frame="physical",
        threshold_grid=(0.6, 0.7),
        transcript=(),
        theorem_grid_condition=theorem,
        theorem_tau_condition=theorem,
    )


def _localization(
    n,
    t,
    registers,
    *,
    d=3,
    U_rec=None,
    theorem=True,
    prior=None,
):
    m = n - t
    U_rec = np.eye(2**m, dtype=complex) if U_rec is None else np.asarray(U_rec)
    bar = np.kron(np.eye(2**t), U_rec)
    prior = prior or main_v2.CopyLedger(
        (("peeling_bell_pool", 20), ("recovery_bell_pool", 40), ("grouping_ordinary_pool", 7))
    )
    used = {q for _cluster, register in registers for q in register}
    return main_v2.LocalizationResult(
        success=True,
        failure_reason=None,
        n=n,
        t=t,
        m=m,
        d=d,
        clusters=tuple(cluster for cluster, _register in registers),
        structures=(),
        cluster_localizations=(),
        J_C=tuple((tuple(cluster), tuple(register)) for cluster, register in registers),
        J_aux=tuple(q for q in range(m) if q not in used),
        K_rec=sum(len(register) for _cluster, register in registers),
        source_basis=np.eye(2 * m, dtype=np.uint8),
        target_basis=np.eye(2 * m, dtype=np.uint8),
        residual_tableau=np.eye(2 * m, dtype=np.uint8),
        synthesis_tableau=np.eye(2 * m, dtype=np.uint8),
        full_tableau=np.eye(2 * n, dtype=np.uint8),
        gates=(),
        U_rec=U_rec,
        bar_U_rec=bar,
        grouping_theorem_preconditions_hold=theorem,
        theorem_localization_preconditions_hold=theorem,
        handoff_valid=True,
        cross_group_direct_sum_holds=True,
        cross_group_symplectic_orthogonality_holds=True,
        global_pairing_holds=True,
        register_partition_holds=True,
        localization_guarantee_holds=True,
        localization_copy_count=0,
        cumulative_copy_ledger=prior,
    )


def _tomography_config(epsilon=10.0, zeta=0.1, **kwargs):
    return main_v2.TomographyConfig(epsilon, zeta, **kwargs)


def _trace_norm(matrix):
    return float(np.linalg.eigvalsh((matrix + matrix.conj().T) / 2.0).__abs__().sum())


def test_phase5_and_phase6_public_api_and_all_integrity():
    intended = {
        "LocalizationConfig",
        "ClusterSymplecticStructure",
        "ClusterLocalization",
        "LocalizationResult",
        "LocalizationInvariantError",
        "validate_grouping_against_recovery",
        "analyze_group_symplectic_structure",
        "localize_grouped_recovery",
        "SyndromeConfig",
        "TomographyConfig",
        "recover_peeling_syndrome",
        "tomograph_localized_registers",
        "recovered_block_tomography",
        "project_eigenvalues_to_simplex",
        "project_to_density_matrix_hs",
    }
    assert intended <= set(main_v2.__all__)
    assert all(hasattr(main_v2, name) for name in main_v2.__all__)


def test_syndrome_budget_formula_validation_and_ceiling():
    n, h_min, zeta = 7, 0.61, 0.03
    expected = math.ceil(2 / h_min * math.log(2 * n / zeta))
    assert main_v2.syndrome_sign_sample_count(n, h_min, zeta) == expected
    with pytest.raises(ValueError):
        main_v2.syndrome_sign_sample_count(n, 0, zeta)
    with pytest.raises(ValueError):
        main_v2.SyndromeConfig(0, h_min)


def test_single_pauli_measurement_and_empirical_mean():
    record = main_v2.measure_pauli_expectation(_source(qt.basis(2, 0), 1), "Z", 50, seed=4)
    assert set(record.outcomes) == {1}
    assert record.empirical_mean == 1.0
    assert record.provenance is main_v2.DataProvenance.EMPIRICAL


def test_syndrome_positive_negative_fresh_batches_ledger_and_seeds():
    state = qt.tensor(qt.basis(2, 0), qt.basis(2, 1))
    peeling = _peeling(2, 2)
    config = main_v2.SyndromeConfig(0.05, 0.6, M_sgn=5, return_details=True)
    first = main_v2.recover_peeling_syndrome(_source(state, 2), peeling, config, seed=8)
    second = main_v2.recover_peeling_syndrome(_source(state, 2), peeling, config, seed=8)
    assert first.success and first.syndrome_bits == (0, 1)
    assert first.empirical_means == (1.0, -1.0)
    assert first.syndrome_sign_pool == 10
    assert first.copy_ledger.as_dict() == {"syndrome_sign_pool": 10}
    assert len(first.records) == 2 and all(record.shots == 5 for record in first.records)
    assert len({record.seed for record in first.records}) == 2
    assert np.array_equal(first.records[0].outcomes, second.records[0].outcomes)
    assert not first.theorem_preconditions_hold  # explicit shots below calibrated count


def test_t_zero_syndrome_is_zero_copy_and_does_not_measure(monkeypatch):
    peeling = _peeling(1, 0)
    monkeypatch.setattr(main_v2, "measure_pauli_expectation", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError()))
    result = main_v2.recover_peeling_syndrome(
        _source(qt.basis(2, 0), 1), peeling, main_v2.SyndromeConfig(0.1, 0.6), seed=2
    )
    assert result.success and result.syndrome_bits == result.empirical_means == ()
    assert result.M_sgn == result.syndrome_sign_pool == 0


@pytest.mark.parametrize("mixed", [False, True])
def test_localized_state_orientation_for_ket_and_density(mixed):
    plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    state = plus * plus.dag() if mixed else plus
    S = np.diag([1.0, 1.0j])
    peeling = _peeling(1, 0)
    localization = _localization(1, 0, (((0,), (0,)),), d=1, U_rec=S)
    actual = main_v2.debug_exact_localized_state(_source(state, 1), peeling, localization)
    Sq = qt.Qobj(S, dims=[[2], [2]])
    expected = Sq.dag() * state if not mixed else Sq.dag() * state * Sq
    assert np.allclose(actual.full(), expected.full())
    opposite = Sq * state if not mixed else Sq * state * Sq.dag()
    assert not np.allclose(actual.full(), opposite.full())


def test_localized_state_applies_ustab_then_bar_urec_in_exact_order():
    H = np.array([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2.0)
    S = np.diag([1.0, 1.0j])
    cnot = main_v2._v1._two_qubit_cnot(2, 0, 1)
    U_stab = np.kron(H, np.eye(2)) @ cnot
    state = qt.tensor(
        (qt.basis(2, 0) + 0.37j * qt.basis(2, 1)).unit(),
        (qt.basis(2, 0) - 0.61 * qt.basis(2, 1)).unit(),
    )
    peeling = _peeling(2, 1, generators=("XI",), U_stab=U_stab)
    localization = _localization(2, 1, (((0,), (0,)),), d=1, U_rec=S)
    actual = main_v2.debug_exact_localized_state(_source(state, 2), peeling, localization)
    expected = localization.bar_U_rec.conj().T @ U_stab.conj().T @ state.full()
    opposite_order = U_stab.conj().T @ localization.bar_U_rec.conj().T @ state.full()
    assert np.allclose(actual.full(), expected)
    assert not np.allclose(actual.full(), opposite_order)


def test_signed_minus_phase_regression_and_production_path():
    # S Y S^dagger = -X.  On physical |+>, localized Y is therefore -1,
    # while the phase-free pullback direction X would incorrectly predict +1.
    plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    S = np.diag([1.0, 1.0j])
    peeling = _peeling(1, 0)
    localization = _localization(1, 0, (((0,), (0,)),), d=1, U_rec=S)
    localized = main_v2.debug_exact_localized_state(_source(plus, 1), peeling, localization)
    assert main_v2.debug_exact_pauli_expectation(localized, "Y") == pytest.approx(-1.0)
    assert main_v2.debug_exact_pauli_expectation(plus, "X") == pytest.approx(1.0)
    result = main_v2.tomograph_localized_registers(
        _source(plus, 1), peeling, localization, _tomography_config(), seed=9
    )
    coefficients = dict(result.estimates[0].pauli_coefficients)
    assert result.success and coefficients["Y"] == -1.0


def test_register_budget_exact_formulas_and_maps():
    budget = main_v2.register_tomography_budget((4,), (0, 2), 3.0, 0.04)
    assert budget.k_C == 2 and budget.N_C_Pauli == 15
    assert budget.tau_C_tom == pytest.approx(3 / (4 * math.sqrt(15)))
    assert budget.M_C_Pauli == math.ceil(2 / budget.tau_C_tom**2 * math.log(30 / 0.04))
    assert budget.L_C == 15 * budget.M_C_Pauli
    loc = _localization(2, 0, (((4,), (0, 1)),), d=2)
    bad = _tomography_config(epsilon_by_cluster=(((9,), 1.0),))
    result = main_v2.tomograph_localized_registers(_source(qt.basis(4, 0), 2), _peeling(2), loc, bad)
    assert not result.success and "map must match" in result.failure_reason


def test_parallel_schedule_max_not_sum_and_exact_local_counts():
    state = qt.tensor(qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 0))
    peeling = _peeling(3)
    registers = (((0,), (0,)), ((1,), (1, 2)))
    localization = _localization(3, 0, registers, d=2)
    config = _tomography_config(
        22.0,
        0.1,
        epsilon_by_cluster=(((0,), 10.0), ((1,), 12.0)),
        zeta_by_cluster=(((0,), 0.04), ((1,), 0.04)),
        return_details=True,
    )
    result = main_v2.tomograph_localized_registers(_source(state, 3), peeling, localization, config, seed=5)
    lengths = [budget.L_C for budget in result.budgets]
    assert result.success and len(set(lengths)) == 2
    assert result.N_bp == max(lengths) < sum(lengths)
    assert result.block_tomography_pool == result.N_bp
    assert result.one_common_copy_per_round
    assert len(result.schedule_transcript) == result.N_bp
    for budget, record in zip(result.budgets, result.records):
        assert len(record.schedule) == budget.L_C
        assert all(len(outcomes) == budget.M_C_Pauli for _pauli, outcomes in record.outcomes_by_pauli)


@pytest.mark.parametrize("m", [0, 2])
def test_no_register_branch_is_zero_copy_maximally_mixed(m):
    n = max(1, m)
    t = n - m
    state = qt.basis(2**n, 0)
    peeling = _peeling(n, t)
    localization = _localization(n, t, (), d=1)
    result = main_v2.tomograph_localized_registers(
        _source(state, n), peeling, localization, _tomography_config(), seed=1
    )
    assert result.success and result.Khat == result.N_bp == result.block_tomography_pool == 0
    expected = np.array([[1.0]]) if m == 0 else np.eye(2**m) / 2**m
    assert np.allclose(result.localized_empirical_estimator.full(), expected)


def test_linear_inversion_exact_coefficients_entangled_state():
    bell = (qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) + qt.tensor(qt.basis(2, 1), qt.basis(2, 1))).unit()
    rho = bell * bell.dag()
    coefficients = {
        pauli: main_v2.debug_exact_pauli_expectation(rho, pauli)
        for pauli in main_v2._local_paulis(2)
    }
    reconstructed = main_v2.linear_inversion_from_pauli_coefficients(2, coefficients)
    assert np.allclose(reconstructed.full(), rho.full())
    assert reconstructed.isherm and reconstructed.tr() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "vector,expected",
    [
        ([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]),
        ([-0.2, 0.4, 0.8], [0.0, 0.3, 0.7]),
        ([-2.0, -1.0, 4.0], [0.0, 0.0, 1.0]),
        ([0.5, 0.5, 0.5, 0.5], [0.25] * 4),
    ],
)
def test_simplex_projection_examples_and_kkt(vector, expected):
    projected = main_v2.project_eigenvalues_to_simplex(vector)
    assert np.allclose(projected, expected)
    assert projected.sum() == pytest.approx(1.0) and np.min(projected) >= 0.0
    active = projected > 1e-12
    theta = np.asarray(vector)[active] - projected[active]
    assert np.max(theta) - np.min(theta) < 1e-10
    threshold = float(theta[0])
    assert np.all(np.asarray(vector)[~active] <= threshold + 1e-10)


def test_simplex_projection_matches_independent_small_grid_reference():
    vector = np.array([-0.35, 0.23, 1.41])
    projected = main_v2.project_eigenvalues_to_simplex(vector)
    grid = np.linspace(0, 1, 1001)
    candidates = np.array([(x, y, 1 - x - y) for x in grid for y in grid if x + y <= 1])
    reference = candidates[np.argmin(np.sum((candidates - vector) ** 2, axis=1))]
    assert np.linalg.norm(projected - vector) <= np.linalg.norm(reference - vector) + 1e-10


def test_density_projection_physical_and_no_farther_from_true_state():
    linear = qt.Qobj([[1.2, 0.2], [0.2, -0.2]], dims=[[2], [2]])
    physical = main_v2.project_to_density_matrix_hs(linear)
    true = qt.ket2dm(qt.basis(2, 0))
    assert physical.isherm and physical.tr() == pytest.approx(1.0)
    assert np.min(np.linalg.eigvalsh(physical.full())) >= -1e-12
    assert np.linalg.norm(physical.full() - true.full()) <= np.linalg.norm(linear.full() - true.full()) + 1e-12


@pytest.mark.parametrize(
    "state,n,d,register,epsilon",
    [
        (qt.basis(2, 0), 1, 1, (0,), 2.0),
        (0.7 * qt.ket2dm(qt.basis(2, 0)) + 0.3 * qt.ket2dm(qt.basis(2, 1)), 1, 1, (0,), 2.0),
        ((qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) + qt.tensor(qt.basis(2, 1), qt.basis(2, 1))).unit(), 2, 2, (0, 1), 4.0),
        (qt.tensor(qt.basis(2, 0), qt.basis(2, 1), qt.basis(2, 0)), 3, 3, (0, 1, 2), 12.0),
    ],
)
def test_empirical_local_tomography_k1_k2_k3(state, n, d, register, epsilon):
    peeling = _peeling(n)
    localization = _localization(n, 0, (((0,), register),), d=d)
    result = main_v2.tomograph_localized_registers(
        _source(state, n), peeling, localization, _tomography_config(epsilon, 0.05), seed=12
    )
    assert result.success
    estimate, budget = result.estimates[0], result.budgets[0]
    exact = state * state.dag() if state.isket else state
    exact_coefficients = {p: main_v2.debug_exact_pauli_expectation(exact, p) for p in main_v2._local_paulis(n)}
    assert max(abs(value - exact_coefficients[p]) for p, value in estimate.pauli_coefficients) <= budget.tau_C_tom
    error = _trace_norm(estimate.nu_hat.full() - exact.full())
    assert error <= budget.epsilon_C
    assert estimate.nu_hat.isherm and estimate.nu_hat.tr() == pytest.approx(1.0)
    assert np.min(np.linalg.eigvalsh(estimate.nu_hat.full())) >= -1e-10


def test_noncontiguous_product_assembly_and_auxiliary_marginals():
    zero = qt.ket2dm(qt.basis(2, 0))
    one = qt.ket2dm(qt.basis(2, 1))
    product = main_v2.assemble_localized_product_estimator((((2,), zero), ((0,), one)), (1,), 3)
    assert np.allclose(product.ptrace(0).full(), one.full())
    assert np.allclose(product.ptrace(1).full(), np.eye(2) / 2)
    assert np.allclose(product.ptrace(2).full(), zero.full())
    assert product.isherm and product.tr() == pytest.approx(1.0)


def test_combined_copy_ledger_seed_independence_and_no_decoding():
    state = qt.tensor(qt.basis(2, 1), qt.basis(2, 0))
    peeling = _peeling(2, 1, generators=("ZI",))
    localization = _localization(2, 1, (((0,), (0,)),), d=1)
    config = main_v2.RecoveredBlockTomographyConfig(
        main_v2.SyndromeConfig(0.1, 0.6, M_sgn=9, return_details=True),
        _tomography_config(return_details=True),
    )
    first = main_v2.recovered_block_tomography(_source(state, 2), peeling, localization, config, syndrome_seed=2, tomography_seed=3)
    changed_syndrome = main_v2.recovered_block_tomography(_source(state, 2), peeling, localization, config, syndrome_seed=4, tomography_seed=3)
    changed_tomography = main_v2.recovered_block_tomography(_source(state, 2), peeling, localization, config, syndrome_seed=2, tomography_seed=5)
    assert first.success and first.syndrome_bits == (1,)
    ledger = first.cumulative_copy_ledger.as_dict()
    assert ledger["peeling_bell_pool"] == 20 and ledger["recovery_bell_pool"] == 40
    assert ledger["grouping_ordinary_pool"] == 7
    assert ledger["syndrome_sign_pool"] == 9
    assert ledger["block_tomography_pool"] == first.tomography.N_bp
    assert first.cumulative_copy_ledger.total == 20 + 40 + 7 + 9 + first.tomography.N_bp
    assert first.tomography.records[0].outcomes_by_pauli == changed_syndrome.tomography.records[0].outcomes_by_pauli
    assert first.syndrome.records[0].outcomes.tolist() == changed_tomography.syndrome.records[0].outcomes.tolist()
    assert not hasattr(first, "rho_hat_bp")
    assert hasattr(main_v2, "full_cebp_tomography")


def test_uncertified_localization_rejected_and_override_is_noncertifying():
    source = _source(qt.basis(2, 0), 1)
    peeling = _peeling(1)
    localization = _localization(1, 0, (((0,), (0,)),), d=1, theorem=False)
    rejected = main_v2.tomograph_localized_registers(source, peeling, localization, _tomography_config())
    override = main_v2.tomograph_localized_registers(
        source,
        peeling,
        localization,
        _tomography_config(allow_uncertified_localization=True),
        seed=1,
    )
    assert not rejected.success and rejected.failure_reason == "localization_not_theorem_certified"
    assert override.success and not override.theorem_preconditions_hold


def test_combined_rejects_uncertified_localization_before_any_measurement(monkeypatch):
    localization = _localization(1, 0, (((0,), (0,)),), d=1, theorem=False)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("upstream rejection measured a physical copy")

    monkeypatch.setattr(main_v2, "measure_pauli_expectation", forbidden)
    monkeypatch.setattr(main_v2, "_joint_disjoint_register_records", forbidden)
    result = main_v2.recovered_block_tomography(
        _source(qt.basis(2, 0), 1),
        _peeling(1),
        localization,
        main_v2.RecoveredBlockTomographyConfig(
            main_v2.SyndromeConfig(0.1, 0.6), _tomography_config()
        ),
    )
    assert not result.success
    assert result.failure_reason == "localization_not_theorem_certified"
    assert result.syndrome.syndrome_sign_pool == result.tomography.N_bp == 0


def test_empirical_path_does_not_use_oracle_helpers(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("empirical Phase 6 used oracle/debug truth")

    monkeypatch.setattr(main_v2, "debug_exact_localized_state", forbidden)
    monkeypatch.setattr(main_v2, "debug_exact_register_marginal", forbidden)
    monkeypatch.setattr(main_v2, "debug_oracle_sector_block_labels", forbidden)
    state = qt.basis(2, 0)
    result = main_v2.recovered_block_tomography(
        _source(state, 1),
        _peeling(1),
        _localization(1, 0, (((0,), (0,)),), d=1),
        main_v2.RecoveredBlockTomographyConfig(
            main_v2.SyndromeConfig(0.1, 0.6), _tomography_config()
        ),
        syndrome_seed=1,
        tomography_seed=2,
    )
    assert result.success


def _valid_phase6_handoff_fixture():
    state = qt.tensor(qt.basis(2, 1), qt.basis(2, 0))
    peeling = _peeling(2, 1, generators=("ZI",))
    localization = _localization(2, 1, (((0,), (0,)),), d=1)
    result = main_v2.recovered_block_tomography(
        _source(state, 2),
        peeling,
        localization,
        main_v2.RecoveredBlockTomographyConfig(
            main_v2.SyndromeConfig(0.1, 0.6, M_sgn=9, return_details=True),
            _tomography_config(return_details=True),
        ),
        syndrome_seed=2,
        tomography_seed=3,
    )
    assert result.success
    return peeling, localization, result


def test_strict_phase6_handoff_validator_accepts_valid_transcript():
    peeling, localization, result = _valid_phase6_handoff_fixture()
    assert main_v2.validate_recovered_block_tomography_handoff(
        peeling, localization, result
    )


@pytest.mark.parametrize("mutation", ["length", "bit", "copies"])
def test_strict_handoff_rejects_malformed_syndrome(mutation):
    peeling, localization, result = _valid_phase6_handoff_fixture()
    syndrome = result.syndrome
    if mutation == "length":
        syndrome = replace(syndrome, syndrome_bits=())
    elif mutation == "bit":
        syndrome = replace(syndrome, syndrome_bits=(2,))
    else:
        syndrome = replace(syndrome, syndrome_sign_pool=8)
    malformed = replace(result, syndrome=syndrome, syndrome_bits=syndrome.syndrome_bits)
    with pytest.raises(ValueError):
        main_v2.validate_recovered_block_tomography_handoff(
            peeling, localization, malformed
        )


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "budget", "N_bp", "physical", "ledger"])
def test_strict_handoff_rejects_malformed_registers_and_ledger(mutation):
    peeling, localization, result = _valid_phase6_handoff_fixture()
    tomography = result.tomography
    malformed = result
    if mutation == "missing":
        malformed = replace(result, register_estimates=())
    elif mutation == "duplicate":
        malformed = replace(
            result,
            register_estimates=result.register_estimates + result.register_estimates,
        )
    elif mutation == "budget":
        bad_budget = replace(tomography.budgets[0], J_C=(1,))
        malformed = replace(result, tomography=replace(tomography, budgets=(bad_budget,)))
    elif mutation == "N_bp":
        malformed = replace(result, tomography=replace(tomography, N_bp=tomography.N_bp + 1))
    elif mutation == "physical":
        estimate = tomography.estimates[0]
        bad = qt.Qobj([[1.2, 0.0], [0.0, -0.2]], dims=[[2], [2]])
        bad_estimate = replace(estimate, nu_hat=bad)
        bad_tomography = replace(tomography, estimates=(bad_estimate,))
        malformed = replace(
            result,
            tomography=bad_tomography,
            register_estimates=((estimate.cluster, estimate.J_C, bad),),
        )
    else:
        ledger = main_v2.CopyLedger(
            (("peeling_bell_pool", 20), ("recovery_bell_pool", 40),
             ("grouping_ordinary_pool", 7), ("syndrome_sign_pool", 9),
             ("block_tomography_pool", tomography.N_bp + 1))
        )
        malformed = replace(result, cumulative_copy_ledger=ledger)
    with pytest.raises(ValueError):
        main_v2.validate_recovered_block_tomography_handoff(
            peeling, localization, malformed
        )


def test_numerical_projection_bound_is_explicit_conservative_and_within_allowance():
    peeling, localization, result = _valid_phase6_handoff_fixture()
    estimate = result.tomography.estimates[0]
    bound = estimate.numerical_projection_error_bound
    assert np.isfinite(bound) and bound > 0.0
    assert bound <= result.tomography.budgets[0].epsilon_C / 2.0
    assert main_v2.validate_recovered_block_tomography_handoff(
        peeling, localization, result
    )


def test_unrealistically_tight_projection_tolerance_fails_explicitly():
    linear = qt.Qobj([[1.2, 0.2], [0.2, -0.2]], dims=[[2], [2]])
    with pytest.raises(ValueError, match="certification floor"):
        main_v2.project_to_density_matrix_hs(linear, tolerance=1e-30)
