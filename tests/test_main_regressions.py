import inspect
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
import qutip as qt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


def test_required_public_api_is_present():
    required = {
        "bell_sampling", "bell_sampling_pure", "empirical_stabilizer_peeling",
        "rank_guided_symplectic_recovery", "single_qubit_tomography",
        "full_recovery_infidelity", "random_clifford_gate",
        "random_clifford_encoded_product_state",
        "random_clifford_encoded_product_pure_state",
        "generic_pure_state_local_pauli_tomography",
    }
    assert all(callable(getattr(main, name, None)) for name in required)


def _pauli_from_xz(x, z):
    out = np.array([[1]], dtype=complex)
    for x_bit, z_bit in zip(x, z):
        local = (1j) ** int(x_bit * z_bit) * np.linalg.matrix_power(main.X2, int(x_bit)) @ np.linalg.matrix_power(main.Z2, int(z_bit))
        out = np.kron(out, local)
    return out


def _assert_tableau_action(tableau, unitary):
    n = tableau.shape[0] // 2
    dimension = 2 ** n
    assert np.allclose(unitary.conj().T @ unitary, np.eye(dimension))
    for x_bits in __import__("itertools").product((0, 1), repeat=n):
        for z_bits in __import__("itertools").product((0, 1), repeat=n):
            x, z = np.array(x_bits), np.array(z_bits)
            transformed = (tableau @ np.concatenate([x, z])) % 2
            actual = unitary @ _pauli_from_xz(x, z) @ unitary.conj().T
            expected = _pauli_from_xz(transformed[:n], transformed[n:])
            phase = np.trace(expected.conj().T @ actual) / dimension
            assert np.allclose(actual, phase / abs(phase) * expected)


def _random_symplectic(n, steps, seed):
    rng = np.random.default_rng(seed)
    tableau = np.eye(2 * n, dtype=np.uint8)
    for _ in range(steps):
        gate = int(rng.integers(0, 2 if n == 1 else 3))
        if gate == 0:
            main.apply_h_right(tableau, int(rng.integers(0, n)))
        elif gate == 1:
            main.apply_s_right(tableau, int(rng.integers(0, n)))
        else:
            control = int(rng.integers(0, n))
            target = int(rng.integers(0, n - 1))
            if target >= control:
                target += 1
            main.apply_cnot_right(tableau, control, target)
    return tableau


def _explicit_two_copy_bell_probabilities(rho, n):
    """Small-n reference retaining the deleted rho-tensor-rho implementation."""
    rho = main._as_density(rho, n)
    rho2 = qt.tensor(rho, rho).permute(
        [axis for qubit in range(n) for axis in (qubit, n + qubit)]
    )
    zero, one = qt.basis(2, 0), qt.basis(2, 1)
    bell_kets = [
        (qt.tensor(zero, zero) + qt.tensor(one, one)).unit(),
        (qt.tensor(zero, zero) - qt.tensor(one, one)).unit(),
        (qt.tensor(zero, one) + qt.tensor(one, zero)).unit(),
        (qt.tensor(zero, one) - qt.tensor(one, zero)).unit(),
    ]
    bell_projectors = [ket * ket.dag() for ket in bell_kets]
    probabilities = np.empty(4 ** n, dtype=float)
    for index in range(4 ** n):
        digits = [(index // (4 ** qubit)) % 4 for qubit in range(n)]
        projector = qt.tensor([bell_projectors[digit] for digit in digits])
        probabilities[index] = float((projector * rho2).tr().real)
    probabilities = np.clip(probabilities, 0.0, None)
    return probabilities / probabilities.sum()


def _sample_bell_records_from_probabilities(probabilities, n, M, seed):
    outcomes = np.random.default_rng(seed).choice(probabilities.size, size=M, p=probabilities)
    eig_table = np.array(
        [[1, -1, 1], [-1, 1, 1], [1, 1, -1], [-1, -1, -1]],
        dtype=np.int8,
    )
    records = np.empty((n, M, 3), dtype=np.int8)
    for qubit in range(n):
        records[qubit] = eig_table[(outcomes // (4 ** qubit)) % 4]
    return records


def _direct_product_pauli_probabilities(state, axes):
    n = len(axes)
    rho = main._as_density(state, n)
    local_bases = [main._single_qubit_pauli_basis_change(axis) for axis in axes]
    probabilities = []
    for bits in itertools.product((0, 1), repeat=n):
        ket = qt.tensor([
            qt.Qobj(basis[:, bit], dims=[[2], [1]])
            for basis, bit in zip(local_bases, bits)
        ])
        probabilities.append(float((ket.dag() * rho * ket).real))
    return np.asarray(probabilities)


def test_clifford_synthesis_and_single_qubit_random_walk():
    for n in (1, 2, 3):
        for seed in range(3):
            gates, tableau, unitary = main.random_clifford_gate(n, steps=5, seed=seed)
            assert np.allclose(unitary.conj().T @ unitary, np.eye(2 ** n))
            if n == 1:
                assert all(gate[0] != "CNOT" for gate in gates)
            # The column tableau records U† P U; the local assertion checks
            # U P U†, hence pass U† here.
            _assert_tableau_action(tableau, unitary.conj().T)
    for seed in range(12):
        _, _, unitary = main.random_clifford_gate(1, steps=4, seed=seed)
        assert np.allclose(unitary.conj().T @ unitary, np.eye(2))


def test_thresholded_ranking_and_symplectic_completion():
    psi = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
    samples = main.bell_sampling_pure(2, 24, psi, seed=6)
    for threshold in (0.0, 0.35, 0.8):
        assert main.paulis_above_threshold_from_B(samples, threshold) == [
            item for item in main.rank_all_sP_from_B(samples) if item[0] >= threshold
        ]

    tableau = _random_symplectic(3, steps=7, seed=4)
    completed = main.complete_isotropic_to_symplectic(tableau[:, 3:5])
    assert main.is_symplectic(completed)
    assert np.array_equal(completed[:, 3:5], tableau[:, 3:5])
    partial = main._complete_partial_symplectic_basis(tableau[:, :2], tableau[:, 3:4])
    assert main.is_symplectic(partial)
    assert np.array_equal(partial[:, :2], tableau[:, :2])
    assert np.array_equal(partial[:, 3:4], tableau[:, 3:4])


def test_tableau_synthesis_uses_the_reduction_convention():
    for n in (1, 2, 3):
        for seed in range(3):
            tableau = _random_symplectic(n, steps=5, seed=seed)
            gates = main.synthesize_clifford_from_tableau(tableau)
            unitary = main.gates_to_unitary(gates, n)
            _assert_tableau_action(tableau, unitary)


def test_seeded_sampling_generators_and_end_to_end_reproducible():
    psi = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
    assert np.array_equal(
        main.bell_sampling_pure(2, 20, psi, seed=9),
        main.bell_sampling_pure(2, 20, psi, seed=9),
    )
    state_a = main.random_clifford_encoded_product_state(4, 0.1, 2, 0, 1, 1, 0, seed=9, return_details=True)
    state_b = main.random_clifford_encoded_product_state(4, 0.1, 2, 0, 1, 1, 0, seed=9, return_details=True)
    assert np.allclose(state_a["rho_encoded"].full(), state_b["rho_encoded"].full())
    assert np.allclose(state_a["Uc"], state_b["Uc"])

    first = main.full_recovery_infidelity(psi, 2, 0.4, 0.2, 30, 30, 30, seed1=1, seed2=2, return_details=True)
    second = main.full_recovery_infidelity(psi, 2, 0.4, 0.2, 30, 30, 30, seed1=1, seed2=2, return_details=True)
    assert first["infidelity"] == second["infidelity"]
    assert np.allclose(first["rho_recovered"].full(), second["rho_recovered"].full())
    assert np.isfinite(first["infidelity"]) and 0 <= first["infidelity"] <= 1


def test_exact_mixed_bell_probabilities_match_two_copy_reference():
    for n in (1, 2, 3):
        rng = np.random.default_rng(100 + n)
        vector = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
        pure = qt.Qobj(vector, dims=[[2] * n, [1] * n]).unit()
        matrix = rng.normal(size=(2 ** n, 2 ** n)) + 1j * rng.normal(size=(2 ** n, 2 ** n))
        mixed = qt.Qobj(matrix @ matrix.conj().T, dims=[[2] * n, [2] * n])
        mixed = mixed / mixed.tr()

        for rho in (pure * pure.dag(), mixed):
            expected = _explicit_two_copy_bell_probabilities(rho, n)
            actual = main._bell_outcome_probabilities(rho, n)
            assert np.max(np.abs(actual - expected)) <= 5e-14
            assert actual.min() >= 0.0
            assert np.isclose(actual.sum(), 1.0)
            assert np.array_equal(
                main.bell_sampling(n, 64, rho, seed=700 + n),
                _sample_bell_records_from_probabilities(expected, n, 64, seed=700 + n),
            )

        pure_general = main._bell_outcome_probabilities(pure * pure.dag(), n)
        pure_direct = main._pure_bell_outcome_probabilities(n, pure)
        assert np.allclose(pure_general, pure_direct, atol=5e-14)
        assert np.array_equal(
            main.bell_sampling(n, 64, pure * pure.dag(), seed=800 + n),
            main.bell_sampling_pure(n, 64, pure, seed=800 + n),
        )

    source = inspect.getsource(main.bell_sampling) + inspect.getsource(main._bell_outcome_probabilities)
    assert "qt.tensor(rho, rho)" not in source
    assert "rho2" not in source


@pytest.mark.parametrize("delta", (0.0, 0.1, 0.30, 0.32, 0.33, 1.0 / 3.0))
def test_mixed_a1_direct_simplex_sampling(delta):
    # Forty seeds per delta exercises hundreds of deterministic samples in total.
    for seed in range(40):
        first = main.random_clifford_encoded_product_state(
            1, delta, 1, 0, 1, 0, 0, seed=seed, return_details=True
        )
        second = main.random_clifford_encoded_product_state(
            1, delta, 1, 0, 1, 0, 0, seed=seed, return_details=True
        )
        bloch = first["bloch_vectors"][0]
        squared = bloch * bloch
        rho = first["local_states"][0]
        assert np.allclose(bloch, second["bloch_vectors"][0])
        assert np.all(squared >= delta - 1e-12)
        assert np.all(squared <= 1.0 - delta + 1e-12)
        assert squared.sum() <= 1.0 + 1e-12
        assert abs(rho.tr() - 1.0) <= 1e-12
        assert np.linalg.eigvalsh(rho.full()).min() >= -1e-12
        if delta == 1.0 / 3.0:
            assert np.allclose(squared, np.full(3, 1.0 / 3.0), atol=1e-15)


def test_mixed_a1_just_above_boundary_errors():
    with pytest.raises(ValueError, match="feasible A1"):
        main.random_clifford_encoded_product_state(
            1, 1.0 / 3.0 + 1e-8, 1, 0, 1, 0, 0, seed=1
        )


@pytest.mark.parametrize(
    ("delta", "counts"),
    ((0.0, (1, 0, 0, 0)), (0.8, (1, 0, 0, 0)), (0.0, (0, 0, 1, 0)), (0.49, (0, 0, 1, 0))),
)
def test_s_and_a2_constraint_regression(delta, counts):
    S, A1, A2, B = counts
    for seed in range(12):
        details = main.random_clifford_encoded_product_state(
            2, delta, 1, S, A1, A2, B, seed=seed, return_details=True
        )
        squared = details["bloch_vectors"][0] ** 2
        rho = details["local_states"][0]
        assert squared.sum() <= 1.0 + 1e-12
        assert np.linalg.eigvalsh(rho.full()).min() >= -1e-12
        if S:
            assert squared.max() >= 1.0 - delta - 1e-12
        else:
            assert np.count_nonzero(squared >= delta - 1e-12) >= 1
            assert np.count_nonzero(squared <= delta + 1e-12) >= 2


def test_mixed_generator_constraints_and_impossible_region():
    for seed in range(16):
        details = main.random_clifford_encoded_product_state(3, 0.1, 4, 1, 1, 1, 1, seed=seed, return_details=True)
        for label, bloch in zip(details["class_assignment"], details["bloch_vectors"]):
            squared = bloch * bloch
            assert squared.sum() <= 1 + 1e-10
            if label == "S":
                assert squared.max() >= 0.9 - 1e-10
            elif label == "A1":
                assert np.all(squared >= 0.1 - 1e-10)
                assert np.all(squared <= 0.9 + 1e-10)
            elif label == "A2":
                assert np.count_nonzero(squared >= 0.1 - 1e-10) >= 1
                assert np.count_nonzero(squared <= 0.1 + 1e-10) >= 2
            else:
                assert np.all(squared <= 0.1 + 1e-10)
    with pytest.raises(ValueError, match="feasible A1"):
        main.random_clifford_encoded_product_state(2, 0.4, 1, 0, 1, 0, 0, seed=1)


@pytest.mark.parametrize("delta", (0.05, 0.2, 0.4, 0.5, 0.8, 1.0))
def test_b_class_sampling_is_feasible_for_all_legal_deltas(delta):
    for seed in range(8):
        details = main.random_clifford_encoded_product_state(2, delta, 1, 0, 0, 0, 1, seed=seed, return_details=True)
        bloch = details["bloch_vectors"][0]
        rho = details["local_states"][0]
        squared = bloch * bloch
        assert np.all(squared <= delta + 1e-10)
        assert squared.sum() <= 1 + 1e-10
        assert abs(rho.tr() - 1) < 1e-10
        assert np.linalg.eigvalsh(rho.full()).min() >= -1e-10


def test_shot_count_tuple_order_and_conservation():
    counts = main.pauli_shot_counts_from_state(np.eye(2) / 2, 11, 13, 17, seed=3)
    assert isinstance(counts, tuple) and len(counts) == 6
    assert counts[0] + counts[1] == 11
    assert counts[2] + counts[3] == 13
    assert counts[4] + counts[5] == 17


def test_product_pauli_probabilities_match_direct_projectors_for_pure_and_mixed_states():
    rng = np.random.default_rng(901)
    for n in (1, 2, 3):
        vector = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
        pure = qt.Qobj(vector, dims=[[2] * n, [1] * n]).unit()
        states = [pure]
        if n <= 2:
            matrix = rng.normal(size=(2 ** n, 2 ** n)) + 1j * rng.normal(size=(2 ** n, 2 ** n))
            mixed = qt.Qobj(matrix @ matrix.conj().T, dims=[[2] * n, [2] * n])
            states.append(mixed / mixed.tr())
        for state in states:
            for axis in ("X", "Y", "Z"):
                axes = (axis,) * n
                actual = main._local_product_pauli_measurement_probabilities(state, axes)
                expected = _direct_product_pauli_probabilities(state, axes)
                assert np.allclose(actual, expected, atol=5e-14)
            shared = main._shared_single_qubit_tomography(
                state, n, 17, seed=910 + n, return_details=True
            )
            assert len(shared["rho_estimates"]) == n
            assert all(rho.dims == [[2], [2]] for rho in shared["rho_estimates"])
            assert all(records.shape == (17, n) for records in shared["records"].values())


def test_shared_tomography_reuses_global_records_and_is_seeded():
    zero, one = qt.basis(2, 0), qt.basis(2, 1)
    ghz = (qt.tensor(zero, zero) + qt.tensor(one, one)).unit()
    correlated_mixed = 0.5 * qt.ket2dm(qt.tensor(zero, zero)) + 0.5 * qt.ket2dm(qt.tensor(one, one))

    for state in (ghz, correlated_mixed):
        first = main._shared_single_qubit_tomography(
            state, 2, 41, seed=902, return_details=True
        )
        second = main._shared_single_qubit_tomography(
            state, 2, 41, seed=902, return_details=True
        )
        assert len(set(first["axis_seeds"].values())) == 3
        for axis in ("X", "Y", "Z"):
            records = first["records"][axis]
            assert records.shape == (41, 2)
            assert np.array_equal(records, second["records"][axis])
        # Both test states have perfectly correlated computational-basis bits.
        assert np.array_equal(first["records"]["Z"][:, 0], first["records"]["Z"][:, 1])

        for qubit, count_tuple in enumerate(first["counts"]):
            expected_counts = []
            for axis in ("X", "Y", "Z"):
                minus = int(first["records"][axis][:, qubit].sum())
                expected_counts.extend((41 - minus, minus))
            assert count_tuple == tuple(expected_counts)
            assert first["rho_estimates"][qubit].dims == [[2], [2]]
            assert np.allclose(
                first["rho_estimates"][qubit].full(),
                second["rho_estimates"][qubit].full(),
            )

    rho_one = 0.5 * (qt.qeye(2) + 0.3 * qt.sigmax() - 0.2 * qt.sigmay() + 0.4 * qt.sigmaz())
    shared = main._shared_single_qubit_tomography(rho_one, 1, 10_000, seed=903)[0]
    standalone = main.single_qubit_tomography(rho_one, 10_000, 10_000, 10_000, seed=903)
    assert np.allclose(shared.full(), standalone.full(), atol=0.03)


@pytest.mark.parametrize("m", (1, 2, 3))
def test_empty_partial_symplectic_basis_returns_identity(m):
    completed = main._complete_partial_symplectic_basis(
        np.zeros((2 * m, 0), dtype=np.uint8),
        np.zeros((2 * m, 0), dtype=np.uint8),
    )
    assert np.array_equal(completed, np.eye(2 * m, dtype=np.uint8))
    assert main.is_symplectic(completed)


def test_zero_residual_recovery_preserves_purity_metadata():
    pure = qt.basis(2, 0)
    mixed = qt.qeye(2) / 2
    pure_result = main.rank_guided_symplectic_recovery(
        pure, 1, 1, 0, 0.2, seed=1, return_details=True
    )
    mixed_result = main.rank_guided_symplectic_recovery(
        mixed, 1, 1, 0, 0.2, seed=1, return_details=True
    )
    assert pure_result["is_pure_input"] is True
    assert mixed_result["is_pure_input"] is False
    assert pure_result["recovered_axes"]["m"] == 0
    assert mixed_result["recovered_axes"]["m"] == 0


def test_peeling_and_recovery_seeded_outputs_are_unchanged():
    plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    plus_y = (qt.basis(2, 0) + 1j * qt.basis(2, 1)).unit()
    state = qt.tensor(plus, plus_y)
    peeling = main.empirical_stabilizer_peeling(
        state, 2, 48, 0.4, seed=303, return_details=True
    )
    assert peeling["g_list"] == ["IY", "XI"]
    recovery = main.rank_guided_symplectic_recovery(
        state, 2, 0, 48, 0.2, seed=404, return_details=True
    )
    assert recovery["recovered_axes"] == {
        "A1_axes_res": [{"x": "IY", "z": "IZ", "y": "IX"}],
        "A2_axes_res": [{"x": "XI"}],
        "A1_axes_full": [{"x": "IY", "z": "IZ", "y": "IX"}],
        "A2_axes_full": [{"x": "XI"}],
        "A1": [1],
        "A2": [2],
        "m": 2,
    }


def test_end_to_end_copy_budgets_for_residual_and_fully_peeled_paths():
    mixed = qt.qeye([2, 2]) / 4
    residual = main.full_recovery_infidelity(
        mixed, 2, 0.1, 0.2, 40, 30, 20, master_seed=0, return_details=True
    )
    repeated = main.full_recovery_infidelity(
        mixed, 2, 0.1, 0.2, 40, 30, 20, master_seed=0, return_details=True
    )
    assert residual["t"] < 2
    assert residual["copy_budget"] == {"bell_1": 80, "bell_2": 60, "tomography": 60, "total": 200}
    assert residual["total_physical_copies"] == 2 * 40 + 2 * 30 + 3 * 20
    assert len(set(residual["stage_seeds"].values())) == 3
    assert np.allclose(residual["rho_recovered"].full(), repeated["rho_recovered"].full())
    assert 0.0 <= residual["infidelity"] <= 1.0
    assert np.isfinite(residual["trace_distance"])

    pure = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
    peeled = main.full_recovery_infidelity(
        pure, 2, 0.4, 0.2, 40, 0, 20, master_seed=904, return_details=True
    )
    assert peeled["t"] == 2
    assert peeled["copy_budget"] == {"bell_1": 80, "bell_2": 0, "tomography": 60, "total": 140}
    assert peeled["copies_tomography"] == 3 * 20
    assert peeled["recovery_output"] is None


def test_fully_peeled_path_and_generic_baseline():
    psi = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
    details = main.full_recovery_infidelity(psi, 2, 0.4, 0.2, 40, 0, 20, seed1=11, seed2=12, return_details=True)
    assert details["t"] == 2
    assert details["recovery_output"] is None
    assert details["is_pure_input"] is True

    first = main.generic_pure_state_local_pauli_tomography(1, qt.basis(2, 0), 20, seed=5, return_details=True)
    second = main.generic_pure_state_local_pauli_tomography(1, qt.basis(2, 0), 20, seed=5, return_details=True)
    assert first["total_samples"] == 60
    assert first["infidelity"] == second["infidelity"]
    assert 0 <= first["infidelity"] <= 1
