import numpy as np
import pytest
import qutip as qt

import main
import main_v2


def _density(state: qt.Qobj) -> qt.Qobj:
    return state * state.dag() if state.isket else state


def test_random_mixed_cebp_generator_invariants_and_reproducibility():
    first = main_v2.random_cebp_state(5, 2, clifford_steps=11, seed=2026)
    second = main_v2.random_cebp_state(5, 2, clifford_steps=11, seed=2026)

    truth = first.oracle_truth
    assert sorted(qubit for block in truth.hidden_partition for qubit in block) == list(range(5))
    assert all(1 <= len(block) <= 2 for block in truth.hidden_partition)
    assert [len(block) for block in truth.hidden_partition] == [
        len(block) for block in second.oracle_truth.hidden_partition
    ]
    assert first.is_ket is False
    assert first.state.dims == [[2] * 5, [2] * 5]
    assert first.state.isherm
    assert np.isclose(first.state.tr(), 1.0)
    assert np.linalg.eigvalsh(first.state.full()).min() >= -1e-10
    assert np.allclose(first.state.full(), second.state.full())
    assert np.allclose(truth.encoder_unitary, second.oracle_truth.encoder_unitary)
    assert first.seed_ledger == second.seed_ledger
    assert first.copy_ledger.total == 0

    for block, block_state in zip(truth.hidden_partition, truth.latent_block_states):
        assert block_state.shape == (2 ** len(block), 2 ** len(block))
        assert block_state.isherm
        assert np.isclose(block_state.tr(), 1.0)
        assert np.linalg.eigvalsh(block_state.full()).min() >= -1e-10


def test_random_pure_cebp_generator_matches_explicit_encoding():
    instance = main_v2.random_cebp_state(
        4,
        3,
        block_sizes=(3, 1),
        pure=True,
        clifford_steps=8,
        seed=91,
    )
    truth = instance.oracle_truth
    assert instance.is_ket is True
    assert instance.state.isket
    # QuTiP 5 canonicalizes a ket's trivial right tensor factors to ``[1]``.
    assert instance.state.dims[0] == [2] * 4
    assert np.prod(instance.state.dims[1]) == 1
    assert np.isclose(instance.state.norm(), 1.0)
    assert truth.hidden_partition == ((0, 1, 2), (3,))
    assert all(block.isket for block in truth.latent_block_states)
    assert truth.latent_state_source == "haar_pure"

    encoder = qt.Qobj(truth.encoder_unitary, dims=[[2] * 4, [2] * 4])
    expected = (encoder * truth.latent_product_state).unit()
    assert np.allclose(_density(instance.state).full(), _density(expected).full())
    assert main_v2.tableau_unitary_convention_holds(
        truth.encoder_tableau, truth.encoder_unitary
    )


def test_explicit_d2_correlated_blocks_are_preserved():
    zero, one = qt.basis(2, 0), qt.basis(2, 1)
    bell = (qt.tensor(zero, zero) + qt.tensor(one, one)).unit()
    bell_density = bell * bell.dag()
    mixed_one = 0.5 * (qt.qeye(2) + 0.4 * qt.sigmay())

    instance = main_v2.random_cebp_state(
        5,
        2,
        block_sizes=(2, 2, 1),
        block_states=(bell, bell_density, mixed_one),
        clifford_steps=7,
        seed=17,
    )
    truth = instance.oracle_truth
    assert truth.hidden_partition == ((0, 1), (2, 3), (4,))
    assert truth.latent_state_source == "user_supplied"
    assert instance.is_ket is False
    assert np.allclose(truth.latent_block_states[0].full(), bell.full())
    assert np.allclose(truth.latent_block_states[1].full(), bell_density.full())
    assert np.allclose(truth.latent_block_states[2].full(), mixed_one.full())
    assert truth.latent_product_state.shape == (32, 32)


def test_explicit_d3_correlated_block_is_preserved():
    zero, one = qt.basis(2, 0), qt.basis(2, 1)
    ghz = (
        qt.tensor(zero, zero, zero) + qt.tensor(one, one, one)
    ).unit()
    plus = (zero + one).unit()
    instance = main_v2.random_cebp_state(
        4,
        3,
        block_sizes=(3, 1),
        block_states=(ghz, plus),
        clifford_steps=6,
        seed=31,
    )
    assert instance.oracle_truth.hidden_partition == ((0, 1, 2), (3,))
    assert np.allclose(instance.oracle_truth.latent_block_states[0].full(), ghz.full())
    assert instance.is_ket is True


def test_rank_one_density_uses_mixed_representation_backend():
    plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    rank_one_density = plus * plus.dag()
    instance = main_v2.random_cebp_state(
        1,
        1,
        block_states=(rank_one_density,),
        clifford_steps=1,
        seed=22,
    )
    assert np.isclose((instance.state * instance.state).tr(), 1.0)
    assert instance.state.isoper
    assert instance.is_ket is False
    assert instance.measurement_source.is_ket is False
    assert not hasattr(instance, "is_pure")


def test_all_supplied_kets_preserve_ket_representation():
    plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    instance = main_v2.random_cebp_state(
        2,
        1,
        block_states=(qt.basis(2, 0), plus),
        pure=False,
        clifford_steps=4,
        seed=4,
    )
    assert instance.is_ket is True
    assert instance.state.isket


def test_learner_view_excludes_oracle_truth_and_exact_state_field():
    instance = main_v2.random_cebp_state(3, 2, seed=7)
    view = instance.learner_view()
    assert view.n == 3 and view.d == 2
    assert view.measurement_source.n == 3
    assert view.measurement_source.is_ket is False
    assert not hasattr(view, "oracle_truth")
    assert not hasattr(view, "state")
    assert "_state" not in repr(view.measurement_source)


def test_oracle_hidden_block_support_uses_encoder_tableau():
    instance = main_v2.random_cebp_state(
        4, 2, block_sizes=(2, 2), clifford_steps=9, seed=12
    )
    truth = instance.oracle_truth
    physical_pauli = "XYZI"
    latent = truth.latent_pauli_vector(physical_pauli)
    expected = (
        truth.encoder_tableau @ main.pauli_to_symplectic_col(physical_pauli)
    ) % 2
    assert np.array_equal(latent, expected)

    n = 4
    touched_qubits = {
        qubit for qubit in range(n) if latent[qubit] or latent[n + qubit]
    }
    expected_blocks = tuple(
        index
        for index, block in enumerate(truth.hidden_partition)
        if touched_qubits.intersection(block)
    )
    assert truth.hidden_block_support(physical_pauli) == expected_blocks
    assert truth.hidden_block_support("IIII") == ()


@pytest.mark.parametrize("n", (1, 2, 3, 4))
def test_encoder_tableau_unitary_convention(n):
    instance = main_v2.random_cebp_state(
        n, max(1, min(2, n)), pure=True, clifford_steps=max(1, 3 * n), seed=100 + n
    )
    truth = instance.oracle_truth
    assert main.is_symplectic(truth.encoder_tableau)
    assert main_v2.tableau_unitary_convention_holds(
        truth.encoder_tableau, truth.encoder_unitary
    )


def test_d_one_generator_has_single_qubit_hidden_blocks():
    instance = main_v2.random_cebp_state(
        4, 1, pure=True, clifford_steps=10, seed=33
    )
    assert instance.oracle_truth.hidden_partition == ((0,), (1,), (2,), (3,))
    assert all(state.shape == (2, 1) for state in instance.oracle_truth.latent_block_states)
    assert instance.state.shape == (16, 1)


def test_copy_ledger_validates_counts_and_totals():
    ledger = main_v2.CopyLedger.from_mapping({"bell_1": 20, "sign": 3})
    assert ledger.total == 23
    assert ledger.as_dict() == {"bell_1": 20, "sign": 3}
    with pytest.raises(ValueError):
        main_v2.CopyLedger((('bad', -1),))
    with pytest.raises(ValueError):
        main_v2.CopyLedger((('same', 1), ('same', 2)))


def test_generator_rejects_invalid_model_inputs():
    with pytest.raises(ValueError, match="sum"):
        main_v2.random_cebp_state(3, 2, block_sizes=(1, 1), seed=1)
    with pytest.raises(ValueError, match="at most"):
        main_v2.random_cebp_state(3, 1, block_sizes=(2, 1), seed=1)
    with pytest.raises(ValueError, match="do not match"):
        main_v2.random_cebp_state(
            3, 2, block_sizes=(1, 2), block_states=(qt.qeye(4) / 4, qt.qeye(2) / 2)
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        main_v2.random_cebp_state(
            1,
            1,
            block_states=(np.diag([1.2, -0.2]),),
            clifford_steps=1,
            seed=1,
        )
    with pytest.raises(ValueError, match="pure=True"):
        main_v2.random_cebp_state(
            1, 1, block_states=(qt.qeye(2) / 2,), pure=True, seed=1
        )
    with pytest.raises(ValueError, match="nonnegative"):
        main_v2.random_cebp_state(1, 1, seed=-1)


def test_validate_instance_detects_tableau_unitary_mismatch():
    instance = main_v2.random_cebp_state(2, 2, pure=True, clifford_steps=5, seed=8)
    truth = instance.oracle_truth
    assert main_v2.tableau_unitary_convention_holds(
        truth.encoder_tableau, truth.encoder_unitary
    )
    non_clifford_phase = np.diag([1.0, 1.0, 1.0, np.exp(0.17j)])
    assert not main_v2.tableau_unitary_convention_holds(
        truth.encoder_tableau, truth.encoder_unitary @ non_clifford_phase
    )
