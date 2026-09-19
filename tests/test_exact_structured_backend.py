"""Dense-reference and 10q scalability checks for the exact structured backend."""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest
import qutip as qt

import main_v2
from cebp_compact import SignedClifford, StructuredCEBPState


def _dense_score_table(state, n):
    return np.asarray(
        [
            main_v2.debug_exact_pauli_score(
                state, main_v2.pauli_index_to_string(index, n)
            )
            for index in range(4**n)
        ],
        dtype=np.float64,
    )


@pytest.mark.parametrize("seed", (17, 29, 41))
def test_structured_all_scores_match_dense_random_small_instances(seed):
    instance = main_v2.random_cebp_state(
        3, 2, block_sizes=(2, 1), clifford_steps=11, seed=seed
    )
    scores = main_v2.debug_exact_score_source(instance).all_scores()
    reference = _dense_score_table(
        instance.materialize_state_debug(max_qubits=3), 3
    )
    assert scores.shape == (4**3,)
    assert scores.dtype == np.float64 and scores.flags.c_contiguous
    assert not scores.flags.writeable and scores[0] == 1.0
    assert np.max(np.abs(scores - reference)) < 1.0e-12


def test_clifford_mapped_score_indices_follow_pauli_string_order():
    ket_a = qt.Qobj(
        np.asarray([1, 1j, -0.5, 0.25j], dtype=complex),
        dims=[[2, 2], [1, 1]],
    ).unit()
    ket_b = (qt.basis(2, 0) + np.exp(0.37j) * qt.basis(2, 1)).unit()
    encoder = SignedClifford.from_gates(
        3, (("H", 0), ("S", 1), ("CNOT", 0, 2), ("CNOT", 2, 1))
    )
    structured = StructuredCEBPState(
        3, ((0, 1), (2,)), (ket_a, ket_b), encoder
    )
    scores = structured.all_squared_pauli_scores()
    dense = structured.materialize_dense_debug(max_qubits=3)
    for index in range(4**3):
        pauli = main_v2.pauli_index_to_string(index, 3)
        assert main_v2.pauli_string_to_index(pauli) == index
        assert scores[index] == pytest.approx(
            main_v2.debug_exact_pauli_score(dense, pauli), abs=1.0e-12
        )


def test_peeled_structured_score_table_matches_dense_reference_and_zero_copies():
    instance = main_v2.random_cebp_state(
        2,
        1,
        block_states=(qt.basis(2, 0), qt.qeye(2) / 2),
        clifford_steps=5,
        seed=10,
    )
    peeling = main_v2.debug_exact_certified_stabilizer_peeling(
        instance,
        main_v2.PeelingConfig(
            h_min=0.6,
            h_max=0.8,
            eta=0.005,
            max_dense_debug_qubits=2,
        ),
    )
    source = main_v2.debug_exact_peeled_score_source(instance, peeling)
    unitary = qt.Qobj(peeling.U_stab, dims=[[2, 2], [2, 2]])
    direct = unitary.dag() * instance.state * unitary
    assert np.max(np.abs(source.all_scores() - _dense_score_table(direct, 2))) < 1.0e-12
    assert peeling.copy_ledger.total == source.copy_ledger.total == 0


def _identity_only_peeling(n):
    score_map = {
        "".join(chars): 1.0 if all(char == "I" for char in chars) else 0.0
        for chars in itertools.product("IXYZ", repeat=n)
    }
    return main_v2.certified_stabilizer_peeling_v2(
        main_v2.PauliScoreMap(
            n=n, score_map=score_map, uniform_radius=0.0, frame="physical"
        ),
        main_v2.PeelingConfig(
            h_min=0.6, h_max=0.8, eta=0.005, max_dense_debug_qubits=n
        ),
    )


def test_structured_signed_moments_and_q2_q3_cumulants_match_dense(monkeypatch):
    instance = main_v2.random_cebp_state(
        3, 3, block_sizes=(3,), clifford_steps=7, seed=91
    )
    peeling = _identity_only_peeling(3)
    peeled = main_v2._peeled_measurement_source(instance.learner_view(), peeling)
    dense = peeled._state_for_backend(max_dense_debug_qubits=3)
    sign, product = main_v2.hermitian_pauli_product("XXI", "ZZI")
    assert (sign, product) == (-1, "YYI")
    assert main_v2.debug_exact_residual_signed_moment(
        instance, peeling, sign, product
    ) == pytest.approx(
        sign * main_v2.debug_exact_pauli_expectation(dense, product), abs=1.0e-12
    )

    original = main_v2._peeled_measurement_source
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(main_v2, "_peeled_measurement_source", counted)
    interface = main_v2.DebugExactResidualCumulantInterface(instance, peeling)
    for observables in (("XXI", "ZZI"), ("XXI", "ZZI", "IIZ")):
        moments = {frozenset(): 1.0}
        for size in range(1, len(observables) + 1):
            for positions in itertools.combinations(range(len(observables)), size):
                local_sign, pauli = main_v2.hermitian_pauli_product(
                    *(observables[position] for position in positions)
                )
                moments[frozenset(positions)] = local_sign * (
                    main_v2.debug_exact_pauli_expectation(dense, pauli)
                )
        reference = main_v2.mixed_cumulant_from_moments(len(observables), moments)
        assert interface.query(observables) == pytest.approx(reference, abs=1.0e-12)
    assert len(calls) == 1
    assert interface.realized_copies == 0
    assert all(copies == 0 for copies in interface.copies_by_order.values())


def test_frozen_10q_score_table_and_peeling_do_not_use_dense_bridge(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    case_dir = root / "Jobs" / "06" / "n=10_d=3"
    spec = importlib.util.spec_from_file_location(
        "exact_backend_10q_instance_io", case_dir / "instance_io.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    instance, _metadata = module.load_instance(
        case_dir / "states" / "init_000.npz", expected_init_id=0
    )

    def forbidden_dense_bridge(*_args, **_kwargs):
        raise AssertionError("10q exact structural stage invoked the dense state bridge")

    monkeypatch.setattr(
        main_v2.SimulatorMeasurementSource,
        "_state_for_backend",
        forbidden_dense_bridge,
    )
    source = main_v2.debug_exact_score_source(instance)
    scores = source.all_scores()
    assert scores.shape == (4**10,)
    assert scores.dtype == np.float64 and scores.nbytes == 4**10 * 8
    assert scores.nbytes == 8 * 1024 * 1024 and scores[0] == 1.0
    peeling = main_v2.debug_exact_certified_stabilizer_peeling(
        instance,
        main_v2.PeelingConfig(
            h_min=0.6,
            h_max=0.8,
            eta=0.04,
            materialize_dense_clifford=False,
        ),
    )
    assert peeling.copy_ledger.total == 0
    assert peeling.enumeration_diagnostics.score_array_bytes == scores.nbytes


def test_small_exact_operational_decoder_keeps_physical_trace_distance_result():
    ket = (
        qt.basis(2, 0) + np.exp(1j * np.pi / 4) * qt.basis(2, 1)
    ).unit()
    instance = main_v2.random_cebp_state(
        1,
        1,
        block_states=(ket,),
        clifford_steps=0,
        seed=1,
        max_dense_debug_qubits=1,
    )
    result = main_v2.debug_exact_operational_diagnostic(
        instance,
        h_min=0.6,
        h_max=0.8,
        theta=0.4,
        eta_test=0.1,
        peeling_grid_intervals=5,
        max_dense_debug_qubits=1,
    )
    assert result.realized_physical_copies == 0
    assert result.localization_registers == (((0,), (0,)),)
    assert result.structural_trace_distance < 1.0e-12
