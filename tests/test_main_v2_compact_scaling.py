from dataclasses import replace
import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import qutip as qt

import cebp_compact
import main_v2
from Optimization import CandidateParameters, OptimizationConfig, derive_candidate
from Optimization.parameterization import candidate_to_end_to_end_config
from Optimization import objective as optimization_objective


def _reference_score_sums(outcomes):
    n, rounds, _ = outcomes.shape
    result = np.empty(4**n, dtype=np.int64)
    for index in range(4**n):
        pauli = cebp_compact.pauli_index_to_string(index, n)
        parity = np.ones(rounds, dtype=np.int64)
        for qubit, character in enumerate(pauli):
            if character != "I":
                parity *= outcomes[qubit, :, main_v2._v1.AXIS_TO_K[character]]
        result[index] = parity.sum()
    return result


@pytest.mark.parametrize("n", range(1, 7))
def test_pauli_index_roundtrip_order_and_symplectic_conversion(n):
    old_order = main_v2._all_pauli_strings(n)
    assert cebp_compact.pauli_index_to_string(0, n) == "I" * n
    for index, pauli in enumerate(old_order):
        assert cebp_compact.pauli_index_to_string(index, n) == pauli
        assert cebp_compact.pauli_string_to_index(pauli) == index
        vector = cebp_compact.pauli_index_to_symplectic_int(index, n)
        assert vector == main_v2._v1.Symplectic(n).to_int(pauli)
        assert cebp_compact.symplectic_int_to_pauli_index(vector, n) == index


@pytest.mark.parametrize("n", range(1, 7))
def test_integer_score_transform_matches_reference_and_threshold_parallelism(n):
    rng = np.random.default_rng(1700 + n)
    categories = rng.integers(0, 4, size=(n, 97))
    outcomes = np.asarray(main_v2._v1._BELL_EIG_TABLE, dtype=np.int8)[categories]
    compact = cebp_compact.bell_score_sums_from_outcomes(outcomes)
    assert np.array_equal(compact, _reference_score_sums(outcomes))
    scores = compact.astype(float) / outcomes.shape[1]
    serial = cebp_compact.threshold_span_reduction(
        scores, 0.1, n, workers=1, chunk_size=max(1, 4**n // 5)
    )
    parallel = cebp_compact.threshold_span_reduction(
        scores, 0.1, n, workers=min(4, 4**n), chunk_size=max(1, 4**n // 5)
    )
    assert serial == parallel


def test_peeling_and_recovery_serial_parallel_structural_identity():
    n = 4
    scores = {pauli: 0.0 for pauli in main_v2._all_pauli_strings(n)}
    scores["I" * n] = 1.0
    scores["ZIII"] = 0.9
    scores["IZII"] = 0.85
    peeling_source = main_v2.PauliScoreMap(
        n=n, score_map=scores, uniform_radius=0.0, frame="physical"
    )
    base = main_v2.PeelingConfig(
        0.6,
        0.8,
        0.01,
        return_details=True,
        materialize_dense_clifford=False,
    )
    serial = main_v2.certified_stabilizer_peeling_v2(
        peeling_source,
        replace(
            base,
            enumeration_execution=main_v2.EnumerationExecutionConfig(workers=1),
        ),
    )
    parallel_results = tuple(
        main_v2.certified_stabilizer_peeling_v2(
            peeling_source,
            replace(
                base,
                enumeration_execution=main_v2.EnumerationExecutionConfig(
                    workers=workers, chunk_size=37
                ),
            ),
        )
        for workers in (2, 4)
    )
    assert all(serial == parallel for parallel in parallel_results)

    recovery_scores = dict(scores)
    recovery_scores.update({"ZZXI": 0.8, "ZZZI": 0.75, "ZZYI": 0.7})
    recovery_source = main_v2.PauliScoreMap(
        n=n, score_map=recovery_scores, uniform_radius=0.0, frame="peeled"
    )
    recovery_base = main_v2.RecoveryConfig(
        theta=0.5,
        return_details=True,
        allow_uncalibrated_peeling=True,
        allow_margin_failure=True,
    )
    recovered_serial = main_v2.rank_guided_sector_recovery(
        recovery_source,
        serial,
        replace(
            recovery_base,
            enumeration_execution=main_v2.EnumerationExecutionConfig(workers=1),
        ),
    )
    recovered_parallel = tuple(
        main_v2.rank_guided_sector_recovery(
            recovery_source,
            parallel_results[index],
            replace(
                recovery_base,
                enumeration_execution=main_v2.EnumerationExecutionConfig(
                    workers=workers
                ),
            ),
        )
        for index, workers in enumerate((2, 4))
    )
    assert all(
        recovered_serial == parallel for parallel in recovered_parallel
    )


@pytest.mark.parametrize("n", (1, 2, 3))
def test_signed_clifford_dense_sign_composition_and_inverse(n):
    gates, tableau, unitary = main_v2._v1.random_clifford_gate(
        n, steps=7 * n, seed=900 + n
    )
    compact = cebp_compact.SignedClifford(n, tableau, tuple(gates))
    for pauli in main_v2._all_pauli_strings(n):
        image = compact.conjugate(pauli)
        actual = (
            unitary.conj().T
            @ main_v2._v1._qutip_pauli_op(n, pauli).full()
            @ unitary
        )
        expected = (
            image.coefficient
            * main_v2._v1._qutip_pauli_op(n, image.pauli).full()
        )
        assert np.allclose(actual, expected, atol=1e-10)
        identity_image = compact.compose(compact.inverse()).conjugate(pauli)
        assert identity_image.phase_exponent == 0 and identity_image.pauli == pauli


@pytest.mark.parametrize("n", (3, 4, 5, 6))
def test_structured_pauli_commuting_and_bell_probabilities_match_dense(n):
    instance = main_v2.random_cebp_state(
        n,
        min(3, n),
        pure=False,
        clifford_steps=4 * n,
        seed=1200 + n,
    )
    structured = instance.measurement_source._structured_state
    assert structured is not None
    dense = instance.state
    rng = np.random.default_rng(1300 + n)
    indices = (
        np.arange(4**n)
        if n <= 4
        else rng.choice(4**n, size=64, replace=False)
    )
    for index in indices:
        pauli = cebp_compact.pauli_index_to_string(int(index), n)
        assert structured.expectation(pauli) == pytest.approx(
            main_v2.debug_exact_pauli_expectation(dense, pauli), abs=1e-9
        )
    commuting = tuple(
        "I" * qubit + "Z" + "I" * (n - qubit - 1)
        for qubit in range(min(3, n))
    )
    structured_joint = structured.commuting_probabilities(commuting)
    dense_source = main_v2.SimulatorMeasurementSource(n, dense.isket, dense)
    _outcomes, dense_joint = main_v2._commuting_tuple_probabilities_backend(
        dense_source, commuting
    )
    assert np.allclose(structured_joint, dense_joint, atol=1e-9)
    assert np.allclose(
        structured.bell_probabilities(),
        main_v2._v1._bell_outcome_probabilities(dense, n),
        atol=1e-9,
    )


def test_materialize_false_end_to_end_never_calls_dense_materializers(monkeypatch):
    instance = main_v2.random_cebp_state(
        2,
        2,
        block_sizes=(2,),
        block_states=(qt.qeye(4) / 4,),
        clifford_steps=3,
        seed=1401,
    )
    monkeypatch.setattr(
        cebp_compact.StructuredCEBPState,
        "materialize_dense_debug",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("structured state was materialized")
        ),
    )
    monkeypatch.setattr(
        cebp_compact.SignedClifford,
        "materialize_dense_debug",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Clifford was materialized")
        ),
    )
    config = main_v2.EndToEndConfig(
        epsilon=0.5,
        delta=0.1,
        seed=1402,
        materialize_dense_estimator=False,
        max_reserved_copies=20_000,
        max_realized_copies=20_000,
        allow_uncertified_execution=True,
        simulation_backend="batched_counts",
        peeling_override=main_v2.PeelingConfig(0.6, 0.8, 0.02, 5_000, 0.1),
        recovery_override=main_v2.RecoveryConfig(
            0.5,
            500,
            0.1,
            allow_uncalibrated_peeling=True,
            allow_margin_failure=True,
        ),
        grouping_override=main_v2.GroupingConfig(
            ell_grp=2,
            eta_test=0.1,
            tau_kappa=0.05,
            delta_grp_ordinary=0.1,
            allow_uncalibrated_recovery=True,
            allow_no_false_merge_margin_failure=True,
        ),
        syndrome_override=main_v2.SyndromeConfig(0.1, 0.6, M_sgn=10),
        tomography_override=main_v2.TomographyConfig(
            2.0,
            0.1,
            allow_uncertified_localization=True,
            materialize_localized_estimator=False,
        ),
    )
    result = main_v2.full_cebp_tomography(instance.learner_view(), config=config)
    assert result.success and result.compact_estimator is not None
    assert result.decoded_density is None and result.localized_estimator is None
    assert result.peeling.U_stab is None
    assert result.localization.U_rec is None and result.localization.bar_U_rec is None
    assert result.tomography.localized_empirical_estimator is None
    assert instance.measurement_source._state is None
    timings = result.performance_diagnostics.as_dict()
    assert {
        "peeling.bell_record_construction",
        "peeling.bell_score_transform",
        "peeling.threshold_scan",
        "recovery.bell_record_construction",
        "recovery.recovery_preparation_sort",
        "recovery.serial_ranked_recovery_loop",
        "peeling",
        "recovery",
        "grouping",
        "localization",
        "syndrome",
        "tomography",
        "overall_end_to_end",
    } <= set(timings)
    assert all(value >= 0.0 for value in timings.values())
    with pytest.raises(ValueError, match="trace-error debug resource guard"):
        main_v2.debug_end_to_end_trace_error(
            result, instance, max_dense_qubits=1
        )
    assert all(
        not hasattr(instance.learner_view(), attribute)
        for attribute in (
            "hidden_partition",
            "latent_block_states",
            "encoder_tableau",
            "encoder_gates",
        )
    )


def test_resource_guard_separation_and_nested_parallelism_policy():
    config = OptimizationConfig(
        max_dense_qubits=3,
        max_enumeration_qubits=7,
        max_oracle_dense_qubits=2,
        inner_enumeration_workers=2,
        candidate_workers=1,
    )
    base = CandidateParameters(
        0.0334, 0.20, 0.001, 1.25, 1.30, 0.80, 0.10, 0.12,
        0.10, 0.60, 0.80,
    )
    derived = derive_candidate(
        base, n=3, d=2, total_copies=config.total_copies,
        optimization_config=config,
    )
    learner = candidate_to_end_to_end_config(
        derived,
        d=2,
        learner_seed=1,
        total_copies=config.total_copies,
        optimization_config=config,
    )
    assert learner.max_enumeration_qubits == 7
    assert learner.max_dense_debug_qubits == 2
    assert learner.enumeration_execution.workers == 2
    with pytest.raises(ValueError, match="Nested parallelism"):
        OptimizationConfig(inner_enumeration_workers=2, candidate_workers=2)


def test_dense_oracle_unavailability_is_explicit_not_learner_failure(monkeypatch):
    config = OptimizationConfig(
        total_copies=500_000,
        max_enumeration_qubits=4,
        max_oracle_dense_qubits=2,
    )
    base = CandidateParameters(
        0.0334, 0.20, 0.001, 1.25, 1.30, 0.80, 0.10, 0.12,
        0.10, 0.60, 0.80,
    )
    derived = derive_candidate(
        base, n=3, d=2, total_copies=config.total_copies,
        optimization_config=config,
    )
    instance = main_v2.random_cebp_state(3, 2, seed=1501)
    fake = SimpleNamespace(
        theorem_certified=False,
        realized_total=0,
        realized_copy_ledger=main_v2.CopyLedger(),
        estimator_available=True,
        success=True,
        failure_stage=None,
        failure_reason=None,
        peeling=None,
        grouping=None,
        localization=None,
        budget_truncated=False,
        truncated_stages=(),
    )
    monkeypatch.setattr(
        optimization_objective, "full_cebp_tomography", lambda *_a, **_k: fake
    )
    monkeypatch.setattr(
        optimization_objective,
        "debug_end_to_end_trace_error",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("dense oracle should not run")
        ),
    )
    evaluation = optimization_objective.evaluate_candidate_on_seed(
        instance,
        derived,
        learner_seed=1,
        total_copies=config.total_copies,
        optimization_config=config,
    )
    assert evaluation.operational_success
    assert not evaluation.oracle_loss_available
    assert evaluation.oracle_loss_requested
    assert not evaluation.loss_computed
    assert not evaluation.ranking_eligible
    assert evaluation.loss is None
    assert evaluation.trace_distance is None
    assert evaluation.failure_stage is None
    assert evaluation.failure_reason is None
    assert "max_oracle_dense_qubits=2" in evaluation.oracle_unavailable_reason
