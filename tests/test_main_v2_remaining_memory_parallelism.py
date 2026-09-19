from dataclasses import replace
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import qutip as qt

import cebp_compact
import main_v2
import Optimization.objective as optimization_objective
from Optimization import (
    CandidateParameters,
    OptimizationConfig,
    aggregate_candidate_evaluations,
    derive_candidate,
    optimize_cebp_parameters,
)


BASE = CandidateParameters(
    0.0334, 0.20, 0.001, 1.25, 1.30, 0.80, 0.10, 0.12,
    0.10, 0.60, 0.80,
)


def _previous_structured_bell_probabilities(structured):
    scores = np.empty(4**structured.n, dtype=np.float64)
    for index in range(scores.size):
        pauli = cebp_compact.pauli_index_to_string(index, structured.n)
        scores[index] = structured.expectation(pauli) ** 2
    characters = np.column_stack(
        (
            np.ones(4, dtype=np.int8),
            np.asarray(main_v2._v1._BELL_EIG_TABLE, dtype=np.int8),
        )
    ).astype(float)
    tensor = scores.reshape((4,) * structured.n)
    for axis in range(structured.n):
        tensor = np.tensordot(characters, tensor, axes=([1], [axis])) / 4.0
        tensor = np.moveaxis(tensor, 0, axis)
    result = np.asarray(tensor, dtype=float).reshape(-1, order="F")
    result = np.clip(result, 0.0, None)
    return result / result.sum()


def _record_from_categories(categories, *, compressed):
    categories = np.asarray(categories, dtype=np.int64)
    n, rounds = categories.shape
    indices = np.zeros(rounds, dtype=np.int64)
    multiplier = 1
    for qubit in range(n):
        indices += multiplier * categories[qubit]
        multiplier *= 4
    counts = np.bincount(indices, minlength=4**n).astype(np.int64)
    outcomes = np.asarray(main_v2._v1._BELL_EIG_TABLE, dtype=np.int8)[categories]
    return main_v2.BellScoreRecord(
        n=n,
        rounds=rounds,
        frame="physical",
        uniform_radius=0.1,
        zeta_bs=0.1,
        backend="synthetic",
        seed=1,
        pool_name="peeling_bell_pool",
        copy_ledger=main_v2.CopyLedger((("peeling_bell_pool", 2 * rounds),)),
        outcomes=None if compressed else outcomes,
        simulation_backend="batched_counts" if compressed else "legacy_shotwise",
        category_counts=counts if compressed else None,
    )


def _t0_peeling(n):
    scores = {pauli: 0.0 for pauli in main_v2._all_pauli_strings(n)}
    scores["I" * n] = 1.0
    result = main_v2.certified_stabilizer_peeling_v2(
        main_v2.PauliScoreMap(n, scores, 0.0, frame="physical"),
        main_v2.PeelingConfig(
            0.6, 0.8, 0.01, materialize_dense_clifford=False
        ),
    )
    assert result.success and result.t == 0
    return result


def test_score_guard_rejects_before_integer_transform(monkeypatch):
    record = _record_from_categories(
        np.random.default_rng(1).integers(0, 4, size=(3, 32)),
        compressed=True,
    )
    called = False

    def forbidden(_self):
        nonlocal called
        called = True
        raise AssertionError("score transform allocated before its guard")

    monkeypatch.setattr(main_v2.BellScoreRecord, "all_score_sums", forbidden)
    config = main_v2.PeelingConfig(
        0.6,
        0.8,
        0.01,
        materialize_dense_clifford=False,
        enumeration_execution=main_v2.EnumerationExecutionConfig(
            max_score_array_bytes=1
        ),
    )
    with pytest.raises(ValueError, match=r"required=512, limit=1"):
        main_v2.certified_stabilizer_peeling_v2(record, config)
    assert not called


def test_integer_cutoff_exactly_preserves_float_division_threshold_semantics():
    values = np.arange(-101, 102, dtype=np.int64)
    for rounds in (3, 7, 97):
        for threshold in (0.1, 1 / 3, np.nextafter(0.5, 1.0), 0.9):
            cutoff = main_v2._score_threshold_cutoff(
                values, float(rounds), threshold
            )
            assert np.array_equal(
                values >= cutoff,
                values.astype(np.float64) / rounds >= threshold,
            )


def test_peeling_preflight_rejects_before_sampling_and_copies(monkeypatch):
    instance = main_v2.random_cebp_state(3, 2, seed=2001)
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("peeling sampler must not run")

    monkeypatch.setattr(main_v2, "sample_bell_scores", forbidden)
    config = main_v2.PeelingConfig(
        0.6, 0.8, 0.01, M1=10, max_enumeration_qubits=2
    )
    with pytest.raises(ValueError, match="before|exceeds|max_enumeration_qubits"):
        main_v2.empirical_certified_stabilizer_peeling(
            instance.learner_view(), config
        )
    assert not called
    assert instance.copy_ledger.total == 0


def test_peeling_score_cap_rejects_before_sampling(monkeypatch):
    instance = main_v2.random_cebp_state(3, 2, seed=2004)
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("peeling sampler must not run")

    monkeypatch.setattr(main_v2, "sample_bell_scores", forbidden)
    config = main_v2.PeelingConfig(
        0.6,
        0.8,
        0.01,
        M1=10,
        enumeration_execution=main_v2.EnumerationExecutionConfig(
            max_score_array_bytes=1
        ),
    )
    with pytest.raises(ValueError, match="score workspace"):
        main_v2.empirical_certified_stabilizer_peeling(
            instance.learner_view(), config
        )
    assert not called


def test_recovery_preflight_rejects_before_fresh_sampling_and_copies(monkeypatch):
    instance = main_v2.random_cebp_state(3, 2, seed=2002)
    peeling = _t0_peeling(3)
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("recovery sampler must not run")

    monkeypatch.setattr(main_v2, "sample_peeled_bell_scores", forbidden)
    config = main_v2.RecoveryConfig(
        0.2,
        M2=10,
        max_enumeration_qubits=2,
        allow_uncalibrated_peeling=True,
        allow_margin_failure=True,
    )
    with pytest.raises(ValueError, match="max_enumeration_qubits"):
        main_v2.empirical_rank_guided_sector_recovery(
            instance.learner_view(), peeling, config
        )
    assert not called
    assert instance.copy_ledger.total == 0


def test_structured_bell_workspace_guard_precedes_sampler(monkeypatch):
    instance = main_v2.random_cebp_state(4, 2, seed=2003)
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("structured sampler must not run")

    monkeypatch.setattr(
        main_v2.SimulatorMeasurementSource, "_sample_bell_for_backend", forbidden
    )
    config = main_v2.PeelingConfig(
        0.6,
        0.8,
        0.01,
        M1=10,
        enumeration_execution=main_v2.EnumerationExecutionConfig(
            max_structured_bell_workspace_bytes=1
        ),
    )
    with pytest.raises(ValueError, match="structured Bell workspace"):
        main_v2.empirical_certified_stabilizer_peeling(
            instance.learner_view(), config
        )
    assert not called


def test_accepted_basis_streams_without_flatnonzero(monkeypatch):
    n = 3
    scores = {pauli: 0.0 for pauli in main_v2._all_pauli_strings(n)}
    scores.update({"III": 1.0, "IZI": 0.9, "ZII": 0.85})
    source = main_v2.PauliScoreMap(n, scores, 0.0, frame="physical")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("normal accepted-basis selection allocated survivors")

    monkeypatch.setattr(np, "flatnonzero", forbidden)
    result = main_v2.certified_stabilizer_peeling_v2(
        source,
        main_v2.PeelingConfig(
            0.6,
            0.8,
            0.01,
            return_details=False,
            materialize_dense_clifford=False,
        ),
    )
    assert result.success
    assert result.generators == ("IZI", "ZII")
    assert result.inner_set == result.outer_set == ()


@pytest.mark.parametrize("n", (3, 4, 5, 6))
@pytest.mark.parametrize("pure", (True, False))
def test_numeric_structured_complete_bell_table_matches_dense(n, pure):
    instance = main_v2.random_cebp_state(
        n, min(3, n), pure=pure, clifford_steps=3 * n, seed=2100 + 10 * n + pure
    )
    structured = instance.oracle_truth._structured_state
    dense = instance.materialize_state_debug(max_qubits=6)
    numeric = structured.bell_probabilities()
    assert np.allclose(
        numeric,
        main_v2._v1._bell_outcome_probabilities(dense, n),
        atol=2e-12,
    )
    assert np.allclose(
        numeric, _previous_structured_bell_probabilities(structured), atol=2e-12
    )


def test_structured_bell_hot_path_never_converts_pauli_strings(monkeypatch):
    instance = main_v2.random_cebp_state(5, 2, seed=2201)
    structured = instance.oracle_truth._structured_state

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Pauli strings entered the complete Bell hot path")

    monkeypatch.setattr(cebp_compact, "pauli_index_to_string", forbidden)
    probabilities, diagnostics = structured.bell_probabilities_with_diagnostics()
    assert probabilities.shape == (4**5,)
    assert np.isclose(probabilities.sum(), 1.0)
    assert {name for name, _seconds in diagnostics} == {
        "latent_squared_moments",
        "symplectic_score_permutation",
        "bell_probability_transform",
    }


def test_compressed_and_raw_bell_records_have_identical_statistics():
    categories = np.random.default_rng(2301).integers(0, 4, size=(5, 10_000))
    raw = _record_from_categories(categories, compressed=False)
    compressed = _record_from_categories(categories, compressed=True)
    assert np.array_equal(raw.all_score_sums(), compressed.all_score_sums())
    assert np.array_equal(raw.all_scores(), compressed.all_scores())
    assert compressed.outcomes is None
    assert compressed.category_counts.nbytes < raw.outcomes.nbytes


def test_batched_structured_sampling_returns_counts_not_raw_shots():
    instance = main_v2.random_cebp_state(4, 2, seed=2302)
    record = main_v2.sample_bell_scores(
        instance.learner_view(),
        50_000,
        seed=2303,
        simulation_backend="batched_counts",
    )
    assert record.outcomes is None
    assert record.category_counts.shape == (4**4,)
    assert int(record.category_counts.sum()) == 50_000
    assert record.copy_ledger.total == 100_000


def test_dense_debug_guards_precede_construction_and_do_not_cache(monkeypatch):
    large = main_v2.random_cebp_state(9, 3, seed=2401)
    with pytest.raises(ValueError, match="resource guard"):
        _ = large.state
    with pytest.raises(ValueError, match="resource guard"):
        _ = large.oracle_truth.encoder_unitary
    with pytest.raises(ValueError, match="resource guard"):
        _ = large.oracle_truth.latent_product_state

    small = main_v2.random_cebp_state(3, 2, seed=2402)
    source = small.measurement_source
    assert source._state is None
    materialized = source._state_for_backend(max_dense_debug_qubits=3)
    assert materialized.shape[0] == 2**3
    assert source._state is None
    small.oracle_truth.materialize_encoder_unitary_debug(max_qubits=3)
    with pytest.raises(ValueError, match="encoder debug resource guard"):
        small.oracle_truth.materialize_encoder_unitary_debug(max_qubits=2)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dense constructor called above guard")

    monkeypatch.setattr(
        cebp_compact.StructuredCEBPState, "materialize_dense_debug", forbidden
    )
    with pytest.raises(ValueError, match="resource guard"):
        large.materialize_state_debug(max_qubits=8)


def test_unavailable_oracle_loss_is_not_aggregated_or_ranked(monkeypatch):
    config = OptimizationConfig(
        total_copies=500_000,
        tuning_seeds=(1,),
        holdout_seeds=(2,),
        halving_seed_counts=(1,),
        number_of_candidates=2,
        max_enumeration_qubits=4,
        max_oracle_dense_qubits=2,
    )
    derived = derive_candidate(
        BASE, n=3, d=2, total_copies=config.total_copies,
        optimization_config=config,
    )
    instance = main_v2.random_cebp_state(3, 2, seed=2501)
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
    evaluation = optimization_objective.evaluate_candidate_on_seed(
        instance, derived, 1, config.total_copies, config
    )
    assert evaluation.operational_success
    assert evaluation.loss is None
    assert not evaluation.loss_computed
    assert not evaluation.ranking_eligible
    with pytest.raises(ValueError, match="not ranking-eligible"):
        aggregate_candidate_evaluations("candidate-a", (evaluation, evaluation))

    with pytest.raises(ValueError, match="requires exact trace-distance ranking"):
        optimize_cebp_parameters(
            instance,
            config.total_copies,
            config,
            initial_candidates=(BASE, replace(BASE, tomography_weight=3.0)),
        )


def test_available_oracle_loss_remains_computed(monkeypatch):
    config = OptimizationConfig(
        total_copies=500_000,
        max_enumeration_qubits=3,
        max_oracle_dense_qubits=3,
    )
    derived = derive_candidate(
        BASE, n=3, d=2, total_copies=config.total_copies,
        optimization_config=config,
    )
    instance = main_v2.random_cebp_state(3, 2, seed=2502)
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
        lambda *_a, **_k: 0.25,
    )
    evaluation = optimization_objective.evaluate_candidate_on_seed(
        instance, derived, 1, config.total_copies, config
    )
    assert evaluation.loss_computed and evaluation.ranking_eligible
    assert evaluation.oracle_loss_available
    assert evaluation.trace_distance == evaluation.loss == pytest.approx(0.125)
    assert evaluation.oracle_evaluation_seconds is not None
    assert evaluation.oracle_evaluation_seconds >= 0.0


def test_candidate_search_control_is_explicitly_serial_and_cli_removed():
    with pytest.raises(ValueError, match="candidate search is explicitly serial"):
        OptimizationConfig(candidate_workers=2)
    from Optimization.run_parameter_optimization import _parser

    assert "candidate_workers" not in {
        action.dest for action in _parser()._actions
    }


def test_learner_stage_code_uses_operational_api_not_hidden_state_fields():
    learner_stages = (
        main_v2.sample_bell_scores,
        main_v2._commuting_tuple_probabilities_backend,
        main_v2._product_pauli_probabilities_backend,
        main_v2.empirical_certified_stabilizer_peeling,
        main_v2.empirical_rank_guided_sector_recovery,
        main_v2.empirical_hierarchical_cumulant_grouping,
    )
    for function in learner_stages:
        source = inspect.getsource(inspect.unwrap(function))
        assert "_structured_state" not in source
        assert "oracle_truth" not in source
