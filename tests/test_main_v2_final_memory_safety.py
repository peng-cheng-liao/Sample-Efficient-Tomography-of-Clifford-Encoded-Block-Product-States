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
    values.update(overrides or {})
    return values


def _synthetic_peeling(n):
    source = main_v2.PauliScoreMap(
        n=n,
        score_map=_score_map(n),
        uniform_radius=0.0,
        frame="physical",
    )
    result = main_v2.certified_stabilizer_peeling_v2(
        source,
        main_v2.PeelingConfig(
            h_min=0.6,
            h_max=0.8,
            eta=0.005,
            return_details=True,
            materialize_dense_clifford=False,
        ),
    )
    assert result.success
    return result


def _recovery_source(n):
    return main_v2.PauliScoreMap(
        n=n,
        score_map=_score_map(n, {"XI"[-n:]: 0.8}),
        uniform_radius=0.0,
        frame="peeled",
    )


def test_workspace_estimator_uses_known_dtype_arithmetic_and_worst_case_survivors():
    estimate = main_v2.estimate_enumeration_workspace(
        2,
        "recovery",
        simulation_backend="batched_counts",
        structured_bell=True,
        safety_factor=1.0,
    )
    table = 4**2 * np.dtype(np.int64).itemsize
    assert estimate.peak_phase == "recovery_ranking"
    assert dict(estimate.dominant_buffers)["worst_case_survivor_indices"] == table
    assert estimate.known_peak_bytes == 7 * table
    assert estimate.safety_margin_bytes == 0
    assert estimate.predicted_peak_bytes == 7 * table


def test_workspace_safety_factor_is_explicit_and_config_validated():
    baseline = main_v2.estimate_enumeration_workspace(
        3, "recovery", safety_factor=1.0
    )
    guarded = main_v2.estimate_enumeration_workspace(
        3, "recovery", safety_factor=1.5
    )
    assert guarded.predicted_peak_bytes == int(
        np.ceil(1.5 * baseline.known_peak_bytes)
    )
    with pytest.raises(ValueError, match="at least 1"):
        main_v2.EnumerationExecutionConfig(
            enumeration_workspace_safety_factor=0.99
        )


def test_peeling_workspace_cap_rejects_before_sampler_or_copy_consumption(monkeypatch):
    instance = main_v2.random_cebp_state(
        2,
        1,
        block_states=(qt.basis(2, 0), qt.basis(2, 0)),
        clifford_steps=0,
        seed=4,
    )
    estimate = main_v2.estimate_enumeration_workspace(
        2, "peeling", bell_rounds=10, safety_factor=1.0
    )
    config = main_v2.PeelingConfig(
        h_min=0.6,
        h_max=0.8,
        eta=0.01,
        M1=10,
        enumeration_execution=main_v2.EnumerationExecutionConfig(
            max_enumeration_workspace_bytes=estimate.predicted_peak_bytes - 1,
            enumeration_workspace_safety_factor=1.0,
        ),
        materialize_dense_clifford=False,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Bell sampler must not run after preflight rejection")

    monkeypatch.setattr(
        main_v2.SimulatorMeasurementSource, "_sample_bell_for_backend", forbidden
    )
    with pytest.raises(
        ValueError,
        match=r"peeling.*n=2.*predicted=.*limit=.*dominant_buffers",
    ):
        main_v2.empirical_certified_stabilizer_peeling(
            instance.learner_view(),
            config,
            simulation_backend="batched_counts",
        )


def test_recovery_workspace_cap_rejects_before_sampler_and_sort(monkeypatch):
    peeling = _synthetic_peeling(2)
    instance = main_v2.random_cebp_state(
        2,
        1,
        block_states=(qt.basis(2, 0), qt.basis(2, 0)),
        clifford_steps=0,
        seed=5,
    )
    estimate = main_v2.estimate_enumeration_workspace(
        2, "recovery", bell_rounds=10, safety_factor=1.0
    )
    config = main_v2.RecoveryConfig(
        theta=0.5,
        M2=10,
        allow_uncalibrated_peeling=True,
        allow_margin_failure=True,
        enumeration_execution=main_v2.EnumerationExecutionConfig(
            max_enumeration_workspace_bytes=estimate.predicted_peak_bytes - 1,
            enumeration_workspace_safety_factor=1.0,
        ),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("guarded allocation/sort helper was invoked")

    monkeypatch.setattr(main_v2, "sample_peeled_bell_scores", forbidden)
    monkeypatch.setattr(main_v2, "_rank_recovery_survivors", forbidden)
    with pytest.raises(
        ValueError,
        match=r"recovery.*n=2.*peak_phase=recovery_ranking",
    ):
        main_v2.empirical_rank_guided_sector_recovery(
            instance.learner_view(),
            peeling,
            config,
            simulation_backend="batched_counts",
        )


def test_small_recovery_below_cap_is_structurally_identical():
    peeling = _synthetic_peeling(2)
    source = main_v2.PauliScoreMap(
        n=2,
        score_map=_score_map(2, {"XI": 0.8, "IX": 0.7, "XX": 0.56}),
        uniform_radius=0.0,
        frame="peeled",
    )
    base = main_v2.RecoveryConfig(
        theta=0.5,
        return_details=True,
        allow_uncalibrated_peeling=True,
        allow_margin_failure=True,
    )
    estimate = main_v2.estimate_enumeration_workspace(
        2, "recovery", empirical=False, safety_factor=1.0, return_details=True
    )
    guarded = replace(
        base,
        enumeration_execution=main_v2.EnumerationExecutionConfig(
            max_enumeration_workspace_bytes=estimate.predicted_peak_bytes,
            enumeration_workspace_safety_factor=1.0,
        ),
    )
    assert main_v2.rank_guided_sector_recovery(
        source, peeling, guarded
    ) == main_v2.rank_guided_sector_recovery(source, peeling, base)


def test_structured_bell_model_includes_probability_count_and_score_coexistence():
    instance = main_v2.random_cebp_state(
        3,
        1,
        block_states=(qt.basis(2, 0),) * 3,
        clifford_steps=1,
        seed=9,
    )
    structured = instance.measurement_source._structured_state
    assert structured is not None
    expected = 2 * 4**3 * np.dtype(np.float64).itemsize
    assert structured.bell_workspace_bytes == expected
    breakdown = dict(structured.bell_workspace_breakdown)
    assert breakdown == {
        "distribution_two_float64_tables": expected,
        "sampling_probability_plus_int64_counts": expected,
        "score_retained_counts_plus_int64_sums": expected,
    }


def test_batched_record_retains_only_counts_and_one_common_copy_ledger():
    instance = main_v2.random_cebp_state(
        2,
        1,
        block_states=(qt.basis(2, 0), qt.qeye(2) / 2),
        clifford_steps=2,
        seed=12,
    )
    record = main_v2.sample_bell_scores(
        instance.learner_view(),
        100_000,
        simulation_backend="batched_counts",
        seed=13,
    )
    assert record.outcomes is None
    assert record.category_counts is not None
    assert record.category_counts.nbytes == 4**2 * np.dtype(np.int64).itemsize
    assert not any("probab" in name for name in vars(record))
    assert record.copy_ledger.as_dict() == {"peeling_bell_pool": 200_000}
    first = record.all_score_sums()
    second = record.all_score_sums()
    assert first is second


def test_backend_equivalence_keeps_exact_copy_accounting_on_same_fixture():
    instance = main_v2.random_cebp_state(
        2,
        1,
        block_states=(qt.basis(2, 0), qt.basis(2, 1)),
        clifford_steps=0,
        seed=18,
    )
    raw = main_v2.sample_bell_scores(
        instance.learner_view(), 500, simulation_backend="legacy_shotwise", seed=19
    )
    compressed = main_v2.sample_bell_scores(
        instance.learner_view(), 500, simulation_backend="batched_counts", seed=19
    )
    assert raw.copy_ledger == compressed.copy_ledger
    assert raw.rounds == compressed.rounds == 500
    np.testing.assert_allclose(
        raw.all_score_sums() / raw.rounds,
        compressed.all_score_sums() / compressed.rounds,
        atol=0.15,
    )
