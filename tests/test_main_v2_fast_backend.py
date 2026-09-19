import itertools
from dataclasses import replace

import numpy as np
import pytest
import qutip as qt

import main_v2


def _t0_peeling(n):
    scores = {pauli: 0.0 for pauli in main_v2._all_pauli_strings(n)}
    scores["I" * n] = 1.0
    result = main_v2.certified_stabilizer_peeling_v2(
        main_v2.PauliScoreMap(n, scores, 0.0, frame="physical"),
        main_v2.PeelingConfig(0.6, 0.8, 0.01),
    )
    assert result.success and result.t == 0
    return result


def _controlled_instance():
    polarization = 0.75
    one = (qt.qeye(2) + polarization * qt.sigmaz()) / 2
    return main_v2.random_cebp_state(
        3,
        2,
        block_sizes=(1, 2),
        block_states=(qt.ket2dm(qt.basis(2, 0)), qt.tensor(one, one)),
        clifford_steps=0,
        seed=1,
    )


def _fast_config(**changes):
    config = main_v2.EndToEndConfig(
        epsilon=0.5,
        delta=0.1,
        seed=7,
        materialize_dense_estimator=True,
        allow_uncertified_execution=True,
        simulation_backend="batched_counts",
        peeling_override=main_v2.PeelingConfig(0.6, 0.8, 0.02, 500, 0.1),
        recovery_override=main_v2.RecoveryConfig(
            0.4,
            500,
            0.1,
            allow_uncalibrated_peeling=True,
            allow_margin_failure=True,
        ),
        syndrome_override=main_v2.SyndromeConfig(0.1, 0.6, M_sgn=7),
        grouping_override=main_v2.GroupingConfig.from_guessed_scale(
            2,
            2.0,
            delta_grp_ordinary=0.1,
            allow_uncalibrated_recovery=True,
            allow_no_false_merge_margin_failure=True,
        ),
        tomography_override=main_v2.TomographyConfig(
            2.0,
            0.1,
            allow_uncertified_localization=True,
            max_dense_qubits=4,
        ),
    )
    return replace(config, **changes)


@pytest.fixture(scope="module")
def fast_run():
    instance = _controlled_instance()
    result = main_v2.full_cebp_tomography(
        instance.learner_view(), config=_fast_config()
    )
    assert result.success
    return instance, result


def test_backend_contract_default_explicit_and_invalid():
    assert main_v2.EndToEndConfig(0.5, 0.1).simulation_backend == "legacy_shotwise"
    assert (
        main_v2.EndToEndConfig(
            0.5, 0.1, simulation_backend="batched_counts"
        ).simulation_backend
        == "batched_counts"
    )
    with pytest.raises(ValueError, match="simulation_backend"):
        main_v2.EndToEndConfig(0.5, 0.1, simulation_backend="unknown")


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_all_bell_scores_transform_matches_direct_score_exactly(n):
    rng = np.random.default_rng(100 + n)
    categories = rng.integers(0, 4, size=(n, 73))
    outcomes = np.empty((n, 73, 3), dtype=np.int8)
    for qubit in range(n):
        outcomes[qubit] = main_v2._v1._BELL_EIG_TABLE[categories[qubit]]
    record = main_v2.BellScoreRecord(
        n=n,
        rounds=73,
        frame="peeled",
        uniform_radius=0.1,
        zeta_bs=0.1,
        backend="synthetic",
        seed=1,
        pool_name="recovery_bell_pool",
        copy_ledger=main_v2.CopyLedger((("recovery_bell_pool", 146),)),
        outcomes=outcomes,
        simulation_backend="batched_counts",
    )
    direct = replace(record, simulation_backend="legacy_shotwise")
    for pauli in main_v2._all_pauli_strings(n):
        assert record.score(pauli) == pytest.approx(direct.score(pauli), abs=1e-12)
    assert record.score("I" * n) == 1.0
    cache = record._all_scores_cache
    record.score("X" + "I" * (n - 1))
    assert record._all_scores_cache is cache


def test_bell_cache_memory_guard_falls_back_to_direct():
    n = main_v2._MAX_BELL_SCORE_CACHE_QUBITS + 1
    outcomes = np.broadcast_to(
        main_v2._v1._BELL_EIG_TABLE[0], (n, 2, 3)
    ).copy()
    record = main_v2.BellScoreRecord(
        n,
        2,
        "peeled",
        0.1,
        0.1,
        "synthetic",
        1,
        "recovery_bell_pool",
        main_v2.CopyLedger((("recovery_bell_pool", 4),)),
        outcomes,
        simulation_backend="batched_counts",
    )
    assert record.score("I" * n) == 1.0
    assert record._all_scores_cache is None


@pytest.mark.parametrize("n,measured", [(3, (0,)), (3, (0, 2)), (4, (1, 2, 3))])
def test_tomography_signed_sum_counts_equals_expanded_shots(n, measured):
    counts = np.arange(1, 2**n + 1, dtype=np.int64)
    expanded = np.repeat(np.arange(2**n), counts)
    expected = 0
    for category in expanded:
        bits = tuple((category >> (n - 1 - qubit)) & 1 for qubit in measured)
        expected += -1 if sum(bits) % 2 else 1
    assert main_v2._signed_sum_from_counts(counts, n, measured) == expected


@pytest.mark.parametrize("q", [2, 3])
def test_grouping_count_record_moments_and_cumulant_equal_expanded(q):
    outcomes = np.asarray(tuple(itertools.product((-1, 1), repeat=q)), dtype=np.int8)
    counts = np.arange(1, 2**q + 1, dtype=np.int64)
    shots = int(counts.sum())
    observables = tuple("I" * index + "Z" + "I" * (q - index - 1) for index in range(q))
    compressed = main_v2.JointPauliCountRecord(
        observables, shots, 1, outcomes, counts
    )
    expanded = main_v2.JointPauliMeasurementRecord(
        observables, shots, 1, np.repeat(outcomes, counts, axis=0)
    )
    assert compressed.all_subset_moments() == expanded.all_subset_moments()
    assert main_v2.empirical_mixed_cumulant_from_record(compressed) == pytest.approx(
        main_v2.empirical_mixed_cumulant_from_record(expanded), abs=1e-15
    )


def test_grouping_count_record_rejects_malformed_counts():
    outcomes = np.asarray(tuple(itertools.product((-1, 1), repeat=2)), dtype=np.int8)
    with pytest.raises(ValueError, match="sum exactly"):
        main_v2.JointPauliCountRecord(("ZI", "IZ"), 4, 1, outcomes, np.array([1, 1, 1, 0]))
    with pytest.raises(ValueError, match="positive"):
        main_v2.JointPauliCountRecord(("ZI", "IZ"), 0, 1, outcomes, np.zeros(4, dtype=int))


def test_each_grouping_query_is_copy_capped_without_partial_second_batch():
    instance = main_v2.random_cebp_state(
        2, 2, block_sizes=(2,), block_states=(qt.qeye(4) / 4,), clifford_steps=0, seed=3
    )
    peeling = _t0_peeling(2)
    shots = main_v2.ordinary_cumulant_sample_count(2, 1.0, 0.1)
    interface = main_v2.EmpiricalOrdinaryCumulantInterface(
        instance.learner_view(),
        peeling,
        tau_kappa=1.0,
        delta_tuple=0.1,
        seed=2,
        simulation_backend="batched_counts",
        max_realized_copies=shots,
    )
    interface.query(("ZI", "IZ"))
    with pytest.raises(main_v2.CopyBudgetExceeded, match="grouping_query"):
        interface.query(("XI", "IX"))
    assert interface.realized_query_count == 1
    assert interface.realized_copies == shots


def test_fast_full_pipeline_nonzero_pools_and_physical_output(fast_run):
    instance, result = fast_run
    ledger = result.realized_copy_ledger.as_dict()
    assert all(
        ledger[name] > 0
        for name in (
            "peeling_bell_pool",
            "recovery_bell_pool",
            "grouping_ordinary_pool",
            "syndrome_sign_pool",
            "block_tomography_pool",
        )
    )
    assert result.realized_total == result.realized_copy_ledger.total
    assert result.tomography.simulation_backend == "batched_counts"
    assert result.tomography.sampling_work_units < result.tomography.N_bp
    assert all(
        isinstance(record, main_v2.RegisterTomographyCountRecord)
        for record in result.tomography.records
    )
    density = result.decoded_density
    eigenvalues = np.linalg.eigvalsh(density.full())
    assert density.isherm and density.tr() == pytest.approx(1.0)
    assert np.min(eigenvalues) >= -1e-10
    assert np.isfinite(main_v2.debug_end_to_end_trace_error(result, instance))
    assert not result.theorem_certified


def test_legacy_and_batched_full_paths_have_identical_copy_ledger(fast_run):
    _instance, fast = fast_run
    instance = _controlled_instance()
    legacy = main_v2.full_cebp_tomography(
        instance.learner_view(),
        config=replace(_fast_config(), simulation_backend="legacy_shotwise"),
    )
    assert legacy.success
    assert legacy.realized_copy_ledger.entries == fast.realized_copy_ledger.entries
    assert legacy.tomography.sampling_work_units == legacy.tomography.N_bp


@pytest.mark.parametrize("backend", ["legacy_shotwise", "batched_counts"])
def test_copy_cap_below_peeling_consumes_zero_and_never_samples(monkeypatch, backend):
    instance = _controlled_instance()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("peeling sampled despite rejected copy batch")

    monkeypatch.setattr(main_v2, "empirical_certified_stabilizer_peeling", forbidden)
    result = main_v2.full_cebp_tomography(
        instance.learner_view(),
        config=replace(
            _fast_config(), simulation_backend=backend, max_realized_copies=999
        ),
    )
    assert not result.success and result.failure_stage == "copy_budget"
    assert result.realized_total == result.realized_copy_ledger.total == 0
    assert "peeling_bell_pool" in result.failure_reason


def test_copy_cap_allows_syndrome_then_stops_before_recovery(monkeypatch):
    instance = _controlled_instance()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("recovery sampled despite rejected copy batch")

    monkeypatch.setattr(main_v2, "empirical_rank_guided_sector_recovery", forbidden)
    result = main_v2.full_cebp_tomography(
        instance.learner_view(), config=_fast_config(max_realized_copies=1007)
    )
    assert not result.success and result.failure_stage == "copy_budget"
    assert result.realized_total == result.realized_copy_ledger.total == 1007
    assert result.peeling is not None and result.syndrome is not None
    assert result.recovery is None and "recovery_bell_pool" in result.failure_reason


def test_copy_cap_stops_before_tomography_and_retains_prior_ledger(fast_run):
    _instance, successful = fast_run
    prior = successful.realized_total - successful.tomography.N_bp
    instance = _controlled_instance()
    result = main_v2.full_cebp_tomography(
        instance.learner_view(), config=_fast_config(max_realized_copies=prior)
    )
    assert not result.success and result.failure_stage == "copy_budget"
    assert result.tomography is None and result.localization is not None
    assert result.realized_total == result.realized_copy_ledger.total == prior
    assert "block_tomography_pool" in result.failure_reason


def test_copy_cap_exactly_full_realized_cost_succeeds(fast_run):
    _instance, successful = fast_run
    instance = _controlled_instance()
    capped = main_v2.full_cebp_tomography(
        instance.learner_view(),
        config=_fast_config(max_realized_copies=successful.realized_total),
    )
    assert capped.success and capped.realized_total == successful.realized_total


def test_batched_tomography_raw_details_are_explicitly_rejected():
    instance = _controlled_instance()
    config = _fast_config(
        tomography_override=replace(
            _fast_config().tomography_override, return_details=True
        )
    )
    result = main_v2.full_cebp_tomography(instance.learner_view(), config=config)
    assert not result.success and result.failure_stage == "tomography"
    assert "does not provide raw per-shot" in result.failure_reason
