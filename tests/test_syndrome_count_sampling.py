import numpy as np
import pytest
import qutip as qt

import main_v2


def _source(state, n):
    state = main_v2._v1._state_to_qobj(state, n)
    return main_v2.SimulatorMeasurementSource(n, state.isket, state)


def _peeling(n, t, generators=None):
    generators = tuple(
        generators
        or ("I" * j + "Z" + "I" * (n - j - 1) for j in range(t))
    )
    return main_v2.PeelingResult(
        success=True,
        failure_reason=None,
        h=0.7,
        lambda_=0.6,
        tau_1=0.01,
        t=t,
        generators=generators,
        generator_symplectic_vectors=tuple(
            tuple(main_v2._v1.pauli_to_symplectic_col(generator))
            for generator in generators
        ),
        certified_span_basis=(),
        inner_set=generators,
        outer_set=generators,
        inner_span_basis=(),
        outer_span_basis=(),
        U_stab=np.eye(2**n, dtype=complex),
        tableau=np.eye(2 * n, dtype=np.uint8),
        gates=(),
        epsilon_peel=0.01,
        M1=10,
        copy_ledger=main_v2.CopyLedger((("peeling_bell_pool", 20),)),
        score_provenance=main_v2.DataProvenance.EMPIRICAL,
        score_frame="physical",
        threshold_grid=(0.6, 0.7),
        transcript=(),
        theorem_grid_condition=True,
        theorem_tau_condition=True,
    )


def test_production_large_shot_path_is_scalar_count_only(monkeypatch):
    shots = 1_000_000_000
    source = _source(qt.tensor(qt.basis(2, 0), qt.basis(2, 1)), 2)
    peeling = _peeling(2, 2)
    calls = []

    class ScalarBinomialRng:
        def binomial(self, n, p):
            calls.append((n, p))
            return n if p == 1.0 else 0

    monkeypatch.setattr(
        main_v2,
        "measure_pauli_expectation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("production requested detailed shot outcomes")
        ),
    )
    monkeypatch.setattr(
        main_v2.np.random, "default_rng", lambda _seed=None: ScalarBinomialRng()
    )
    result = main_v2.recover_peeling_syndrome(
        source,
        peeling,
        main_v2.SyndromeConfig(0.05, 0.6, M_sgn=shots),
    )

    assert result.success
    assert calls == [(shots, 1.0), (shots, 0.0)]
    assert result.records == ()
    assert result.empirical_means == (1.0, -1.0)
    assert result.syndrome_bits == (0, 1)
    assert result.M_sgn == shots
    assert result.syndrome_sign_pool == 2 * shots
    assert result.copy_ledger.as_dict() == {"syndrome_sign_pool": 2 * shots}


def test_count_only_empirical_mean_uses_binomial_sufficient_statistic(monkeypatch):
    calls = []

    class FixedBinomialRng:
        def binomial(self, n, p):
            calls.append((n, p))
            return 7

    plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    monkeypatch.setattr(
        main_v2.np.random, "default_rng", lambda _seed=None: FixedBinomialRng()
    )
    empirical_mean = main_v2._sample_pauli_empirical_mean(
        _source(plus, 1), "Z", 10, seed=4
    )
    assert calls == [(10, 0.5)]
    assert empirical_mean == pytest.approx((2 * 7 - 10) / 10)


def test_count_only_exact_probability_edges():
    zero = _source(qt.basis(2, 0), 1)
    one = _source(qt.basis(2, 1), 1)
    assert main_v2._sample_pauli_empirical_mean(zero, "Z", 10_000, seed=1) == 1.0
    assert main_v2._sample_pauli_empirical_mean(one, "Z", 10_000, seed=1) == -1.0


def test_count_only_distribution_sanity():
    rho = qt.Qobj([[0.625, 0.0], [0.0, 0.375]], dims=[[2], [2]])
    source = _source(rho, 1)
    means = [
        main_v2._sample_pauli_empirical_mean(source, "Z", 2_000, seed=seed)
        for seed in range(128)
    ]
    assert np.mean(means) == pytest.approx(0.25, abs=0.015)
    assert all(np.isfinite(mean) and -1.0 <= mean <= 1.0 for mean in means)


def test_t_zero_remains_zero_copy_and_skips_count_sampler(monkeypatch):
    monkeypatch.setattr(
        main_v2,
        "_sample_pauli_empirical_mean",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("t=0 invoked the count sampler")
        ),
    )
    result = main_v2.recover_peeling_syndrome(
        _source(qt.basis(2, 0), 1),
        _peeling(1, 0),
        main_v2.SyndromeConfig(0.05, 0.6),
        seed=9,
    )
    assert result.success
    assert result.M_sgn == result.syndrome_sign_pool == 0
    assert result.records == result.empirical_means == result.syndrome_bits == ()
    assert result.copy_ledger.as_dict() == {"syndrome_sign_pool": 0}


def test_detailed_path_retains_full_measurement_records(monkeypatch):
    monkeypatch.setattr(
        main_v2,
        "_sample_pauli_empirical_mean",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("detailed path invoked count-only sampling")
        ),
    )
    result = main_v2.recover_peeling_syndrome(
        _source(qt.basis(2, 0), 1),
        _peeling(1, 1),
        main_v2.SyndromeConfig(0.05, 0.6, M_sgn=7, return_details=True),
        seed=10,
    )
    assert result.success and result.empirical_means == (1.0,)
    assert len(result.records) == 1
    assert isinstance(result.records[0], main_v2.PauliMeasurementRecord)
    assert result.records[0].outcomes.shape == (7,)
    assert result.syndrome_sign_pool == 7
