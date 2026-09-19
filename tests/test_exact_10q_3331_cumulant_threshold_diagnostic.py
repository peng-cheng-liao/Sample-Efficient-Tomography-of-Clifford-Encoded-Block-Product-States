import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import qutip as qt

import main_v2
import run_exact_10q_3331_cumulant_threshold_diagnostic as diagnostic


def _three_triple_sectors():
    return (
        main_v2.RecoveredSector(0, "XII", "ZII", "YII"),
        main_v2.RecoveredSector(1, "IXI", "IZI", "IYI"),
        main_v2.RecoveredSector(2, "IIX", "IIZ", "IIY"),
    )


def _identity_only_peeling(n):
    score_map = {
        "".join(chars): 1.0 if all(char == "I" for char in chars) else 0.0
        for chars in itertools.product("IXYZ", repeat=n)
    }
    result = main_v2.certified_stabilizer_peeling_v2(
        main_v2.PauliScoreMap(
            n=n, score_map=score_map, uniform_radius=0.0, frame="physical"
        ),
        main_v2.PeelingConfig(h_min=0.6, h_max=0.8, eta=0.005),
    )
    assert result.success and result.t == 0
    return result


def test_exact_K2_enumeration_uses_generated_cluster_group_semantics(monkeypatch):
    recovery = SimpleNamespace(sectors=_three_triple_sectors(), m=3)
    interface = main_v2.CallableResidualCumulantInterface(
        lambda observables: 0.25 if observables == ("XXI", "IIX") else 0.0
    )
    original = main_v2.generated_cluster_pauli_group
    calls = []

    def recorded(cluster, sectors, **kwargs):
        calls.append(tuple(cluster))
        return original(cluster, sectors, **kwargs)

    monkeypatch.setattr(main_v2, "generated_cluster_pauli_group", recorded)
    result = diagnostic.enumerate_cluster_cumulant_maximum(
        interface, recovery, ((0, 1), (2,))
    )
    assert calls == [(0, 1), (2,)]
    assert result.group_sizes == (15, 3) and result.candidate_count == 45
    assert result.magnitude == pytest.approx(0.25)
    assert result.witness.observables == ("XXI", "IIX")


def test_exact_K3_enumeration_uses_original_singleton_groups(monkeypatch):
    recovery = SimpleNamespace(sectors=_three_triple_sectors(), m=3)
    interface = main_v2.CallableResidualCumulantInterface(
        lambda observables: -0.4 if observables == ("XII", "IXI", "IIX") else 0.0
    )
    original = main_v2.generated_cluster_pauli_group
    calls = []

    def recorded(cluster, sectors, **kwargs):
        calls.append(tuple(cluster))
        return original(cluster, sectors, **kwargs)

    monkeypatch.setattr(main_v2, "generated_cluster_pauli_group", recorded)
    result = diagnostic.enumerate_cluster_cumulant_maximum(
        interface, recovery, ((0,), (1,), (2,))
    )
    assert calls == [(0,), (1,), (2,)]
    assert result.group_sizes == (3, 3, 3) and result.candidate_count == 27
    assert result.signed_value == pytest.approx(-0.4)
    assert result.witness.observables == ("XII", "IXI", "IIX")


def test_threshold_comparison_at_and_around_eta_is_deterministic():
    eta = 0.2
    tol = diagnostic.COMPARISON_TOLERANCE
    assert diagnostic.threshold_relation(eta, eta) == "equal_within_tolerance"
    assert diagnostic.threshold_relation(eta + 0.5 * tol, eta) == "equal_within_tolerance"
    assert diagnostic.threshold_relation(eta - 0.5 * tol, eta) == "equal_within_tolerance"
    assert diagnostic.threshold_relation(eta + 2.0 * tol, eta) == "above"
    assert diagnostic.threshold_relation(eta - 2.0 * tol, eta) == "below"


def test_cross_block_exact_cumulants_are_zero_on_block_product_fixture():
    zero, one = qt.basis(2, 0), qt.basis(2, 1)
    bell = (qt.tensor(zero, zero) + qt.tensor(one, one)).unit()
    plus = (zero + np.exp(0.31j) * one).unit()
    instance = main_v2.random_cebp_state(
        3,
        2,
        block_sizes=(2, 1),
        block_states=(bell * bell.dag(), plus * plus.dag()),
        clifford_steps=0,
        seed=7331,
    )
    peeling = _identity_only_peeling(3)
    recovery = SimpleNamespace(sectors=_three_triple_sectors(), m=3)
    interface = main_v2.DebugExactResidualCumulantInterface(instance, peeling)
    labels = ((0,), (0,), (1,))
    k2_cross = diagnostic.exhaustive_cross_block_maximum(
        interface, recovery, labels, 2
    )
    k3_cross = diagnostic.exhaustive_cross_block_maximum(
        interface, recovery, labels, 3
    )
    assert k2_cross.candidate_count == 18
    assert k3_cross.candidate_count == 27
    assert k2_cross.magnitude < diagnostic.EXACT_ZERO_TOLERANCE
    assert k3_cross.magnitude < diagnostic.EXACT_ZERO_TOLERANCE
    assert abs(k2_cross.signed_value) == k2_cross.magnitude
    assert abs(k3_cross.signed_value) == k3_cross.magnitude
    assert interface.realized_copies == 0


def test_eta_sweep_critical_value_construction_is_deterministic():
    first = diagnostic.critical_eta_values((0.1, 0.25, 0.1), 0.179)
    second = diagnostic.critical_eta_values((0.1, 0.25, 0.1), 0.179)
    assert first == second == tuple(sorted(set(first)))
    assert 0.0 in first and 0.179 in first
    assert 0.1 - 1.0e-6 in first and 0.1 + 1.0e-6 in first
    assert 0.25 - 1.0e-6 in first and 0.25 + 1.0e-6 in first


def test_oracle_labels_cannot_affect_actual_grouping_call(monkeypatch):
    interface = SimpleNamespace(realized_copies=0)
    grouping = SimpleNamespace(
        success=True,
        failure_reason=None,
        grouping_budget_truncated=False,
        grouping_copy_ledger=main_v2.CopyLedger(),
    )
    captured = {}

    monkeypatch.setattr(
        main_v2,
        "DebugExactResidualCumulantInterface",
        lambda instance, peeling: interface,
    )

    def fake_grouping(recovery, peeling, supplied_interface, config):
        captured["args"] = (recovery, peeling, supplied_interface, config)
        return grouping

    monkeypatch.setattr(main_v2, "hierarchical_cumulant_grouping", fake_grouping)
    monkeypatch.setattr(
        main_v2,
        "debug_oracle_sector_block_labels",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("oracle labels consulted during grouping")
        ),
    )
    result, returned_interface = diagnostic.run_grouping_at_eta(
        object(), object(), object(), 0.125
    )
    assert result is grouping and returned_interface is interface
    assert captured["args"][3].eta_test == pytest.approx(0.125)
    assert captured["args"][3].return_details
