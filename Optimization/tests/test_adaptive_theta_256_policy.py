from dataclasses import fields
from copy import deepcopy
import math
from types import SimpleNamespace

import numpy as np
import pytest

import Optimization.parameterization as parameterization
from Optimization.checkpoint import CheckpointStore
from Optimization.parameterization import (
    CandidateParameters,
    FixedBudgetCandidateParameters,
    OptimizationConfig,
    SearchSpace,
    derive_candidate,
    sample_candidate,
)
from Optimization.specification import OptimizationMode, OptimizationObjective
from Optimization.search import _checkpoint_metadata


def fixed_config(total_copies: int) -> OptimizationConfig:
    return OptimizationConfig(
        total_copies=total_copies,
        objective=OptimizationObjective(
            mode=OptimizationMode.FIXED_BUDGET_MIN_ERROR,
            copy_ceiling=total_copies,
        ),
    )


def fixed_candidate(*, multiplier=1.5, recovery_weight=1.0):
    return FixedBudgetCandidateParameters(
        h_min=0.60,
        h_max=0.90,
        theta_tau_multiplier=multiplier,
        eta_test=0.10,
        peel_weight=1.0,
        recovery_weight=recovery_weight,
        grouping_weight=1.0,
        syndrome_weight=1.0,
        tomography_weight=1.0,
    )


def test_fixed_budget_sampling_uses_log_uniform_multiplier(monkeypatch):
    calls = []

    def controlled_log_uniform(_rng, bounds):
        calls.append(tuple(bounds))
        return math.sqrt(float(bounds[0]) * float(bounds[1]))

    monkeypatch.setattr(parameterization, "_log_uniform", controlled_log_uniform)
    sampled = sample_candidate(
        np.random.default_rng(7), SearchSpace(), mode="fixed_budget_min_error"
    )
    assert isinstance(sampled, FixedBudgetCandidateParameters)
    assert calls[0] == (1.5, 128.0)
    assert sampled.theta_tau_multiplier == pytest.approx(math.sqrt(1.5 * 128.0))
    assert 1.5 <= sampled.theta_tau_multiplier <= 128.0
    assert "theta" not in {field.name for field in fields(sampled)}


def test_strict_sampling_retains_absolute_theta_coordinate():
    sampled = sample_candidate(np.random.default_rng(7), SearchSpace())
    assert isinstance(sampled, CandidateParameters)
    assert 0.03 <= sampled.theta <= 0.50
    assert not hasattr(sampled, "theta_tau_multiplier")


@pytest.mark.parametrize("d", (1, 2, 3))
def test_same_adaptive_formula_applies_to_all_d(d):
    total = 100_000_000
    multiplier = 3.25
    derived = derive_candidate(
        fixed_candidate(multiplier=multiplier),
        n=8,
        d=d,
        total_copies=total,
        optimization_config=fixed_config(total),
    )
    expected = min(0.50, multiplier * derived.tau_rank)
    assert derived.theta == pytest.approx(expected)
    assert derived.theta_tau_multiplier == multiplier
    assert derived.theta_over_tau_rank == pytest.approx(
        derived.theta / derived.tau_rank
    )
    assert 0.0 < derived.theta < 1.0


def test_recovery_allocation_controls_tau_and_effective_theta_at_fixed_budget():
    total = 100_000_000
    low = derive_candidate(
        fixed_candidate(recovery_weight=0.25),
        n=8,
        d=2,
        total_copies=total,
        optimization_config=fixed_config(total),
    )
    high = derive_candidate(
        fixed_candidate(recovery_weight=4.0),
        n=8,
        d=2,
        total_copies=total,
        optimization_config=fixed_config(total),
    )
    assert low.M2 != high.M2
    assert low.tau_rank != high.tau_rank
    assert low.theta != high.theta
    for derived in (low, high):
        assert derived.theta == pytest.approx(
            min(0.50, derived.theta_tau_multiplier * derived.tau_rank)
        )


def test_cap_and_high_budget_below_old_absolute_floor():
    small_total = 100
    capped = derive_candidate(
        fixed_candidate(multiplier=128.0),
        n=5,
        d=2,
        total_copies=small_total,
        optimization_config=fixed_config(small_total),
    )
    assert capped.theta == 0.50
    assert capped.theta_over_tau_rank < capped.theta_tau_multiplier

    large_total = 100_000_000
    resolved = derive_candidate(
        fixed_candidate(multiplier=1.5),
        n=8,
        d=3,
        total_copies=large_total,
        optimization_config=fixed_config(large_total),
    )
    assert resolved.theta < 0.03


def test_fixed_error_legacy_absolute_theta_is_canonicalized_to_multiplier():
    total = 100_000_000
    strict = CandidateParameters(
        alpha_peel=0.05,
        alpha_rank=0.20,
        alpha_sgn=0.01,
        c_peel=1.25,
        c_rank=1.25,
        h_min=0.70,
        h_span=0.10,
        theta=0.40,
        eta_test=0.10,
        kappa_ratio=0.60,
        epsilon_tom=1.0,
    )
    config = OptimizationConfig(
        total_copies=total,
        objective=OptimizationObjective(
            mode=OptimizationMode.FIXED_ERROR_MIN_COPIES,
            copy_ceiling=total,
            error_target=0.1,
        ),
    )
    derived = derive_candidate(
        strict, n=8, d=2, total_copies=total, optimization_config=config
    )
    assert derived.theta == 0.40
    assert derived.theta_tau_multiplier == pytest.approx(0.40 / derived.tau_rank)
    assert derived.theta_over_tau_rank == pytest.approx(0.40 / derived.tau_rank)
    assert derived.execution_policy == "fixed_budget_graceful"
    assert sum(dict(derived.fixed_budget_stage_caps).values()) == total


def test_old_fixed_budget_semantic_checkpoint_metadata_is_rejected(tmp_path):
    config = fixed_config(100_000)
    current = _checkpoint_metadata(
        instance=SimpleNamespace(n=8, d=2),
        objective=config.effective_objective,
        config=config,
        search_space=SearchSpace(),
        records=(),
    )
    assert current["fixed_budget_parameterization_schema"] == 5
    assert current["evaluation_identity_schema"] == 7
    old = deepcopy(current)
    old["fixed_budget_parameterization_schema"] = 4
    old["evaluation_identity_schema"] = 6
    path = tmp_path / "old-v2.json"
    CheckpointStore(
        str(path), expected_metadata=old, every_n_evaluations=1
    ).save({}, {}, status="running")
    with pytest.raises(ValueError, match="Incompatible checkpoint"):
        CheckpointStore(
            str(path), expected_metadata=current, every_n_evaluations=1
        ).load()
