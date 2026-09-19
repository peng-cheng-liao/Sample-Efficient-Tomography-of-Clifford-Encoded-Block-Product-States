from dataclasses import fields, replace

import numpy as np
import pytest
import qutip as qt

import main_v2
from Optimization.checkpoint import SCHEMA_VERSION
from Optimization.parameterization import (
    CandidateParameters,
    FixedBudgetCandidateParameters,
    OptimizationConfig,
    SearchSpace,
    derive_candidate,
    sample_candidate,
)
from Optimization.progressive_search import PROGRESSIVE_SEARCH_SCHEMA
from tests.test_main_v2_grouping import (
    _manuscript_fixture,
    _manual_recovery,
    _synthetic_peeling,
)


def _config(total=10_000):
    return OptimizationConfig(
        total_copies=total,
        tuning_seeds=(1,),
        holdout_seeds=(2,),
        halving_seed_counts=(1,),
        number_of_candidates=1,
    )


def test_fixed_budget_candidate_has_only_active_coordinates_and_is_reproducible():
    expected = {
        "h_min", "h_max", "theta_tau_multiplier", "eta_test", "peel_weight",
        "recovery_weight", "grouping_weight", "syndrome_weight",
        "tomography_weight",
    }
    first = sample_candidate(
        np.random.default_rng(17), SearchSpace(), mode="fixed_budget_min_error"
    )
    second = sample_candidate(
        np.random.default_rng(17), SearchSpace(), mode="fixed_budget_min_error"
    )
    assert isinstance(first, FixedBudgetCandidateParameters)
    assert {item.name for item in fields(first)} == expected
    assert first == second


def test_legacy_fixed_candidates_canonicalize_without_theorem_only_identity():
    base = CandidateParameters(
        0.01, 0.02, 0.003, 1.1, 1.2, 0.65, 0.10, 0.02, 0.08, 0.5, 0.7
    )
    changed = replace(
        base,
        alpha_peel=0.09,
        alpha_rank=0.20,
        alpha_sgn=0.02,
        c_peel=1.9,
        c_rank=1.8,
        kappa_ratio=1.4,
        epsilon_tom=1.7,
    )
    config = _config()
    left = derive_candidate(base, n=3, d=2, total_copies=10_000, optimization_config=config)
    right = derive_candidate(changed, n=3, d=2, total_copies=10_000, optimization_config=config)
    assert isinstance(left.parameters, FixedBudgetCandidateParameters)
    assert left == right


def test_fixed_budget_relaxes_theorem_margins_but_caps_sum_exactly():
    candidate = FixedBudgetCandidateParameters(
        h_min=0.80,
        h_max=0.90,
        theta_tau_multiplier=0.001,
        eta_test=0.01,
        peel_weight=1,
        recovery_weight=1,
        grouping_weight=1,
        syndrome_weight=1,
        tomography_weight=1,
    )
    config = _config(100)
    derived = derive_candidate(
        candidate, n=5, d=2, total_copies=100, optimization_config=config
    )
    assert derived.h_min + derived.tau1 >= 1.0
    assert derived.theta <= derived.tau_rank
    assert sum(dict(derived.fixed_budget_stage_caps).values()) == 100
    assert derived.M_sgn is None


def test_fixed_budget_grouping_uses_local_pool_without_theorem_batch(monkeypatch):
    instance, peeling, recovery = _manuscript_fixture()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("fixed-budget grouping called theorem shot calibration")

    monkeypatch.setattr(main_v2, "ordinary_cumulant_sample_count", forbidden)
    cap = 41
    result = main_v2.empirical_hierarchical_cumulant_grouping(
        instance.learner_view(),
        peeling,
        recovery,
        main_v2.GroupingConfig(
            ell_grp=3,
            eta_test=0.2,
            tau_kappa=0.05,
            allow_uncalibrated_recovery=True,
            allow_no_false_merge_margin_failure=True,
            sampling_policy="fixed_budget",
        ),
        seed=19,
        simulation_backend="batched_counts",
        fixed_budget_local_cap=cap,
    )
    assert result.success
    assert 0 < result.realized_grouping_copies <= cap
    assert result.sampling_policy == "fixed_budget"
    assert result.ordinary_copy_upper_bound == 0
    assert not result.theorem_grouping_preconditions_hold


def test_fixed_budget_grouping_refines_cumulative_tuple_statistics():
    instance, peeling, recovery = _manuscript_fixture()
    cap = 100
    result = main_v2.empirical_hierarchical_cumulant_grouping(
        instance.learner_view(),
        peeling,
        recovery,
        main_v2.GroupingConfig(
            ell_grp=3,
            eta_test=1e-12,
            tau_kappa=0.05,
            allow_uncalibrated_recovery=True,
            allow_no_false_merge_margin_failure=True,
            sampling_policy="fixed_budget",
        ),
        seed=19,
        simulation_backend="batched_counts",
        fixed_budget_local_cap=cap,
    )
    assert result.grouping_complete
    assert result.exploratory_query_count == 3
    assert result.realized_grouping_copies == cap
    assert sum(copies for _order, copies in result.copies_by_order) == cap
    assert result.min_shots_per_queried_tuple > 1

    interface = main_v2.FixedBudgetEmpiricalCumulantInterface(
        instance.learner_view(),
        peeling,
        local_copy_cap=20,
        seed=23,
        simulation_backend="batched_counts",
    )
    interface.set_remaining_query_opportunities(2)
    interface.query(("ZII", "IZI"))
    before = interface.shots_by_tuple[("ZII", "IZI")]
    interface.top_up_balanced(interface.remaining_copies)
    assert before == 10
    assert interface.shots_by_tuple[("ZII", "IZI")] == 20
    assert interface.refinement_top_up_count == 1


def test_fixed_budget_grouping_has_no_tau_or_delta_execution_dependency(monkeypatch):
    instance, peeling, recovery = _manuscript_fixture()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("certified sample count reached fixed-budget execution")

    monkeypatch.setattr(main_v2, "ordinary_cumulant_sample_count", forbidden)
    common = dict(
        ell_grp=3,
        eta_test=0.2,
        allow_uncalibrated_recovery=True,
        sampling_policy="fixed_budget",
    )
    results = tuple(
        main_v2.empirical_hierarchical_cumulant_grouping(
            instance.learner_view(),
            peeling,
            recovery,
            main_v2.GroupingConfig(
                **common,
                tau_kappa=tau,
                delta_grp_ordinary=delta,
            ),
            seed=29,
            simulation_backend="batched_counts",
            fixed_budget_local_cap=47,
        )
        for tau, delta in ((None, None), (0.19, 0.2), (float("nan"), 2.0))
    )
    assert all(result.success for result in results)
    assert len({result.clusters for result in results}) == 1
    assert len({result.realized_grouping_copies for result in results}) == 1
    assert len({result.copies_by_order for result in results}) == 1
    with pytest.raises(ValueError, match="positive"):
        main_v2.GroupingConfig(
            **{**common, "eta_test": 0.0}, tau_kappa=None, delta_grp_ordinary=None
        )
    with pytest.raises(ValueError, match="certified tau_kappa"):
        main_v2.GroupingConfig(
            ell_grp=3,
            eta_test=0.2,
            tau_kappa=None,
            sampling_policy="certified_tau",
        )


def test_online_grouping_opportunities_drop_skipped_tuple_and_old_partition_work():
    peeling = _synthetic_peeling(3)
    recovery = _manual_recovery(
        peeling,
        (
            main_v2.RecoveredSector(0, "XII", "ZII", "YII"),
            main_v2.RecoveredSector(1, "IXI", "IZI", "IYI"),
            main_v2.RecoveredSector(2, "IIX", "IIZ", "IIY"),
        ),
    )

    class DeterministicBudgetInterface:
        provenance = main_v2.DataProvenance.EMPIRICAL

        def __init__(self, cap):
            self.cap = cap
            self.remaining = cap
            self.pending = 1
            self.values = {}
            self.history = []
            self.query_count_by_order = {}
            self.copies_by_order = {}

        @property
        def realized_query_count(self):
            return len(self.values)

        @property
        def realized_copies(self):
            return self.cap - self.remaining

        def set_remaining_query_opportunities(self, count):
            self.pending = count

        def query(self, observables):
            key = tuple(observables)
            if key not in self.values:
                shots = max(1, self.remaining // self.pending)
                if shots > self.remaining:
                    raise main_v2.GroupingBudgetExhausted(
                        "grouping_query", self.realized_copies, shots, self.cap
                    )
                self.remaining -= shots
                self.values[key] = 1.0
                self.history.append((self.pending, shots))
                order = len(key)
                self.query_count_by_order[order] = self.query_count_by_order.get(order, 0) + 1
                self.copies_by_order[order] = self.copies_by_order.get(order, 0) + shots
            return self.values[key]

    first = DeterministicBudgetInterface(5_400)
    config = main_v2.GroupingConfig(
        ell_grp=3,
        eta_test=0.1,
        tau_kappa=None,
        delta_grp_ordinary=None,
        allow_uncalibrated_recovery=True,
        sampling_policy="fixed_budget",
    )
    result = main_v2.hierarchical_cumulant_grouping(
        recovery, peeling, first, config
    )
    second = DeterministicBudgetInterface(5_400)
    replay = main_v2.hierarchical_cumulant_grouping(
        recovery, peeling, second, config
    )
    initial_opportunities, first_shots = first.history[0]
    next_opportunities, next_shots = first.history[1]
    stale_after_one_query = initial_opportunities - 1
    assert initial_opportunities == 54
    assert next_opportunities == 18
    assert stale_after_one_query - next_opportunities > 1
    assert next_shots >= (5_400 - first_shots) // stale_after_one_query
    assert result.clusters == ((0, 1, 2),)
    assert result.realized_grouping_copies <= 5_400
    assert second.history == first.history and replay.clusters == result.clusters


def test_exact_operational_diagnostic_is_deterministic_and_zero_copy():
    instance = main_v2.random_cebp_state(
        2,
        2,
        block_sizes=(1, 1),
        block_states=(qt.qeye(2) / 2, qt.qeye(2) / 2),
        clifford_steps=0,
        seed=23,
    )
    kwargs = dict(h_min=0.6, h_max=0.9, theta=0.2, eta_test=0.1)
    first = main_v2.debug_exact_operational_diagnostic(instance, **kwargs)
    second = main_v2.debug_exact_operational_diagnostic(instance, **kwargs)
    assert first == second
    assert first.realized_physical_copies == 0
    assert first.structural_trace_distance == 0.0


def test_exact_operational_diagnostic_uses_exact_syndrome_product_for_t_positive():
    biased = qt.Qobj(np.diag([0.9, 0.1]), dims=[[2], [2]])
    instance = main_v2.random_cebp_state(
        2,
        1,
        block_sizes=(1, 1),
        block_states=(biased, qt.qeye(2) / 2),
        clifford_steps=0,
        seed=31,
    )
    diagnostic = main_v2.debug_exact_operational_diagnostic(
        instance, h_min=0.6, h_max=0.9, theta=0.2, eta_test=0.1
    )
    assert diagnostic.peeling_t == 1
    assert diagnostic.syndrome_expectations == pytest.approx((0.8,))
    assert diagnostic.syndrome_bits == (0,)
    assert diagnostic.syndrome_bits == tuple(
        0 if value >= 0.0 else 1 for value in diagnostic.syndrome_expectations
    )
    assert diagnostic.structural_trace_distance == pytest.approx(0.1)


def test_semantic_schema_versions_are_bumped():
    assert SCHEMA_VERSION == 7
    assert PROGRESSIVE_SEARCH_SCHEMA == 5
