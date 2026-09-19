from copy import deepcopy

import pytest

import run_local_10q_eta_pair_diagnostic as diagnostic


@pytest.fixture(scope="module")
def provenance():
    return diagnostic.resolve_prior_candidate()


def test_prior_candidate_resolution_recovers_unique_five_weights(provenance):
    assert provenance["raw_draw_index"] == 89
    assert provenance["archived_candidate_id"] == "candidate-0058"
    assert provenance["h_min"] == 0.7645886990214604
    assert provenance["h_max"] == 0.9941139265711931
    assert provenance["theta"] == 0.03043433244408696
    assert provenance["eta_test"] == diagnostic.ETA_OLD
    assert provenance["stage_weights"] == pytest.approx({
        "peeling": 1.439235197153638,
        "recovery": 2.984067963842604,
        "grouping": 0.4911433777404421,
        "syndrome": 0.05906142042626675,
        "tomography": 0.5435380730562828,
    })


def test_candidate_cloning_changes_only_eta_test(provenance):
    old = diagnostic.candidate_record(provenance, diagnostic.ETA_OLD)
    low = diagnostic.candidate_record(provenance, diagnostic.ETA_LOW)
    assert diagnostic.changed_candidate_fields(old, low) == ("eta_test",)
    assert old["eta_test"] == diagnostic.ETA_OLD
    assert low["eta_test"] == diagnostic.ETA_LOW


def test_identical_stage_caps_across_paired_arms(provenance):
    seed = provenance["canonical_measurement_seed"]
    old_caps, low_caps = diagnostic.paired_stage_caps(
        provenance, diagnostic.EXPECTED_BUDGET, seed
    )
    assert old_caps == low_caps == diagnostic.EXPECTED_STAGE_CAPS
    assert sum(cap for _stage, cap in old_caps) == diagnostic.EXPECTED_BUDGET


def test_same_seed_and_same_pregrouping_configuration(provenance):
    seed = provenance["canonical_measurement_seed"]
    config = diagnostic.build_optimization_config(
        provenance, diagnostic.EXPECTED_BUDGET, seed
    )
    old_record = diagnostic.candidate_record(provenance, diagnostic.ETA_OLD)
    low_record = diagnostic.candidate_record(provenance, diagnostic.ETA_LOW)
    old = diagnostic.derive_candidate(
        diagnostic.make_candidate(old_record), n=10, d=3,
        total_copies=diagnostic.EXPECTED_BUDGET, optimization_config=config,
    )
    low = diagnostic.derive_candidate(
        diagnostic.make_candidate(low_record), n=10, d=3,
        total_copies=diagnostic.EXPECTED_BUDGET, optimization_config=config,
    )
    assert config.tuning_seeds == (seed,)
    assert config.simulation_backend == "batched_counts"
    assert (old.M1, old.M2, old.h_min, old.h_max, old.theta) == (
        low.M1, low.M2, low.h_min, low.h_max, low.theta
    )
    assert old.fixed_budget_stage_caps == low.fixed_budget_stage_caps
    assert old.normalized_stage_weights == low.normalized_stage_weights
    assert old.eta_test == diagnostic.ETA_OLD
    assert low.eta_test == diagnostic.ETA_LOW


def _arm(distance, *, exact=False, false_merges=0, splits=1, pure=4):
    return {
        "trace_distance": distance,
        "oracle_post_hoc": {
            "exact_oracle_partition": exact,
            "false_merge_count": false_merges,
            "true_block_split_count": splits,
            "block_pure_cluster_count": pure,
        },
    }


def test_paired_result_comparison_invariants():
    old = {
        "peeling": {"t": 0}, "recovery": {"L": 10},
        "assigned_stage_caps": [["peeling", 1]], "measurement_seed": 7,
        "stage_seed_ledger": {"peeling_seed": 11}, "backend": "batched_counts",
        "candidate": {"h_min": 0.7, "eta_test": diagnostic.ETA_OLD},
    }
    low = deepcopy(old)
    low["candidate"]["eta_test"] = diagnostic.ETA_LOW
    comparison = diagnostic.compare_pregrouping(old, low)
    assert all(comparison.values())
    low["recovery"]["L"] = 9
    assert not diagnostic.compare_pregrouping(old, low)["recovery_equal"]


@pytest.mark.parametrize(
    "old,low,invariants,expected",
    [
        (_arm(0.3, splits=1), _arm(0.1, exact=True, splits=0), {"paired": True}, "A"),
        (_arm(0.3, splits=1), _arm(0.299, exact=True, splits=0), {"paired": True}, "B"),
        (_arm(0.2), _arm(0.3, false_merges=1), {"paired": True}, "C"),
        (_arm(0.2), _arm(0.2), {"paired": True}, "D"),
        (_arm(0.2), _arm(0.1, exact=True), {"paired": False}, "E"),
    ],
)
def test_scientific_classification_helper(old, low, invariants, expected):
    assert diagnostic.classify_pair(old, low, invariants)[0] == expected
