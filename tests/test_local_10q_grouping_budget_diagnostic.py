from copy import deepcopy

from tests import run_local_10q_grouping_budget_diagnostic as diagnostic


def _arm(caps, total, *, false_merges, false_splits, distance):
    return {
        "state_identity": {"path": "Jobs/06/n=10_d=3/states/init_000.npz"},
        "candidate": {
            "h_min": 0.7645886990214604,
            "h_max": 0.9941139265711931,
            "theta": 0.03043433244408696,
            "eta_test": 0.15,
        },
        "eta_test": 0.15,
        "measurement_seed": 9443020599540107709,
        "stage_seed_ledger": {
            "peeling_seed": 1145022205596711276,
            "recovery_seed": 7642224393080596217,
            "grouping_seed": 1116043267847408079,
            "syndrome_seed": 11719803265182213442,
            "tomography_seed": 16072299670706725501,
        },
        "backend": "batched_counts",
        "assigned_stage_caps": caps,
        "nominal_total": total,
        "peeling": {"selected_h": 0.7645886990214604, "t": 0},
        "recovery": {"L": 10, "recovered_span_rank": 20},
        "oracle_post_hoc": {
            "exact_oracle_partition": false_merges == 0 and false_splits == 0,
            "false_merge_count": false_merges,
            "true_block_split_count": false_splits,
            "block_pure_cluster_count": 4 if false_merges == 0 else 2,
        },
        "trace_distance": distance,
    }


def _pair(**changes):
    test_a = _arm(
        diagnostic.EXPECTED_STAGE_CAPS,
        diagnostic.EXPECTED_BUDGET,
        false_merges=1,
        false_splits=0,
        distance=0.5,
    )
    test_b = _arm(
        diagnostic.EXPANDED_GROUPING_STAGE_CAPS,
        diagnostic.EXPECTED_BUDGET_B,
        false_merges=0,
        false_splits=0,
        distance=0.3,
    )
    for target, values in changes.items():
        arm = test_a if target == "test_a" else test_b
        arm.update(values)
    return test_a, test_b


def test_exact_pair_stage_caps_and_total_b():
    assert diagnostic.EXPECTED_STAGE_CAPS == (
        ("peeling", 1_304_353),
        ("recovery", 2_704_407),
        ("grouping", 445_114),
        ("syndrome", 53_526),
        ("tomography", 492_600),
    )
    assert diagnostic.EXPANDED_GROUPING_STAGE_CAPS == (
        ("peeling", 1_304_353),
        ("recovery", 2_704_407),
        ("grouping", 2_000_000),
        ("syndrome", 53_526),
        ("tomography", 492_600),
    )
    assert diagnostic.EXPECTED_BUDGET_B == 6_554_886


def test_pair_config_and_pregrouping_outputs_are_identical_except_grouping_cap():
    test_a, test_b = _pair()
    invariants = diagnostic.compare_pregrouping(test_a, test_b)
    assert all(invariants.values())


def test_pairing_inconsistency_is_classification_e():
    test_a, test_b = _pair()
    test_b = deepcopy(test_b)
    test_b["measurement_seed"] += 1
    invariants = diagnostic.compare_pregrouping(test_a, test_b)
    assert not invariants["measurement_seed_equal"]
    assert diagnostic.classify_pair(test_a, test_b, invariants)[0] == "E"


def test_more_grouping_copies_fix_false_merge_is_classification_a():
    test_a, test_b = _pair()
    invariants = diagnostic.compare_pregrouping(test_a, test_b)
    assert diagnostic.classify_pair(test_a, test_b, invariants)[0] == "A"


def test_more_grouping_copies_help_without_full_fix_is_classification_b():
    test_a, test_b = _pair()
    test_b["oracle_post_hoc"].update(
        false_merge_count=1,
        true_block_split_count=0,
        exact_oracle_partition=False,
        block_pure_cluster_count=3,
    )
    invariants = diagnostic.compare_pregrouping(test_a, test_b)
    assert diagnostic.classify_pair(test_a, test_b, invariants)[0] == "B"


def test_unchanged_and_worse_results_classify_c_and_d():
    test_a, test_b = _pair()
    test_b["oracle_post_hoc"] = deepcopy(test_a["oracle_post_hoc"])
    test_b["trace_distance"] = test_a["trace_distance"]
    invariants = diagnostic.compare_pregrouping(test_a, test_b)
    assert diagnostic.classify_pair(test_a, test_b, invariants)[0] == "C"

    test_b["oracle_post_hoc"]["false_merge_count"] = 2
    assert diagnostic.classify_pair(test_a, test_b, invariants)[0] == "D"
