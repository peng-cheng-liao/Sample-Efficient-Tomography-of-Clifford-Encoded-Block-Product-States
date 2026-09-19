import json
from types import SimpleNamespace
import zipfile

import pytest

import run_exact_10q_3331_diagnostic as diagnostic


def test_explicit_completed_candidate_thresholds_are_loaded(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "status": "success",
                "task": {"init_id": 0},
                "best_candidate_id": "candidate-7",
                "best_parameters": {
                    "h_min": 0.7,
                    "h_max": 0.9,
                    "theta": 0.2,
                    "eta_test": 0.1,
                },
            }
        ),
        encoding="utf-8",
    )
    thresholds = diagnostic.resolve_thresholds(0, path)
    assert (thresholds.h_min, thresholds.h_max) == (0.7, 0.9)
    assert thresholds.grid_intervals == 5
    assert thresholds.source_record == "candidate-7"


def test_missing_completed_candidate_is_a_policy_error(tmp_path, monkeypatch):
    case_dir = tmp_path / "n=10_d=3"
    (case_dir / "results").mkdir(parents=True)
    monkeypatch.setattr(diagnostic, "CASE_DIR", case_dir)
    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    with pytest.raises(diagnostic.ThresholdSourceError, match="No completed calibration result.json"):
        diagnostic.resolve_thresholds(0, None)


def test_missing_original_uses_authorized_reconstructed_record(tmp_path, monkeypatch):
    case_dir = tmp_path / "Jobs" / "06" / "n=10_d=3"
    (case_dir / "results").mkdir(parents=True)
    record = tmp_path / "Reports" / diagnostic.RECONSTRUCTED_THRESHOLD_NAME
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(
            {
                "init_id": 0,
                "provenance": "reconstructed_from_prior_recorded_calibration_values",
                "peeling_grid_intervals": 5,
                "best_candidate_id": "reconstructed-prior-init-000-budget-5000000",
                "best_parameters": {
                    "h_min": 0.7645886990214604,
                    "h_max": 0.9941139265711931,
                    "theta": 0.03043433244408696,
                    "eta_test": 0.17971039918570025,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    monkeypatch.setattr(diagnostic, "CASE_DIR", case_dir)
    thresholds = diagnostic.resolve_thresholds(0, None)
    assert thresholds.provenance == "reconstructed_from_prior_recorded_calibration_values"
    assert thresholds.h_min == pytest.approx(0.7645886990214604)
    assert thresholds.source == str(record.resolve())


def _result(kind, budget, h_min):
    return {
        "status": "success",
        "task": {
            "task_kind": kind,
            "init_id": 0,
            "budget": budget,
            "effective_budget": budget,
        },
        "best_candidate_id": f"{kind}-{budget}",
        "best_parameters": {
            "h_min": h_min,
            "h_max": 0.99,
            "theta": 0.03,
            "eta_test": 0.18,
        },
    }


def test_discovery_uses_highest_calibration_budget_and_excludes_benchmark(
    tmp_path, monkeypatch
):
    case_dir = tmp_path / "Jobs" / "06" / "n=10_d=3"
    low = case_dir / "results" / "calibration" / "low" / "result.json"
    low.parent.mkdir(parents=True)
    low.write_text(json.dumps(_result("calibration", 100_000, 0.71)))
    benchmark = case_dir / "results" / "benchmark" / "result.json"
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text(json.dumps(_result("benchmark", 9_000_000, 0.72)))
    archive_path = tmp_path / "old_results.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "results/calibration/init_000/budget_005000000/result.json",
            json.dumps(_result("calibration", 5_000_000, 0.7645886990214604)),
        )
    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    monkeypatch.setattr(diagnostic, "CASE_DIR", case_dir)
    thresholds = diagnostic.resolve_thresholds(0, None)
    assert thresholds.h_min == pytest.approx(0.7645886990214604)
    assert "old_results.zip::" in thresholds.source
    assert thresholds.source_record == "calibration-5000000"


def test_explicit_archive_rejects_conflicting_highest_budget_records(tmp_path):
    archive_path = tmp_path / "conflicting.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("a/result.json", json.dumps(_result("calibration", 5_000_000, 0.76)))
        archive.writestr("b/result.json", json.dumps(_result("calibration", 5_000_000, 0.77)))
    with pytest.raises(diagnostic.ThresholdSourceError, match="Conflicting.*a/result.json.*b/result.json"):
        diagnostic.resolve_thresholds(0, archive_path)


def test_explicit_archive_identical_highest_budget_records_are_deterministic(tmp_path):
    archive_path = tmp_path / "identical.zip"
    payload = json.dumps(_result("calibration", 5_000_000, 0.76))
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("z/result.json", payload)
        archive.writestr("a/result.json", payload)
        archive.writestr("low/result.json", json.dumps(_result("calibration", 100, 0.71)))
    thresholds = diagnostic.resolve_thresholds(0, archive_path)
    assert thresholds.h_min == pytest.approx(0.76)
    assert thresholds.source.endswith("::a/result.json")


def _thresholds(tmp_path):
    return diagnostic.Thresholds(
        0.7,
        0.9,
        0.2,
        0.1,
        5,
        str(tmp_path / "result.json"),
        "sha",
        "id",
        "completed_calibration_result",
    )


def test_t_positive_grouping_target_uses_residual_sector_partition():
    labels = ((0,), (1,), (0,), (2,))
    target = diagnostic._oracle_residual_clusters(labels)
    assert target == ((0, 2), (1,), (3,))
    assert diagnostic._same_partition(((3,), (1,), (0, 2)), target)
    assert not diagnostic._same_partition(((0, 1), (2,), (3,)), target)


def test_exact_syndrome_signs_and_completed_case_classification():
    source = SimpleNamespace(
        _expectation_for_backend=lambda pauli: {"XI": 0.25, "IZ": -0.5}[pauli]
    )
    expectations, bits = diagnostic._exact_syndrome(
        SimpleNamespace(measurement_source=source),
        SimpleNamespace(generators=("XI", "IZ")),
    )
    assert expectations == (0.25, -0.5) and bits == (0, 1)
    good_flags = {"direct_sum": True, "pairing": True}
    assert diagnostic._classify_completed_result(
        sector_pure=False,
        grouping_matches_residual=True,
        localization_flags=good_flags,
        trace_distance=0.0,
    ) == "CASE B"
    assert diagnostic._classify_completed_result(
        sector_pure=True,
        grouping_matches_residual=False,
        localization_flags=good_flags,
        trace_distance=0.0,
    ) == "CASE C"
    assert diagnostic._classify_completed_result(
        sector_pure=True,
        grouping_matches_residual=True,
        localization_flags={"direct_sum": False},
        trace_distance=0.0,
    ) == "CASE D"
    assert diagnostic._classify_completed_result(
        sector_pure=True,
        grouping_matches_residual=True,
        localization_flags=good_flags,
        trace_distance=1.0e-4,
    ) == "CASE E"
    assert diagnostic._classify_completed_result(
        sector_pure=True,
        grouping_matches_residual=True,
        localization_flags=good_flags,
        trace_distance=0.0,
    ) == "CASE A"


def test_scientific_peeling_and_recovery_failures_are_results(tmp_path, monkeypatch):
    peeling_failure = SimpleNamespace(
        success=False,
        failure_reason="no_certified_threshold",
        copy_ledger=diagnostic.main_v2.CopyLedger(),
    )
    monkeypatch.setattr(
        diagnostic.main_v2,
        "debug_exact_certified_stabilizer_peeling",
        lambda *_args, **_kwargs: peeling_failure,
    )
    result = diagnostic.run_exact(object(), _thresholds(tmp_path), False)
    assert result["case"] == "PEELING FAILURE"
    assert result["failure_stage"] == "peeling" and not result["passed"]

    peeling_success = SimpleNamespace(
        success=True,
        failure_reason=None,
        t=0,
        copy_ledger=diagnostic.main_v2.CopyLedger(),
    )
    recovery_failure = SimpleNamespace(
        success=False,
        failure_reason="threshold_span_incomplete",
        sectors=(),
        copy_ledger=diagnostic.main_v2.CopyLedger(),
    )
    monkeypatch.setattr(
        diagnostic.main_v2,
        "debug_exact_certified_stabilizer_peeling",
        lambda *_args, **_kwargs: peeling_success,
    )
    monkeypatch.setattr(
        diagnostic.main_v2,
        "debug_exact_rank_guided_sector_recovery",
        lambda *_args, **_kwargs: recovery_failure,
    )
    result = diagnostic.run_exact(object(), _thresholds(tmp_path), False)
    assert result["case"] == "CASE B"
    assert result["failure_stage"] == "recovery" and not result["passed"]
