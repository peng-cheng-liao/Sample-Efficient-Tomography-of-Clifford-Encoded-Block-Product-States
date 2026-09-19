#!/usr/bin/env python3
"""Run the zero-copy exact structural diagnostic on a frozen 10q instance."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import shlex
import subprocess
import sys
import time
from typing import Any, Iterable
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "Jobs" / "06" / "n=10_d=3"
REPORT_PATH = ROOT / "Reports" / "EXACT_10Q_3331_FINAL_DIAGNOSTIC_REPORT.txt"
RECONSTRUCTED_THRESHOLD_NAME = "EXACT_10Q_3331_RECONSTRUCTED_THRESHOLD_RECORD.json"
EXPECTED_PARTITION = (3, 3, 3, 1)
TRACE_TOLERANCE = 1.0e-10
DEFAULT_GRID_INTERVALS = 5

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main_v2  # noqa: E402


@dataclass(frozen=True)
class Thresholds:
    h_min: float
    h_max: float
    theta: float
    eta_test: float
    grid_intervals: int
    source: str
    source_sha256: str
    source_record: str
    provenance: str


class ThresholdSourceError(RuntimeError):
    """No unambiguous current operational threshold record is available."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_instance_io():
    path = CASE_DIR / "instance_io.py"
    spec = importlib.util.spec_from_file_location("cebp_n10_instance_io", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen-state loader: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record_init_id(payload: dict[str, Any]) -> int | None:
    candidates = (
        payload.get("init_id"),
        payload.get("task", {}).get("init_id") if isinstance(payload.get("task"), dict) else None,
        payload.get("instance", {}).get("init_id") if isinstance(payload.get("instance"), dict) else None,
    )
    for value in candidates:
        if value is not None:
            return int(value)
    return None


def _extract_threshold_payload(
    payload: dict[str, Any],
    init_index: int,
    *,
    source: str,
    source_sha256: str,
) -> Thresholds:
    if not isinstance(payload, dict):
        raise ThresholdSourceError(f"Threshold JSON is not an object: {source}")
    record_init = _record_init_id(payload)
    if record_init is not None and record_init != init_index:
        raise ThresholdSourceError(
            f"Threshold record init_id={record_init} does not match init_{init_index:03d}."
        )
    record: Any = payload.get("best_parameters", payload.get("parameters", payload))
    if not isinstance(record, dict):
        raise ThresholdSourceError(f"No candidate parameter object in {source}")
    required = ("h_min", "theta", "eta_test")
    missing = [name for name in required if name not in record]
    if missing:
        raise ThresholdSourceError(f"Missing {missing} in threshold record {source}")
    if "h_max" in record:
        h_max = float(record["h_max"])
    elif "h_span" in record:
        h_max = float(record["h_min"]) + float(record["h_span"])
    else:
        raise ThresholdSourceError(f"Missing h_max/h_span in threshold record {source}")
    intervals = int(
        payload.get(
            "peeling_grid_intervals",
            payload.get("optimization_config", {}).get(
                "peeling_grid_intervals", DEFAULT_GRID_INTERVALS
            )
            if isinstance(payload.get("optimization_config"), dict)
            else DEFAULT_GRID_INTERVALS,
        )
    )
    values = Thresholds(
        h_min=float(record["h_min"]),
        h_max=h_max,
        theta=float(record["theta"]),
        eta_test=float(record["eta_test"]),
        grid_intervals=intervals,
        source=source,
        source_sha256=source_sha256,
        source_record=str(payload.get("best_candidate_id", "candidate record")),
        provenance=str(payload.get("provenance", "completed_calibration_result")),
    )
    if not 0.5 < values.h_min < values.h_max < 1.0:
        raise ThresholdSourceError("Threshold record violates 1/2 < h_min < h_max < 1.")
    if not 0.0 < values.theta < 1.0 or values.eta_test <= 0.0:
        raise ThresholdSourceError("Threshold record has invalid theta/eta_test.")
    if values.grid_intervals <= 0:
        raise ThresholdSourceError("peeling_grid_intervals must be positive.")
    return values


def _extract_thresholds(path: Path, init_index: int) -> Thresholds:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _extract_threshold_payload(
        payload,
        init_index,
        source=str(path.resolve()),
        source_sha256=_sha256(path),
    )


def _task_metadata(payload: dict[str, Any], source: str) -> tuple[str | None, int | None]:
    task = payload.get("task") if isinstance(payload.get("task"), dict) else {}
    kind = task.get("task_kind", payload.get("task_kind"))
    if kind is None:
        lowered = source.lower().replace("\\", "/")
        if "/calibration/" in lowered:
            kind = "calibration"
        elif "/benchmark/" in lowered:
            kind = "benchmark"
    budget = task.get(
        "effective_budget",
        task.get(
            "budget",
            payload.get(
                "requested_copy_budget",
                payload.get("total_copies", payload.get("budget")),
            ),
        ),
    )
    return (None if kind is None else str(kind), None if budget is None else int(budget))


def _completed_calibration_candidate(
    payload: dict[str, Any], init_index: int, source: str
) -> tuple[int, tuple[float, float, float, float]] | None:
    if payload.get("status") not in ("success", "scientific_failure"):
        return None
    if _record_init_id(payload) != init_index:
        return None
    if not isinstance(payload.get("best_parameters"), dict):
        return None
    kind, budget = _task_metadata(payload, source)
    if kind != "calibration" or budget is None:
        return None
    try:
        threshold = _extract_threshold_payload(
            payload,
            init_index,
            source=source,
            source_sha256="pending",
        )
    except (KeyError, TypeError, ValueError, ThresholdSourceError):
        return None
    return budget, (threshold.h_min, threshold.h_max, threshold.theta, threshold.eta_test)


def resolve_thresholds(init_index: int, explicit: Path | None) -> Thresholds:
    """Use an explicit record or the highest-budget frozen calibration result.

    Discovery is metadata-driven and deterministic.  It searches direct JSON
    records plus archived ``result.json`` members, excludes worker-scaling
    benchmark tasks, and never consults filesystem modification times.
    """
    if explicit is not None:
        if not explicit.is_file():
            raise ThresholdSourceError(f"Explicit threshold record does not exist: {explicit}")
        if zipfile.is_zipfile(explicit):
            records: list[
                tuple[int, tuple[float, float, float, float], str, bytes, dict[str, Any]]
            ] = []
            with zipfile.ZipFile(explicit) as archive:
                for member in sorted(archive.namelist()):
                    if not member.lower().endswith("result.json"):
                        continue
                    raw = archive.read(member)
                    try:
                        payload = json.loads(raw.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    metadata = _completed_calibration_candidate(
                        payload, init_index, f"{explicit.resolve()}::{member}"
                    )
                    if metadata is not None:
                        records.append((metadata[0], metadata[1], member, raw, payload))
            if not records:
                raise ThresholdSourceError(
                    f"No completed calibration result for init_{init_index:03d} in {explicit}."
                )
            highest_budget = max(item[0] for item in records)
            highest = [item for item in records if item[0] == highest_budget]
            threshold_tuples = {item[1] for item in highest}
            if len(threshold_tuples) != 1:
                conflicts = ", ".join(
                    f"{member}={values}"
                    for _budget, values, member, _raw, _payload in sorted(
                        highest, key=lambda item: item[2]
                    )
                )
                raise ThresholdSourceError(
                    "Conflicting completed calibration records in explicit archive at "
                    f"budget {highest_budget}: {conflicts}"
                )
            highest.sort(key=lambda item: item[2])
            budget, _values, member, raw, payload = highest[0]
            return _extract_threshold_payload(
                payload,
                init_index,
                source=f"{explicit.resolve()}::{member}",
                source_sha256=hashlib.sha256(raw).hexdigest(),
            )
        return _extract_thresholds(explicit, init_index)

    candidates: list[tuple[int, int, str, str, dict[str, Any]]] = []
    direct_paths = set(CASE_DIR.rglob("result.json"))
    reports_dir = ROOT / "Reports"
    if reports_dir.is_dir():
        direct_paths.update(reports_dir.rglob("result.json"))
    for path in sorted(direct_paths):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source = str(path.resolve())
        metadata = _completed_calibration_candidate(payload, init_index, source)
        if metadata is not None:
            budget, _values = metadata
            candidates.append((budget, 0, source, _sha256(path), payload))
    for archive_path in sorted(ROOT.rglob("*.zip")):
        try:
            with zipfile.ZipFile(archive_path) as archive:
                for member in sorted(archive.namelist()):
                    if not member.lower().endswith("result.json"):
                        continue
                    raw = archive.read(member)
                    try:
                        payload = json.loads(raw.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    source = f"{archive_path.resolve()}::{member}"
                    metadata = _completed_calibration_candidate(
                        payload, init_index, source
                    )
                    if metadata is not None:
                        budget, _values = metadata
                        candidates.append(
                            (budget, 1, source, hashlib.sha256(raw).hexdigest(), payload)
                        )
        except (OSError, zipfile.BadZipFile):
            continue
    if not candidates:
        reconstructed = ROOT / "Reports" / RECONSTRUCTED_THRESHOLD_NAME
        if init_index == 0 and reconstructed.is_file():
            thresholds = _extract_thresholds(reconstructed, init_index)
            if thresholds.provenance != "reconstructed_from_prior_recorded_calibration_values":
                raise ThresholdSourceError(
                    f"Reconstructed threshold record has invalid provenance: {reconstructed}"
                )
            return thresholds
        raise ThresholdSourceError(
            "No completed calibration result.json with best_parameters for init_"
            f"{init_index:03d} exists in Jobs/06/n=10_d=3 or repository archives, "
            "and no authorized reconstructed prior-record record is available."
        )
    highest_budget = max(item[0] for item in candidates)
    highest = [item for item in candidates if item[0] == highest_budget]
    values = {
        _completed_calibration_candidate(item[4], init_index, item[2])[1]
        for item in highest
    }
    if len(values) != 1:
        raise ThresholdSourceError(
            f"Conflicting completed calibration records exist at budget {highest_budget}: "
            + ", ".join(item[2] for item in sorted(highest, key=lambda item: item[2]))
        )
    # Prefer the direct repository record over an identical archived duplicate,
    # then use the canonical source path solely as a stable storage tie-break.
    highest.sort(key=lambda item: (item[1], item[2]))
    budget, _archive_rank, source, digest, payload = highest[0]
    return _extract_threshold_payload(
        payload, init_index, source=source, source_sha256=digest
    )


def _trace_distance(first, second) -> float:
    difference = np.asarray(first.full() - second.full(), dtype=complex)
    return 0.5 * float(np.linalg.svd(difference, compute_uv=False).sum())


def _labels_for_cluster(
    cluster: Iterable[int], labels: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    return tuple(sorted({label for sector in cluster for label in labels[int(sector)]}))


def _format_pairs(values: Iterable[tuple[Any, Any]]) -> str:
    return ", ".join(f"{name}={value}" for name, value in values)


def _oracle_residual_clusters(
    sector_labels: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    """Return oracle-only residual sector groups after learner grouping is done."""
    if any(len(labels) != 1 for labels in sector_labels):
        return ()
    by_block: dict[int, list[int]] = {}
    for sector_id, labels in enumerate(sector_labels):
        by_block.setdefault(labels[0], []).append(sector_id)
    return tuple(tuple(by_block[label]) for label in sorted(by_block))


def _same_partition(
    learned: Iterable[Iterable[int]], target: Iterable[Iterable[int]]
) -> bool:
    return {
        frozenset(int(value) for value in cluster) for cluster in learned
    } == {
        frozenset(int(value) for value in cluster) for cluster in target
    }


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _git_head() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNAVAILABLE"


def _exact_syndrome(instance: Any, peeling: Any) -> tuple[tuple[float, ...], tuple[int, ...]]:
    expectations = tuple(
        float(instance.measurement_source._expectation_for_backend(generator))
        for generator in peeling.generators
    )
    return expectations, tuple(0 if value >= 0.0 else 1 for value in expectations)


def _classify_completed_result(
    *,
    sector_pure: bool,
    grouping_matches_residual: bool,
    localization_flags: dict[str, bool],
    trace_distance: float,
) -> str:
    if not sector_pure:
        return "CASE B"
    if not grouping_matches_residual:
        return "CASE C"
    if not all(localization_flags.values()):
        return "CASE D"
    if trace_distance > TRACE_TOLERANCE:
        return "CASE E"
    return "CASE A"


def _write_blocked_report(
    *, init_index: int, state_path: Path, metadata: dict[str, Any], reason: str
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "EXACT 10Q 3-3-3-1 STRUCTURAL DIAGNOSTIC",
                "=" * 47,
                "",
                "Status: BLOCKED before structural execution",
                f"Init index: {init_index}",
                f"Frozen state: {state_path.relative_to(ROOT)}",
                f"Frozen state SHA-256: {metadata['file_sha256']}",
                f"Content fingerprint: {metadata['content_fingerprint']}",
                "Verified n: 10",
                "Verified d: 3",
                "Verified hidden partition: [3, 3, 3, 1]",
                "Verified latent block sizes: [3, 3, 3, 1]",
                f"Compact Clifford gate count: {metadata['clifford_gate_count']}",
                "Dense full-state materialization: NOT PERFORMED",
                "",
                "THRESHOLD POLICY BLOCKER",
                "------------------------",
                reason,
                "No h_min, h_max, theta, or eta_test values were guessed or tuned.",
                "",
                "STAGE STATUS",
                "------------",
                "Exact peeling: NOT RUN",
                "Exact recovery: NOT RUN",
                "Oracle sector validation: NOT RUN",
                "Exact grouping: NOT RUN",
                "Localization: NOT RUN",
                "Exact syndrome: NOT RUN",
                "Exact register marginals: NOT RUN",
                "Compact estimator/decode: NOT RUN",
                "Exact trace distance: NOT AVAILABLE",
                "Physical learner copies consumed: 0",
                "Finite-budget empirical run: NOT RUN",
                "Scientific CASE A/B/C/D/E: NOT ASSIGNABLE BEFORE THRESHOLD RESOLUTION",
                "",
                "STRUCTURAL EXACT-DATA DIAGNOSTIC: FAIL",
                "Reason: missing unambiguous current operational threshold candidate.",
                "",
            )
        ),
        encoding="utf-8",
    )


def run_exact(instance, thresholds: Thresholds, details: bool) -> dict[str, Any]:
    """Execute exact stages, returning structural failures as diagnostic data."""

    started = time.perf_counter()
    result: dict[str, Any] = {
        "peeling": None,
        "recovery": None,
        "sector_labels": (),
        "grouping": None,
        "cluster_labels": (),
        "oracle_residual_clusters": (),
        "localization": None,
        "syndrome_expectations": (),
        "syndrome_bits": (),
        "register_marginal_summaries": (),
        "trace_distance": None,
        "localized_trace_distance": None,
        "stage_ledgers": {},
        "localization_flags": {},
        "sector_pure": False,
        "cluster_pure": False,
        "target_grouping": False,
        "dense_final_comparison_entered": False,
        "peak_rss_bytes": None,
        "case": "PEELING FAILURE",
        "passed": False,
        "failure_stage": None,
        "failure_reason": None,
        "stage_times": {},
    }

    def finish(stage: str | None = None, reason: str | None = None) -> dict[str, Any]:
        def copy_total(value: Any, ledger_name: str) -> int:
            ledger = getattr(value, ledger_name, None)
            return 0 if ledger is None else int(ledger.total)

        result["stage_ledgers"] = {
            "peeling": copy_total(result["peeling"], "copy_ledger"),
            "recovery": copy_total(result["recovery"], "copy_ledger"),
            "grouping": copy_total(result["grouping"], "grouping_copy_ledger"),
            "syndrome": 0,
            "marginals": 0,
            "instance": copy_total(instance, "copy_ledger"),
        }
        if any(result["stage_ledgers"].values()):
            raise RuntimeError(
                f"Exact path consumed physical copies: {result['stage_ledgers']}"
            )
        result["failure_stage"] = stage
        result["failure_reason"] = reason
        result["elapsed"] = time.perf_counter() - started
        result["peak_rss_bytes"] = _peak_rss_bytes()
        return result

    grid_eta = (thresholds.h_max - thresholds.h_min) / thresholds.grid_intervals
    stage_started = time.perf_counter()
    peeling = main_v2.debug_exact_certified_stabilizer_peeling(
        instance,
        main_v2.PeelingConfig(
            h_min=thresholds.h_min,
            h_max=thresholds.h_max,
            eta=grid_eta,
            M1=None,
            zeta_bs=0.05,
            return_details=details,
            max_dense_debug_qubits=10,
            materialize_dense_clifford=False,
        ),
    )
    result["stage_times"]["exact_peeling"] = time.perf_counter() - stage_started
    result["peeling"] = peeling
    if not peeling.success:
        return finish("peeling", peeling.failure_reason)

    stage_started = time.perf_counter()
    recovery = main_v2.debug_exact_rank_guided_sector_recovery(
        instance,
        peeling,
        main_v2.RecoveryConfig(
            theta=thresholds.theta,
            M2=None,
            zeta_rank=0.05,
            return_details=details,
            allow_uncalibrated_peeling=True,
            allow_margin_failure=True,
        ),
    )
    result["stage_times"]["exact_recovery"] = time.perf_counter() - stage_started
    result["recovery"] = recovery
    if not recovery.success:
        result["case"] = "CASE B"
        return finish("recovery", recovery.failure_reason)

    # Oracle metadata is deliberately attached only after recovery is complete.
    sector_labels = main_v2.debug_oracle_sector_block_labels(
        instance, peeling, recovery
    )
    result["sector_labels"] = sector_labels
    sector_pure = all(len(labels) == 1 for labels in sector_labels)
    result["sector_pure"] = sector_pure

    stage_started = time.perf_counter()
    grouping = main_v2.debug_exact_hierarchical_cumulant_grouping(
        instance,
        peeling,
        recovery,
        main_v2.GroupingConfig(
            ell_grp=instance.d,
            eta_test=thresholds.eta_test,
            tau_kappa=0.0,
            return_details=details,
            allow_uncalibrated_recovery=True,
            allow_no_false_merge_margin_failure=True,
        ),
    )
    result["stage_times"]["exact_grouping"] = time.perf_counter() - stage_started
    result["grouping"] = grouping
    if not grouping.success:
        result["case"] = "CASE B" if not sector_pure else "CASE C"
        return finish("grouping", grouping.failure_reason)

    cluster_labels = tuple(
        _labels_for_cluster(cluster, sector_labels) for cluster in grouping.clusters
    )
    cluster_pure = all(len(labels) == 1 for labels in cluster_labels)
    oracle_residual_clusters = _oracle_residual_clusters(sector_labels)
    target_grouping = sector_pure and _same_partition(
        grouping.clusters, oracle_residual_clusters
    )
    result.update(
        cluster_labels=cluster_labels,
        cluster_pure=cluster_pure,
        oracle_residual_clusters=oracle_residual_clusters,
        target_grouping=target_grouping,
    )

    stage_started = time.perf_counter()
    localization = main_v2.localize_grouped_recovery(
        recovery,
        grouping,
        d=instance.d,
        config=main_v2.LocalizationConfig(
            allow_uncertified_grouping=True,
            verify_dense_unitary=False,
            max_dense_qubits=10,
            materialize_dense_clifford=False,
            return_details=details,
        ),
    )
    result["stage_times"]["localization"] = time.perf_counter() - stage_started
    result["localization"] = localization
    localization_flags = {
        "direct_sum": localization.cross_group_direct_sum_holds,
        "cross_group_symplectic_orthogonality": (
            localization.cross_group_symplectic_orthogonality_holds
        ),
        "global_pairing": localization.global_pairing_holds,
        "register_partition": localization.register_partition_holds,
        "localization_guarantee": localization.localization_guarantee_holds,
        "handoff_valid": localization.handoff_valid,
    }
    result["localization_flags"] = localization_flags
    if not localization.success:
        result["case"] = (
            "CASE B"
            if not sector_pure
            else "CASE C"
            if not (cluster_pure and target_grouping)
            else "CASE D"
        )
        return finish("localization", localization.failure_reason)

    stage_started = time.perf_counter()
    syndrome_expectations, syndrome_bits = _exact_syndrome(instance, peeling)
    register_estimates = tuple(
        (
            tuple(cluster),
            tuple(register),
            main_v2.debug_exact_register_marginal(
                instance.learner_view(),
                peeling,
                localization,
                register,
                max_dense_debug_qubits=10,
            ),
        )
        for cluster, register in localization.J_C
    )
    result["register_marginal_summaries"] = tuple(
        (
            cluster,
            register,
            tuple(state.shape),
            float(np.real(state.tr())),
        )
        for cluster, register, state in register_estimates
    )
    result["stage_times"]["exact_syndrome_and_marginals"] = (
        time.perf_counter() - stage_started
    )
    compact = main_v2.CompactCEBPEstimator(
        n=localization.n,
        t=localization.t,
        m=localization.m,
        U_stab=None if peeling.U_stab is None else np.asarray(peeling.U_stab),
        bar_U_rec=None
        if localization.bar_U_rec is None
        else np.asarray(localization.bar_U_rec),
        syndrome_bits=syndrome_bits,
        register_estimates=register_estimates,
        J_aux=tuple(localization.J_aux),
        peeling_gates=tuple(peeling.gates),
        recovery_gates=tuple(localization.gates),
        peeling_clifford=peeling.signed_clifford,
        recovery_clifford=localization.signed_clifford,
    )

    # Structural stages above are compact.  The guarded final decode explicitly
    # materializes dense 10q Clifford unitaries plus true/estimated states.
    stage_started = time.perf_counter()
    result["dense_final_comparison_entered"] = True
    exact_state = instance.materialize_state_debug(max_qubits=10)
    exact_density = exact_state * exact_state.dag() if exact_state.isket else exact_state
    decoded = main_v2.materialize_compact_cebp_estimator(
        compact, max_dense_qubits=10
    )
    result["stage_times"]["dense_physical_materialization"] = (
        time.perf_counter() - stage_started
    )
    stage_started = time.perf_counter()
    trace_distance = _trace_distance(exact_density, decoded)
    # Unitary invariance gives the same localized structural distance without a
    # second full-state materialization.
    localized_distance = trace_distance
    result["stage_times"]["physical_trace_distance"] = (
        time.perf_counter() - stage_started
    )
    del exact_state, exact_density, decoded
    case = _classify_completed_result(
        sector_pure=sector_pure,
        grouping_matches_residual=cluster_pure and target_grouping,
        localization_flags=localization_flags,
        trace_distance=trace_distance,
    )
    result.update(
        syndrome_expectations=syndrome_expectations,
        syndrome_bits=syndrome_bits,
        trace_distance=trace_distance,
        localized_trace_distance=localized_distance,
        localization_flags=localization_flags,
        case=case,
        passed=case == "CASE A",
    )
    if case == "CASE D":
        return finish("localization_invariants", str(localization_flags))
    if case == "CASE E":
        return finish("exact_reconstruction", "nonzero exact structural error")
    if case in ("CASE B", "CASE C"):
        return finish("structural_validation", "oracle-only validation mismatch")
    return finish()


def _write_completed_report(
    *,
    init_index: int,
    state_path: Path,
    metadata: dict[str, Any],
    thresholds: Thresholds,
    result: dict[str, Any],
) -> None:
    peeling = result["peeling"]
    recovery = result["recovery"]
    grouping = result["grouping"]
    localization = result["localization"]
    cluster_sizes = () if grouping is None else tuple(map(len, grouping.clusters))
    j_aux = () if localization is None else tuple(localization.J_aux)
    sector_summary = (
        ()
        if recovery is None
        else tuple((sector.sector_id, sector.kind) for sector in recovery.sectors)
    )
    localization_summary = (
        ()
        if localization is None
        else tuple(
            (entry.cluster, entry.k_C, entry.J_C)
            for entry in localization.cluster_localizations
        )
    )
    dense_entered = bool(result["dense_final_comparison_entered"])
    interpretation = {
        "CASE A": "Recovery, residual grouping, localization, and reconstruction all agree with the oracle-only validation; the exact structural error is numerically zero.",
        "CASE B": "Recovery is the first structural bottleneck: it failed or produced a sector that is not hidden-block pure.",
        "CASE C": "Recovery is block-pure, but exact cumulant grouping does not reproduce the oracle residual sector partition.",
        "CASE D": "Recovery and grouping validate, but localization failed or violated an invariant.",
        "CASE E": "All structural stages validate, but the standard compact estimator has non-negligible exact physical trace distance.",
        "PEELING FAILURE": "Exact peeling did not produce an operational certified span at the frozen thresholds.",
    }.get(result["case"], "See failure_stage and failure_reason.")
    lines = [
        "EXACT 10Q 3-3-3-1 FINAL STRUCTURAL DIAGNOSTIC",
        "=" * 53,
        "",
        f"Run timestamp: {result.get('run_timestamp', 'UNAVAILABLE')}",
        f"Git base commit: {result.get('git_base_commit', 'UNAVAILABLE')}",
        f"Git final commit: {result.get('git_final_commit', 'UNAVAILABLE')}",
        f"Init index: {init_index}",
        f"Frozen state: {state_path.relative_to(ROOT)}",
        f"Frozen state SHA-256: {metadata['file_sha256']}",
        f"Content fingerprint: {metadata['content_fingerprint']}",
        f"Master seed: {metadata['master_seed']}",
        f"Block seeds: {metadata['block_seeds']}",
        f"Clifford seed: {metadata['clifford_seed']}",
        "n=10; d=3; hidden partition=[3, 3, 3, 1]",
        f"Latent blocks: 4; sizes={list(EXPECTED_PARTITION)}",
        f"Compact Clifford gate count: {metadata['clifford_gate_count']}",
        "",
        "THRESHOLD PROVENANCE",
        "--------------------",
        f"Provenance: {thresholds.provenance}",
        f"Source: {thresholds.source}",
        f"Source SHA-256: {thresholds.source_sha256}",
        f"Source record: {thresholds.source_record}",
        "Original completed result artifact was not found; the authorized prior-recorded 5M values are used unchanged."
        if thresholds.provenance == "reconstructed_from_prior_recorded_calibration_values"
        else "An authentic completed calibration record was resolved deterministically.",
        f"h_min={thresholds.h_min:.17g}",
        f"h_max={thresholds.h_max:.17g}",
        f"grid_intervals={thresholds.grid_intervals}",
        f"grid_spacing={(thresholds.h_max-thresholds.h_min)/thresholds.grid_intervals:.17g}",
        f"theta={thresholds.theta:.17g}",
        f"eta_test={thresholds.eta_test:.17g}",
        "These are prior frozen calibration thresholds, not newly optimized thresholds.",
        "Thresholds were not selected using the hidden partition or exact state.",
        "",
        "PEELING",
        "-------",
        f"success={None if peeling is None else peeling.success}",
        f"failure_reason={None if peeling is None else peeling.failure_reason}",
        f"selected_h={None if peeling is None else peeling.h}",
        f"lambda={None if peeling is None else peeling.lambda_}",
        f"t={None if peeling is None else peeling.t}",
        f"generators={None if peeling is None else peeling.generators}",
        f"generator_count={None if peeling is None else len(peeling.generators)}",
        f"certified_span_rank={None if peeling is None else len(peeling.certified_span_basis)}",
        f"theorem_grid_condition={None if peeling is None else peeling.theorem_grid_condition}",
        f"theorem_tau_condition={None if peeling is None else peeling.theorem_tau_condition}",
        f"theorem_certified={None if peeling is None else peeling.theorem_certified}",
        f"score_table_bytes={4**10 * np.dtype(np.float64).itemsize}",
        f"copy_ledger={None if peeling is None else peeling.copy_ledger.entries}",
        "",
        "RECOVERY",
        "--------",
        f"success={None if recovery is None else recovery.success}; L={None if recovery is None else len(recovery.sectors)}",
        f"failure_reason={None if recovery is None else recovery.failure_reason}",
        f"sector_id_kind={sector_summary}",
        f"recovered_span_rank={None if recovery is None else len(recovery.recovered_span_basis)}",
        f"threshold_span_complete={None if recovery is None else recovery.threshold_span_complete}",
        f"threshold_margin_holds={None if recovery is None else recovery.threshold_margin_holds}",
        f"ranking_gap_margin_holds={None if recovery is None else recovery.ranking_gap_margin_holds}",
        f"theorem_recovery_preconditions_hold={None if recovery is None else recovery.theorem_recovery_preconditions_hold}",
        f"oracle_sector_block_labels={result['sector_labels']}",
        f"all_sectors_block_pure={result['sector_pure']}",
        f"copy_ledger={None if recovery is None else recovery.copy_ledger.entries}",
        "",
        "GROUPING",
        "--------",
        f"success={None if grouping is None else grouping.success}",
        f"failure_reason={None if grouping is None else grouping.failure_reason}",
        f"clusters={None if grouping is None else grouping.clusters}",
        f"cluster_sizes={cluster_sizes}",
        f"cluster_oracle_labels={result['cluster_labels']}",
        f"all_clusters_block_pure={result['cluster_pure']}",
        f"oracle_residual_sector_clusters={result['oracle_residual_clusters']}",
        f"matches_oracle_residual_grouping={result['target_grouping']}",
        "For t>0 this comparison uses oracle residual sector labels, not the original [3,3,3,1] sizes.",
        f"realized_query_count={None if grouping is None else grouping.realized_query_count}",
        f"query_count_by_order={None if grouping is None else grouping.query_count_by_order}",
        f"merge_rounds={None if grouping is None else grouping.merge_rounds}",
        f"grouping_complete={None if grouping is None else grouping.grouping_complete}",
        f"grouping_budget_truncated={None if grouping is None else grouping.grouping_budget_truncated}",
        f"copy_ledger={None if grouping is None else grouping.grouping_copy_ledger.entries}",
        "",
        "LOCALIZATION",
        "------------",
        f"success={None if localization is None else localization.success}",
        f"failure_reason={None if localization is None else localization.failure_reason}",
        f"J_C={None if localization is None else localization.J_C}",
        f"cluster_k_C_registers={localization_summary}",
        f"J_aux={j_aux}",
        f"J_aux_size={len(j_aux)}",
        f"K_rec={None if localization is None else localization.K_rec}",
        f"invariants={_format_pairs(result['localization_flags'].items())}",
        f"grouping_theorem_preconditions_hold={None if localization is None else localization.grouping_theorem_preconditions_hold}",
        f"theorem_localization_preconditions_hold={None if localization is None else localization.theorem_localization_preconditions_hold}",
        "localization_copy_count=0",
        "",
        "EXACT RECONSTRUCTION",
        "--------------------",
        f"syndrome_expectations={result['syndrome_expectations']}",
        f"syndrome_bits={result['syndrome_bits']}",
        "t>0 convention: explicit product |b_exact><b_exact|; t=0: empty prefix",
        "Register factors: exact localized marginals",
        f"register_marginal_summaries=(cluster, register, shape, trace): {result['register_marginal_summaries']}",
        "J_aux factor: maximally mixed",
        "Estimator: CompactCEBPEstimator + standard physical decoder",
        f"localized_structural_trace_distance={result['localized_trace_distance']}",
        f"exact_physical_trace_distance={result['trace_distance']}",
        f"zero_tolerance={TRACE_TOLERANCE:.17g}",
        f"numerically_zero={result['trace_distance'] is not None and result['trace_distance'] <= TRACE_TOLERANCE}",
        f"zero_copy_verification={result['stage_ledgers']}",
        "Finite-budget empirical run: NOT RUN",
        "",
        "RUNTIME AND MEMORY",
        "------------------",
        f"stage_wall_times={result['stage_times']}",
        f"structural_runtime_seconds={result['elapsed']:.6f}",
        f"total_runtime_seconds={result.get('total_elapsed', result['elapsed']):.6f}",
        f"peak_rss_bytes={result['peak_rss_bytes']}",
        f"peak_rss_mib={None if result['peak_rss_bytes'] is None else result['peak_rss_bytes'] / (1024**2):.6f}" if result['peak_rss_bytes'] is not None else "peak_rss_mib=UNAVAILABLE",
        f"dense_final_comparison_entered={dense_entered}",
        "Structural peeling/recovery/grouping/localization/marginals use compact or structured backends.",
        "The bounded final n=10 comparison materializes dense true/decoded states and two dense 1024x1024 Clifford unitaries plus decoder/workspace; each complex128 1024x1024 matrix is about 16 MiB before overhead. Dense objects are not retained after the comparison."
        if dense_entered
        else "The bounded dense n=10 final comparison was not entered because an earlier structural stage failed.",
        f"failure_stage={result['failure_stage']}",
        f"failure_reason={result['failure_reason']}",
        "",
        "SCIENTIFIC INTERPRETATION",
        "-------------------------",
        result["case"],
        interpretation,
        "No threshold was oracle-retuned and no finite-budget empirical tomography was run.",
        "",
        "IMPLEMENTATION ISSUES",
        "---------------------",
        "None during execution."
        if result["failure_stage"] is None
        else f"None. Scientific/structural stop: {result['failure_stage']} — {result['failure_reason']}",
        "",
        "EXACT COMMAND",
        "-------------",
        result.get("exact_command", "UNAVAILABLE"),
        "",
        "STRUCTURAL EXACT-DATA DIAGNOSTIC: " + ("PASS" if result["passed"] else "FAIL"),
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-index", type=int, default=0)
    parser.add_argument(
        "--candidate-file",
        type=Path,
        help="explicit completed result/candidate JSON; otherwise discover results/result.json",
    )
    parser.add_argument("--details", action="store_true")
    return parser


def _write_exception_report(
    *,
    init_index: int,
    state_path: Path,
    metadata: dict[str, Any],
    thresholds: Thresholds,
    error: Exception,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "EXACT 10Q 3-3-3-1 STRUCTURAL DIAGNOSTIC",
                "=" * 47,
                "",
                "Status: FAIL — implementation exception",
                f"Init index: {init_index}",
                f"Frozen state: {state_path.relative_to(ROOT)}",
                f"Frozen state SHA-256: {metadata['file_sha256']}",
                f"Threshold source: {thresholds.source}",
                "",
                "IMPLEMENTATION EXCEPTION",
                "------------------------",
                f"{type(error).__name__}: {error}",
                "",
            )
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    program_started = time.perf_counter()
    run_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    exact_command = shlex.join(
        [sys.executable, str(Path(__file__).resolve()), *effective_argv]
    )
    git_base_commit = _git_head()
    args = _parser().parse_args(effective_argv)
    if not 0 <= args.init_index <= 19:
        raise SystemExit("--init-index must lie in [0, 19]")
    instance_io = _load_instance_io()
    state_path = CASE_DIR / "states" / f"init_{args.init_index:03d}.npz"
    instance, metadata = instance_io.load_instance(
        state_path, expected_init_id=args.init_index
    )
    partition = tuple(len(block) for block in instance.oracle_truth.hidden_partition)
    if (instance.n, instance.d, partition) != (10, 3, EXPECTED_PARTITION):
        raise RuntimeError("Frozen instance metadata/partition invariant failed.")
    if len(instance.oracle_truth.latent_block_states) != 4:
        raise RuntimeError("Frozen instance does not contain four latent blocks.")
    try:
        thresholds = resolve_thresholds(args.init_index, args.candidate_file)
    except ThresholdSourceError as error:
        _write_blocked_report(
            init_index=args.init_index,
            state_path=state_path,
            metadata=metadata,
            reason=str(error),
        )
        print(
            f"init={args.init_index:03d} BLOCKED threshold_source_missing "
            "t=NA L=NA clusters=NA J_aux=NA D_exact=NA"
        )
        if args.details:
            print(str(error), file=sys.stderr)
        return 2
    loading_elapsed = time.perf_counter() - program_started
    try:
        result = run_exact(instance, thresholds, args.details)
    except Exception as error:
        _write_exception_report(
            init_index=args.init_index,
            state_path=state_path,
            metadata=metadata,
            thresholds=thresholds,
            error=error,
        )
        print(
            f"init={args.init_index:03d} FAIL implementation_exception "
            f"type={type(error).__name__} message={error}",
            file=sys.stderr,
        )
        if args.details:
            raise
        return 3
    result["stage_times"] = {
        "state_and_candidate_loading": loading_elapsed,
        **result["stage_times"],
    }
    result["run_timestamp"] = run_timestamp
    result["git_base_commit"] = git_base_commit
    result["git_final_commit"] = _git_head()
    result["exact_command"] = exact_command
    result["total_elapsed"] = time.perf_counter() - program_started
    _write_completed_report(
        init_index=args.init_index,
        state_path=state_path,
        metadata=metadata,
        thresholds=thresholds,
        result=result,
    )
    peeling = result["peeling"]
    recovery = result["recovery"]
    grouping = result["grouping"]
    localization = result["localization"]
    t = None if peeling is None else peeling.t
    sector_count = None if recovery is None else len(recovery.sectors)
    kinds = None if recovery is None else tuple(sector.kind for sector in recovery.sectors)
    labels = result["sector_labels"]
    cluster_sizes = () if grouping is None else tuple(map(len, grouping.clusters))
    j_aux_size = None if localization is None else len(localization.J_aux)
    trace_distance = result["trace_distance"]
    print(
        f"init={args.init_index:03d} provenance={thresholds.provenance} "
        f"h=[{thresholds.h_min:.6g},{thresholds.h_max:.6g}] "
        f"theta={thresholds.theta:.6g} eta_test={thresholds.eta_test:.6g} "
        f"t={t} L={sector_count} kinds={kinds} labels={labels} "
        f"clusters={cluster_sizes} J_aux={j_aux_size} syndrome={result['syndrome_bits']} "
        f"D_exact={trace_distance} {result['case']} "
        f"runtime={result['total_elapsed']:.6f}s "
        f"{'PASS' if result['passed'] else 'FAIL'}"
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
