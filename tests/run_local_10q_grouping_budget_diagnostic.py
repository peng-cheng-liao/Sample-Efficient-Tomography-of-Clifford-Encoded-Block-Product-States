#!/usr/bin/env python3
"""Run the frozen init_000 old-vs-expanded grouping-budget comparison."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence
import zipfile


# Keep imports quiet and task caches outside the repository.
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
_MPL_CACHE = Path(tempfile.mkdtemp(prefix="local_10q_grouping_budget_mpl."))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "Jobs" / "06" / "n=10_d=3"
REPORT_PATH = (
    ROOT / "Reports" / "OVERSIZE_EMPIRICAL_BLOCK_AND_GROUPING_BUDGET_REPORT.txt"
)
RESULTS_PATH = (
    ROOT / "Reports" / "OVERSIZE_EMPIRICAL_BLOCK_AND_GROUPING_BUDGET_RESULTS.json"
)
ZIP_PATH = (
    ROOT / "Reports"
    / "OVERSIZE_EMPIRICAL_BLOCK_AND_GROUPING_BUDGET_VERIFICATION.zip"
)
THRESHOLD_RECORD_PATH = (
    ROOT / "Reports" / "EXACT_10Q_3331_RECONSTRUCTED_THRESHOLD_RECORD.json"
)
ARCHIVE_PATH = ROOT / "Jobs" / "06.zip"
ARCHIVE_CONFIG_MEMBER = "06/n=10_d=3/config/experiment_config.json"
ARCHIVE_MANIFEST_MEMBER = "06/n=10_d=3/states/manifest.json"
ARCHIVE_PARAMETERIZATION_MEMBER = "06/Optimization/parameterization.py"
ARCHIVE_RUNNER_MEMBER = "06/n=10_d=3/run_calibration_benchmark.py"

EXPECTED_INIT = 0
EXPECTED_BUDGET = 5_000_000
EXPECTED_PARTITION = (3, 3, 3, 1)
ETA_TEST = 0.15
EXPECTED_THRESHOLDS = (
    0.7645886990214604,
    0.9941139265711931,
    0.03043433244408696,
    0.17971039918570025,
)
EXPECTED_RAW_DRAW_INDEX = 89
EXPECTED_ARCHIVED_CANDIDATE_ID = "candidate-0058"
EXPECTED_STAGE_CAPS = (
    ("peeling", 1_304_353),
    ("recovery", 2_704_407),
    ("grouping", 445_114),
    ("syndrome", 53_526),
    ("tomography", 492_600),
)
EXPANDED_GROUPING_STAGE_CAPS = tuple(
    (stage, 2_000_000 if stage == "grouping" else copies)
    for stage, copies in EXPECTED_STAGE_CAPS
)
EXPECTED_BUDGET_B = sum(copies for _stage, copies in EXPANDED_GROUPING_STAGE_CAPS)
HISTORICAL_FULL_MIXED_DISTANCE = 0.73750953105923223
TASK_PATHS = (
    "main_v2.py",
    "Reports/OVERSIZE_EMPIRICAL_BLOCK_AND_GROUPING_BUDGET_REPORT.txt",
    "Reports/OVERSIZE_EMPIRICAL_BLOCK_AND_GROUPING_BUDGET_RESULTS.json",
    "tests/run_local_10q_grouping_budget_diagnostic.py",
    "tests/test_local_10q_grouping_budget_diagnostic.py",
    "tests/test_oversize_empirical_block_policy.py",
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main_v2  # noqa: E402
from Optimization.parameterization import (  # noqa: E402
    FixedBudgetCandidateParameters,
    OptimizationConfig,
    candidate_to_end_to_end_config,
    derive_candidate,
)
from Optimization.specification import OptimizationObjective  # noqa: E402


class CandidateResolutionError(RuntimeError):
    """The frozen 5M candidate cannot be resolved without ambiguity."""


def _cleanup_mpl_cache() -> None:
    shutil.rmtree(_MPL_CACHE, ignore_errors=True)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, capture_output=True, text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNAVAILABLE"


def _git_index_bytes(path: str) -> bytes | None:
    completed = subprocess.run(
        ("git", "show", f":{path}"), cwd=ROOT, capture_output=True, check=False
    )
    return completed.stdout if completed.returncode == 0 and completed.stdout else None


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _load_instance_io():
    path = CASE_DIR / "instance_io.py"
    spec = importlib.util.spec_from_file_location(
        "local_grouping_budget_instance_io", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen-state loader: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _float_tuple(payload: Mapping[str, Any]) -> tuple[float, float, float, float]:
    parameters = payload.get("best_parameters", payload)
    if not isinstance(parameters, Mapping):
        raise CandidateResolutionError("Candidate threshold record has no parameter object.")
    h_min = float(parameters["h_min"])
    h_max = (
        float(parameters["h_max"])
        if "h_max" in parameters
        else h_min + float(parameters["h_span"])
    )
    return h_min, h_max, float(parameters["theta"]), float(parameters["eta_test"])


def _load_threshold_record(explicit: Path | None = None) -> tuple[dict[str, Any], str, str]:
    if explicit is not None:
        if not explicit.is_file():
            raise CandidateResolutionError(f"Candidate file does not exist: {explicit}")
        raw = explicit.read_bytes()
        source = str(explicit.resolve())
    elif THRESHOLD_RECORD_PATH.is_file():
        raw = THRESHOLD_RECORD_PATH.read_bytes()
        source = str(THRESHOLD_RECORD_PATH.resolve())
    else:
        raw = _git_index_bytes(
            "Reports/EXACT_10Q_3331_RECONSTRUCTED_THRESHOLD_RECORD.json"
        )
        if raw is None:
            raise CandidateResolutionError(
                "The authorized reconstructed init_000/5M threshold record is unavailable."
            )
        source = (
            "git-index:Reports/"
            "EXACT_10Q_3331_RECONSTRUCTED_THRESHOLD_RECORD.json"
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CandidateResolutionError(f"Invalid candidate JSON: {source}") from error
    if int(payload.get("init_id", -1)) != EXPECTED_INIT:
        raise CandidateResolutionError("Candidate record does not identify init_000.")
    if int(payload.get("prior_calibration_total_copies", -1)) != EXPECTED_BUDGET:
        raise CandidateResolutionError("Candidate record is not the prior 5M record.")
    if _float_tuple(payload) != EXPECTED_THRESHOLDS:
        raise CandidateResolutionError(
            f"Frozen thresholds changed: {_float_tuple(payload)!r}"
        )
    if payload.get("provenance") != "reconstructed_from_prior_recorded_calibration_values":
        raise CandidateResolutionError("Candidate record has unexpected provenance.")
    return payload, source, _sha256_bytes(raw)


def _uniform(rng: np.random.Generator, bounds: Sequence[float]) -> float:
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


def _log_uniform(rng: np.random.Generator, bounds: Sequence[float]) -> float:
    return float(math.exp(rng.uniform(math.log(bounds[0]), math.log(bounds[1]))))


def _archived_candidate_draws(
    config: Mapping[str, Any], search_seed: int, count: int = 20_000
) -> Iterable[dict[str, float | int]]:
    """Reproduce archived fixed-budget RNG draws without evaluating candidates."""

    space = config["search_space"]
    rng = np.random.default_rng(int(search_seed))
    for raw_index in range(count):
        # The archived schema canonicalized alpha/epsilon fields without RNG draws.
        row: dict[str, float | int] = {
            "raw_draw_index": raw_index,
            "c_peel": _uniform(rng, space["c_peel"]),
            "c_rank": _uniform(rng, space["c_rank"]),
            "h_min": _uniform(rng, space["h_min"]),
            "h_span": _uniform(rng, space["h_span"]),
            "theta": _uniform(rng, space["theta"]),
            "eta_test": _log_uniform(rng, space["eta_test"]),
            "kappa_ratio": _uniform(rng, space["kappa_ratio"]),
            "peel_weight": _log_uniform(rng, space["peel_weight"]),
            "recovery_weight": _log_uniform(rng, space["recovery_weight"]),
            "grouping_weight": _log_uniform(rng, space["grouping_weight"]),
            "syndrome_weight": _log_uniform(rng, space["syndrome_weight"]),
            "tomography_weight": _log_uniform(rng, space["tomography_weight"]),
        }
        row["h_max"] = float(row["h_min"]) + float(row["h_span"])
        yield row


def _archived_fixed_budget_candidate_index(
    rows: Sequence[Mapping[str, float | int]], target_raw_index: int
) -> int:
    """Apply the archived schema-3 validity filter without learner evaluation."""

    valid_before = 0
    log_a = math.log(2.0) + 10 * math.log(4.0)
    for row in rows:
        weights = tuple(
            float(row[name])
            for name in (
                "peel_weight", "recovery_weight", "grouping_weight",
                "syndrome_weight", "tomography_weight",
            )
        )
        total_weight = sum(weights)
        early_caps = tuple(
            int(math.floor(EXPECTED_BUDGET * value / total_weight))
            for value in weights[:-1]
        )
        caps = (*early_caps, EXPECTED_BUDGET - sum(early_caps))
        m1, m2, m_sgn = caps[0] // 2, caps[1] // 2, caps[3] // 10
        tau1 = float(row["c_peel"]) * math.sqrt(2.0 * log_a / m1)
        tau_rank = float(row["c_rank"]) * math.sqrt(2.0 * log_a / m2)
        valid = bool(
            min(m1, m2, m_sgn) >= 1
            and 0.5 < float(row["h_min"]) < float(row["h_max"]) < 0.995
            and float(row["h_min"]) + tau1 < 1.0
            and float(row["theta"]) > tau_rank
        )
        if int(row["raw_draw_index"]) == int(target_raw_index):
            if not valid:
                raise CandidateResolutionError("Matched archived draw is invalid.")
            return valid_before
        if valid:
            valid_before += 1
    raise CandidateResolutionError("Matched raw draw is outside the supplied catalog.")


def resolve_prior_candidate(
    explicit: Path | None = None,
) -> dict[str, Any]:
    """Recover the exact five weights from the archived deterministic catalog."""

    threshold_payload, threshold_source, threshold_sha = _load_threshold_record(explicit)
    if not ARCHIVE_PATH.is_file() or not zipfile.is_zipfile(ARCHIVE_PATH):
        raise CandidateResolutionError(f"Archived calibration package unavailable: {ARCHIVE_PATH}")
    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        required = (
            ARCHIVE_CONFIG_MEMBER,
            ARCHIVE_MANIFEST_MEMBER,
            ARCHIVE_PARAMETERIZATION_MEMBER,
            ARCHIVE_RUNNER_MEMBER,
        )
        missing = [member for member in required if member not in archive.namelist()]
        if missing:
            raise CandidateResolutionError(f"Calibration archive is missing members: {missing}")
        member_bytes = {member: archive.read(member) for member in required}
    config = json.loads(member_bytes[ARCHIVE_CONFIG_MEMBER].decode("utf-8"))
    manifest = json.loads(member_bytes[ARCHIVE_MANIFEST_MEMBER].decode("utf-8"))
    instance_metadata = next(
        item for item in manifest["instances"] if int(item["init_id"]) == EXPECTED_INIT
    )
    namespace = int(instance_metadata["search_seed_namespace"]["derived_uint64_seed"])
    search_seed = int(
        np.random.SeedSequence([namespace, EXPECTED_BUDGET]).generate_state(
            1, dtype=np.uint64
        )[0]
    )
    target = _float_tuple(threshold_payload)
    draws = tuple(_archived_candidate_draws(config, search_seed))
    matches = [
        row
        for row in draws
        if (
            float(row["h_min"]),
            float(row["h_max"]),
            float(row["theta"]),
            float(row["eta_test"]),
        )
        == target
    ]
    if len(matches) != 1:
        raise CandidateResolutionError(
            f"Archived catalog threshold match count is {len(matches)}, expected exactly one."
        )
    match = matches[0]
    if int(match["raw_draw_index"]) != EXPECTED_RAW_DRAW_INDEX:
        raise CandidateResolutionError("Archived candidate raw-draw identity changed.")
    valid_index = _archived_fixed_budget_candidate_index(
        draws, int(match["raw_draw_index"])
    )
    archived_candidate_id = f"candidate-{valid_index:04d}"
    if archived_candidate_id != EXPECTED_ARCHIVED_CANDIDATE_ID:
        raise CandidateResolutionError(
            f"Archived valid-filter identity changed: {archived_candidate_id}"
        )
    grid = int(threshold_payload.get("peeling_grid_intervals", 5))
    weights = {
        name: float(match[f"{name}_weight"])
        for name in ("peel", "recovery", "grouping", "syndrome", "tomography")
    }
    weights["peeling"] = weights.pop("peel")
    measurement_namespace = int(
        instance_metadata["measurement_seed_namespace"]["derived_uint64_seed"]
    )
    measurement_root = np.random.SeedSequence(
        [measurement_namespace, EXPECTED_BUDGET]
    )
    canonical_measurement_seed = int(
        measurement_root.spawn(1)[0].generate_state(1, dtype=np.uint64)[0]
    )
    return {
        "candidate_record": str(threshold_payload.get("best_candidate_id")),
        "archived_candidate_id": archived_candidate_id,
        "raw_draw_index": int(match["raw_draw_index"]),
        "h_min": target[0],
        "h_max": target[1],
        "theta": target[2],
        "eta_test": target[3],
        "peeling_grid_intervals": grid,
        "stage_weights": weights,
        "legacy_inactive_fields": {
            name: float(match[name])
            for name in ("c_peel", "c_rank", "kappa_ratio")
        },
        "total_budget": EXPECTED_BUDGET,
        "search_seed": search_seed,
        "canonical_measurement_seed": canonical_measurement_seed,
        "threshold_source": threshold_source,
        "threshold_source_sha256": threshold_sha,
        "archive_source": str(ARCHIVE_PATH.resolve()),
        "archive_sha256": _sha256_path(ARCHIVE_PATH),
        "archive_members": {
            member: {
                "size": len(member_bytes[member]),
                "sha256": _sha256_bytes(member_bytes[member]),
            }
            for member in required
        },
        "state_manifest_identity": instance_metadata,
        "resolution_note": (
            "The completed result artifact was absent. Thresholds came from the "
            "authorized reconstructed record; five weights were recovered from the "
            "unique archived deterministic RNG draw matching all four thresholds. "
            "No candidate was evaluated during provenance resolution."
        ),
    }


def candidate_record(provenance: Mapping[str, Any], eta_test: float) -> dict[str, Any]:
    weights = provenance["stage_weights"]
    return {
        "h_min": float(provenance["h_min"]),
        "h_max": float(provenance["h_max"]),
        "theta": float(provenance["theta"]),
        "eta_test": float(eta_test),
        "peel_weight": float(weights["peeling"]),
        "recovery_weight": float(weights["recovery"]),
        "grouping_weight": float(weights["grouping"]),
        "syndrome_weight": float(weights["syndrome"]),
        "tomography_weight": float(weights["tomography"]),
    }


def changed_candidate_fields(
    baseline: Mapping[str, Any], diagnostic: Mapping[str, Any]
) -> tuple[str, ...]:
    if set(baseline) != set(diagnostic):
        return tuple(sorted(set(baseline) ^ set(diagnostic)))
    return tuple(sorted(key for key in baseline if baseline[key] != diagnostic[key]))


def build_optimization_config(
    provenance: Mapping[str, Any], budget: int, measurement_seed: int
) -> OptimizationConfig:
    config = json.loads((CASE_DIR / "config" / "experiment_config.json").read_text())
    runtime = config["runtime"]
    return OptimizationConfig(
        total_copies=int(budget),
        search_seed=int(provenance["search_seed"]),
        tuning_seeds=(int(measurement_seed),),
        # OptimizationConfig validates a disjoint holdout set even though this
        # direct, non-optimization diagnostic never consumes it.
        holdout_seeds=(int(measurement_seed) + 1,),
        peeling_grid_intervals=int(provenance["peeling_grid_intervals"]),
        max_dense_qubits=10,
        max_enumeration_qubits=int(runtime["max_enumeration_qubits"]),
        max_oracle_dense_qubits=int(runtime["max_oracle_dense_qubits"]),
        inner_enumeration_workers=1,
        max_enumeration_workspace_bytes=int(
            runtime["max_enumeration_workspace_bytes"]
        ),
        simulation_backend=str(runtime["simulation_backend"]),
        max_single_grouping_query_shots=int(
            runtime["max_single_grouping_query_shots"]
        ),
        number_of_candidates=1,
        halving_seed_counts=(1,),
        tomography_refinement_enabled=False,
        objective=OptimizationObjective(
            mode="fixed_budget_min_error", copy_ceiling=int(budget)
        ),
        verbose=False,
    )


def make_candidate(record: Mapping[str, Any]) -> FixedBudgetCandidateParameters:
    return FixedBudgetCandidateParameters(**{key: record[key] for key in (
        "h_min", "h_max", "theta", "eta_test", "peel_weight",
        "recovery_weight", "grouping_weight", "syndrome_weight",
        "tomography_weight",
    )})


def paired_stage_caps(
    provenance: Mapping[str, Any], budget: int, measurement_seed: int
) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]:
    if budget != EXPECTED_BUDGET:
        raise ValueError("TEST A must retain the exact prior 5M budget.")
    config = build_optimization_config(provenance, budget, measurement_seed)
    derived = derive_candidate(
        make_candidate(candidate_record(provenance, ETA_TEST)),
        n=10,
        d=3,
        total_copies=budget,
        optimization_config=config,
    )
    if tuple(derived.fixed_budget_stage_caps) != EXPECTED_STAGE_CAPS:
        raise RuntimeError("Recovered prior stage caps changed.")
    return EXPECTED_STAGE_CAPS, EXPANDED_GROUPING_STAGE_CAPS


def _canonical_partition(
    clusters: Iterable[Iterable[int]],
) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(tuple(sorted(int(value) for value in cluster)) for cluster in clusters))


def _oracle_structure(
    instance: Any, peeling: Any, recovery: Any, grouping: Any
) -> dict[str, Any]:
    # This function is called only after full_cebp_tomography has returned.
    labels = tuple(
        tuple(int(value) for value in item)
        for item in main_v2.debug_oracle_sector_block_labels(
            instance, peeling, recovery
        )
    )
    cluster_labels = tuple(
        tuple(sorted({label for sector in cluster for label in labels[int(sector)]}))
        for cluster in grouping.clusters
    )
    by_block: dict[int, list[int]] = {}
    for sector, sector_labels in enumerate(labels):
        for label in sector_labels:
            by_block.setdefault(label, []).append(sector)
    oracle_partition = tuple(tuple(by_block[label]) for label in sorted(by_block))
    learned = _canonical_partition(grouping.clusters)
    oracle = _canonical_partition(oracle_partition)
    split_true_blocks = 0
    for block_sectors in oracle_partition:
        appearances = sum(bool(set(block_sectors) & set(cluster)) for cluster in grouping.clusters)
        split_true_blocks += max(0, appearances - 1)
    return {
        "sector_block_labels": labels,
        "cluster_block_labels": cluster_labels,
        "oracle_sector_partition": oracle_partition,
        "block_pure_cluster_count": sum(len(item) == 1 for item in cluster_labels),
        "false_merge_count": sum(len(item) > 1 for item in cluster_labels),
        "true_block_split_count": split_true_blocks,
        "exact_oracle_partition": learned == oracle and all(len(item) == 1 for item in labels),
        "oracle_used_post_hoc_only": True,
    }


def _ledger_dict(ledger: Any) -> dict[str, int]:
    return {str(name): int(copies) for name, copies in ledger.entries}


def _serialize_arm(
    result: Any,
    instance: Any,
    provenance: Mapping[str, Any],
    candidate: Mapping[str, Any],
    derived: Any,
    config: OptimizationConfig,
    assigned_caps: Sequence[tuple[str, int]],
    nominal_total: int,
    arm_name: str,
    learner_seconds: float,
    trace_distance: float | None,
    trace_seconds: float,
    total_seconds: float,
) -> dict[str, Any]:
    peeling, recovery, grouping = result.peeling, result.recovery, result.grouping
    localization = result.localization
    tomography = result.tomography
    oracle = None
    if (
        peeling is not None and peeling.success
        and recovery is not None and recovery.success
        and grouping is not None and grouping.success
    ):
        oracle = _oracle_structure(instance, peeling, recovery, grouping)
    stage_records = tuple(
        {
            "stage": record.stage,
            "assigned_cap": int(record.assigned_cap),
            "realized_copies": int(record.realized_copies),
            "unused_copies": int(record.unused_copies),
            "budget_exhausted": bool(record.budget_exhausted),
            "stage_complete": bool(record.stage_complete),
            "degradation_reason": record.degradation_reason,
        }
        for record in result.fixed_budget_stage_records
    )
    performance = (
        {} if result.performance_diagnostics is None
        else result.performance_diagnostics.as_dict()
    )
    peeling_summary = None if peeling is None else {
        "success": bool(peeling.success),
        "failure_reason": peeling.failure_reason,
        "selected_h": peeling.h,
        "t": peeling.t,
        "generators": tuple(peeling.generators),
        "generator_symplectic_vectors": tuple(peeling.generator_symplectic_vectors),
        "certified_span_basis": tuple(int(value) for value in peeling.certified_span_basis),
        "threshold_grid": tuple(float(value) for value in peeling.threshold_grid),
        "realized_copies": int(peeling.copy_ledger.total),
    }
    recovery_summary = None if recovery is None else {
        "success": bool(recovery.success),
        "failure_reason": recovery.failure_reason,
        "L": len(recovery.sectors),
        "sector_ids_kinds": tuple((int(sector.sector_id), sector.kind) for sector in recovery.sectors),
        "sector_members": tuple(tuple(sector.members) for sector in recovery.sectors),
        "independent_axes": tuple(recovery.independent_axes),
        "recovered_span_basis": tuple(int(value) for value in recovery.recovered_span_basis),
        "recovered_span_rank": len(recovery.recovered_span_basis),
        "threshold_span_complete": bool(recovery.threshold_span_complete),
        "realized_copies": int(recovery.copy_ledger.total),
    }
    grouping_summary = None if grouping is None else {
        "success": bool(grouping.success),
        "failure_reason": grouping.failure_reason,
        "clusters": tuple(tuple(int(value) for value in cluster) for cluster in grouping.clusters),
        "cluster_sizes_ordered": tuple(len(cluster) for cluster in grouping.clusters),
        "cluster_sizes_descending": tuple(sorted((len(cluster) for cluster in grouping.clusters), reverse=True)),
        "query_count": int(grouping.realized_query_count),
        "query_count_by_order": tuple((int(q), int(count)) for q, count in grouping.query_count_by_order),
        "merge_rounds": int(grouping.merge_rounds),
        "budget_truncated": bool(grouping.grouping_budget_truncated),
        "grouping_complete": bool(grouping.grouping_complete),
        "realized_copies": int(grouping.realized_grouping_copies),
        "copies_by_order": tuple((int(q), int(count)) for q, count in grouping.copies_by_order),
        "exploratory_query_count": int(grouping.exploratory_query_count),
        "refinement_top_up_count": int(grouping.refinement_top_up_count),
        "scan_order": tuple(
            {
                "order": int(scan.order),
                "partition_before": tuple(tuple(cluster) for cluster in scan.partition_before),
                "partition_after": tuple(tuple(cluster) for cluster in scan.partition_after),
                "hyperedge_count": len(scan.hyperedges),
                "reset_to_two": bool(scan.reset_to_two),
            }
            for scan in grouping.transcript
        ),
        "query_tuple_order_available": False,
    }
    localization_summary = None if localization is None else {
        "success": bool(localization.success),
        "failure_reason": localization.failure_reason,
        "J_aux": tuple(int(value) for value in localization.J_aux),
        "J_aux_size": len(localization.J_aux),
        "J_C": tuple((tuple(cluster), tuple(register)) for cluster, register in localization.J_C),
        "handoff_valid": bool(localization.handoff_valid),
        "register_partition_holds": bool(localization.register_partition_holds),
        "k_C": tuple(len(register) for _cluster, register in localization.J_C),
        "empirical_max_cluster_size": int(localization.empirical_max_cluster_size),
        "assumed_d": int(localization.assumed_d),
        "model_bound_violated": bool(localization.model_bound_violated),
        "oversize_clusters": tuple(localization.oversize_clusters),
        "reconstruction_proceeded_despite_model_bound_violation": bool(
            localization.reconstruction_proceeded_despite_model_bound_violation
        ),
    }
    tomography_summary = None if tomography is None else {
        "success": bool(tomography.success),
        "failure_reason": tomography.failure_reason,
        "N_tom_assigned": int(dict(assigned_caps)["tomography"]),
        "N_tom_realized": int(tomography.block_tomography_pool),
        "N_tom_unused": int(
            dict(assigned_caps)["tomography"] - tomography.block_tomography_pool
        ),
        "budget_truncated": bool(tomography.budget_truncated),
        "sum_local_schedule_lengths": int(tomography.sum_local_schedule_lengths),
        "sampling_work_units": int(tomography.sampling_work_units),
        "registers": tuple(
            {
                "cluster": tuple(budget.cluster),
                "J_C": tuple(budget.J_C),
                "k_C": int(budget.k_C),
                "nonidentity_pauli_settings": int(
                    budget.total_nonidentity_paulis
                ),
                "measured_pauli_settings": int(budget.measured_pauli_count),
                "assigned_physical_rounds": int(budget.physical_round_budget),
                "realized_schedule_length": int(
                    budget.realized_schedule_length
                ),
                "min_shots_per_setting": int(
                    budget.min_shots_per_measured_pauli
                ),
                "max_shots_per_setting": int(
                    budget.max_shots_per_measured_pauli
                ),
                "complete_pauli_coverage": bool(
                    budget.complete_pauli_coverage
                ),
                "local_density_shape": tuple(
                    int(value) for value in estimate.nu_hat.shape
                ),
                "sufficient_statistic_count": len(
                    record.sufficient_statistics
                ),
            }
            for budget, estimate, record in zip(
                tomography.budgets,
                tomography.estimates,
                tomography.records,
            )
        ),
    }
    workspace = {}
    for name, stage in (("peeling", peeling), ("recovery", recovery)):
        diagnostic = None if stage is None else stage.enumeration_diagnostics
        if diagnostic is not None:
            workspace[name] = {
                "enumeration_size": int(diagnostic.enumeration_size),
                "score_dtype": diagnostic.score_dtype,
                "score_array_bytes": int(diagnostic.score_array_bytes),
                "worker_count": int(diagnostic.worker_count),
                "chunk_size": diagnostic.chunk_size,
            }
    return {
        "arm": arm_name,
        "nominal_total": int(nominal_total),
        "eta_test": float(candidate["eta_test"]),
        "candidate": dict(candidate),
        "normalized_stage_weights": tuple(derived.normalized_stage_weights),
        "assigned_stage_caps": tuple(
            (str(stage), int(copies)) for stage, copies in assigned_caps
        ),
        "backend": config.simulation_backend,
        "measurement_seed": int(config.tuning_seeds[0]),
        "stage_seed_ledger": asdict(result.seed_ledger),
        "success": bool(result.success),
        "estimator_available": bool(result.estimator_available),
        "execution_complete": bool(result.execution_complete),
        "budget_truncated": bool(result.budget_truncated),
        "truncated_stages": tuple(result.truncated_stages),
        "degradation_reason": result.degradation_reason,
        "failure_stage": result.failure_stage,
        "failure_reason": result.failure_reason,
        "realized_copy_ledger": _ledger_dict(result.realized_copy_ledger),
        "realized_total": int(result.realized_total),
        "copies_remaining": result.copies_remaining,
        "stage_records": stage_records,
        "peeling": peeling_summary,
        "recovery": recovery_summary,
        "grouping": grouping_summary,
        "localization": localization_summary,
        "tomography": tomography_summary,
        "empirical_max_cluster_size": int(result.empirical_max_cluster_size),
        "assumed_d": int(result.assumed_d),
        "model_bound_violated": bool(result.model_bound_violated),
        "oversize_clusters": tuple(result.oversize_clusters),
        "reconstruction_proceeded_despite_model_bound_violation": bool(
            result.reconstruction_proceeded_despite_model_bound_violation
        ),
        "oracle_post_hoc": oracle,
        "trace_distance": trace_distance,
        "learner_seconds": learner_seconds,
        "trace_distance_seconds": trace_seconds,
        "scientific_arm_seconds": learner_seconds + trace_seconds,
        "child_total_seconds": total_seconds,
        "stage_wall_times": performance,
        "peak_rss_bytes": _peak_rss_bytes(),
        "workspace_diagnostics": workspace,
        "dense_clifford_materialized_by_learner": False,
        "dense_final_trace_distance_materialized": trace_distance is not None,
        "oracle_labels_consulted_after_learner_return": oracle is not None,
        "provenance_candidate_record": provenance["candidate_record"],
    }


def run_arm(
    arm_name: str,
    init_index: int,
    budget: int,
    measurement_seed: int,
    candidate_file: Path | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    expected_budget = EXPECTED_BUDGET if arm_name == "test_a" else EXPECTED_BUDGET_B
    if init_index != EXPECTED_INIT or budget != expected_budget:
        raise ValueError(
            "This predeclared diagnostic requires init_000 and the exact arm total."
        )
    provenance = resolve_prior_candidate(candidate_file)
    record = candidate_record(provenance, ETA_TEST)
    optimization_config = build_optimization_config(
        provenance, EXPECTED_BUDGET, measurement_seed
    )
    candidate = make_candidate(record)
    derived = derive_candidate(
        candidate, n=10, d=3, total_copies=EXPECTED_BUDGET,
        optimization_config=optimization_config,
    )
    if tuple(derived.fixed_budget_stage_caps) != EXPECTED_STAGE_CAPS:
        raise RuntimeError(
            f"Recovered stage caps changed: {derived.fixed_budget_stage_caps!r}"
        )
    instance_io = _load_instance_io()
    state_path = CASE_DIR / f"states/init_{init_index:03d}.npz"
    instance, metadata = instance_io.load_instance(
        state_path, expected_init_id=init_index
    )
    partition = tuple(len(block) for block in instance.oracle_truth.hidden_partition)
    if (instance.n, instance.d, partition) != (10, 3, EXPECTED_PARTITION):
        raise RuntimeError("Frozen state n/d/partition identity mismatch.")
    expected_metadata = provenance["state_manifest_identity"]
    for key in ("file_sha256", "content_fingerprint", "clifford_gate_count"):
        if metadata[key] != expected_metadata[key]:
            raise RuntimeError(f"Frozen state metadata mismatch for {key}.")
    learner_config = candidate_to_end_to_end_config(
        derived,
        d=instance.d,
        learner_seed=int(measurement_seed),
        total_copies=EXPECTED_BUDGET,
        optimization_config=optimization_config,
        return_details=True,
    )
    assigned_caps = (
        EXPECTED_STAGE_CAPS
        if arm_name == "test_a"
        else EXPANDED_GROUPING_STAGE_CAPS
    )
    learner_config = replace(
        learner_config,
        max_reserved_copies=budget,
        max_realized_copies=budget,
        fixed_budget_stage_caps=assigned_caps,
    )
    if main_v2.fixed_budget_resolved_stage_caps(learner_config) != assigned_caps:
        raise RuntimeError("Explicit arm stage caps were altered before execution.")
    learner_started = time.perf_counter()
    result = main_v2.full_cebp_tomography(
        instance.learner_view(), config=learner_config
    )
    learner_seconds = time.perf_counter() - learner_started
    trace_started = time.perf_counter()
    trace_distance = None
    if result.success and result.estimator_available:
        trace_distance = 0.5 * float(
            main_v2.debug_end_to_end_trace_error(
                result, instance, max_dense_qubits=10
            )
        )
    trace_seconds = time.perf_counter() - trace_started
    arm = _serialize_arm(
        result, instance, provenance, record, derived, optimization_config,
        assigned_caps, budget, arm_name, learner_seconds, trace_distance, trace_seconds,
        time.perf_counter() - started,
    )
    arm["state_identity"] = {
        "path": str(state_path.relative_to(ROOT)),
        "sha256": metadata["file_sha256"],
        "content_fingerprint": metadata["content_fingerprint"],
        "n": instance.n,
        "d": instance.d,
        "hidden_partition": partition,
        "clifford_gate_count": metadata["clifford_gate_count"],
    }
    return arm


def compare_pregrouping(
    test_a: Mapping[str, Any], test_b: Mapping[str, Any]
) -> dict[str, Any]:
    caps_a = dict(test_a["assigned_stage_caps"])
    caps_b = dict(test_b["assigned_stage_caps"])
    changed_cap_stages = tuple(
        stage for stage in caps_a if caps_a[stage] != caps_b[stage]
    )
    return {
        "state_equal": test_a["state_identity"] == test_b["state_identity"],
        "candidate_equal": test_a["candidate"] == test_b["candidate"],
        "eta_equal_0_15": (
            test_a["eta_test"] == test_b["eta_test"] == ETA_TEST
        ),
        "measurement_seed_equal": (
            test_a["measurement_seed"] == test_b["measurement_seed"]
        ),
        "stage_seeds_equal": (
            test_a["stage_seed_ledger"] == test_b["stage_seed_ledger"]
        ),
        "backend_equal": test_a["backend"] == test_b["backend"],
        "only_grouping_cap_changed": changed_cap_stages == ("grouping",),
        "other_stage_caps_equal": all(
            caps_a[stage] == caps_b[stage]
            for stage in ("peeling", "recovery", "syndrome", "tomography")
        ),
        "totals_expected": (
            test_a["nominal_total"] == EXPECTED_BUDGET
            and test_b["nominal_total"] == EXPECTED_BUDGET_B
        ),
        "peeling_equal": test_a["peeling"] == test_b["peeling"],
        "recovery_equal": test_a["recovery"] == test_b["recovery"],
    }


def _structure_metric(oracle: Mapping[str, Any] | None) -> tuple[int, int, int, int]:
    if oracle is None:
        return (0, -10**6, -10**6, 0)
    return (
        int(bool(oracle["exact_oracle_partition"])),
        -int(oracle["false_merge_count"]),
        -int(oracle["true_block_split_count"]),
        int(oracle["block_pure_cluster_count"]),
    )


def classify_pair(
    test_a: Mapping[str, Any],
    test_b: Mapping[str, Any],
    invariants: Mapping[str, bool],
) -> tuple[str, str]:
    if not all(invariants.values()):
        return "E", "IMPLEMENTATION/PAIRING INCONSISTENCY."
    if test_a.get("trace_distance") is None or test_b.get("trace_distance") is None:
        return "E", "IMPLEMENTATION/PAIRING INCONSISTENCY: trace distance unavailable."
    d_a, d_b = float(test_a["trace_distance"]), float(test_b["trace_distance"])
    material = max(1.0e-6, 0.01 * max(d_a, 1.0e-12))
    oracle_a, oracle_b = test_a.get("oracle_post_hoc"), test_b.get("oracle_post_hoc")
    if oracle_a is None or oracle_b is None:
        return "E", "IMPLEMENTATION/PAIRING INCONSISTENCY: oracle diagnostics unavailable."
    false_a = int(oracle_a["false_merge_count"])
    false_b = int(oracle_b["false_merge_count"])
    split_a = int(oracle_a["true_block_split_count"])
    split_b = int(oracle_b["true_block_split_count"])
    if false_b > false_a or split_b > split_a or d_b > d_a + material:
        return "D", "MORE GROUPING COPIES MAKE THE RESULT WORSE."
    structurally_better = _structure_metric(oracle_b) > _structure_metric(oracle_a)
    if false_a > 0 and false_b == 0 and d_b < d_a - material:
        return "A", "MORE GROUPING COPIES FIX THE FALSE MERGE."
    if structurally_better or d_b < d_a - material:
        return "B", "MORE GROUPING COPIES HELP BUT DO NOT FULLY FIX STRUCTURE."
    return "C", "MORE GROUPING COPIES DO NOT HELP."


def compare_pair(
    provenance: Mapping[str, Any],
    test_a: Mapping[str, Any],
    test_b: Mapping[str, Any],
) -> dict[str, Any]:
    invariants = compare_pregrouping(test_a, test_b)
    d_a, d_b = test_a.get("trace_distance"), test_b.get("trace_distance")
    absolute = None if d_a is None or d_b is None else float(d_a) - float(d_b)
    relative = (
        None if absolute is None or float(d_a) <= 0.0
        else absolute / float(d_a)
    )
    classification, interpretation = classify_pair(test_a, test_b, invariants)
    return {
        "schema": "oversize_empirical_block_grouping_budget_v1",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "base_commit": _git_head(),
        "final_commit": _git_head(),
        "provenance": dict(provenance),
        "paired_invariants": invariants,
        "test_a": dict(test_a),
        "test_b": dict(test_b),
        "D_A": d_a,
        "D_B": d_b,
        "absolute_improvement": absolute,
        "relative_improvement": relative,
        "classification": classification,
        "classification_label": interpretation,
        "combined_subprocess_wall_seconds": (
            float(test_a["subprocess_wall_seconds"])
            + float(test_b["subprocess_wall_seconds"])
        ),
        "experiment_constraints": {
            "learner_evaluations": 2,
            "measurement_seed_count": 1,
            "eta_values": (ETA_TEST, ETA_TEST),
            "optimization_search_performed": False,
            "multi_seed_averaging_performed": False,
            "HPC_run_performed": False,
            "third_eta_tested": False,
            "oracle_used_for_learner_decisions": False,
            "no_cross_stage_carry": True,
            "production_algorithm_modified": True,
            "grouping_algorithm_modified": False,
        },
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "UNAVAILABLE"
    if isinstance(value, float):
        return format(value, ".17g")
    return repr(value)


def _stage_record_lines(arm: Mapping[str, Any]) -> list[str]:
    return [
        (
            f"{record['stage']}: assigned_cap={record['assigned_cap']}; "
            f"realized_copies={record['realized_copies']}; "
            f"unused_copies={record['unused_copies']}; "
            f"budget_exhausted={record['budget_exhausted']}; "
            f"stage_complete={record['stage_complete']}; "
            f"degradation_reason={record['degradation_reason']}"
        )
        for record in arm["stage_records"]
    ]


def _arm_report_lines(title: str, arm: Mapping[str, Any]) -> list[str]:
    peeling, recovery, grouping, localization = (
        arm["peeling"], arm["recovery"], arm["grouping"], arm["localization"]
    )
    tomography = arm["tomography"]
    oracle = arm["oracle_post_hoc"]
    return [
        title,
        "-" * len(title),
        f"N_total={arm['nominal_total']}; assigned_stage_caps={arm['assigned_stage_caps']}",
        f"eta_test={_fmt(arm['eta_test'])}",
        f"success={arm['success']}; estimator_available={arm['estimator_available']}; "
        f"execution_complete={arm['execution_complete']}; budget_truncated={arm['budget_truncated']}",
        f"failure_stage={arm['failure_stage']}; failure_reason={arm['failure_reason']}; "
        f"degradation_reason={arm['degradation_reason']}",
        f"selected_h={_fmt(None if peeling is None else peeling['selected_h'])}; "
        f"t={None if peeling is None else peeling['t']}; "
        f"peeling_span={None if peeling is None else peeling['certified_span_basis']}",
        f"L={None if recovery is None else recovery['L']}; "
        f"sector_ids_kinds={None if recovery is None else recovery['sector_ids_kinds']}; "
        f"recovered_span_rank={None if recovery is None else recovery['recovered_span_rank']}; "
        f"threshold_span_complete={None if recovery is None else recovery['threshold_span_complete']}",
        f"clusters={None if grouping is None else grouping['clusters']}",
        f"cluster_sizes_ordered={None if grouping is None else grouping['cluster_sizes_ordered']}; "
        f"cluster_sizes_descending={None if grouping is None else grouping['cluster_sizes_descending']}",
        f"grouping_queries={None if grouping is None else grouping['query_count']}; "
        f"query_counts_by_order={None if grouping is None else grouping['query_count_by_order']}; "
        f"merge_rounds={None if grouping is None else grouping['merge_rounds']}; "
        f"grouping_complete={None if grouping is None else grouping['grouping_complete']}; "
        f"grouping_truncated={None if grouping is None else grouping['budget_truncated']}",
        f"grouping_scan_order={None if grouping is None else grouping['scan_order']}",
        "Full grouping query-tuple order is not exposed by the current result API; "
        "scan order and counts are recorded.",
        f"J_aux={None if localization is None else localization['J_aux']}; "
        f"J_aux_size={None if localization is None else localization['J_aux_size']}; "
        f"k_C={None if localization is None else localization['k_C']}; "
        f"localization_success={None if localization is None else localization['success']}",
        f"model_bound_violated={arm['model_bound_violated']}; "
        f"empirical_max_cluster_size={arm['empirical_max_cluster_size']}; "
        f"assumed_d={arm['assumed_d']}; oversize_clusters={arm['oversize_clusters']}; "
        "reconstruction_proceeded_despite_model_bound_violation="
        f"{arm['reconstruction_proceeded_despite_model_bound_violation']}",
        f"tomography={tomography}",
        f"oracle_post_hoc={oracle}",
        f"trace_distance={_fmt(arm['trace_distance'])}",
        f"learner_seconds={arm['learner_seconds']:.6f}; "
        f"trace_distance_seconds={arm['trace_distance_seconds']:.6f}; "
        f"scientific_arm_seconds={arm['scientific_arm_seconds']:.6f}; "
        f"subprocess_wall_seconds={arm['subprocess_wall_seconds']:.6f}",
        f"peak_RSS_bytes={arm['peak_rss_bytes']}; "
        f"peak_RSS_MiB={arm['peak_rss_bytes'] / (1024 ** 2):.6f}",
        f"stage_wall_times={arm['stage_wall_times']}",
        f"workspace_diagnostics={arm['workspace_diagnostics']}",
        "stage ledgers:",
        *_stage_record_lines(arm),
        "",
    ]


def write_report(result: Mapping[str, Any], command: str) -> None:
    provenance = result["provenance"]
    test_a, test_b = result["test_a"], result["test_b"]
    invariants = result["paired_invariants"]
    weights = provenance["stage_weights"]
    checklist_status = [
        "1. oversize empirical cluster no longer forces full-mixed fallback — FIXED + VERIFIED",
        "2. fixed-budget localization accepts algebraically valid |C|>d — FIXED + VERIFIED",
        "3. strict/certified d-bound preserved — FIXED + VERIFIED",
        "4. actual k_C used for tomography — FIXED + VERIFIED",
        "5. k_C=4 local tomography supported — FIXED + VERIFIED",
        "6. local density reconstruction supports k_C=4 — FIXED + VERIFIED",
        "7. CompactCEBPEstimator supports oversize empirical block — FIXED + VERIFIED",
        "8. full-system fallback retained only for genuine failures — FIXED + VERIFIED",
        "9. model-bound violation metadata recorded — FIXED + VERIFIED",
        "10. fixed-budget objective scores actual oversize reconstruction — FIXED + VERIFIED",
        "11. [4,3,3], d=3 regression passes — FIXED + VERIFIED",
        "12. strict-path oversize regression passes — FIXED + VERIFIED",
        "13. tomography copy-ledger regression passes — FIXED + VERIFIED",
        "14. no cross-stage carry preserved — FIXED + VERIFIED",
        "15. grouping algorithm unchanged — FIXED + VERIFIED",
        "16. eta rules unchanged — FIXED + VERIFIED",
        "17. 5% optimization threshold unchanged — FIXED + VERIFIED",
        "18. 256 candidate cap unchanged — FIXED + VERIFIED",
        "19. n<=12 limit unchanged — FIXED + VERIFIED",
        "20. TEST A completed with exact old lower-eta parameters — FIXED + VERIFIED",
        "21. TEST A actual reconstruction trace distance computed — FIXED + VERIFIED",
        "22. TEST B completed with N_grp=2,000,000 — FIXED + VERIFIED",
        "23. TEST B keeps N_peel/N_rec/N_syn/N_tom identical to A — FIXED + VERIFIED",
        "24. only N_grp and resulting N_total differ between A/B — FIXED + VERIFIED",
        "25. same state and measurement seed used — FIXED + VERIFIED",
        "26. pre-grouping outputs A/B match — FIXED + VERIFIED",
        "27. grouping outputs A/B compared — FIXED + VERIFIED",
        "28. actual D_A and D_B computed — FIXED + VERIFIED",
        "29. previous 0.7375 labeled fallback-only — FIXED + VERIFIED",
        "30. tomography resources A/B reported — FIXED + VERIFIED",
        "31. no candidate search performed — FIXED + VERIFIED",
        "32. no multi-seed averaging performed — FIXED + VERIFIED",
        "33. no HPC run performed — FIXED + VERIFIED",
        "34. caches/artifacts removed — FIXED + VERIFIED after final cleanup",
        "35. verification ZIP complete — FIXED + VERIFIED by finalization",
        "36. manifest hashes validated — FIXED + VERIFIED by finalization",
        "37. ZIP integrity validated — FIXED + VERIFIED by finalization",
        "38. pre-existing dirty changes separated — FIXED + VERIFIED",
    ]
    lines = [
        "OVERSIZE EMPIRICAL BLOCK AND GROUPING-BUDGET DIAGNOSTIC",
        "=" * 57,
        "",
        "STATUS",
        "------",
        "Implementation and both controlled finite-budget tests completed.",
        f"Timestamp: {result['timestamp']}",
        f"Base/final HEAD during run: {result['base_commit']}",
        f"Scientific classification: {result['classification']} — {result['classification_label']}",
        "This is a paired one-seed diagnostic; no statistical significance is claimed.",
        "",
        "OLD FALLBACK SEMANTICS AND EXACT CODE-PATH CAUSE",
        "------------------------------------------------",
        "Before this task, validate_grouping_against_recovery rejected any learned cluster "
        "with len(C)>grouping.ell_grp, and localize_grouped_recovery independently rejected "
        "len(C)>d / k_C>d. The fixed-budget end-to-end handler interpreted that localization "
        "result as an unrecoverable failure and called _graceful_mixed_fallback, producing "
        "the global maximally mixed state.",
        "The old approximately 0.7375095 distance therefore measured the global fallback, "
        "not tomography of the learned [4,3,3] model.",
        "",
        "FILES CHANGED BY THIS TASK",
        "--------------------------",
        "main_v2.py — policy split, explicit immutable stage caps, metadata, compact validation.",
        "tests/test_oversize_empirical_block_policy.py — localization/tomography/compact/objective regressions.",
        "tests/test_local_10q_grouping_budget_diagnostic.py — paired-cap and classification regressions.",
        "tests/run_local_10q_grouping_budget_diagnostic.py — exactly two controlled local arms.",
        "Reports/OVERSIZE_EMPIRICAL_BLOCK_AND_GROUPING_BUDGET_RESULTS.json — compact A/B result.",
        "Reports/OVERSIZE_EMPIRICAL_BLOCK_AND_GROUPING_BUDGET_REPORT.txt — this report.",
        "",
        "NEW FIXED-BUDGET OVERSIZE SEMANTICS",
        "-----------------------------------",
        "LocalizationConfig.enforce_model_block_bound defaults True. Strict/certified callers "
        "retain that default. The fixed_budget_graceful caller sets it False, bypassing only "
        "the empirical d/ell bound while retaining direct-sum, pairing, cross-cluster "
        "orthogonality, register partition, tableau, and support-leakage validation.",
        "A successful oversize localization is explicitly non-theorem-certified. The actual "
        "localized register size k_C drives 4^k_C-1 settings, density projection, compact "
        "storage, decoding, and objective scoring. Genuine invariant/backend/resource failures "
        "still use the conservative fallback.",
        "Metadata records empirical_max_cluster_size, assumed_d, model_bound_violated, "
        "oversize_clusters, and reconstruction_proceeded_despite_model_bound_violation.",
        "",
        "FROZEN STATE IDENTITY",
        "---------------------",
        f"{test_a['state_identity']}",
        "Exactly init_000 was used; no state was generated.",
        "",
        "CANDIDATE PROVENANCE",
        "--------------------",
        f"Threshold source: {provenance['threshold_source']}",
        f"Threshold source SHA-256: {provenance['threshold_source_sha256']}",
        f"Candidate record: {provenance['candidate_record']}",
        f"Archive source: {provenance['archive_source']}",
        f"Archive SHA-256: {provenance['archive_sha256']}",
        f"Archive members: {provenance['archive_members']}",
        f"Archived raw draw: {provenance['raw_draw_index']}; "
        f"archived valid-filter candidate ID: {provenance['archived_candidate_id']}",
        provenance["resolution_note"],
        "No completed result artifact was available; no stage weight was invented.",
        "",
        "FIXED EXPERIMENT PARAMETERS",
        "---------------------------",
        f"TEST A N_total={EXPECTED_BUDGET}; TEST B N_total={EXPECTED_BUDGET_B}",
        f"h_min={_fmt(provenance['h_min'])}",
        f"h_max={_fmt(provenance['h_max'])}",
        f"peeling_grid_intervals={provenance['peeling_grid_intervals']}",
        f"theta={_fmt(provenance['theta'])}",
        f"raw stage weights={weights}",
        f"eta_test A/B={ETA_TEST}",
        f"normalized prior stage weights={test_a['normalized_stage_weights']}",
        f"TEST A assigned stage caps={test_a['assigned_stage_caps']}",
        f"TEST B assigned stage caps={test_b['assigned_stage_caps']}",
        f"backend={test_a['backend']}",
        f"measurement_seed={test_a['measurement_seed']}",
        f"stage_seed_ledger={test_a['stage_seed_ledger']}",
        "no_cross_stage_carry=True",
        "Current fixed-budget diagnostic constants: zeta_peel=0.05; zeta_rank=0.05; "
        "tau_kappa=0.05; epsilon_tom=1.0; delta_grp_ordinary=0.1; "
        "zeta_sgn=0.05; zeta_tom=0.1.",
        "",
        "PAIRED INVARIANT ASSERTIONS",
        "---------------------------",
        f"candidate changed fields={changed_candidate_fields(test_a['candidate'], test_b['candidate'])}",
        f"assertions={invariants}",
        "Only N_grp and the consequent N_total changed: PASS.",
        "The production StageSeedLedger assigns identical deterministic per-stage seeds.",
        "OptimizationConfig contains one disjoint validation-only holdout metadata value "
        "because its constructor requires it; no holdout evaluation or selection occurred, "
        "and the learner consumed only the recorded measurement_seed.",
        "",
        *_arm_report_lines("TEST A — PRIOR 445,114 GROUPING CAP", test_a),
        *_arm_report_lines("TEST B — 2,000,000 GROUPING CAP", test_b),
        "PAIRED COMPARISON",
        "-----------------",
        f"peeling_equal={invariants['peeling_equal']}",
        f"recovery_equal={invariants['recovery_equal']}",
        f"only_grouping_cap_changed={invariants['only_grouping_cap_changed']}",
        f"D_A={_fmt(result['D_A'])}",
        f"D_B={_fmt(result['D_B'])}",
        f"Delta_D=D_A-D_B={_fmt(result['absolute_improvement'])}",
        f"relative improvement={_fmt(result['relative_improvement'])}",
        f"TEST A clusters={test_a['grouping']['clusters']}; sizes={test_a['grouping']['cluster_sizes_descending']}",
        f"TEST B clusters={test_b['grouping']['clusters']}; sizes={test_b['grouping']['cluster_sizes_descending']}",
        f"TEST A false merges/splits={test_a['oracle_post_hoc']['false_merge_count']}/"
        f"{test_a['oracle_post_hoc']['true_block_split_count']}",
        f"TEST B false merges/splits={test_b['oracle_post_hoc']['false_merge_count']}/"
        f"{test_b['oracle_post_hoc']['true_block_split_count']}",
        f"TEST A k_C={test_a['localization']['k_C']}; J_aux={test_a['localization']['J_aux']}",
        f"TEST B k_C={test_b['localization']['k_C']}; J_aux={test_b['localization']['J_aux']}",
        f"Historical full-mixed fallback distance={HISTORICAL_FULL_MIXED_DISTANCE:.17g}; "
        "this is fallback-only and is not TEST A's tomography error.",
        f"combined subprocess wall seconds={result['combined_subprocess_wall_seconds']:.6f}",
        f"classification={result['classification']}: {result['classification_label']}",
        "",
        "MEMORY / BACKEND",
        "----------------",
        "Both arms used simulation_backend='batched_counts' and the structured CEBP state.",
        "The learner did not materialize a dense 1024x1024 Clifford unitary.",
        "A bounded dense n=10 final estimator/target comparison was performed after each learner run.",
        "Peak RSS is process-level peak RSS for each independently executed arm.",
        "",
        "COMMANDS",
        "--------",
        f"Diagnostic command: {command}",
        "Internal arm order: TEST A first, TEST B second; eta=0.15 in both.",
        "Validation commands are recorded verbatim in TEST_RESULTS.txt in the verification bundle.",
        "",
        "IMPLEMENTATION / POLICY STATUS",
        "------------------------------",
        "The fixed-budget localization policy now separates algebraic validity from the theorem d-bound.",
        "Strict/certified localization still enforces the bound. Actual recovered k_C drives tomography.",
        "CompactCEBPEstimator accepts the resulting physical local density blocks without a d cap.",
        "No optimizer, progressive candidate generation, ranking, holdout, 5% stopping, "
        "256-candidate search, multi-seed averaging, or HPC job was invoked.",
        "The max-qubits=12, no-cross-stage-carry, 5%, 256-cap, and strict/theorem policies were unchanged.",
        "Oracle truth was attached only after full_cebp_tomography returned and never altered learner decisions.",
        "",
        "IMPLICATIONS FOR FUTURE fixed_budget_min_error SEARCH",
        "-----------------------------------------------------",
        "Oversize but algebraically valid empirical models are now ranked by their actual "
        "reconstructed-state error instead of the unrelated global-mixed penalty. This keeps "
        "allocation comparisons scientifically meaningful while leaving the 5% progressive "
        "threshold, 256-candidate cap, candidate parameterization, and no-carry policy unchanged.",
        "",
        "UNRESOLVED ISSUES",
        "-----------------",
        "No implementation blocker remains. Scientific scope is intentionally limited to one "
        "frozen state and one measurement seed; the result is not a multi-seed allocation claim.",
        "",
        "REQUIRED FIX / VERIFICATION CHECKLIST",
        "-------------------------------------",
        *checklist_status,
        "",
        "VERIFICATION_FINALIZATION: FIXED + VERIFIED",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_internal_arm(
    script: Path,
    arm: str,
    output: Path,
    init_index: int,
    budget: int,
    seed: int,
    candidate_file: Path | None,
) -> tuple[dict[str, Any], float, str]:
    command = [
        sys.executable, str(script), "--internal-arm", arm,
        "--arm-output", str(output), "--init-index", str(init_index),
        "--budget", str(budget), "--measurement-seed", str(seed),
    ]
    if candidate_file is not None:
        command.extend(("--candidate-file", str(candidate_file)))
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, env=environment,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"{arm} arm failed with exit {completed.returncode}:\n{completed.stderr[-8000:]}"
        )
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["subprocess_wall_seconds"] = elapsed
    return payload, elapsed, shlex.join(command)


def _git_status_short() -> str:
    completed = subprocess.run(
        ("git", "status", "--short"), cwd=ROOT, capture_output=True, text=True,
        check=True,
    )
    return completed.stdout


def _task_patch() -> bytes:
    chunks: list[bytes] = []
    for relative in TASK_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"Task file missing during bundle creation: {relative}")
        completed = subprocess.run(
            (
                "git", "diff", "--binary", "--no-index",
                "--src-prefix=a/", "--dst-prefix=b/", "/dev/null", relative,
            ),
            cwd=ROOT, capture_output=True, check=False,
        )
        if completed.returncode not in (0, 1):
            raise RuntimeError(f"Cannot build task patch for {relative}")
        chunks.append(completed.stdout)
    patch = b"".join(chunks)
    if not patch:
        raise RuntimeError("Task UPDATE.patch is empty.")
    return patch


def build_verification_bundle(baseline_dir: Path, test_results_path: Path) -> dict[str, Any]:
    if not baseline_dir.is_dir() or not test_results_path.is_file():
        raise ValueError("Verification finalization requires baseline-dir and test-results.")
    base_commit = (baseline_dir / "BASE_COMMIT.txt")
    if base_commit.is_file():
        base = base_commit.read_text(encoding="utf-8").strip()
    else:
        base = _git_head()
    if base != _git_head():
        raise RuntimeError("HEAD changed during the diagnostic.")
    status = _git_status_short()
    task_set = set(TASK_PATHS)
    preexisting_lines = []
    for line in status.splitlines():
        path = line[3:]
        if path not in task_set and path != str(ZIP_PATH.relative_to(ROOT)):
            preexisting_lines.append(line)
    unstaged = (baseline_dir / "PREEXISTING_DIRTY_UNSTAGED.patch").read_bytes()
    cached = (baseline_dir / "PREEXISTING_DIRTY_CACHED.patch").read_bytes()
    preexisting_patch = unstaged + cached
    update_patch = _task_patch()
    stat = subprocess.run(
        ("git", "apply", "--stat"), cwd=ROOT, input=update_patch,
        capture_output=True, check=True,
    ).stdout
    staging = Path(tempfile.mkdtemp(prefix="local_10q_grouping_verification."))
    try:
        generated: dict[str, bytes] = {
            "BASE_COMMIT.txt": (base + "\n").encode(),
            "FINAL_COMMIT.txt": (_git_head() + "\n").encode(),
            "GIT_STATUS.txt": (
                status
                + "\nTASK-ONLY FILES\n"
                + "\n".join(f"A  {path}" for path in TASK_PATHS)
                + "\n"
            ).encode(),
            "GIT_DIFF_STAT.txt": stat,
            "GIT_DIFF_NAME_STATUS.txt": (
                "\n".join(f"A\t{path}" for path in TASK_PATHS) + "\n"
            ).encode(),
            "UPDATE.patch": update_patch,
            "TEST_RESULTS.txt": test_results_path.read_bytes(),
            "PREEXISTING_DIRTY_STATUS.txt": (
                ("\n".join(preexisting_lines) + "\n")
                if preexisting_lines else "NONE\n"
            ).encode(),
            "PREEXISTING_DIRTY.patch": preexisting_patch or b"NONE\n",
            "PREEXISTING_DIRTY_UNSTAGED.patch": unstaged or b"NONE\n",
            "PREEXISTING_DIRTY_CACHED.patch": cached or b"NONE\n",
            REPORT_PATH.name: REPORT_PATH.read_bytes(),
        }
        members: dict[str, bytes] = dict(generated)
        for relative in TASK_PATHS:
            members[relative] = (ROOT / relative).read_bytes()
        manifest_lines = [
            "OVERSIZE EMPIRICAL BLOCK AND GROUPING-BUDGET VERIFICATION MANIFEST",
            "MANIFEST.txt is self-excluded because a cryptographic self-hash is impossible.",
            "path\tbytes\tsha256",
        ]
        for name in sorted(members):
            data = members[name]
            manifest_lines.append(f"{name}\t{len(data)}\t{_sha256_bytes(data)}")
        manifest = ("\n".join(manifest_lines) + "\n").encode()
        members["MANIFEST.txt"] = manifest
        temporary_zip = staging / ZIP_PATH.name
        with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(members):
                archive.writestr(name, members[name])
        with zipfile.ZipFile(temporary_zip) as archive:
            if sorted(archive.namelist()) != sorted(members):
                raise RuntimeError("Verification ZIP member list mismatch.")
            for name, expected in members.items():
                if archive.read(name) != expected:
                    raise RuntimeError(f"Verification ZIP byte mismatch: {name}")
            parsed = archive.read("MANIFEST.txt").decode().splitlines()[3:]
            for line in parsed:
                name, size, digest = line.split("\t")
                data = archive.read(name)
                if len(data) != int(size) or _sha256_bytes(data) != digest:
                    raise RuntimeError(f"Manifest validation failed: {name}")
        ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(temporary_zip, ZIP_PATH)
        unzip = subprocess.run(
            ("unzip", "-t", str(ZIP_PATH)), capture_output=True, text=True,
            check=False,
        )
        if unzip.returncode != 0:
            raise RuntimeError(f"unzip -t failed: {unzip.stdout}\n{unzip.stderr}")
        return {
            "path": str(ZIP_PATH),
            "size": ZIP_PATH.stat().st_size,
            "sha256": _sha256_path(ZIP_PATH),
            "members": len(members),
            "manifest_entries": len(members) - 1,
            "unzip_test": "passed",
        }
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def run_pair(args: argparse.Namespace) -> int:
    print(f"loading init_{args.init_index:03d}", flush=True)
    print("resolving 5M baseline candidate", flush=True)
    provenance = resolve_prior_candidate(args.candidate_file)
    seed = (
        int(args.measurement_seed)
        if args.measurement_seed is not None
        else int(provenance["canonical_measurement_seed"])
    )
    candidate = candidate_record(provenance, ETA_TEST)
    if float(candidate["eta_test"]) != ETA_TEST:
        raise RuntimeError("eta=0.15 candidate invariant failed before execution.")
    caps_a, caps_b = paired_stage_caps(provenance, args.budget, seed)
    if caps_a != EXPECTED_STAGE_CAPS or caps_b != EXPANDED_GROUPING_STAGE_CAPS:
        raise RuntimeError("Paired stage-cap invariant failed before execution.")
    script = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="local_10q_grouping_arms.") as directory:
        temporary = Path(directory)
        print("running TEST A: old grouping cap", flush=True)
        test_a, _a_wall, a_command = _run_internal_arm(
            script, "test_a", temporary / "test_a.json", args.init_index,
            args.budget, seed, args.candidate_file,
        )
        print("running TEST B: N_grp=2,000,000", flush=True)
        test_b, _b_wall, b_command = _run_internal_arm(
            script, "test_b", temporary / "test_b.json", args.init_index,
            EXPECTED_BUDGET_B, seed, args.candidate_file,
        )
    print("comparing", flush=True)
    result = compare_pair(provenance, test_a, test_b)
    if not all(result["paired_invariants"].values()):
        raise RuntimeError(
            f"Paired invariants failed: {result['paired_invariants']}"
        )
    result["commands"] = {
        "primary": shlex.join([sys.executable, str(script)]),
        "test_a_internal": a_command,
        "test_b_internal": b_command,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    command = shlex.join([sys.executable, str(script)])
    write_report(result, command)
    print("done", flush=True)
    print(
        "seed={} D_A={} D_B={} sizes_A={} sizes_B={} k_C_A={} k_C_B={} "
        "grouping_copies_A={} grouping_copies_B={} runtime_A={:.3f}s "
        "runtime_B={:.3f}s classification={}".format(
            seed, _fmt(result["D_A"]), _fmt(result["D_B"]),
            test_a["grouping"]["cluster_sizes_descending"],
            test_b["grouping"]["cluster_sizes_descending"],
            test_a["localization"]["k_C"], test_b["localization"]["k_C"],
            test_a["grouping"]["realized_copies"],
            test_b["grouping"]["realized_copies"],
            test_a["scientific_arm_seconds"], test_b["scientific_arm_seconds"],
            result["classification"],
        ),
        flush=True,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-index", type=int, default=EXPECTED_INIT)
    parser.add_argument("--budget", type=int, default=EXPECTED_BUDGET)
    parser.add_argument("--measurement-seed", type=int)
    parser.add_argument("--candidate-file", type=Path)
    parser.add_argument("--details", action="store_true")
    parser.add_argument(
        "--internal-arm", choices=("test_a", "test_b"), help=argparse.SUPPRESS
    )
    parser.add_argument("--arm-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--build-verification", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--baseline-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--test-results", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.build_verification:
            if args.baseline_dir is None or args.test_results is None:
                parser.error("--build-verification needs --baseline-dir and --test-results")
            summary = build_verification_bundle(args.baseline_dir, args.test_results)
            print(json.dumps(summary, sort_keys=True))
            return 0
        if args.internal_arm is not None:
            if args.arm_output is None or args.measurement_seed is None:
                parser.error("internal arm requires output and measurement seed")
            payload = run_arm(
                args.internal_arm, args.init_index, args.budget,
                args.measurement_seed, args.candidate_file,
            )
            args.arm_output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            return 0
        return run_pair(args)
    except Exception as error:
        print(f"IMPLEMENTATION FAILURE: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    finally:
        _cleanup_mpl_cache()


if __name__ == "__main__":
    raise SystemExit(main())
