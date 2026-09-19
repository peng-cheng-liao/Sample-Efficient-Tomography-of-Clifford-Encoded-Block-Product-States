"""Private concise JSON checkpoint support for long optimization searches."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Tuple

from .objective import SeedEvaluation
from .parameterization import PreflightResourceCheck


SCHEMA_VERSION = 7


def _seed_evaluation_to_json(evaluation: SeedEvaluation) -> Dict[str, Any]:
    value = asdict(evaluation)
    value["realized_copy_ledger"] = [list(item) for item in evaluation.realized_copy_ledger]
    value["cluster_sizes"] = list(evaluation.cluster_sizes)
    value["register_sizes"] = list(evaluation.register_sizes)
    value["truncated_stages"] = list(evaluation.truncated_stages)
    value["execution_resource_policy"] = [
        list(item) for item in evaluation.execution_resource_policy
    ]
    value["fixed_budget_stage_records"] = [
        list(item) for item in evaluation.fixed_budget_stage_records
    ]
    value["grouping_diagnostics"] = [
        list(item) for item in evaluation.grouping_diagnostics
    ]
    return value


def _seed_evaluation_from_json(value: Mapping[str, Any]) -> SeedEvaluation:
    data = dict(value)
    data["preflight_resource_check"] = PreflightResourceCheck(
        **dict(data["preflight_resource_check"])
    )
    data["realized_copy_ledger"] = tuple(
        (str(name), int(count)) for name, count in data["realized_copy_ledger"]
    )
    data["cluster_sizes"] = tuple(int(item) for item in data["cluster_sizes"])
    data["register_sizes"] = tuple(int(item) for item in data["register_sizes"])
    data["truncated_stages"] = tuple(str(item) for item in data.get("truncated_stages", ()))
    data["execution_resource_policy"] = tuple(
        (str(name), str(value))
        for name, value in data.get("execution_resource_policy", ())
    )
    data["fixed_budget_stage_records"] = tuple(
        (
            str(stage), int(cap), int(realized), int(unused),
            bool(exhausted), bool(complete), reason,
        )
        for stage, cap, realized, unused, exhausted, complete, reason
        in data.get("fixed_budget_stage_records", ())
    )
    data["grouping_diagnostics"] = tuple(
        (str(name), str(item))
        for name, item in data.get("grouping_diagnostics", ())
    )
    return SeedEvaluation(**data)


class CheckpointStore:
    """Compatibility-guarded atomic JSON persistence."""

    def __init__(
        self,
        path: str,
        *,
        expected_metadata: Mapping[str, Any],
        every_n_evaluations: int,
    ) -> None:
        self.path = Path(path)
        self.expected_metadata = dict(expected_metadata)
        self.every_n_evaluations = int(every_n_evaluations)
        self._since_save = 0

    def load(self) -> Tuple[Dict[Tuple[str, int], SeedEvaluation], Dict[str, Any]]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise FileNotFoundError(f"Checkpoint does not exist: {self.path}") from error
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                "Incompatible checkpoint schema_version: "
                f"expected {SCHEMA_VERSION}, found {payload.get('schema_version')!r}."
            )
        if payload.get("compatibility") != self.expected_metadata:
            raise ValueError("Incompatible checkpoint metadata or fingerprint.")
        cache: Dict[Tuple[str, int], SeedEvaluation] = {}
        for item in payload.get("evaluation_cache", []):
            key = (str(item["fingerprint"]), int(item["learner_seed"]))
            cache[key] = _seed_evaluation_from_json(item["evaluation"])
        return cache, dict(payload.get("run_state", {}))

    def note_completed(
        self,
        cache: Mapping[Tuple[str, int], SeedEvaluation],
        run_state: Mapping[str, Any],
    ) -> None:
        self._since_save += 1
        if self._since_save >= self.every_n_evaluations:
            self.save(cache, run_state, status="running")

    def save(
        self,
        cache: Mapping[Tuple[str, int], SeedEvaluation],
        run_state: Mapping[str, Any],
        *,
        status: str,
    ) -> None:
        if status not in {"running", "complete"}:
            raise ValueError("Checkpoint status must be running or complete.")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "compatibility": self.expected_metadata,
            "evaluation_cache": [
                {
                    "fingerprint": fingerprint,
                    "learner_seed": seed,
                    "evaluation": _seed_evaluation_to_json(evaluation),
                }
                for (fingerprint, seed), evaluation in sorted(cache.items())
            ],
            "run_state": dict(run_state),
            "status": status,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        self._since_save = 0
