#!/usr/bin/env python3
"""Run the controlled 6q fixed-error benchmark after adaptive-policy fixes."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Optimization.objective as objective_module  # noqa: E402
from Optimization import optimize_cebp_parameters_progressive  # noqa: E402
from tests.fixed_error_6q_adaptive_budget_validation import (  # noqa: E402
    run_fixed_error_6q_adaptive_budget_validation as base,
)


OUTPUT_ROOT = ROOT / "tests" / "fixed_error_6q_adaptive_budget_validation_v2"
PREVIOUS_ROOT = ROOT / "tests" / "fixed_error_6q_adaptive_budget_validation" / "benchmark"
PREVIOUS_PATHS = {d: PREVIOUS_ROOT / f"d{d}.json" for d in (1, 2, 3)}


def previous_metrics(d: int) -> dict[str, Any]:
    payload = json.loads(PREVIOUS_PATHS[d].read_text())
    if d == 1:
        cost = payload["partial_cost_diagnostics_from_last_checkpoint"]
        return {
            "status": payload["status"],
            "runtime_seconds": payload["runtime_seconds"],
            "unique_lambda_budget_trials": cost["unique_lambda_budget_trials"],
            "unique_lambda_budget_seed_evaluations": cost[
                "confirmed_unique_lambda_budget_seed_evaluations"
            ],
            "semantic_cache_hits": cost["semantic_cache_hits"],
            "largest_tested_budget": cost["largest_completed_tested_budget"],
            "average_budget_trials_per_candidate": cost[
                "average_unique_budget_trials_per_structural_candidate"
            ],
            "estimated_min_budget": None,
            "note": "Partial lower-bound cost at the point of raw OverflowError.",
        }
    cost = payload["cost_diagnostics"]
    return {
        "status": payload["status"],
        "runtime_seconds": payload["runtime_seconds"],
        "unique_lambda_budget_trials": cost["unique_lambda_budget_trials"],
        "unique_lambda_budget_seed_evaluations": cost[
            "unique_lambda_budget_seed_evaluations"
        ],
        "semantic_cache_hits": cost["semantic_cache_hits"],
        "largest_tested_budget": cost["largest_tested_budget"],
        "average_budget_trials_per_candidate": cost[
            "average_unique_budget_trials_per_structural_candidate"
        ],
        "estimated_min_budget": payload["threshold_result"]["estimated_min_budget"],
        "note": "Completed immediately previous 16-candidate adaptive benchmark.",
    }


def reduction(old: float | int | None, new: float | int | None) -> float | None:
    if old is None or new is None or float(old) == 0.0:
        return None
    return (float(old) - float(new)) / float(old) * 100.0


def state_identity(d: int, prior: dict[str, Any]):
    """Reproduce prior states, tolerating the documented d=1 last-bit digest drift."""

    try:
        return base.state_identity(d, prior)
    except RuntimeError:
        pass
    instance = base.state_runner.build_instance(d)
    repeated = base.state_runner.build_instance(d)
    difference = float(np.max(np.abs(instance.state.full() - repeated.state.full())))
    ledger = base.json_safe(instance.seed_ledger)
    cfg = prior["configuration"]
    checks = {
        "state_generation_seed_matches": cfg["state_generation_seed"] == base.STATE_SEEDS[d],
        "seed_ledger_matches": ledger == cfg["state_seed_ledger"],
        "raw_dense_digest_matches": (
            base.state_runner.state_digest(instance) == prior["state_digest_sha256"]
        ),
        "repeated_raw_dense_digest_matches": (
            base.state_runner.state_digest(repeated)
            == base.state_runner.state_digest(instance)
        ),
        "repeated_state_max_abs_difference": difference,
        "encoder_sampling_matches": (
            instance.oracle_truth.encoder_sampling == cfg["encoder_sampling"]
        ),
        "encoder_steps_matches": (
            int(instance.oracle_truth.encoder_steps) == int(cfg["encoder_steps"])
        ),
        "hidden_partition_matches": (
            [list(block) for block in instance.oracle_truth.hidden_partition]
            == cfg["exact_hidden_partition"]
        ),
        "current_state_digest_sha256": base.state_runner.state_digest(instance),
        "prior_state_digest_sha256": prior["state_digest_sha256"],
        "state_seed_ledger": ledger,
        "raw_digest_caveat": (
            "Exact seeds, ledger, encoder configuration, and partition match; "
            "two regenerations differ by at most the recorded floating last bits."
        ),
    }
    required = (
        checks["state_generation_seed_matches"],
        checks["seed_ledger_matches"],
        checks["encoder_sampling_matches"],
        checks["encoder_steps_matches"],
        checks["hidden_partition_matches"],
        difference <= 1e-14,
    )
    if not all(required):
        raise RuntimeError(f"d={d} scientific state identity could not be reproduced.")
    return instance, checks


def augment_payload(payload: dict[str, Any], result: Any, d: int) -> None:
    diagnostics = result.search_metadata.fixed_error_computational_diagnostics
    threshold = base.select_final_threshold(result)
    cost = payload["cost_diagnostics"]
    cost.update(
        {
            "cumulative_expansion_max_per_candidate": (
                diagnostics.cumulative_expansion_max_per_candidate
            ),
            "numeric_safety_event_count": diagnostics.numeric_safety_event_count,
            "incumbent_probe_count": diagnostics.incumbent_probe_count,
            "incumbent_confirmation_probe_count": (
                diagnostics.incumbent_confirmation_probe_count
            ),
            "incumbent_pruned_candidate_count": (
                diagnostics.incumbent_pruned_candidate_count
            ),
            "maximum_tested_physical_budget": (
                diagnostics.maximum_tested_physical_budget
            ),
        }
    )
    payload["configuration"].update(
        {
            "incumbent_confirmation_probes": 1,
            "incumbent_pruning_policy": (
                "probe at N_best, permit one probe at ceil(2*N_best), then "
                "heuristically prune only if no feasible observation exists"
            ),
            "numeric_sampling_safety_policy": (
                "runtime-derived min(NumPy intp max, C-long max) before learner"
            ),
        }
    )
    payload["threshold_result"].update(
        {
            "cumulative_expansion_trial_count": (
                threshold.cumulative_expansion_trial_count
            ),
            "largest_tested_budget": threshold.largest_tested_budget,
            "terminal_search_status": threshold.terminal_search_status,
            "incumbent_budget_hint": threshold.incumbent_budget_hint,
            "incumbent_guided_start": threshold.incumbent_guided_start,
            "incumbent_probe_count": threshold.incumbent_probe_count,
            "incumbent_confirmation_probe_count": (
                threshold.incumbent_confirmation_probe_count
            ),
            "incumbent_pruned": threshold.incumbent_pruned,
            "numeric_safety_event_count": threshold.numeric_safety_event_count,
        }
    )
    prior = previous_metrics(d)
    current = {
        "status": payload["status"],
        "runtime_seconds": payload["runtime_seconds"],
        "unique_lambda_budget_trials": cost["unique_lambda_budget_trials"],
        "unique_lambda_budget_seed_evaluations": cost[
            "unique_lambda_budget_seed_evaluations"
        ],
        "semantic_cache_hits": cost["semantic_cache_hits"],
        "largest_tested_budget": cost["largest_tested_budget"],
        "average_budget_trials_per_candidate": cost[
            "average_unique_budget_trials_per_structural_candidate"
        ],
        "estimated_min_budget": payload["threshold_result"][
            "estimated_min_budget"
        ],
    }
    payload["immediately_previous_benchmark_comparison"] = {
        "previous_result_path": str(PREVIOUS_PATHS[d].relative_to(ROOT)),
        "previous": prior,
        "new": current,
        "budget_trial_reduction_percentage": reduction(
            prior["unique_lambda_budget_trials"],
            current["unique_lambda_budget_trials"],
        ),
        "learner_evaluation_reduction_percentage": reduction(
            prior["unique_lambda_budget_seed_evaluations"],
            current["unique_lambda_budget_seed_evaluations"],
        ),
        "runtime_reduction_percentage": reduction(
            prior["runtime_seconds"], current["runtime_seconds"]
        ),
        "maximum_tested_budget_reduction_percentage": reduction(
            prior["largest_tested_budget"], current["largest_tested_budget"]
        ),
    }
    payload["acceptance_checks"] = {
        "no_raw_numeric_overflow": True,
        "cumulative_expansion_limit_respected": (
            diagnostics.cumulative_expansion_max_per_candidate
            <= payload["configuration"]["max_budget_expansion_rounds"]
        ),
        "incumbent_pruning_activated": (
            diagnostics.incumbent_pruned_candidate_count > 0
        ),
        "d1_grouping_allocation_zero": (
            None
            if d != 1
            else payload["physical_budget"]["per_stage_allocated_copies"][
                "grouping"
            ]
            == 0
        ),
        "final_threshold_feasible": bool(
            payload["final_16_seed_metrics"]
            and payload["final_16_seed_metrics"]["combined_final_feasibility"]
        ),
        "no_holdout_execution": not payload["holdout"]["learner_seed_calls"],
        "full_256_not_launched": payload["outer_search"]["final_pool_size"] == 16,
    }


def run_case(d: int, *, resume: bool, overwrite: bool = False) -> int:
    output_path = OUTPUT_ROOT / f"d{d}.json"
    checkpoint_path = OUTPUT_ROOT / f"d{d}.checkpoint.json"
    if output_path.exists() and not resume and not overwrite:
        raise FileExistsError(f"Fresh run refuses to overwrite {output_path}")
    if resume and not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing resume checkpoint: {checkpoint_path}")
    if checkpoint_path.exists() and not resume:
        raise FileExistsError(f"Fresh run refuses existing checkpoint: {checkpoint_path}")

    scientific_prior = base.load_prior(d)
    instance, state_checks = state_identity(d, scientific_prior)
    config = replace(
        base.build_config(d, "benchmark", resume=resume),
        checkpoint_path=str(checkpoint_path),
        checkpoint_key=f"6q-fixed-error-adaptive-policy-v2-d{d}-{base.STATE_SEEDS[d]}",
    )
    progressive = base.build_progressive(scientific_prior, "benchmark")
    search_space = base.build_search_space(scientific_prior)
    hashes_before = base.production_hashes()
    previous_hashes_before = {
        str(path.relative_to(ROOT)): base.sha256(path)
        for path in PREVIOUS_PATHS.values()
    }
    tracked_seeds: list[int] = []
    original = objective_module.evaluate_candidate_on_seed

    def tracked(*args, **kwargs):
        tracked_seeds.append(int(args[2]))
        return original(*args, **kwargs)

    print(f"[v2 d={d}] start", flush=True)
    started = time.perf_counter()
    objective_module.evaluate_candidate_on_seed = tracked
    try:
        result = optimize_cebp_parameters_progressive(
            instance,
            base.LEGACY_TOTAL_COPIES_API_ARGUMENT,
            config,
            search_space,
            progressive_config=progressive,
        )
        runtime = time.perf_counter() - started
        payload = base.result_payload(
            d=d,
            stage="benchmark",
            prior=scientific_prior,
            state_checks=state_checks,
            config=config,
            progressive=progressive,
            result=result,
            runtime=runtime,
            tracked_seeds=tracked_seeds,
        )
        payload["schema_version"] = 2
        payload["benchmark_version"] = "adaptive_efficiency_safety_v2"
        augment_payload(payload, result, d)
        code = 0
    except Exception as error:
        runtime = time.perf_counter() - started
        payload = {
            "schema_version": 2,
            "benchmark_version": "adaptive_efficiency_safety_v2",
            "status": "structured_runtime_failure",
            "configuration": {
                "n": base.N,
                "d": d,
                "partition": list(base.PARTITIONS[d]),
            },
            "state_identity": state_checks,
            "runtime_seconds": runtime,
            "failure": {
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
                "raw_overflow_error_escaped_controller": isinstance(
                    error, OverflowError
                ),
            },
        }
        code = 3
    finally:
        objective_module.evaluate_candidate_on_seed = original

    hashes_after = base.production_hashes()
    previous_hashes_after = {
        str(path.relative_to(ROOT)): base.sha256(path)
        for path in PREVIOUS_PATHS.values()
    }
    payload.update(
        {
            "production_hashes_before": hashes_before,
            "production_hashes_after": hashes_after,
            "production_files_unchanged_during_benchmark": (
                hashes_before == hashes_after
            ),
            "previous_result_hashes_before": previous_hashes_before,
            "previous_result_hashes_after": previous_hashes_after,
            "previous_results_unchanged": (
                previous_hashes_before == previous_hashes_after
            ),
            "tracked_tuning_learner_call_count": len(tracked_seeds),
        }
    )
    base.write_json(output_path, payload)
    if code == 0:
        cost = payload["cost_diagnostics"]
        print(
            f"[v2 d={d}] complete {runtime:.3f}s; "
            f"N={payload['threshold_result']['estimated_min_budget']}; "
            f"trials={cost['unique_lambda_budget_trials']}; "
            f"evals={cost['unique_lambda_budget_seed_evaluations']}; "
            f"pruned={cost['incumbent_pruned_candidate_count']}; "
            f"numeric={cost['numeric_safety_event_count']}",
            flush=True,
        )
        checkpoint_path.unlink(missing_ok=True)
    else:
        print(
            f"[v2 d={d}] stopped cleanly after {runtime:.3f}s: "
            f"{payload['failure']['exception_type']}: "
            f"{payload['failure']['exception_message']}",
            flush=True,
        )
    return code


def build_comparison() -> None:
    cases = {}
    for d in (1, 2, 3):
        path = OUTPUT_ROOT / f"d{d}.json"
        payload = json.loads(path.read_text()) if path.exists() else None
        cases[f"d{d}"] = {
            "result_path": str(path.relative_to(ROOT)),
            "status": None if payload is None else payload.get("status"),
            "comparison": (
                None
                if payload is None
                else payload.get("immediately_previous_benchmark_comparison")
            ),
            "acceptance_checks": (
                None if payload is None else payload.get("acceptance_checks")
            ),
        }
    base.write_json(
        OUTPUT_ROOT / "comparison.json",
        {
            "schema_version": 2,
            "description": (
                "Immediately previous versus cumulative-safety/incumbent-aware "
                "controlled 6q fixed-error 16-candidate benchmark"
            ),
            "full_256_runs_launched": False,
            "cases": cases,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("1", "2", "3", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--build-comparison", action="store_true")
    args = parser.parse_args()
    base.OUTPUT_ROOT = OUTPUT_ROOT
    if args.build_comparison and all(
        (OUTPUT_ROOT / f"d{d}.json").exists() for d in (1, 2, 3)
    ) and args.case == "all" and not args.resume:
        build_comparison()
        return 0
    cases = (1, 2, 3) if args.case == "all" else (int(args.case),)
    codes = []
    for d in cases:
        try:
            codes.append(
                run_case(d, resume=args.resume, overwrite=args.overwrite)
            )
        except Exception as error:
            print(f"[v2 d={d}] setup failure: {type(error).__name__}: {error}")
            codes.append(4)
    if args.build_comparison:
        build_comparison()
    return max(codes, default=0)


if __name__ == "__main__":
    raise SystemExit(main())
