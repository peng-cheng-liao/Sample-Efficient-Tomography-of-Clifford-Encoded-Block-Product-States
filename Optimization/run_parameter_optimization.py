#!/usr/bin/env python3
"""Lightweight CLI for a deterministic optimizer smoke experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import qutip as qt

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main_v2 import MAX_SUPPORTED_END_TO_END_QUBITS, random_cebp_state
from Optimization.parameterization import OptimizationConfig, SearchSpace
from Optimization.search import optimize_cebp_parameters


def build_smoke_instance():
    """Return the deterministic nontrivial (1,2)-block remediation fixture."""

    zero = qt.basis(2, 0).proj()
    phi_plus = (
        qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
        + qt.tensor(qt.basis(2, 1), qt.basis(2, 1))
    ).unit()
    p = 0.55
    bell_mixture = (1.0 - p) * qt.qeye([2, 2]) / 4.0 + p * phi_plus.proj()
    return random_cebp_state(
        n=3,
        d=2,
        block_sizes=(1, 2),
        block_states=(zero, bell_mixture),
        clifford_steps=6,
        seed=20260810,
    )


def smoke_search_space() -> SearchSpace:
    """Conservative mechanics-test ranges, not state-specific oracle ranges."""

    return SearchSpace(
        alpha_peel=(0.025, 0.045),
        alpha_rank=(0.14, 0.23),
        alpha_sgn=(0.0005, 0.0015),
        c_peel=(1.12, 1.40),
        c_rank=(1.12, 1.45),
        h_min=(0.76, 0.84),
        h_span=(0.06, 0.12),
        theta=(0.10, 0.17),
        eta_test=(0.09, 0.14),
        kappa_ratio=(0.50, 0.70),
        epsilon_tom=(0.75, 1.30),
    )


def _supported_qubit_count(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= MAX_SUPPORTED_END_TO_END_QUBITS:
        raise argparse.ArgumentTypeError(
            "qubit limit must lie in [1, "
            f"{MAX_SUPPORTED_END_TO_END_QUBITS}]"
        )
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run the small deterministic mechanics test")
    parser.add_argument("--total-copies", type=int, default=None, help="hard physical-copy budget")
    parser.add_argument("--candidates", type=int, default=None, help="number of random candidates")
    parser.add_argument("--search-seed", type=int, default=271828, help="parameter-search RNG seed")
    parser.add_argument(
        "--max-enumeration-qubits",
        type=_supported_qubit_count,
        default=MAX_SUPPORTED_END_TO_END_QUBITS,
    )
    parser.add_argument(
        "--max-oracle-dense-qubits",
        type=_supported_qubit_count,
        default=MAX_SUPPORTED_END_TO_END_QUBITS,
    )
    parser.add_argument("--inner-enumeration-workers", type=int, default=1)
    parser.add_argument(
        "--simulation-backend",
        choices=("batched_counts", "legacy_shotwise"),
        default="batched_counts",
        help="Bell simulator backend; batched_counts is recommended for large M/n",
    )
    parser.add_argument("--max-score-array-bytes", type=int, default=None)
    parser.add_argument(
        "--max-structured-bell-workspace-bytes", type=int, default=None
    )
    parser.add_argument("--max-enumeration-workspace-bytes", type=int, default=None)
    parser.add_argument(
        "--enumeration-workspace-safety-factor", type=float, default=1.25
    )
    parser.add_argument("--verbose", action="store_true", help="show round lines and a compact top-candidate table")
    return parser


def _format_float(value: float) -> str:
    return f"{value:.8g}"


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    total_copies = args.total_copies or (300_000 if args.smoke else 500_000)
    candidates = args.candidates or (8 if args.smoke else 12)
    if total_copies <= 0 or candidates <= 0 or args.search_seed < 0:
        raise SystemExit("copy budget/candidate count must be positive and search seed nonnegative")

    instance = build_smoke_instance()
    config = OptimizationConfig(
        total_copies=total_copies,
        search_seed=args.search_seed,
        tuning_seeds=(10001, 10002),
        holdout_seeds=(20001, 20002, 20003),
        number_of_candidates=candidates,
        halving_seed_counts=(1, 2),
        retention_fraction=0.5,
        max_enumeration_qubits=args.max_enumeration_qubits,
        max_oracle_dense_qubits=args.max_oracle_dense_qubits,
        inner_enumeration_workers=args.inner_enumeration_workers,
        simulation_backend=args.simulation_backend,
        max_score_array_bytes=args.max_score_array_bytes,
        max_structured_bell_workspace_bytes=(
            args.max_structured_bell_workspace_bytes
        ),
        max_enumeration_workspace_bytes=args.max_enumeration_workspace_bytes,
        enumeration_workspace_safety_factor=(
            args.enumeration_workspace_safety_factor
        ),
        tomography_refinement_enabled=False if args.smoke else True,
        tomography_refinement_steps=4,
        execution_safety_factor=20.0,
        verbose=args.verbose,
    )
    space = smoke_search_space() if args.smoke else SearchSpace()
    print(
        f"CEBP parameter optimization: n={instance.n} d={instance.d} "
        f"budget={total_copies} candidates={candidates}"
    )
    print(
        "execution policy: "
        f"backend={config.simulation_backend} "
        f"enumeration_workers={config.inner_enumeration_workers} "
        f"max_enumeration_qubits={config.max_enumeration_qubits} "
        f"max_oracle_dense_qubits={config.max_oracle_dense_qubits} "
        f"workspace_cap={config.max_enumeration_workspace_bytes}"
    )
    result = optimize_cebp_parameters(instance, total_copies, config, space)

    for item in result.search_metadata.round_summaries:
        print(
            f"round {item.round_index}: alive={item.candidates_alive} "
            f"seeds={item.seeds_per_candidate} best={item.best_candidate_id} "
            f"mean={item.best_mean_loss:.6g} max_copies={item.best_max_copies}"
        )

    if args.verbose:
        highest_fidelity = max(
            item.n_seeds_evaluated for item in result.all_candidate_summaries
        )
        print(f"top candidate summaries at common fidelity n_seeds={highest_fidelity}:")
        ranked = sorted(
            (
                item for item in result.all_candidate_summaries
                if item.n_seeds_evaluated == highest_fidelity
            ),
            key=lambda item: (
                not item.all_budget_feasible,
                item.mean_loss,
                item.max_loss,
                -item.success_rate,
                item.max_realized_copies,
                item.candidate_id,
            ),
        )
        for item in ranked[: min(5, len(ranked))]:
            print(
                f"  {item.candidate_id}: seeds={item.n_seeds_evaluated} mean={item.mean_loss:.6g} "
                f"success={item.success_rate:.3f} max_copies={item.max_realized_copies}"
                f" budget_feasible={item.all_budget_feasible}"
            )

    tuning = result.tuning_evaluation
    holdout = result.holdout_evaluation
    print("OPTIMIZATION SMOKE RESULT")
    print("-------------------------")
    print(f"best candidate ID: {result.best_candidate_id}")
    print(f"total copy budget: {total_copies}")
    derived = result.best_derived_candidate
    print(
        "best derived settings: "
        f"M1={derived.M1} M2={derived.M2} M_sgn={derived.M_sgn} "
        f"tau1={derived.tau1:.6g} tau_rank={derived.tau_rank:.6g} "
        f"tau_kappa={derived.tau_kappa:.6g} epsilon_tom={derived.epsilon_tom:.6g}"
    )
    print(f"tuning mean loss: {_format_float(tuning.mean_loss)}")
    print(f"tuning success rate: {_format_float(tuning.success_rate)}")
    print(f"tuning max copies: {tuning.max_realized_copies}")
    print(f"tuning max utilization: {_format_float(tuning.max_copy_utilization)}")
    if holdout is None:
        print("holdout: not evaluated for fixed_error_min_copies")
    else:
        print(f"holdout mean loss: {_format_float(holdout.mean_loss)}")
        print(f"holdout success rate: {_format_float(holdout.success_rate)}")
        print(f"holdout max copies: {holdout.max_realized_copies}")
        print(f"holdout max utilization: {_format_float(holdout.max_copy_utilization)}")
        print(f"holdout budget robustness: {'PASS' if holdout.budget_robust else 'FAIL'}")
    representative = tuning.seed_evaluations[0]
    ledger = dict(representative.realized_copy_ledger)
    print("selected realized ledger: " + json.dumps(ledger, sort_keys=True))
    print(
        "structural summary: "
        f"t={representative.recovered_t} clusters={representative.cluster_sizes} "
        f"registers={representative.register_sizes} J_aux={representative.j_aux_size}"
    )
    pools_nonzero = all(
        ledger.get(name, 0) > 0
        for name in ("grouping_ordinary_pool", "syndrome_sign_pool", "block_tomography_pool")
    )
    print(f"grouping/sign/tomography pools nonzero: {pools_nonzero}")
    print(f"final trace distance: {_format_float(tuning.mean_loss)}")
    print(f"actual learner runs: {result.actual_learner_run_count}")
    print(f"preflight rejections: {result.preflight_rejection_count}")
    acceptable = (
        tuning.success_rate > 0.0
        and tuning.all_budget_feasible
        and pools_nonzero
        and representative.trace_distance is not None
    )
    return 0 if acceptable else 1


if __name__ == "__main__":
    raise SystemExit(main())
