#!/usr/bin/env python3
"""Six-qubit Phase-7 demo with genuine third-order cumulant grouping.

The learner receives only ``CEBPLearnerView``.  Hidden block labels and exact
states are consulted only for explicitly labeled design checks and post-run
evaluation of the already-produced learner transcript.
"""

from __future__ import annotations

import dataclasses
import math
import sys
from enum import Enum
from typing import Any, Iterable

import numpy as np
import qutip as qt

from main_v2 import (
    EndToEndConfig,
    GroupingConfig,
    PeelingConfig,
    RecoveryConfig,
    SyndromeConfig,
    TomographyConfig,
    bell_score_uniform_radius,
    calibrated_end_to_end_schedule,
    debug_end_to_end_trace_error,
    debug_exact_pauli_expectation,
    debug_oracle_sector_block_labels,
    full_cebp_tomography,
    random_cebp_state,
)


# ---------------------------------------------------------------------------
# One fixed practical execution.  No value is selected using oracle truth.
# ---------------------------------------------------------------------------
N = 6
D = 3
BLOCK_SIZES = (3, 3)
P_B = 0.55
CLIFFORD_STEPS = 18
INSTANCE_SEED = 20260820

DEMO_EPSILON = 0.90
DEMO_DELTA = 0.10
LEARNER_SEED = 20260821

PEELING_CONFIG = PeelingConfig(
    h_min=0.85,
    h_max=0.95,
    eta=0.02,
    M1=8_000,
    zeta_bs=0.05,
    return_details=True,
)
RECOVERY_CONFIG = RecoveryConfig(
    theta=0.15,
    M2=50_000,
    zeta_rank=0.05,
    return_details=True,
    allow_uncalibrated_peeling=True,
    allow_margin_failure=True,
)
GROUPING_CONFIG = GroupingConfig(
    ell_grp=3,
    eta_test=0.06,
    tau_kappa=0.05,
    delta_grp_ordinary=0.10,
    return_details=True,
    allow_uncalibrated_recovery=True,
    allow_no_false_merge_margin_failure=True,
)
SYNDROME_CONFIG = SyndromeConfig(
    zeta_sgn=0.05,
    h_min=0.85,
    M_sgn=100,
    return_details=True,
)
TOMOGRAPHY_CONFIG = TomographyConfig(
    epsilon_tom=1.20,
    zeta_tom=0.10,
    allow_uncertified_localization=True,
    return_details=True,
    max_dense_qubits=6,
)
END_TO_END_CONFIG = EndToEndConfig(
    epsilon=DEMO_EPSILON,
    delta=DEMO_DELTA,
    seed=LEARNER_SEED,
    return_details=True,
    materialize_dense_estimator=True,
    max_dense_qubits=6,
    allow_uncertified_execution=True,
    peeling_override=PEELING_CONFIG,
    recovery_override=RECOVERY_CONFIG,
    grouping_override=GROUPING_CONFIG,
    syndrome_override=SYNDROME_CONFIG,
    tomography_override=TOMOGRAPHY_CONFIG,
)

FORMAL_SCHEDULE_FIELDS = (
    "branch", "n", "d", "epsilon", "delta", "R_ub", "A_d", "q_d",
    "Gamma_d", "epsilon_tom", "theta_0", "theta", "eta_s", "lambda_0",
    "h_min", "h_max", "peeling_eta", "eta_test", "tau_kappa", "tau_mu",
    "ell_grp", "zeta_peel", "zeta_rank", "zeta_grp", "zeta_sgn",
    "zeta_tom", "M1", "tau_1", "M2", "tau_rank", "M_sgn",
    "N_test_max_wc", "grouping_per_query_copies", "N_grp_wc", "N_P_wc",
    "tau_bp_wc", "M_P_wc", "N_bp_wc",
)
POOL_NAMES = (
    "peeling_bell_pool",
    "recovery_bell_pool",
    "grouping_ordinary_pool",
    "syndrome_sign_pool",
    "block_tomography_pool",
)


def heading(title: str, character: str = "=") -> None:
    print()
    print(character * 72)
    print(title)
    print(character * 72)


def fmt(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision=6, suppress_small=True)
    if isinstance(value, float):
        return f"{value:.17g}"
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def print_named_fields(obj: Any, names: Iterable[str], indent: str = "  ") -> None:
    for name in names:
        print(f"{indent}{name} = {fmt(getattr(obj, name))}")


def print_dataclass_config(label: str, config: Any) -> None:
    print(f"{label}:")
    for field in dataclasses.fields(config):
        print(f"  {field.name} = {fmt(getattr(config, field.name))}")


def pauli_operator(pauli: str) -> qt.Qobj:
    operators = {
        "I": qt.qeye(2),
        "X": qt.sigmax(),
        "Y": qt.sigmay(),
        "Z": qt.sigmaz(),
    }
    return qt.tensor(*(operators[axis] for axis in pauli))


def expectation(state: qt.Qobj, pauli: str) -> float:
    return float(np.real(qt.expect(pauli_operator(pauli), state)))


def density_eigenvalues(state: qt.Qobj) -> tuple[float, ...]:
    return tuple(float(x) for x in np.linalg.eigvalsh(state.full()))


def purity(state: qt.Qobj) -> float:
    density = state.proj() if state.isket else state
    return float(np.real((density * density).tr()))


def validate_density(state: qt.Qobj, dims: list[int]) -> None:
    matrix = state.full()
    assert state.dims == [dims, dims]
    assert np.allclose(matrix, matrix.conj().T, atol=1e-12)
    assert min(np.linalg.eigvalsh(matrix)) >= -1e-12
    assert np.isclose(float(np.real(state.tr())), 1.0, atol=1e-12)


def block_a_state() -> qt.Qobj:
    probabilities = np.array([13, 5, 5, 1, 5, 1, 1, 1], dtype=float) / 32.0
    return qt.Qobj(np.diag(probabilities), dims=[[2, 2, 2], [2, 2, 2]])


def bell_mixture(p: float) -> qt.Qobj:
    zero, one = qt.basis(2, 0), qt.basis(2, 1)
    phi_plus = (qt.tensor(zero, zero) + qt.tensor(one, one)).unit()
    rho = (1.0 - p) * qt.qeye([2, 2]) / 4.0 + p * phi_plus.proj()
    rho.dims = [[2, 2], [2, 2]]
    return rho


def internal_block_b_clifford() -> qt.Qobj:
    hadamard = qt.Qobj(np.array([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2.0))
    h0 = qt.tensor(hadamard, qt.qeye(2), qt.qeye(2))
    cnot = np.zeros((8, 8), dtype=complex)
    for column in range(8):
        bits = [(column >> shift) & 1 for shift in (2, 1, 0)]
        if bits[0]:
            bits[1] ^= 1
        row = (bits[0] << 2) | (bits[1] << 1) | bits[2]
        cnot[row, column] = 1.0
    return qt.Qobj(cnot, dims=[[2, 2, 2], [2, 2, 2]]) * h0


def block_a_design_diagnostics(rho_a: qt.Qobj) -> dict[str, float]:
    moments = {
        "Z1": expectation(rho_a, "ZII"),
        "Z2": expectation(rho_a, "IZI"),
        "Z3": expectation(rho_a, "IIZ"),
        "Z1Z2": expectation(rho_a, "ZZI"),
        "Z1Z3": expectation(rho_a, "ZIZ"),
        "Z2Z3": expectation(rho_a, "IZZ"),
        "Z1Z2Z3": expectation(rho_a, "ZZZ"),
    }
    cumulants = {
        "kappa(Z1,Z2)": moments["Z1Z2"] - moments["Z1"] * moments["Z2"],
        "kappa(Z1,Z3)": moments["Z1Z3"] - moments["Z1"] * moments["Z3"],
        "kappa(Z2,Z3)": moments["Z2Z3"] - moments["Z2"] * moments["Z3"],
        "kappa(Z1,Z2,Z3)": (
            moments["Z1Z2Z3"]
            - moments["Z1Z2"] * moments["Z3"]
            - moments["Z1Z3"] * moments["Z2"]
            - moments["Z2Z3"] * moments["Z1"]
            + 2.0 * moments["Z1"] * moments["Z2"] * moments["Z3"]
        ),
    }
    assert np.allclose(
        tuple(moments.values()),
        (0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.0),
        atol=1e-12,
    )
    assert np.allclose(
        (cumulants["kappa(Z1,Z2)"], cumulants["kappa(Z1,Z3)"],
         cumulants["kappa(Z2,Z3)"]),
        (0.0, 0.0, 0.0),
        atol=1e-12,
    )
    assert np.isclose(cumulants["kappa(Z1,Z2,Z3)"], -0.125, atol=1e-12)
    print("BLOCK A moments:")
    for name, value in moments.items():
        print(f"  <{name}> = {value:.12g}")
    print("BLOCK A cumulants:")
    for name, value in cumulants.items():
        print(f"  {name} = {value:.12g}")
    print("BLOCK A squared scores: s(Z_i)=0.25, s(Z_i Z_j)=0.0625, "
          "s(Z1 Z2 Z3)=0")
    return {**moments, **cumulants}


def block_b_design_diagnostics(bell: qt.Qobj, rho_b: qt.Qobj) -> None:
    xx = expectation(bell, "XX")
    yy = expectation(bell, "YY")
    zz = expectation(bell, "ZZ")
    assert np.allclose((xx, yy, zz), (P_B, -P_B, P_B), atol=1e-12)
    print(f"BLOCK B p_B = {P_B}")
    print(f"BLOCK B Bell component: <XX>={xx:.12g}, <YY>={yy:.12g}, "
          f"<ZZ>={zz:.12g}")
    print(f"BLOCK B p_B^2 = {P_B**2:.12g}")
    print(f"BLOCK B kappa(XX,ZZ) = {P_B - P_B**2:.12g}")
    print(f"BLOCK B purity = {purity(rho_b):.12g}")
    print("BLOCK B internal V_B = H(local qubit 0), then CNOT(0 -> 1)")


def print_formal_schedule(schedule: Any) -> None:
    heading("FORMAL THEOREM-CALIBRATED SCHEDULE\n(AUDIT ONLY — NOT EXECUTED)")
    print_named_fields(schedule, FORMAL_SCHEDULE_FIELDS)
    print("  reservation:")
    print_named_fields(schedule.reservation, (*POOL_NAMES, "total"), indent="    ")
    print("FORMAL SCHEDULE EXECUTED: NO")
    print("PRACTICAL MANUAL OVERRIDES EXECUTED: YES")


def print_practical_parameters() -> None:
    heading("PRACTICAL FIXED DEMO PARAMETERS")
    print(f"DEMO_EPSILON = {DEMO_EPSILON}")
    print(f"DEMO_DELTA = {DEMO_DELTA}")
    print(f"INSTANCE_SEED = {INSTANCE_SEED}")
    print(f"LEARNER_SEED = {LEARNER_SEED}")
    print_dataclass_config("PeelingConfig", PEELING_CONFIG)
    print_dataclass_config("RecoveryConfig", RECOVERY_CONFIG)
    print_dataclass_config("GroupingConfig", GROUPING_CONFIG)
    print_dataclass_config("SyndromeConfig", SYNDROME_CONFIG)
    print_dataclass_config("TomographyConfig", TOMOGRAPHY_CONFIG)
    print("actual concentration radii:")
    print(f"  tau1_actual = "
          f"{bell_score_uniform_radius(PEELING_CONFIG.M1, N, PEELING_CONFIG.zeta_bs):.12g}")
    print(f"  tau_rank_actual = "
          f"{bell_score_uniform_radius(RECOVERY_CONFIG.M2, N, RECOVERY_CONFIG.zeta_rank):.12g}")


def print_global_result(result: Any) -> None:
    heading("GLOBAL RESULT / STAGE SEEDS")
    print_named_fields(
        result,
        ("branch", "success", "theorem_certified", "failure_stage", "failure_reason"),
    )
    print("  StageSeedLedger:")
    print_named_fields(
        result.seed_ledger,
        ("master_seed", "peeling_seed", "syndrome_seed", "recovery_seed",
         "grouping_seed", "tomography_seed"),
        indent="    ",
    )


def print_peeling(peeling: Any) -> None:
    heading("PHASE 2 — PEELING")
    print_named_fields(
        peeling,
        ("success", "failure_reason", "h", "lambda_", "tau_1", "t",
         "generators", "generator_symplectic_vectors", "epsilon_peel", "M1",
         "score_provenance", "score_frame", "theorem_grid_condition",
         "theorem_tau_condition", "theorem_preconditions_hold", "theorem_certified"),
    )
    if peeling.transcript:
        print("  threshold transcript:")
        print("    h          inner outer irank orank same  isotropic accepted reason")
        for row in peeling.transcript:
            print(f"    {row.h:<10.6g} {row.inner_count:<5} {row.outer_count:<5} "
                  f"{row.inner_rank:<5} {row.outer_rank:<5} "
                  f"{str(row.spans_equal):<5} {str(row.isotropic):<9} "
                  f"{str(row.accepted):<8} {row.reason}")


def print_syndrome(syndrome: Any, instance: Any, peeling: Any) -> None:
    heading("SYNDROME")
    print_named_fields(
        syndrome,
        ("success", "failure_reason", "empirical_means", "syndrome_bits", "M_sgn",
         "calibrated_M_sgn", "syndrome_sign_pool", "theorem_preconditions_hold"),
    )
    exact = tuple(
        debug_exact_pauli_expectation(instance.state, generator)
        for generator in peeling.generators
    )
    print(f"  exact physical generator expectations = {exact} "
          "[ORACLE / EVALUATION ONLY]")


def print_recovery(recovery: Any, labels: tuple[tuple[int, ...], ...] | None) -> None:
    heading("PHASE 3 — RECOVERY")
    print_named_fields(
        recovery,
        ("success", "failure_reason", "n", "t", "m", "theta", "tau_rank", "M2",
         "zeta_rank", "ranked_survivor_count", "threshold_margin_holds",
         "ranking_gap_margin_holds", "theorem_recovery_preconditions_hold",
         "threshold_span_complete", "independent_axes", "recovered_span_basis"),
    )
    print("  sectors:")
    print("    id kind       x       z       y")
    for sector in recovery.sectors:
        print(f"    {sector.sector_id:<2} {sector.kind:<10} {sector.x:<7} "
              f"{str(sector.z):<7} {str(sector.y):<7}")
    if recovery.transcript:
        print("  recovery transcript:")
        print("    full_pauli score       residual dressed action affected")
        for row in recovery.transcript:
            print(f"    {row.full_pauli:<11} {row.score:<11.7g} "
                  f"{str(row.residual_pauli):<9} {str(row.dressed_pauli):<9} "
                  f"{row.action:<26} {row.affected_sector_ids}")
    if labels is not None:
        print("  ORACLE / EVALUATION ONLY — sector hidden-block labels:")
        for sector, sector_labels in zip(recovery.sectors, labels):
            print(f"    sector {sector.sector_id} -> {sector_labels}")
        print(f"  pure BLOCK A sector count = {sum(label == (0,) for label in labels)}")
        print(f"  pure BLOCK B sector count = {sum(label == (1,) for label in labels)}")


def flatten_clusters(clusters: Iterable[Iterable[int]]) -> frozenset[int]:
    return frozenset(sector for cluster in clusters for sector in cluster)


def grouping_third_order_audit(
    grouping: Any,
    label_map: dict[int, tuple[int, ...]],
) -> dict[str, Any]:
    block_a_ids = frozenset(key for key, value in label_map.items() if value == (0,))
    block_b_ids = frozenset(key for key, value in label_map.items() if value == (1,))
    q2_a_witnesses = []
    q3_a_witnesses = []
    q2_b_witnesses = []
    b_merge_scans = []
    a_merge_scans = []
    for scan_index, scan in enumerate(grouping.transcript, start=1):
        for witness in scan.hyperedges:
            support = flatten_clusters(witness.clusters)
            if scan.order == 2 and support <= block_a_ids:
                q2_a_witnesses.append((scan_index, witness))
            if scan.order == 3 and support == block_a_ids:
                q3_a_witnesses.append((scan_index, witness))
            if scan.order == 2 and support == block_b_ids:
                q2_b_witnesses.append((scan_index, witness))
        if scan.order == 2 and any(
            frozenset(cluster) == block_b_ids for cluster in scan.partition_after
        ) and not any(
            frozenset(cluster) == block_b_ids for cluster in scan.partition_before
        ):
            b_merge_scans.append((scan_index, scan))
        if scan.order == 3 and any(
            frozenset(cluster) == block_a_ids for cluster in scan.partition_after
        ) and not any(
            frozenset(cluster) == block_a_ids for cluster in scan.partition_before
        ):
            a_merge_scans.append((scan_index, scan))
    block_b_merged_q2 = bool(q2_b_witnesses and b_merge_scans)
    reset_after_b = any(scan.reset_to_two for _index, scan in b_merge_scans)
    no_block_a_q2 = not q2_a_witnesses
    block_a_q3 = bool(q3_a_witnesses)
    q3_magnitude_close = any(
        abs(abs(witness.cumulant) - 0.125) <= 0.04
        for _scan_index, witness in q3_a_witnesses
    )
    block_a_merged_q3 = bool(a_merge_scans) and any(
        scan_index == merge_index
        for scan_index, _witness in q3_a_witnesses
        for merge_index, _scan in a_merge_scans
    )
    final_sets = {frozenset(cluster) for cluster in grouping.clusters}
    final_expected = block_a_ids in final_sets and block_b_ids in final_sets
    demonstrated = bool(
        len(block_a_ids) == 3
        and len(block_b_ids) == 2
        and block_b_merged_q2
        and reset_after_b
        and no_block_a_q2
        and block_a_q3
        and q3_magnitude_close
        and block_a_merged_q3
        and final_expected
    )
    return {
        "block_a_ids": block_a_ids,
        "block_b_ids": block_b_ids,
        "q2_a_witnesses": tuple(q2_a_witnesses),
        "q3_a_witnesses": tuple(q3_a_witnesses),
        "q2_b_witnesses": tuple(q2_b_witnesses),
        "b_merge_scans": tuple(b_merge_scans),
        "a_merge_scans": tuple(a_merge_scans),
        "block_b_merged_q2": block_b_merged_q2,
        "reset_after_b": reset_after_b,
        "no_block_a_q2": no_block_a_q2,
        "block_a_q3": block_a_q3,
        "q3_magnitude_close": q3_magnitude_close,
        "block_a_merged_q3": block_a_merged_q3,
        "final_expected": final_expected,
        "third_order_grouping_demonstrated": demonstrated,
    }


def print_grouping(
    grouping: Any,
    label_map: dict[int, tuple[int, ...]] | None,
) -> dict[str, Any] | None:
    heading("PHASE 4 — GROUPING / THIRD-ORDER CENTRAL OUTPUT")
    print_named_fields(
        grouping,
        ("success", "failure_reason", "L", "ell_grp", "eta_test", "tau_kappa",
         "beta_peel", "clusters", "realized_query_count", "query_count_by_order",
         "realized_grouping_copies", "copies_by_order", "N_test_max",
         "ordinary_copy_upper_bound", "merge_rounds", "no_false_merge_condition_holds",
         "recovery_theorem_preconditions_hold", "theorem_grouping_preconditions_hold"),
    )
    print("  complete GroupingScan transcript:")
    for index, scan in enumerate(grouping.transcript, start=1):
        print(f"    SCAN {index}")
        print(f"      q/order = {scan.order}")
        print(f"      partition_before = {scan.partition_before}")
        print(f"      active_clusters = {scan.active_clusters}")
        print("      hyperedges:")
        for witness in scan.hyperedges:
            print(f"        clusters={witness.clusters} observables={witness.observables} "
                  f"cumulant={witness.cumulant:.12g}")
        if not scan.hyperedges:
            print("        (none)")
        print(f"      connected_components = {scan.connected_components}")
        print(f"      partition_after = {scan.partition_after}")
        print(f"      reset_to_two = {scan.reset_to_two}")
    if label_map is None:
        return None
    audit = grouping_third_order_audit(grouping, label_map)
    heading("EXPLICIT q=3 GROUPING AUDIT — ORACLE / EVALUATION ONLY", "-")
    print(f"BLOCK-A sector IDs = {tuple(sorted(audit['block_a_ids']))}")
    print(f"BLOCK-B sector IDs = {tuple(sorted(audit['block_b_ids']))}")
    print(f"A. q=2 retained edge wholly inside BLOCK A: "
          f"{'YES' if audit['q2_a_witnesses'] else 'NO'}")
    print(f"B. q=3 retained edge exactly joining BLOCK A: "
          f"{'YES' if audit['block_a_q3'] else 'NO'}")
    print("C. BLOCK-A q=3 witness details:")
    for scan_index, witness in audit["q3_a_witnesses"]:
        print(f"  scan={scan_index} observables={witness.observables} "
              f"empirical_cumulant={witness.cumulant:.12g}")
    if not audit["q3_a_witnesses"]:
        print("  (none)")
    print(f"   q=3 witness magnitude close to 1/8 (tolerance 0.04): "
          f"{audit['q3_magnitude_close']}")
    print(f"D. q=3 connected component caused size-3 BLOCK-A merge: "
          f"{audit['block_a_merged_q3']}")
    print(f"E. BLOCK B merged at q=2: {audit['block_b_merged_q2']}")
    print(f"F. algorithm reset q->2 after BLOCK-B merge: {audit['reset_after_b']}")
    print("final cluster oracle-label unions:")
    for cluster in grouping.clusters:
        union = tuple(sorted({label for sector_id in cluster
                              for label in label_map[sector_id]}))
        print(f"  cluster {cluster} -> labels {union}; block_pure={len(union) == 1}")
    print(f"third_order_grouping_demonstrated = "
          f"{audit['third_order_grouping_demonstrated']}")
    print("THIRD-ORDER GROUPING TARGET: "
          f"{'PASS' if audit['third_order_grouping_demonstrated'] else 'FAIL'}")
    return audit


def print_localization(localization: Any) -> None:
    heading("PHASE 5 — LOCALIZATION")
    print_named_fields(
        localization,
        ("success", "failure_reason", "n", "t", "m", "d", "K_rec", "J_C",
         "J_aux", "grouping_theorem_preconditions_hold",
         "theorem_localization_preconditions_hold", "handoff_valid",
         "cross_group_direct_sum_holds", "cross_group_symplectic_orthogonality_holds",
         "global_pairing_holds", "register_partition_holds",
         "localization_guarantee_holds", "localization_copy_count"),
    )
    print(f"  len(gates) = {len(localization.gates)}")
    print("  ClusterSymplecticStructure values:")
    for structure in localization.structures:
        print(f"    cluster={structure.cluster} dimension={structure.dimension} "
              f"symplectic_rank={structure.symplectic_rank} h_C={structure.h_C} "
              f"q_C={structure.q_C} k_C={structure.k_C}")
        print(f"      radical_basis={structure.radical_basis}")
        print(f"      hyperbolic_pairs={structure.hyperbolic_pairs}")
    print("  ClusterLocalization values:")
    for item in localization.cluster_localizations:
        print(f"    cluster={item.cluster} J_C={item.J_C} h_C={item.h_C} "
              f"q_C={item.q_C} k_C={item.k_C} "
              f"target_pair_qubits={item.target_pair_qubits}")


def print_tomography(tomography: Any) -> None:
    heading("PHASE 6 — TOMOGRAPHY")
    print_named_fields(
        tomography,
        ("success", "failure_reason", "Khat", "N_bp", "sum_local_schedule_lengths",
         "one_common_copy_per_round", "block_tomography_pool",
         "theorem_preconditions_hold", "J_aux"),
    )
    ratio = (tomography.N_bp / tomography.sum_local_schedule_lengths
             if tomography.sum_local_schedule_lengths else 0.0)
    print(f"  N_bp / sum_C L_C = {ratio:.12g}")
    print("  RegisterTomographyBudget values:")
    for budget in tomography.budgets:
        print_named_fields(
            budget,
            ("cluster", "J_C", "k_C", "N_C_Pauli", "epsilon_C", "zeta_C",
             "tau_C_tom", "M_C_Pauli", "L_C"),
            indent="    ",
        )
        print("    --")
    print("  RegisterTomographyEstimate values:")
    for estimate in tomography.estimates:
        linear_eigs = tuple(float(x) for x in np.linalg.eigvalsh(estimate.nu_hat_lin.full()))
        projected_eigs = tuple(float(x) for x in np.linalg.eigvalsh(estimate.nu_hat.full()))
        print(f"    cluster={estimate.cluster} J_C={estimate.J_C} k_C={estimate.k_C}")
        print(f"      eigenvalues(nu_hat_lin)={linear_eigs}")
        print(f"      eigenvalues(nu_hat)={projected_eigs}")
        print(f"      purity Tr(nu_hat^2)={purity(estimate.nu_hat):.12g}")
        print(f"      numerical_projection_error_bound="
              f"{estimate.numerical_projection_error_bound:.12g}")
        print("      learned Pauli coefficients:")
        for pauli, coefficient in sorted(estimate.pauli_coefficients):
            print(f"        {pauli} {coefficient:.12g}")


def print_certificates(result: Any) -> None:
    heading("PHASE 7 — STRUCTURAL / END-TO-END CERTIFICATES")
    if result.structural_certificate is None or result.end_to_end_certificate is None:
        print("  unavailable because the learner did not complete")
        return
    print("  StructuralCertificate:")
    print_named_fields(
        result.structural_certificate,
        ("branch", "Khat", "epsilon_peel", "theta_rec", "E_peel", "E_miss",
         "beta_peel", "xi_s", "xi_eff", "F_d_xi_eff", "split_terms",
         "E_split_cert", "E_struct_cert"),
        indent="    ",
    )
    print("  EndToEndCertificate:")
    print_named_fields(
        result.end_to_end_certificate,
        ("certified_trace_norm_bound", "target_epsilon", "failure_budgets",
         "total_failure_bound", "target_delta", "theorem_preconditions",
         "copy_reservation_valid", "realized_within_reserved", "theorem_certified"),
        indent="    ",
    )
    print("THIS MANUAL-OVERRIDE RUN IS NOT THEOREM-CERTIFIED.")


def print_copy_table(title: str, values: dict[str, int]) -> None:
    print(title)
    print("  pool                         copies")
    print("  ---------------------------------------------")
    for name in POOL_NAMES:
        print(f"  {name:<29} {values.get(name, 0)}")
    print("  ---------------------------------------------")
    print(f"  {'TOTAL':<29} {sum(values.get(name, 0) for name in POOL_NAMES)}")


def print_copy_accounting(result: Any) -> None:
    heading("COPY ACCOUNTING")
    assert result.realized_total == result.realized_copy_ledger.total
    print_copy_table("REALIZED COPY LEDGER", result.realized_copy_ledger.as_dict())
    reservation = result.worst_case_reservation
    formal = {name: int(getattr(reservation, name)) for name in POOL_NAMES}
    print_copy_table("FORMAL WORST-CASE RESERVATION", formal)
    print(f"actual/formal total ratio = "
          f"{result.realized_total / reservation.total:.12e}")
    if result.tomography is not None:
        total = result.tomography.sum_local_schedule_lengths
        print(f"N_bp = {result.tomography.N_bp}")
        print(f"sum_C L_C = {total}")
        print(f"N_bp / sum_C L_C = {result.tomography.N_bp / total:.12g}")


def print_available_diagnostics(
    result: Any,
    instance: Any,
) -> tuple[dict[int, tuple[int, ...]] | None, dict[str, Any] | None]:
    print_global_result(result)
    labels = None
    label_map = None
    audit = None
    if result.peeling is not None:
        print_peeling(result.peeling)
    if result.syndrome is not None and result.peeling is not None:
        print_syndrome(result.syndrome, instance, result.peeling)
    if result.recovery is not None and result.peeling is not None:
        if result.recovery.success:
            labels = debug_oracle_sector_block_labels(instance, result.peeling, result.recovery)
            label_map = {
                sector.sector_id: sector_labels
                for sector, sector_labels in zip(result.recovery.sectors, labels)
            }
        print_recovery(result.recovery, labels)
    if result.grouping is not None:
        audit = print_grouping(result.grouping, label_map)
    if result.localization is not None:
        print_localization(result.localization)
    if result.tomography is not None:
        print_tomography(result.tomography)
    print_certificates(result)
    print_copy_accounting(result)
    return label_map, audit


def validate_completed_result(
    result: Any,
    label_map: dict[int, tuple[int, ...]],
) -> None:
    assert result.success
    assert not result.theorem_certified
    assert result.decoded_density is not None
    validate_density(result.decoded_density, [2] * N)
    assert result.realized_total == result.realized_copy_ledger.total
    assert label_map
    for name in (
        "handoff_valid", "cross_group_direct_sum_holds",
        "cross_group_symplectic_orthogonality_holds", "global_pairing_holds",
        "register_partition_holds", "localization_guarantee_holds",
    ):
        assert getattr(result.localization, name), f"localization invariant failed: {name}"


def main() -> None:
    rho_a = block_a_state()
    bell = bell_mixture(P_B)
    zero_dm = qt.basis(2, 0).proj()
    v_b = internal_block_b_clifford()
    rho_b = v_b * qt.tensor(zero_dm, bell) * v_b.dag()
    rho_b = 0.5 * (rho_b + rho_b.dag())
    rho_b = rho_b / rho_b.tr()
    rho_b.dims = [[2, 2, 2], [2, 2, 2]]
    validate_density(rho_a, [2, 2, 2])
    validate_density(rho_b, [2, 2, 2])

    heading("CONTROLLED 6-QUBIT TARGET DESIGN\n"
            "ORACLE / DESIGN INFORMATION — NOT GIVEN TO LEARNER")
    block_a_design_diagnostics(rho_a)
    block_b_design_diagnostics(bell, rho_b)
    print(f"BLOCK A purity = {purity(rho_a):.12g}")
    print(f"BLOCK A eigenvalues = {density_eigenvalues(rho_a)}")
    print(f"BLOCK B eigenvalues = {density_eigenvalues(rho_b)}")
    print("both latent density validations = PASS")

    instance = random_cebp_state(
        n=N,
        d=D,
        block_sizes=BLOCK_SIZES,
        block_states=(rho_a, rho_b),
        pure=False,
        clifford_steps=CLIFFORD_STEPS,
        seed=INSTANCE_SEED,
    )
    formal_schedule = calibrated_end_to_end_schedule(
        n=N, d=D, epsilon=DEMO_EPSILON, delta=DEMO_DELTA
    )
    print_formal_schedule(formal_schedule)
    print_practical_parameters()

    truth = instance.oracle_truth
    heading("ORACLE / EVALUATION ONLY — NOT GIVEN TO LEARNER")
    print(f"hidden partition = {truth.hidden_partition}")
    print(f"latent block sizes = {tuple(len(block) for block in truth.hidden_partition)}")
    print(f"latent block purities = {tuple(purity(x) for x in truth.latent_block_states)}")
    print(f"latent product purity = {purity(truth.latent_product_state):.12g}")
    print(f"target encoded purity = {purity(instance.state):.12g}")
    print(f"encoder sampling mode = {truth.encoder_sampling}")
    print(f"encoder steps = {truth.encoder_steps}")
    print("encoder gate list:")
    for index, gate in enumerate(truth.encoder_gates):
        print(f"  {index:02d}: {gate}")
    print(f"encoder seed ledger = {instance.seed_ledger}")
    print(f"backend = {'ket' if instance.is_ket else 'density'}")

    learner_view = instance.learner_view()
    heading("LEARNER INPUT")
    print(f"n={learner_view.n}")
    print(f"d={learner_view.d}")
    print("hidden partition unavailable")
    print("latent states unavailable")
    print("encoder unavailable")
    assert not hasattr(learner_view, "oracle_truth")

    heading("RUNNING REAL PHASE-7 END-TO-END LEARNER")
    result = full_cebp_tomography(learner_view, config=END_TO_END_CONFIG)
    label_map, audit = print_available_diagnostics(result, instance)
    if not result.success:
        heading("DEMO FAILURE")
        print(f"failure_stage = {result.failure_stage}")
        print(f"failure_reason = {result.failure_reason}")
        raise SystemExit(1)
    assert label_map is not None
    assert audit is not None
    validate_completed_result(result, label_map)

    unhalved_error = debug_end_to_end_trace_error(result, instance)
    trace_distance = 0.5 * unhalved_error
    qutip_root_fidelity = float(qt.metrics.fidelity(result.decoded_density, instance.state))
    squared_uhlmann_fidelity = qutip_root_fidelity**2
    frobenius_error = float(
        np.linalg.norm(result.decoded_density.full() - instance.state.full(), ord="fro")
    )
    assert all(math.isfinite(value) for value in (
        unhalved_error, trace_distance, qutip_root_fidelity,
        squared_uhlmann_fidelity, frobenius_error,
    ))
    cluster_sizes = tuple(sorted(len(cluster) for cluster in result.grouping.clusters))
    register_sizes = tuple(sorted(len(register) for _cluster, register in result.localization.J_C))
    block_a_count = sum(value == (0,) for value in label_map.values())
    block_b_count = sum(value == (1,) for value in label_map.values())

    heading("FINAL 6-QUBIT RECONSTRUCTION RESULT")
    print(f"operational_success = {result.success}")
    print(f"theorem_certified = {result.theorem_certified}")
    print(f"recovered_t = {result.peeling.t}")
    print(f"recovered_BLOCK_A_sector_count = {block_a_count}")
    print(f"recovered_BLOCK_B_sector_count = {block_b_count}")
    print(f"final_cluster_count = {len(result.grouping.clusters)}")
    print(f"final_cluster_sizes = {cluster_sizes}")
    print(f"final_register_sizes = {register_sizes}")
    print(f"J_aux_size = {len(result.localization.J_aux)}")
    print(f"third_order_grouping_demonstrated = "
          f"{audit['third_order_grouping_demonstrated']}")
    print(f"unhalved_trace_norm_error = {unhalved_error:.12g}")
    print(f"conventional_trace_distance = {trace_distance:.12g}")
    print(f"qutip_fidelity_root = {qutip_root_fidelity:.12g} "
          f"[QuTiP {qt.__version__} returns sqrt(Jozsa fidelity)]")
    print(f"squared_uhlmann_fidelity = {squared_uhlmann_fidelity:.12g} "
          "[F=(Tr sqrt(sqrt(rho)sigma sqrt(rho)))^2]")
    print(f"frobenius_error = {frobenius_error:.12g}")
    print(f"realized_total_copies = {result.realized_total}")
    print(f"formal_reserved_copies = {result.worst_case_reservation.total}")
    print(f"actual/formal_ratio = "
          f"{result.realized_total / result.worst_case_reservation.total:.12e}")
    print("THIRD-ORDER GROUPING TARGET: "
          f"{'PASS' if audit['third_order_grouping_demonstrated'] else 'FAIL'}")
    print(f"SMALL-ERROR TARGET: {'PASS' if trace_distance <= 0.10 else 'FAIL'}")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as error:
        print(f"DEMO VALIDATION ERROR: {error}", file=sys.stderr)
        raise
