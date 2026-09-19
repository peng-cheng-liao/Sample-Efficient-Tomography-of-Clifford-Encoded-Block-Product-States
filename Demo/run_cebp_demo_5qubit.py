#!/usr/bin/env python3
"""Deterministic, manual-budget five-qubit Phase-7 CEBP demonstration.

The learner is given only ``CEBPLearnerView``.  Exact states, the hidden
partition, and the encoder are used only in explicitly marked diagnostics
before or after the real end-to-end learner call.
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
# Fixed practical-demo parameters.  These are not selected at runtime.
# ---------------------------------------------------------------------------
N = 5
D = 3
BLOCK_SIZES = (3, 2)
P3 = 0.60
P2 = 0.55
CLIFFORD_STEPS = 15
INSTANCE_SEED = 20260810

DEMO_EPSILON = 0.80
DEMO_DELTA = 0.10
LEARNER_SEED = 20260811

PEELING_CONFIG = PeelingConfig(
    h_min=0.85,
    h_max=0.95,
    eta=0.02,
    M1=5_000,
    zeta_bs=0.05,
    return_details=True,
)
RECOVERY_CONFIG = RecoveryConfig(
    theta=0.15,
    M2=30_000,
    zeta_rank=0.05,
    return_details=True,
    allow_uncalibrated_peeling=True,
    allow_margin_failure=True,
)
GROUPING_CONFIG = GroupingConfig(
    ell_grp=3,
    eta_test=0.10,
    tau_kappa=0.12,
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
    epsilon_tom=0.80,
    zeta_tom=0.10,
    allow_uncertified_localization=True,
    return_details=True,
    max_dense_qubits=5,
)
END_TO_END_CONFIG = EndToEndConfig(
    epsilon=DEMO_EPSILON,
    delta=DEMO_DELTA,
    seed=LEARNER_SEED,
    return_details=True,
    materialize_dense_estimator=True,
    max_dense_qubits=5,
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
    print(character * 66)
    print(title)
    print(character * 66)


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


def bell_mixture(p: float) -> qt.Qobj:
    """Return (1-p) I/4 + p |Phi+><Phi+| with two-qubit dims."""
    zero, one = qt.basis(2, 0), qt.basis(2, 1)
    phi_plus = (qt.tensor(zero, zero) + qt.tensor(one, one)).unit()
    rho = (1.0 - p) * qt.qeye([2, 2]) / 4.0 + p * phi_plus.proj()
    rho.dims = [[2, 2], [2, 2]]
    return rho


def internal_three_qubit_clifford() -> qt.Qobj:
    """Return CNOT(0->1) H(0), using qubit 0 as the most-significant bit."""
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


def eigenvalues(state: qt.Qobj) -> tuple[float, ...]:
    return tuple(float(x) for x in np.linalg.eigvalsh(state.full()))


def purity(state: qt.Qobj) -> float:
    density = state.proj() if state.isket else state
    return float(np.real((density * density).tr()))


def validate_density(state: qt.Qobj, expected_dims: list[int]) -> None:
    matrix = state.full()
    assert state.dims == [expected_dims, expected_dims]
    assert np.allclose(matrix, matrix.conj().T, atol=1e-12)
    assert min(np.linalg.eigvalsh(matrix)) >= -1e-12
    assert np.isclose(float(np.real(state.tr())), 1.0, atol=1e-12)


def bell_diagnostics(label: str, rho: qt.Qobj, p: float) -> None:
    values = {}
    paulis = {"X": qt.sigmax(), "Y": qt.sigmay(), "Z": qt.sigmaz()}
    for axes in ("XX", "YY", "ZZ"):
        op = qt.tensor(*(paulis[axis] for axis in axes))
        values[axes] = float(np.real(qt.expect(op, rho)))
    print(f"{label}: <XX>={values['XX']:.9f}, <YY>={values['YY']:.9f}, "
          f"<ZZ>={values['ZZ']:.9f}, squared signal={p*p:.9f}, "
          f"kappa(XX,ZZ)={p-p*p:.9f}")
    assert np.allclose(
        (values["XX"], values["YY"], values["ZZ"]), (p, -p, p), atol=1e-12
    )


def print_formal_schedule(schedule: Any) -> None:
    heading("FORMAL THEOREM-CALIBRATED SCHEDULE\n(AUDIT ONLY — NOT EXECUTED)")
    print_named_fields(schedule, FORMAL_SCHEDULE_FIELDS)
    print("  reservation:")
    print_named_fields(schedule.reservation, (*POOL_NAMES, "total"), indent="    ")
    print("FORMAL SCHEDULE EXECUTED: NO")
    print("PRACTICAL MANUAL OVERRIDES EXECUTED: YES")


def print_practical_parameters() -> None:
    heading("PRACTICAL DEMO PARAMETERS")
    print(f"DEMO_EPSILON = {DEMO_EPSILON}")
    print(f"DEMO_DELTA = {DEMO_DELTA}")
    print(f"LEARNER_SEED = {LEARNER_SEED}")
    print_dataclass_config("PeelingConfig", PEELING_CONFIG)
    print_dataclass_config("RecoveryConfig", RECOVERY_CONFIG)
    print_dataclass_config("GroupingConfig", GROUPING_CONFIG)
    print_dataclass_config("SyndromeConfig", SYNDROME_CONFIG)
    print_dataclass_config("TomographyConfig", TOMOGRAPHY_CONFIG)
    tau1 = bell_score_uniform_radius(PEELING_CONFIG.M1, N, PEELING_CONFIG.zeta_bs)
    tau_rank = bell_score_uniform_radius(
        RECOVERY_CONFIG.M2, N, RECOVERY_CONFIG.zeta_rank
    )
    print(f"tau1_actual = {tau1:.12g}")
    print(f"tau_rank_actual = {tau_rank:.12g}")
    print("DESIGN DIAGNOSTICS ONLY:")
    for name, value in (
        ("p3^2", P3**2), ("p2^2", P2**2),
        ("p3^2*p2^2", P3**2 * P2**2),
        ("p3-p3^2", P3 - P3**2), ("p2-p2^2", P2 - P2**2),
    ):
        print(f"  {name} = {value:.12g}")


def print_copy_table(title: str, values: dict[str, int]) -> None:
    print(title)
    print("  pool                         copies")
    print("  -----------------------------------------")
    for name in POOL_NAMES:
        print(f"  {name:<29} {values.get(name, 0)}")
    print("  -----------------------------------------")
    print(f"  {'TOTAL':<29} {sum(values.get(name, 0) for name in POOL_NAMES)}")


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
        print("    h          inner outer irank orank same isotropic accepted reason")
        for row in peeling.transcript:
            print(
                f"    {row.h:<10.6g} {row.inner_count:<5} {row.outer_count:<5} "
                f"{row.inner_rank:<5} {row.outer_rank:<5} {str(row.spans_equal):<5} "
                f"{str(row.isotropic):<9} {str(row.accepted):<8} {row.reason}"
            )


def print_syndrome(syndrome: Any, instance: Any, peeling: Any) -> None:
    heading("SYNDROME")
    true_expectations = tuple(
        debug_exact_pauli_expectation(instance.state, generator)
        for generator in peeling.generators
    )
    print("  true generator expectations = "
          f"{fmt(true_expectations)}  [ORACLE / EVALUATION ONLY]")
    print_named_fields(
        syndrome,
        ("success", "failure_reason", "empirical_means", "syndrome_bits", "M_sgn",
         "calibrated_M_sgn", "syndrome_sign_pool", "theorem_preconditions_hold"),
    )


def print_recovery(recovery: Any, labels: tuple[tuple[int, ...], ...] | None) -> None:
    heading("PHASE 3 — RECOVERY")
    print_named_fields(
        recovery,
        ("success", "failure_reason", "n", "t", "m", "theta", "tau_rank",
         "lambda_", "M2", "zeta_rank", "ranked_survivor_count",
         "threshold_margin_holds", "ranking_gap_margin_holds",
         "theorem_recovery_preconditions_hold", "threshold_span_complete",
         "recovered_span_basis", "independent_axes"),
    )
    print("  recovered sectors:")
    print("    id kind       x      z      y")
    for sector in recovery.sectors:
        print(f"    {sector.sector_id:<2} {sector.kind:<10} {sector.x:<6} "
              f"{str(sector.z):<6} {str(sector.y):<6}")
    if recovery.transcript:
        print("  recovery transcript:")
        print("    full_pauli score       residual dressed action affected")
        for row in recovery.transcript:
            print(f"    {row.full_pauli:<10} {row.score:<11.7g} "
                  f"{str(row.residual_pauli):<8} {str(row.dressed_pauli):<8} "
                  f"{row.action:<26} {row.affected_sector_ids}")
    if labels is not None:
        print("  ORACLE / EVALUATION ONLY — hidden-block labels:")
        for sector, sector_labels in zip(recovery.sectors, labels):
            print(f"    sector {sector.sector_id} -> {sector_labels}")


def print_grouping(grouping: Any, label_map: dict[int, tuple[int, ...]] | None) -> None:
    heading("PHASE 4 — GROUPING")
    print_named_fields(
        grouping,
        ("success", "failure_reason", "L", "ell_grp", "eta_test", "tau_kappa",
         "beta_peel", "eta_s", "xi_s", "no_false_merge_condition_holds",
         "theorem_grouping_preconditions_hold", "clusters", "realized_query_count",
         "query_count_by_order", "realized_grouping_copies", "copies_by_order",
         "N_test_max", "ordinary_copy_upper_bound", "merge_rounds"),
    )
    print("  retained HyperedgeWitness values:")
    for witness in grouping.hyperedge_witnesses:
        print(f"    order={witness.order} clusters={witness.clusters} "
              f"observables={witness.observables} cumulant={witness.cumulant:.12g}")
    if not grouping.hyperedge_witnesses:
        print("    (none)")
    if label_map is not None:
        print("  ORACLE / EVALUATION ONLY — final-cluster hidden-block unions:")
        for cluster in grouping.clusters:
            union = tuple(sorted({label for sector_id in cluster
                                  for label in label_map[sector_id]}))
            print(f"    cluster {cluster} -> labels {union}; block_pure={len(union) == 1}")


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
    print(f"  N_bp / sum_local_schedule_lengths = {ratio:.12g}")
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
        lin_eigs = tuple(float(x) for x in np.linalg.eigvalsh(estimate.nu_hat_lin.full()))
        projected_eigs = tuple(float(x) for x in np.linalg.eigvalsh(estimate.nu_hat.full()))
        print(f"    cluster={estimate.cluster} J_C={estimate.J_C} k_C={estimate.k_C}")
        print(f"      numerical_projection_error_bound="
              f"{estimate.numerical_projection_error_bound:.12g}")
        print(f"      eigenvalues(nu_hat_lin)={lin_eigs}")
        print(f"      eigenvalues(nu_hat)={projected_eigs}")
        print(f"      purity Tr(nu_hat^2)={purity(estimate.nu_hat):.12g}")
        print("      learned local Pauli coefficients:")
        for pauli, coefficient in sorted(estimate.pauli_coefficients):
            print(f"        {pauli} {coefficient:.12g}")


def print_certificates(result: Any) -> None:
    heading("PHASE 7 — STRUCTURAL / END-TO-END CERTIFICATE")
    structural = result.structural_certificate
    certificate = result.end_to_end_certificate
    if structural is None or certificate is None:
        print("  certificate unavailable because the learner did not complete")
        return
    print("  StructuralCertificate:")
    print_named_fields(
        structural,
        ("branch", "Khat", "epsilon_peel", "theta_rec", "E_peel", "E_miss",
         "beta_peel", "xi_s", "xi_eff", "F_d_xi_eff", "split_terms",
         "E_split_cert", "E_struct_cert"),
        indent="    ",
    )
    print("  EndToEndCertificate:")
    print_named_fields(
        certificate,
        ("certified_trace_norm_bound", "target_epsilon", "failure_budgets",
         "total_failure_bound", "target_delta", "theorem_preconditions",
         "copy_reservation_valid", "realized_within_reserved", "theorem_certified"),
        indent="    ",
    )
    print("  NOTE: certificate is NOT a theorem guarantee for this manual-override run.")


def print_copy_accounting(result: Any) -> None:
    heading("COPY ACCOUNTING")
    realized = result.realized_copy_ledger.as_dict()
    assert result.realized_total == result.realized_copy_ledger.total
    print_copy_table("REALIZED COPY LEDGER", realized)
    reservation = result.worst_case_reservation
    formal = {name: int(getattr(reservation, name)) for name in POOL_NAMES}
    print_copy_table("FORMAL WORST-CASE RESERVATION", formal)
    ratio = result.realized_total / reservation.total
    print(f"actual/formal total ratio = {ratio:.12e}")
    if result.tomography is not None:
        print(f"N_bp = {result.tomography.N_bp}")
        print(f"sum_C L_C = {result.tomography.sum_local_schedule_lengths}")


def print_available_diagnostics(result: Any, instance: Any) -> tuple[Any, Any]:
    print_global_result(result)
    labels = None
    label_map = None
    if result.peeling is not None:
        print_peeling(result.peeling)
    if result.syndrome is not None and result.peeling is not None:
        print_syndrome(result.syndrome, instance, result.peeling)
    if result.recovery is not None and result.peeling is not None:
        if result.recovery.success:
            labels = debug_oracle_sector_block_labels(
                instance, result.peeling, result.recovery
            )
            label_map = {
                sector.sector_id: sector_labels
                for sector, sector_labels in zip(result.recovery.sectors, labels)
            }
        print_recovery(result.recovery, labels)
    if result.grouping is not None:
        print_grouping(result.grouping, label_map)
    if result.localization is not None:
        print_localization(result.localization)
    if result.tomography is not None:
        print_tomography(result.tomography)
    print_certificates(result)
    print_copy_accounting(result)
    return labels, label_map


def validate_completed_result(result: Any, instance: Any, label_map: dict[int, tuple[int, ...]]) -> None:
    assert result.success
    assert not result.theorem_certified
    assert result.decoded_density is not None
    validate_density(result.decoded_density, [2] * N)
    assert result.realized_total == result.realized_copy_ledger.total
    loc = result.localization
    assert loc is not None
    for name in (
        "handoff_valid", "cross_group_direct_sum_holds",
        "cross_group_symplectic_orthogonality_holds", "global_pairing_holds",
        "register_partition_holds", "localization_guarantee_holds",
    ):
        assert getattr(loc, name), f"returned structural invariant failed: {name}"
    for cluster in result.grouping.clusters:
        union = {label for sector_id in cluster for label in label_map[sector_id]}
        assert union, "oracle evaluation found an unlabeled cluster"


def main() -> None:
    rho_phi_3 = bell_mixture(P3)
    rho2 = bell_mixture(P2)
    zero_dm = qt.basis(2, 0).proj()
    v3 = internal_three_qubit_clifford()
    rho3 = v3 * qt.tensor(zero_dm, rho_phi_3) * v3.dag()
    rho3 = 0.5 * (rho3 + rho3.dag())
    rho3 = rho3 / rho3.tr()
    rho3.dims = [[2, 2, 2], [2, 2, 2]]
    validate_density(rho3, [2, 2, 2])
    validate_density(rho2, [2, 2])

    heading("CONTROLLED BLOCK CONSTRUCTION — DESIGN DIAGNOSTICS ONLY")
    bell_diagnostics("three-qubit block's Bell-mixture component", rho_phi_3, P3)
    bell_diagnostics("two-qubit block", rho2, P2)
    print(f"rho3 Hermitian/PSD/trace-one/dims validation = PASS")
    print(f"rho3 purity = {purity(rho3):.12g}")
    print("internal Clifford V3 = H(local qubit 0), then CNOT(0 -> 1)")

    instance = random_cebp_state(
        n=N,
        d=D,
        block_sizes=BLOCK_SIZES,
        block_states=(rho3, rho2),
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
    print(f"true hidden partition = {truth.hidden_partition}")
    print(f"latent block sizes = {tuple(len(block) for block in truth.hidden_partition)}")
    print(f"p3 = {P3}; p2 = {P2}")
    print(f"latent block purities = {tuple(purity(x) for x in truth.latent_block_states)}")
    print(f"latent block eigenvalues = {tuple(eigenvalues(x) for x in truth.latent_block_states)}")
    print(f"latent product purity = {purity(truth.latent_product_state):.12g}")
    print(f"encoder sampling mode = {truth.encoder_sampling}")
    print(f"encoder steps = {truth.encoder_steps}")
    print("encoder gate list:")
    for index, gate in enumerate(truth.encoder_gates):
        print(f"  {index:02d}: {gate}")
    print(f"encoder seed ledger = {instance.seed_ledger}")
    print(f"encoded target purity = {purity(instance.state):.12g}")
    print(f"backend = {'ket' if instance.is_ket else 'density'}")

    learner_view = instance.learner_view()
    heading("LEARNER INPUT")
    print(f"n={learner_view.n}")
    print(f"d={learner_view.d}")
    print("oracle partition hidden: YES")
    print("latent states hidden: YES")
    print("encoder hidden: YES")
    assert not hasattr(learner_view, "oracle_truth")

    heading("RUNNING REAL PHASE-7 END-TO-END LEARNER")
    result = full_cebp_tomography(learner_view, config=END_TO_END_CONFIG)
    _labels, label_map = print_available_diagnostics(result, instance)
    if not result.success:
        heading("DEMO FAILURE")
        print(f"failure_stage = {result.failure_stage}")
        print(f"failure_reason = {result.failure_reason}")
        raise SystemExit(1)
    assert label_map is not None
    validate_completed_result(result, instance, label_map)

    unhalved_error = debug_end_to_end_trace_error(result, instance)
    trace_distance = 0.5 * unhalved_error
    qutip_root_fidelity = float(qt.metrics.fidelity(result.decoded_density, instance.state))
    squared_uhlmann_fidelity = qutip_root_fidelity**2
    frobenius_error = float(
        np.linalg.norm(result.decoded_density.full() - instance.state.full(), ord="fro")
    )
    assert all(math.isfinite(x) for x in (
        unhalved_error, trace_distance, qutip_root_fidelity,
        squared_uhlmann_fidelity, frobenius_error,
    ))
    loc = result.localization
    register_sizes = tuple(len(register) for _cluster, register in loc.J_C)

    heading("FINAL RECONSTRUCTION RESULT")
    print(f"operational_success = {result.success}")
    print(f"theorem_certified = {result.theorem_certified}")
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
    print(f"final_cluster_count = {len(result.grouping.clusters)}")
    print(f"final_register_sizes = {register_sizes}")
    print(f"J_aux_size = {len(loc.J_aux)}")
    print(f"recovered_t = {result.peeling.t}")
    print(f"SMALL-ERROR TARGET: {'PASS' if trace_distance <= 0.10 else 'NOT MET'}")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as error:
        print(f"DEMO VALIDATION ERROR: {error}", file=sys.stderr)
        raise
