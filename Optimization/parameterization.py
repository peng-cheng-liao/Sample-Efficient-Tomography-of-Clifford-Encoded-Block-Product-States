"""Candidate sampling and conversion to the validated learner's public API."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, replace
import math
from typing import Optional, Tuple, Union

import numpy as np

from main_v2 import (
    adaptive_cumulant_test_bound,
    EnumerationExecutionConfig,
    EndToEndConfig,
    ExecutionPolicy,
    FixedBudgetStageWeights,
    fixed_budget_nominal_stage_caps,
    GroupingConfig,
    GroupingSamplingPolicy,
    MAX_SUPPORTED_END_TO_END_QUBITS,
    ordinary_cumulant_sample_count,
    ordinary_grouping_copy_upper_bound,
    PeelingConfig,
    RecoveryConfig,
    register_tomography_budget,
    SimulationBackend,
    simplified_cumulant_test_bound,
    SyndromeConfig,
    TomographyConfig,
    bell_score_uniform_radius,
)

from .specification import (
    OptimizationMode,
    OptimizationObjective,
    default_fixed_budget_objective,
)


Range = Tuple[float, float]


class InvalidCandidateError(ValueError):
    """A sampled point cannot define a valid practical learner execution."""


def _check_range(name: str, bounds: Range, *, positive: bool = False) -> None:
    if len(bounds) != 2:
        raise ValueError(f"{name} must be a (low, high) pair.")
    low, high = map(float, bounds)
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ValueError(f"{name} must have finite low < high bounds.")
    if positive and low <= 0.0:
        raise ValueError(f"{name} must have positive bounds.")


@dataclass(frozen=True)
class SearchSpace:
    """Broad, state-agnostic ranges for practical manual overrides."""

    alpha_peel: Range = (0.002, 0.12)
    alpha_rank: Range = (0.005, 0.25)
    alpha_sgn: Range = (0.0001, 0.03)
    c_peel: Range = (1.02, 2.0)
    c_rank: Range = (1.02, 2.0)
    h_min: Range = (0.55, 0.95)
    h_span: Range = (0.02, 0.30)
    h_max: Range = (0.60, 0.99)
    theta: Range = (0.03, 0.50)
    theta_tau_multiplier: Range = (1.5, 128.0)
    eta_test: Range = (0.02, 0.30)
    kappa_ratio: Range = (0.35, 1.50)
    epsilon_tom: Range = (0.35, 1.80)
    peel_weight: Range = (0.20, 5.0)
    recovery_weight: Range = (0.20, 5.0)
    grouping_weight: Range = (0.20, 5.0)
    syndrome_weight: Range = (0.05, 2.0)
    tomography_weight: Range = (0.20, 5.0)

    def __post_init__(self) -> None:
        for name in (
            "alpha_peel", "alpha_rank", "alpha_sgn", "c_peel", "c_rank",
            "h_min", "h_span", "h_max", "theta", "theta_tau_multiplier",
            "eta_test", "kappa_ratio",
            "epsilon_tom",
            "peel_weight", "recovery_weight", "grouping_weight",
            "syndrome_weight", "tomography_weight",
        ):
            _check_range(name, getattr(self, name), positive=True)


@dataclass(frozen=True)
class CandidateParameters:
    alpha_peel: float
    alpha_rank: float
    alpha_sgn: float
    c_peel: float
    c_rank: float
    h_min: float
    h_span: float
    theta: float
    eta_test: float
    kappa_ratio: float
    epsilon_tom: float
    peel_weight: float = 1.0
    recovery_weight: float = 1.0
    grouping_weight: float = 1.0
    syndrome_weight: float = 0.25
    tomography_weight: float = 2.0

    @property
    def h_max(self) -> float:
        return float(self.h_min + self.h_span)

    def with_epsilon_tom(self, epsilon_tom: float) -> "CandidateParameters":
        return replace(self, epsilon_tom=float(epsilon_tom))


@dataclass(frozen=True)
class FixedBudgetCandidateParameters:
    """The complete active numerical fixed-budget search point.

    Normalization removes one degree of freedom from the five weights, leaving
    eight effective continuous degrees of freedom in total.
    """

    h_min: float
    h_max: float
    theta_tau_multiplier: float
    eta_test: float
    peel_weight: float = 1.0
    recovery_weight: float = 1.0
    grouping_weight: float = 1.0
    syndrome_weight: float = 0.25
    tomography_weight: float = 2.0


@dataclass(frozen=True, init=False)
class FixedErrorCandidateParameters:
    """Persistent structural coordinates for modern fixed-error search.

    The structural coordinates intentionally match ``FixedBudgetCandidateParameters``.
    Physical budget is deliberately absent: each learner trial supplies it
    transiently to :func:`derive_candidate`.
    """

    h_min: float
    h_max: float
    theta_tau_multiplier: float
    eta_test: float
    peel_weight: float = 1.0
    recovery_weight: float = 1.0
    grouping_weight: float = 1.0
    syndrome_weight: float = 0.25
    tomography_weight: float = 2.0

    def __init__(
        self,
        h_min: float,
        h_max: float,
        theta_tau_multiplier: float,
        eta_test: float,
        peel_weight: float = 1.0,
        recovery_weight: float = 1.0,
        grouping_weight: float = 1.0,
        syndrome_weight: float = 0.25,
        tomography_weight: float = 2.0,
        *,
        N_candidate: Optional[int] = None,
    ) -> None:
        """Build structural parameters, ignoring a readable legacy budget field."""

        if N_candidate is not None and (
            isinstance(N_candidate, bool)
            or not isinstance(N_candidate, (int, np.integer))
            or int(N_candidate) <= 0
        ):
            raise ValueError("Legacy N_candidate metadata must be positive or None.")
        for name, value in (
            ("h_min", h_min),
            ("h_max", h_max),
            ("theta_tau_multiplier", theta_tau_multiplier),
            ("eta_test", eta_test),
            ("peel_weight", peel_weight),
            ("recovery_weight", recovery_weight),
            ("grouping_weight", grouping_weight),
            ("syndrome_weight", syndrome_weight),
            ("tomography_weight", tomography_weight),
        ):
            object.__setattr__(self, name, float(value))


OptimizationCandidate = Union[
    CandidateParameters,
    FixedBudgetCandidateParameters,
    FixedErrorCandidateParameters,
]


@dataclass(frozen=True)
class OptimizationConfig:
    """Fixed experiment controls; defaults intentionally describe a small run.

    Exhaustive simulator memory policy is expressed in bytes. The unified
    ``max_enumeration_workspace_bytes`` cap is preferred; the score-only and
    structured-Bell caps remain available for backward-compatible stricter
    policies. Optimization defaults to compressed ``batched_counts`` records;
    ``legacy_shotwise`` remains an explicit reference/debug backend.
    """

    total_copies: int = 500_000
    search_seed: int = 271828
    tuning_seeds: Tuple[int, ...] = (10001, 10002)
    holdout_seeds: Tuple[int, ...] = (20001, 20002)
    peeling_grid_intervals: int = 5
    delta_grp_ordinary: float = 0.10
    zeta_sgn: float = 0.05
    zeta_tom: float = 0.10
    fixed_budget_zeta_peel: float = 0.05
    fixed_budget_zeta_rank: float = 0.05
    fixed_budget_tau_kappa_diagnostic: float = 0.05
    fixed_budget_epsilon_tom_diagnostic: float = 1.0
    minimum_M1: int = 8
    minimum_M2: int = 8
    minimum_M_sgn: int = 4
    max_dense_qubits: int = 8
    max_enumeration_qubits: int | None = MAX_SUPPORTED_END_TO_END_QUBITS
    max_oracle_dense_qubits: int | None = MAX_SUPPORTED_END_TO_END_QUBITS
    inner_enumeration_workers: int = 1
    candidate_workers: InitVar[int | None] = None
    max_score_array_bytes: int | None = None
    max_structured_bell_workspace_bytes: int | None = None
    max_enumeration_workspace_bytes: int | None = None
    enumeration_workspace_safety_factor: float = 1.25
    simulation_backend: SimulationBackend = "batched_counts"
    failure_loss: float = 1.0
    number_of_candidates: int = 8
    halving_seed_counts: Tuple[int, ...] = (1, 2)
    retention_fraction: float = 0.5
    tomography_refinement_enabled: bool = False
    tomography_refinement_steps: int = 4
    budget_refinement_enabled: bool = True
    budget_refinement_steps: int = 16
    budget_refinement_tolerance: int = 1
    fixed_error_initial_budget_hint: int | None = None
    fixed_error_budget_growth_factor: float = 2.0
    fixed_error_budget_relative_tolerance: float = 0.01
    fixed_error_max_budget_expansion_rounds: int = 32
    fixed_error_incumbent_confirmation_probes: int = 1
    fixed_error_hard_safety_budget: int | None = None
    target_copy_utilization: float = 0.99
    objective: Optional[OptimizationObjective] = None
    max_preflight_estimated_copies: int | None = None
    execution_safety_factor: float = 8.0
    max_predicted_grouping_copies: int | None = None
    max_predicted_tomography_copies: int | None = None
    max_predicted_total_copies: int | None = None
    max_single_grouping_query_shots: int | None = 2_000_000
    checkpoint_path: str | None = None
    resume_from_checkpoint: bool = False
    checkpoint_every_n_evaluations: int = 1
    checkpoint_key: str = ""
    verbose: bool = False

    def __post_init__(self, candidate_workers: int | None) -> None:
        integer_fields = (
            "total_copies", "peeling_grid_intervals", "minimum_M1", "minimum_M2",
            "minimum_M_sgn", "max_dense_qubits", "number_of_candidates",
            "tomography_refinement_steps",
            "budget_refinement_steps", "budget_refinement_tolerance",
            "fixed_error_max_budget_expansion_rounds",
            "fixed_error_incumbent_confirmation_probes",
            "checkpoint_every_n_evaluations",
            "inner_enumeration_workers",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.max_enumeration_qubits is None:
            object.__setattr__(
                self, "max_enumeration_qubits", MAX_SUPPORTED_END_TO_END_QUBITS
            )
        elif int(self.max_enumeration_qubits) <= 0:
            raise ValueError("max_enumeration_qubits must be positive.")
        if int(self.max_enumeration_qubits) > MAX_SUPPORTED_END_TO_END_QUBITS:
            raise ValueError(
                "max_enumeration_qubits cannot exceed the supported optimization "
                f"maximum of {MAX_SUPPORTED_END_TO_END_QUBITS}."
            )
        if self.max_oracle_dense_qubits is None:
            object.__setattr__(
                self, "max_oracle_dense_qubits", MAX_SUPPORTED_END_TO_END_QUBITS
            )
        elif int(self.max_oracle_dense_qubits) <= 0:
            raise ValueError("max_oracle_dense_qubits must be positive.")
        if int(self.max_oracle_dense_qubits) > MAX_SUPPORTED_END_TO_END_QUBITS:
            raise ValueError(
                "max_oracle_dense_qubits cannot exceed the supported optimization "
                f"maximum of {MAX_SUPPORTED_END_TO_END_QUBITS}."
            )
        if candidate_workers is not None and (
            isinstance(candidate_workers, bool)
            or not isinstance(candidate_workers, (int, np.integer))
            or int(candidate_workers) <= 0
        ):
            raise ValueError("candidate_workers must be a positive integer when supplied.")
        if candidate_workers not in (None, 1):
            raise ValueError(
                "Nested parallelism is not exposed: candidate_workers is deprecated "
                "and candidate search is explicitly serial; use "
                "inner_enumeration_workers for deterministic shared-memory all-Pauli "
                "reductions."
            )
        fixed_error = bool(
            self.objective is not None
            and self.objective.mode is OptimizationMode.FIXED_ERROR_MIN_COPIES
        )
        seeds = (self.search_seed, *self.tuning_seeds, *self.holdout_seeds)
        for seed in seeds:
            if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
                raise ValueError("All RNG seeds must be nonnegative integers.")
        if not self.tuning_seeds:
            raise ValueError("Tuning seeds must be nonempty.")
        if not fixed_error and not self.holdout_seeds:
            raise ValueError(
                "Holdout seeds must be nonempty outside fixed_error_min_copies."
            )
        if len(set(self.tuning_seeds)) != len(self.tuning_seeds):
            raise ValueError("Tuning seeds must be unique.")
        if len(set(self.holdout_seeds)) != len(self.holdout_seeds):
            raise ValueError("Holdout seeds must be unique.")
        if set(self.tuning_seeds) & set(self.holdout_seeds):
            raise ValueError("Tuning and holdout seed sets must be disjoint.")
        counts = tuple(int(value) for value in self.halving_seed_counts)
        if not counts or any(value <= 0 for value in counts):
            raise ValueError("halving_seed_counts must contain positive integers.")
        if tuple(sorted(set(counts))) != counts:
            raise ValueError("halving_seed_counts must be strictly increasing.")
        if counts[-1] > len(self.tuning_seeds):
            raise ValueError("Halving schedule exceeds the tuning seed count.")
        for name in (
            "delta_grp_ordinary", "zeta_sgn", "zeta_tom",
            "fixed_budget_zeta_peel", "fixed_budget_zeta_rank",
        ):
            if not 0.0 < float(getattr(self, name)) < 1.0:
                raise ValueError(f"{name} must lie in (0,1).")
        for name in (
            "fixed_budget_tau_kappa_diagnostic",
            "fixed_budget_epsilon_tom_diagnostic",
        ):
            if not math.isfinite(float(getattr(self, name))) or float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not 0.0 < float(self.failure_loss) <= 1.0:
            raise ValueError("failure_loss must equal 1.0.")
        if float(self.failure_loss) != 1.0:
            raise ValueError("failure_loss is fixed at 1.0.")
        if not 0.0 < float(self.retention_fraction) <= 1.0:
            raise ValueError("retention_fraction must lie in (0,1].")
        if not 0.0 < float(self.target_copy_utilization) <= 1.0:
            raise ValueError("target_copy_utilization must lie in (0,1].")
        for name in (
            "max_score_array_bytes",
            "max_structured_bell_workspace_bytes",
            "max_enumeration_workspace_bytes",
            "max_preflight_estimated_copies",
            "max_predicted_grouping_copies",
            "max_predicted_tomography_copies",
            "max_predicted_total_copies",
            "max_single_grouping_query_shots",
            "fixed_error_initial_budget_hint",
            "fixed_error_hard_safety_budget",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
                raise ValueError(f"{name} must be positive or None.")
        if (
            not math.isfinite(self.enumeration_workspace_safety_factor)
            or self.enumeration_workspace_safety_factor < 1.0
        ):
            raise ValueError(
                "enumeration_workspace_safety_factor must be finite and at least 1."
            )
        if not math.isfinite(self.execution_safety_factor) or self.execution_safety_factor <= 0.0:
            raise ValueError("execution_safety_factor must be finite and positive.")
        if (
            not math.isfinite(float(self.fixed_error_budget_growth_factor))
            or float(self.fixed_error_budget_growth_factor) <= 1.0
        ):
            raise ValueError("fixed_error_budget_growth_factor must exceed 1.")
        if (
            not math.isfinite(float(self.fixed_error_budget_relative_tolerance))
            or not 0.0 < float(self.fixed_error_budget_relative_tolerance) < 1.0
        ):
            raise ValueError(
                "fixed_error_budget_relative_tolerance must lie in (0,1)."
            )
        if self.objective is not None:
            if not isinstance(self.objective, OptimizationObjective):
                raise TypeError("objective must be an OptimizationObjective or None.")
            if (
                self.objective.mode is OptimizationMode.FIXED_BUDGET_MIN_ERROR
                and int(self.objective.copy_ceiling) != int(self.total_copies)
            ):
                raise ValueError("objective.copy_ceiling must equal total_copies.")
        if self.simulation_backend not in ("legacy_shotwise", "batched_counts"):
            raise ValueError(
                "simulation_backend must be 'legacy_shotwise' or 'batched_counts'."
            )
        if self.checkpoint_path is not None:
            if not isinstance(self.checkpoint_path, str) or not self.checkpoint_path.strip():
                raise ValueError("checkpoint_path must be a nonempty path string or None.")
        if not isinstance(self.checkpoint_key, str):
            raise TypeError("checkpoint_key must be str.")
        if self.checkpoint_path is not None:
            if not self.checkpoint_key.strip():
                raise ValueError(
                    "checkpoint_key must be nonempty and uniquely identify the "
                    "target state/run whenever checkpointing is enabled."
                )
            if self.checkpoint_key != self.checkpoint_key.strip():
                raise ValueError("checkpoint_key must not have leading or trailing whitespace.")
        if self.resume_from_checkpoint and self.checkpoint_path is None:
            raise ValueError("resume_from_checkpoint requires checkpoint_path.")
        for name in (
            "tomography_refinement_enabled",
            "budget_refinement_enabled",
            "resume_from_checkpoint",
            "verbose",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool.")

    @property
    def effective_objective(self) -> OptimizationObjective:
        """Return the explicit objective, including legacy-call compatibility."""

        if self.objective is not None:
            return self.objective
        return default_fixed_budget_objective(
            self.total_copies,
            budget_utilization_target=self.target_copy_utilization,
        )


@dataclass(frozen=True)
class DerivedCandidate:
    parameters: OptimizationCandidate
    M1: int
    tau1: float
    zeta_peel: float
    h_min: float
    h_max: float
    eta_peel: float
    M2: int
    tau_rank: float
    zeta_rank: float
    theta: float
    theta_tau_multiplier: Optional[float]
    theta_over_tau_rank: float
    eta_test: float
    tau_kappa: float
    delta_grp_ordinary: float
    M_sgn: Optional[int]
    zeta_sgn: float
    epsilon_tom: float
    zeta_tom: float
    peeling_physical_copies: int
    recovery_physical_copies: int
    worst_case_sign_reservation: int
    execution_policy: str = ExecutionPolicy.STRICT.value
    fixed_budget_parameterization_schema: int = 5
    fixed_budget_stage_caps: Tuple[Tuple[str, int], ...] = ()
    normalized_stage_weights: Tuple[Tuple[str, float], ...] = ()
    physical_copy_budget: int = 0

    @property
    def mandatory_bell_copies(self) -> int:
        """Copies always requested by the two nonadaptive Bell stages."""

        return self.peeling_physical_copies + self.recovery_physical_copies

    @property
    def preflight_fixed_reservation(self) -> int:
        """Bell copies plus the worst-case (not realized-minimum) sign pool."""

        return self.mandatory_bell_copies + self.worst_case_sign_reservation


@dataclass(frozen=True)
class PreflightResourceCheck:
    """External resource screen; it is not an in-progress execution cap."""

    mandatory_bell_copies: int
    worst_case_sign_reservation: int
    preflight_fixed_reservation: int
    grouping_safety_estimate: int
    tomography_safety_estimate: int
    preflight_safety_estimate: int
    rigorous_grouping_upper_bound: int
    rigorous_tomography_upper_bound: int
    rigorous_total_upper_bound: int
    single_grouping_query_shots_estimate: int
    safety_limit: int
    runtime_safety_rejected: bool
    mathematical_budget_rejected: bool
    safe_to_execute: bool
    reason: str
    guarantee_level: str


NUMERIC_SAMPLING_SAFETY_POLICY_SCHEMA = 1


def backend_sampling_count_limit() -> int:
    """Return the conservative count limit shared by NumPy/C samplers.

    NumPy's multinomial implementation accepts a C ``long`` count while array
    sizes use ``intp``.  The smaller runtime-derived bound is therefore the
    portable representability limit for counts passed by the learner.
    """

    return int(min(np.iinfo(np.intp).max, np.iinfo(np.dtype("l")).max))


def unsafe_backend_sampling_quantities(
    derived: DerivedCandidate,
    *,
    total_copies: int,
) -> Tuple[Tuple[str, int], ...]:
    """List derived physical counts that cannot be represented by the backend."""

    quantities = [
        ("total_physical_budget", int(total_copies)),
        ("physical_copy_budget", int(derived.physical_copy_budget or total_copies)),
        ("M1_bell_pair_count", int(derived.M1)),
        ("M2_bell_pair_count", int(derived.M2)),
        ("peeling_physical_copies", int(derived.peeling_physical_copies)),
        ("recovery_physical_copies", int(derived.recovery_physical_copies)),
        ("worst_case_sign_reservation", int(derived.worst_case_sign_reservation)),
    ]
    quantities.extend(
        (f"stage_cap_{stage}", int(count))
        for stage, count in derived.fixed_budget_stage_caps
    )
    limit = backend_sampling_count_limit()
    return tuple(
        (name, value)
        for name, value in quantities
        if value < 0 or value > limit
    )


def _uniform(rng: np.random.Generator, bounds: Range) -> float:
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


def _log_uniform(rng: np.random.Generator, bounds: Range) -> float:
    return float(math.exp(rng.uniform(math.log(bounds[0]), math.log(bounds[1]))))


def sample_candidate(
    rng: np.random.Generator,
    search_space: SearchSpace,
    *,
    mode: Optional[OptimizationMode | str] = None,
    copy_ceiling: Optional[int] = None,
    d: int = 2,
) -> OptimizationCandidate:
    """Sample a mode-specific point without consulting state truth."""

    normalized_mode = None if mode is None else OptimizationMode(mode)
    fixed_budget = normalized_mode is OptimizationMode.FIXED_BUDGET_MIN_ERROR
    fixed_error = normalized_mode is OptimizationMode.FIXED_ERROR_MIN_COPIES

    if fixed_budget or fixed_error:
        h_min = _uniform(rng, search_space.h_min)
        h_max = _uniform(rng, search_space.h_max)
        common = dict(
            h_min=h_min,
            h_max=h_max,
            theta_tau_multiplier=_log_uniform(
                rng, search_space.theta_tau_multiplier
            ),
            eta_test=_log_uniform(rng, search_space.eta_test),
            peel_weight=_log_uniform(rng, search_space.peel_weight),
            recovery_weight=_log_uniform(rng, search_space.recovery_weight),
            grouping_weight=_log_uniform(rng, search_space.grouping_weight),
            syndrome_weight=_log_uniform(rng, search_space.syndrome_weight),
            tomography_weight=_log_uniform(rng, search_space.tomography_weight),
        )
        if fixed_budget:
            return FixedBudgetCandidateParameters(**common)
        return FixedErrorCandidateParameters(**common)

    return CandidateParameters(
        alpha_peel=_log_uniform(rng, search_space.alpha_peel),
        alpha_rank=_log_uniform(rng, search_space.alpha_rank),
        alpha_sgn=_log_uniform(rng, search_space.alpha_sgn),
        c_peel=_uniform(rng, search_space.c_peel),
        c_rank=_uniform(rng, search_space.c_rank),
        h_min=_uniform(rng, search_space.h_min),
        h_span=_uniform(rng, search_space.h_span),
        theta=_uniform(rng, search_space.theta),
        eta_test=_log_uniform(rng, search_space.eta_test),
        kappa_ratio=_uniform(rng, search_space.kappa_ratio),
        epsilon_tom=_log_uniform(rng, search_space.epsilon_tom),
        peel_weight=_log_uniform(rng, search_space.peel_weight),
        recovery_weight=_log_uniform(rng, search_space.recovery_weight),
        grouping_weight=_log_uniform(rng, search_space.grouping_weight),
        syndrome_weight=_log_uniform(rng, search_space.syndrome_weight),
        tomography_weight=_log_uniform(rng, search_space.tomography_weight),
    )


def _derive_concentration(
    *, rounds: int, n: int, scale: float, label: str
) -> Tuple[float, float]:
    if scale <= 1.0 or not math.isfinite(scale):
        raise InvalidCandidateError(f"{label} concentration scale must exceed 1.")
    log_a = math.log(2.0) + n * math.log(4.0)
    tau = float(scale * math.sqrt(2.0 * log_a / rounds))
    log_zeta = (1.0 - scale * scale) * log_a
    zeta = float(math.exp(log_zeta))
    if not 0.0 < zeta < 1.0:
        raise InvalidCandidateError(f"{label} implied zeta is outside (0,1).")
    reconstructed = bell_score_uniform_radius(rounds, n, zeta)
    if not math.isclose(reconstructed, tau, rel_tol=2e-13, abs_tol=2e-15):
        raise InvalidCandidateError(f"{label} tau/zeta inversion failed.")
    return tau, zeta


def _d1_fixed_budget_caps(
    total_copies: int,
    candidate: OptimizationCandidate,
) -> Tuple[Tuple[Tuple[str, int], ...], Tuple[Tuple[str, float], ...]]:
    """Allocate d=1 over its four active stages; grouping is structural only."""

    active = (
        ("peeling", float(candidate.peel_weight)),
        ("recovery", float(candidate.recovery_weight)),
        ("syndrome", float(candidate.syndrome_weight)),
        ("tomography", float(candidate.tomography_weight)),
    )
    total_weight = sum(weight for _stage, weight in active)
    fractions = {stage: weight / total_weight for stage, weight in active}
    peeling = int(math.floor(total_copies * fractions["peeling"]))
    recovery = int(math.floor(total_copies * fractions["recovery"]))
    syndrome = int(math.floor(total_copies * fractions["syndrome"]))
    tomography = int(total_copies) - peeling - recovery - syndrome
    caps = (
        ("peeling", peeling),
        ("recovery", recovery),
        ("grouping", 0),
        ("syndrome", syndrome),
        ("tomography", tomography),
    )
    normalized = (
        ("peeling", fractions["peeling"]),
        ("recovery", fractions["recovery"]),
        ("grouping", 0.0),
        ("syndrome", fractions["syndrome"]),
        ("tomography", fractions["tomography"]),
    )
    if sum(cap for _stage, cap in caps) != int(total_copies):
        raise RuntimeError("d=1 fixed-budget caps do not sum to total_copies.")
    return caps, normalized


def minimum_practical_copy_budget(
    candidate: FixedBudgetCandidateParameters | FixedErrorCandidateParameters,
    *,
    d: int,
) -> int:
    """Return the smallest integer budget with nonempty practical stage pools."""

    if isinstance(d, bool) or not isinstance(d, (int, np.integer)) or int(d) <= 0:
        raise ValueError("d must be a positive integer.")
    weights = FixedBudgetStageWeights(
        peeling=candidate.peel_weight,
        recovery=candidate.recovery_weight,
        grouping=candidate.grouping_weight,
        syndrome=candidate.syndrome_weight,
        tomography=candidate.tomography_weight,
    )

    def sufficient(total: int) -> bool:
        caps = (
            _d1_fixed_budget_caps(total, candidate)[0]
            if int(d) == 1
            else fixed_budget_nominal_stage_caps(total, weights)
        )
        cap_map = dict(caps)
        return bool(
            cap_map["peeling"] >= 2
            and cap_map["recovery"] >= 2
            and cap_map["syndrome"] >= 1
            and cap_map["tomography"] >= (3 if int(d) == 1 else 1)
            and (int(d) == 1 or cap_map["grouping"] >= 1)
        )

    high = 1
    while not sufficient(high):
        high *= 2
        if high > 2**62:
            raise InvalidCandidateError(
                "Unable to derive a finite practical minimum copy budget."
            )
    low = 0
    while high - low > 1:
        middle = (low + high) // 2
        if sufficient(middle):
            high = middle
        else:
            low = middle
    return int(high)


def derive_candidate(
    candidate: OptimizationCandidate,
    *,
    n: int,
    d: int,
    total_copies: int,
    optimization_config: OptimizationConfig,
    physical_budget: Optional[int] = None,
) -> DerivedCandidate:
    """Derive all learner settings and reject invalid/budget-impossible points."""

    if n <= 0 or d <= 0 or total_copies <= 0:
        raise ValueError("n, d, and total_copies must be positive.")
    if int(n) > int(optimization_config.max_enumeration_qubits):
        raise ValueError(
            f"n={n} exceeds max_enumeration_qubits="
            f"{optimization_config.max_enumeration_qubits}."
        )
    objective_mode = optimization_config.effective_objective.mode
    fixed_budget = objective_mode is OptimizationMode.FIXED_BUDGET_MIN_ERROR
    fixed_error = objective_mode is OptimizationMode.FIXED_ERROR_MIN_COPIES
    if not fixed_error and int(total_copies) != int(optimization_config.total_copies):
        raise ValueError("total_copies disagrees with OptimizationConfig.total_copies.")
    practical = fixed_budget or fixed_error
    if d == 1 and not practical:
        raise NotImplementedError(
            "d=1 optimization is supported only for fixed_budget_min_error and "
            "fixed_error_min_copies; "
            "theorem-oriented/strict d=1 optimization remains unsupported."
        )
    legacy_fixed_candidate = practical and isinstance(
        candidate, CandidateParameters
    )
    if fixed_error and not isinstance(
        candidate, (CandidateParameters, FixedErrorCandidateParameters)
    ):
        raise InvalidCandidateError(
            "fixed_error_min_copies requires a practical or legacy-compatible candidate."
        )
    if not practical and not isinstance(candidate, CandidateParameters):
        raise InvalidCandidateError(
            "The theorem-oriented objective requires CandidateParameters."
        )
    values = tuple(candidate.__dict__.values())
    if not all(math.isfinite(float(value)) for value in values):
        raise InvalidCandidateError("Candidate values must all be finite.")
    if not practical and not (
        candidate.alpha_peel > 0
        and candidate.alpha_rank > 0
        and candidate.alpha_sgn > 0
    ):
        raise InvalidCandidateError("Allocation fractions must be positive.")
    stage_weights = FixedBudgetStageWeights(
        peeling=candidate.peel_weight,
        recovery=candidate.recovery_weight,
        grouping=candidate.grouping_weight,
        syndrome=candidate.syndrome_weight,
        tomography=candidate.tomography_weight,
    )
    if physical_budget is not None and (
        isinstance(physical_budget, bool)
        or not isinstance(physical_budget, (int, np.integer))
        or int(physical_budget) <= 0
    ):
        raise InvalidCandidateError("physical_budget must be a positive integer.")
    if fixed_error:
        physical_budget = int(
            total_copies if physical_budget is None else physical_budget
        )
    elif physical_budget is not None and int(physical_budget) != int(total_copies):
        raise ValueError("physical_budget is only variable in fixed_error_min_copies.")
    else:
        physical_budget = int(total_copies)
    if practical and d == 1:
        stage_caps, normalized_stage_weights = _d1_fixed_budget_caps(
            physical_budget, candidate
        )
    else:
        stage_caps = fixed_budget_nominal_stage_caps(physical_budget, stage_weights)
        normalized_stage_weights = stage_weights.normalized()
    cap_map = dict(stage_caps)
    if practical:
        M1 = cap_map["peeling"] // 2
        M2 = cap_map["recovery"] // 2
        M_sgn = None
        if M1 < 1 or M2 < 1:
            raise InvalidCandidateError(
                "Normalized fixed-budget Bell caps leave an empty required stage."
            )
    else:
        M1 = max(optimization_config.minimum_M1, int(math.floor(candidate.alpha_peel * total_copies / 2.0)))
        M2 = max(optimization_config.minimum_M2, int(math.floor(candidate.alpha_rank * total_copies / 2.0)))
        M_sgn = max(
            optimization_config.minimum_M_sgn,
            int(math.floor(candidate.alpha_sgn * total_copies / max(1, n))),
        )
    if practical:
        zeta_peel = float(optimization_config.fixed_budget_zeta_peel)
        zeta_rank = float(optimization_config.fixed_budget_zeta_rank)
        tau1 = bell_score_uniform_radius(M1, n, zeta_peel)
        tau_rank = bell_score_uniform_radius(M2, n, zeta_rank)
        if legacy_fixed_candidate:
            theta_tau_multiplier_value = float(candidate.theta / tau_rank)
            practical_values = dict(
                h_min=float(candidate.h_min),
                h_max=float(candidate.h_max),
                theta_tau_multiplier=theta_tau_multiplier_value,
                eta_test=float(candidate.eta_test),
                peel_weight=float(candidate.peel_weight),
                recovery_weight=float(candidate.recovery_weight),
                grouping_weight=float(candidate.grouping_weight),
                syndrome_weight=float(candidate.syndrome_weight),
                tomography_weight=float(candidate.tomography_weight),
            )
            candidate = (
                FixedErrorCandidateParameters(**practical_values)
                if fixed_error
                else FixedBudgetCandidateParameters(**practical_values)
            )
        if candidate.theta_tau_multiplier <= 0.0:
            raise InvalidCandidateError("theta_tau_multiplier must be positive.")
        theta_tau_multiplier: Optional[float] = float(
            candidate.theta_tau_multiplier
        )
        theta_effective = min(0.50, theta_tau_multiplier * tau_rank)
    else:
        tau1, zeta_peel = _derive_concentration(
            rounds=M1, n=n, scale=candidate.c_peel, label="peeling"
        )
        tau_rank, zeta_rank = _derive_concentration(
            rounds=M2, n=n, scale=candidate.c_rank, label="recovery"
        )
        theta_tau_multiplier = None
        theta_effective = float(candidate.theta)

    h_min, h_max = float(candidate.h_min), float(candidate.h_max)
    if not 0.5 < h_min < h_max < 1.0:
        raise InvalidCandidateError("Require 1/2 < h_min < h_max < 1.")
    if not practical and h_max >= 0.995:
        raise InvalidCandidateError("The strict parameterization requires h_max < 0.995.")
    if not practical and h_min + tau1 >= 1.0:
        raise InvalidCandidateError("h_min + tau1 must be below one.")
    eta_peel = (h_max - h_min) / optimization_config.peeling_grid_intervals
    if not 0.0 < theta_effective < 1.0:
        raise InvalidCandidateError("theta must lie in (0,1).")
    if not practical and theta_effective <= tau_rank:
        raise InvalidCandidateError("theta must exceed tau_rank.")
    if candidate.eta_test <= 0.0:
        raise InvalidCandidateError("eta_test must be positive.")
    if practical:
        tau_kappa = float(optimization_config.fixed_budget_tau_kappa_diagnostic)
        epsilon_tom = float(optimization_config.fixed_budget_epsilon_tom_diagnostic)
    else:
        tau_kappa = candidate.kappa_ratio * candidate.eta_test
        if not (candidate.kappa_ratio > 0.0 and tau_kappa > 0.0):
            raise InvalidCandidateError("Require kappa_ratio > 0 and tau_kappa > 0.")
        if candidate.epsilon_tom <= 0.0:
            raise InvalidCandidateError("epsilon_tom must be positive.")
        epsilon_tom = float(candidate.epsilon_tom)

    derived = DerivedCandidate(
        parameters=candidate,
        M1=M1,
        tau1=tau1,
        zeta_peel=zeta_peel,
        h_min=h_min,
        h_max=h_max,
        eta_peel=eta_peel,
        M2=M2,
        tau_rank=tau_rank,
        zeta_rank=zeta_rank,
        theta=theta_effective,
        theta_tau_multiplier=theta_tau_multiplier,
        theta_over_tau_rank=float(theta_effective / tau_rank),
        eta_test=float(candidate.eta_test),
        tau_kappa=float(tau_kappa),
        delta_grp_ordinary=float(optimization_config.delta_grp_ordinary),
        M_sgn=M_sgn,
        zeta_sgn=float(optimization_config.zeta_sgn),
        epsilon_tom=epsilon_tom,
        zeta_tom=float(optimization_config.zeta_tom),
        peeling_physical_copies=2 * M1,
        recovery_physical_copies=2 * M2,
        worst_case_sign_reservation=(0 if M_sgn is None else n * M_sgn),
        execution_policy=(
            ExecutionPolicy.FIXED_BUDGET_GRACEFUL.value
            if practical
            else ExecutionPolicy.STRICT.value
        ),
        fixed_budget_stage_caps=stage_caps if practical else (),
        normalized_stage_weights=normalized_stage_weights if practical else (),
        physical_copy_budget=physical_budget,
    )
    return derived


def preflight_resource_check(
    derived: DerivedCandidate,
    *,
    n: int,
    d: int,
    total_copies: int,
    optimization_config: OptimizationConfig,
) -> PreflightResourceCheck:
    """Compute rigorous bounds plus a practical, explicitly non-hard estimate."""

    if int(n) > int(optimization_config.max_enumeration_qubits):
        raise ValueError(
            f"n={n} exceeds max_enumeration_qubits="
            f"{optimization_config.max_enumeration_qubits}."
        )
    graceful_fixed_budget = (
        optimization_config.effective_objective.mode
        in (
            OptimizationMode.FIXED_BUDGET_MIN_ERROR,
            OptimizationMode.FIXED_ERROR_MIN_COPIES,
        )
        and derived.execution_policy == ExecutionPolicy.FIXED_BUDGET_GRACEFUL.value
    )
    if d == 1:
        if not graceful_fixed_budget:
            raise NotImplementedError(
                "d=1 preflight is supported only for the practical graceful modes; "
                "theorem-oriented/strict d=1 preflight remains unsupported."
            )
        cap_map = dict(derived.fixed_budget_stage_caps)
        canonical_stages = {
            "peeling", "recovery", "grouping", "syndrome", "tomography"
        }
        malformed = bool(
            set(cap_map) != canonical_stages
            or sum(cap_map.values()) != int(total_copies)
            or cap_map.get("grouping") != 0
            or derived.peeling_physical_copies > cap_map.get("peeling", -1)
            or derived.recovery_physical_copies > cap_map.get("recovery", -1)
        )
        insufficient = bool(
            not malformed
            and (
                cap_map["peeling"] < 2
                or cap_map["recovery"] < 2
                or cap_map["syndrome"] < 1
                or cap_map["tomography"] < 3
            )
        )
        safe = not malformed and not insufficient
        reason = (
            "malformed_d1_fixed_budget_stage_allocation"
            if malformed
            else (
                "d1_fixed_budget_stage_allocation_below_primitive_minimum"
                if insufficient
                else "passed_d1_graceful_fixed_budget_parameter_and_runtime_screen"
            )
        )
        fixed = derived.preflight_fixed_reservation
        tomography_cap = max(0, int(cap_map.get("tomography", 0)))
        return PreflightResourceCheck(
            mandatory_bell_copies=derived.mandatory_bell_copies,
            worst_case_sign_reservation=derived.worst_case_sign_reservation,
            preflight_fixed_reservation=fixed,
            grouping_safety_estimate=0,
            tomography_safety_estimate=tomography_cap,
            preflight_safety_estimate=int(total_copies),
            rigorous_grouping_upper_bound=0,
            rigorous_tomography_upper_bound=tomography_cap,
            rigorous_total_upper_bound=int(total_copies),
            single_grouping_query_shots_estimate=0,
            safety_limit=int(total_copies),
            runtime_safety_rejected=False,
            mathematical_budget_rejected=not safe,
            safe_to_execute=safe,
            reason=reason,
            guarantee_level=(
                "d=1 explicit four-stage caps rigorously bound practical execution; "
                "grouping is identically zero and conditional tomography spends "
                "at most its physical cap"
            ),
        )
    fixed = derived.preflight_fixed_reservation
    rigorous_tests = simplified_cumulant_test_bound(n, d)
    rigorous_grouping = ordinary_grouping_copy_upper_bound(
        rigorous_tests,
        d,
        derived.tau_kappa,
        derived.delta_grp_ordinary,
    )

    def tomography_pool(register_count: int, register_size: int) -> int:
        budget = register_tomography_budget(
            (0,),
            tuple(range(register_size)),
            derived.epsilon_tom / register_count,
            derived.zeta_tom / register_count,
        )
        return int(budget.L_C)

    rigorous_tomography = max(
        tomography_pool(K, min(d, n - K + 1))
        for K in range(1, n + 1)
    )
    estimated_L = min(n, max(2, d))
    estimated_tests = adaptive_cumulant_test_bound(estimated_L, d)
    grouping_estimate = ordinary_grouping_copy_upper_bound(
        estimated_tests,
        d,
        derived.tau_kappa,
        derived.delta_grp_ordinary,
    )
    estimated_K = max(1, math.ceil(n / d))
    estimated_k = min(d, n - estimated_K + 1)
    tomography_estimate = tomography_pool(estimated_K, estimated_k)
    safety_estimate = fixed + grouping_estimate + tomography_estimate
    rigorous_total = fixed + rigorous_grouping + rigorous_tomography
    legacy_safety_limit = (
        int(optimization_config.max_preflight_estimated_copies)
        if optimization_config.max_preflight_estimated_copies is not None
        else int(math.floor(optimization_config.execution_safety_factor * total_copies))
    )
    total_safety_limit = min(
        legacy_safety_limit,
        int(optimization_config.max_predicted_total_copies)
        if optimization_config.max_predicted_total_copies is not None
        else legacy_safety_limit,
    )
    single_query_estimate = ordinary_cumulant_sample_count(
        max(1, min(d, estimated_L)),
        derived.tau_kappa,
        derived.delta_grp_ordinary / max(1, estimated_tests),
    )
    cap_map = dict(derived.fixed_budget_stage_caps)
    malformed_graceful_allocation = bool(
        graceful_fixed_budget
        and (
            set(cap_map) != {"peeling", "recovery", "grouping", "syndrome", "tomography"}
            or sum(cap_map.values()) != int(total_copies)
            or derived.peeling_physical_copies > cap_map.get("peeling", -1)
            or derived.recovery_physical_copies > cap_map.get("recovery", -1)
            or derived.worst_case_sign_reservation > cap_map.get("syndrome", -1)
        )
    )
    if malformed_graceful_allocation:
        safe = False
        reason = "malformed_fixed_budget_stage_allocation"
        runtime_rejected = False
        budget_rejected = True
    elif graceful_fixed_budget:
        safe = True
        reason = "passed_graceful_fixed_budget_parameter_and_runtime_screen"
        runtime_rejected = False
        budget_rejected = False
    elif fixed >= total_copies:
        safe = False
        reason = "preflight_fixed_reservation_exhausts_or_exceeds_total_budget"
        runtime_rejected = False
        budget_rejected = True
    elif (
        optimization_config.max_predicted_grouping_copies is not None
        and grouping_estimate > optimization_config.max_predicted_grouping_copies
    ):
        safe = False
        reason = "predicted_grouping_copies_exceed_runtime_safety_limit"
        runtime_rejected = True
        budget_rejected = False
    elif (
        optimization_config.max_predicted_tomography_copies is not None
        and tomography_estimate > optimization_config.max_predicted_tomography_copies
    ):
        safe = False
        reason = "predicted_tomography_copies_exceed_runtime_safety_limit"
        runtime_rejected = True
        budget_rejected = False
    elif (
        optimization_config.max_single_grouping_query_shots is not None
        and single_query_estimate > optimization_config.max_single_grouping_query_shots
    ):
        safe = False
        reason = "single_grouping_query_shots_exceed_runtime_safety_limit"
        runtime_rejected = True
        budget_rejected = False
    elif safety_estimate > total_safety_limit:
        safe = False
        reason = "predicted_total_copies_exceed_runtime_safety_limit"
        runtime_rejected = True
        budget_rejected = False
    else:
        safe = True
        reason = "passed_external_preflight_safety_screen"
        runtime_rejected = False
        budget_rejected = False
    return PreflightResourceCheck(
        mandatory_bell_copies=derived.mandatory_bell_copies,
        worst_case_sign_reservation=derived.worst_case_sign_reservation,
        preflight_fixed_reservation=fixed,
        grouping_safety_estimate=grouping_estimate,
        tomography_safety_estimate=tomography_estimate,
        preflight_safety_estimate=safety_estimate,
        rigorous_grouping_upper_bound=rigorous_grouping,
        rigorous_tomography_upper_bound=rigorous_tomography,
        rigorous_total_upper_bound=rigorous_total,
        single_grouping_query_shots_estimate=single_query_estimate,
        safety_limit=total_safety_limit,
        runtime_safety_rejected=runtime_rejected,
        mathematical_budget_rejected=budget_rejected,
        safe_to_execute=safe,
        reason=reason,
        guarantee_level=(
            "graceful normalized stage caps rigorously bound actual execution; "
            "nominal tolerance copy estimates are diagnostic only"
            if graceful_fixed_budget
            else "fixed count and worst-case bounds are rigorous; safety estimate is "
            "a configurable computational guard, not a stage-wise execution cap"
        ),
    )


def predict_tomography_pool_copies(
    register_sizes: Tuple[int, ...] | list[int],
    epsilon_tom: float,
    zeta_tom: float,
) -> int:
    """Predict the exact simultaneous Phase-6 physical-copy pool.

    This mirrors ``main_v2``'s uniform per-register allocations and returns
    ``max_C L_C`` because disjoint registers are measured simultaneously.
    """

    sizes = tuple(int(size) for size in register_sizes)
    if any(size <= 0 for size in sizes):
        raise ValueError("register_sizes must contain positive integers.")
    if not math.isfinite(float(epsilon_tom)) or epsilon_tom <= 0.0:
        raise ValueError("epsilon_tom must be finite and positive.")
    if not 0.0 < float(zeta_tom) < 1.0:
        raise ValueError("zeta_tom must lie in (0,1).")
    register_count = len(sizes)
    if register_count == 0:
        return 0
    local_epsilon = float(epsilon_tom) / register_count
    local_zeta = float(zeta_tom) / register_count
    return max(
        int(
            register_tomography_budget(
                (index,), tuple(range(size)), local_epsilon, local_zeta
            ).L_C
        )
        for index, size in enumerate(sizes)
    )


def _allocation_fraction_for_exact_floor(
    target_rounds: int,
    *,
    total_copies: int,
    physical_copies_per_round: int,
) -> float:
    """Choose the midpoint of the open-safe interval mapping to one floor."""

    if isinstance(target_rounds, bool) or int(target_rounds) <= 0:
        raise ValueError("Manual round counts must be positive integers.")
    scale = float(total_copies) / float(physical_copies_per_round)
    alpha = (float(target_rounds) + 0.5) / scale
    if int(math.floor(alpha * scale)) != int(target_rounds):
        raise RuntimeError("Failed to construct a robust exact-floor allocation fraction.")
    return float(alpha)


def _concentration_scale_from_zeta(*, n: int, zeta: float, label: str) -> float:
    if not (math.isfinite(zeta) and 0.0 < zeta < 1.0):
        raise ValueError(f"{label} zeta must lie in (0,1).")
    log_a = math.log(2.0) + int(n) * math.log(4.0)
    scale_squared = 1.0 - math.log(float(zeta)) / log_a
    scale = math.sqrt(scale_squared)
    if not scale > 1.0:
        raise ValueError(f"{label} concentration scale must exceed one.")
    return float(scale)


def candidate_from_manual_configs(
    *,
    n: int,
    d: int,
    total_copies: int,
    peeling_config: PeelingConfig,
    recovery_config: RecoveryConfig,
    grouping_config: GroupingConfig,
    syndrome_config: SyndromeConfig,
    tomography_config: TomographyConfig,
) -> CandidateParameters:
    """Serialize representable legacy manual configs without consulting truth.

    Allocation fractions are placed at the midpoint of the interval whose
    floor reproduces each manual count.  When supplied to a practical objective,
    ``derive_candidate`` canonicalizes this compatibility object to the shared
    normalized-stage parameterization; its theorem-only fields do not remain
    operational search coordinates.
    """

    if n <= 0 or d <= 0 or total_copies <= 0:
        raise ValueError("n, d, and total_copies must be positive.")
    if d == 1:
        raise NotImplementedError(
            "Optimization V1 supports d>=2; d=1 requires a separate conditional "
            "one-qubit tomography budget parameterization."
        )
    expected_eta = (
        float(peeling_config.h_max) - float(peeling_config.h_min)
    ) / OptimizationConfig(total_copies=total_copies).peeling_grid_intervals
    if not math.isclose(float(peeling_config.eta), expected_eta, rel_tol=1e-12, abs_tol=1e-15):
        raise InvalidCandidateError(
            "Manual peeling eta is not representable by the optimizer grid."
        )
    if grouping_config.ell_grp != d:
        raise InvalidCandidateError("Manual ell_grp must equal d.")
    if not math.isclose(
        float(syndrome_config.h_min), float(peeling_config.h_min),
        rel_tol=0.0, abs_tol=1e-15,
    ):
        raise InvalidCandidateError("Manual syndrome and peeling h_min values differ.")
    if grouping_config.eta_test <= 0.0 or grouping_config.tau_kappa <= 0.0:
        raise InvalidCandidateError("Manual empirical grouping thresholds must be positive.")
    if tomography_config.epsilon_tom <= 0.0:
        raise InvalidCandidateError("Manual epsilon_tom must be positive.")

    defaults = OptimizationConfig(total_copies=total_copies)
    fixed_values = (
        ("delta_grp_ordinary", grouping_config.delta_grp_ordinary, defaults.delta_grp_ordinary),
        ("zeta_sgn", syndrome_config.zeta_sgn, defaults.zeta_sgn),
        ("zeta_tom", tomography_config.zeta_tom, defaults.zeta_tom),
    )
    for name, actual, expected in fixed_values:
        if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-15):
            raise InvalidCandidateError(
                f"Manual {name}={actual} differs from the optimizer-fixed value {expected}."
            )

    manual_peel_cap = 2 * int(peeling_config.M1)
    manual_recovery_cap = 2 * int(recovery_config.M2)
    manual_syndrome_cap = n * int(syndrome_config.M_sgn)
    manual_tomography_weight = (
        total_copies
        - manual_peel_cap
        - manual_recovery_cap
        - manual_syndrome_cap
        - 1.0
    )
    if manual_tomography_weight <= 0.0:
        raise InvalidCandidateError("Manual fixed stages leave no normalized tomography weight.")
    candidate = CandidateParameters(
        alpha_peel=_allocation_fraction_for_exact_floor(
            peeling_config.M1,
            total_copies=total_copies,
            physical_copies_per_round=2,
        ),
        alpha_rank=_allocation_fraction_for_exact_floor(
            recovery_config.M2,
            total_copies=total_copies,
            physical_copies_per_round=2,
        ),
        alpha_sgn=_allocation_fraction_for_exact_floor(
            syndrome_config.M_sgn,
            total_copies=total_copies,
            physical_copies_per_round=n,
        ),
        c_peel=_concentration_scale_from_zeta(
            n=n, zeta=float(peeling_config.zeta_bs), label="peeling"
        ),
        c_rank=_concentration_scale_from_zeta(
            n=n, zeta=float(recovery_config.zeta_rank), label="recovery"
        ),
        h_min=float(peeling_config.h_min),
        h_span=float(peeling_config.h_max - peeling_config.h_min),
        theta=float(recovery_config.theta),
        eta_test=float(grouping_config.eta_test),
        kappa_ratio=float(grouping_config.tau_kappa / grouping_config.eta_test),
        epsilon_tom=float(tomography_config.epsilon_tom),
        peel_weight=manual_peel_cap + 0.25,
        recovery_weight=manual_recovery_cap + 0.25,
        grouping_weight=0.25,
        syndrome_weight=manual_syndrome_cap + 0.25,
        tomography_weight=manual_tomography_weight,
    )
    return candidate


def candidate_to_end_to_end_config(
    derived: DerivedCandidate,
    *,
    d: int,
    learner_seed: int,
    total_copies: int,
    optimization_config: OptimizationConfig,
    return_details: bool = False,
) -> EndToEndConfig:
    """Build only public main_v2 configs with explicit manual-override flags."""

    physical_budget = int(derived.physical_copy_budget or total_copies)

    enumeration = EnumerationExecutionConfig(
        workers=optimization_config.inner_enumeration_workers,
        max_score_array_bytes=optimization_config.max_score_array_bytes,
        max_structured_bell_workspace_bytes=(
            optimization_config.max_structured_bell_workspace_bytes
        ),
        max_enumeration_workspace_bytes=(
            optimization_config.max_enumeration_workspace_bytes
        ),
        enumeration_workspace_safety_factor=(
            optimization_config.enumeration_workspace_safety_factor
        ),
    )
    peeling = PeelingConfig(
        h_min=derived.h_min,
        h_max=derived.h_max,
        eta=derived.eta_peel,
        M1=derived.M1,
        zeta_bs=derived.zeta_peel,
        return_details=return_details,
        max_enumeration_qubits=int(optimization_config.max_enumeration_qubits),
        enumeration_execution=enumeration,
        materialize_dense_clifford=False,
        max_dense_debug_qubits=int(optimization_config.max_oracle_dense_qubits),
    )
    recovery = RecoveryConfig(
        theta=derived.theta,
        M2=derived.M2,
        zeta_rank=derived.zeta_rank,
        max_enumeration_qubits=int(optimization_config.max_enumeration_qubits),
        enumeration_execution=enumeration,
        allow_uncalibrated_peeling=True,
        allow_margin_failure=True,
        return_details=return_details,
    )
    grouping = GroupingConfig(
        ell_grp=d,
        eta_test=derived.eta_test,
        tau_kappa=derived.tau_kappa,
        delta_grp_ordinary=derived.delta_grp_ordinary,
        allow_uncalibrated_recovery=True,
        allow_no_false_merge_margin_failure=True,
        sampling_policy=(
            GroupingSamplingPolicy.FIXED_BUDGET.value
            if derived.execution_policy
            == ExecutionPolicy.FIXED_BUDGET_GRACEFUL.value
            else GroupingSamplingPolicy.CERTIFIED_TAU.value
        ),
        return_details=return_details,
    )
    syndrome = SyndromeConfig(
        zeta_sgn=derived.zeta_sgn,
        h_min=derived.h_min,
        M_sgn=derived.M_sgn,
        return_details=return_details,
    )
    tomography = TomographyConfig(
        epsilon_tom=derived.epsilon_tom,
        zeta_tom=derived.zeta_tom,
        allow_uncertified_localization=True,
        max_dense_qubits=int(optimization_config.max_oracle_dense_qubits),
        materialize_localized_estimator=False,
        return_details=return_details,
    )
    # epsilon/delta remain mandatory schedule metadata, but every execution
    # stage is explicitly overridden by practical, non-certified settings.
    return EndToEndConfig(
        epsilon=0.5,
        delta=0.2,
        seed=int(learner_seed),
        return_details=return_details,
        materialize_dense_estimator=False,
        max_dense_qubits=optimization_config.max_dense_qubits,
        max_enumeration_qubits=int(optimization_config.max_enumeration_qubits),
        max_dense_debug_qubits=int(optimization_config.max_oracle_dense_qubits),
        enumeration_execution=enumeration,
        max_reserved_copies=physical_budget,
        max_realized_copies=physical_budget,
        simulation_backend=optimization_config.simulation_backend,
        execution_policy=derived.execution_policy,
        fixed_budget_stage_weights=(
            FixedBudgetStageWeights(
                peeling=derived.parameters.peel_weight,
                recovery=derived.parameters.recovery_weight,
                grouping=derived.parameters.grouping_weight,
                syndrome=derived.parameters.syndrome_weight,
                tomography=derived.parameters.tomography_weight,
            )
            if derived.execution_policy
            == ExecutionPolicy.FIXED_BUDGET_GRACEFUL.value
            else None
        ),
        fixed_budget_stage_caps=(
            tuple(derived.fixed_budget_stage_caps)
            if d == 1
            and derived.execution_policy
            == ExecutionPolicy.FIXED_BUDGET_GRACEFUL.value
            else None
        ),
        allow_uncertified_execution=True,
        peeling_override=peeling,
        recovery_override=recovery,
        grouping_override=grouping,
        syndrome_override=syndrome,
        tomography_override=tomography,
    )
