"""General-block CEBP simulator through calibrated end-to-end tomography.

Phase 1 supplies the state-model boundary: a product of arbitrary latent states
on blocks of size at most ``d``, followed by an unknown Clifford encoding.
Phase 2 adds the manuscript's certified stabilizer-peeling stage.  Phase 3
adds full-Pauli rank-guided recovery of visible singleton/triple sectors.
Phase 4 adds signed residual moments and exact/ordinary empirical hierarchical
mixed-cumulant grouping.  Phase 5 adds deterministic simultaneous symplectic
localization and disjoint empirical-register allocation.  Phase 6 adds fresh
signed syndrome measurements and joint tomography of those disjoint registers.
Phase 7 composes the calibrated five-pool learner, compact physical decoder,
specialized conditional d=1 path, and learner-visible theorem certificate.

The existing :mod:`main` module remains the source of validated low-level
Clifford and QuTiP primitives.  Oracle-only simulation truth is nested under
``CEBPInstance.oracle_truth`` and is omitted from ``CEBPLearnerView``.
"""

from __future__ import annotations

import itertools
import math
import time
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import lru_cache, wraps
from types import MappingProxyType
from typing import Callable, Iterable, Literal, Mapping, Optional, Protocol, Sequence, Tuple, Union

import numpy as np
import qutip as qt

import main as _v1
from cebp_compact import (
    SignedClifford,
    StreamingGF2Basis,
    StructuredCEBPState,
    bell_score_sums_from_counts,
    bell_score_sums_from_outcomes,
    pauli_index_to_string,
    pauli_index_to_symplectic_int,
    pauli_string_to_index,
    threshold_span_reduction,
)


RngSeed = Optional[int]
StateLike = Union[np.ndarray, qt.Qobj]
SimulationBackend = Literal["legacy_shotwise", "batched_counts"]
MAX_SUPPORTED_END_TO_END_QUBITS = 12
_MAX_P2_ENUMERATION_QUBITS = MAX_SUPPORTED_END_TO_END_QUBITS
_MAX_P3_ENUMERATION_QUBITS = MAX_SUPPORTED_END_TO_END_QUBITS
_MAX_BELL_SCORE_CACHE_QUBITS = 10
_MAX_P4_GENERATED_GROUP_SIZE = 4**6


@dataclass(frozen=True)
class EnumerationExecutionConfig:
    """Deterministic controls for exhaustive all-Pauli post-processing.

    Candidate search remains explicitly serial to avoid duplicated ``4**n``
    workspaces. Standalone and optimization callers may opt into native/threaded
    fixed-threshold reductions here. ``max_score_array_bytes`` and
    ``max_structured_bell_workspace_bytes`` are backward-compatible hard caps
    for those individual resources. ``max_enumeration_workspace_bytes`` is the
    preferred modeled peak cap across known simultaneously-live Bell,
    score-transform, survivor, and ranking buffers. The safety factor reserves
    space for NumPy/Numba scratch that cannot be measured exactly in advance.
    """

    workers: int = 1
    chunk_size: Optional[int] = None
    max_score_array_bytes: Optional[int] = None
    max_structured_bell_workspace_bytes: Optional[int] = None
    max_enumeration_workspace_bytes: Optional[int] = None
    enumeration_workspace_safety_factor: float = 1.25
    benchmark: bool = False

    def __post_init__(self) -> None:
        _validate_positive_integer("workers", self.workers)
        if self.chunk_size is not None:
            _validate_positive_integer("chunk_size", self.chunk_size)
        if self.max_score_array_bytes is not None:
            _validate_positive_integer(
                "max_score_array_bytes", self.max_score_array_bytes
            )
        if self.max_structured_bell_workspace_bytes is not None:
            _validate_positive_integer(
                "max_structured_bell_workspace_bytes",
                self.max_structured_bell_workspace_bytes,
            )
        if self.max_enumeration_workspace_bytes is not None:
            _validate_positive_integer(
                "max_enumeration_workspace_bytes",
                self.max_enumeration_workspace_bytes,
            )
        if (
            not np.isfinite(self.enumeration_workspace_safety_factor)
            or float(self.enumeration_workspace_safety_factor) < 1.0
        ):
            raise ValueError(
                "enumeration_workspace_safety_factor must be finite and at least 1."
            )
        if not isinstance(self.benchmark, bool):
            raise TypeError("benchmark must be bool.")


@dataclass(frozen=True)
class WorkspacePhaseEstimate:
    """Known simultaneously-live buffers in one exhaustive stage phase."""

    phase: str
    buffers: Tuple[Tuple[str, int], ...]

    @property
    def known_bytes(self) -> int:
        return sum(int(size) for _name, size in self.buffers)


@dataclass(frozen=True)
class EnumerationWorkspaceEstimate:
    """Conservative modeled peak, not a guarantee of total process RSS."""

    n: int
    stage: str
    simulation_backend: SimulationBackend
    enumeration_size: int
    phases: Tuple[WorkspacePhaseEstimate, ...]
    safety_factor: float
    peak_phase: str
    known_peak_bytes: int
    safety_margin_bytes: int
    predicted_peak_bytes: int
    dominant_buffers: Tuple[Tuple[str, int], ...]


def estimate_enumeration_workspace(
    n: int,
    stage: str,
    *,
    simulation_backend: SimulationBackend = "batched_counts",
    bell_rounds: Optional[int] = None,
    structured_bell: bool = True,
    empirical: bool = True,
    return_details: bool = False,
    safety_factor: float = 1.25,
) -> EnumerationWorkspaceEstimate:
    """Model the peak known workspace for peeling or rank-guided recovery.

    The model enumerates buffers that can coexist instead of applying an
    unexplained multiplier. It assumes the validated implementation uses one
    shared all-Pauli workspace regardless of ``workers``. NumPy/Numba may use
    additional allocator/runtime memory, represented only by ``safety_factor``;
    this estimate is therefore not a process-memory or OOM guarantee.
    """

    n = _validate_positive_integer("n", n)
    if stage not in ("peeling", "recovery"):
        raise ValueError("stage must be 'peeling' or 'recovery'.")
    simulation_backend = _validate_simulation_backend(simulation_backend)
    if not np.isfinite(safety_factor) or float(safety_factor) < 1.0:
        raise ValueError("safety_factor must be finite and at least 1.")
    if empirical and simulation_backend == "legacy_shotwise":
        if bell_rounds is None:
            raise ValueError("legacy_shotwise workspace modeling requires bell_rounds.")
        bell_rounds = _validate_positive_integer("bell_rounds", bell_rounds)

    entries = 4**n
    float_table = entries * np.dtype(np.float64).itemsize
    int_table = entries * np.dtype(np.int64).itemsize
    index_table = entries * np.dtype(np.intp).itemsize
    bool_table = entries * np.dtype(np.bool_).itemsize
    phases = []

    if empirical and structured_bell:
        phases.append(
            WorkspacePhaseEstimate(
                "structured_bell_distribution",
                (
                    ("latent_or_physical_float64_table", float_table),
                    ("mapped_or_transform_float64_table", float_table),
                ),
            )
        )
    if empirical and simulation_backend == "batched_counts":
        phases.extend(
            (
                WorkspacePhaseEstimate(
                    "bell_category_sampling",
                    (
                        ("bell_probability_float64_table", float_table),
                        ("multinomial_int64_counts", int_table),
                    ),
                ),
                WorkspacePhaseEstimate(
                    "compressed_record_construction",
                    (
                        ("sampled_int64_counts", int_table),
                        ("immutable_record_int64_counts", int_table),
                    ),
                ),
                WorkspacePhaseEstimate(
                    "bell_score_transform",
                    (
                        ("retained_int64_counts", int_table),
                        ("all_pauli_int64_score_sums", int_table),
                    ),
                ),
            )
        )
        retained_record = (("retained_int64_counts", int_table),)
    elif empirical:
        assert bell_rounds is not None
        raw_outcomes = n * bell_rounds * 3 * np.dtype(np.int8).itemsize
        sampled_indices = bell_rounds * np.dtype(np.int64).itemsize
        category_matches = n * bell_rounds * 4 * np.dtype(np.bool_).itemsize
        decoded_categories = n * bell_rounds * np.dtype(np.int64).itemsize
        phases.extend(
            (
                WorkspacePhaseEstimate(
                    "legacy_bell_sampling",
                    (
                        ("bell_probability_float64_table", float_table),
                        ("sampled_category_indices", sampled_indices),
                        ("raw_int8_bell_outcomes", raw_outcomes),
                    ),
                ),
                WorkspacePhaseEstimate(
                    "legacy_record_construction",
                    (
                        ("sampled_raw_int8_bell_outcomes", raw_outcomes),
                        ("immutable_raw_int8_bell_outcomes", raw_outcomes),
                    ),
                ),
                WorkspacePhaseEstimate(
                    "legacy_score_category_decode",
                    (
                        ("retained_raw_int8_bell_outcomes", raw_outcomes),
                        ("local_category_match_bool_table", category_matches),
                        ("decoded_local_int64_categories", decoded_categories),
                    ),
                ),
                WorkspacePhaseEstimate(
                    "legacy_score_histogram",
                    (
                        ("retained_raw_int8_bell_outcomes", raw_outcomes),
                        ("decoded_local_int64_categories", decoded_categories),
                        ("per_round_int64_category_indices", sampled_indices),
                        ("int64_category_histogram", int_table),
                    ),
                ),
                WorkspacePhaseEstimate(
                    "legacy_bell_score_transform",
                    (
                        ("retained_raw_int8_bell_outcomes", raw_outcomes),
                        ("int64_category_histogram", int_table),
                        ("all_pauli_int64_score_sums", int_table),
                    ),
                ),
            )
        )
        retained_record = (("retained_raw_int8_bell_outcomes", raw_outcomes),)
    else:
        retained_record = ()

    score_buffer = ("all_pauli_score_or_sum_table", int_table)
    if stage == "peeling":
        buffers = (*retained_record, score_buffer)
        if return_details:
            buffers = (
                *buffers,
                ("threshold_comparison_bool_table", bool_table),
                ("detail_index_table", index_table),
            )
        phases.append(WorkspacePhaseEstimate("peeling_threshold_scan", buffers))
    else:
        phases.extend(
            (
                WorkspacePhaseEstimate(
                    "recovery_survivor_extraction",
                    (
                        *retained_record,
                        score_buffer,
                        ("threshold_comparison_bool_table", bool_table),
                        ("worst_case_survivor_indices", index_table),
                    ),
                ),
                WorkspacePhaseEstimate(
                    "recovery_ranking",
                    (
                        *retained_record,
                        score_buffer,
                        ("worst_case_survivor_indices", index_table),
                        ("fancy_indexed_score_temporary", int_table),
                        ("descending_score_key", int_table),
                        ("ranking_permutation", index_table),
                        ("sort_or_sorted_index_scratch", index_table),
                    ),
                ),
            )
        )

    peak = max(phases, key=lambda item: item.known_bytes)
    known_peak = peak.known_bytes
    predicted = int(math.ceil(known_peak * float(safety_factor)))
    return EnumerationWorkspaceEstimate(
        n=n,
        stage=stage,
        simulation_backend=simulation_backend,
        enumeration_size=entries,
        phases=tuple(phases),
        safety_factor=float(safety_factor),
        peak_phase=peak.phase,
        known_peak_bytes=known_peak,
        safety_margin_bytes=predicted - known_peak,
        predicted_peak_bytes=predicted,
        dominant_buffers=peak.buffers,
    )


@dataclass(frozen=True)
class EnumerationDiagnostics:
    """Lightweight opt-in observability for one exhaustive Pauli stage."""

    n: int
    enumeration_size: int
    score_dtype: str
    score_array_bytes: int
    worker_count: int
    chunk_size: Optional[int]
    stage_wall_times: Tuple[Tuple[str, float], ...] = ()


@dataclass(frozen=True)
class PerformanceDiagnostics:
    """Diagnostic-only wall times for one end-to-end learner execution."""

    stage_wall_times: Tuple[Tuple[str, float], ...]

    def as_dict(self) -> dict[str, float]:
        return {name: float(seconds) for name, seconds in self.stage_wall_times}


_STAGE_TIMING_CONTEXT: ContextVar[Optional[list[Tuple[str, float]]]] = ContextVar(
    "cebp_stage_timing_context", default=None
)


def _time_pipeline_stage(name: str):
    """Record a stage total only while the public end-to-end wrapper is active."""

    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                collector = _STAGE_TIMING_CONTEXT.get()
                if collector is not None:
                    collector.append((name, time.perf_counter() - started))

        return wrapped

    return decorate


class DataProvenance(str, Enum):
    """Origin of a numerical quantity exposed by a future learner stage."""

    EXACT = "EXACT"
    EMPIRICAL = "EMPIRICAL"


class ExecutionPolicy(str, Enum):
    """How a physical-copy ceiling is interpreted by the end-to-end learner.

    ``STRICT`` preserves the calibrated complete-batch semantics.  The explicit
    ``FIXED_BUDGET_GRACEFUL`` policy instead treats a finite copy budget as an
    estimator resource: undersampling lowers precision but does not by itself
    make the estimator unavailable.
    """

    STRICT = "strict"
    FIXED_BUDGET_GRACEFUL = "fixed_budget_graceful"


class GroupingSamplingPolicy(str, Enum):
    """Physical-shot semantics for learner-facing cumulant grouping."""

    CERTIFIED_TAU = "certified_tau"
    FIXED_BUDGET = "fixed_budget"


@dataclass(frozen=True)
class FixedBudgetStageWeights:
    """Positive coordinates normalized into five deterministic stage caps."""

    peeling: float = 1.0
    recovery: float = 1.0
    grouping: float = 1.0
    syndrome: float = 1.0
    tomography: float = 1.0

    def __post_init__(self) -> None:
        values = tuple(float(getattr(self, name)) for name in self.__dataclass_fields__)
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("Every fixed-budget stage weight must be finite and positive.")

    def normalized(self) -> Tuple[Tuple[str, float], ...]:
        values = tuple(
            (name, float(getattr(self, name))) for name in self.__dataclass_fields__
        )
        total = sum(value for _name, value in values)
        return tuple((name, value / total) for name, value in values)


@dataclass(frozen=True)
class FixedBudgetStageRecord:
    """Auditable immutable local-cap accounting for one numerical stage."""

    stage: str
    assigned_cap: int
    realized_copies: int
    unused_copies: int
    budget_exhausted: bool
    stage_complete: bool
    degradation_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.assigned_cap < 0 or self.realized_copies < 0:
            raise ValueError("Fixed-budget stage counts must be nonnegative.")
        if self.realized_copies > self.assigned_cap:
            raise ValueError("A fixed-budget stage exceeded its immutable local cap.")
        if self.unused_copies != self.assigned_cap - self.realized_copies:
            raise ValueError("unused_copies must equal assigned_cap-realized_copies.")
        if self.budget_exhausted and self.unused_copies != 0:
            raise ValueError("An exhausted stage cannot have unused local copies.")


def _validate_simulation_backend(value: str) -> SimulationBackend:
    if value not in ("legacy_shotwise", "batched_counts"):
        raise ValueError(
            "simulation_backend must be 'legacy_shotwise' or 'batched_counts'."
        )
    return value


@dataclass(frozen=True)
class SignedPauliProduct:
    """Exact canonical product ``i**phase_exponent * pauli``.

    This signed layer is deliberately separate from the phase-free binary
    symplectic arithmetic used by recovery.  The exponent is stored modulo
    four, avoiding floating-point comparisons of Pauli phases.
    """

    phase_exponent: int
    pauli: str

    def __post_init__(self) -> None:
        if isinstance(self.phase_exponent, bool) or not isinstance(
            self.phase_exponent, (int, np.integer)
        ):
            raise TypeError("phase_exponent must be an integer modulo four.")
        if not isinstance(self.pauli, str) or not self.pauli:
            raise ValueError("A signed Pauli product needs a nonempty Pauli string.")
        _validate_pauli_string(self.pauli, len(self.pauli))
        object.__setattr__(self, "phase_exponent", int(self.phase_exponent) % 4)

    @property
    def coefficient(self) -> complex:
        """Return the exact fourth-root-of-unity coefficient as a complex value."""
        return (1.0 + 0.0j, 1.0j, -1.0 + 0.0j, -1.0j)[self.phase_exponent]


@dataclass(frozen=True)
class SeedLedger:
    """Deterministic random streams allocated to a simulated CEBP instance."""

    master_seed: RngSeed
    partition_seed: RngSeed
    latent_state_seed: RngSeed
    clifford_seed: RngSeed


@dataclass(frozen=True)
class CopyLedger:
    """Immutable stage-by-stage physical-copy accounting.

    Phase 1 creates no measurement records, so generated instances use an
    empty ledger.  The type is introduced now to keep future measurement
    stages from passing around unstructured integer dictionaries.
    """

    entries: Tuple[Tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        names = set()
        for name, copies in self.entries:
            if not name or name in names:
                raise ValueError("Copy-ledger stage names must be nonempty and unique.")
            if not isinstance(copies, (int, np.integer)) or int(copies) < 0:
                raise ValueError("Copy-ledger counts must be nonnegative integers.")
            names.add(name)

    @classmethod
    def from_mapping(cls, values: Mapping[str, int]) -> "CopyLedger":
        return cls(tuple((str(name), int(count)) for name, count in values.items()))

    @property
    def total(self) -> int:
        return sum(copies for _, copies in self.entries)

    def as_dict(self) -> dict[str, int]:
        return dict(self.entries)

    def with_entry(self, name: str, copies: int) -> "CopyLedger":
        """Return a new ledger with one additional uniquely named stage."""
        if name in self.as_dict():
            raise ValueError(f"Copy-ledger stage {name!r} already exists.")
        return CopyLedger(self.entries + ((name, int(copies)),))


@dataclass(frozen=True)
class SimulatorMeasurementSource:
    """Opaque state-backed measurement source for learner-facing simulation.

    The exact density matrix/ket is deliberately stored in a private field.
    Future empirical learner stages should request measurements from this
    object instead of reading oracle metadata or accepting a ``CEBPInstance``.
    ``is_ket`` is strictly a representation/backend-dispatch flag: a rank-one
    density operator is physically pure but has ``is_ket=False``.
    """

    n: int
    is_ket: bool
    _state: Optional[qt.Qobj] = field(default=None, repr=False, compare=False)
    _structured_state: Optional[StructuredCEBPState] = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("Measurement-source n must be positive.")
        if (self._state is None) == (self._structured_state is None):
            raise ValueError(
                "A measurement source needs exactly one dense or structured backend."
            )
        if self._structured_state is not None and self._structured_state.n != self.n:
            raise ValueError("Structured backend dimension disagrees with source n.")

    def _state_for_backend(self, *, max_dense_debug_qubits: int = 8) -> qt.Qobj:
        """Explicit small-n reference bridge for legacy/debug measurement code."""

        if self.n > int(max_dense_debug_qubits):
            raise ValueError("dense state debug resource guard exceeded.")
        if self._state is not None:
            return self._state
        assert self._structured_state is not None
        return self._structured_state.materialize_dense_debug(
            max_qubits=max_dense_debug_qubits
        )

    @property
    def structured_bell_workspace_bytes(self) -> Optional[int]:
        """Return the backend's deterministic Bell workspace, if structured."""

        if self._structured_state is None:
            return None
        return self._structured_state.bell_workspace_bytes

    def _sample_bell_for_backend(
        self,
        rounds: int,
        seed: RngSeed,
        *,
        simulation_backend: SimulationBackend,
        max_structured_bell_workspace_bytes: Optional[int],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Tuple[Tuple[str, float], ...]]:
        """Operational Bell interface; hidden state structure stays backend-private."""

        started = time.perf_counter()
        if self._structured_state is not None:
            if simulation_backend == "batched_counts":
                counts, diagnostics = self._structured_state.sample_bell_counts(
                    rounds,
                    seed,
                    max_workspace_bytes=max_structured_bell_workspace_bytes,
                )
                return None, counts, diagnostics + (
                    ("bell_record_construction", time.perf_counter() - started),
                )
            outcomes = self._structured_state.sample_bell(
                rounds,
                seed,
                max_workspace_bytes=max_structured_bell_workspace_bytes,
            )
            return outcomes, None, (
                ("bell_record_construction", time.perf_counter() - started),
            )
        state = self._state_for_backend()
        if simulation_backend == "batched_counts":
            probabilities = np.asarray(
                _v1._bell_outcome_probabilities(state, self.n), dtype=float
            )
            counts = np.asarray(
                np.random.default_rng(seed).multinomial(rounds, probabilities),
                dtype=np.int64,
            )
            counts.setflags(write=False)
            return None, counts, (
                ("bell_record_construction", time.perf_counter() - started),
            )
        if self.is_ket:
            outcomes = _v1.bell_sampling_pure(
                n=self.n, M=rounds, psi=state, seed=seed
            )
        else:
            outcomes = _v1.bell_sampling(
                n=self.n, M=rounds, rho=state, seed=seed
            )
        return outcomes, None, (
            ("bell_record_construction", time.perf_counter() - started),
        )

    def _commuting_probabilities_for_backend(
        self, observables: Sequence[str]
    ) -> np.ndarray:
        """Operational joint-measurement distribution for a commuting tuple."""

        if self._structured_state is not None:
            return self._structured_state.commuting_probabilities(observables)
        values = tuple(observables)
        state = self._state_for_backend()
        operators = tuple(_v1._qutip_pauli_op(self.n, pauli) for pauli in values)
        identity = qt.qeye([2] * self.n)
        probabilities = []
        for outcome in itertools.product((-1, 1), repeat=len(values)):
            projector = identity
            for eigenvalue, operator in zip(outcome, operators):
                projector = projector * (identity + eigenvalue * operator) / 2.0
            probability = complex(qt.expect(projector, state))
            if abs(probability.imag) > 1e-9:
                raise RuntimeError("Joint Pauli probability has an imaginary part.")
            probabilities.append(float(probability.real))
        result = np.asarray(probabilities, dtype=float)
        result = np.clip(result, 0.0, None)
        result /= result.sum()
        return result

    @property
    def uses_structured_backend(self) -> bool:
        return self._structured_state is not None

    def _expectation_for_backend(self, pauli: str) -> float:
        if self._structured_state is not None:
            return self._structured_state.expectation(pauli)
        assert self._state is not None
        return debug_exact_pauli_expectation(self._state, pauli)

    def _transformed_by_dagger(
        self, clifford: SignedClifford
    ) -> "SimulatorMeasurementSource":
        if self._structured_state is None:
            raise ValueError("Compact transformation requires a structured backend.")
        transformed = self._structured_state.transformed_by_dagger(clifford)
        return SimulatorMeasurementSource(
            n=self.n,
            is_ket=self.is_ket,
            _structured_state=transformed,
        )


@dataclass(frozen=True)
class CEBPLearnerView:
    """The information and operational access available to the learner."""

    n: int
    d: int
    measurement_source: SimulatorMeasurementSource


class PauliScoreSource(Protocol):
    """Structural interface consumed by the common peeling core."""

    n: int
    frame: str
    provenance: DataProvenance
    uniform_radius: float
    bell_rounds: int
    copy_ledger: CopyLedger

    def score(self, pauli: str) -> float:
        """Return the score associated with one phase-free Pauli string."""


@dataclass(frozen=True)
class BellScoreRecord:
    """One shared empirical Bell record for all Pauli-score queries.

    ``outcomes`` has V1 shape ``(n, rounds, 3)`` with axes ordered ``X,Y,Z``.
    One round consumes two physical copies, recorded under the explicitly
    named stage pool.  The array is copied and made read-only so multiple
    adaptive score queries provably reuse the same immutable measurement data.
    """

    n: int
    rounds: int
    frame: str
    uniform_radius: float
    zeta_bs: float
    backend: str
    seed: RngSeed
    pool_name: str
    copy_ledger: CopyLedger
    outcomes: Optional[np.ndarray] = field(repr=False, compare=False)
    provenance: DataProvenance = field(
        default=DataProvenance.EMPIRICAL, init=False
    )
    simulation_backend: SimulationBackend = "legacy_shotwise"
    category_counts: Optional[np.ndarray] = field(
        default=None, repr=False, compare=False
    )
    stage_wall_times: Tuple[Tuple[str, float], ...] = ()
    _all_scores_cache: Optional[np.ndarray] = field(
        default=None, init=False, repr=False, compare=False
    )
    _all_score_sums_cache: Optional[np.ndarray] = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        n = _validate_positive_integer("n", self.n)
        rounds = _validate_positive_integer("rounds", self.rounds)
        if not self.frame:
            raise ValueError("A Bell-score record must have a nonempty frame label.")
        if not (0.0 < float(self.zeta_bs) < 1.0):
            raise ValueError("zeta_bs must lie in (0,1).")
        if not np.isfinite(self.uniform_radius) or self.uniform_radius < 0.0:
            raise ValueError("uniform_radius must be finite and nonnegative.")
        if (self.outcomes is None) == (self.category_counts is None):
            raise ValueError("A Bell record needs exactly one raw or compressed representation.")
        if self.outcomes is not None:
            outcomes = np.asarray(self.outcomes, dtype=np.int8)
            if outcomes.shape != (n, rounds, 3):
                raise ValueError(f"outcomes must have shape {(n, rounds, 3)}.")
            if not np.all(np.isin(outcomes, (-1, 1))):
                raise ValueError("Bell outcomes must be eigenvalues in {-1,+1}.")
            outcomes = outcomes.copy()
            outcomes.setflags(write=False)
            object.__setattr__(self, "outcomes", outcomes)
        else:
            counts = np.asarray(self.category_counts, dtype=np.int64)
            if counts.shape != (4**n,) or np.any(counts < 0):
                raise ValueError("Compressed Bell counts must have shape (4**n,).")
            if int(counts.sum()) != rounds:
                raise ValueError("Compressed Bell counts must sum to rounds.")
            counts = counts.copy()
            counts.setflags(write=False)
            object.__setattr__(self, "category_counts", counts)
        if self.pool_name not in ("peeling_bell_pool", "recovery_bell_pool"):
            raise ValueError("Bell pool_name must identify peeling or recovery.")
        if self.copy_ledger.as_dict() != {self.pool_name: 2 * rounds}:
            raise ValueError("A Bell record must account for exactly 2*rounds copies.")
        _validate_simulation_backend(self.simulation_backend)

    @property
    def bell_rounds(self) -> int:
        return self.rounds

    def score(self, pauli: str) -> float:
        """Evaluate the Bell parity mean without collecting new samples."""
        _validate_pauli_string(pauli, self.n)
        if self.outcomes is None or (
            self.simulation_backend == "batched_counts"
            and self.n <= _MAX_BELL_SCORE_CACHE_QUBITS
        ):
            return float(self.all_score_sums()[pauli_string_to_index(pauli)] / self.rounds)
        assert self.outcomes is not None
        parity = np.ones(self.rounds, dtype=np.int16)
        for qubit, character in enumerate(pauli):
            if character != "I":
                parity *= self.outcomes[qubit, :, _v1.AXIS_TO_K[character]]
        return float(parity.mean())

    def scores(self, paulis: Iterable[str]) -> Tuple[float, ...]:
        return tuple(self.score(pauli) for pauli in paulis)

    def all_score_sums(self) -> np.ndarray:
        """Return the shared record's exact contiguous integer sufficient statistics."""

        if self._all_score_sums_cache is None:
            sums = (
                bell_score_sums_from_counts(self.category_counts, self.n)
                if self.category_counts is not None
                else bell_score_sums_from_outcomes(self.outcomes)
            )
            object.__setattr__(
                self,
                "_all_score_sums_cache",
                sums,
            )
        return self._all_score_sums_cache

    def all_scores(self) -> np.ndarray:
        """Return one read-only contiguous mean-score array in fixed Pauli order."""

        if self._all_scores_cache is None:
            scores = np.asarray(self.all_score_sums(), dtype=np.float64) / self.rounds
            scores.setflags(write=False)
            object.__setattr__(self, "_all_scores_cache", scores)
        return self._all_scores_cache


@dataclass(frozen=True)
class DebugExactPauliScoreSource:
    """Oracle/debug-only exact scores; never used by the empirical path."""

    n: int
    _debug_state: Optional[qt.Qobj] = field(default=None, repr=False, compare=False)
    _structured_state: Optional[StructuredCEBPState] = field(
        default=None, repr=False, compare=False
    )
    frame: str = "physical"
    uniform_radius: float = 0.0
    bell_rounds: int = 0
    copy_ledger: CopyLedger = field(default_factory=CopyLedger)
    provenance: DataProvenance = field(default=DataProvenance.EXACT, init=False)
    _all_scores_cache: Optional[np.ndarray] = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _validate_positive_integer("n", self.n)
        if not self.frame:
            raise ValueError("An exact score source must have a nonempty frame label.")
        if self.uniform_radius != 0.0 or self.bell_rounds != 0:
            raise ValueError("Exact/debug score sources use radius and Bell rounds zero.")
        if self.copy_ledger.total != 0:
            raise ValueError("Exact/debug score sources consume no learner copies.")
        if (self._debug_state is None) == (self._structured_state is None):
            raise ValueError("An exact score source needs one dense or structured backend.")
        if self._structured_state is not None:
            if self._structured_state.n != self.n:
                raise ValueError("Structured exact score source dimension mismatch.")
        else:
            normalized = _v1._state_to_qobj(self._debug_state, self.n)
            object.__setattr__(self, "_debug_state", normalized)

    def score(self, pauli: str) -> float:
        _validate_pauli_string(pauli, self.n)
        if self._structured_state is not None:
            return float(self.all_scores()[pauli_string_to_index(pauli)])
        assert self._debug_state is not None
        return debug_exact_pauli_score(self._debug_state, pauli)

    def all_scores(self) -> np.ndarray:
        """Return one exact contiguous score table when the backend supports it."""

        if self._all_scores_cache is None:
            if self._structured_state is not None:
                scores = self._structured_state.all_squared_pauli_scores()
            else:
                assert self._debug_state is not None
                scores = np.empty(4**self.n, dtype=np.float64)
                for index in range(scores.size):
                    scores[index] = debug_exact_pauli_score(
                        self._debug_state, pauli_index_to_string(index, self.n)
                    )
                scores.setflags(write=False)
            object.__setattr__(self, "_all_scores_cache", scores)
        return self._all_scores_cache


@dataclass(frozen=True)
class PauliScoreMap:
    """Deterministic complete score map for certification unit tests/debugging."""

    n: int
    score_map: Mapping[str, float] = field(repr=False, compare=False)
    uniform_radius: float
    provenance: DataProvenance = DataProvenance.EXACT
    frame: str = "synthetic_debug"
    bell_rounds: int = 0
    copy_ledger: CopyLedger = field(default_factory=CopyLedger)

    def __post_init__(self) -> None:
        n = _validate_positive_integer("n", self.n)
        if not np.isfinite(self.uniform_radius) or self.uniform_radius < 0.0:
            raise ValueError("uniform_radius must be finite and nonnegative.")
        expected = set(_all_pauli_strings(n))
        values = dict(self.score_map)
        if set(values) != expected:
            missing = len(expected.difference(values))
            extra = len(set(values).difference(expected))
            raise ValueError(
                f"score_map must contain all 4^n Paulis (missing={missing}, extra={extra})."
            )
        for pauli, value in values.items():
            if not np.isfinite(value) or not (-1.0 <= float(value) <= 1.0):
                raise ValueError(f"Invalid score value for {pauli!r}.")
        if self.provenance is DataProvenance.EMPIRICAL:
            if self.bell_rounds <= 0:
                raise ValueError("Empirical score maps require positive bell_rounds.")
            if self.copy_ledger.total != 2 * self.bell_rounds:
                raise ValueError("Empirical score maps must account for 2*bell_rounds copies.")
        elif self.bell_rounds != 0 or self.copy_ledger.total != 0:
            raise ValueError("Exact/debug score maps consume no learner copies.")
        object.__setattr__(self, "score_map", MappingProxyType(values))

    def score(self, pauli: str) -> float:
        _validate_pauli_string(pauli, self.n)
        return float(self.score_map[pauli])


@dataclass(frozen=True)
class PeelingConfig:
    """Manuscript threshold window, grid, and empirical Bell budget."""

    h_min: float = 0.6
    h_max: float = 0.9
    eta: float = 0.01
    M1: Optional[int] = None
    zeta_bs: float = 0.05
    return_details: bool = False
    max_enumeration_qubits: int = _MAX_P2_ENUMERATION_QUBITS
    enumeration_execution: EnumerationExecutionConfig = field(
        default_factory=EnumerationExecutionConfig
    )
    materialize_dense_clifford: bool = True
    max_dense_debug_qubits: int = 8

    def __post_init__(self) -> None:
        if not (0.5 < float(self.h_min) < float(self.h_max) < 1.0):
            raise ValueError("Require 1/2 < h_min < h_max < 1.")
        if not np.isfinite(self.eta) or self.eta <= 0.0:
            raise ValueError("eta must be finite and positive.")
        if self.M1 is not None:
            _validate_positive_integer("M1", self.M1)
        if not (0.0 < float(self.zeta_bs) < 1.0):
            raise ValueError("zeta_bs must lie in (0,1).")
        _validate_positive_integer("max_enumeration_qubits", self.max_enumeration_qubits)
        if not isinstance(self.enumeration_execution, EnumerationExecutionConfig):
            raise TypeError("enumeration_execution must be EnumerationExecutionConfig.")
        if not isinstance(self.materialize_dense_clifford, bool):
            raise TypeError("materialize_dense_clifford must be bool.")
        _validate_positive_integer("max_dense_debug_qubits", self.max_dense_debug_qubits)

    @classmethod
    def calibrated(
        cls,
        n: int,
        *,
        h_min: float,
        h_max: float,
        zeta_bs: float,
        return_details: bool = False,
        max_enumeration_qubits: int = _MAX_P2_ENUMERATION_QUBITS,
    ) -> "PeelingConfig":
        """Construct the sufficient grid and M1 budget from the manuscript."""
        n = _validate_positive_integer("n", n)
        if not (0.5 < float(h_min) < float(h_max) < 1.0):
            raise ValueError("Require 1/2 < h_min < h_max < 1.")
        if not (0.0 < float(zeta_bs) < 1.0):
            raise ValueError("zeta_bs must lie in (0,1).")
        delta_h = float(h_max - h_min)
        eta = delta_h / (4.0 * (2 * n + 1))
        rounds = int(
            math.ceil(
                128.0
                * (2 * n + 1) ** 2
                / (delta_h**2)
                * math.log(2.0 * (4**n) / zeta_bs)
            )
        )
        return cls(
            h_min=h_min,
            h_max=h_max,
            eta=eta,
            M1=rounds,
            zeta_bs=zeta_bs,
            return_details=return_details,
            max_enumeration_qubits=max_enumeration_qubits,
        )

    @property
    def threshold_grid(self) -> Tuple[float, ...]:
        delta_h = float(self.h_max - self.h_min)
        count = int(math.ceil(delta_h / float(self.eta)))
        mesh = delta_h / count
        return tuple(float(self.h_min + index * mesh) for index in range(count + 1))

    @property
    def actual_mesh(self) -> float:
        grid = self.threshold_grid
        return grid[1] - grid[0]


@dataclass(frozen=True)
class PeelingThresholdAttempt:
    """Compact transcript for one grid point."""

    h: float
    inner_count: int
    outer_count: int
    inner_rank: int
    outer_rank: int
    spans_equal: bool
    isotropic: Optional[bool]
    accepted: bool
    reason: str


@dataclass(frozen=True)
class PeelingResult:
    """Learner-visible certified-peeling result with explicit failure state."""

    success: bool
    failure_reason: Optional[str]
    h: Optional[float]
    lambda_: Optional[float]
    tau_1: float
    t: Optional[int]
    generators: Tuple[str, ...]
    generator_symplectic_vectors: Tuple[Tuple[int, ...], ...]
    certified_span_basis: Tuple[int, ...]
    inner_set: Tuple[str, ...]
    outer_set: Tuple[str, ...]
    inner_span_basis: Tuple[int, ...]
    outer_span_basis: Tuple[int, ...]
    U_stab: Optional[np.ndarray] = field(repr=False, compare=False)
    tableau: Optional[np.ndarray] = field(repr=False, compare=False)
    gates: Tuple[Tuple, ...]
    epsilon_peel: Optional[float]
    M1: int
    copy_ledger: CopyLedger
    score_provenance: DataProvenance
    score_frame: str
    threshold_grid: Tuple[float, ...]
    transcript: Tuple[PeelingThresholdAttempt, ...]
    theorem_grid_condition: bool
    theorem_tau_condition: bool
    signed_clifford: Optional[SignedClifford] = field(
        default=None, repr=False, compare=False
    )
    enumeration_diagnostics: Optional[EnumerationDiagnostics] = field(
        default=None, compare=False
    )

    @property
    def theorem_preconditions_hold(self) -> bool:
        """Whether the sufficient grid and concentration conditions hold.

        This is intentionally separate from ``success``: manual/debug settings
        may identify an operationally stable span without satisfying the
        manuscript's sufficient calibration conditions.
        """
        return self.theorem_grid_condition and self.theorem_tau_condition

    @property
    def theorem_certified(self) -> bool:
        """Whether peeling both succeeded and met theorem preconditions."""
        return self.success and self.theorem_preconditions_hold

    def __post_init__(self) -> None:
        if not self.success and self.tableau is None and self.signed_clifford is not None:
            object.__setattr__(self, "signed_clifford", None)
        if self.success and self.signed_clifford is None and self.tableau is not None:
            n = np.asarray(self.tableau).shape[0] // 2
            try:
                compact = SignedClifford(n, np.asarray(self.tableau), tuple(self.gates))
            except ValueError:
                compact = None
            object.__setattr__(self, "signed_clifford", compact)
        if self.success:
            if self.failure_reason is not None:
                raise ValueError("A successful result cannot have a failure reason.")
            if None in (self.h, self.lambda_, self.t, self.epsilon_peel):
                raise ValueError("A successful result is missing certified fields.")
            if self.t != len(self.generators):
                raise ValueError("t must equal the returned generator count.")
            if self.tableau is None or self.signed_clifford is None:
                raise ValueError("A successful result must contain compact Clifford data.")
        else:
            if not self.failure_reason:
                raise ValueError("A failed result must state a failure reason.")
            if any(value is not None for value in (self.h, self.lambda_, self.t, self.epsilon_peel)):
                raise ValueError("A failed result cannot contain fake certified values.")
            if self.U_stab is not None or self.tableau is not None or self.signed_clifford is not None:
                raise ValueError("A failed result cannot contain a synthesized Clifford.")
        for name in ("U_stab", "tableau"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value).copy()
                array.setflags(write=False)
                object.__setattr__(self, name, array)


@dataclass(frozen=True)
class RecoveryConfig:
    """Configuration for manuscript rank-guided sector recovery."""

    theta: float
    M2: Optional[int] = None
    zeta_rank: float = 0.05
    return_details: bool = False
    max_enumeration_qubits: int = _MAX_P3_ENUMERATION_QUBITS
    allow_uncalibrated_peeling: bool = False
    allow_margin_failure: bool = False
    enumeration_execution: EnumerationExecutionConfig = field(
        default_factory=EnumerationExecutionConfig
    )

    def __post_init__(self) -> None:
        if not (0.0 < float(self.theta) < 1.0):
            raise ValueError("theta must lie strictly in (0,1).")
        if self.M2 is not None:
            _validate_positive_integer("M2", self.M2)
        if not (0.0 < float(self.zeta_rank) < 1.0):
            raise ValueError("zeta_rank must lie in (0,1).")
        _validate_positive_integer("max_enumeration_qubits", self.max_enumeration_qubits)
        if not isinstance(self.allow_uncalibrated_peeling, bool):
            raise TypeError("allow_uncalibrated_peeling must be bool.")
        if not isinstance(self.allow_margin_failure, bool):
            raise TypeError("allow_margin_failure must be bool.")
        if not isinstance(self.enumeration_execution, EnumerationExecutionConfig):
            raise TypeError("enumeration_execution must be EnumerationExecutionConfig.")

    @classmethod
    def calibrated(
        cls,
        n: int,
        *,
        theta_0: float,
        lambda_0: float,
        zeta_rank: float,
        return_details: bool = False,
        max_enumeration_qubits: int = _MAX_P3_ENUMERATION_QUBITS,
    ) -> "RecoveryConfig":
        """Use Eq. ``eq:supp-calibrated-second-bell-rounds`` exactly."""
        n = _validate_positive_integer("n", n)
        if not (0.0 < float(theta_0) < 1.0):
            raise ValueError("theta_0 must lie in (0,1).")
        if not (0.0 < float(lambda_0) < 1.0):
            raise ValueError("lambda_0 must lie in (0,1).")
        if not (0.0 < float(zeta_rank) < 1.0):
            raise ValueError("zeta_rank must lie in (0,1).")
        rounds = int(
            math.ceil(
                128.0
                / (float(lambda_0) ** 2 * float(theta_0) ** 2)
                * math.log(2.0 * (4**n) / float(zeta_rank))
            )
        )
        return cls(
            theta=float(theta_0),
            M2=rounds,
            zeta_rank=float(zeta_rank),
            return_details=return_details,
            max_enumeration_qubits=max_enumeration_qubits,
        )


@dataclass(frozen=True)
class RecoveredSector:
    """One learner-visible residual singleton or completed Pauli triple."""

    sector_id: int
    x: str
    z: Optional[str] = None
    y: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.sector_id, bool) or not isinstance(
            self.sector_id, (int, np.integer)
        ) or int(self.sector_id) < 0:
            raise ValueError("sector_id must be a nonnegative integer.")
        if not isinstance(self.x, str) or not self.x:
            raise ValueError("A recovered sector needs a nonidentity x axis.")
        _validate_pauli_string(self.x, len(self.x))
        if set(self.x) == {"I"}:
            raise ValueError("A recovered axis cannot be identity.")
        if (self.z is None) != (self.y is None):
            raise ValueError("z and y must either both be present or both be absent.")
        if self.z is not None:
            _validate_pauli_string(self.z, len(self.x))
            _validate_pauli_string(self.y, len(self.x))
            if set(self.z) == {"I"} or set(self.y) == {"I"}:
                raise ValueError("Completed-triple axes must be nonidentity.")

    @property
    def kind(self) -> str:
        return "TRIPLE" if self.z is not None else "SINGLETON"

    @property
    def completed(self) -> bool:
        return self.z is not None

    @property
    def members(self) -> Tuple[str, ...]:
        if self.z is None:
            return (self.x,)
        return (self.x, self.z, self.y)

    @property
    def independent_axes(self) -> Tuple[str, ...]:
        if self.z is None:
            return (self.x,)
        return (self.x, self.z)


@dataclass(frozen=True)
class RecoveryCandidate:
    """One full ranked peeled Pauli and its parsed residual action."""

    full_pauli: str
    score: float
    prefix_bits: Optional[Tuple[int, ...]]
    residual_pauli: Optional[str]


@dataclass(frozen=True)
class RecoveryStep:
    """Optional deterministic transcript entry for one ranked candidate."""

    full_pauli: str
    score: float
    residual_pauli: Optional[str]
    dressed_pauli: Optional[str]
    action: str
    affected_sector_ids: Tuple[int, ...] = ()


@dataclass(frozen=True)
class RecoveryResult:
    """Learner-visible output of Phase 3 rank-guided recovery."""

    success: bool
    failure_reason: Optional[str]
    n: int
    t: int
    m: int
    theta: float
    tau_rank: float
    M2: int
    zeta_rank: float
    lambda_: float
    sectors: Tuple[RecoveredSector, ...]
    independent_axes: Tuple[str, ...]
    recovered_span_basis: Tuple[int, ...]
    ranked_survivor_count: int
    score_provenance: DataProvenance
    score_frame: str
    threshold_margin_holds: bool
    ranking_gap_margin_holds: bool
    theorem_recovery_preconditions_hold: bool
    threshold_span_complete: bool
    copy_ledger: CopyLedger
    cumulative_copy_ledger: CopyLedger
    ranked_candidates: Tuple[RecoveryCandidate, ...] = ()
    transcript: Tuple[RecoveryStep, ...] = ()
    enumeration_diagnostics: Optional[EnumerationDiagnostics] = field(
        default=None, compare=False
    )

    def __post_init__(self) -> None:
        if self.n < 1 or not (0 <= self.t <= self.n) or self.m != self.n - self.t:
            raise ValueError("Recovery dimensions are inconsistent.")
        if self.success:
            if self.failure_reason is not None:
                raise ValueError("A successful recovery cannot have a failure reason.")
            if not self.threshold_span_complete:
                raise ValueError("A successful recovery must satisfy span completeness.")
        else:
            if not self.failure_reason:
                raise ValueError("A failed recovery must state a failure reason.")
            if self.sectors or self.independent_axes or self.recovered_span_basis:
                raise ValueError("A failed recovery cannot expose fake recovered sectors.")


class RecoveryPreconditionError(ValueError):
    """Raised before empirical sampling when theorem preconditions fail."""


class CopyBudgetExceeded(RuntimeError):
    """Raised before a physical batch that would exceed the execution cap."""

    def __init__(self, stage: str, realized_so_far: int, next_requested: int, cap: int):
        self.stage = stage
        self.realized_so_far = int(realized_so_far)
        self.next_requested = int(next_requested)
        self.cap = int(cap)
        super().__init__(
            f"stage={stage}; realized_so_far={realized_so_far}; "
            f"next_requested={next_requested}; cap={cap}"
        )


class GroupingBudgetExhausted(CopyBudgetExceeded):
    """Internal control flow for a graceful stop before an unstarted query."""


def _check_execution_copy_cap(
    cap: Optional[int], realized_so_far: int, next_requested: int, stage: str
) -> None:
    if next_requested < 0 or realized_so_far < 0:
        raise ValueError("Copy-cap accounting values must be nonnegative.")
    if cap is not None and realized_so_far + next_requested > cap:
        raise CopyBudgetExceeded(stage, realized_so_far, next_requested, cap)


@dataclass(frozen=True)
class GroupingConfig:
    """Configuration for exact/debug or ordinary empirical Phase-4 grouping."""

    ell_grp: Optional[int] = None
    eta_test: float = 0.0
    tau_kappa: Optional[float] = 0.0
    delta_grp_ordinary: Optional[float] = 0.05
    return_details: bool = False
    allow_uncalibrated_recovery: bool = False
    allow_no_false_merge_margin_failure: bool = False
    exact_zero_tolerance: float = 1e-12
    max_generated_group_size: int = _MAX_P4_GENERATED_GROUP_SIZE
    eta_s: Optional[float] = None
    eta_irr: Optional[float] = None
    sampling_policy: str = GroupingSamplingPolicy.CERTIFIED_TAU.value

    def __post_init__(self) -> None:
        if self.ell_grp is not None:
            _validate_positive_integer("ell_grp", self.ell_grp)
        try:
            policy = GroupingSamplingPolicy(self.sampling_policy)
        except ValueError as error:
            raise ValueError(
                "sampling_policy must be 'certified_tau' or 'fixed_budget'."
            ) from error
        if not np.isfinite(self.eta_test):
            raise ValueError("eta_test must be finite.")
        if policy is GroupingSamplingPolicy.FIXED_BUDGET:
            if self.eta_test <= 0.0:
                raise ValueError("fixed-budget empirical eta_test must be positive.")
        else:
            if self.eta_test < 0.0:
                raise ValueError("eta_test must be finite and nonnegative.")
            if (
                self.tau_kappa is None
                or not np.isfinite(self.tau_kappa)
                or self.tau_kappa < 0.0
            ):
                raise ValueError("certified tau_kappa must be finite and nonnegative.")
            if self.delta_grp_ordinary is None or not (
                0.0 < float(self.delta_grp_ordinary) < 1.0
            ):
                raise ValueError("certified delta_grp_ordinary must lie in (0,1).")
        if not np.isfinite(self.exact_zero_tolerance) or self.exact_zero_tolerance < 0.0:
            raise ValueError("exact_zero_tolerance must be finite and nonnegative.")
        _validate_positive_integer(
            "max_generated_group_size", self.max_generated_group_size
        )
        for name in (
            "return_details",
            "allow_uncalibrated_recovery",
            "allow_no_false_merge_margin_failure",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool.")
        for name in ("eta_s", "eta_irr"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive when supplied.")
        if (
            policy is GroupingSamplingPolicy.CERTIFIED_TAU
            and self.eta_s is not None
            and not (
                np.isclose(self.eta_test, self.eta_s / 2.0)
                and np.isclose(self.tau_kappa, self.eta_s / 4.0)
            )
        ):
            raise ValueError("eta_s requires eta_test=eta_s/2 and tau_kappa=eta_s/4.")

    @classmethod
    def from_guessed_scale(
        cls,
        d: int,
        eta_s: float,
        *,
        delta_grp_ordinary: float = 0.05,
        return_details: bool = False,
        allow_uncalibrated_recovery: bool = False,
        allow_no_false_merge_margin_failure: bool = False,
        exact_zero_tolerance: float = 1e-12,
        max_generated_group_size: int = _MAX_P4_GENERATED_GROUP_SIZE,
    ) -> "GroupingConfig":
        """Use the manuscript guessed-scale schedule exactly."""
        d = _validate_positive_integer("d", d)
        if not np.isfinite(eta_s) or eta_s <= 0.0:
            raise ValueError("eta_s must be finite and positive.")
        return cls(
            ell_grp=d,
            eta_test=float(eta_s) / 2.0,
            tau_kappa=float(eta_s) / 4.0,
            delta_grp_ordinary=delta_grp_ordinary,
            return_details=return_details,
            allow_uncalibrated_recovery=allow_uncalibrated_recovery,
            allow_no_false_merge_margin_failure=allow_no_false_merge_margin_failure,
            exact_zero_tolerance=exact_zero_tolerance,
            max_generated_group_size=max_generated_group_size,
            eta_s=float(eta_s),
        )


@dataclass(frozen=True)
class GeneratedPauliGroup:
    """Deterministic phase-free generated group ``P(A)`` for one cluster."""

    cluster: Tuple[int, ...]
    rank: int
    paulis: Tuple[str, ...]

    @property
    def nonidentity(self) -> Tuple[str, ...]:
        if not self.paulis:
            return ()
        identity = "I" * len(self.paulis[0])
        return tuple(pauli for pauli in self.paulis if pauli != identity)


@dataclass(frozen=True)
class JointPauliMeasurementRecord:
    """One fresh ordinary joint-measurement batch for a labeled tuple."""

    observables: Tuple[str, ...]
    shots: int
    seed: RngSeed
    outcomes: np.ndarray = field(repr=False, compare=False)
    frame: str = field(default="peeled_residual", init=False)
    provenance: DataProvenance = field(default=DataProvenance.EMPIRICAL, init=False)

    def __post_init__(self) -> None:
        shots = _validate_positive_integer("shots", self.shots)
        if not self.observables:
            raise ValueError("A joint-measurement tuple cannot be empty.")
        m = len(self.observables[0])
        for observable in self.observables:
            _validate_pauli_string(observable, m)
            if set(observable) <= {"I"}:
                raise ValueError("Joint-measurement tuple members must be nonidentity.")
        outcomes = np.asarray(self.outcomes, dtype=np.int8)
        if outcomes.shape != (shots, len(self.observables)):
            raise ValueError("Joint outcomes have the wrong shape.")
        if not np.all(np.isin(outcomes, (-1, 1))):
            raise ValueError("Joint outcomes must lie in {-1,+1}.")
        outcomes = outcomes.copy()
        outcomes.setflags(write=False)
        object.__setattr__(self, "outcomes", outcomes)

    @property
    def q(self) -> int:
        return len(self.observables)

    @property
    def copies(self) -> int:
        return self.shots

    def subset_moment(self, positions: Iterable[int]) -> float:
        labels = tuple(sorted(int(position) for position in positions))
        if len(set(labels)) != len(labels) or any(
            position < 0 or position >= self.q for position in labels
        ):
            raise ValueError("Subset positions must be distinct labels in range(q).")
        if not labels:
            return 1.0
        return float(np.prod(self.outcomes[:, labels], axis=1).mean())

    def all_subset_moments(self) -> Mapping[frozenset[int], float]:
        values = {frozenset(): 1.0}
        for size in range(1, self.q + 1):
            for positions in itertools.combinations(range(self.q), size):
                values[frozenset(positions)] = self.subset_moment(positions)
        return MappingProxyType(values)


@dataclass(frozen=True)
class JointPauliCountRecord:
    """Compressed ordinary joint-measurement sufficient statistics."""

    observables: Tuple[str, ...]
    shots: int
    seed: RngSeed
    outcome_values: np.ndarray = field(repr=False, compare=False)
    counts: np.ndarray = field(repr=False, compare=False)
    frame: str = field(default="peeled_residual", init=False)
    provenance: DataProvenance = field(default=DataProvenance.EMPIRICAL, init=False)

    def __post_init__(self) -> None:
        shots = _validate_positive_integer("shots", self.shots)
        if not self.observables:
            raise ValueError("A joint-measurement tuple cannot be empty.")
        m = len(self.observables[0])
        for observable in self.observables:
            _validate_pauli_string(observable, m)
            if set(observable) <= {"I"}:
                raise ValueError("Joint-measurement tuple members must be nonidentity.")
        outcomes = np.asarray(self.outcome_values, dtype=np.int8)
        counts = np.asarray(self.counts)
        expected_bins = 2 ** len(self.observables)
        if outcomes.shape != (expected_bins, len(self.observables)):
            raise ValueError("Compressed joint outcomes have the wrong shape.")
        if not np.all(np.isin(outcomes, (-1, 1))):
            raise ValueError("Compressed joint outcomes must lie in {-1,+1}.")
        if counts.shape != (expected_bins,) or not np.issubdtype(counts.dtype, np.integer):
            raise ValueError("counts must be an integer vector with one entry per outcome.")
        counts = counts.astype(np.int64, copy=True)
        if np.any(counts < 0) or int(counts.sum()) != shots:
            raise ValueError("counts must be nonnegative and sum exactly to shots.")
        outcomes = outcomes.copy()
        outcomes.setflags(write=False)
        counts.setflags(write=False)
        object.__setattr__(self, "outcome_values", outcomes)
        object.__setattr__(self, "counts", counts)

    @property
    def q(self) -> int:
        return len(self.observables)

    @property
    def copies(self) -> int:
        return self.shots

    def subset_moment(self, positions: Iterable[int]) -> float:
        labels = tuple(sorted(int(position) for position in positions))
        if len(set(labels)) != len(labels) or any(
            position < 0 or position >= self.q for position in labels
        ):
            raise ValueError("Subset positions must be distinct labels in range(q).")
        if not labels:
            return 1.0
        signs = np.prod(self.outcome_values[:, labels], axis=1, dtype=np.int64)
        return float(np.dot(self.counts, signs) / self.shots)

    def all_subset_moments(self) -> Mapping[frozenset[int], float]:
        values = {frozenset(): 1.0}
        for size in range(1, self.q + 1):
            for positions in itertools.combinations(range(self.q), size):
                values[frozenset(positions)] = self.subset_moment(positions)
        return MappingProxyType(values)


@dataclass(frozen=True)
class HyperedgeWitness:
    """First deterministic operator witness for one test hyperedge."""

    order: int
    clusters: Tuple[Tuple[int, ...], ...]
    observables: Tuple[str, ...]
    cumulant: float


@dataclass(frozen=True)
class GroupingScan:
    """One order scan in the hierarchical grouping transcript."""

    order: int
    partition_before: Tuple[Tuple[int, ...], ...]
    active_clusters: Tuple[Tuple[int, ...], ...]
    hyperedges: Tuple[HyperedgeWitness, ...]
    connected_components: Tuple[Tuple[Tuple[int, ...], ...], ...]
    partition_after: Tuple[Tuple[int, ...], ...]
    reset_to_two: bool


@dataclass(frozen=True)
class GroupingResult:
    """Learner-visible Phase-4 partition and its theorem/copy transcript."""

    success: bool
    failure_reason: Optional[str]
    L: int
    ell_grp: int
    clusters: Tuple[Tuple[int, ...], ...]
    eta_test: float
    tau_kappa: float
    beta_peel: float
    eta_s: Optional[float]
    xi_s: Optional[float]
    no_false_merge_condition_holds: bool
    exact_recovery_window_holds: Optional[bool]
    recovery_theorem_preconditions_hold: bool
    theorem_grouping_preconditions_hold: bool
    recovery_score_provenance: DataProvenance
    moment_provenance: DataProvenance
    realized_query_count: int
    query_count_by_order: Tuple[Tuple[int, int], ...]
    realized_grouping_copies: int
    copies_by_order: Tuple[Tuple[int, int], ...]
    N_test_max: int
    N_test_simplified_bound: int
    delta_tuple: Optional[float]
    ordinary_copy_upper_bound: int
    grouping_copy_ledger: CopyLedger
    cumulative_copy_ledger: CopyLedger
    merge_rounds: int
    transcript: Tuple[GroupingScan, ...] = ()
    hyperedge_witnesses: Tuple[HyperedgeWitness, ...] = ()
    grouping_complete: bool = True
    grouping_budget_truncated: bool = False
    sampling_policy: str = GroupingSamplingPolicy.CERTIFIED_TAU.value
    exploratory_query_count: int = 0
    refinement_top_up_count: int = 0
    min_shots_per_queried_tuple: int = 0
    max_shots_per_queried_tuple: int = 0
    mean_shots_per_queried_tuple: float = 0.0

    def __post_init__(self) -> None:
        if self.L < 0 or self.ell_grp < 1:
            raise ValueError("Grouping dimensions are invalid.")
        if self.success:
            if self.failure_reason is not None:
                raise ValueError("Successful grouping cannot have a failure reason.")
            flattened = tuple(sector for cluster in self.clusters for sector in cluster)
            if len(flattened) != self.L or len(set(flattened)) != self.L:
                raise ValueError("Successful grouping clusters must partition L sectors.")
        else:
            if not self.failure_reason:
                raise ValueError("Failed grouping must state a failure reason.")
            if self.theorem_grouping_preconditions_hold:
                raise ValueError("Failed grouping cannot claim theorem preconditions.")


@dataclass(frozen=True)
class LocalizationConfig:
    """Deterministic Phase-5 localization controls (no measurement budgets)."""

    allow_uncertified_grouping: bool = False
    return_details: bool = False
    verify_dense_unitary: bool = True
    max_dense_qubits: int = 8
    max_generated_group_size: int = _MAX_P4_GENERATED_GROUP_SIZE
    materialize_dense_clifford: bool = True
    enforce_model_block_bound: bool = True

    def __post_init__(self) -> None:
        for name in (
            "allow_uncertified_grouping",
            "return_details",
            "verify_dense_unitary",
            "materialize_dense_clifford",
            "enforce_model_block_bound",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool.")
        _validate_positive_integer("max_dense_qubits", self.max_dense_qubits)
        _validate_positive_integer(
            "max_generated_group_size", self.max_generated_group_size
        )


@dataclass(frozen=True)
class ClusterSymplecticStructure:
    """Restricted binary symplectic structure of one empirical group ``V_C``."""

    cluster: Tuple[int, ...]
    ordered_independent_axes: Tuple[str, ...]
    axis_vectors: Tuple[Tuple[int, ...], ...]
    span_basis: Tuple[int, ...]
    restricted_gram: np.ndarray = field(repr=False, compare=False)
    dimension: int = 0
    symplectic_rank: int = 0
    h_C: int = 0
    q_C: int = 0
    k_C: int = 0
    radical_basis: Tuple[Tuple[int, ...], ...] = ()
    hyperbolic_pairs: Tuple[
        Tuple[Tuple[int, ...], Tuple[int, ...]], ...
    ] = ()

    def __post_init__(self) -> None:
        gram = np.asarray(self.restricted_gram, dtype=np.uint8) % 2
        if gram.shape != (self.dimension, self.dimension):
            raise ValueError("Restricted symplectic Gram matrix has the wrong shape.")
        gram = gram.copy()
        gram.setflags(write=False)
        object.__setattr__(self, "restricted_gram", gram)


@dataclass(frozen=True)
class ClusterLocalization:
    """One empirical group's source pairs and allocated residual register."""

    cluster: Tuple[int, ...]
    h_C: int
    q_C: int
    k_C: int
    J_C: Tuple[int, ...]
    source_hyperbolic_pairs: Tuple[
        Tuple[Tuple[int, ...], Tuple[int, ...]], ...
    ]
    source_radical_axes: Tuple[Tuple[int, ...], ...]
    radical_partners: Tuple[Tuple[int, ...], ...]
    source_pair_indices: Tuple[int, ...]
    target_pair_qubits: Tuple[int, ...]


@dataclass(frozen=True)
class LocalizationResult:
    """Learner-visible Phase-5 simultaneous localization transcript."""

    success: bool
    failure_reason: Optional[str]
    n: int
    t: int
    m: int
    d: int
    clusters: Tuple[Tuple[int, ...], ...]
    structures: Tuple[ClusterSymplecticStructure, ...]
    cluster_localizations: Tuple[ClusterLocalization, ...]
    J_C: Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], ...]
    J_aux: Tuple[int, ...]
    K_rec: int
    source_basis: Optional[np.ndarray] = field(repr=False, compare=False)
    target_basis: Optional[np.ndarray] = field(repr=False, compare=False)
    residual_tableau: Optional[np.ndarray] = field(repr=False, compare=False)
    synthesis_tableau: Optional[np.ndarray] = field(repr=False, compare=False)
    full_tableau: Optional[np.ndarray] = field(repr=False, compare=False)
    gates: Tuple[Tuple, ...]
    U_rec: Optional[np.ndarray] = field(repr=False, compare=False)
    bar_U_rec: Optional[np.ndarray] = field(repr=False, compare=False)
    grouping_theorem_preconditions_hold: bool
    theorem_localization_preconditions_hold: bool
    handoff_valid: bool
    cross_group_direct_sum_holds: bool
    cross_group_symplectic_orthogonality_holds: bool
    global_pairing_holds: bool
    register_partition_holds: bool
    localization_guarantee_holds: bool
    localization_copy_count: int
    cumulative_copy_ledger: CopyLedger
    verification_transcript: Tuple[str, ...] = ()
    signed_clifford: Optional[SignedClifford] = field(
        default=None, repr=False, compare=False
    )
    empirical_max_cluster_size: int = 0
    assumed_d: int = 0
    model_bound_violated: bool = False
    oversize_clusters: Tuple[Tuple[Tuple[int, ...], int], ...] = ()
    reconstruction_proceeded_despite_model_bound_violation: bool = False

    def __post_init__(self) -> None:
        if self.n < 1 or not (0 <= self.t <= self.n) or self.m != self.n - self.t:
            raise ValueError("Localization dimensions are inconsistent.")
        if self.d < 1 or self.localization_copy_count != 0:
            raise ValueError("Phase-5 localization must have d>=1 and zero copy cost.")
        empirical_max = int(self.empirical_max_cluster_size)
        if empirical_max < 0:
            raise ValueError("empirical_max_cluster_size must be nonnegative.")
        if empirical_max == 0 and self.clusters:
            empirical_max = max(len(cluster) for cluster in self.clusters)
        oversize = tuple(
            (tuple(cluster), int(size))
            for cluster, size in self.oversize_clusters
        )
        if not oversize and self.clusters:
            oversize = tuple(
                (tuple(cluster), len(cluster))
                for cluster in self.clusters
                if len(cluster) > self.d
            )
        if any(size <= self.d or size != len(cluster) for cluster, size in oversize):
            raise ValueError("oversize_clusters metadata is inconsistent with d.")
        object.__setattr__(self, "empirical_max_cluster_size", empirical_max)
        object.__setattr__(self, "assumed_d", self.d)
        object.__setattr__(self, "model_bound_violated", empirical_max > self.d)
        object.__setattr__(self, "oversize_clusters", oversize)
        object.__setattr__(
            self,
            "reconstruction_proceeded_despite_model_bound_violation",
            bool(self.success and empirical_max > self.d),
        )
        compact_array_names = (
            "source_basis",
            "target_basis",
            "residual_tableau",
            "synthesis_tableau",
            "full_tableau",
        )
        dense_array_names = (
            "U_rec",
            "bar_U_rec",
        )
        array_names = compact_array_names + dense_array_names
        if self.success and self.signed_clifford is None and self.full_tableau is not None:
            try:
                compact = SignedClifford(
                    self.n, np.asarray(self.full_tableau), tuple(self.gates)
                )
            except ValueError:
                compact = None
            object.__setattr__(self, "signed_clifford", compact)
        if self.success:
            if self.failure_reason is not None:
                raise ValueError("Successful localization cannot have a failure reason.")
            if any(getattr(self, name) is None for name in compact_array_names):
                raise ValueError("Successful localization is missing compact Clifford data.")
            if not all(
                (
                    self.handoff_valid,
                    self.cross_group_direct_sum_holds,
                    self.cross_group_symplectic_orthogonality_holds,
                    self.global_pairing_holds,
                    self.register_partition_holds,
                    self.localization_guarantee_holds,
                )
            ):
                raise ValueError("Successful localization must satisfy every invariant.")
        else:
            if not self.failure_reason:
                raise ValueError("Failed localization must state a failure reason.")
            if any(getattr(self, name) is not None for name in array_names) or self.signed_clifford is not None:
                raise ValueError("Failed localization cannot expose fake Clifford data.")
            if self.structures or self.cluster_localizations or self.J_C or self.J_aux:
                raise ValueError("Failed localization cannot expose fake registers.")
        for name in array_names:
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value).copy()
                array.setflags(write=False)
                object.__setattr__(self, name, array)


class LocalizationInvariantError(ValueError):
    """Raised when learned Phase-3/4 structure cannot be localized safely."""


@dataclass(frozen=True)
class SyndromeConfig:
    """Fresh signed-measurement budget for peeled-generator syndromes."""

    zeta_sgn: float
    h_min: float
    M_sgn: Optional[int] = None
    return_details: bool = False

    def __post_init__(self) -> None:
        if not (0.0 < float(self.zeta_sgn) < 1.0):
            raise ValueError("zeta_sgn must lie in (0,1).")
        if not (0.5 < float(self.h_min) < 1.0):
            raise ValueError("h_min must lie in (1/2,1).")
        if self.M_sgn is not None:
            if (
                isinstance(self.M_sgn, bool)
                or not isinstance(self.M_sgn, (int, np.integer))
                or int(self.M_sgn) < 0
            ):
                raise ValueError("M_sgn must be a nonnegative integer or None.")
        if not isinstance(self.return_details, bool):
            raise TypeError("return_details must be bool.")


@dataclass(frozen=True)
class PauliMeasurementRecord:
    """One fresh physical single-Pauli measurement batch."""

    pauli: str
    shots: int
    seed: RngSeed
    outcomes: np.ndarray = field(repr=False, compare=False)
    frame: str = field(default="physical", init=False)
    provenance: DataProvenance = field(default=DataProvenance.EMPIRICAL, init=False)

    def __post_init__(self) -> None:
        shots = _validate_positive_integer("shots", self.shots)
        _validate_pauli_string(self.pauli, len(self.pauli))
        outcomes = np.asarray(self.outcomes, dtype=np.int8)
        if outcomes.shape != (shots,) or not np.all(np.isin(outcomes, (-1, 1))):
            raise ValueError("Pauli outcomes must be a length-shots +/-1 array.")
        outcomes = outcomes.copy()
        outcomes.setflags(write=False)
        object.__setattr__(self, "outcomes", outcomes)

    @property
    def empirical_mean(self) -> float:
        return float(self.outcomes.mean())


@dataclass(frozen=True)
class SyndromeResult:
    """Learner-visible signed syndrome estimate and exact copy accounting."""

    success: bool
    failure_reason: Optional[str]
    t: int
    syndrome_bits: Tuple[int, ...]
    empirical_means: Tuple[float, ...]
    M_sgn: int
    zeta_sgn: float
    h_min: float
    calibrated_M_sgn: int
    theorem_preconditions_hold: bool
    syndrome_sign_pool: int
    copy_ledger: CopyLedger
    cumulative_copy_ledger: CopyLedger
    provenance: DataProvenance
    records: Tuple[PauliMeasurementRecord, ...] = ()


@dataclass(frozen=True)
class TomographyConfig:
    """Accuracy allocation and numerical controls for empirical registers.

    Empty per-cluster maps select the manuscript's uniform allocation
    ``epsilon_C=epsilon_tom/Khat`` and ``zeta_C=zeta_tom/Khat``.
    Explicit maps are immutable tuples keyed by the stable cluster tuple.
    """

    epsilon_tom: float
    zeta_tom: float
    epsilon_by_cluster: Tuple[Tuple[Tuple[int, ...], float], ...] = ()
    zeta_by_cluster: Tuple[Tuple[Tuple[int, ...], float], ...] = ()
    allow_uncertified_localization: bool = False
    return_details: bool = False
    max_dense_qubits: int = 8
    numerical_tolerance: float = 1e-10
    materialize_localized_estimator: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.epsilon_tom) or self.epsilon_tom <= 0.0:
            raise ValueError("epsilon_tom must be finite and positive.")
        if not (0.0 < float(self.zeta_tom) < 1.0):
            raise ValueError("zeta_tom must lie in (0,1).")
        for name, values in (
            ("epsilon_by_cluster", self.epsilon_by_cluster),
            ("zeta_by_cluster", self.zeta_by_cluster),
        ):
            keys = [tuple(cluster) for cluster, _value in values]
            if len(keys) != len(set(keys)):
                raise ValueError(f"{name} contains duplicate cluster keys.")
        for _cluster, value in self.epsilon_by_cluster:
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("Every epsilon_C must be finite and positive.")
        for _cluster, value in self.zeta_by_cluster:
            if not (0.0 < float(value) < 1.0):
                raise ValueError("Every zeta_C must lie in (0,1).")
        for name in (
            "allow_uncertified_localization",
            "return_details",
            "materialize_localized_estimator",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool.")
        _validate_positive_integer("max_dense_qubits", self.max_dense_qubits)
        if not np.isfinite(self.numerical_tolerance) or self.numerical_tolerance <= 0.0:
            raise ValueError("numerical_tolerance must be finite and positive.")


@dataclass(frozen=True)
class RegisterTomographyBudget:
    """Exact manuscript Pauli-tomography schedule for one register."""

    cluster: Tuple[int, ...]
    J_C: Tuple[int, ...]
    k_C: int
    N_C_Pauli: int
    epsilon_C: float
    zeta_C: float
    tau_C_tom: float
    M_C_Pauli: int
    L_C: int


@dataclass(frozen=True)
class FixedBudgetRegisterTomographyBudget:
    """Actual fixed-shot schedule metadata without strict accuracy targets."""

    cluster: Tuple[int, ...]
    J_C: Tuple[int, ...]
    k_C: int
    total_nonidentity_paulis: int
    measured_pauli_count: int
    physical_round_budget: int
    realized_schedule_length: int
    min_shots_per_measured_pauli: int
    max_shots_per_measured_pauli: int
    complete_pauli_coverage: bool
    budget_truncated: bool

    def __post_init__(self) -> None:
        if self.k_C <= 0 or self.total_nonidentity_paulis != 4**self.k_C - 1:
            raise ValueError("Fixed-budget tomography dimensions are inconsistent.")
        if not 0 <= self.measured_pauli_count <= self.total_nonidentity_paulis:
            raise ValueError("Measured Pauli count is outside its valid range.")
        if min(
            self.physical_round_budget,
            self.realized_schedule_length,
            self.min_shots_per_measured_pauli,
            self.max_shots_per_measured_pauli,
        ) < 0:
            raise ValueError("Fixed-budget tomography counts must be nonnegative.")
        if self.realized_schedule_length > self.physical_round_budget:
            raise ValueError("A local schedule exceeds its physical-round budget.")
        if self.measured_pauli_count == 0:
            if self.min_shots_per_measured_pauli or self.max_shots_per_measured_pauli:
                raise ValueError("An empty schedule cannot report positive shots.")
        elif not (
            1 <= self.min_shots_per_measured_pauli
            <= self.max_shots_per_measured_pauli
        ):
            raise ValueError("Measured-Pauli shot range is invalid.")
        complete = self.measured_pauli_count == self.total_nonidentity_paulis
        if self.complete_pauli_coverage != complete:
            raise ValueError("complete_pauli_coverage disagrees with measured settings.")
        if self.budget_truncated != (not complete):
            raise ValueError("budget_truncated must mean incomplete Pauli coverage.")


@dataclass(frozen=True)
class RegisterTomographyRecord:
    """Signed outcomes collected for one register in the joint schedule."""

    cluster: Tuple[int, ...]
    J_C: Tuple[int, ...]
    schedule: Tuple[str, ...]
    outcomes_by_pauli: Tuple[Tuple[str, Tuple[int, ...]], ...]
    provenance: DataProvenance = field(default=DataProvenance.EMPIRICAL, init=False)


@dataclass(frozen=True)
class RegisterTomographyCountRecord:
    """Compressed per-Pauli shot counts and signed sums for one register."""

    cluster: Tuple[int, ...]
    J_C: Tuple[int, ...]
    schedule_runs: Tuple[Tuple[str, int], ...]
    sufficient_statistics: Tuple[Tuple[str, int, int], ...]
    provenance: DataProvenance = field(default=DataProvenance.EMPIRICAL, init=False)

    def __post_init__(self) -> None:
        if any(repetitions <= 0 for _pauli, repetitions in self.schedule_runs):
            raise ValueError("Tomography schedule runs must have positive lengths.")
        for pauli, shots, signed_sum in self.sufficient_statistics:
            _validate_pauli_string(pauli, len(self.J_C))
            _validate_positive_integer("shots", shots)
            if abs(int(signed_sum)) > shots or (shots - int(signed_sum)) % 2:
                raise ValueError("Tomography signed sums are incompatible with shot counts.")

    @property
    def pauli_coefficients(self) -> Mapping[str, float]:
        return MappingProxyType(
            {pauli: float(signed_sum / shots) for pauli, shots, signed_sum in self.sufficient_statistics}
        )


@dataclass(frozen=True)
class RegisterTomographyEstimate:
    """Linear and physical estimates for one localized empirical register."""

    cluster: Tuple[int, ...]
    J_C: Tuple[int, ...]
    k_C: int
    pauli_coefficients: Tuple[Tuple[str, float], ...]
    nu_hat_lin: qt.Qobj = field(repr=False, compare=False)
    nu_star: qt.Qobj = field(repr=False, compare=False)
    nu_hat: qt.Qobj = field(repr=False, compare=False)
    numerical_projection_error_bound: float = 0.0


@dataclass(frozen=True)
class TomographyResult:
    """Localized register tomography and residual product estimator."""

    success: bool
    failure_reason: Optional[str]
    Khat: int
    budgets: Tuple[
        Union[RegisterTomographyBudget, FixedBudgetRegisterTomographyBudget], ...
    ]
    records: Tuple[Union[RegisterTomographyRecord, RegisterTomographyCountRecord], ...]
    estimates: Tuple[RegisterTomographyEstimate, ...]
    J_aux: Tuple[int, ...]
    localized_empirical_estimator: Optional[qt.Qobj] = field(
        repr=False, compare=False
    )
    N_bp: int = 0
    sum_local_schedule_lengths: int = 0
    one_common_copy_per_round: bool = False
    theorem_preconditions_hold: bool = False
    block_tomography_pool: int = 0
    copy_ledger: CopyLedger = field(default_factory=CopyLedger)
    cumulative_copy_ledger: CopyLedger = field(default_factory=CopyLedger)
    provenance: DataProvenance = DataProvenance.EMPIRICAL
    schedule_transcript: Tuple[Tuple[int, Tuple[Tuple[Tuple[int, ...], str], ...]], ...] = ()
    sampling_work_units: int = 0
    simulation_backend: SimulationBackend = "legacy_shotwise"
    measured_pauli_counts: Tuple[Tuple[Tuple[int, ...], int], ...] = ()
    total_pauli_counts: Tuple[Tuple[Tuple[int, ...], int], ...] = ()
    budget_truncated: bool = False


@dataclass(frozen=True)
class RecoveredBlockTomographyConfig:
    """Combined Phase-6 configuration; no end-to-end decoding controls."""

    syndrome: SyndromeConfig
    tomography: TomographyConfig


@dataclass(frozen=True)
class RecoveredBlockTomographyResult:
    """Compact Phase-6 output in localized coordinates only."""

    success: bool
    failure_reason: Optional[str]
    syndrome: SyndromeResult
    tomography: TomographyResult
    syndrome_bits: Tuple[int, ...]
    register_estimates: Tuple[Tuple[Tuple[int, ...], Tuple[int, ...], qt.Qobj], ...]
    J_aux: Tuple[int, ...]
    cumulative_copy_ledger: CopyLedger
    theorem_preconditions_hold: bool


@dataclass(frozen=True)
class StageSeedLedger:
    """Stable master-seed split for the five stochastic learner stages."""

    master_seed: RngSeed
    peeling_seed: int
    syndrome_seed: int
    recovery_seed: int
    grouping_seed: int
    tomography_seed: int


@dataclass(frozen=True)
class WorstCaseCopyReservation:
    """Data-independent physical-copy reservation computed before sampling."""

    peeling_bell_pool: int
    recovery_bell_pool: int
    grouping_ordinary_pool: int
    syndrome_sign_pool: int
    block_tomography_pool: int
    total: int

    def __post_init__(self) -> None:
        values = (
            self.peeling_bell_pool,
            self.recovery_bell_pool,
            self.grouping_ordinary_pool,
            self.syndrome_sign_pool,
            self.block_tomography_pool,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
            for value in values
        ):
            raise ValueError("Worst-case pool reservations must be nonnegative integers.")
        if self.total != sum(int(value) for value in values):
            raise ValueError("Worst-case reservation total is inconsistent.")

    def as_dict(self) -> dict[str, int]:
        return {
            "peeling_bell_pool": int(self.peeling_bell_pool),
            "recovery_bell_pool": int(self.recovery_bell_pool),
            "grouping_ordinary_pool": int(self.grouping_ordinary_pool),
            "syndrome_sign_pool": int(self.syndrome_sign_pool),
            "block_tomography_pool": int(self.block_tomography_pool),
        }


@dataclass(frozen=True)
class EndToEndSchedule:
    """Complete learner-known calibrated schedule from ``n,d,epsilon,delta``."""

    branch: str
    n: int
    d: int
    epsilon: float
    delta: float
    R_ub: int
    A_d: int
    q_d: int
    Gamma_d: int
    epsilon_tom: float
    theta_0: float
    theta: float
    eta_s: Optional[float]
    lambda_0: float
    h_min: float
    h_max: float
    peeling_eta: float
    eta_test: float
    tau_kappa: float
    tau_mu: float
    ell_grp: int
    zeta_peel: float
    zeta_rank: float
    zeta_grp: float
    zeta_sgn: float
    zeta_tom: float
    M1: int
    tau_1: float
    M2: int
    tau_rank: float
    M_sgn: int
    N_test_max_wc: int
    grouping_per_query_copies: int
    N_grp_wc: int
    N_P_wc: int
    tau_bp_wc: float
    M_P_wc: int
    N_bp_wc: int
    d1_accepted_per_setting: int
    d1_attempts_per_setting: int
    reservation: WorstCaseCopyReservation


@dataclass(frozen=True)
class EndToEndConfig:
    """Execution controls; calibrated schedules remain the source of truth."""

    epsilon: float
    delta: float
    seed: RngSeed = None
    return_details: bool = False
    materialize_dense_estimator: bool = False
    max_dense_qubits: int = 8
    max_enumeration_qubits: int = MAX_SUPPORTED_END_TO_END_QUBITS
    max_dense_debug_qubits: Optional[int] = None
    enumeration_execution: EnumerationExecutionConfig = field(
        default_factory=EnumerationExecutionConfig
    )
    max_reserved_copies: Optional[int] = 10_000_000
    max_realized_copies: Optional[int] = None
    simulation_backend: SimulationBackend = "legacy_shotwise"
    execution_policy: Union[ExecutionPolicy, str] = ExecutionPolicy.STRICT
    fixed_budget_stage_weights: Optional[FixedBudgetStageWeights] = None
    fixed_budget_stage_caps: Optional[Tuple[Tuple[str, int], ...]] = None
    allow_uncertified_execution: bool = False
    peeling_override: Optional[PeelingConfig] = None
    recovery_override: Optional[RecoveryConfig] = None
    grouping_override: Optional[GroupingConfig] = None
    syndrome_override: Optional[SyndromeConfig] = None
    tomography_override: Optional[TomographyConfig] = None
    d1_accepted_per_setting_override: Optional[int] = None
    d1_attempts_per_setting_override: Optional[int] = None

    def __post_init__(self) -> None:
        if not (0.0 < float(self.epsilon) < 1.0):
            raise ValueError("epsilon must lie in (0,1).")
        if not (0.0 < float(self.delta) < 1.0):
            raise ValueError("delta must lie in (0,1).")
        _validate_seed(self.seed)
        for name in (
            "return_details",
            "materialize_dense_estimator",
            "allow_uncertified_execution",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool.")
        _validate_positive_integer("max_dense_qubits", self.max_dense_qubits)
        _validate_positive_integer(
            "max_enumeration_qubits", self.max_enumeration_qubits
        )
        if int(self.max_enumeration_qubits) > MAX_SUPPORTED_END_TO_END_QUBITS:
            raise ValueError(
                "max_enumeration_qubits cannot exceed the supported end-to-end "
                f"maximum of {MAX_SUPPORTED_END_TO_END_QUBITS}."
            )
        if self.max_dense_debug_qubits is None:
            object.__setattr__(self, "max_dense_debug_qubits", self.max_dense_qubits)
        else:
            _validate_positive_integer(
                "max_dense_debug_qubits", self.max_dense_debug_qubits
            )
        if not isinstance(self.enumeration_execution, EnumerationExecutionConfig):
            raise TypeError("enumeration_execution must be EnumerationExecutionConfig.")
        if self.max_reserved_copies is not None:
            _validate_positive_integer("max_reserved_copies", self.max_reserved_copies)
        if self.max_realized_copies is not None:
            _validate_positive_integer("max_realized_copies", self.max_realized_copies)
        _validate_simulation_backend(self.simulation_backend)
        try:
            policy = ExecutionPolicy(self.execution_policy)
        except ValueError as error:
            raise ValueError(f"Unsupported execution_policy: {self.execution_policy!r}.") from error
        object.__setattr__(self, "execution_policy", policy)
        if self.fixed_budget_stage_weights is not None and not isinstance(
            self.fixed_budget_stage_weights, FixedBudgetStageWeights
        ):
            raise TypeError("fixed_budget_stage_weights must be FixedBudgetStageWeights or None.")
        if self.fixed_budget_stage_caps is not None:
            raw_caps = tuple(self.fixed_budget_stage_caps)
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                for _stage, value in raw_caps
            ):
                raise TypeError("fixed_budget_stage_caps values must be integers.")
            caps = tuple((str(stage), int(value)) for stage, value in raw_caps)
            expected_stages = (
                "peeling",
                "recovery",
                "grouping",
                "syndrome",
                "tomography",
            )
            if tuple(stage for stage, _value in caps) != expected_stages:
                raise ValueError(
                    "fixed_budget_stage_caps must list every fixed-budget stage "
                    "once in canonical order."
                )
            if any(value < 0 for _stage, value in caps):
                raise ValueError("fixed_budget_stage_caps must be nonnegative.")
            object.__setattr__(self, "fixed_budget_stage_caps", caps)
        if policy is ExecutionPolicy.FIXED_BUDGET_GRACEFUL:
            if self.max_realized_copies is None:
                raise ValueError("fixed_budget_graceful requires max_realized_copies.")
            if self.fixed_budget_stage_caps is not None:
                if sum(
                    value for _stage, value in self.fixed_budget_stage_caps
                ) != int(self.max_realized_copies):
                    raise ValueError(
                        "Explicit fixed-budget stage caps must sum to "
                        "max_realized_copies."
                    )
            elif self.fixed_budget_stage_weights is None:
                object.__setattr__(self, "fixed_budget_stage_weights", FixedBudgetStageWeights())
        elif self.fixed_budget_stage_caps is not None:
            raise ValueError(
                "fixed_budget_stage_caps are available only under "
                "fixed_budget_graceful."
            )
        for name in (
            "d1_accepted_per_setting_override",
            "d1_attempts_per_setting_override",
        ):
            value = getattr(self, name)
            if value is not None:
                _validate_positive_integer(name, value)
        if (
            self.d1_accepted_per_setting_override is not None
            and self.d1_attempts_per_setting_override is not None
            and self.d1_attempts_per_setting_override
            < self.d1_accepted_per_setting_override
        ):
            raise ValueError("d1 attempted shots must be at least accepted shots.")

    @property
    def uses_execution_overrides(self) -> bool:
        return any(
            value is not None
            for value in (
                self.peeling_override,
                self.recovery_override,
                self.grouping_override,
                self.syndrome_override,
                self.tomography_override,
                self.d1_accepted_per_setting_override,
                self.d1_attempts_per_setting_override,
            )
        )


@dataclass(frozen=True)
class StructuralCertificate:
    """Learner-computable structural error certificate."""

    branch: str
    Khat: int
    epsilon_peel: float
    theta_rec: float
    E_peel: float
    E_miss: float
    beta_peel: float
    xi_s: float
    xi_eff: float
    F_d_xi_eff: float
    split_terms: Tuple[Tuple[Tuple[int, ...], int, float], ...]
    E_split_cert: float
    E_struct_cert: float


@dataclass(frozen=True)
class EndToEndCertificate:
    """Final trace-norm, failure, and reservation certificate."""

    structural: StructuralCertificate
    epsilon_tom: float
    certified_trace_norm_bound: float
    target_epsilon: float
    failure_budgets: Tuple[Tuple[str, float], ...]
    total_failure_bound: float
    target_delta: float
    theorem_preconditions: Tuple[Tuple[str, bool], ...]
    copy_reservation_valid: bool
    realized_within_reserved: bool
    theorem_certified: bool


@dataclass(frozen=True)
class CompactCEBPEstimator:
    """Compact physical estimator from Eq. supp-abstract-output-tuple."""

    n: int
    t: int
    m: int
    U_stab: Optional[np.ndarray] = field(repr=False, compare=False)
    bar_U_rec: Optional[np.ndarray] = field(repr=False, compare=False)
    syndrome_bits: Tuple[int, ...]
    register_estimates: Tuple[
        Tuple[Tuple[int, ...], Tuple[int, ...], qt.Qobj], ...
    ]
    J_aux: Tuple[int, ...]
    peeling_gates: Tuple[Tuple, ...] = ()
    recovery_gates: Tuple[Tuple, ...] = ()
    peeling_clifford: Optional[SignedClifford] = field(
        default=None, repr=False, compare=False
    )
    recovery_clifford: Optional[SignedClifford] = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.n < 1 or not (0 <= self.t <= self.n) or self.m != self.n - self.t:
            raise ValueError("Compact estimator dimensions are inconsistent.")
        if len(self.syndrome_bits) != self.t or any(
            bit not in (0, 1) for bit in self.syndrome_bits
        ):
            raise ValueError("Compact estimator syndrome is malformed.")
        if self.U_stab is None and self.peeling_clifford is None:
            raise ValueError("Compact estimator is missing its peeling Clifford.")
        if self.bar_U_rec is None and self.recovery_clifford is None:
            raise ValueError("Compact estimator is missing its recovery Clifford.")
        residual_registers = []
        for cluster, register, estimate in self.register_estimates:
            cluster = tuple(int(value) for value in cluster)
            register = tuple(int(value) for value in register)
            if not cluster or not register:
                raise ValueError(
                    "Compact empirical blocks require nonempty labels and registers."
                )
            if len(set(register)) != len(register):
                raise ValueError(
                    "A compact empirical register contains duplicate qubits."
                )
            qobj = _v1._state_to_qobj(estimate, len(register))
            density = qobj * qobj.dag() if qobj.isket else qobj
            if density.shape != (2 ** len(register), 2 ** len(register)):
                raise ValueError("A compact empirical block has the wrong dimension.")
            residual_registers.extend(register)
        auxiliary = tuple(int(value) for value in self.J_aux)
        if len(auxiliary) != len(set(auxiliary)):
            raise ValueError("Compact auxiliary register contains duplicate qubits.")
        residual_registers.extend(auxiliary)
        if sorted(residual_registers) != list(range(self.m)):
            raise ValueError(
                "Compact empirical and auxiliary registers must partition range(m)."
            )
        for name, value in (("U_stab", self.U_stab), ("bar_U_rec", self.bar_U_rec)):
            if value is None:
                continue
            value = np.asarray(value, dtype=complex)
            if value.shape != (2**self.n, 2**self.n):
                raise ValueError(f"{name} has the wrong dimension.")
            frozen = value.copy()
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)


@dataclass(frozen=True)
class ConditionalOneQubitTomographyResult:
    """Attempted-copy transcript for the manuscript's d=1 postselection path."""

    success: bool
    failure_reason: Optional[str]
    estimates: Tuple[RegisterTomographyEstimate, ...]
    J_aux: Tuple[int, ...]
    localized_empirical_estimator: Optional[qt.Qobj] = field(
        repr=False, compare=False
    )
    accepted_per_setting: Tuple[int, int, int] = (0, 0, 0)
    target_accepted_per_setting: int = 0
    attempts_per_setting: int = 0
    attempted_copies: int = 0
    copy_ledger: CopyLedger = field(default_factory=CopyLedger)
    theorem_preconditions_hold: bool = False


@dataclass(frozen=True)
class EndToEndResult:
    """Operational Phase-7 result, including explicit statistical failures."""

    success: bool
    failure_stage: Optional[str]
    failure_reason: Optional[str]
    branch: str
    n: int
    d: int
    epsilon: float
    delta: float
    schedule: EndToEndSchedule
    seed_ledger: StageSeedLedger
    worst_case_reservation: WorstCaseCopyReservation
    peeling: Optional[PeelingResult] = None
    syndrome: Optional[SyndromeResult] = None
    recovery: Optional[RecoveryResult] = None
    grouping: Optional[GroupingResult] = None
    localization: Optional[LocalizationResult] = None
    tomography: Optional[Union[TomographyResult, ConditionalOneQubitTomographyResult]] = None
    phase6_handoff: Optional[RecoveredBlockTomographyResult] = None
    compact_estimator: Optional[CompactCEBPEstimator] = None
    localized_estimator: Optional[qt.Qobj] = field(default=None, repr=False, compare=False)
    decoded_density: Optional[qt.Qobj] = field(default=None, repr=False, compare=False)
    structural_certificate: Optional[StructuralCertificate] = None
    end_to_end_certificate: Optional[EndToEndCertificate] = None
    realized_copy_ledger: CopyLedger = field(default_factory=CopyLedger)
    realized_total: int = 0
    reservation_slack: Optional[int] = None
    theorem_certified: bool = False
    estimator_available: bool = False
    execution_complete: bool = False
    budget_truncated: bool = False
    truncated_stages: Tuple[str, ...] = ()
    degradation_reason: Optional[str] = None
    copies_remaining: Optional[int] = None
    fixed_budget_stage_records: Tuple[FixedBudgetStageRecord, ...] = ()
    performance_diagnostics: Optional[PerformanceDiagnostics] = field(
        default=None, compare=False
    )
    empirical_max_cluster_size: int = 0
    assumed_d: int = 0
    model_bound_violated: bool = False
    oversize_clusters: Tuple[Tuple[Tuple[int, ...], int], ...] = ()
    reconstruction_proceeded_despite_model_bound_violation: bool = False

    def __post_init__(self) -> None:
        clusters = (
            tuple(self.grouping.clusters)
            if self.grouping is not None and self.grouping.success
            else ()
        )
        empirical_max = max((len(cluster) for cluster in clusters), default=0)
        oversize = tuple(
            (tuple(cluster), len(cluster))
            for cluster in clusters
            if len(cluster) > self.d
        )
        proceeded = bool(
            oversize
            and self.localization is not None
            and self.localization.success
            and self.compact_estimator is not None
        )
        object.__setattr__(self, "empirical_max_cluster_size", empirical_max)
        object.__setattr__(self, "assumed_d", self.d)
        object.__setattr__(self, "model_bound_violated", bool(oversize))
        object.__setattr__(self, "oversize_clusters", oversize)
        object.__setattr__(
            self,
            "reconstruction_proceeded_despite_model_bound_violation",
            proceeded,
        )


class ResidualCumulantInterface(Protocol):
    """Structural interface consumed by the common grouping algorithm."""

    provenance: DataProvenance
    realized_query_count: int
    realized_copies: int
    query_count_by_order: Mapping[int, int]
    copies_by_order: Mapping[int, int]

    def query(self, observables: Sequence[str]) -> float:
        """Return one real labeled mixed cumulant."""


@dataclass(frozen=True)
class CEBPOracleTruth:
    """Hidden simulation truth used only for validation.

    ``encoder_tableau`` uses V1's column convention: for a binary Pauli column
    ``a``, ``U_c^dagger P(a) U_c = +/- P(F a)``.
    """

    hidden_partition: Tuple[Tuple[int, ...], ...]
    latent_block_states: Tuple[qt.Qobj, ...] = field(repr=False, compare=False)
    encoder_gates: Tuple[Tuple, ...]
    encoder_tableau: np.ndarray = field(repr=False, compare=False)
    encoder_sampling: str
    encoder_steps: int
    latent_state_source: str
    _structured_state: StructuredCEBPState = field(repr=False, compare=False)
    max_dense_debug_qubits: int = 8
    _latent_product_state: Optional[qt.Qobj] = field(
        default=None, repr=False, compare=False
    )
    _encoder_unitary: Optional[np.ndarray] = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _validate_positive_integer(
            "max_dense_debug_qubits", self.max_dense_debug_qubits
        )

    @property
    def latent_product_state(self) -> qt.Qobj:
        """Bounded small-n convenience view of the global latent product."""

        return self.materialize_latent_product_debug(
            max_qubits=self.max_dense_debug_qubits
        )

    def materialize_latent_product_debug(self, *, max_qubits: int) -> qt.Qobj:
        """Explicitly materialize the global latent product under a real cap."""

        max_qubits = _validate_positive_integer("max_qubits", max_qubits)
        if self._structured_state.n > max_qubits:
            raise ValueError("dense latent-product debug resource guard exceeded.")
        if self._latent_product_state is None:
            value, _is_ket = _tensor_latent_blocks(self.latent_block_states)
            object.__setattr__(self, "_latent_product_state", value)
        return self._latent_product_state

    @property
    def encoder_unitary(self) -> np.ndarray:
        """Bounded small-n convenience view of the full-system encoder."""

        return self.materialize_encoder_unitary_debug(
            max_qubits=self.max_dense_debug_qubits
        )

    def materialize_encoder_unitary_debug(self, *, max_qubits: int) -> np.ndarray:
        """Explicitly materialize the encoder unitary under a real cap."""

        max_qubits = _validate_positive_integer("max_qubits", max_qubits)
        if self._structured_state.n > max_qubits:
            raise ValueError("dense encoder debug resource guard exceeded.")
        if self._encoder_unitary is None:
            value = self._structured_state.encoder.materialize_dense_debug(
                max_qubits=max_qubits
            )
            value.setflags(write=False)
            object.__setattr__(self, "_encoder_unitary", value)
        return self._encoder_unitary

    def latent_pauli_vector(self, physical_pauli: str) -> np.ndarray:
        """Return the phase-free latent Pauli vector for a physical Pauli."""
        n = len(self.hidden_partition_qubits)
        _validate_pauli_string(physical_pauli, n)
        column = _v1.pauli_to_symplectic_col(physical_pauli)
        return ((self.encoder_tableau @ column) % 2).astype(np.uint8)

    @property
    def hidden_partition_qubits(self) -> Tuple[int, ...]:
        return tuple(qubit for block in self.hidden_partition for qubit in block)

    def hidden_block_support(self, physical_pauli: str) -> Tuple[int, ...]:
        """Oracle-only hidden-block labels touched by a physical Pauli."""
        vector = self.latent_pauli_vector(physical_pauli)
        n = vector.size // 2
        support = {i for i in range(n) if vector[i] or vector[n + i]}
        return tuple(
            block_index
            for block_index, block in enumerate(self.hidden_partition)
            if any(qubit in support for qubit in block)
        )


@dataclass(frozen=True)
class CEBPInstance:
    """A simulated CEBP state plus isolated oracle validation metadata.

    ``is_ket`` describes the encoded state's representation and the eligible
    Bell-sampling backend.  It is not a mathematical-purity certificate.
    """

    n: int
    d: int
    measurement_source: SimulatorMeasurementSource
    is_ket: bool
    seed_ledger: SeedLedger
    copy_ledger: CopyLedger
    oracle_truth: CEBPOracleTruth = field(repr=False, compare=False)
    max_dense_debug_qubits: int = 8

    def __post_init__(self) -> None:
        _validate_positive_integer(
            "max_dense_debug_qubits", self.max_dense_debug_qubits
        )

    @property
    def state(self) -> qt.Qobj:
        """Bounded small-n convenience view, never used by learner stages."""

        return self.materialize_state_debug(
            max_qubits=self.max_dense_debug_qubits
        )

    def materialize_state_debug(self, *, max_qubits: int) -> qt.Qobj:
        """Explicitly materialize the encoded state without retaining a cache."""

        max_qubits = _validate_positive_integer("max_qubits", max_qubits)
        return self.measurement_source._state_for_backend(
            max_dense_debug_qubits=max_qubits
        )

    def learner_view(self) -> CEBPLearnerView:
        """Return an object with no hidden partition, latent states, or encoder."""
        return CEBPLearnerView(
            n=self.n,
            d=self.d,
            measurement_source=self.measurement_source,
        )


def _validate_positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _validate_seed(seed: RngSeed) -> RngSeed:
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer or None.")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be nonnegative.")
    return seed


def _validate_pauli_string(pauli: str, n: int) -> None:
    if not isinstance(pauli, str) or len(pauli) != n:
        raise ValueError(f"Pauli must be an IXYZ string of length {n}.")
    if any(character not in _v1.PAULI_CHARS for character in pauli):
        raise ValueError("Pauli strings may contain only I, X, Y, and Z.")


def _all_pauli_strings(n: int) -> Tuple[str, ...]:
    """Return the fixed lexicographic order I<X<Y<Z used by Phase 2."""
    return tuple("".join(chars) for chars in itertools.product(_v1.PAULI_CHARS, repeat=n))


def _all_pauli_scores_from_bell_outcomes(outcomes: np.ndarray) -> np.ndarray:
    """Evaluate every Bell parity by the exact local-character tensor transform."""
    values = np.asarray(outcomes, dtype=np.int8)
    sums = bell_score_sums_from_outcomes(values)
    scores = np.asarray(sums, dtype=np.float64) / values.shape[1]
    scores.setflags(write=False)
    return scores


def _compact_score_array(
    source: PauliScoreSource,
    n: int,
    execution: EnumerationExecutionConfig,
) -> Tuple[np.ndarray, float]:
    """Return one guarded score array and its common denominator.

    The byte guard covers the single required contiguous length-``4**n``
    primary array. Empirical Bell records retain exact int64 parity sums and
    use ``rounds`` as the denominator; exact/debug sources use one float64
    array with denominator one. The check occurs before either allocation or
    sufficient-statistic construction.
    """

    required_bytes = (4**n) * np.dtype(np.int64).itemsize
    if (
        execution.max_score_array_bytes is not None
        and required_bytes > execution.max_score_array_bytes
    ):
        raise ValueError(
            "all-Pauli score workspace exceeds max_score_array_bytes "
            f"(required={required_bytes}, limit={execution.max_score_array_bytes})."
        )
    if isinstance(source, BellScoreRecord):
        values = source.all_score_sums()
        denominator = float(source.rounds)
    elif callable(getattr(source, "all_scores", None)):
        values = np.asarray(source.all_scores(), dtype=np.float64)
        if values.shape != (4**n,) or not values.flags.c_contiguous:
            raise ValueError("all_scores() must return one contiguous length-4**n array.")
        if not np.all(np.isfinite(values)):
            raise ValueError("all_scores() returned a non-finite score.")
        if float(values.min()) < -1e-12 or float(values.max()) > 1.0 + 1e-12:
            raise ValueError("all_scores() returned a score outside [0,1].")
        denominator = 1.0
    else:
        values = np.empty(4**n, dtype=np.float64)
        for index in range(values.size):
            pauli = pauli_index_to_string(index, n)
            value = float(source.score(pauli))
            if not np.isfinite(value) or not (-1.0 - 1e-12 <= value <= 1.0 + 1e-12):
                raise ValueError(
                    f"Score source returned invalid value {value} for {pauli}."
                )
            values[index] = value
        values.setflags(write=False)
        denominator = 1.0
    return values, denominator


def _enforce_modeled_enumeration_workspace(
    execution: EnumerationExecutionConfig,
    estimate: EnumerationWorkspaceEstimate,
) -> None:
    """Apply the preferred peak-workspace cap with an actionable breakdown."""

    limit = execution.max_enumeration_workspace_bytes
    if limit is None or estimate.predicted_peak_bytes <= int(limit):
        return
    dominant = ", ".join(
        f"{name}={size}" for name, size in estimate.dominant_buffers
    )
    raise ValueError(
        f"{estimate.stage} modeled enumeration workspace exceeds "
        "max_enumeration_workspace_bytes "
        f"(n={estimate.n}, predicted={estimate.predicted_peak_bytes}, "
        f"limit={int(limit)}, peak_phase={estimate.peak_phase}, "
        f"safety_factor={estimate.safety_factor:g}, "
        f"dominant_buffers=[{dominant}])."
    )


def _workspace_estimate_for_score_source(
    source: PauliScoreSource,
    *,
    n: int,
    stage: str,
    execution: EnumerationExecutionConfig,
    return_details: bool,
) -> EnumerationWorkspaceEstimate:
    empirical = isinstance(source, BellScoreRecord)
    backend: SimulationBackend = (
        source.simulation_backend if empirical else "batched_counts"
    )
    return estimate_enumeration_workspace(
        n,
        stage,
        simulation_backend=backend,
        bell_rounds=source.rounds if empirical else None,
        structured_bell=False,
        empirical=empirical,
        return_details=return_details,
        safety_factor=execution.enumeration_workspace_safety_factor,
    )


def _score_threshold_cutoff(
    values: np.ndarray, denominator: float, threshold: float
) -> Union[int, float]:
    """Return a compact cutoff exactly matching ``value/denominator >= threshold``."""

    if not np.issubdtype(np.asarray(values).dtype, np.integer):
        return float(threshold)
    rounds = int(denominator)
    if rounds <= 0 or float(rounds) != float(denominator):
        raise ValueError("Integer score sufficient statistics need an integer denominator.")
    cutoff = int(math.ceil(float(threshold) * rounds))
    while float(cutoff) / rounds < float(threshold):
        cutoff += 1
    while float(cutoff - 1) / rounds >= float(threshold):
        cutoff -= 1
    return cutoff


def _preflight_exhaustive_empirical_stage(
    source: Union[CEBPLearnerView, SimulatorMeasurementSource],
    *,
    n: int,
    max_enumeration_qubits: int,
    execution: EnumerationExecutionConfig,
    stage: str,
    stage_kind: str,
    rounds: int,
    simulation_backend: SimulationBackend,
    return_details: bool,
) -> SimulatorMeasurementSource:
    """Reject deterministic implementation-memory limits before consuming copies.

    Guard precedence is enumeration qubits, legacy score-array cap, legacy
    structured-Bell cap, then the preferred modeled total-workspace cap.
    """

    measurement_source = _coerce_measurement_source(source)
    if measurement_source.n != n:
        raise ValueError(f"{stage} source dimension disagrees with n.")
    if n > max_enumeration_qubits:
        raise ValueError(
            f"{stage} exhaustively enumerates 4^n Paulis; n={n} exceeds "
            f"max_enumeration_qubits={max_enumeration_qubits}."
        )
    required_score_bytes = (4**n) * np.dtype(np.int64).itemsize
    if (
        execution.max_score_array_bytes is not None
        and required_score_bytes > execution.max_score_array_bytes
    ):
        raise ValueError(
            f"{stage} all-Pauli score workspace exceeds max_score_array_bytes "
            f"(required={required_score_bytes}, "
            f"limit={execution.max_score_array_bytes})."
        )
    required_bell_bytes = measurement_source.structured_bell_workspace_bytes
    if (
        required_bell_bytes is not None
        and execution.max_structured_bell_workspace_bytes is not None
        and required_bell_bytes
        > execution.max_structured_bell_workspace_bytes
    ):
        raise ValueError(
            f"{stage} structured Bell workspace exceeds "
            "max_structured_bell_workspace_bytes "
            f"(required={required_bell_bytes}, "
            f"limit={execution.max_structured_bell_workspace_bytes})."
        )
    estimate = estimate_enumeration_workspace(
        n,
        stage_kind,
        simulation_backend=simulation_backend,
        bell_rounds=rounds,
        structured_bell=required_bell_bytes is not None,
        empirical=True,
        return_details=return_details,
        safety_factor=execution.enumeration_workspace_safety_factor,
    )
    _enforce_modeled_enumeration_workspace(execution, estimate)
    return measurement_source


def bell_score_uniform_radius(rounds: int, n: int, zeta_bs: float) -> float:
    """Return the uniform Bell-score radius from Eq. (supp-tauM)."""
    rounds = _validate_positive_integer("rounds", rounds)
    n = _validate_positive_integer("n", n)
    if not (0.0 < float(zeta_bs) < 1.0):
        raise ValueError("zeta_bs must lie in (0,1).")
    return float(math.sqrt(2.0 * math.log(2.0 * (4 ** n) / zeta_bs) / rounds))


def _coerce_measurement_source(
    source: Union[CEBPLearnerView, SimulatorMeasurementSource],
) -> SimulatorMeasurementSource:
    if isinstance(source, CEBPLearnerView):
        if source.n != source.measurement_source.n:
            raise ValueError("Learner-view and measurement-source sizes disagree.")
        return source.measurement_source
    if isinstance(source, SimulatorMeasurementSource):
        return source
    raise TypeError(
        "Empirical measurement requires CEBPLearnerView or SimulatorMeasurementSource."
    )


def sample_bell_scores(
    source: Union[CEBPLearnerView, SimulatorMeasurementSource],
    rounds: int,
    *,
    frame: str = "physical",
    zeta_bs: float = 0.05,
    seed: RngSeed = None,
    pool_name: str = "peeling_bell_pool",
    simulation_backend: SimulationBackend = "legacy_shotwise",
    max_structured_bell_workspace_bytes: Optional[int] = None,
) -> BellScoreRecord:
    """Collect one learner-facing Bell pool and expose reusable score queries."""
    measurement_source = _coerce_measurement_source(source)
    rounds = _validate_positive_integer("rounds", rounds)
    seed = _validate_seed(seed)
    if not frame:
        raise ValueError("frame must be nonempty.")
    if pool_name not in ("peeling_bell_pool", "recovery_bell_pool"):
        raise ValueError("pool_name must identify peeling or recovery.")
    simulation_backend = _validate_simulation_backend(simulation_backend)

    if max_structured_bell_workspace_bytes is not None:
        _validate_positive_integer(
            "max_structured_bell_workspace_bytes",
            max_structured_bell_workspace_bytes,
        )
    outcomes, counts, stage_wall_times = measurement_source._sample_bell_for_backend(
        rounds,
        seed,
        simulation_backend=simulation_backend,
        max_structured_bell_workspace_bytes=max_structured_bell_workspace_bytes,
    )
    backend = "pure_ket" if measurement_source.is_ket else "mixed_density"

    return BellScoreRecord(
        n=measurement_source.n,
        rounds=rounds,
        frame=frame,
        uniform_radius=bell_score_uniform_radius(rounds, measurement_source.n, zeta_bs),
        zeta_bs=float(zeta_bs),
        backend=backend,
        seed=seed,
        pool_name=pool_name,
        copy_ledger=CopyLedger(((pool_name, 2 * rounds),)),
        outcomes=outcomes,
        simulation_backend=simulation_backend,
        category_counts=counts,
        stage_wall_times=stage_wall_times,
    )


def debug_exact_pauli_expectation(state: StateLike, pauli: str) -> float:
    """Oracle/debug-only exact signed Pauli expectation."""
    _validate_pauli_string(pauli, len(pauli))
    state_qobj = _v1._state_to_qobj(state, len(pauli))
    operator = _v1._qutip_pauli_op(len(pauli), pauli)
    value = complex(qt.expect(operator, state_qobj))
    if abs(value.imag) > 1e-9:
        raise RuntimeError("A Hermitian Pauli expectation acquired an imaginary part.")
    return float(value.real)


def debug_exact_pauli_score(state: StateLike, pauli: str) -> float:
    """Oracle/debug-only exact squared Pauli expectation."""
    expectation = debug_exact_pauli_expectation(state, pauli)
    return float(expectation * expectation)


def debug_exact_score_source(
    instance: CEBPInstance,
    *,
    frame: str = "physical",
) -> DebugExactPauliScoreSource:
    """Build an explicitly oracle/debug score source from a simulator instance."""
    if not isinstance(instance, CEBPInstance):
        raise TypeError("debug_exact_score_source requires a CEBPInstance.")
    source = instance.measurement_source
    if source._structured_state is not None:
        return DebugExactPauliScoreSource(
            n=instance.n, _structured_state=source._structured_state, frame=frame
        )
    return DebugExactPauliScoreSource(
        n=instance.n, _debug_state=source._state, frame=frame
    )


def _infer_state_qubits(state: StateLike) -> int:
    if isinstance(state, qt.Qobj):
        rows, columns = state.shape
    else:
        array = np.asarray(state)
        if array.ndim == 1:
            rows, columns = array.size, 1
        elif array.ndim == 2:
            rows, columns = array.shape
        else:
            raise ValueError("A block state must be a ket vector or square density matrix.")

    if columns not in (1, rows):
        raise ValueError("A block state must be a ket vector or square density matrix.")
    if rows <= 0 or rows & (rows - 1):
        raise ValueError("A block-state dimension must be a positive power of two.")
    return rows.bit_length() - 1


def _validate_density_psd(state: qt.Qobj, *, tolerance: float = 1e-10) -> None:
    eigenvalues = np.linalg.eigvalsh(state.full())
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError("Density matrices must be positive semidefinite.")


def _normalize_block_state(state: StateLike, qubits: int) -> qt.Qobj:
    normalized = _v1._state_to_qobj(state, qubits)
    if not normalized.isket:
        _validate_density_psd(normalized)
    return normalized


def _random_pure_block(qubits: int, rng: np.random.Generator) -> qt.Qobj:
    dimension = 2 ** qubits
    vector = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
    vector /= np.linalg.norm(vector)
    return qt.Qobj(vector.reshape((dimension, 1)), dims=[[2] * qubits, [1] * qubits])


def _random_mixed_block(qubits: int, rng: np.random.Generator) -> qt.Qobj:
    """Draw a full-rank Hilbert--Schmidt random density matrix."""
    dimension = 2 ** qubits
    ginibre = (
        rng.normal(size=(dimension, dimension))
        + 1j * rng.normal(size=(dimension, dimension))
    )
    matrix = ginibre @ ginibre.conj().T
    matrix /= np.trace(matrix)
    return qt.Qobj(matrix, dims=[[2] * qubits, [2] * qubits])


def _random_block_sizes(n: int, d: int, rng: np.random.Generator) -> Tuple[int, ...]:
    remaining = n
    sizes = []
    while remaining:
        size = int(rng.integers(1, min(d, remaining) + 1))
        sizes.append(size)
        remaining -= size
    return tuple(sizes)


def _validate_block_sizes(n: int, d: int, block_sizes: Iterable[int]) -> Tuple[int, ...]:
    sizes = tuple(_validate_positive_integer("block size", size) for size in block_sizes)
    if not sizes:
        raise ValueError("block_sizes must contain at least one block.")
    if sum(sizes) != n:
        raise ValueError(f"block_sizes must sum to n={n}.")
    if max(sizes) > d:
        raise ValueError(f"Every block size must be at most d={d}.")
    return sizes


def _contiguous_partition(block_sizes: Sequence[int]) -> Tuple[Tuple[int, ...], ...]:
    partition = []
    start = 0
    for size in block_sizes:
        partition.append(tuple(range(start, start + size)))
        start += size
    return tuple(partition)


def _tensor_latent_blocks(block_states: Sequence[qt.Qobj]) -> Tuple[qt.Qobj, bool]:
    all_pure = all(state.isket for state in block_states)
    if all_pure:
        product = qt.tensor(list(block_states)).unit()
        n = sum(_infer_state_qubits(state) for state in block_states)
        product.dims = [[2] * n, [1] * n]
        return product, True

    density_blocks = [state * state.dag() if state.isket else state for state in block_states]
    product = qt.tensor(density_blocks)
    n = sum(_infer_state_qubits(state) for state in block_states)
    product.dims = [[2] * n, [2] * n]
    product = 0.5 * (product + product.dag())
    product = product / product.tr()
    return product, False


def _symplectic_column_to_pauli(column: np.ndarray) -> str:
    column = np.asarray(column, dtype=np.uint8).reshape(-1) % 2
    if column.size % 2:
        raise ValueError("A symplectic column must have even length.")
    n = column.size // 2
    characters = []
    for x, z in zip(column[:n], column[n:]):
        characters.append(("I", "Z", "X", "Y")[2 * int(x) + int(z)])
    return "".join(characters)


def tableau_unitary_convention_holds(
    tableau: np.ndarray,
    unitary: np.ndarray,
    *,
    atol: float = 1e-8,
) -> bool:
    """Check ``U^dagger P(a) U = +/- P(Fa)`` on canonical generators."""
    tableau = np.asarray(tableau, dtype=np.uint8) % 2
    if tableau.ndim != 2 or tableau.shape[0] != tableau.shape[1] or tableau.shape[0] % 2:
        return False
    n = tableau.shape[0] // 2
    dimension = 2 ** n
    unitary = np.asarray(unitary, dtype=complex)
    if unitary.shape != (dimension, dimension):
        return False
    if not _v1.is_symplectic(tableau):
        return False
    if not np.allclose(unitary.conj().T @ unitary, np.eye(dimension), atol=atol):
        return False

    for column_index in range(2 * n):
        source = np.zeros(2 * n, dtype=np.uint8)
        source[column_index] = 1
        source_op = _v1._qutip_pauli_op(n, _symplectic_column_to_pauli(source)).full()
        target = (tableau @ source) % 2
        target_op = _v1._qutip_pauli_op(n, _symplectic_column_to_pauli(target)).full()
        actual = unitary.conj().T @ source_op @ unitary
        overlap = np.trace(target_op.conj().T @ actual) / dimension
        if abs(overlap.imag) > atol or abs(abs(overlap.real) - 1.0) > 10 * atol:
            return False
        if not np.allclose(actual, np.sign(overlap.real) * target_op, atol=10 * atol):
            return False
    return True


def canonical_gf2_span_basis(
    paulis: Iterable[str],
    n: int,
) -> Tuple[int, ...]:
    """Return a canonical RREF basis for a phase-free Pauli span.

    Rows use bit positions ``[x_0,...,x_{n-1},z_0,...,z_{n-1}]``.  Comparing
    these tuples compares actual subspaces, not only their dimensions.
    """
    n = _validate_positive_integer("n", n)
    symplectic = _v1.Symplectic(n)
    vectors = []
    for pauli in paulis:
        _validate_pauli_string(pauli, n)
        vector = symplectic.to_int(pauli)
        if vector:
            vectors.append(vector)
    if not vectors:
        return ()
    matrix = np.array(
        [[(vector >> bit) & 1 for bit in range(2 * n)] for vector in vectors],
        dtype=np.uint8,
    )
    reduced, _ = _v1._gf2_rref(matrix)
    canonical = []
    for row in reduced:
        if row.any():
            canonical.append(sum(int(bit) << index for index, bit in enumerate(row)))
    return tuple(canonical)


def gf2_pauli_spans_equal(
    left: Iterable[str],
    right: Iterable[str],
    n: int,
) -> bool:
    """Compare Pauli spans by canonical subspace representation."""
    return canonical_gf2_span_basis(left, n) == canonical_gf2_span_basis(right, n)


def _first_independent_pauli_basis(paulis: Iterable[str], n: int) -> Tuple[str, ...]:
    """Select the first independent basis in the supplied deterministic order."""
    span = _v1.GF2Basis()
    symplectic = _v1.Symplectic(n)
    selected = []
    for pauli in paulis:
        vector = symplectic.to_int(pauli)
        if vector and not span.contains(vector):
            span.add(vector)
            selected.append(pauli)
    return tuple(selected)


def _pauli_family_is_isotropic(paulis: Sequence[str], n: int) -> bool:
    symplectic = _v1.Symplectic(n)
    vectors = [symplectic.to_int(pauli) for pauli in paulis]
    return all(
        symplectic.commutes_int(vectors[left], vectors[right])
        for left in range(len(vectors))
        for right in range(left + 1, len(vectors))
    )


def peeling_clifford_mapping_holds(
    generators: Sequence[str],
    U_stab: np.ndarray,
    *,
    atol: float = 1e-8,
) -> bool:
    """Check the manuscript orientation ``U_stab^dag g_j U_stab = Z_j``."""
    if not generators:
        unitary = np.asarray(U_stab, dtype=complex)
        dimension = unitary.shape[0]
        return unitary.ndim == 2 and unitary.shape == (dimension, dimension) and np.allclose(
            unitary, np.eye(dimension), atol=atol
        )
    n = len(generators[0])
    unitary = np.asarray(U_stab, dtype=complex)
    if unitary.shape != (2 ** n, 2 ** n):
        return False
    Uq = qt.Qobj(unitary, dims=[[2] * n, [2] * n])
    for index, generator in enumerate(generators):
        _validate_pauli_string(generator, n)
        target = ["I"] * n
        target[index] = "Z"
        actual = Uq.dag() * _v1._qutip_pauli_op(n, generator) * Uq
        if not np.allclose(actual.full(), _v1._qutip_pauli_op(n, "".join(target)).full(), atol=atol):
            return False
    return True


def _synthesize_peeling_clifford(
    generators: Sequence[str],
    n: int,
    *,
    materialize_dense: bool = True,
    max_dense_debug_qubits: int = 8,
) -> Tuple[Optional[np.ndarray], np.ndarray, Tuple[Tuple, ...], SignedClifford]:
    if not generators:
        compact = SignedClifford.identity(n)
        return (
            np.eye(2 ** n, dtype=complex) if materialize_dense else None,
            np.eye(2 * n, dtype=np.uint8),
            (),
            compact,
        )
    columns = np.column_stack(
        [_v1.pauli_to_symplectic_col(generator) for generator in generators]
    ).astype(np.uint8)
    tableau = _v1.complete_isotropic_to_symplectic(columns)
    gates = _v1.synthesize_clifford_from_tableau(tableau)
    compact = SignedClifford.from_gates(n, tuple(gates))
    sign_gates = []
    for index, generator in enumerate(generators):
        target = "I" * index + "Z" + "I" * (n - index - 1)
        image = compact.conjugate(generator)
        if image.pauli != target or image.phase_exponent not in (0, 2):
            raise RuntimeError("Peeling synthesis returned an invalid signed image.")
        if image.phase_exponent == 2:
            sign_gates.append(("X", index))
            compact = SignedClifford.from_gates(n, tuple(gates + sign_gates))
    all_gates = tuple(gates + sign_gates)
    if not _v1.is_symplectic(tableau):
        raise RuntimeError("Peeling synthesis returned a nonsymplectic tableau.")
    if not np.array_equal(tableau[:, n : n + len(generators)], columns):
        raise RuntimeError("Peeling tableau does not preserve the prescribed Z columns.")
    for index, generator in enumerate(generators):
        target = "I" * index + "Z" + "I" * (n - index - 1)
        image = compact.conjugate(generator)
        if image.phase_exponent != 0 or image.pauli != target:
            raise RuntimeError(
                "Synthesized compact U_stab violates U_stab^dag g_j U_stab = Z_j."
            )
    unitary = None
    if materialize_dense:
        unitary = compact.materialize_dense_debug(
            max_qubits=max_dense_debug_qubits
        )
        if not peeling_clifford_mapping_holds(generators, unitary):
            raise RuntimeError(
                "Dense U_stab violates U_stab^dag g_j U_stab = Z_j."
            )
    return unitary, tableau, all_gates, compact


def _source_metadata(source: PauliScoreSource) -> Tuple[int, DataProvenance, float, int, CopyLedger, str]:
    try:
        n = _validate_positive_integer("score-source n", source.n)
        provenance = DataProvenance(source.provenance)
        radius = float(source.uniform_radius)
        rounds = int(source.bell_rounds)
        ledger = source.copy_ledger
        frame = str(source.frame)
    except AttributeError as error:
        raise TypeError("score_source does not implement the PauliScoreSource interface.") from error
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("The score-source uniform radius must be finite and nonnegative.")
    if rounds < 0 or not isinstance(ledger, CopyLedger) or not frame:
        raise ValueError("Invalid score-source metadata.")
    if provenance is DataProvenance.EMPIRICAL and ledger.total != 2 * rounds:
        raise ValueError("Empirical score sources must report exactly 2*M copies.")
    if provenance is DataProvenance.EXACT and (rounds != 0 or ledger.total != 0):
        raise ValueError("Exact/debug score sources cannot report learner copies.")
    return n, provenance, radius, rounds, ledger, frame


def certified_stabilizer_peeling_v2(
    score_source: PauliScoreSource,
    config: PeelingConfig,
) -> PeelingResult:
    """Run Algorithm ``alg:supp-stabilizer-peeling`` on one score source.

    All ``4^n`` scores are evaluated once, then reused at every grid point.
    This is the manuscript-faithful small-system implementation and is guarded
    by ``config.max_enumeration_qubits``.
    """
    if not isinstance(config, PeelingConfig):
        raise TypeError("config must be a PeelingConfig.")
    n, provenance, tau_1, rounds, ledger, frame = _source_metadata(score_source)
    if n > config.max_enumeration_qubits:
        raise ValueError(
            "Certified peeling exhaustively enumerates 4^n Paulis; "
            f"n={n} exceeds max_enumeration_qubits={config.max_enumeration_qubits}."
        )
    _enforce_modeled_enumeration_workspace(
        config.enumeration_execution,
        _workspace_estimate_for_score_source(
            score_source,
            n=n,
            stage="peeling",
            execution=config.enumeration_execution,
            return_details=config.return_details,
        ),
    )

    score_started = time.perf_counter()
    score_values, score_denominator = _compact_score_array(
        score_source, n, config.enumeration_execution
    )
    score_elapsed = time.perf_counter() - score_started

    grid = config.threshold_grid
    delta_h = float(config.h_max - config.h_min)
    theorem_grid = config.eta <= delta_h / (4.0 * (2 * n + 1)) + 1e-15
    theorem_tau = tau_1 <= delta_h / (8.0 * (2 * n + 1)) + 1e-15
    transcript = []

    threshold_started = time.perf_counter()
    for h in grid:
        inner_cutoff = _score_threshold_cutoff(
            score_values, score_denominator, h + tau_1
        )
        outer_cutoff = _score_threshold_cutoff(
            score_values, score_denominator, h - tau_1
        )
        inner_count, inner_span = threshold_span_reduction(
            score_values,
            inner_cutoff,
            n,
            workers=config.enumeration_execution.workers,
            chunk_size=config.enumeration_execution.chunk_size,
        )
        outer_count, outer_span = threshold_span_reduction(
            score_values,
            outer_cutoff,
            n,
            workers=config.enumeration_execution.workers,
            chunk_size=config.enumeration_execution.chunk_size,
        )
        spans_equal = inner_span == outer_span
        if not spans_equal:
            transcript.append(
                PeelingThresholdAttempt(
                    h=h,
                    inner_count=inner_count,
                    outer_count=outer_count,
                    inner_rank=len(inner_span),
                    outer_rank=len(outer_span),
                    spans_equal=False,
                    isotropic=None,
                    accepted=False,
                    reason="inner_outer_spans_differ",
                )
            )
            continue

        selected_span = StreamingGF2Basis(2 * n)
        selected_indices = []
        accepted_cutoff = inner_cutoff
        for index in range(score_values.size):
            if score_values[index] < accepted_cutoff:
                continue
            vector = pauli_index_to_symplectic_int(index, n)
            if vector and selected_span.add(vector):
                selected_indices.append(index)
                if selected_span.rank == len(inner_span):
                    break
        generators = tuple(
            pauli_index_to_string(index, n) for index in selected_indices
        )
        if canonical_gf2_span_basis(generators, n) != inner_span:
            raise RuntimeError("Deterministic inner-set basis does not span the certified subspace.")
        isotropic = _pauli_family_is_isotropic(generators, n)
        if not isotropic:
            transcript.append(
                PeelingThresholdAttempt(
                    h=h,
                    inner_count=inner_count,
                    outer_count=outer_count,
                    inner_rank=len(inner_span),
                    outer_rank=len(outer_span),
                    spans_equal=True,
                    isotropic=False,
                    accepted=False,
                    reason="certified_span_non_isotropic",
                )
            )
            continue

        unitary, tableau, gates, compact_clifford = _synthesize_peeling_clifford(
            generators,
            n,
            materialize_dense=config.materialize_dense_clifford,
            max_dense_debug_qubits=config.max_dense_debug_qubits,
        )
        transcript.append(
            PeelingThresholdAttempt(
                h=h,
                inner_count=inner_count,
                outer_count=outer_count,
                inner_rank=len(inner_span),
                outer_rank=len(outer_span),
                spans_equal=True,
                isotropic=True,
                accepted=True,
                reason="accepted",
            )
        )
        lambda_ = 1.0 - h
        vectors = tuple(
            tuple(int(bit) for bit in _v1.pauli_to_symplectic_col(generator))
            for generator in generators
        )
        inner_details = ()
        outer_details = ()
        if config.return_details:
            inner_details = tuple(
                pauli_index_to_string(int(index), n)
                for index in np.flatnonzero(score_values >= accepted_cutoff)
            )
            outer_details = tuple(
                pauli_index_to_string(int(index), n)
                for index in np.flatnonzero(
                    score_values >= outer_cutoff
                )
            )
        diagnostics = EnumerationDiagnostics(
                n=n,
                enumeration_size=4**n,
                score_dtype=str(score_values.dtype),
                score_array_bytes=int(score_values.nbytes),
                worker_count=config.enumeration_execution.workers,
                chunk_size=config.enumeration_execution.chunk_size,
                stage_wall_times=(
                    *(
                        score_source.stage_wall_times
                        if isinstance(score_source, BellScoreRecord)
                        else ()
                    ),
                    ("bell_score_transform", score_elapsed),
                    ("threshold_scan", time.perf_counter() - threshold_started),
                ),
        )
        return PeelingResult(
            success=True,
            failure_reason=None,
            h=h,
            lambda_=lambda_,
            tau_1=tau_1,
            t=len(generators),
            generators=generators,
            generator_symplectic_vectors=vectors,
            certified_span_basis=inner_span,
            inner_set=inner_details,
            outer_set=outer_details,
            inner_span_basis=inner_span,
            outer_span_basis=outer_span,
            U_stab=unitary,
            tableau=tableau,
            gates=gates,
            epsilon_peel=len(generators) * lambda_ / 2.0,
            M1=rounds,
            copy_ledger=ledger,
            score_provenance=provenance,
            score_frame=frame,
            threshold_grid=grid,
            transcript=tuple(transcript),
            theorem_grid_condition=theorem_grid,
            theorem_tau_condition=theorem_tau,
            signed_clifford=compact_clifford,
            enumeration_diagnostics=diagnostics,
        )

    diagnostics = EnumerationDiagnostics(
            n=n,
            enumeration_size=4**n,
            score_dtype=str(score_values.dtype),
            score_array_bytes=int(score_values.nbytes),
            worker_count=config.enumeration_execution.workers,
            chunk_size=config.enumeration_execution.chunk_size,
            stage_wall_times=(
                *(
                    score_source.stage_wall_times
                    if isinstance(score_source, BellScoreRecord)
                    else ()
                ),
                ("bell_score_transform", score_elapsed),
                ("threshold_scan", time.perf_counter() - threshold_started),
            ),
    )
    return PeelingResult(
        success=False,
        failure_reason="no_certified_threshold",
        h=None,
        lambda_=None,
        tau_1=tau_1,
        t=None,
        generators=(),
        generator_symplectic_vectors=(),
        certified_span_basis=(),
        inner_set=(),
        outer_set=(),
        inner_span_basis=(),
        outer_span_basis=(),
        U_stab=None,
        tableau=None,
        gates=(),
        epsilon_peel=None,
        M1=rounds,
        copy_ledger=ledger,
        score_provenance=provenance,
        score_frame=frame,
        threshold_grid=grid,
        transcript=tuple(transcript),
        theorem_grid_condition=theorem_grid,
        theorem_tau_condition=theorem_tau,
        signed_clifford=None,
        enumeration_diagnostics=diagnostics,
    )


@_time_pipeline_stage("peeling")
def empirical_certified_stabilizer_peeling(
    source: Union[CEBPLearnerView, SimulatorMeasurementSource],
    config: PeelingConfig,
    *,
    seed: RngSeed = None,
    simulation_backend: SimulationBackend = "legacy_shotwise",
) -> PeelingResult:
    """Collect the fresh M1 Bell pool and run empirical certified peeling."""
    if config.M1 is None:
        raise ValueError("Empirical certified peeling requires config.M1.")
    measurement_source = _coerce_measurement_source(source)
    _preflight_exhaustive_empirical_stage(
        source,
        n=measurement_source.n,
        max_enumeration_qubits=config.max_enumeration_qubits,
        execution=config.enumeration_execution,
        stage="Certified peeling",
        stage_kind="peeling",
        rounds=config.M1,
        simulation_backend=simulation_backend,
        return_details=config.return_details,
    )
    record = sample_bell_scores(
        source,
        config.M1,
        frame="physical",
        zeta_bs=config.zeta_bs,
        seed=seed,
        simulation_backend=simulation_backend,
        max_structured_bell_workspace_bytes=(
            config.enumeration_execution.max_structured_bell_workspace_bytes
        ),
    )
    return certified_stabilizer_peeling_v2(record, config)


def debug_exact_certified_stabilizer_peeling(
    instance: CEBPInstance,
    config: PeelingConfig,
) -> PeelingResult:
    """Oracle/debug structural peeling with exact scores and zero copy cost."""
    return certified_stabilizer_peeling_v2(debug_exact_score_source(instance), config)


def phase_free_pauli_product(*paulis: str) -> str:
    """Multiply equal-length Paulis modulo their Hermitian phase."""
    if not paulis:
        raise ValueError("At least one Pauli is required.")
    n = len(paulis[0])
    for pauli in paulis:
        _validate_pauli_string(pauli, n)
    symplectic = _v1.Symplectic(n)
    value = 0
    for pauli in paulis:
        value ^= symplectic.to_int(pauli)
    return symplectic.to_str(value)


_SINGLE_QUBIT_SIGNED_PRODUCT = {
    ("I", "I"): (0, "I"),
    ("I", "X"): (0, "X"),
    ("I", "Y"): (0, "Y"),
    ("I", "Z"): (0, "Z"),
    ("X", "I"): (0, "X"),
    ("Y", "I"): (0, "Y"),
    ("Z", "I"): (0, "Z"),
    ("X", "X"): (0, "I"),
    ("Y", "Y"): (0, "I"),
    ("Z", "Z"): (0, "I"),
    ("X", "Y"): (1, "Z"),
    ("Y", "X"): (3, "Z"),
    ("Y", "Z"): (1, "X"),
    ("Z", "Y"): (3, "X"),
    ("Z", "X"): (1, "Y"),
    ("X", "Z"): (3, "Y"),
}


def signed_pauli_product(*paulis: str) -> SignedPauliProduct:
    """Multiply equal-length canonical Paulis with their exact ``i`` phase."""
    if not paulis:
        raise ValueError("At least one Pauli is required.")
    n = len(paulis[0])
    for pauli in paulis:
        _validate_pauli_string(pauli, n)
    phase = 0
    value = "I" * n
    for pauli in paulis:
        characters = []
        for left, right in zip(value, pauli):
            local_phase, character = _SINGLE_QUBIT_SIGNED_PRODUCT[(left, right)]
            phase = (phase + local_phase) % 4
            characters.append(character)
        value = "".join(characters)
    return SignedPauliProduct(phase, value)


def hermitian_pauli_product(*paulis: str) -> Tuple[int, str]:
    """Reduce a Hermitian Pauli product to ``(sign, canonical_pauli)``.

    Products with an imaginary phase are not Hermitian and are rejected.  A
    mutually commuting tuple always has a real phase, including subset
    products used by the Phase-4 signed-moment layer.
    """
    product = signed_pauli_product(*paulis)
    if product.phase_exponent not in (0, 2):
        raise ValueError("Pauli product has phase +/-i and is not Hermitian.")
    return (1 if product.phase_exponent == 0 else -1, product.pauli)


def _validate_recovery_for_grouping(
    recovery: RecoveryResult,
    *,
    allow_uncalibrated_recovery: bool,
) -> bool:
    """Enforce the Phase-4 empirical recovery boundary.

    The returned flag records whether theorem recovery preconditions truly
    hold.  An explicit debug/manual override permits execution but never
    changes or upgrades that flag.
    """
    if not isinstance(recovery, RecoveryResult):
        raise TypeError("recovery must be a RecoveryResult.")
    if not recovery.success:
        raise RecoveryPreconditionError("PRECONDITION recovery_failed")
    theorem_valid = bool(recovery.theorem_recovery_preconditions_hold)
    if not theorem_valid and not allow_uncalibrated_recovery:
        raise RecoveryPreconditionError(
            "PRECONDITION recovery_not_theorem_calibrated"
        )
    return theorem_valid


def validate_grouping_against_recovery(
    grouping: GroupingResult,
    recovery: RecoveryResult,
    *,
    enforce_cluster_size_bound: bool = True,
) -> bool:
    """Validate the exact learned-structure handoff into localization.

    ``GroupingResult`` can validate only its own cardinality bookkeeping.  The
    stronger boundary check here compares its stable sector IDs with the
    actual ``RecoveryResult`` and rechecks the Phase-3 symplectic invariants.
    No oracle labels or state data enter this validation.
    """
    if not isinstance(grouping, GroupingResult):
        raise TypeError("grouping must be a GroupingResult.")
    if not isinstance(recovery, RecoveryResult):
        raise TypeError("recovery must be a RecoveryResult.")
    if not recovery.success:
        raise RecoveryPreconditionError("PRECONDITION recovery_failed")
    if not grouping.success:
        raise RecoveryPreconditionError("PRECONDITION grouping_failed")

    sectors = tuple(recovery.sectors)
    sector_ids = tuple(sector.sector_id for sector in sectors)
    if len(set(sector_ids)) != len(sector_ids):
        raise ValueError("Recovery contains duplicate sector IDs.")
    if grouping.L != len(sectors):
        raise ValueError("Grouping L does not match the recovery sector count.")
    if not recovered_sectors_are_symplectically_valid(sectors):
        raise ValueError("Recovery sectors are not symplectically valid.")
    if any(len(member) != recovery.m for sector in sectors for member in sector.members):
        raise ValueError("Recovery sector axes do not have residual length m.")
    if tuple(recovery.independent_axes) != recovered_sector_axes(sectors):
        raise ValueError("Recovery independent_axes disagrees with its sectors.")
    if tuple(recovery.recovered_span_basis) != recovered_sector_span_basis(
        sectors, recovery.m
    ):
        raise ValueError("Recovery span basis disagrees with its sectors.")

    flattened = []
    for cluster in grouping.clusters:
        if not cluster:
            raise ValueError("Grouping contains an empty cluster.")
        if len(set(cluster)) != len(cluster):
            raise ValueError("Grouping cluster contains duplicate sector IDs.")
        if enforce_cluster_size_bound and len(cluster) > grouping.ell_grp:
            raise ValueError("Grouping cluster exceeds ell_grp.")
        flattened.extend(cluster)
    if len(flattened) != len(set(flattened)):
        raise ValueError("A recovery sector appears in more than one cluster.")
    grouped_ids = set(flattened)
    recovery_ids = set(sector_ids)
    unknown = grouped_ids - recovery_ids
    missing = recovery_ids - grouped_ids
    if unknown:
        raise ValueError(f"Grouping contains unknown sector IDs: {sorted(unknown)}.")
    if missing:
        raise ValueError(f"Grouping omits recovery sector IDs: {sorted(missing)}.")
    return True


def recovered_sector_axes(sectors: Sequence[RecoveredSector]) -> Tuple[str, ...]:
    """Return ``Ax(G)`` in stable sector order (derived y axes excluded)."""
    return tuple(axis for sector in sectors for axis in sector.independent_axes)


def recovered_sector_span_basis(
    sectors: Sequence[RecoveredSector],
    m: Optional[int] = None,
) -> Tuple[int, ...]:
    """Return the canonical GF(2) representation of ``V(G)``."""
    axes = recovered_sector_axes(sectors)
    if axes:
        inferred = len(axes[0])
        if m is not None and m != inferred:
            raise ValueError("Requested residual size disagrees with sector axes.")
        return canonical_gf2_span_basis(axes, inferred)
    if m is not None and m < 0:
        raise ValueError("m must be nonnegative.")
    return ()


def pauli_in_recovered_span(pauli: str, sectors: Sequence[RecoveredSector]) -> bool:
    """Test phase-free GF(2) membership in ``V(G)``."""
    _validate_pauli_string(pauli, len(pauli))
    if not sectors:
        return set(pauli) <= {"I"}
    axes = recovered_sector_axes(sectors)
    if any(len(axis) != len(pauli) for axis in axes):
        raise ValueError("Pauli and recovered sectors have different sizes.")
    basis = _v1.GF2Basis()
    symplectic = _v1.Symplectic(len(pauli))
    for axis in axes:
        basis.add(symplectic.to_int(axis))
    return basis.contains(symplectic.to_int(pauli))


def recovered_sectors_are_symplectically_valid(
    sectors: Sequence[RecoveredSector],
) -> bool:
    """Check the four validity clauses preceding Algorithm Phase 3."""
    sectors = tuple(sectors)
    if not sectors:
        return True
    m = len(sectors[0].x)
    if len({sector.sector_id for sector in sectors}) != len(sectors):
        return False
    if any(len(member) != m for sector in sectors for member in sector.members):
        return False
    symplectic = _v1.Symplectic(m)
    for sector in sectors:
        if sector.completed:
            if sector.y != phase_free_pauli_product(sector.x, sector.z):
                return False
            if not all(
                symplectic.anticommutes(left, right)
                for left, right in itertools.combinations(sector.members, 2)
            ):
                return False
    for left_index, left in enumerate(sectors):
        for right in sectors[left_index + 1 :]:
            if not all(
                symplectic.commutes(left_member, right_member)
                for left_member in left.members
                for right_member in right.members
            ):
                return False
    axes = recovered_sector_axes(sectors)
    return len(canonical_gf2_span_basis(axes, m)) == len(axes)


def dress_residual_pauli(
    pauli: str,
    sectors: Sequence[RecoveredSector],
) -> str:
    """Apply Eq. ``eq:supp-triple-dressing-map`` phase-free."""
    _validate_pauli_string(pauli, len(pauli))
    if not recovered_sectors_are_symplectically_valid(sectors):
        raise ValueError("Dressing requires a symplectically valid sector collection.")
    symplectic = _v1.Symplectic(len(pauli))
    dressed = pauli
    for sector in sectors:
        if not sector.completed:
            continue
        original = symplectic.to_int(pauli)
        value = symplectic.to_int(dressed)
        if symplectic.anticommutes_int(original, symplectic.to_int(sector.z)):
            value ^= symplectic.to_int(sector.x)
        if symplectic.anticommutes_int(original, symplectic.to_int(sector.x)):
            value ^= symplectic.to_int(sector.z)
        dressed = symplectic.to_str(value)
    if any(
        not symplectic.commutes(dressed, member)
        for sector in sectors
        if sector.completed
        for member in sector.members
    ):
        raise RuntimeError("Completed-sector dressing failed to neutralize a triple.")
    return dressed


def singleton_anticommutation_set(
    sectors: Sequence[RecoveredSector],
    dressed_pauli: str,
) -> Tuple[int, ...]:
    """Return stable sector IDs in ``A_G(P^circ)``."""
    _validate_pauli_string(dressed_pauli, len(dressed_pauli))
    symplectic = _v1.Symplectic(len(dressed_pauli))
    return tuple(
        sector.sector_id
        for sector in sectors
        if not sector.completed and symplectic.anticommutes(dressed_pauli, sector.x)
    )


def _add_dressed_recovery_direction(
    sectors: Sequence[RecoveredSector],
    dressed_pauli: str,
) -> Tuple[Tuple[RecoveredSector, ...], str, Tuple[int, ...]]:
    """Append a singleton or perform the manuscript pivot-rebasing step."""
    old = tuple(sectors)
    if set(dressed_pauli) <= {"I"} or pauli_in_recovered_span(dressed_pauli, old):
        raise RuntimeError("Recovery attempted to add a dependent dressed direction.")
    anticommuting = singleton_anticommutation_set(old, dressed_pauli)
    if not anticommuting:
        new_id = 0 if not old else max(sector.sector_id for sector in old) + 1
        updated = old + (RecoveredSector(new_id, dressed_pauli),)
        action = "append_singleton"
        affected = (new_id,)
    else:
        pivot_id = anticommuting[0]
        pivot = next(sector for sector in old if sector.sector_id == pivot_id)
        anticommuting_set = set(anticommuting)
        rebased = []
        for sector in old:
            if sector.sector_id == pivot_id:
                rebased.append(
                    RecoveredSector(
                        sector_id=pivot_id,
                        x=sector.x,
                        z=dressed_pauli,
                        y=phase_free_pauli_product(sector.x, dressed_pauli),
                    )
                )
            elif sector.sector_id in anticommuting_set:
                rebased.append(
                    RecoveredSector(
                        sector_id=sector.sector_id,
                        x=phase_free_pauli_product(sector.x, pivot.x),
                    )
                )
            else:
                rebased.append(sector)
        updated = tuple(rebased)
        action = "complete_pivot"
        affected = anticommuting
        if recovered_sector_span_basis(old) != recovered_sector_span_basis(
            tuple(sector for sector in updated if sector.sector_id != pivot_id)
            + (RecoveredSector(pivot_id, pivot.x),)
        ):
            raise RuntimeError("Pivot rebasing failed to preserve the old span.")
    if not recovered_sectors_are_symplectically_valid(updated):
        raise RuntimeError("Recovery step violated a symplectic sector invariant.")
    return updated, action, affected


def parse_peeled_full_pauli(
    full_pauli: str,
    t: int,
) -> Tuple[Optional[Tuple[int, ...]], Optional[str], Optional[str]]:
    """Parse ``A=Z^c tensor P`` or return a deterministic skip reason."""
    n = len(full_pauli)
    _validate_pauli_string(full_pauli, n)
    if not (0 <= t <= n):
        raise ValueError("t must lie in [0,n].")
    prefix = full_pauli[:t]
    if any(character not in ("I", "Z") for character in prefix):
        return None, None, "not_prefix_Z_form"
    bits = tuple(1 if character == "Z" else 0 for character in prefix)
    residual = full_pauli[t:]
    if not residual or set(residual) <= {"I"}:
        return bits, residual, "prefix_only"
    return bits, residual, None


def _peeling_dimension(peeling: PeelingResult) -> int:
    if not peeling.success or peeling.tableau is None or peeling.t is None:
        raise RecoveryPreconditionError("PRECONDITION peeling_failed")
    size = int(np.asarray(peeling.tableau).shape[0])
    if size % 2:
        raise RecoveryPreconditionError("PRECONDITION invalid_peeling_tableau")
    return size // 2


def _recovery_margin_flags(
    peeling: PeelingResult,
    theta: float,
    tau_rank: float,
) -> Tuple[bool, bool]:
    if not peeling.success or peeling.lambda_ is None:
        return False, False
    threshold = float(theta) - float(tau_rank) > float(peeling.lambda_)
    ranking = (
        float(peeling.lambda_) * (float(theta) - float(tau_rank))
        - 2.0 * float(tau_rank)
        > 0.0
    )
    return threshold, ranking


def _enforce_recovery_preconditions(
    peeling: PeelingResult,
    config: RecoveryConfig,
    n: int,
    tau_rank: float,
) -> Tuple[bool, bool]:
    if _peeling_dimension(peeling) != n:
        raise RecoveryPreconditionError("PRECONDITION peeling_dimension_mismatch")
    if not peeling.theorem_certified and not config.allow_uncalibrated_peeling:
        raise RecoveryPreconditionError("PRECONDITION peeling_not_theorem_calibrated")
    threshold, ranking = _recovery_margin_flags(peeling, config.theta, tau_rank)
    if not threshold and not config.allow_margin_failure:
        raise RecoveryPreconditionError("PRECONDITION threshold_margin_failed")
    if not ranking and not config.allow_margin_failure:
        raise RecoveryPreconditionError("PRECONDITION ranking_gap_margin_failed")
    return threshold, ranking


def sample_peeled_bell_scores(
    source: Union[CEBPLearnerView, SimulatorMeasurementSource],
    peeling: PeelingResult,
    rounds: int,
    *,
    zeta_rank: float,
    seed: RngSeed = None,
    simulation_backend: SimulationBackend = "legacy_shotwise",
    max_structured_bell_workspace_bytes: Optional[int] = None,
) -> BellScoreRecord:
    """Collect a fresh Bell pool on ``U_stab^dag rho U_stab``."""
    measurement_source = _coerce_measurement_source(source)
    n = _peeling_dimension(peeling)
    if measurement_source.n != n:
        raise RecoveryPreconditionError("PRECONDITION source_dimension_mismatch")
    backend_source = _peeled_measurement_source(measurement_source, peeling)
    return sample_bell_scores(
        backend_source,
        rounds,
        frame="peeled",
        zeta_bs=zeta_rank,
        seed=seed,
        pool_name="recovery_bell_pool",
        simulation_backend=simulation_backend,
        max_structured_bell_workspace_bytes=max_structured_bell_workspace_bytes,
    )


def debug_exact_peeled_score_source(
    instance: CEBPInstance,
    peeling: PeelingResult,
) -> DebugExactPauliScoreSource:
    """Build a zero-copy exact/debug score source for the actual peeled state."""
    if not isinstance(instance, CEBPInstance):
        raise TypeError("debug_exact_peeled_score_source requires a CEBPInstance.")
    n = _peeling_dimension(peeling)
    if instance.n != n:
        raise RecoveryPreconditionError("PRECONDITION instance_dimension_mismatch")
    peeled = _peeled_measurement_source(instance.learner_view(), peeling)
    if peeled._structured_state is not None:
        return DebugExactPauliScoreSource(
            n=n, _structured_state=peeled._structured_state, frame="peeled"
        )
    return DebugExactPauliScoreSource(n=n, _debug_state=peeled._state, frame="peeled")


def _rank_recovery_survivors(
    score_values: np.ndarray,
    survivor_cutoff: Union[int, float],
) -> np.ndarray:
    """Return the exact manuscript ranking with auditable temporary lifetimes."""

    threshold_mask = score_values >= survivor_cutoff
    survivor_indices = np.flatnonzero(threshold_mask)
    del threshold_mask
    # Identity is fixed index zero and always has empirical/exact score one.
    # A slice avoids the former full filtered copy without changing survivors.
    if survivor_indices.size and survivor_indices[0] == 0:
        survivor_indices = survivor_indices[1:]
    elif np.any(survivor_indices == 0):
        survivor_indices = survivor_indices[survivor_indices != 0]
    indexed_scores = score_values[survivor_indices]
    descending_score_key = np.negative(indexed_scores)
    del indexed_scores
    ranking = np.lexsort((survivor_indices, descending_score_key))
    del descending_score_key
    ranked_survivors = survivor_indices[ranking]
    del ranking, survivor_indices
    ranked_survivors.setflags(write=False)
    return ranked_survivors


def rank_guided_sector_recovery(
    score_source: PauliScoreSource,
    peeling: PeelingResult,
    config: RecoveryConfig,
) -> RecoveryResult:
    """Run Algorithm ``alg:supp-block-rank-guided-recovery`` literally."""
    if not isinstance(config, RecoveryConfig):
        raise TypeError("config must be a RecoveryConfig.")
    n, provenance, tau_rank, rounds, ledger, frame = _source_metadata(score_source)
    if frame != "peeled":
        raise RecoveryPreconditionError("PRECONDITION invalid_score_frame")
    if n > config.max_enumeration_qubits:
        raise ValueError(
            "Rank-guided recovery exhaustively enumerates 4^n Paulis; "
            f"n={n} exceeds max_enumeration_qubits={config.max_enumeration_qubits}."
        )
    _enforce_modeled_enumeration_workspace(
        config.enumeration_execution,
        _workspace_estimate_for_score_source(
            score_source,
            n=n,
            stage="recovery",
            execution=config.enumeration_execution,
            return_details=config.return_details,
        ),
    )
    if provenance is DataProvenance.EMPIRICAL:
        if ledger.as_dict() != {"recovery_bell_pool": 2 * rounds}:
            raise ValueError("Empirical recovery requires a distinct recovery_bell_pool.")
        if config.M2 is not None and rounds != config.M2:
            raise ValueError("Recovery score rounds disagree with config.M2.")
        if isinstance(score_source, BellScoreRecord) and not np.isclose(
            score_source.zeta_bs, config.zeta_rank
        ):
            raise ValueError("Recovery score failure budget disagrees with zeta_rank.")
    threshold_margin, ranking_margin = _enforce_recovery_preconditions(
        peeling, config, n, tau_rank
    )
    t = int(peeling.t)
    m = n - t

    preparation_started = time.perf_counter()
    score_values, score_denominator = _compact_score_array(
        score_source, n, config.enumeration_execution
    )
    survivor_cutoff = _score_threshold_cutoff(
        score_values, score_denominator, config.theta
    )
    survivor_indices = _rank_recovery_survivors(score_values, survivor_cutoff)
    preparation_elapsed = time.perf_counter() - preparation_started

    sectors: Tuple[RecoveredSector, ...] = ()
    candidates = []
    transcript = []
    recovered_basis = StreamingGF2Basis(2 * max(m, 1))
    residual_symplectic = _v1.Symplectic(m) if m else None
    ranked_loop_started = time.perf_counter()
    for full_index in survivor_indices:
        full_pauli = pauli_index_to_string(int(full_index), n)
        score = float(score_values[int(full_index)] / score_denominator)
        prefix_bits, residual, skip_reason = parse_peeled_full_pauli(full_pauli, t)
        if config.return_details:
            candidates.append(
                RecoveryCandidate(full_pauli, score, prefix_bits, residual)
            )
        if skip_reason is not None:
            if config.return_details:
                transcript.append(
                    RecoveryStep(full_pauli, score, residual, None, skip_reason)
                )
            continue
        assert residual_symplectic is not None and residual is not None
        residual_vector = residual_symplectic.to_int(residual)
        if recovered_basis.contains(residual_vector):
            if config.return_details:
                transcript.append(
                    RecoveryStep(full_pauli, score, residual, None, "already_in_span")
                )
            continue
        dressed = dress_residual_pauli(residual, sectors)
        sectors, action, affected = _add_dressed_recovery_direction(sectors, dressed)
        if not recovered_basis.add(residual_vector):
            raise RuntimeError("Recovery step did not add exactly the candidate span.")
        if config.return_details:
            transcript.append(
                RecoveryStep(full_pauli, score, residual, dressed, action, affected)
            )

    completeness = True
    for full_index in survivor_indices:
        full_pauli = pauli_index_to_string(int(full_index), n)
        _bits, residual, reason = parse_peeled_full_pauli(full_pauli, t)
        if reason is None:
            assert residual is not None and residual_symplectic is not None
            if not recovered_basis.contains(residual_symplectic.to_int(residual)):
                completeness = False
                break
    if not recovered_sectors_are_symplectically_valid(sectors):
        raise RuntimeError("Final recovery collection is not symplectically valid.")
    if not completeness:
        raise RuntimeError("Threshold-span completeness check failed.")

    cumulative = CopyLedger(peeling.copy_ledger.entries + ledger.entries)
    theorem_preconditions = (
        peeling.theorem_certified and threshold_margin and ranking_margin
    )
    diagnostics = EnumerationDiagnostics(
            n=n,
            enumeration_size=4**n,
            score_dtype=str(score_values.dtype),
            score_array_bytes=int(score_values.nbytes),
            worker_count=config.enumeration_execution.workers,
            chunk_size=config.enumeration_execution.chunk_size,
            stage_wall_times=(
                *(
                    score_source.stage_wall_times
                    if isinstance(score_source, BellScoreRecord)
                    else ()
                ),
                ("recovery_preparation_sort", preparation_elapsed),
                ("serial_ranked_recovery_loop", time.perf_counter() - ranked_loop_started),
            ),
    )
    return RecoveryResult(
        success=True,
        failure_reason=None,
        n=n,
        t=t,
        m=m,
        theta=float(config.theta),
        tau_rank=float(tau_rank),
        M2=rounds,
        zeta_rank=float(config.zeta_rank),
        lambda_=float(peeling.lambda_),
        sectors=sectors,
        independent_axes=recovered_sector_axes(sectors),
        recovered_span_basis=recovered_sector_span_basis(sectors, m),
        ranked_survivor_count=int(survivor_indices.size),
        score_provenance=provenance,
        score_frame=frame,
        threshold_margin_holds=threshold_margin,
        ranking_gap_margin_holds=ranking_margin,
        theorem_recovery_preconditions_hold=theorem_preconditions,
        threshold_span_complete=completeness,
        copy_ledger=ledger,
        cumulative_copy_ledger=cumulative,
        ranked_candidates=tuple(candidates),
        transcript=tuple(transcript),
        enumeration_diagnostics=diagnostics,
    )


@_time_pipeline_stage("recovery")
def empirical_rank_guided_sector_recovery(
    source: Union[CEBPLearnerView, SimulatorMeasurementSource],
    peeling: PeelingResult,
    config: RecoveryConfig,
    *,
    seed: RngSeed = None,
    simulation_backend: SimulationBackend = "legacy_shotwise",
) -> RecoveryResult:
    """Collect the fresh peeled M2 pool and perform empirical recovery."""
    if config.M2 is None:
        raise ValueError("Empirical rank-guided recovery requires config.M2.")
    n = _peeling_dimension(peeling)
    tau_rank = bell_score_uniform_radius(config.M2, n, config.zeta_rank)
    _enforce_recovery_preconditions(peeling, config, n, tau_rank)
    _preflight_exhaustive_empirical_stage(
        source,
        n=n,
        max_enumeration_qubits=config.max_enumeration_qubits,
        execution=config.enumeration_execution,
        stage="Rank-guided recovery",
        stage_kind="recovery",
        rounds=config.M2,
        simulation_backend=simulation_backend,
        return_details=config.return_details,
    )
    record = sample_peeled_bell_scores(
        source,
        peeling,
        config.M2,
        zeta_rank=config.zeta_rank,
        seed=seed,
        simulation_backend=simulation_backend,
        max_structured_bell_workspace_bytes=(
            config.enumeration_execution.max_structured_bell_workspace_bytes
        ),
    )
    return rank_guided_sector_recovery(record, peeling, config)


def debug_exact_rank_guided_sector_recovery(
    instance: CEBPInstance,
    peeling: PeelingResult,
    config: RecoveryConfig,
    *,
    allow_uncalibrated_peeling: bool = True,
) -> RecoveryResult:
    """Run exact zero-copy structural recovery on the actual peeled state."""
    debug_config = replace(
        config,
        M2=None,
        allow_uncalibrated_peeling=allow_uncalibrated_peeling,
    )
    return rank_guided_sector_recovery(
        debug_exact_peeled_score_source(instance, peeling),
        peeling,
        debug_config,
    )


def debug_oracle_residual_block_labels(
    instance: CEBPInstance,
    peeling: PeelingResult,
    residual_pauli: str,
) -> Tuple[int, ...]:
    """Oracle-only quotient-aware hidden labels over every Z-prefix lift."""
    n = _peeling_dimension(peeling)
    t = int(peeling.t)
    m = n - t
    _validate_pauli_string(residual_pauli, m)
    labels = set()
    for bits in itertools.product((0, 1), repeat=t):
        prefix = "".join("Z" if bit else "I" for bit in bits)
        peeled = prefix + residual_pauli
        peeled_column = _v1.pauli_to_symplectic_col(peeled)
        physical_column = (np.asarray(peeling.tableau) @ peeled_column) % 2
        physical = _symplectic_column_to_pauli(physical_column)
        support = instance.oracle_truth.hidden_block_support(physical)
        if len(support) == 1:
            labels.add(support[0])
    return tuple(sorted(labels))


def debug_oracle_sector_block_labels(
    instance: CEBPInstance,
    peeling: PeelingResult,
    recovery: RecoveryResult,
) -> Tuple[Tuple[int, ...], ...]:
    """Oracle-only block-label diagnostics, never used by learner recovery."""
    return tuple(
        tuple(
            sorted(
                {
                    label
                    for member in sector.members
                    for label in debug_oracle_residual_block_labels(
                        instance, peeling, member
                    )
                }
            )
        )
        for sector in recovery.sectors
    )


def generated_cluster_pauli_group(
    cluster: Iterable[int],
    sectors: Sequence[RecoveredSector],
    *,
    max_group_size: int = _MAX_P4_GENERATED_GROUP_SIZE,
) -> GeneratedPauliGroup:
    """Enumerate ``P(A)`` from independent sector axes in deterministic order."""
    cluster_ids = tuple(sorted(int(sector_id) for sector_id in cluster))
    if not cluster_ids or len(set(cluster_ids)) != len(cluster_ids):
        raise ValueError("A cluster must contain distinct sector IDs.")
    lookup = {sector.sector_id: sector for sector in sectors}
    if len(lookup) != len(tuple(sectors)) or any(
        sector_id not in lookup for sector_id in cluster_ids
    ):
        raise ValueError("Cluster refers to an unknown or duplicate sector ID.")
    selected = tuple(lookup[sector_id] for sector_id in cluster_ids)
    if not recovered_sectors_are_symplectically_valid(selected):
        raise ValueError("Generated groups require symplectically valid sectors.")
    m = len(selected[0].x)
    basis = canonical_gf2_span_basis(recovered_sector_axes(selected), m)
    rank = len(basis)
    cardinality = 1 << rank
    if cardinality > int(max_group_size):
        raise ValueError(
            f"Generated group size {cardinality} exceeds guard {max_group_size}."
        )
    if cardinality > 4 ** len(cluster_ids):
        raise RuntimeError("Generated group violates the manuscript cardinality bound.")
    symplectic = _v1.Symplectic(m)
    values = []
    for mask in range(cardinality):
        value = 0
        for index, axis in enumerate(basis):
            if (mask >> index) & 1:
                value ^= axis
        values.append(symplectic.to_str(value))
    paulis = tuple(sorted(set(values)))
    if len(paulis) != cardinality:
        raise RuntimeError("GF(2) generated-group enumeration produced duplicates.")
    return GeneratedPauliGroup(cluster_ids, rank, paulis)


def labeled_set_partitions(q: int) -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
    """Return deterministic partitions of the labeled positions ``range(q)``."""
    if isinstance(q, bool) or not isinstance(q, (int, np.integer)) or int(q) < 0:
        raise ValueError("q must be a nonnegative integer.")
    q = int(q)
    partitions: Tuple[Tuple[Tuple[int, ...], ...], ...] = ((),)
    for label in range(q):
        updated = []
        for partition in partitions:
            for block_index in range(len(partition)):
                blocks = list(partition)
                blocks[block_index] = blocks[block_index] + (label,)
                updated.append(tuple(blocks))
            updated.append(partition + ((label,),))
        partitions = tuple(updated)
    return partitions


@lru_cache(maxsize=None)
def _stirling_second_kind(q: int, k: int) -> int:
    if q == k == 0:
        return 1
    if q <= 0 or k <= 0 or k > q:
        return 0
    return _stirling_second_kind(q - 1, k - 1) + k * _stirling_second_kind(q - 1, k)


def cumulant_gamma(q: int) -> int:
    """Return the ordered-Bell perturbation constant ``Gamma_q`` exactly."""
    q = _validate_positive_integer("q", q)
    return sum(_stirling_second_kind(q, k) * math.factorial(k) for k in range(1, q + 1))


def grouping_beta_peel(peeling: PeelingResult, ell_grp: int) -> float:
    """Return ``4 Gamma_ell sqrt(epsilon_peel)`` from the peeling transcript."""
    ell_grp = _validate_positive_integer("ell_grp", ell_grp)
    if not peeling.success or peeling.epsilon_peel is None:
        raise RecoveryPreconditionError("PRECONDITION peeling_failed")
    if peeling.epsilon_peel < 0.0:
        raise ValueError("epsilon_peel must be nonnegative.")
    return float(4.0 * cumulant_gamma(ell_grp) * math.sqrt(peeling.epsilon_peel))


def mixed_cumulant_from_moments(
    q: int,
    moments: Mapping[Iterable[int], float],
) -> float:
    """Evaluate the labeled set-partition cumulant from all subset moments."""
    q = _validate_positive_integer("q", q)
    normalized = {frozenset(key): float(value) for key, value in moments.items()}
    normalized.setdefault(frozenset(), 1.0)
    labels = frozenset(range(q))
    required = {
        frozenset(subset)
        for size in range(1, q + 1)
        for subset in itertools.combinations(range(q), size)
    }
    if not required.issubset(normalized):
        raise ValueError("Moment family is missing labeled nonempty subsets.")
    if any(not key.issubset(labels) for key in normalized):
        raise ValueError("Moment family contains labels outside range(q).")
    total = 0.0
    for partition in labeled_set_partitions(q):
        blocks = len(partition)
        coefficient = math.factorial(blocks - 1) * (-1 if (blocks - 1) % 2 else 1)
        term = 1.0
        for block in partition:
            term *= normalized[frozenset(block)]
        total += coefficient * term
    if not np.isfinite(total):
        raise ValueError("Cumulant evaluation produced a non-finite result.")
    return float(total)


def _validate_commuting_residual_tuple(observables: Sequence[str], m: int) -> Tuple[str, ...]:
    values = tuple(observables)
    if not values:
        raise ValueError("A residual observable tuple cannot be empty.")
    symplectic = _v1.Symplectic(m)
    for observable in values:
        _validate_pauli_string(observable, m)
    if any(
        not symplectic.commutes(left, right)
        for left, right in itertools.combinations(values, 2)
    ):
        raise ValueError("Residual cumulant observables must commute pairwise.")
    return values


def _peeled_measurement_source(
    source: Union[CEBPLearnerView, SimulatorMeasurementSource],
    peeling: PeelingResult,
) -> SimulatorMeasurementSource:
    measurement_source = _coerce_measurement_source(source)
    n = _peeling_dimension(peeling)
    if measurement_source.n != n:
        raise RecoveryPreconditionError("PRECONDITION source_dimension_mismatch")
    if measurement_source.uses_structured_backend and peeling.signed_clifford is not None:
        for index, generator in enumerate(peeling.generators):
            target = "I" * index + "Z" + "I" * (n - index - 1)
            image = peeling.signed_clifford.conjugate(generator)
            if image.phase_exponent != 0 or image.pauli != target:
                raise RuntimeError("Compact peeling Clifford orientation check failed.")
        return measurement_source._transformed_by_dagger(peeling.signed_clifford)
    if peeling.U_stab is None or not peeling_clifford_mapping_holds(
        peeling.generators, peeling.U_stab
    ):
        raise RuntimeError("Peeling Clifford orientation check failed.")
    Uq = qt.Qobj(np.asarray(peeling.U_stab), dims=[[2] * n, [2] * n])
    state = measurement_source._state_for_backend()
    if measurement_source.is_ket:
        peeled_state = (Uq.dag() * state).unit()
        peeled_state.dims = [[2] * n, [1] * n]
    else:
        peeled_state = Uq.dag() * state * Uq
        peeled_state = 0.5 * (peeled_state + peeled_state.dag())
        peeled_state = peeled_state / peeled_state.tr()
        peeled_state.dims = [[2] * n, [2] * n]
    return SimulatorMeasurementSource(n, measurement_source.is_ket, peeled_state)


def debug_exact_residual_signed_moment(
    instance: CEBPInstance,
    peeling: PeelingResult,
    sign: int,
    residual_pauli: str,
) -> float:
    """Evaluate ``mu_res(sign * P)`` exactly at zero physical-copy cost."""
    if not isinstance(instance, CEBPInstance):
        raise TypeError("Exact residual moments require a CEBPInstance debug harness.")
    if sign not in (-1, 1):
        raise ValueError("Hermitian Pauli sign must be +1 or -1.")
    n = _peeling_dimension(peeling)
    t = int(peeling.t)
    m = n - t
    _validate_pauli_string(residual_pauli, m)
    peeled = _peeled_measurement_source(instance.learner_view(), peeling)
    return _debug_exact_residual_signed_moment_from_source(
        peeled, t, sign, residual_pauli
    )


def _debug_exact_residual_signed_moment_from_source(
    peeled: SimulatorMeasurementSource,
    t: int,
    sign: int,
    residual_pauli: str,
) -> float:
    """Backend-dispatched exact moment on one already-peeled source."""

    if sign not in (-1, 1):
        raise ValueError("Hermitian Pauli sign must be +1 or -1.")
    _validate_pauli_string(residual_pauli, peeled.n - int(t))
    expectation = peeled._expectation_for_backend("I" * int(t) + residual_pauli)
    return float(sign * expectation)


def _debug_exact_residual_subset_moments_from_source(
    peeled: SimulatorMeasurementSource,
    t: int,
    observables: Sequence[str],
) -> Mapping[frozenset[int], float]:
    values = _validate_commuting_residual_tuple(observables, peeled.n - int(t))
    moments = {frozenset(): 1.0}
    for size in range(1, len(values) + 1):
        for positions in itertools.combinations(range(len(values)), size):
            sign, pauli = hermitian_pauli_product(
                *(values[position] for position in positions)
            )
            moments[frozenset(positions)] = (
                _debug_exact_residual_signed_moment_from_source(
                    peeled, t, sign, pauli
                )
            )
    return MappingProxyType(moments)


def debug_exact_residual_subset_moments(
    instance: CEBPInstance,
    peeling: PeelingResult,
    observables: Sequence[str],
) -> Mapping[frozenset[int], float]:
    """Return all signed subset moments for one exact labeled commuting tuple."""
    if not isinstance(instance, CEBPInstance):
        raise TypeError("Exact residual moments require a CEBPInstance debug harness.")
    peeled = _peeled_measurement_source(instance.learner_view(), peeling)
    return _debug_exact_residual_subset_moments_from_source(
        peeled, int(peeling.t), observables
    )


def debug_exact_residual_mixed_cumulant(
    instance: CEBPInstance,
    peeling: PeelingResult,
    observables: Sequence[str],
) -> float:
    """Evaluate the manuscript residual mixed cumulant exactly/debug-only."""
    values = tuple(observables)
    return mixed_cumulant_from_moments(
        len(values), debug_exact_residual_subset_moments(instance, peeling, values)
    )


def measure_signed_pauli_tuple(
    source: Union[CEBPLearnerView, SimulatorMeasurementSource],
    peeling: PeelingResult,
    observables: Sequence[str],
    shots: int,
    *,
    seed: RngSeed = None,
    simulation_backend: SimulationBackend = "legacy_shotwise",
) -> Union[JointPauliMeasurementRecord, JointPauliCountRecord]:
    """Jointly measure an arbitrary commuting residual tuple on fresh copies."""
    shots = _validate_positive_integer("shots", shots)
    seed = _validate_seed(seed)
    simulation_backend = _validate_simulation_backend(simulation_backend)
    n = _peeling_dimension(peeling)
    t = int(peeling.t)
    m = n - t
    values = _validate_commuting_residual_tuple(observables, m)
    if any(set(observable) <= {"I"} for observable in values):
        raise ValueError("Joint-measurement tuple members must be nonidentity.")
    peeled = _peeled_measurement_source(source, peeling)
    full_paulis = tuple("I" * t + observable for observable in values)
    outcome_values, probabilities_array = _commuting_tuple_probabilities_backend(
        peeled, full_paulis
    )
    if np.min(probabilities_array) < -1e-9:
        raise RuntimeError("Joint Pauli measurement produced a negative probability.")
    probabilities_array = np.clip(probabilities_array, 0.0, None)
    total = float(probabilities_array.sum())
    if not np.isfinite(total) or total <= 0.0 or abs(total - 1.0) > 1e-7:
        raise RuntimeError("Joint Pauli measurement probabilities are invalid.")
    probabilities_array /= total
    rng = np.random.default_rng(seed)
    if simulation_backend == "batched_counts":
        counts = rng.multinomial(shots, probabilities_array)
        return JointPauliCountRecord(
            values,
            shots,
            seed,
            np.asarray(outcome_values, dtype=np.int8),
            counts,
        )
    sampled = rng.choice(len(outcome_values), size=shots, p=probabilities_array)
    outcomes = np.asarray([outcome_values[index] for index in sampled], dtype=np.int8)
    return JointPauliMeasurementRecord(values, shots, seed, outcomes)


def _commuting_tuple_probabilities_backend(
    source: SimulatorMeasurementSource,
    paulis: Sequence[str],
) -> Tuple[Tuple[Tuple[int, ...], ...], np.ndarray]:
    """Exact joint probabilities for the already-selected commuting query."""

    values = tuple(paulis)
    if not values:
        raise ValueError("A joint measurement needs at least one observable.")
    symplectic = _v1.Symplectic(source.n)
    for pauli in values:
        _validate_pauli_string(pauli, source.n)
    if any(
        not symplectic.commutes(left, right)
        for left, right in itertools.combinations(values, 2)
    ):
        raise ValueError("Joint measurement observables must commute.")
    outcomes = tuple(itertools.product((-1, 1), repeat=len(values)))
    probabilities = source._commuting_probabilities_for_backend(values)
    if probabilities.min() < -1e-9:
        raise RuntimeError("Joint Pauli probability is negative.")
    probabilities = np.clip(probabilities, 0.0, None)
    probabilities /= probabilities.sum()
    return outcomes, probabilities


def empirical_mixed_cumulant_from_record(
    record: Union[JointPauliMeasurementRecord, JointPauliCountRecord]
) -> float:
    """Use one joint record for all subset moments of its labeled tuple."""
    return mixed_cumulant_from_moments(record.q, record.all_subset_moments())


def ordinary_cumulant_sample_count(
    q: int,
    tau_kappa: float,
    delta_tuple: float,
) -> int:
    """Return the exact ceiling in ``eq:supp-fixed-tuple-ordinary-copies``."""
    q = _validate_positive_integer("q", q)
    if not np.isfinite(tau_kappa) or tau_kappa <= 0.0:
        raise ValueError("tau_kappa must be finite and positive.")
    if not (0.0 < float(delta_tuple) < 1.0):
        raise ValueError("delta_tuple must lie in (0,1).")
    gamma = cumulant_gamma(q)
    return int(
        math.ceil(
            2.0
            * gamma**2
            / float(tau_kappa) ** 2
            * math.log(2.0 * (2**q - 1) / float(delta_tuple))
        )
    )


def adaptive_cumulant_test_bound(L: int, ell_grp: int) -> int:
    """Return ``N_test_max`` from the manuscript adaptive-query bound."""
    if isinstance(L, bool) or not isinstance(L, (int, np.integer)) or int(L) < 0:
        raise ValueError("L must be a nonnegative integer.")
    ell_grp = _validate_positive_integer("ell_grp", ell_grp)
    L = int(L)
    return int(
        L
        * 4**ell_grp
        * sum(math.comb(L, q) for q in range(2, min(ell_grp, L) + 1))
    )


def simplified_cumulant_test_bound(n: int, ell_grp: int) -> int:
    n = _validate_positive_integer("n", n)
    ell_grp = _validate_positive_integer("ell_grp", ell_grp)
    return int(ell_grp * n ** (ell_grp + 1) * 4**ell_grp)


def ordinary_grouping_copy_upper_bound(
    N_test_max: int,
    ell_grp: int,
    tau_kappa: float,
    delta_grp_ordinary: float,
) -> int:
    """Return the sufficient worst-case ordinary grouping-copy bound."""
    if isinstance(N_test_max, bool) or not isinstance(
        N_test_max, (int, np.integer)
    ) or int(N_test_max) < 0:
        raise ValueError("N_test_max must be a nonnegative integer.")
    if int(N_test_max) == 0:
        return 0
    per_query = ordinary_cumulant_sample_count(
        ell_grp, tau_kappa, float(delta_grp_ordinary) / int(N_test_max)
    )
    return int(N_test_max) * per_query


class DebugExactResidualCumulantInterface:
    """Zero-copy exact/debug residual cumulants with labeled-tuple caching."""

    provenance = DataProvenance.EXACT

    def __init__(self, instance: CEBPInstance, peeling: PeelingResult) -> None:
        if not isinstance(instance, CEBPInstance):
            raise TypeError("Exact cumulants require a CEBPInstance debug harness.")
        self._instance = instance
        self._peeling = peeling
        self._peeled_source = _peeled_measurement_source(
            instance.learner_view(), peeling
        )
        self._cache: dict[Tuple[str, ...], float] = {}
        self.query_count_by_order: dict[int, int] = {}
        self.copies_by_order: dict[int, int] = {}

    @property
    def realized_query_count(self) -> int:
        return len(self._cache)

    @property
    def realized_copies(self) -> int:
        return 0

    @property
    def queried_values(self) -> Mapping[Tuple[str, ...], float]:
        return MappingProxyType(dict(self._cache))

    def query(self, observables: Sequence[str]) -> float:
        key = tuple(observables)
        if key not in self._cache:
            value = mixed_cumulant_from_moments(
                len(key),
                _debug_exact_residual_subset_moments_from_source(
                    self._peeled_source, int(self._peeling.t), key
                ),
            )
            self._cache[key] = float(value)
            self.query_count_by_order[len(key)] = (
                self.query_count_by_order.get(len(key), 0) + 1
            )
            self.copies_by_order.setdefault(len(key), 0)
        return self._cache[key]


class EmpiricalOrdinaryCumulantInterface:
    """Fresh-batch ordinary empirical cumulants with exact labeled-tuple cache keys."""

    provenance = DataProvenance.EMPIRICAL

    def __init__(
        self,
        source: Union[CEBPLearnerView, SimulatorMeasurementSource],
        peeling: PeelingResult,
        *,
        tau_kappa: float,
        delta_tuple: float,
        seed: RngSeed = None,
        simulation_backend: SimulationBackend = "legacy_shotwise",
        max_realized_copies: Optional[int] = None,
        realized_before_grouping: int = 0,
        graceful_budget: bool = False,
    ) -> None:
        if not np.isfinite(tau_kappa) or tau_kappa <= 0.0:
            raise ValueError("Empirical cumulants require tau_kappa > 0.")
        if not (0.0 < float(delta_tuple) < 1.0):
            raise ValueError("delta_tuple must lie in (0,1).")
        self._source = _coerce_measurement_source(source)
        self._peeling = peeling
        self.tau_kappa = float(tau_kappa)
        self.delta_tuple = float(delta_tuple)
        self.seed = _validate_seed(seed)
        self.simulation_backend = _validate_simulation_backend(simulation_backend)
        self.max_realized_copies = max_realized_copies
        self.realized_before_grouping = int(realized_before_grouping)
        self.graceful_budget = bool(graceful_budget)
        self._rng = np.random.default_rng(self.seed)
        self._cache: dict[Tuple[str, ...], float] = {}
        self._records: dict[
            Tuple[str, ...], Union[JointPauliMeasurementRecord, JointPauliCountRecord]
        ] = {}
        self.query_count_by_order: dict[int, int] = {}
        self.copies_by_order: dict[int, int] = {}

    @property
    def realized_query_count(self) -> int:
        return len(self._cache)

    @property
    def realized_copies(self) -> int:
        return sum(record.copies for record in self._records.values())

    @property
    def records(self) -> Tuple[Union[JointPauliMeasurementRecord, JointPauliCountRecord], ...]:
        return tuple(self._records.values())

    @property
    def queried_values(self) -> Mapping[Tuple[str, ...], float]:
        return MappingProxyType(dict(self._cache))

    def query(self, observables: Sequence[str]) -> float:
        key = tuple(observables)
        if key not in self._cache:
            q = len(key)
            shots = ordinary_cumulant_sample_count(
                q, self.tau_kappa, self.delta_tuple
            )
            try:
                _check_execution_copy_cap(
                    self.max_realized_copies,
                    self.realized_before_grouping + self.realized_copies,
                    shots,
                    "grouping_query",
                )
            except CopyBudgetExceeded as error:
                if not self.graceful_budget:
                    raise
                raise GroupingBudgetExhausted(
                    error.stage,
                    error.realized_so_far,
                    error.next_requested,
                    error.cap,
                ) from error
            query_seed = int(self._rng.integers(0, np.iinfo(np.int64).max))
            record = measure_signed_pauli_tuple(
                self._source,
                self._peeling,
                key,
                shots,
                seed=query_seed,
                simulation_backend=self.simulation_backend,
            )
            self._records[key] = record
            self._cache[key] = empirical_mixed_cumulant_from_record(record)
            self.query_count_by_order[q] = self.query_count_by_order.get(q, 0) + 1
            self.copies_by_order[q] = self.copies_by_order.get(q, 0) + shots
        return self._cache[key]


class FixedBudgetEmpiricalCumulantInterface:
    """Cumulative empirical cumulants backed by one immutable local pool.

    Fresh-query batches are authorized only by the remaining local budget and
    the current adaptive trajectory.  No theorem sample-count formula is
    consulted.  Repeated measurements add to, rather than replace, each exact
    labeled tuple's sufficient statistics.
    """

    provenance = DataProvenance.EMPIRICAL

    def __init__(
        self,
        source: Union[CEBPLearnerView, SimulatorMeasurementSource],
        peeling: PeelingResult,
        *,
        local_copy_cap: int,
        seed: RngSeed = None,
        simulation_backend: SimulationBackend = "legacy_shotwise",
    ) -> None:
        if isinstance(local_copy_cap, bool) or int(local_copy_cap) < 0:
            raise ValueError("local_copy_cap must be a nonnegative integer.")
        self._source = _coerce_measurement_source(source)
        self._peeling = peeling
        self.local_copy_cap = int(local_copy_cap)
        self.seed = _validate_seed(seed)
        self.simulation_backend = _validate_simulation_backend(simulation_backend)
        self._rng = np.random.default_rng(self.seed)
        self._moments: dict[Tuple[str, ...], dict[frozenset[int], float]] = {}
        self._shots: dict[Tuple[str, ...], int] = {}
        self._cache: dict[Tuple[str, ...], float] = {}
        self._records: list[
            Union[JointPauliMeasurementRecord, JointPauliCountRecord]
        ] = []
        self._remaining_opportunities = 1
        self._opportunity_history: list[Tuple[Tuple[str, ...], int, int]] = []
        self.query_count_by_order: dict[int, int] = {}
        self.copies_by_order: dict[int, int] = {}
        self.refinement_top_up_count = 0

    @property
    def realized_query_count(self) -> int:
        return len(self._shots)

    @property
    def realized_copies(self) -> int:
        return sum(self._shots.values())

    @property
    def remaining_copies(self) -> int:
        return self.local_copy_cap - self.realized_copies

    @property
    def queried_values(self) -> Mapping[Tuple[str, ...], float]:
        return MappingProxyType(dict(self._cache))

    @property
    def shots_by_tuple(self) -> Mapping[Tuple[str, ...], int]:
        return MappingProxyType(dict(self._shots))

    @property
    def records(self) -> Tuple[
        Union[JointPauliMeasurementRecord, JointPauliCountRecord], ...
    ]:
        return tuple(self._records)

    @property
    def opportunity_history(self) -> Tuple[Tuple[Tuple[str, ...], int, int], ...]:
        """Fresh-query ``(label, remaining opportunities, assigned shots)``."""

        return tuple(self._opportunity_history)

    def set_remaining_query_opportunities(self, count: int) -> None:
        if isinstance(count, bool) or int(count) <= 0:
            raise ValueError("remaining query opportunities must be positive.")
        self._remaining_opportunities = int(count)

    def _measure_into(self, key: Tuple[str, ...], shots: int, *, refinement: bool) -> None:
        if shots <= 0 or shots > self.remaining_copies:
            raise ValueError("Fixed-budget grouping top-up is outside the local pool.")
        query_seed = int(self._rng.integers(0, np.iinfo(np.int64).max))
        record = measure_signed_pauli_tuple(
            self._source,
            self._peeling,
            key,
            shots,
            seed=query_seed,
            simulation_backend=self.simulation_backend,
        )
        new_moments = record.all_subset_moments()
        old_shots = self._shots.get(key, 0)
        total_shots = old_shots + shots
        if old_shots:
            old_moments = self._moments[key]
            cumulative = {
                subset: old_moments[subset] + shots * float(value)
                for subset, value in new_moments.items()
            }
        else:
            cumulative = {
                subset: shots * float(value) for subset, value in new_moments.items()
            }
            self.query_count_by_order[len(key)] = (
                self.query_count_by_order.get(len(key), 0) + 1
            )
        self._moments[key] = cumulative
        self._shots[key] = total_shots
        self.copies_by_order[len(key)] = self.copies_by_order.get(len(key), 0) + shots
        averaged = {subset: value / total_shots for subset, value in cumulative.items()}
        self._cache[key] = mixed_cumulant_from_moments(len(key), averaged)
        self._records.append(record)
        if refinement:
            self.refinement_top_up_count += 1

    def query(self, observables: Sequence[str]) -> float:
        key = tuple(observables)
        if key not in self._cache:
            if self.remaining_copies <= 0:
                raise GroupingBudgetExhausted(
                    "grouping_query", self.realized_copies, 1, self.local_copy_cap
                )
            shots = max(1, self.remaining_copies // self._remaining_opportunities)
            self._opportunity_history.append(
                (key, int(self._remaining_opportunities), int(shots))
            )
            self._measure_into(key, shots, refinement=False)
        return self._cache[key]

    def top_up_balanced(self, copy_count: int) -> int:
        """Spend at most ``copy_count`` deterministically across known tuples."""

        amount = min(max(0, int(copy_count)), self.remaining_copies)
        keys = tuple(sorted(self._shots))
        if amount == 0 or not keys:
            return 0
        base, extra = divmod(amount, len(keys))
        used = 0
        for index, key in enumerate(keys):
            shots = base + (1 if index < extra else 0)
            if shots:
                self._measure_into(key, shots, refinement=True)
                used += shots
        return used


class CallableResidualCumulantInterface:
    """Deterministic exact interface for focused algorithmic fixtures."""

    provenance = DataProvenance.EXACT

    def __init__(self, callback: Callable[[Tuple[str, ...]], float]) -> None:
        if not callable(callback):
            raise TypeError("callback must be callable.")
        self._callback = callback
        self._cache: dict[Tuple[str, ...], float] = {}
        self.query_count_by_order: dict[int, int] = {}
        self.copies_by_order: dict[int, int] = {}

    @property
    def realized_query_count(self) -> int:
        return len(self._cache)

    @property
    def realized_copies(self) -> int:
        return 0

    def query(self, observables: Sequence[str]) -> float:
        key = tuple(observables)
        if key not in self._cache:
            value = float(self._callback(key))
            if not np.isfinite(value):
                raise ValueError("Synthetic cumulant callback returned a non-finite value.")
            self._cache[key] = value
            self.query_count_by_order[len(key)] = (
                self.query_count_by_order.get(len(key), 0) + 1
            )
            self.copies_by_order.setdefault(len(key), 0)
        return self._cache[key]


def hypergraph_connected_components(
    vertices: Sequence[Tuple[int, ...]],
    hyperedges: Sequence[Sequence[Tuple[int, ...]]],
) -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
    """Return deterministic nontrivial connected components of a hypergraph."""
    ordered_vertices = tuple(sorted(tuple(vertex) for vertex in vertices))
    parent = {vertex: vertex for vertex in ordered_vertices}
    touched: set[Tuple[int, ...]] = set()

    def find(vertex: Tuple[int, ...]) -> Tuple[int, ...]:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    def union(left: Tuple[int, ...], right: Tuple[int, ...]) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root

    for edge in hyperedges:
        normalized = tuple(sorted(tuple(vertex) for vertex in edge))
        if len(normalized) < 2 or any(vertex not in parent for vertex in normalized):
            raise ValueError("Hyperedges must contain at least two known vertices.")
        touched.update(normalized)
        for vertex in normalized[1:]:
            union(normalized[0], vertex)
    groups: dict[Tuple[int, ...], list[Tuple[int, ...]]] = {}
    for vertex in sorted(touched):
        groups.setdefault(find(vertex), []).append(vertex)
    return tuple(sorted((tuple(values) for values in groups.values()), key=lambda x: x[0]))


def _canonical_cluster_partition(
    clusters: Iterable[Iterable[int]],
) -> Tuple[Tuple[int, ...], ...]:
    normalized = tuple(tuple(sorted(int(value) for value in cluster)) for cluster in clusters)
    if any(not cluster or len(set(cluster)) != len(cluster) for cluster in normalized):
        raise ValueError("Clusters must be nonempty sets of sector IDs.")
    flattened = tuple(value for cluster in normalized for value in cluster)
    if len(flattened) != len(set(flattened)):
        raise ValueError("Cluster partition contains a duplicate sector ID.")
    return tuple(sorted(normalized))


def _grouping_ledgers(
    recovery: RecoveryResult, grouping_copies: int
) -> Tuple[CopyLedger, CopyLedger]:
    grouping = CopyLedger((("grouping_ordinary_pool", int(grouping_copies)),))
    cumulative = CopyLedger(recovery.cumulative_copy_ledger.entries + grouping.entries)
    return grouping, cumulative


def _grouping_tau_diagnostic(config: GroupingConfig) -> float:
    """Return finite display metadata without authorizing fixed-budget shots."""

    try:
        value = float(config.tau_kappa)
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0


def _grouping_failure(
    recovery: RecoveryResult,
    peeling: PeelingResult,
    config: GroupingConfig,
    ell_grp: int,
    reason: str,
    provenance: DataProvenance,
) -> GroupingResult:
    L = len(recovery.sectors)
    beta = grouping_beta_peel(peeling, ell_grp) if peeling.success else math.inf
    ntest = adaptive_cumulant_test_bound(L, ell_grp)
    grouping, cumulative = _grouping_ledgers(recovery, 0)
    return GroupingResult(
        success=False,
        failure_reason=reason,
        L=L,
        ell_grp=ell_grp,
        clusters=(),
        eta_test=float(config.eta_test),
        tau_kappa=_grouping_tau_diagnostic(config),
        beta_peel=float(beta),
        eta_s=config.eta_s,
        xi_s=None if config.eta_s is None else 3.0 * config.eta_s / 4.0,
        no_false_merge_condition_holds=False,
        exact_recovery_window_holds=None,
        recovery_theorem_preconditions_hold=bool(
            recovery.theorem_recovery_preconditions_hold
        ),
        theorem_grouping_preconditions_hold=False,
        recovery_score_provenance=recovery.score_provenance,
        moment_provenance=provenance,
        realized_query_count=0,
        query_count_by_order=(),
        realized_grouping_copies=0,
        copies_by_order=(),
        N_test_max=ntest,
        N_test_simplified_bound=simplified_cumulant_test_bound(recovery.n, ell_grp),
        delta_tuple=None,
        ordinary_copy_upper_bound=0,
        grouping_copy_ledger=grouping,
        cumulative_copy_ledger=cumulative,
        merge_rounds=0,
        sampling_policy=config.sampling_policy,
    )


def _remaining_grouping_query_opportunities(
    partition: Sequence[Tuple[int, ...]],
    start_order: int,
    ell_grp: int,
    recovery: RecoveryResult,
    max_generated_group_size: int,
) -> int:
    """Trajectory-local tuple count used only by the numerical allocator."""

    active = tuple(cluster for cluster in partition if len(cluster) < ell_grp)
    group_sizes = {
        cluster: len(
            generated_cluster_pauli_group(
                cluster,
                recovery.sectors,
                max_group_size=max_generated_group_size,
            ).nonidentity
        )
        for cluster in active
    }
    total = 0
    for order in range(start_order, ell_grp + 1):
        for selected in itertools.combinations(active, order):
            if len(set().union(*(set(cluster) for cluster in selected))) <= ell_grp:
                total += math.prod(group_sizes[cluster] for cluster in selected)
    return int(total)


@dataclass
class _GroupingOpportunityState:
    """Exact remaining scan opportunities for the current adaptive partition.

    The state tracks only the still-reachable portion of the current order.
    Before a witness, later orders on the same partition remain possible.  A
    first witness guarantees a merge/reset, so those later-order opportunities
    are removed immediately.  Finishing a selected cluster tuple also skips
    every unqueried operator tuple after its first witness.
    """

    operator_counts: Tuple[int, ...]
    higher_order_opportunities: int
    selected_index: int = 0
    operator_index: int = 0
    merge_guaranteed: bool = False

    @property
    def remaining(self) -> int:
        if self.selected_index >= len(self.operator_counts):
            return 1
        current = self.operator_counts[self.selected_index] - self.operator_index
        later_selected = sum(self.operator_counts[self.selected_index + 1 :])
        later_orders = 0 if self.merge_guaranteed else self.higher_order_opportunities
        return max(1, int(current + later_selected + later_orders))

    def record_executed_query(self) -> None:
        if self.selected_index >= len(self.operator_counts):
            raise RuntimeError("Grouping opportunity state is past the current scan.")
        self.operator_index += 1
        if self.operator_index > self.operator_counts[self.selected_index]:
            raise RuntimeError("Grouping opportunity state over-consumed a selection.")

    def finish_selected(self, *, witnessed: bool) -> None:
        if witnessed:
            self.merge_guaranteed = True
        self.selected_index += 1
        self.operator_index = 0


def hierarchical_cumulant_grouping(
    recovery: RecoveryResult,
    peeling: PeelingResult,
    cumulant_interface: ResidualCumulantInterface,
    config: GroupingConfig,
) -> GroupingResult:
    """Run ``alg:supp-hierarchical-cumulant-grouping`` with q-reset literally."""
    if not isinstance(config, GroupingConfig):
        raise TypeError("config must be a GroupingConfig.")
    if config.ell_grp is None:
        raise ValueError("The common grouping core requires a resolved ell_grp.")
    ell_grp = int(config.ell_grp)
    provenance = DataProvenance(cumulant_interface.provenance)
    if not recovery.success:
        return _grouping_failure(
            recovery, peeling, config, ell_grp, "recovery_failed", provenance
        )
    try:
        recovery_theorem = _validate_recovery_for_grouping(
            recovery,
            allow_uncalibrated_recovery=(
                config.allow_uncalibrated_recovery
                or provenance is DataProvenance.EXACT
            ),
        )
    except RecoveryPreconditionError:
        return _grouping_failure(
            recovery,
            peeling,
            config,
            ell_grp,
            "recovery_not_theorem_calibrated",
            provenance,
        )
    initial_L = len(recovery.sectors)
    trivial_partition = initial_L <= 1 or ell_grp == 1
    beta = grouping_beta_peel(peeling, ell_grp)
    fixed_budget_sampling = (
        config.sampling_policy == GroupingSamplingPolicy.FIXED_BUDGET.value
    )
    tau_diagnostic = _grouping_tau_diagnostic(config)
    # No false merge is vacuous when the algorithm cannot test or perform a
    # merge.  In particular, the d=1 specialization consumes zero ordinary
    # copies even when the numerical nontrivial-branch margin would fail.
    no_false = trivial_partition or (
        beta + tau_diagnostic < float(config.eta_test)
    )
    if (
        provenance is DataProvenance.EMPIRICAL
        and not fixed_budget_sampling
        and not no_false
        and not config.allow_no_false_merge_margin_failure
    ):
        return _grouping_failure(
            recovery,
            peeling,
            config,
            ell_grp,
            "no_false_merge_margin_failed",
            provenance,
        )

    sector_ids = tuple(sorted(sector.sector_id for sector in recovery.sectors))
    partition = _canonical_cluster_partition((sector_id,) for sector_id in sector_ids)
    initial_L = len(partition)
    ntest = adaptive_cumulant_test_bound(initial_L, ell_grp)
    simplified = simplified_cumulant_test_bound(recovery.n, ell_grp)
    if ntest > simplified:
        raise RuntimeError("Adaptive cumulant bound exceeds its manuscript simplification.")
    delta_tuple = (
        float(config.delta_grp_ordinary) / ntest
        if provenance is DataProvenance.EMPIRICAL and ntest > 0 and not fixed_budget_sampling
        else None
    )
    copy_bound = (
        ordinary_grouping_copy_upper_bound(
            ntest, ell_grp, tau_diagnostic, float(config.delta_grp_ordinary)
        )
        if provenance is DataProvenance.EMPIRICAL and ntest > 0 and not fixed_budget_sampling
        else 0
    )
    scans = []
    witnesses = []
    merge_rounds = 0
    q = ell_grp + 1 if trivial_partition else 2
    budget_truncated = False
    while q <= ell_grp:
        scan_order = q
        before = partition
        active = tuple(cluster for cluster in partition if len(cluster) < ell_grp)
        order_witnesses = []
        edges = []
        selected_plans = []
        for selected in itertools.combinations(active, q):
            if len(set().union(*(set(cluster) for cluster in selected))) > ell_grp:
                continue
            operator_groups = tuple(
                generated_cluster_pauli_group(
                    cluster,
                    recovery.sectors,
                    max_group_size=config.max_generated_group_size,
                ).nonidentity
                for cluster in selected
            )
            selected_plans.append(
                (tuple(selected), operator_groups, math.prod(map(len, operator_groups)))
            )
        higher_opportunities = (
            0
            if q >= ell_grp
            else _remaining_grouping_query_opportunities(
                partition,
                q + 1,
                ell_grp,
                recovery,
                config.max_generated_group_size,
            )
        )
        opportunity_state = _GroupingOpportunityState(
            tuple(plan[2] for plan in selected_plans),
            higher_opportunities,
        )
        for selected, operator_groups, _operator_count in selected_plans:
            witnessed = False
            for observables in itertools.product(*operator_groups):
                _validate_commuting_residual_tuple(observables, recovery.m)
                try:
                    setter = getattr(
                        cumulant_interface,
                        "set_remaining_query_opportunities",
                        None,
                    )
                    if setter is not None:
                        setter(opportunity_state.remaining)
                    value = float(cumulant_interface.query(observables))
                except GroupingBudgetExhausted:
                    budget_truncated = True
                    break
                opportunity_state.record_executed_query()
                comparison_value = value
                if (
                    provenance is DataProvenance.EXACT
                    and abs(comparison_value) <= config.exact_zero_tolerance
                ):
                    comparison_value = 0.0
                if abs(comparison_value) > config.eta_test:
                    witness = HyperedgeWitness(q, tuple(selected), tuple(observables), value)
                    order_witnesses.append(witness)
                    witnesses.append(witness)
                    edges.append(tuple(selected))
                    witnessed = True
                    break
            opportunity_state.finish_selected(witnessed=witnessed)
            if budget_truncated:
                break
        if budget_truncated:
            break
        components = hypergraph_connected_components(active, edges)
        if not components:
            after = partition
            reset = False
            q += 1
        else:
            component_vertices = {cluster for component in components for cluster in component}
            merged = [
                tuple(sorted(sector for cluster in component for sector in cluster))
                for component in components
            ]
            unchanged = [cluster for cluster in partition if cluster not in component_vertices]
            partition = _canonical_cluster_partition(unchanged + merged)
            after = partition
            reset = True
            merge_rounds += 1
            q = 2
        scans.append(
            GroupingScan(
                order=scan_order,
                partition_before=before,
                active_clusters=active,
                hyperedges=tuple(order_witnesses),
                connected_components=components,
                partition_after=after,
                reset_to_two=reset,
            )
        )

    realized_queries = int(cumulant_interface.realized_query_count)
    realized_copies = int(cumulant_interface.realized_copies)
    if not fixed_budget_sampling and realized_queries > ntest:
        raise RuntimeError("Realized adaptive query count exceeds N_test_max.")
    if provenance is DataProvenance.EXACT and realized_copies != 0:
        raise RuntimeError("Exact grouping cannot consume physical copies.")
    if (
        provenance is DataProvenance.EMPIRICAL
        and not fixed_budget_sampling
        and realized_copies > copy_bound
    ):
        raise RuntimeError("Realized grouping copies exceed the ordinary upper bound.")
    grouping_ledger, cumulative = _grouping_ledgers(recovery, realized_copies)
    exact_window = None
    if config.eta_irr is not None:
        exact_window = (
            None
            if fixed_budget_sampling
            else (
                beta + tau_diagnostic < config.eta_test
                and config.eta_test < config.eta_irr - tau_diagnostic
            )
        )
    theorem_grouping = (
        provenance is DataProvenance.EMPIRICAL
        and not fixed_budget_sampling
        and recovery_theorem
        and no_false
        and not budget_truncated
    )
    return GroupingResult(
        success=True,
        failure_reason=None,
        L=initial_L,
        ell_grp=ell_grp,
        clusters=partition,
        eta_test=float(config.eta_test),
        tau_kappa=tau_diagnostic,
        beta_peel=beta,
        eta_s=config.eta_s,
        xi_s=None if config.eta_s is None else 3.0 * config.eta_s / 4.0,
        no_false_merge_condition_holds=no_false,
        exact_recovery_window_holds=exact_window,
        recovery_theorem_preconditions_hold=recovery_theorem,
        theorem_grouping_preconditions_hold=theorem_grouping,
        recovery_score_provenance=recovery.score_provenance,
        moment_provenance=provenance,
        realized_query_count=realized_queries,
        query_count_by_order=tuple(sorted(cumulant_interface.query_count_by_order.items())),
        realized_grouping_copies=realized_copies,
        copies_by_order=tuple(sorted(cumulant_interface.copies_by_order.items())),
        N_test_max=ntest,
        N_test_simplified_bound=simplified,
        delta_tuple=delta_tuple,
        ordinary_copy_upper_bound=copy_bound,
        grouping_copy_ledger=grouping_ledger,
        cumulative_copy_ledger=cumulative,
        merge_rounds=merge_rounds,
        transcript=tuple(scans) if config.return_details else (),
        hyperedge_witnesses=tuple(witnesses) if config.return_details else (),
        grouping_complete=not budget_truncated,
        grouping_budget_truncated=budget_truncated,
        sampling_policy=config.sampling_policy,
        exploratory_query_count=realized_queries,
        refinement_top_up_count=int(
            getattr(cumulant_interface, "refinement_top_up_count", 0)
        ),
        min_shots_per_queried_tuple=(
            min(getattr(cumulant_interface, "shots_by_tuple", {}).values())
            if getattr(cumulant_interface, "shots_by_tuple", {})
            else 0
        ),
        max_shots_per_queried_tuple=(
            max(getattr(cumulant_interface, "shots_by_tuple", {}).values())
            if getattr(cumulant_interface, "shots_by_tuple", {})
            else 0
        ),
        mean_shots_per_queried_tuple=(
            float(np.mean(tuple(getattr(cumulant_interface, "shots_by_tuple", {}).values())))
            if getattr(cumulant_interface, "shots_by_tuple", {})
            else 0.0
        ),
    )


def debug_exact_hierarchical_cumulant_grouping(
    instance: CEBPInstance,
    peeling: PeelingResult,
    recovery: RecoveryResult,
    config: Optional[GroupingConfig] = None,
) -> GroupingResult:
    """Run exact/debug zero-copy grouping without upgrading theorem flags."""
    if config is None:
        config = GroupingConfig(ell_grp=instance.d)
    elif config.ell_grp is None:
        config = replace(config, ell_grp=instance.d)
    interface = DebugExactResidualCumulantInterface(instance, peeling)
    return hierarchical_cumulant_grouping(recovery, peeling, interface, config)


@_time_pipeline_stage("grouping")
def empirical_hierarchical_cumulant_grouping(
    source: Union[CEBPLearnerView, SimulatorMeasurementSource],
    peeling: PeelingResult,
    recovery: RecoveryResult,
    config: GroupingConfig,
    *,
    seed: RngSeed = None,
    simulation_backend: SimulationBackend = "legacy_shotwise",
    max_realized_copies: Optional[int] = None,
    realized_before_grouping: int = 0,
    graceful_budget: bool = False,
    fixed_budget_local_cap: Optional[int] = None,
) -> GroupingResult:
    """Run learner-facing ordinary grouping with fresh adaptive tuple batches."""
    if not isinstance(config, GroupingConfig):
        raise TypeError("config must be a GroupingConfig.")
    if config.ell_grp is None:
        if not isinstance(source, CEBPLearnerView):
            raise TypeError("Default ell_grp=d requires a CEBPLearnerView.")
        config = replace(config, ell_grp=source.d)
    ell_grp = int(config.ell_grp)
    try:
        _coerce_measurement_source(source)
    except (TypeError, ValueError):
        return _grouping_failure(
            recovery,
            peeling,
            config,
            ell_grp,
            "invalid_measurement_source",
            DataProvenance.EMPIRICAL,
        )
    if not recovery.success:
        return _grouping_failure(
            recovery,
            peeling,
            config,
            ell_grp,
            "recovery_failed",
            DataProvenance.EMPIRICAL,
        )
    if (
        not recovery.theorem_recovery_preconditions_hold
        and not config.allow_uncalibrated_recovery
    ):
        return _grouping_failure(
            recovery,
            peeling,
            config,
            ell_grp,
            "recovery_not_theorem_calibrated",
            DataProvenance.EMPIRICAL,
        )
    fixed_budget_sampling = (
        config.sampling_policy == GroupingSamplingPolicy.FIXED_BUDGET.value
    )
    if fixed_budget_sampling:
        if config.eta_test <= 0.0:
            raise ValueError("fixed-budget empirical eta_test must be positive.")
        if fixed_budget_local_cap is None:
            raise ValueError("fixed_budget grouping requires fixed_budget_local_cap.")
        interface = FixedBudgetEmpiricalCumulantInterface(
            source,
            peeling,
            local_copy_cap=fixed_budget_local_cap,
            seed=seed,
            simulation_backend=simulation_backend,
        )
        result = hierarchical_cumulant_grouping(recovery, peeling, interface, config)
        exploratory_queries = interface.realized_query_count
        previous_signature = (result.clusters, interface.realized_query_count)
        for _iteration in range(16):
            if interface.remaining_copies <= 0 or interface.realized_query_count == 0:
                break
            used = interface.top_up_balanced(max(1, interface.remaining_copies // 2))
            if used <= 0:
                break
            replay = hierarchical_cumulant_grouping(recovery, peeling, interface, config)
            signature = (replay.clusters, interface.realized_query_count)
            result = replay
            if signature == previous_signature:
                interface.top_up_balanced(interface.remaining_copies)
                result = hierarchical_cumulant_grouping(
                    recovery, peeling, interface, config
                )
                break
            previous_signature = signature
        shots = tuple(interface.shots_by_tuple.values())
        return replace(
            result,
            exploratory_query_count=exploratory_queries,
            refinement_top_up_count=interface.refinement_top_up_count,
            min_shots_per_queried_tuple=min(shots) if shots else 0,
            max_shots_per_queried_tuple=max(shots) if shots else 0,
            mean_shots_per_queried_tuple=float(np.mean(shots)) if shots else 0.0,
        )

    trivial_partition = len(recovery.sectors) <= 1 or ell_grp == 1
    beta = grouping_beta_peel(peeling, ell_grp)
    if (
        not trivial_partition
        and
        not beta + config.tau_kappa < config.eta_test
        and not config.allow_no_false_merge_margin_failure
    ):
        return _grouping_failure(
            recovery,
            peeling,
            config,
            ell_grp,
            "no_false_merge_margin_failed",
            DataProvenance.EMPIRICAL,
        )
    L = len(recovery.sectors)
    ntest = adaptive_cumulant_test_bound(L, ell_grp)
    if config.delta_grp_ordinary is None:
        raise ValueError("certified grouping requires delta_grp_ordinary.")
    delta_tuple = config.delta_grp_ordinary / ntest if ntest > 0 else 0.5
    if ntest > 0 and (config.tau_kappa is None or config.tau_kappa <= 0.0):
        return _grouping_failure(
            recovery,
            peeling,
            config,
            ell_grp,
            "invalid_tau_kappa",
            DataProvenance.EMPIRICAL,
        )

    interface = EmpiricalOrdinaryCumulantInterface(
        source,
        peeling,
        tau_kappa=(1.0 if ntest == 0 else float(config.tau_kappa)),
        delta_tuple=delta_tuple,
        seed=seed,
        simulation_backend=simulation_backend,
        max_realized_copies=max_realized_copies,
        realized_before_grouping=realized_before_grouping,
        graceful_budget=graceful_budget,
    )
    try:
        return hierarchical_cumulant_grouping(recovery, peeling, interface, config)
    except CopyBudgetExceeded as error:
        failed = _grouping_failure(
            recovery,
            peeling,
            config,
            ell_grp,
            f"copy_budget: {error}",
            DataProvenance.EMPIRICAL,
        )
        grouping_ledger, cumulative = _grouping_ledgers(
            recovery, interface.realized_copies
        )
        return replace(
            failed,
            realized_query_count=interface.realized_query_count,
            query_count_by_order=tuple(sorted(interface.query_count_by_order.items())),
            realized_grouping_copies=interface.realized_copies,
            copies_by_order=tuple(sorted(interface.copies_by_order.items())),
            grouping_copy_ledger=grouping_ledger,
            cumulative_copy_ledger=cumulative,
        )


def _symplectic_form_matrix(m: int) -> np.ndarray:
    identity = np.eye(m, dtype=np.uint8)
    zero = np.zeros((m, m), dtype=np.uint8)
    return np.block([[zero, identity], [identity, zero]]).astype(np.uint8)


def _symplectic_pairing(left: np.ndarray, right: np.ndarray, m: int) -> int:
    return int((np.asarray(left, dtype=np.uint8) @ _symplectic_form_matrix(m) @ np.asarray(right, dtype=np.uint8)) % 2)


def _vector_tuple(vector: np.ndarray) -> Tuple[int, ...]:
    return tuple(int(value) for value in np.asarray(vector, dtype=np.uint8) % 2)


def _vector_span_signature(
    vectors: Sequence[np.ndarray], width: int
) -> Tuple[Tuple[int, ...], ...]:
    if not vectors:
        return ()
    matrix = np.vstack([np.asarray(vector, dtype=np.uint8) % 2 for vector in vectors])
    if matrix.shape[1] != width:
        raise LocalizationInvariantError("Binary vector has the wrong residual width.")
    reduced, _ = _v1._gf2_rref(matrix)
    return tuple(_vector_tuple(row) for row in reduced if np.any(row))


def _restricted_symplectic_parameters(gram: np.ndarray) -> Tuple[int, int, int]:
    """Return ``(rank, h, q)`` and reject malformed/odd restricted forms."""
    matrix = np.asarray(gram, dtype=np.uint8) % 2
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise LocalizationInvariantError("Restricted symplectic form must be square.")
    rank = int(_v1._gf2_rank(matrix))
    if rank % 2:
        raise LocalizationInvariantError("Restricted symplectic rank must be even.")
    if np.any(np.diag(matrix)) or not np.array_equal(matrix, matrix.T):
        raise LocalizationInvariantError("Restricted form is not alternating symplectic.")
    h_C = rank // 2
    q_C = matrix.shape[0] - rank
    return rank, h_C, q_C


def analyze_group_symplectic_structure(
    recovery: RecoveryResult,
    cluster: Iterable[int],
) -> ClusterSymplecticStructure:
    """Compute the manuscript ``V_C``, radical, and hyperbolic decomposition."""
    if not isinstance(recovery, RecoveryResult):
        raise TypeError("recovery must be a RecoveryResult.")
    if not recovery.success:
        raise RecoveryPreconditionError("PRECONDITION recovery_failed")
    cluster_ids = tuple(sorted(int(sector_id) for sector_id in cluster))
    if not cluster_ids or len(set(cluster_ids)) != len(cluster_ids):
        raise LocalizationInvariantError(
            "A localization cluster must contain distinct sector IDs."
        )
    lookup = {sector.sector_id: sector for sector in recovery.sectors}
    if len(lookup) != len(recovery.sectors) or any(
        sector_id not in lookup for sector_id in cluster_ids
    ):
        raise LocalizationInvariantError("Localization cluster contains an unknown ID.")
    selected = tuple(lookup[sector_id] for sector_id in cluster_ids)
    if not recovered_sectors_are_symplectically_valid(selected):
        raise LocalizationInvariantError("Cluster sectors are not symplectically valid.")

    m = recovery.m
    axes = recovered_sector_axes(selected)
    if any(len(axis) != m for axis in axes):
        raise LocalizationInvariantError("Cluster axis does not have residual length m.")
    vectors = tuple(_v1.pauli_to_symplectic_col(axis) for axis in axes)
    basis_matrix = np.column_stack(vectors).astype(np.uint8) % 2
    dimension = int(_v1._gf2_rank(basis_matrix))
    if dimension != len(vectors):
        raise LocalizationInvariantError("A grouped recovered axis was lost or duplicated.")
    J = _symplectic_form_matrix(m)
    gram = (basis_matrix.T @ J @ basis_matrix) % 2
    rank, h_C, q_C = _restricted_symplectic_parameters(gram)

    null_coefficients = _v1._gf2_nullspace_basis(gram)
    generic_radical = tuple(
        (basis_matrix @ coefficient) % 2 for coefficient in null_coefficients
    )
    if len(generic_radical) != q_C or _v1._gf2_rank(
        np.column_stack(generic_radical)
        if generic_radical
        else np.zeros((2 * m, 0), dtype=np.uint8)
    ) != q_C:
        raise LocalizationInvariantError("Explicit radical basis has wrong dimension.")
    if any(
        _symplectic_pairing(radical, vector, m)
        for radical in generic_radical
        for vector in vectors
    ):
        raise LocalizationInvariantError("Computed radical is not orthogonal to V_C.")

    triple_sectors = tuple(sector for sector in selected if sector.completed)
    singleton_sectors = tuple(sector for sector in selected if not sector.completed)
    hyperbolic_pairs = tuple(
        (
            _v1.pauli_to_symplectic_col(sector.x),
            _v1.pauli_to_symplectic_col(str(sector.z)),
        )
        for sector in triple_sectors
    )
    sector_radicals = tuple(
        _v1.pauli_to_symplectic_col(sector.x) for sector in singleton_sectors
    )
    if h_C != len(hyperbolic_pairs) or q_C != len(sector_radicals):
        raise LocalizationInvariantError(
            "Generic h_C/q_C disagrees with the recovered sector structure."
        )
    if _vector_span_signature(generic_radical, 2 * m) != _vector_span_signature(
        sector_radicals, 2 * m
    ):
        raise LocalizationInvariantError(
            "Generic radical disagrees with singleton-sector directions."
        )
    adapted = tuple(
        vector for pair in hyperbolic_pairs for vector in pair
    ) + sector_radicals
    if _vector_span_signature(adapted, 2 * m) != _vector_span_signature(
        vectors, 2 * m
    ):
        raise LocalizationInvariantError("Adapted hyperbolic/radical basis lost V_C.")
    for pair_index, (left, right) in enumerate(hyperbolic_pairs):
        if _symplectic_pairing(left, right, m) != 1:
            raise LocalizationInvariantError("Recovered triple is not hyperbolic.")
        other_vectors = tuple(
            vector
            for other_index, pair in enumerate(hyperbolic_pairs)
            if other_index != pair_index
            for vector in pair
        ) + sector_radicals
        if any(
            _symplectic_pairing(left, vector, m)
            or _symplectic_pairing(right, vector, m)
            for vector in other_vectors
        ):
            raise LocalizationInvariantError("Adapted hyperbolic planes are not orthogonal.")

    k_C = h_C + q_C
    if k_C != len(cluster_ids):
        raise LocalizationInvariantError("Manuscript identity k_C=|C| failed.")
    return ClusterSymplecticStructure(
        cluster=cluster_ids,
        ordered_independent_axes=axes,
        axis_vectors=tuple(_vector_tuple(vector) for vector in vectors),
        span_basis=canonical_gf2_span_basis(axes, m),
        restricted_gram=gram,
        dimension=dimension,
        symplectic_rank=rank,
        h_C=h_C,
        q_C=q_C,
        k_C=k_C,
        radical_basis=tuple(_vector_tuple(vector) for vector in sector_radicals),
        hyperbolic_pairs=tuple(
            (_vector_tuple(left), _vector_tuple(right))
            for left, right in hyperbolic_pairs
        ),
    )


def _gf2_inverse_matrix(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.uint8) % 2
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise LocalizationInvariantError("GF(2) inverse requires a square matrix.")
    size = value.shape[0]
    if size == 0:
        return value.copy()
    augmented = np.concatenate((value.copy(), np.eye(size, dtype=np.uint8)), axis=1)
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if augmented[row, column]), None
        )
        if pivot is None:
            raise LocalizationInvariantError("Binary basis matrix is singular.")
        if pivot != column:
            augmented[[column, pivot]] = augmented[[pivot, column]]
        for row in range(size):
            if row != column and augmented[row, column]:
                augmented[row] ^= augmented[column]
    return augmented[:, size:]


def _canonical_pauli_vector(m: int, qubit: int, axis: str) -> np.ndarray:
    vector = np.zeros(2 * m, dtype=np.uint8)
    vector[qubit if axis == "X" else m + qubit] = 1
    return vector


def _residual_support(vector: np.ndarray, m: int) -> Tuple[int, ...]:
    value = np.asarray(vector, dtype=np.uint8) % 2
    return tuple(
        qubit for qubit in range(m) if value[qubit] or value[m + qubit]
    )


def _full_residual_tableau(residual: np.ndarray, n: int, t: int) -> np.ndarray:
    full = np.eye(2 * n, dtype=np.uint8)
    residual_indices = tuple(range(t, n)) + tuple(range(n + t, 2 * n))
    full[np.ix_(residual_indices, residual_indices)] = residual
    return full


def _localization_failure(
    recovery: RecoveryResult,
    grouping: GroupingResult,
    d: int,
    reason: str,
    *,
    handoff_valid: bool = False,
) -> LocalizationResult:
    theorem = bool(grouping.theorem_grouping_preconditions_hold)
    empirical_max = max((len(cluster) for cluster in grouping.clusters), default=0)
    oversize = tuple(
        (tuple(cluster), len(cluster))
        for cluster in grouping.clusters
        if len(cluster) > d
    )
    return LocalizationResult(
        success=False,
        failure_reason=reason,
        n=recovery.n,
        t=recovery.t,
        m=recovery.m,
        d=d,
        clusters=(),
        structures=(),
        cluster_localizations=(),
        J_C=(),
        J_aux=(),
        K_rec=0,
        source_basis=None,
        target_basis=None,
        residual_tableau=None,
        synthesis_tableau=None,
        full_tableau=None,
        gates=(),
        U_rec=None,
        bar_U_rec=None,
        grouping_theorem_preconditions_hold=theorem,
        theorem_localization_preconditions_hold=False,
        handoff_valid=handoff_valid,
        cross_group_direct_sum_holds=False,
        cross_group_symplectic_orthogonality_holds=False,
        global_pairing_holds=False,
        register_partition_holds=False,
        localization_guarantee_holds=False,
        localization_copy_count=0,
        cumulative_copy_ledger=grouping.cumulative_copy_ledger,
        empirical_max_cluster_size=empirical_max,
        assumed_d=d,
        model_bound_violated=bool(oversize),
        oversize_clusters=oversize,
    )


@_time_pipeline_stage("localization")
def localize_grouped_recovery(
    recovery: RecoveryResult,
    grouping: GroupingResult,
    *,
    d: int,
    config: Optional[LocalizationConfig] = None,
) -> LocalizationResult:
    """Simultaneously localize every learned ``P(C)`` using one Clifford.

    Binary columns use ``[x_0,...,x_{m-1}|z_0,...,z_{m-1}]``.  The returned
    tableau ``R`` and dense unitary obey the manuscript orientation
    ``U_rec^dagger P(a) U_rec = +/- P(R a)``.
    """
    if not isinstance(recovery, RecoveryResult):
        raise TypeError("recovery must be a RecoveryResult.")
    if not isinstance(grouping, GroupingResult):
        raise TypeError("grouping must be a GroupingResult.")
    d = _validate_positive_integer("d", d)
    if config is None:
        config = LocalizationConfig()
    if not isinstance(config, LocalizationConfig):
        raise TypeError("config must be a LocalizationConfig.")
    try:
        validate_grouping_against_recovery(
            grouping,
            recovery,
            enforce_cluster_size_bound=config.enforce_model_block_bound,
        )
    except (RecoveryPreconditionError, ValueError) as error:
        return _localization_failure(
            recovery, grouping, d, f"invalid_grouping_recovery_handoff: {error}"
        )
    handoff_valid = True
    model_bound_violated = any(len(cluster) > d for cluster in grouping.clusters)
    if config.enforce_model_block_bound and model_bound_violated:
        return _localization_failure(
            recovery,
            grouping,
            d,
            "cluster_exceeds_known_block_cap_d",
            handoff_valid=True,
        )
    theorem_upstream = bool(grouping.theorem_grouping_preconditions_hold)
    if not theorem_upstream and not config.allow_uncertified_grouping:
        return _localization_failure(
            recovery,
            grouping,
            d,
            "grouping_not_theorem_certified",
            handoff_valid=True,
        )
    if config.materialize_dense_clifford and recovery.n > config.max_dense_qubits:
        return _localization_failure(
            recovery,
            grouping,
            d,
            "dense_clifford_resource_guard_exceeded",
            handoff_valid=True,
        )

    m = recovery.m
    clusters = tuple(tuple(cluster) for cluster in grouping.clusters)
    try:
        structures = tuple(
            analyze_group_symplectic_structure(recovery, cluster)
            for cluster in clusters
        )
        if config.enforce_model_block_bound and any(
            structure.k_C > d for structure in structures
        ):
            raise LocalizationInvariantError("Computed k_C exceeds known block cap d.")

        all_axes = tuple(
            axis for structure in structures for axis in structure.ordered_independent_axes
        )
        dimension_sum = sum(structure.dimension for structure in structures)
        global_rank = (
            int(
                _v1._gf2_rank(
                    np.column_stack(
                        [_v1.pauli_to_symplectic_col(axis) for axis in all_axes]
                    )
                )
            )
            if all_axes
            else 0
        )
        direct_sum = global_rank == dimension_sum == len(recovery.independent_axes)
        if m > 0:
            direct_sum = direct_sum and (
                canonical_gf2_span_basis(all_axes, m)
                == tuple(recovery.recovered_span_basis)
            )
        elif all_axes or recovery.recovered_span_basis:
            direct_sum = False
        if not direct_sum:
            raise LocalizationInvariantError("Cross-group direct-sum invariant failed.")

        J = _symplectic_form_matrix(m)
        cross_orthogonal = True
        for left_index, left in enumerate(structures):
            left_vectors = [np.asarray(vector, dtype=np.uint8) for vector in left.axis_vectors]
            for right in structures[left_index + 1 :]:
                right_vectors = [
                    np.asarray(vector, dtype=np.uint8) for vector in right.axis_vectors
                ]
                if any(
                    int((u @ J @ v) % 2)
                    for u in left_vectors
                    for v in right_vectors
                ):
                    cross_orthogonal = False
        if not cross_orthogonal:
            raise LocalizationInvariantError(
                "Cross-group symplectic orthogonality invariant failed."
            )

        registers = []
        next_register = 0
        for structure in structures:
            register = tuple(range(next_register, next_register + structure.k_C))
            registers.append(register)
            next_register += structure.k_C
        K_rec = next_register
        if K_rec > m:
            raise LocalizationInvariantError("K_rec exceeds residual qubit count m.")
        J_aux = tuple(range(K_rec, m))
        register_partition = (
            len({qubit for register in registers for qubit in register}) == K_rec
            and tuple(qubit for register in registers for qubit in register) + J_aux
            == tuple(range(m))
            and all(len(register) == structure.k_C for register, structure in zip(registers, structures))
        )
        if not register_partition:
            raise LocalizationInvariantError("Residual register partition is invalid.")

        hyperbolic_specs = []
        radical_specs = []
        for structure_index, structure in enumerate(structures):
            for local_index, pair in enumerate(structure.hyperbolic_pairs):
                hyperbolic_specs.append(
                    (
                        structure_index,
                        local_index,
                        np.asarray(pair[0], dtype=np.uint8),
                        np.asarray(pair[1], dtype=np.uint8),
                    )
                )
            for local_index, radical in enumerate(structure.radical_basis):
                radical_specs.append(
                    (
                        structure_index,
                        local_index,
                        np.asarray(radical, dtype=np.uint8),
                    )
                )
        primary_vectors = [spec[2] for spec in hyperbolic_specs] + [
            spec[2] for spec in radical_specs
        ]
        paired_vectors = [spec[3] for spec in hyperbolic_specs]
        Xcols = (
            np.column_stack(primary_vectors).astype(np.uint8)
            if primary_vectors
            else np.zeros((2 * m, 0), dtype=np.uint8)
        )
        Zpaired = (
            np.column_stack(paired_vectors).astype(np.uint8)
            if paired_vectors
            else np.zeros((2 * m, 0), dtype=np.uint8)
        )
        source_basis = _v1._complete_partial_symplectic_basis(Xcols, Zpaired)
        if not _v1.is_symplectic(source_basis) and m > 0:
            raise LocalizationInvariantError("Global source completion is not symplectic.")

        pair_targets = []
        pair_indices_by_structure: list[list[int]] = [[] for _ in structures]
        target_qubits_by_structure: list[list[int]] = [[] for _ in structures]
        for pair_index, (structure_index, local_index, _left, _right) in enumerate(
            hyperbolic_specs
        ):
            target = registers[structure_index][local_index]
            pair_targets.append(target)
            pair_indices_by_structure[structure_index].append(pair_index)
            target_qubits_by_structure[structure_index].append(target)
        hyperbolic_count = len(hyperbolic_specs)
        for radical_offset, (structure_index, local_index, _radical) in enumerate(
            radical_specs
        ):
            pair_index = hyperbolic_count + radical_offset
            target = registers[structure_index][structures[structure_index].h_C + local_index]
            pair_targets.append(target)
            pair_indices_by_structure[structure_index].append(pair_index)
            target_qubits_by_structure[structure_index].append(target)
        pair_targets.extend(J_aux)
        if tuple(sorted(pair_targets)) != tuple(range(m)):
            raise LocalizationInvariantError("Source/target pair allocation is not bijective.")
        target_X = [_canonical_pauli_vector(m, qubit, "X") for qubit in pair_targets]
        target_Z = [_canonical_pauli_vector(m, qubit, "Z") for qubit in pair_targets]
        target_basis = (
            np.column_stack(target_X + target_Z).astype(np.uint8)
            if m
            else np.zeros((0, 0), dtype=np.uint8)
        )
        if not _v1.is_symplectic(target_basis) and m > 0:
            raise LocalizationInvariantError("Global target basis is not symplectic.")
        source_inverse = _gf2_inverse_matrix(source_basis)
        residual_tableau = (target_basis @ source_inverse) % 2
        synthesis_tableau = _gf2_inverse_matrix(residual_tableau)
        if m > 0 and not _v1.is_symplectic(residual_tableau):
            raise LocalizationInvariantError("Residual localization map is not symplectic.")
        if not np.array_equal((residual_tableau @ source_basis) % 2, target_basis):
            raise LocalizationInvariantError("Source vectors do not map to targets.")

        if m == 0:
            gates: Tuple[Tuple, ...] = ()
            residual_compact = None
            U_rec = (
                np.ones((1, 1), dtype=complex)
                if config.materialize_dense_clifford
                else None
            )
        else:
            gates = tuple(_v1.synthesize_clifford_from_tableau(synthesis_tableau))
            residual_compact = SignedClifford.from_gates(m, gates)
            if not np.array_equal(residual_compact.tableau, residual_tableau):
                raise LocalizationInvariantError(
                    "Compact U_rec disagrees with the localization tableau."
                )
            U_rec = (
                residual_compact.materialize_dense_debug(
                    max_qubits=config.max_dense_qubits
                )
                if config.materialize_dense_clifford
                else None
            )
        if config.verify_dense_unitary and U_rec is not None and not tableau_unitary_convention_holds(
            residual_tableau, U_rec
        ):
            raise LocalizationInvariantError(
                "Dense U_rec disagrees with U_rec^dagger P U_rec localization orientation."
            )

        full_tableau = _full_residual_tableau(residual_tableau, recovery.n, recovery.t)
        full_compact = (
            SignedClifford.identity(recovery.n)
            if m == 0
            else residual_compact.embedded(recovery.n, recovery.t)
        )
        if not np.array_equal(full_compact.tableau, full_tableau):
            raise LocalizationInvariantError("Full compact localization tableau mismatch.")
        bar_U_rec = (
            np.kron(
                np.eye(2**recovery.t, dtype=complex),
                np.asarray(U_rec, dtype=complex),
            )
            if U_rec is not None
            else None
        )
        if config.verify_dense_unitary and bar_U_rec is not None and not tableau_unitary_convention_holds(
            full_tableau, bar_U_rec
        ):
            raise LocalizationInvariantError("Full-n bar_U_rec convention check failed.")

        localization_holds = True
        for structure, register in zip(structures, registers):
            allowed = set(register)
            generated = generated_cluster_pauli_group(
                structure.cluster,
                recovery.sectors,
                max_group_size=config.max_generated_group_size,
            )
            for pauli in generated.paulis:
                vector = _v1.pauli_to_symplectic_col(pauli)
                localized = (residual_tableau @ vector) % 2
                if not set(_residual_support(localized, m)).issubset(allowed):
                    localization_holds = False
                    break
        if not localization_holds:
            raise LocalizationInvariantError(
                "A generated empirical-group Pauli leaks outside J_C."
            )

        cluster_localizations = []
        radical_index_lookup = {
            (structure_index, local_index): hyperbolic_count + offset
            for offset, (structure_index, local_index, _radical) in enumerate(radical_specs)
        }
        for structure_index, (structure, register) in enumerate(
            zip(structures, registers)
        ):
            partners = tuple(
                _vector_tuple(
                    source_basis[
                        :,
                        m + radical_index_lookup[(structure_index, local_index)],
                    ]
                )
                for local_index in range(structure.q_C)
            )
            cluster_localizations.append(
                ClusterLocalization(
                    cluster=structure.cluster,
                    h_C=structure.h_C,
                    q_C=structure.q_C,
                    k_C=structure.k_C,
                    J_C=register,
                    source_hyperbolic_pairs=structure.hyperbolic_pairs,
                    source_radical_axes=structure.radical_basis,
                    radical_partners=partners,
                    source_pair_indices=tuple(pair_indices_by_structure[structure_index]),
                    target_pair_qubits=tuple(
                        target_qubits_by_structure[structure_index]
                    ),
                )
            )

        transcript = ()
        if config.return_details:
            transcript = (
                "columns=[X_0,...,X_{m-1}|Z_0,...,Z_{m-1}]",
                "source hyperbolic primaries precede all radical primaries",
                "radical partners and auxiliary pairs completed simultaneously",
                "U_rec^dagger P(source) U_rec = +/- P(target)",
                "bar_U_rec=I_prefix tensor U_rec",
            )
        return LocalizationResult(
            success=True,
            failure_reason=None,
            n=recovery.n,
            t=recovery.t,
            m=m,
            d=d,
            clusters=clusters,
            structures=structures,
            cluster_localizations=tuple(cluster_localizations),
            J_C=tuple(
                (structure.cluster, register)
                for structure, register in zip(structures, registers)
            ),
            J_aux=J_aux,
            K_rec=K_rec,
            source_basis=source_basis,
            target_basis=target_basis,
            residual_tableau=residual_tableau,
            synthesis_tableau=synthesis_tableau,
            full_tableau=full_tableau,
            gates=gates,
            U_rec=U_rec,
            bar_U_rec=bar_U_rec,
            grouping_theorem_preconditions_hold=theorem_upstream,
            theorem_localization_preconditions_hold=(
                theorem_upstream and not model_bound_violated
            ),
            handoff_valid=handoff_valid,
            cross_group_direct_sum_holds=direct_sum,
            cross_group_symplectic_orthogonality_holds=cross_orthogonal,
            global_pairing_holds=True,
            register_partition_holds=register_partition,
            localization_guarantee_holds=localization_holds,
            localization_copy_count=0,
            cumulative_copy_ledger=grouping.cumulative_copy_ledger,
            verification_transcript=transcript,
            signed_clifford=full_compact,
            empirical_max_cluster_size=max(
                (len(cluster) for cluster in clusters), default=0
            ),
            assumed_d=d,
            model_bound_violated=model_bound_violated,
            oversize_clusters=tuple(
                (tuple(cluster), len(cluster))
                for cluster in clusters
                if len(cluster) > d
            ),
            reconstruction_proceeded_despite_model_bound_violation=(
                model_bound_violated
            ),
        )
    except (LocalizationInvariantError, ValueError, RuntimeError) as error:
        return _localization_failure(
            recovery,
            grouping,
            d,
            f"localization_invariant_failed: {error}",
            handoff_valid=handoff_valid,
        )


def syndrome_sign_sample_count(n: int, h_min: float, zeta_sgn: float) -> int:
    """Exact ceiling in Eq. ``eq:supp-syndrome-sign-sample-choice``."""
    n = _validate_positive_integer("n", n)
    if not (0.5 < float(h_min) < 1.0):
        raise ValueError("h_min must lie in (1/2,1).")
    if not (0.0 < float(zeta_sgn) < 1.0):
        raise ValueError("zeta_sgn must lie in (0,1).")
    return int(math.ceil(2.0 / float(h_min) * math.log(2.0 * n / zeta_sgn)))


def measure_pauli_expectation(
    source: Union[CEBPLearnerView, SimulatorMeasurementSource],
    pauli: str,
    shots: int,
    *,
    seed: RngSeed = None,
) -> PauliMeasurementRecord:
    """Measure one physical Hermitian Pauli on ``shots`` fresh copies."""
    measurement_source = _coerce_measurement_source(source)
    shots = _validate_positive_integer("shots", shots)
    seed = _validate_seed(seed)
    _validate_pauli_string(pauli, measurement_source.n)
    mean = float(
        np.clip(measurement_source._expectation_for_backend(pauli), -1.0, 1.0)
    )
    rng = np.random.default_rng(seed)
    outcomes = np.where(rng.random(shots) < (1.0 + mean) / 2.0, 1, -1)
    return PauliMeasurementRecord(pauli, shots, seed, outcomes)


def _sample_pauli_empirical_mean(
    source: Union[CEBPLearnerView, SimulatorMeasurementSource],
    pauli: str,
    shots: int,
    *,
    seed: RngSeed = None,
) -> float:
    """Sample the exact count-only sufficient statistic for one Pauli batch.

    A binomial draw preserves the distribution of ``shots`` independent
    two-outcome measurements while avoiding materialized per-shot outcomes.
    The RNG primitive intentionally differs from ``measure_pauli_expectation``.
    """
    measurement_source = _coerce_measurement_source(source)
    shots = _validate_positive_integer("shots", shots)
    seed = _validate_seed(seed)
    _validate_pauli_string(pauli, measurement_source.n)
    expectation = float(measurement_source._expectation_for_backend(pauli))
    if not np.isfinite(expectation):
        raise RuntimeError("A Hermitian Pauli expectation must be finite.")
    mean = float(np.clip(expectation, -1.0, 1.0))
    p_plus = float(np.clip((1.0 + mean) / 2.0, 0.0, 1.0))
    plus_count = int(np.random.default_rng(seed).binomial(shots, p_plus))
    return float((2 * plus_count - shots) / shots)


def _syndrome_failure(
    peeling: PeelingResult,
    config: SyndromeConfig,
    reason: str,
    prior_copy_ledger: CopyLedger,
) -> SyndromeResult:
    return SyndromeResult(
        success=False,
        failure_reason=reason,
        t=0 if peeling.t is None else int(peeling.t),
        syndrome_bits=(),
        empirical_means=(),
        M_sgn=0,
        zeta_sgn=float(config.zeta_sgn),
        h_min=float(config.h_min),
        calibrated_M_sgn=0,
        theorem_preconditions_hold=False,
        syndrome_sign_pool=0,
        copy_ledger=CopyLedger((("syndrome_sign_pool", 0),)),
        cumulative_copy_ledger=prior_copy_ledger.with_entry("syndrome_sign_pool", 0),
        provenance=DataProvenance.EMPIRICAL,
    )


@_time_pipeline_stage("syndrome")
def recover_peeling_syndrome(
    source: Union[CEBPLearnerView, SimulatorMeasurementSource],
    peeling: PeelingResult,
    config: SyndromeConfig,
    *,
    seed: RngSeed = None,
    prior_copy_ledger: Optional[CopyLedger] = None,
) -> SyndromeResult:
    """Recover peeled syndrome bits from fresh signed physical measurements."""
    if not isinstance(peeling, PeelingResult):
        raise TypeError("peeling must be a PeelingResult.")
    if not isinstance(config, SyndromeConfig):
        raise TypeError("config must be a SyndromeConfig.")
    seed = _validate_seed(seed)
    prior = peeling.copy_ledger if prior_copy_ledger is None else prior_copy_ledger
    if "syndrome_sign_pool" in prior.as_dict():
        raise ValueError("prior ledger already contains syndrome_sign_pool.")
    if not peeling.success or peeling.t is None:
        return _syndrome_failure(peeling, config, "peeling_failed", prior)
    measurement_source = _coerce_measurement_source(source)
    n = _peeling_dimension(peeling)
    if measurement_source.n != n:
        return _syndrome_failure(peeling, config, "source_dimension_mismatch", prior)
    t = int(peeling.t)
    calibrated = syndrome_sign_sample_count(n, config.h_min, config.zeta_sgn)
    if t == 0:
        stage = CopyLedger((("syndrome_sign_pool", 0),))
        return SyndromeResult(
            True,
            None,
            0,
            (),
            (),
            0,
            float(config.zeta_sgn),
            float(config.h_min),
            calibrated,
            bool(
                peeling.theorem_preconditions_hold
                and peeling.h is not None
                and config.h_min <= float(peeling.h)
            ),
            0,
            stage,
            prior.with_entry("syndrome_sign_pool", 0),
            DataProvenance.EMPIRICAL,
            (),
        )
    M_sgn = calibrated if config.M_sgn is None else int(config.M_sgn)
    if M_sgn <= 0:
        return _syndrome_failure(peeling, config, "positive_M_sgn_required", prior)
    child_seeds = _v1._child_seeds(seed, t) if seed is not None else [None] * t
    if config.return_details:
        records = tuple(
            measure_pauli_expectation(
                measurement_source, generator, M_sgn, seed=child_seed
            )
            for generator, child_seed in zip(peeling.generators, child_seeds)
        )
        means = tuple(record.empirical_mean for record in records)
    else:
        records = ()
        means = tuple(
            _sample_pauli_empirical_mean(
                measurement_source, generator, M_sgn, seed=child_seed
            )
            for generator, child_seed in zip(peeling.generators, child_seeds)
        )
    bits = tuple(0 if mean >= 0.0 else 1 for mean in means)
    copies = t * M_sgn
    stage = CopyLedger((("syndrome_sign_pool", copies),))
    return SyndromeResult(
        True,
        None,
        t,
        bits,
        means,
        M_sgn,
        float(config.zeta_sgn),
        float(config.h_min),
        calibrated,
        bool(
            peeling.theorem_preconditions_hold
            and peeling.h is not None
            and config.h_min <= float(peeling.h)
            and M_sgn >= calibrated
        ),
        copies,
        stage,
        prior.with_entry("syndrome_sign_pool", copies),
        DataProvenance.EMPIRICAL,
        records,
    )


def localized_measurement_source(
    source: Union[CEBPLearnerView, SimulatorMeasurementSource],
    peeling: PeelingResult,
    localization: LocalizationResult,
) -> SimulatorMeasurementSource:
    """Apply ``bar_U_rec^dagger U_stab^dagger`` in the measurement backend."""
    if not isinstance(localization, LocalizationResult) or not localization.success:
        raise ValueError("Successful localization is required.")
    n = _peeling_dimension(peeling)
    if (localization.n, localization.t, localization.m) != (
        n,
        int(peeling.t),
        n - int(peeling.t),
    ):
        raise ValueError("Peeling and localization dimensions disagree.")
    peeled = _peeled_measurement_source(source, peeling)
    if peeled.uses_structured_backend and localization.signed_clifford is not None:
        return peeled._transformed_by_dagger(localization.signed_clifford)
    if localization.bar_U_rec is None:
        raise ValueError("Dense localization fallback is unavailable.")
    bar = np.asarray(localization.bar_U_rec, dtype=complex)
    if bar.shape != (2**n, 2**n):
        raise ValueError("bar_U_rec has the wrong full-system dimension.")
    bar_q = qt.Qobj(bar, dims=[[2] * n, [2] * n])
    state = peeled._state_for_backend()
    if peeled.is_ket:
        localized = (bar_q.dag() * state).unit()
        localized.dims = [[2] * n, [1] * n]
    else:
        localized = bar_q.dag() * state * bar_q
        localized = 0.5 * (localized + localized.dag())
        localized = localized / localized.tr()
        localized.dims = [[2] * n, [2] * n]
    return SimulatorMeasurementSource(n, peeled.is_ket, localized)


def debug_exact_localized_state(
    source: Union[CEBPLearnerView, SimulatorMeasurementSource],
    peeling: PeelingResult,
    localization: LocalizationResult,
    *,
    max_dense_debug_qubits: int = 8,
) -> qt.Qobj:
    """Debug-only materialization of Eq. ``eq:supp-computable-localized-state``."""
    return localized_measurement_source(
        source, peeling, localization
    )._state_for_backend(
        max_dense_debug_qubits=max_dense_debug_qubits
    )


def debug_exact_register_marginal(
    source: Union[CEBPLearnerView, SimulatorMeasurementSource],
    peeling: PeelingResult,
    localization: LocalizationResult,
    J_C: Sequence[int],
    *,
    max_dense_debug_qubits: int = 8,
) -> qt.Qobj:
    """Debug-only exact marginal on a localized residual register."""
    register = tuple(int(qubit) for qubit in J_C)
    if len(set(register)) != len(register) or any(
        qubit < 0 or qubit >= localization.m for qubit in register
    ):
        raise ValueError("J_C must contain distinct residual-register indices.")
    keep = [localization.t + qubit for qubit in register]
    if not keep:
        return qt.Qobj([[1.0]], dims=[[1], [1]])
    localized_source = localized_measurement_source(source, peeling, localization)
    if localized_source.uses_structured_backend:
        # Reconstruct only the requested k-qubit factor from its 4**k Pauli
        # coefficients.  This avoids a full 10-qubit state for every register.
        k = len(keep)
        marginal = qt.Qobj(
            np.zeros((2**k, 2**k), dtype=complex),
            dims=[[2] * k, [2] * k],
        )
        for index in range(4**k):
            local_pauli = pauli_index_to_string(index, k)
            characters = ["I"] * localized_source.n
            for qubit, character in zip(keep, local_pauli):
                characters[qubit] = character
            coefficient = localized_source._expectation_for_backend(
                "".join(characters)
            )
            marginal += coefficient * _v1._qutip_pauli_op(k, local_pauli)
        marginal /= 2**k
        marginal = 0.5 * (marginal + marginal.dag())
        marginal /= marginal.tr()
        marginal.dims = [[2] * k, [2] * k]
        return marginal
    state = localized_source._state_for_backend(
        max_dense_debug_qubits=max_dense_debug_qubits
    )
    density = state * state.dag() if state.isket else state
    marginal = density.ptrace(keep)
    marginal.dims = [[2] * len(keep), [2] * len(keep)]
    return marginal


@dataclass(frozen=True)
class ExactOperationalDiagnostic:
    """Oracle-only zero-copy structural run for threshold diagnosis."""

    peeling_t: int
    peeling_span: Tuple[int, ...]
    peeling_generators: Tuple[str, ...]
    recovery_sector_count: int
    recovery_sector_types: Tuple[str, ...]
    recovery_span: Tuple[int, ...]
    oracle_sector_block_labels: Tuple[Tuple[int, ...], ...]
    grouping_partition: Tuple[Tuple[int, ...], ...]
    localization_registers: Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], ...]
    J_aux: Tuple[int, ...]
    syndrome_bits: Tuple[int, ...]
    syndrome_expectations: Tuple[float, ...]
    structural_trace_distance: float
    realized_physical_copies: int = 0


def debug_exact_operational_diagnostic(
    instance: CEBPInstance,
    *,
    h_min: float,
    h_max: float,
    theta: float,
    eta_test: float,
    peeling_grid_intervals: int = 5,
    max_dense_debug_qubits: int = 8,
) -> ExactOperationalDiagnostic:
    """Replay the operational thresholds with exact stage data only.

    Oracle labels are attached only after all structural learner decisions.
    They are diagnostic output and never feed peeling, recovery, grouping, or
    localization.
    """

    if not 0.5 < float(h_min) < float(h_max) < 1.0:
        raise ValueError("Require 1/2 < h_min < h_max < 1.")
    if not 0.0 < float(theta) < 1.0 or float(eta_test) <= 0.0:
        raise ValueError("theta must lie in (0,1) and eta_test must be positive.")
    intervals = _validate_positive_integer(
        "peeling_grid_intervals", peeling_grid_intervals
    )
    peeling = debug_exact_certified_stabilizer_peeling(
        instance,
        PeelingConfig(
            h_min=float(h_min),
            h_max=float(h_max),
            eta=(float(h_max) - float(h_min)) / intervals,
            M1=None,
            zeta_bs=0.05,
            max_dense_debug_qubits=max_dense_debug_qubits,
            materialize_dense_clifford=False,
        ),
    )
    if not peeling.success:
        raise RuntimeError("Exact operational peeling was inconclusive.")
    recovery = debug_exact_rank_guided_sector_recovery(
        instance,
        peeling,
        RecoveryConfig(
            theta=float(theta),
            M2=None,
            zeta_rank=0.05,
            allow_uncalibrated_peeling=True,
            allow_margin_failure=True,
        ),
    )
    if not recovery.success:
        raise RuntimeError("Exact operational recovery was inconclusive.")
    grouping = debug_exact_hierarchical_cumulant_grouping(
        instance,
        peeling,
        recovery,
        GroupingConfig(
            ell_grp=instance.d,
            eta_test=float(eta_test),
            tau_kappa=0.0,
            allow_uncalibrated_recovery=True,
            allow_no_false_merge_margin_failure=True,
        ),
    )
    if not grouping.success:
        raise RuntimeError("Exact operational grouping was inconclusive.")
    localization = localize_grouped_recovery(
        recovery,
        grouping,
        d=instance.d,
        config=LocalizationConfig(
            allow_uncertified_grouping=True,
            verify_dense_unitary=False,
            max_dense_qubits=max_dense_debug_qubits,
            materialize_dense_clifford=False,
        ),
    )
    if not localization.success:
        raise RuntimeError("Exact operational localization was inconclusive.")
    factors = tuple(
        (
            register,
            debug_exact_register_marginal(
                instance.learner_view(),
                peeling,
                localization,
                register,
                max_dense_debug_qubits=max_dense_debug_qubits,
            ),
        )
        for _cluster, register in localization.J_C
    )
    syndrome_expectations = tuple(
        instance.measurement_source._expectation_for_backend(generator)
        for generator in peeling.generators
    )
    syndrome_bits = tuple(
        0 if expectation >= 0.0 else 1 for expectation in syndrome_expectations
    )
    compact = CompactCEBPEstimator(
        n=localization.n,
        t=localization.t,
        m=localization.m,
        U_stab=(
            None if peeling.U_stab is None else np.asarray(peeling.U_stab, dtype=complex)
        ),
        bar_U_rec=(
            None
            if localization.bar_U_rec is None
            else np.asarray(localization.bar_U_rec, dtype=complex)
        ),
        syndrome_bits=syndrome_bits,
        register_estimates=tuple(
            (tuple(cluster), tuple(register), marginal)
            for (cluster, register), (_same_register, marginal) in zip(
                localization.J_C, factors
            )
        ),
        J_aux=tuple(localization.J_aux),
        peeling_gates=tuple(peeling.gates),
        recovery_gates=tuple(localization.gates),
        peeling_clifford=peeling.signed_clifford,
        recovery_clifford=localization.signed_clifford,
    )
    decoded = materialize_compact_cebp_estimator(
        compact, max_dense_qubits=max_dense_debug_qubits
    )
    exact_state = instance.materialize_state_debug(
        max_qubits=max_dense_debug_qubits
    )
    target_density = (
        exact_state * exact_state.dag() if exact_state.isket else exact_state
    )
    difference = np.asarray(target_density.full() - decoded.full(), dtype=complex)
    trace_distance = 0.5 * float(np.linalg.svd(difference, compute_uv=False).sum())
    sector_types = tuple(sector.kind for sector in recovery.sectors)
    return ExactOperationalDiagnostic(
        peeling_t=int(peeling.t),
        peeling_span=tuple(peeling.certified_span_basis),
        peeling_generators=tuple(peeling.generators),
        recovery_sector_count=len(recovery.sectors),
        recovery_sector_types=sector_types,
        recovery_span=tuple(recovery.recovered_span_basis),
        oracle_sector_block_labels=debug_oracle_sector_block_labels(
            instance, peeling, recovery
        ),
        grouping_partition=tuple(grouping.clusters),
        localization_registers=tuple(localization.J_C),
        J_aux=tuple(localization.J_aux),
        syndrome_bits=syndrome_bits,
        syndrome_expectations=syndrome_expectations,
        structural_trace_distance=trace_distance,
    )


def register_tomography_budget(
    cluster: Sequence[int],
    J_C: Sequence[int],
    epsilon_C: float,
    zeta_C: float,
) -> RegisterTomographyBudget:
    """Construct the exact per-register budget from the manuscript proof."""
    cluster = tuple(int(value) for value in cluster)
    register = tuple(int(value) for value in J_C)
    k_C = len(register)
    if not cluster or k_C <= 0:
        raise ValueError("A nonempty cluster and register are required.")
    if len(set(register)) != k_C or any(value < 0 for value in register):
        raise ValueError("J_C must contain distinct nonnegative qubits.")
    if not np.isfinite(epsilon_C) or epsilon_C <= 0.0:
        raise ValueError("epsilon_C must be finite and positive.")
    if not (0.0 < float(zeta_C) < 1.0):
        raise ValueError("zeta_C must lie in (0,1).")
    count = 4**k_C - 1
    tau = float(epsilon_C) / (4.0 * math.sqrt(count))
    shots = int(math.ceil(2.0 / tau**2 * math.log(2.0 * count / zeta_C)))
    return RegisterTomographyBudget(
        cluster,
        register,
        k_C,
        count,
        float(epsilon_C),
        float(zeta_C),
        tau,
        shots,
        count * shots,
    )


def _tomography_budgets(
    localization: LocalizationResult, config: TomographyConfig
) -> Tuple[RegisterTomographyBudget, ...]:
    registers = tuple((tuple(cluster), tuple(J_C)) for cluster, J_C in localization.J_C)
    Khat = len(registers)
    if Khat == 0:
        if config.epsilon_by_cluster or config.zeta_by_cluster:
            raise ValueError("Empty localization cannot have per-cluster budgets.")
        return ()
    epsilon_map = dict(config.epsilon_by_cluster)
    zeta_map = dict(config.zeta_by_cluster)
    expected = {cluster for cluster, _register in registers}
    if epsilon_map and set(epsilon_map) != expected:
        raise ValueError("epsilon_C map must match localization clusters exactly.")
    if zeta_map and set(zeta_map) != expected:
        raise ValueError("zeta_C map must match localization clusters exactly.")
    if not epsilon_map:
        epsilon_map = {cluster: config.epsilon_tom / Khat for cluster in expected}
    if not zeta_map:
        zeta_map = {cluster: config.zeta_tom / Khat for cluster in expected}
    if sum(epsilon_map.values()) > config.epsilon_tom + 1e-12:
        raise ValueError("Local epsilon_C budgets exceed epsilon_tom.")
    if sum(zeta_map.values()) > config.zeta_tom + 1e-12:
        raise ValueError("Local zeta_C budgets exceed zeta_tom.")
    budgets = tuple(
        register_tomography_budget(
            cluster, register, epsilon_map[cluster], zeta_map[cluster]
        )
        for cluster, register in registers
    )
    if any(budget.k_C > localization.d for budget in budgets):
        raise ValueError("A tomography register exceeds the known block cap d.")
    return budgets


def project_eigenvalues_to_simplex(values: Sequence[float]) -> np.ndarray:
    """Euclidean projection of a real vector onto the probability simplex."""
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("Simplex projection requires a nonempty finite vector.")
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered)
    indices = np.arange(1, vector.size + 1)
    active = np.nonzero(ordered - (cumulative - 1.0) / indices > 0.0)[0]
    rho = int(active[-1])
    theta = float((cumulative[rho] - 1.0) / (rho + 1))
    projected = np.maximum(vector - theta, 0.0)
    projected /= projected.sum()
    return projected


def project_to_density_matrix_hs(
    matrix: Union[np.ndarray, qt.Qobj], *, tolerance: float = 1e-10
) -> qt.Qobj:
    """Represent the Hilbert--Schmidt projection as a physical matrix.

    This public convenience wrapper returns the represented matrix.  The
    theorem-facing tomography path additionally records the conservative
    floating-point representation bound produced by
    :func:`_project_to_density_matrix_hs_with_error_bound`.
    """
    projected, _bound = _project_to_density_matrix_hs_with_error_bound(
        matrix, tolerance=tolerance
    )
    return projected


def _project_to_density_matrix_hs_with_error_bound(
    matrix: Union[np.ndarray, qt.Qobj], *, tolerance: float = 1e-10
) -> Tuple[qt.Qobj, float]:
    """Return the represented projection and a numerical trace-norm bound.

    ``nu_star`` and ``nu_hat`` use the same computed matrix as a simulator
    representation convention.  The returned nonzero allowance nevertheless
    certifies the floating-point eigensolve, simplex reconstruction, and final
    Hermitian cleanup at the requested tolerance; it is not an oracle error.
    """
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    qobj = matrix if isinstance(matrix, qt.Qobj) else qt.Qobj(np.asarray(matrix))
    array = np.asarray(qobj.full(), dtype=complex)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("Density projection requires a square matrix.")
    if not np.allclose(array, array.conj().T, atol=tolerance):
        raise ValueError("Density projection requires a Hermitian matrix.")
    array = 0.5 * (array + array.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(array)
    projected_values = project_eigenvalues_to_simplex(eigenvalues)
    raw_projected = (eigenvectors * projected_values) @ eigenvectors.conj().T
    projected = 0.5 * (raw_projected + raw_projected.conj().T)
    dimension = projected.shape[0]
    qubits = int(round(math.log2(dimension))) if dimension > 1 else 0
    if 2**qubits != dimension:
        raise ValueError("Projected density dimension must be a power of two.")
    scale = max(1.0, float(np.linalg.norm(array, ord=2)))
    floating_floor = float(
        32.0 * np.finfo(float).eps * max(1, dimension**2) * scale
    )
    cleanup = raw_projected - projected
    cleanup_trace_norm = float(np.linalg.svd(cleanup, compute_uv=False).sum())
    numerical_bound = cleanup_trace_norm + floating_floor
    trace_residual = abs(complex(np.trace(projected)) - 1.0)
    hermitian_residual = float(np.linalg.norm(projected - projected.conj().T, ord=2))
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(projected)))
    observed_residual = max(
        trace_residual,
        hermitian_residual,
        max(0.0, -minimum_eigenvalue),
        cleanup_trace_norm,
        floating_floor,
    )
    if tolerance < observed_residual:
        raise ValueError(
            "numerical_tolerance is below the floating-point projection "
            "certification floor."
        )
    dims = [[2] * qubits, [2] * qubits] if qubits else [[1], [1]]
    return qt.Qobj(projected, dims=dims), numerical_bound


def _local_paulis(k_C: int) -> Tuple[str, ...]:
    identity = "I" * k_C
    return tuple(pauli for pauli in _all_pauli_strings(k_C) if pauli != identity)


def linear_inversion_from_pauli_coefficients(
    k_C: int, coefficients: Mapping[str, float]
) -> qt.Qobj:
    """Reconstruct a Hermitian trace-one local matrix from all Pauli moments."""
    k_C = _validate_positive_integer("k_C", k_C)
    expected = _local_paulis(k_C)
    if set(coefficients) != set(expected) or len(coefficients) != len(expected):
        raise ValueError("Coefficient map must contain every nonidentity Pauli once.")
    dimension = 2**k_C
    array = np.eye(dimension, dtype=complex)
    for pauli in expected:
        value = float(coefficients[pauli])
        if not np.isfinite(value) or abs(value) > 1.0 + 1e-10:
            raise ValueError("Pauli coefficients must be finite values in [-1,1].")
        array += value * _v1._qutip_pauli_op(k_C, pauli).full()
    array /= dimension
    array = 0.5 * (array + array.conj().T)
    if abs(np.trace(array) - 1.0) > 1e-9:
        raise RuntimeError("Linear inversion did not produce trace one.")
    return qt.Qobj(array, dims=[[2] * k_C, [2] * k_C])


def assemble_localized_product_estimator(
    factors: Sequence[Tuple[Sequence[int], qt.Qobj]],
    J_aux: Sequence[int],
    m: int,
) -> qt.Qobj:
    """Assemble factors on explicit residual registers, including permutations."""
    if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or int(m) < 0:
        raise ValueError("m must be a nonnegative integer.")
    m = int(m)
    normalized = []
    flattened = []
    for register, state in factors:
        register = tuple(int(value) for value in register)
        if not register:
            raise ValueError("Empirical factors need nonempty registers.")
        qobj = _v1._state_to_qobj(state, len(register))
        density = qobj * qobj.dag() if qobj.isket else qobj
        normalized.append(density)
        flattened.extend(register)
    auxiliary = tuple(int(value) for value in J_aux)
    flattened.extend(auxiliary)
    if len(flattened) != m or sorted(flattened) != list(range(m)):
        raise ValueError("Empirical and auxiliary registers must partition range(m).")
    if auxiliary:
        normalized.append(qt.qeye([2] * len(auxiliary)) / (2 ** len(auxiliary)))
    if m == 0:
        return qt.Qobj([[1.0]], dims=[[1], [1]])
    product = qt.tensor(normalized) if len(normalized) > 1 else normalized[0]
    permutation = [flattened.index(qubit) for qubit in range(m)]
    product = product.permute(permutation)
    product = 0.5 * (product + product.dag())
    product = product / product.tr()
    product.dims = [[2] * m, [2] * m]
    return product


def _product_pauli_probabilities_backend(
    source: SimulatorMeasurementSource, axes: Tuple[str, ...]
) -> np.ndarray:
    """Exact local-product probabilities without requiring a global density."""

    if source.uses_structured_backend:
        observables = tuple(
            "I" * qubit + axis + "I" * (source.n - qubit - 1)
            for qubit, axis in enumerate(axes)
        )
        # Structured tuple probabilities use (-1,+1) lexicographic outcomes;
        # computational basis indices use (+1,-1), which reverses that vector.
        return source._commuting_probabilities_for_backend(observables)[::-1]
    return _v1._local_product_pauli_measurement_probabilities(
        source._state_for_backend(), axes
    )


def _joint_disjoint_register_records(
    localized: SimulatorMeasurementSource,
    localization: LocalizationResult,
    budgets: Tuple[RegisterTomographyBudget, ...],
    *,
    seed: RngSeed,
    return_details: bool,
) -> Tuple[Tuple[RegisterTomographyRecord, ...], int, Tuple]:
    schedules = tuple(
        tuple(
            pauli
            for pauli in _local_paulis(budget.k_C)
            for _ in range(budget.M_C_Pauli)
        )
        for budget in budgets
    )
    N_bp = max((len(schedule) for schedule in schedules), default=0)
    collected = [
        {pauli: [] for pauli in _local_paulis(budget.k_C)} for budget in budgets
    ]
    rng = np.random.default_rng(seed)
    probability_cache: dict[Tuple[str, ...], Tuple[Tuple[Tuple[int, ...], ...], np.ndarray]] = {}
    transcript = []
    for round_index in range(N_bp):
        active = []
        full_paulis = []
        for index, (budget, schedule) in enumerate(zip(budgets, schedules)):
            if round_index >= len(schedule):
                continue
            pauli = schedule[round_index]
            active.append((index, budget, pauli))
            full = ["I"] * localized.n
            for local_index, character in enumerate(pauli):
                if character != "I":
                    full[localization.t + budget.J_C[local_index]] = character
            full_paulis.append("".join(full))
        query = tuple(full_paulis)
        if query not in probability_cache:
            probability_cache[query] = _commuting_tuple_probabilities_backend(
                localized, query
            )
        outcome_values, probabilities = probability_cache[query]
        sample = int(rng.choice(len(outcome_values), p=probabilities))
        sampled_outcome = outcome_values[sample]
        round_detail = []
        for outcome, (index, budget, pauli) in zip(sampled_outcome, active):
            collected[index][pauli].append(int(outcome))
            round_detail.append((budget.cluster, pauli))
        if return_details:
            transcript.append((round_index, tuple(round_detail)))
    records = tuple(
        RegisterTomographyRecord(
            budget.cluster,
            budget.J_C,
            schedule,
            tuple((pauli, tuple(collected[index][pauli])) for pauli in _local_paulis(budget.k_C)),
        )
        for index, (budget, schedule) in enumerate(zip(budgets, schedules))
    )
    return records, N_bp, tuple(transcript)


def _tomography_joint_segments(
    localized_n: int,
    localization: LocalizationResult,
    budgets: Tuple[RegisterTomographyBudget, ...],
) -> Tuple[Tuple[int, Tuple[str, ...], Tuple[Tuple[int, RegisterTomographyBudget, str], ...]], ...]:
    """Merge compact local Pauli runs into maximal joint-setting segments."""
    local_paulis = tuple(_local_paulis(budget.k_C) for budget in budgets)
    positions = [0] * len(budgets)
    remaining = [budget.M_C_Pauli if paulis else 0 for budget, paulis in zip(budgets, local_paulis)]
    segments = []
    while any(position < len(paulis) for position, paulis in zip(positions, local_paulis)):
        active_indices = [
            index
            for index, paulis in enumerate(local_paulis)
            if positions[index] < len(paulis)
        ]
        length = min(remaining[index] for index in active_indices)
        axes = ["Z"] * localized_n
        active = []
        for index in active_indices:
            budget = budgets[index]
            pauli = local_paulis[index][positions[index]]
            active.append((index, budget, pauli))
            for local_index, character in enumerate(pauli):
                if character != "I":
                    axes[localization.t + budget.J_C[local_index]] = character
        segments.append((length, tuple(axes), tuple(active)))
        for index in active_indices:
            remaining[index] -= length
            if remaining[index] == 0:
                positions[index] += 1
                if positions[index] < len(local_paulis[index]):
                    remaining[index] = budgets[index].M_C_Pauli
    return tuple(segments)


def _signed_sum_from_counts(
    counts: np.ndarray, n: int, measured_qubits: Sequence[int]
) -> int:
    """Return the exact +/-1 parity sum represented by computational counts."""
    count_values = np.asarray(counts, dtype=np.int64)
    if count_values.shape != (2**n,) or np.any(count_values < 0):
        raise ValueError("counts must be a nonnegative length-2**n vector.")
    positions = tuple(int(qubit) for qubit in measured_qubits)
    if len(set(positions)) != len(positions) or any(
        qubit < 0 or qubit >= n for qubit in positions
    ):
        raise ValueError("measured_qubits must be distinct entries in range(n).")
    categories = np.arange(2**n, dtype=np.int64)
    parity = np.zeros(2**n, dtype=np.int8)
    for qubit in positions:
        parity ^= ((categories >> (n - 1 - qubit)) & 1).astype(np.int8)
    signs = 1 - 2 * parity.astype(np.int64)
    return int(np.dot(count_values, signs))


def _signed_sum_from_joint_counts(
    counts: np.ndarray, outcomes: Sequence[Sequence[int]], position: int
) -> int:
    values = np.asarray(counts, dtype=np.int64)
    signs = np.asarray(outcomes, dtype=np.int8)
    if signs.ndim != 2 or values.shape != (signs.shape[0],):
        raise ValueError("Joint counts/outcomes have inconsistent shapes.")
    return int(np.dot(values, signs[:, int(position)].astype(np.int64)))


def _joint_disjoint_register_records_batched(
    localized: SimulatorMeasurementSource,
    localization: LocalizationResult,
    budgets: Tuple[RegisterTomographyBudget, ...],
    *,
    seed: RngSeed,
) -> Tuple[Tuple[RegisterTomographyCountRecord, ...], int, int]:
    """Sample each maximal joint-basis segment with one multinomial draw."""
    segments = _tomography_joint_segments(localized.n, localization, budgets)
    N_bp = sum(length for length, _axes, _active in segments)
    expected = max((budget.L_C for budget in budgets), default=0)
    if N_bp != expected:
        raise RuntimeError("Compact joint tomography schedule changed max_C L_C.")
    statistics = [
        {pauli: [0, 0] for pauli in _local_paulis(budget.k_C)}
        for budget in budgets
    ]
    rng = np.random.default_rng(seed)
    probability_cache: dict[Tuple[str, ...], Tuple[Tuple[Tuple[int, ...], ...], np.ndarray]] = {}
    for length, axes, active in segments:
        full_paulis = []
        for _index, budget, pauli in active:
            full = ["I"] * localized.n
            for local_index, character in enumerate(pauli):
                if character != "I":
                    full[localization.t + budget.J_C[local_index]] = character
            full_paulis.append("".join(full))
        query = tuple(full_paulis)
        if query not in probability_cache:
            probability_cache[query] = _commuting_tuple_probabilities_backend(
                localized, query
            )
        outcomes, probabilities = probability_cache[query]
        counts = rng.multinomial(length, probabilities)
        for position, (index, budget, pauli) in enumerate(active):
            statistics[index][pauli][0] += length
            statistics[index][pauli][1] += _signed_sum_from_joint_counts(
                counts, outcomes, position
            )
    records = tuple(
        RegisterTomographyCountRecord(
            budget.cluster,
            budget.J_C,
            tuple((pauli, budget.M_C_Pauli) for pauli in _local_paulis(budget.k_C)),
            tuple(
                (pauli, statistics[index][pauli][0], statistics[index][pauli][1])
                for pauli in _local_paulis(budget.k_C)
            ),
        )
        for index, budget in enumerate(budgets)
    )
    return records, N_bp, len(segments)


def _tomography_failure(
    localization: LocalizationResult,
    reason: str,
    prior: CopyLedger,
) -> TomographyResult:
    ledger = CopyLedger((("block_tomography_pool", 0),))
    return TomographyResult(
        False,
        reason,
        0,
        (),
        (),
        (),
        (),
        None,
        cumulative_copy_ledger=prior.with_entry("block_tomography_pool", 0),
        copy_ledger=ledger,
    )


def _fixed_budget_register_runs(
    localization: LocalizationResult,
    physical_rounds: int,
    *,
    seed: RngSeed,
) -> Tuple[Tuple[Tuple[str, int], ...], ...]:
    """Allocate deterministic, nearly balanced local Pauli runs.

    Every nonempty disjoint register receives ``physical_rounds`` simultaneous
    rounds.  When fewer than ``4**k-1`` rounds are available, the seed chooses
    a deterministic subset and unmeasured coefficients remain neutral (zero).
    """

    if isinstance(physical_rounds, bool) or int(physical_rounds) < 0:
        raise ValueError("physical_rounds must be a nonnegative integer.")
    rounds = int(physical_rounds)
    rng = np.random.default_rng(seed)
    schedules = []
    for _cluster, register in localization.J_C:
        paulis = _local_paulis(len(register))
        if not paulis or rounds == 0:
            schedules.append(())
            continue
        order = tuple(paulis[index] for index in rng.permutation(len(paulis)))
        quotient, remainder = divmod(rounds, len(order))
        schedules.append(
            tuple(
                (pauli, quotient + (index < remainder))
                for index, pauli in enumerate(order)
                if quotient + (index < remainder) > 0
            )
        )
    return tuple(schedules)


def _joint_fixed_budget_records_batched(
    localized: SimulatorMeasurementSource,
    localization: LocalizationResult,
    schedules: Tuple[Tuple[Tuple[str, int], ...], ...],
    *,
    seed: RngSeed,
) -> Tuple[Tuple[RegisterTomographyCountRecord, ...], int, int]:
    """Sample arbitrary compact local schedules with joint physical rounds."""

    positions = [0] * len(schedules)
    remaining = [runs[0][1] if runs else 0 for runs in schedules]
    statistics = [dict() for _runs in schedules]
    rng = np.random.default_rng(seed)
    probability_cache: dict[Tuple[str, ...], Tuple[Tuple[Tuple[int, ...], ...], np.ndarray]] = {}
    physical_rounds = 0
    work_units = 0
    while any(position < len(runs) for position, runs in zip(positions, schedules)):
        active_indices = [
            index
            for index, runs in enumerate(schedules)
            if positions[index] < len(runs)
        ]
        length = min(remaining[index] for index in active_indices)
        active = []
        full_paulis = []
        for index in active_indices:
            cluster, register = localization.J_C[index]
            pauli = schedules[index][positions[index]][0]
            active.append((index, tuple(cluster), tuple(register), pauli))
            full = ["I"] * localized.n
            for local_index, character in enumerate(pauli):
                if character != "I":
                    full[localization.t + register[local_index]] = character
            full_paulis.append("".join(full))
        query = tuple(full_paulis)
        if query not in probability_cache:
            probability_cache[query] = _commuting_tuple_probabilities_backend(
                localized, query
            )
        outcomes, probabilities = probability_cache[query]
        counts = rng.multinomial(length, probabilities)
        for position, (index, _cluster, register, pauli) in enumerate(active):
            old_shots, old_signed_sum = statistics[index].get(pauli, (0, 0))
            statistics[index][pauli] = (
                old_shots + length,
                old_signed_sum
                + _signed_sum_from_joint_counts(counts, outcomes, position),
            )
        physical_rounds += length
        work_units += 1
        for index in active_indices:
            remaining[index] -= length
            if remaining[index] == 0:
                positions[index] += 1
                if positions[index] < len(schedules[index]):
                    remaining[index] = schedules[index][positions[index]][1]
    expected = max((sum(count for _pauli, count in runs) for runs in schedules), default=0)
    if physical_rounds != expected:
        raise RuntimeError("Fixed-budget joint schedule changed max_C L_C.")
    records = tuple(
        RegisterTomographyCountRecord(
            tuple(cluster),
            tuple(register),
            schedules[index],
            tuple(
                (pauli, statistics[index][pauli][0], statistics[index][pauli][1])
                for pauli, _count in schedules[index]
            ),
        )
        for index, (cluster, register) in enumerate(localization.J_C)
    )
    return records, physical_rounds, work_units


@_time_pipeline_stage("tomography")
def tomograph_localized_registers_fixed_budget(
    source: Union[CEBPLearnerView, SimulatorMeasurementSource],
    peeling: PeelingResult,
    localization: LocalizationResult,
    physical_rounds: int,
    config: TomographyConfig,
    *,
    seed: RngSeed = None,
    prior_copy_ledger: Optional[CopyLedger] = None,
) -> TomographyResult:
    """Return physical local estimates using at most a supplied copy remainder.

    Unlike theorem tomography this API never converts ``epsilon_tom`` into a
    required batch.  Missing Pauli coefficients are set to zero before the
    existing Hilbert--Schmidt density projection.
    """

    if not isinstance(config, TomographyConfig):
        raise TypeError("config must be a TomographyConfig.")
    if not isinstance(localization, LocalizationResult):
        raise TypeError("localization must be a LocalizationResult.")
    rounds = int(physical_rounds)
    if isinstance(physical_rounds, bool) or rounds < 0:
        raise ValueError("physical_rounds must be a nonnegative integer.")
    prior = localization.cumulative_copy_ledger if prior_copy_ledger is None else prior_copy_ledger
    if "block_tomography_pool" in prior.as_dict():
        raise ValueError("prior ledger already contains block_tomography_pool.")
    if not peeling.success or not localization.success:
        return _tomography_failure(localization, "invalid_graceful_tomography_input", prior)
    if config.materialize_localized_estimator and localization.n > config.max_dense_qubits:
        return _tomography_failure(localization, "dense_resource_guard_exceeded", prior)
    try:
        schedules = _fixed_budget_register_runs(localization, rounds, seed=seed)
        localized = localized_measurement_source(source, peeling, localization)
        records, N_bp, work_units = _joint_fixed_budget_records_batched(
            localized, localization, schedules, seed=seed
        )
        estimates = []
        budgets = []
        measured_counts = []
        total_counts = []
        for (cluster, register), record, runs in zip(
            localization.J_C, records, schedules
        ):
            cluster = tuple(cluster)
            register = tuple(register)
            k_C = len(register)
            total = 4**k_C - 1
            stats = {
                pauli: (shots, signed_sum)
                for pauli, shots, signed_sum in record.sufficient_statistics
            }
            measured = len(stats)
            coefficients = {
                pauli: (
                    float(stats[pauli][1] / stats[pauli][0])
                    if pauli in stats
                    else 0.0
                )
                for pauli in _local_paulis(k_C)
            }
            linear = linear_inversion_from_pauli_coefficients(
                k_C, coefficients
            )
            projected, numerical_bound = _project_to_density_matrix_hs_with_error_bound(
                linear, tolerance=config.numerical_tolerance
            )
            estimates.append(
                RegisterTomographyEstimate(
                    cluster,
                    register,
                    k_C,
                    tuple(coefficients.items()),
                    linear,
                    projected,
                    projected,
                    numerical_bound,
                )
            )
            shot_counts = tuple(count for _pauli, count in runs)
            budgets.append(
                FixedBudgetRegisterTomographyBudget(
                    cluster=cluster,
                    J_C=register,
                    k_C=k_C,
                    total_nonidentity_paulis=total,
                    measured_pauli_count=measured,
                    physical_round_budget=rounds,
                    realized_schedule_length=sum(shot_counts),
                    min_shots_per_measured_pauli=min(shot_counts, default=0),
                    max_shots_per_measured_pauli=max(shot_counts, default=0),
                    complete_pauli_coverage=measured == total,
                    budget_truncated=measured < total,
                )
            )
            measured_counts.append((cluster, measured))
            total_counts.append((cluster, total))
        estimator = (
            assemble_localized_product_estimator(
                tuple((estimate.J_C, estimate.nu_hat) for estimate in estimates),
                localization.J_aux,
                localization.m,
            )
            if config.materialize_localized_estimator
            else None
        )
        truncated = any(budget.budget_truncated for budget in budgets)
        stage = CopyLedger((("block_tomography_pool", N_bp),))
        return TomographyResult(
            success=True,
            failure_reason=None,
            Khat=len(budgets),
            budgets=tuple(budgets),
            records=records,
            estimates=tuple(estimates),
            J_aux=localization.J_aux,
            localized_empirical_estimator=estimator,
            N_bp=N_bp,
            sum_local_schedule_lengths=sum(sum(count for _pauli, count in runs) for runs in schedules),
            one_common_copy_per_round=True,
            theorem_preconditions_hold=False,
            block_tomography_pool=N_bp,
            copy_ledger=stage,
            cumulative_copy_ledger=prior.with_entry("block_tomography_pool", N_bp),
            provenance=DataProvenance.EMPIRICAL,
            sampling_work_units=work_units,
            simulation_backend="batched_counts",
            measured_pauli_counts=tuple(measured_counts),
            total_pauli_counts=tuple(total_counts),
            budget_truncated=truncated,
        )
    except (ValueError, RuntimeError) as error:
        return _tomography_failure(
            localization, f"tomography_invariant_failed: {error}", prior
        )


@_time_pipeline_stage("tomography")
def tomograph_localized_registers(
    source: Union[CEBPLearnerView, SimulatorMeasurementSource],
    peeling: PeelingResult,
    localization: LocalizationResult,
    config: TomographyConfig,
    *,
    seed: RngSeed = None,
    prior_copy_ledger: Optional[CopyLedger] = None,
    simulation_backend: SimulationBackend = "legacy_shotwise",
) -> TomographyResult:
    """Tomograph disjoint localized registers using one common copy per round."""
    if not isinstance(config, TomographyConfig):
        raise TypeError("config must be a TomographyConfig.")
    if not isinstance(localization, LocalizationResult):
        raise TypeError("localization must be a LocalizationResult.")
    seed = _validate_seed(seed)
    simulation_backend = _validate_simulation_backend(simulation_backend)
    prior = (
        localization.cumulative_copy_ledger
        if prior_copy_ledger is None
        else prior_copy_ledger
    )
    if "block_tomography_pool" in prior.as_dict():
        raise ValueError("prior ledger already contains block_tomography_pool.")
    if not peeling.success:
        return _tomography_failure(localization, "peeling_failed", prior)
    if not localization.success:
        return _tomography_failure(localization, "localization_failed", prior)
    certified = bool(localization.theorem_localization_preconditions_hold)
    if not certified and not config.allow_uncertified_localization:
        return _tomography_failure(localization, "localization_not_theorem_certified", prior)
    if config.materialize_localized_estimator and localization.n > config.max_dense_qubits:
        return _tomography_failure(localization, "dense_resource_guard_exceeded", prior)
    if simulation_backend == "batched_counts" and config.return_details:
        return _tomography_failure(
            localization,
            "batched_counts does not provide raw per-shot tomography transcript; "
            "use legacy_shotwise for exact raw details",
            prior,
        )
    try:
        budgets = _tomography_budgets(localization, config)
        localized = localized_measurement_source(source, peeling, localization)
        if simulation_backend == "legacy_shotwise":
            records, N_bp, transcript = _joint_disjoint_register_records(
                localized,
                localization,
                budgets,
                seed=seed,
                return_details=config.return_details,
            )
            sampling_work_units = N_bp
        else:
            records, N_bp, sampling_work_units = _joint_disjoint_register_records_batched(
                localized, localization, budgets, seed=seed
            )
            transcript = ()
        estimates = []
        for budget, record in zip(budgets, records):
            if isinstance(record, RegisterTomographyRecord):
                outcome_map = dict(record.outcomes_by_pauli)
                if any(len(values) != budget.M_C_Pauli for values in outcome_map.values()):
                    raise RuntimeError("A local Pauli did not receive its exact shot budget.")
                coefficients = tuple(
                    (pauli, float(np.mean(outcome_map[pauli])))
                    for pauli in _local_paulis(budget.k_C)
                )
            else:
                stats = {pauli: (shots, signed_sum) for pauli, shots, signed_sum in record.sufficient_statistics}
                if any(stats[pauli][0] != budget.M_C_Pauli for pauli in _local_paulis(budget.k_C)):
                    raise RuntimeError("A local Pauli did not receive its exact shot budget.")
                coefficients = tuple(
                    (pauli, float(stats[pauli][1] / stats[pauli][0]))
                    for pauli in _local_paulis(budget.k_C)
                )
            linear = linear_inversion_from_pauli_coefficients(
                budget.k_C, dict(coefficients)
            )
            projected, numerical_bound = _project_to_density_matrix_hs_with_error_bound(
                linear, tolerance=config.numerical_tolerance
            )
            if numerical_bound > budget.epsilon_C / 2.0:
                raise ValueError(
                    "Numerical projection allowance exceeds epsilon_C/2."
                )
            estimates.append(
                RegisterTomographyEstimate(
                    budget.cluster,
                    budget.J_C,
                    budget.k_C,
                    coefficients,
                    linear,
                    projected,
                    projected,
                    numerical_bound,
                )
            )
        estimator = (
            assemble_localized_product_estimator(
                tuple((estimate.J_C, estimate.nu_hat) for estimate in estimates),
                localization.J_aux,
                localization.m,
            )
            if config.materialize_localized_estimator
            else None
        )
        stage = CopyLedger((("block_tomography_pool", N_bp),))
        return TomographyResult(
            True,
            None,
            len(budgets),
            budgets,
            records,
            tuple(estimates),
            localization.J_aux,
            estimator,
            N_bp,
            sum(budget.L_C for budget in budgets),
            True,
            bool(certified),
            N_bp,
            stage,
            prior.with_entry("block_tomography_pool", N_bp),
            DataProvenance.EMPIRICAL,
            transcript,
            sampling_work_units,
            simulation_backend,
        )
    except (ValueError, RuntimeError) as error:
        return _tomography_failure(
            localization, f"tomography_invariant_failed: {error}", prior
        )


def recovered_block_tomography(
    source: Union[CEBPLearnerView, SimulatorMeasurementSource],
    peeling: PeelingResult,
    localization: LocalizationResult,
    config: RecoveredBlockTomographyConfig,
    *,
    syndrome_seed: RngSeed = None,
    tomography_seed: RngSeed = None,
) -> RecoveredBlockTomographyResult:
    """Run Phase 6 only and return a compact localized-coordinate estimate."""
    if not isinstance(config, RecoveredBlockTomographyConfig):
        raise TypeError("config must be a RecoveredBlockTomographyConfig.")
    upstream_reason = None
    if not localization.success:
        upstream_reason = "localization_failed"
    elif (
        not localization.theorem_localization_preconditions_hold
        and not config.tomography.allow_uncertified_localization
    ):
        upstream_reason = "localization_not_theorem_certified"
    if upstream_reason is not None:
        syndrome = _syndrome_failure(
            peeling,
            config.syndrome,
            upstream_reason,
            localization.cumulative_copy_ledger,
        )
        tomography = _tomography_failure(
            localization, upstream_reason, syndrome.cumulative_copy_ledger
        )
        return RecoveredBlockTomographyResult(
            False,
            upstream_reason,
            syndrome,
            tomography,
            (),
            (),
            (),
            tomography.cumulative_copy_ledger,
            False,
        )
    syndrome = recover_peeling_syndrome(
        source,
        peeling,
        config.syndrome,
        seed=syndrome_seed,
        prior_copy_ledger=localization.cumulative_copy_ledger,
    )
    if not syndrome.success:
        tomography = _tomography_failure(
            localization, "syndrome_failed", syndrome.cumulative_copy_ledger
        )
    else:
        tomography = tomograph_localized_registers(
            source,
            peeling,
            localization,
            config.tomography,
            seed=tomography_seed,
            prior_copy_ledger=syndrome.cumulative_copy_ledger,
        )
    success = syndrome.success and tomography.success
    reason = None if success else tomography.failure_reason or syndrome.failure_reason
    compact = tuple(
        (estimate.cluster, estimate.J_C, estimate.nu_hat)
        for estimate in tomography.estimates
    )
    return RecoveredBlockTomographyResult(
        success,
        reason,
        syndrome,
        tomography,
        syndrome.syndrome_bits,
        compact,
        localization.J_aux if localization.success else (),
        tomography.cumulative_copy_ledger,
        bool(
            success
            and syndrome.theorem_preconditions_hold
            and tomography.theorem_preconditions_hold
        ),
    )


def _is_physical_density_matrix(
    state: qt.Qobj, qubits: int, *, tolerance: float = 1e-9
) -> bool:
    """Return whether ``state`` is a physical ``qubits``-qubit density matrix."""
    if not isinstance(state, qt.Qobj) or state.isket:
        return False
    dimension = 2**qubits
    if state.shape != (dimension, dimension):
        return False
    array = np.asarray(state.full(), dtype=complex)
    if not np.all(np.isfinite(array)):
        return False
    if not np.allclose(array, array.conj().T, atol=tolerance, rtol=0.0):
        return False
    if abs(complex(np.trace(array)) - 1.0) > tolerance:
        return False
    return bool(np.min(np.linalg.eigvalsh(0.5 * (array + array.conj().T))) >= -tolerance)


def validate_recovered_block_tomography_handoff(
    peeling: PeelingResult,
    localization: LocalizationResult,
    phase6_result: RecoveredBlockTomographyResult,
    *,
    tolerance: float = 1e-9,
) -> bool:
    """Strictly validate the learned Phase-6 transcript passed to Phase 7.

    The validator deliberately consumes only learner-visible stage outputs.
    Malformed cross-stage relations raise :class:`ValueError`; a valid
    theorem-facing handoff returns ``True``.
    """
    if not isinstance(peeling, PeelingResult):
        raise TypeError("peeling must be a PeelingResult.")
    if not isinstance(localization, LocalizationResult):
        raise TypeError("localization must be a LocalizationResult.")
    if not isinstance(phase6_result, RecoveredBlockTomographyResult):
        raise TypeError("phase6_result must be a RecoveredBlockTomographyResult.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    syndrome = phase6_result.syndrome
    tomography = phase6_result.tomography
    if not (peeling.success and localization.success and phase6_result.success):
        raise ValueError("Every stage in a Phase-7 handoff must be successful.")
    if not (syndrome.success and tomography.success):
        raise ValueError("Phase-6 syndrome and tomography must both succeed.")
    n = _peeling_dimension(peeling)
    t = int(peeling.t if peeling.t is not None else -1)
    if localization.n != n or localization.t != t or localization.m != n - t:
        raise ValueError("Peeling/localization dimensions do not agree.")
    if syndrome.t != t or len(syndrome.syndrome_bits) != t:
        raise ValueError("Syndrome dimension does not match the peeled prefix.")
    if any(bit not in (0, 1) for bit in syndrome.syndrome_bits):
        raise ValueError("Syndrome bits must be binary.")
    if phase6_result.syndrome_bits != syndrome.syndrome_bits:
        raise ValueError("The compact syndrome differs from SyndromeResult.")

    expected_registers = {
        (tuple(cluster), tuple(register)) for cluster, register in localization.J_C
    }
    Khat = len(expected_registers)
    if Khat != len(localization.J_C):
        raise ValueError("Localization contains a duplicate cluster/register.")
    if tomography.Khat != Khat:
        raise ValueError("Tomography Khat does not match localization.")
    collections = (tomography.budgets, tomography.records, tomography.estimates)
    if any(len(values) != Khat for values in collections):
        raise ValueError("Tomography budget/record/estimate counts do not match Khat.")

    def register_keys(values: Sequence[object]) -> Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], ...]:
        return tuple((tuple(value.cluster), tuple(value.J_C)) for value in values)

    for values in collections:
        keys = register_keys(values)
        if len(set(keys)) != len(keys) or set(keys) != expected_registers:
            raise ValueError("Tomography registers do not match localization exactly.")
    compact_keys = tuple(
        (tuple(cluster), tuple(register))
        for cluster, register, _state in phase6_result.register_estimates
    )
    if len(set(compact_keys)) != len(compact_keys) or set(compact_keys) != expected_registers:
        raise ValueError("Compact register estimates are missing or duplicated.")
    estimates_by_key = {
        (tuple(estimate.cluster), tuple(estimate.J_C)): estimate.nu_hat
        for estimate in tomography.estimates
    }
    for cluster, register, state in phase6_result.register_estimates:
        expected_state = estimates_by_key[(tuple(cluster), tuple(register))]
        if not isinstance(state, qt.Qobj) or not np.allclose(
            state.full(), expected_state.full(), atol=tolerance, rtol=0.0
        ):
            raise ValueError("Compact and tomography register estimates differ.")
    if tuple(phase6_result.J_aux) != tuple(localization.J_aux):
        raise ValueError("Auxiliary register does not match localization.")
    if tuple(tomography.J_aux) != tuple(localization.J_aux):
        raise ValueError("Tomography auxiliary register does not match localization.")

    for budget, record, estimate in zip(
        tomography.budgets, tomography.records, tomography.estimates
    ):
        if budget.k_C != len(budget.J_C) or budget.N_C_Pauli != 4**budget.k_C - 1:
            raise ValueError("A tomography budget has an invalid Pauli count.")
        if budget.L_C != budget.N_C_Pauli * budget.M_C_Pauli:
            raise ValueError("A tomography budget has an invalid schedule length.")
        expected_paulis = set(_local_paulis(budget.k_C))
        if isinstance(record, RegisterTomographyRecord):
            outcomes = record.outcomes_by_pauli
            keys = tuple(pauli for pauli, _values in outcomes)
            if len(keys) != len(set(keys)) or set(keys) != expected_paulis:
                raise ValueError("A tomography record does not contain every Pauli once.")
            if set(record.schedule) != expected_paulis or len(record.schedule) != budget.L_C:
                raise ValueError("A tomography record schedule is malformed.")
            for _pauli, values in outcomes:
                if len(values) != budget.M_C_Pauli or any(value not in (-1, 1) for value in values):
                    raise ValueError("A tomography outcome batch has the wrong shots/signs.")
        elif isinstance(record, RegisterTomographyCountRecord):
            keys = tuple(pauli for pauli, _shots, _signed_sum in record.sufficient_statistics)
            if len(keys) != len(set(keys)) or set(keys) != expected_paulis:
                raise ValueError("A compressed tomography record is missing a Pauli.")
            if set(pauli for pauli, _length in record.schedule_runs) != expected_paulis:
                raise ValueError("A compressed tomography schedule is malformed.")
            if sum(length for _pauli, length in record.schedule_runs) != budget.L_C:
                raise ValueError("A compressed tomography schedule has the wrong length.")
            if any(shots != budget.M_C_Pauli for _pauli, shots, _sum in record.sufficient_statistics):
                raise ValueError("A compressed tomography statistic has the wrong shots.")
        else:
            raise TypeError("Unknown tomography record representation.")
        if estimate.k_C != budget.k_C or not _is_physical_density_matrix(
            estimate.nu_hat, budget.k_C, tolerance=tolerance
        ):
            raise ValueError("A register estimate is not a physical density matrix.")
        bound = float(estimate.numerical_projection_error_bound)
        if not np.isfinite(bound) or bound < 0.0 or bound > budget.epsilon_C / 2.0:
            raise ValueError("A numerical projection bound is not theorem-safe.")
    expected_N_bp = max((budget.L_C for budget in tomography.budgets), default=0)
    if tomography.N_bp != expected_N_bp or tomography.block_tomography_pool != expected_N_bp:
        raise ValueError("Tomography N_bp/copy pool does not equal max_C L_C.")
    if tomography.copy_ledger.as_dict() != {"block_tomography_pool": expected_N_bp}:
        raise ValueError("Tomography stage ledger is malformed.")
    if t == 0:
        if syndrome.M_sgn != 0 or syndrome.syndrome_sign_pool != 0:
            raise ValueError("The empty syndrome must consume zero copies.")
    elif syndrome.syndrome_sign_pool != t * syndrome.M_sgn:
        raise ValueError("Syndrome copy pool does not equal t*M_sgn.")
    if syndrome.copy_ledger.as_dict() != {
        "syndrome_sign_pool": syndrome.syndrome_sign_pool
    }:
        raise ValueError("Syndrome stage ledger is malformed.")
    if (
        tomography.localized_empirical_estimator is not None
        and not _is_physical_density_matrix(
            tomography.localized_empirical_estimator,
            localization.m,
            tolerance=tolerance,
        )
    ):
        raise ValueError("The optional localized residual estimator is not physical.")

    expected_pool_names = (
        "peeling_bell_pool",
        "recovery_bell_pool",
        "grouping_ordinary_pool",
        "syndrome_sign_pool",
        "block_tomography_pool",
    )
    ledger = phase6_result.cumulative_copy_ledger
    if tuple(name for name, _copies in ledger.entries) != expected_pool_names:
        raise ValueError("The cumulative Phase-6 ledger has wrong/missing pools.")
    if tomography.cumulative_copy_ledger != ledger:
        raise ValueError("Tomography and Phase-6 cumulative ledgers differ.")
    values = ledger.as_dict()
    if values["syndrome_sign_pool"] != syndrome.syndrome_sign_pool:
        raise ValueError("Cumulative syndrome copies disagree with SyndromeResult.")
    if values["block_tomography_pool"] != expected_N_bp:
        raise ValueError("Cumulative tomography copies disagree with N_bp.")
    if ledger.total != sum(values[name] for name in expected_pool_names):
        raise ValueError("Cumulative Phase-6 ledger total is inconsistent.")
    return True


def _stage_seed_ledger(seed: RngSeed) -> StageSeedLedger:
    """Split the master seed in the fixed peel/sign/rank/group/tom order."""
    master = _validate_seed(seed)
    children = _v1._child_seeds(master, 5)
    return StageSeedLedger(master, *children)


def _ceil_positive(value: float, name: str) -> int:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} is not a finite positive sample count.")
    return int(math.ceil(value))


def calibrated_end_to_end_schedule(
    n: int, d: int, epsilon: float, delta: float
) -> EndToEndSchedule:
    """Construct the exact data-independent Phase-7 schedule.

    The generic branch implements Eqs. supp-calibrated-known-bounds through
    supp-block-product-main-copy-count-worst-case.  The d=1 branch uses the
    separate four-event calibration in Corollary
    supp-single-qubit-hidden-block-tomography.  Its fixed attempted-copy
    postselection count is a conservative explicit realization of the
    manuscript's stated ``O(n^2 epsilon^-2 log(n/delta))`` schedule.
    """
    n = _validate_positive_integer("n", n)
    d = _validate_positive_integer("d", d)
    if d > n:
        raise ValueError("d must not exceed n.")
    if not (0.0 < float(epsilon) < 1.0):
        raise ValueError("epsilon must lie in (0,1).")
    if not (0.0 < float(delta) < 1.0):
        raise ValueError("delta must lie in (0,1).")
    epsilon = float(epsilon)
    delta = float(delta)
    epsilon_tom = epsilon / 4.0
    R_ub = n + 1
    A_d = n * 2**d
    Gamma_d = cumulant_gamma(d)
    if d == 1:
        branch = "d1_specialized"
        q_d = 0
        theta_0 = epsilon**2 / (128.0 * n**2)
        lambda_0 = theta_0 / 4.0
        eta_s = None
        eta_test = tau_kappa = tau_mu = 0.0
        ell_grp = 1
        zeta_peel = zeta_rank = zeta_sgn = zeta_tom = delta / 4.0
        zeta_grp = 0.0
    else:
        branch = "generic_d_ge_2"
        q_d = 2**d - d - 1
        theta_0 = epsilon**2 / (128.0 * R_ub**2 * A_d**2)
        eta_s = math.expm1(
            math.log1p(epsilon / (8.0 * R_ub * A_d)) / q_d
        )
        lambda_0 = min(
            theta_0 / 4.0,
            epsilon**2 / (128.0 * n * R_ub**2),
            eta_s**2 / (512.0 * n * Gamma_d**2),
        )
        eta_test = eta_s / 2.0
        tau_kappa = eta_s / 4.0
        tau_mu = eta_s / (4.0 * Gamma_d)
        ell_grp = d
        zeta_peel = zeta_rank = zeta_grp = zeta_sgn = zeta_tom = delta / 5.0
    theta = theta_0
    h_min = 1.0 - lambda_0
    h_max = 1.0 - lambda_0 / 2.0
    peeling_eta = lambda_0 / (8.0 * (2 * n + 1))
    log_paulis_peel = math.log(2.0 * 4**n / zeta_peel)
    M1 = _ceil_positive(
        512.0 * (2 * n + 1) ** 2 / lambda_0**2 * log_paulis_peel,
        "M1",
    )
    M2 = _ceil_positive(
        128.0 / (lambda_0**2 * theta_0**2)
        * math.log(2.0 * 4**n / zeta_rank),
        "M2",
    )
    tau_1 = bell_score_uniform_radius(M1, n, zeta_peel)
    tau_rank = bell_score_uniform_radius(M2, n, zeta_rank)
    M_sgn = syndrome_sign_sample_count(n, h_min, zeta_sgn)
    if d == 1:
        N_test_max_wc = grouping_per_query = N_grp_wc = 0
        N_P_wc = 3
        tau_bp_wc = epsilon_tom / (2.0 * math.sqrt(3.0) * n)
        accepted = _ceil_positive(
            2.0 / tau_bp_wc**2 * math.log(6.0 * n / zeta_tom),
            "d1 accepted shots",
        )
        attempts = _ceil_positive(
            4.0 * (accepted + math.log(3.0 / zeta_tom)),
            "d1 attempted shots",
        )
        M_P_wc = accepted
        N_bp_wc = 3 * attempts
        d1_accepted_per_setting = accepted
        d1_attempts_per_setting = attempts
    else:
        N_test_max_wc = int(
            n * 4**d * sum(math.comb(n, q) for q in range(2, d + 1))
        )
        if N_test_max_wc:
            grouping_per_query = _ceil_positive(
                32.0 * Gamma_d**2 / eta_s**2
                * math.log(
                    2.0 * (2**d - 1) * N_test_max_wc / zeta_grp
                ),
                "grouping per-query copies",
            )
        else:
            grouping_per_query = 0
        N_grp_wc = N_test_max_wc * grouping_per_query
        N_P_wc = 4**d - 1
        tau_bp_wc = epsilon_tom / (4.0 * n * math.sqrt(N_P_wc))
        M_P_wc = _ceil_positive(
            2.0 / tau_bp_wc**2
            * math.log(2.0 * n * N_P_wc / zeta_tom),
            "worst-case tomography shots",
        )
        N_bp_wc = N_P_wc * M_P_wc
        d1_accepted_per_setting = d1_attempts_per_setting = 0
    reservation = WorstCaseCopyReservation(
        2 * M1,
        2 * M2,
        N_grp_wc,
        n * M_sgn,
        N_bp_wc,
        2 * M1 + 2 * M2 + N_grp_wc + n * M_sgn + N_bp_wc,
    )
    return EndToEndSchedule(
        branch,
        n,
        d,
        epsilon,
        delta,
        R_ub,
        A_d,
        q_d,
        Gamma_d,
        epsilon_tom,
        theta_0,
        theta,
        eta_s,
        lambda_0,
        h_min,
        h_max,
        peeling_eta,
        eta_test,
        tau_kappa,
        tau_mu,
        ell_grp,
        zeta_peel,
        zeta_rank,
        zeta_grp,
        zeta_sgn,
        zeta_tom,
        M1,
        tau_1,
        M2,
        tau_rank,
        M_sgn,
        N_test_max_wc,
        grouping_per_query,
        N_grp_wc,
        N_P_wc,
        tau_bp_wc,
        M_P_wc,
        N_bp_wc,
        d1_accepted_per_setting,
        d1_attempts_per_setting,
        reservation,
    )


def calibrated_asymptotic_summary(d: int) -> Mapping[str, str]:
    """Return the manuscript scaling audit, never a runtime allocator."""
    d = _validate_positive_integer("d", d)
    if d == 1:
        values = {
            "M1": "O_tilde(n^7 epsilon^-4)",
            "M2": "O_tilde(n^9 epsilon^-8)",
            "conditional_tomography": "O_tilde(n^2 epsilon^-2)",
            "grouping": "0",
            "total": "O_tilde(n^9 epsilon^-8)",
        }
    else:
        values = {
            "ordinary_access": (
                "O_tilde(n^3 Gamma_d^4 R_ub^8 A_d^8 q_d^4 epsilon^-8 "
                "+ d^2 R_ub^2 n^(d+3) 4^d q_d^2 epsilon^-2 "
                "(4d^2/(e^2(log 2)^2))^d + n^2 16^d epsilon^-2 + n)"
            ),
            "total": (
                "O_tilde(2^(O(d log d)) "
                "(n^19 epsilon^-8 + n^(d+5) epsilon^-2))"
            ),
        }
    return MappingProxyType(values)


def compute_structural_certificate(
    schedule: EndToEndSchedule,
    peeling: PeelingResult,
    recovery: RecoveryResult,
    grouping: GroupingResult,
    localization: LocalizationResult,
) -> StructuralCertificate:
    """Compute Eq. supp-localization-structural-certificate from learned data."""
    if not all(
        isinstance(value, expected)
        for value, expected in (
            (schedule, EndToEndSchedule),
            (peeling, PeelingResult),
            (recovery, RecoveryResult),
            (grouping, GroupingResult),
            (localization, LocalizationResult),
        )
    ):
        raise TypeError("Structural certificate inputs have the wrong types.")
    if not (peeling.success and recovery.success and grouping.success and localization.success):
        raise ValueError("Structural certificate requires successful learned stages.")
    epsilon_peel = float(peeling.epsilon_peel)
    theta_rec = float(schedule.theta + recovery.tau_rank)
    Khat = len(localization.J_C)
    if epsilon_peel >= 1.0:
        return StructuralCertificate(
            "trivial_fallback",
            Khat,
            epsilon_peel,
            theta_rec,
            2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            (),
            0.0,
            2.0,
        )
    E_peel = 2.0 * math.sqrt(epsilon_peel)
    if schedule.branch == "d1_specialized":
        E_miss = schedule.n * math.sqrt(3.0 * theta_rec)
        return StructuralCertificate(
            "d1_specialized",
            Khat,
            epsilon_peel,
            theta_rec,
            E_peel,
            E_miss,
            0.0,
            0.0,
            0.0,
            0.0,
            (),
            0.0,
            E_peel + E_miss,
        )
    E_miss = schedule.n * 2**schedule.d * math.sqrt(theta_rec)
    exact_branch = grouping.moment_provenance is DataProvenance.EXACT
    if exact_branch:
        return StructuralCertificate(
            "exact_grouping",
            Khat,
            epsilon_peel,
            theta_rec,
            E_peel,
            E_miss,
            grouping.beta_peel,
            0.0,
            grouping.beta_peel,
            0.0,
            (),
            0.0,
            E_peel + E_miss,
        )
    beta_peel = float(grouping.beta_peel)
    xi_s = float(
        grouping.xi_s
        if grouping.xi_s is not None
        else 3.0 * float(schedule.eta_s) / 4.0
    )
    xi_eff = xi_s + beta_peel
    F_value = math.expm1(schedule.q_d * math.log1p(xi_eff))
    split_terms = tuple(
        (
            tuple(cluster),
            len(register),
            F_value
            * math.sqrt(max(0.0, float(4**schedule.d - 4 ** len(register)))),
        )
        for cluster, register in localization.J_C
    )
    E_split = sum(value for _cluster, _k, value in split_terms)
    return StructuralCertificate(
        "guessed_scale",
        Khat,
        epsilon_peel,
        theta_rec,
        E_peel,
        E_miss,
        beta_peel,
        xi_s,
        xi_eff,
        F_value,
        split_terms,
        E_split,
        E_peel + E_miss + E_split,
    )


def _localized_estimator_from_compact(estimator: CompactCEBPEstimator) -> qt.Qobj:
    residual = assemble_localized_product_estimator(
        tuple((register, state) for _cluster, register, state in estimator.register_estimates),
        estimator.J_aux,
        estimator.m,
    )
    if estimator.t:
        prefix_ket = qt.tensor(
            tuple(qt.basis(2, bit) for bit in estimator.syndrome_bits)
        )
        prefix = qt.ket2dm(prefix_ket)
        prefix_array = prefix.full()
    else:
        prefix_array = np.ones((1, 1), dtype=complex)
    residual_array = residual.full() if estimator.m else np.ones((1, 1), dtype=complex)
    localized = np.kron(prefix_array, residual_array)
    return qt.Qobj(localized, dims=[[2] * estimator.n, [2] * estimator.n])


def materialize_compact_cebp_estimator(
    estimator: CompactCEBPEstimator, *, max_dense_qubits: int = 8
) -> qt.Qobj:
    """Materialize and decode a compact estimator for a guarded small system."""
    if not isinstance(estimator, CompactCEBPEstimator):
        raise TypeError("estimator must be a CompactCEBPEstimator.")
    max_dense_qubits = _validate_positive_integer(
        "max_dense_qubits", max_dense_qubits
    )
    if estimator.n > max_dense_qubits:
        raise ValueError("dense estimator resource guard exceeded.")
    localized = _localized_estimator_from_compact(estimator)
    U_stab = estimator.U_stab
    if U_stab is None:
        assert estimator.peeling_clifford is not None
        U_stab = estimator.peeling_clifford.materialize_dense_debug(
            max_qubits=max_dense_qubits
        )
    bar_U_rec = estimator.bar_U_rec
    if bar_U_rec is None:
        assert estimator.recovery_clifford is not None
        bar_U_rec = estimator.recovery_clifford.materialize_dense_debug(
            max_qubits=max_dense_qubits
        )
    decoder = U_stab @ bar_U_rec
    decoded = decoder @ localized.full() @ decoder.conj().T
    decoded = 0.5 * (decoded + decoded.conj().T)
    return qt.Qobj(decoded, dims=[[2] * estimator.n, [2] * estimator.n])


def _compact_estimator(
    peeling: PeelingResult,
    localization: LocalizationResult,
    syndrome: SyndromeResult,
    estimates: Sequence[RegisterTomographyEstimate],
) -> CompactCEBPEstimator:
    return CompactCEBPEstimator(
        localization.n,
        localization.t,
        localization.m,
        (
            None
            if peeling.U_stab is None
            else np.asarray(peeling.U_stab, dtype=complex)
        ),
        (
            None
            if localization.bar_U_rec is None
            else np.asarray(localization.bar_U_rec, dtype=complex)
        ),
        tuple(syndrome.syndrome_bits),
        tuple(
            (tuple(estimate.cluster), tuple(estimate.J_C), estimate.nu_hat)
            for estimate in estimates
        ),
        tuple(localization.J_aux),
        tuple(peeling.gates),
        tuple(localization.gates),
        peeling.signed_clifford,
        localization.signed_clifford,
    )


def _copy_reservation_comparison(
    realized: CopyLedger, reservation: WorstCaseCopyReservation, *, d1: bool
) -> bool:
    reserved = reservation.as_dict()
    values = realized.as_dict()
    aliases = {
        "conditional_one_qubit_pool": "block_tomography_pool",
    }
    for name, copies in values.items():
        target = aliases.get(name, name)
        if target not in reserved or copies > reserved[target]:
            return False
    if d1 and values.get("grouping_ordinary_pool", 0) != 0:
        return False
    return realized.total <= reservation.total


def build_end_to_end_certificate(
    schedule: EndToEndSchedule,
    structural: StructuralCertificate,
    *,
    theorem_preconditions: Mapping[str, bool],
    realized_copy_ledger: CopyLedger,
    operational_success: bool,
) -> EndToEndCertificate:
    """Build the manuscript conditional trace-norm and union-bound certificate."""
    if schedule.branch == "d1_specialized":
        error_bound = structural.E_struct_cert + schedule.epsilon_tom
        failures = (
            ("peeling", schedule.zeta_peel),
            ("rank", schedule.zeta_rank),
            ("syndrome", schedule.zeta_sgn),
            ("tomography", schedule.zeta_tom),
        )
    else:
        error_bound = (structural.Khat + 1) * structural.E_struct_cert + schedule.epsilon_tom
        failures = (
            ("peeling", schedule.zeta_peel),
            ("rank", schedule.zeta_rank),
            ("grouping", schedule.zeta_grp),
            ("syndrome", schedule.zeta_sgn),
            ("tomography", schedule.zeta_tom),
        )
    total_failure = float(sum(value for _name, value in failures))
    reservation_valid = _copy_reservation_comparison(
        realized_copy_ledger,
        schedule.reservation,
        d1=schedule.branch == "d1_specialized",
    )
    preconditions = tuple((str(name), bool(value)) for name, value in theorem_preconditions.items())
    machine_slack = 64.0 * np.finfo(float).eps * max(1.0, schedule.epsilon)
    certified = bool(
        operational_success
        and all(value for _name, value in preconditions)
        and reservation_valid
        and error_bound <= schedule.epsilon + machine_slack
        and total_failure <= schedule.delta + machine_slack
    )
    return EndToEndCertificate(
        structural,
        schedule.epsilon_tom,
        float(error_bound),
        schedule.epsilon,
        failures,
        total_failure,
        schedule.delta,
        preconditions,
        reservation_valid,
        reservation_valid,
        certified,
    )


@_time_pipeline_stage("tomography")
def _conditional_one_qubit_tomography(
    source: Union[CEBPLearnerView, SimulatorMeasurementSource],
    peeling: PeelingResult,
    localization: LocalizationResult,
    syndrome: SyndromeResult,
    schedule: EndToEndSchedule,
    *,
    seed: RngSeed,
    max_dense_qubits: int,
    materialize_localized_estimator: bool = True,
    target_accepted_override: Optional[int] = None,
    attempts_override: Optional[int] = None,
    budget_native: bool = False,
) -> ConditionalOneQubitTomographyResult:
    """Three-setting d=1 tomography conditioned on the learned syndrome.

    ``budget_native`` consumes all accepted samples within the supplied attempt
    cap and returns a projected estimate even when an axis has no accepted
    samples.  The default theorem-oriented path retains its accepted-shot target.
    """
    if materialize_localized_estimator and localization.n > max_dense_qubits:
        return ConditionalOneQubitTomographyResult(
            False,
            "dense_resource_guard_exceeded",
            (),
            (),
            None,
        )
    registers = tuple((tuple(cluster), tuple(register)) for cluster, register in localization.J_C)
    if any(len(register) != 1 for _cluster, register in registers):
        return ConditionalOneQubitTomographyResult(
            False, "d1_register_not_single_qubit", (), (), None
        )
    if not registers:
        estimator = (
            assemble_localized_product_estimator((), localization.J_aux, localization.m)
            if materialize_localized_estimator
            else None
        )
        return ConditionalOneQubitTomographyResult(
            True,
            None,
            (),
            localization.J_aux,
            estimator,
            (0, 0, 0),
            0,
            0,
            0,
            CopyLedger((("conditional_one_qubit_pool", 0),)),
            bool(
                not budget_native
                and localization.theorem_localization_preconditions_hold
            ),
        )
    localized = localized_measurement_source(source, peeling, localization)
    rng = np.random.default_rng(seed)
    target = (
        schedule.d1_accepted_per_setting
        if target_accepted_override is None
        else int(target_accepted_override)
    )
    attempts = (
        schedule.d1_attempts_per_setting
        if attempts_override is None
        else int(attempts_override)
    )
    accepted_counts = []
    values = {
        (cluster, register): {axis: [] for axis in "XYZ"}
        for cluster, register in registers
    }
    signed_sums = {
        (cluster, register): {axis: 0 for axis in "XYZ"}
        for cluster, register in registers
    }
    expected_prefix = tuple(
        1 if bit == 0 else -1 for bit in syndrome.syndrome_bits
    )
    for axis in "XYZ":
        query = tuple(
            "I" * qubit + "Z" + "I" * (localization.n - qubit - 1)
            for qubit in range(localization.t)
        ) + tuple(
            "I" * (localization.t + register[0])
            + axis
            + "I" * (localization.n - localization.t - register[0] - 1)
            for _cluster, register in registers
        )
        outcome_values, probabilities = _commuting_tuple_probabilities_backend(
            localized, query
        )
        accepted = 0
        if budget_native:
            outcome_counts = rng.multinomial(attempts, probabilities)
            for outcome, count in zip(outcome_values, outcome_counts):
                count = int(count)
                if count == 0 or tuple(outcome[: localization.t]) != expected_prefix:
                    continue
                accepted += count
                for register_index, (key, register) in enumerate(registers):
                    signed_sums[(key, register)][axis] += (
                        count * int(outcome[localization.t + register_index])
                    )
        else:
            for _attempt in range(attempts):
                sample = int(rng.choice(len(outcome_values), p=probabilities))
                outcome = outcome_values[sample]
                if tuple(outcome[: localization.t]) != expected_prefix:
                    continue
                if accepted < target:
                    for register_index, (key, register) in enumerate(registers):
                        values[(key, register)][axis].append(
                            int(outcome[localization.t + register_index])
                        )
                accepted += 1
        accepted_counts.append(accepted)
    attempted = 3 * attempts
    ledger = CopyLedger((("conditional_one_qubit_pool", attempted),))
    if not budget_native and any(count < target for count in accepted_counts):
        return ConditionalOneQubitTomographyResult(
            False,
            "insufficient_postselected_shots",
            (),
            localization.J_aux,
            None,
            tuple(accepted_counts),
            target,
            attempts,
            attempted,
            ledger,
            False,
        )
    estimates = []
    epsilon_C = schedule.epsilon_tom / len(registers)
    for cluster, register in registers:
        coefficients = tuple(
            (
                axis,
                float(
                    signed_sums[(cluster, register)][axis]
                    / accepted_counts[axis_index]
                )
                if budget_native and accepted_counts[axis_index] > 0
                else (
                    0.0
                    if budget_native
                    else float(np.mean(values[(cluster, register)][axis][:target]))
                ),
            )
            for axis_index, axis in enumerate("XYZ")
        )
        linear = linear_inversion_from_pauli_coefficients(1, dict(coefficients))
        projected, numerical_bound = _project_to_density_matrix_hs_with_error_bound(
            linear,
            tolerance=(1e-10 if budget_native else min(1e-10, epsilon_C / 4.0)),
        )
        if not budget_native and numerical_bound > epsilon_C / 2.0:
            return ConditionalOneQubitTomographyResult(
                False,
                "numerical_projection_allowance_exceeded",
                (),
                localization.J_aux,
                None,
                tuple(accepted_counts),
                target,
                attempts,
                attempted,
                ledger,
                False,
            )
        estimates.append(
            RegisterTomographyEstimate(
                cluster,
                register,
                1,
                coefficients,
                linear,
                projected,
                projected,
                numerical_bound,
            )
        )
    estimator = (
        assemble_localized_product_estimator(
            tuple((estimate.J_C, estimate.nu_hat) for estimate in estimates),
            localization.J_aux,
            localization.m,
        )
        if materialize_localized_estimator
        else None
    )
    return ConditionalOneQubitTomographyResult(
        True,
        None,
        tuple(estimates),
        localization.J_aux,
        estimator,
        tuple(accepted_counts),
        0 if budget_native else target,
        attempts,
        attempted,
        ledger,
        bool(
            not budget_native
            and localization.theorem_localization_preconditions_hold
        ),
    )


def _end_to_end_failure(
    schedule: EndToEndSchedule,
    seeds: StageSeedLedger,
    stage: str,
    reason: str,
    *,
    peeling: Optional[PeelingResult] = None,
    syndrome: Optional[SyndromeResult] = None,
    recovery: Optional[RecoveryResult] = None,
    grouping: Optional[GroupingResult] = None,
    localization: Optional[LocalizationResult] = None,
    tomography: Optional[Union[TomographyResult, ConditionalOneQubitTomographyResult]] = None,
    ledger: Optional[CopyLedger] = None,
) -> EndToEndResult:
    realized = CopyLedger() if ledger is None else ledger
    return EndToEndResult(
        success=False,
        failure_stage=stage,
        failure_reason=reason,
        branch=schedule.branch,
        n=schedule.n,
        d=schedule.d,
        epsilon=schedule.epsilon,
        delta=schedule.delta,
        schedule=schedule,
        seed_ledger=seeds,
        worst_case_reservation=schedule.reservation,
        peeling=peeling,
        syndrome=syndrome,
        recovery=recovery,
        grouping=grouping,
        localization=localization,
        tomography=tomography,
        realized_copy_ledger=realized,
        realized_total=realized.total,
        reservation_slack=schedule.reservation.total - realized.total,
        theorem_certified=False,
    )


def _execution_configs(
    schedule: EndToEndSchedule, config: EndToEndConfig
) -> Tuple[PeelingConfig, RecoveryConfig, GroupingConfig, SyndromeConfig, TomographyConfig]:
    peeling = config.peeling_override or PeelingConfig(
        h_min=schedule.h_min,
        h_max=schedule.h_max,
        eta=schedule.peeling_eta,
        M1=schedule.M1,
        zeta_bs=schedule.zeta_peel,
        return_details=config.return_details,
    )
    peeling = replace(
        peeling,
        max_enumeration_qubits=int(config.max_enumeration_qubits),
        enumeration_execution=config.enumeration_execution,
        materialize_dense_clifford=False,
        max_dense_debug_qubits=int(config.max_dense_debug_qubits),
    )
    recovery = config.recovery_override or RecoveryConfig(
        theta=schedule.theta,
        M2=schedule.M2,
        zeta_rank=schedule.zeta_rank,
        return_details=config.return_details,
    )
    recovery = replace(
        recovery,
        max_enumeration_qubits=int(config.max_enumeration_qubits),
        enumeration_execution=config.enumeration_execution,
    )
    if schedule.branch == "d1_specialized":
        grouping = config.grouping_override or GroupingConfig(
            ell_grp=1,
            eta_test=0.0,
            tau_kappa=0.0,
            delta_grp_ordinary=max(schedule.zeta_tom, np.finfo(float).eps),
            return_details=config.return_details,
            allow_uncalibrated_recovery=config.allow_uncertified_execution,
        )
    else:
        grouping = config.grouping_override or GroupingConfig.from_guessed_scale(
            schedule.d,
            float(schedule.eta_s),
            delta_grp_ordinary=schedule.zeta_grp,
            return_details=config.return_details,
        )
    syndrome = config.syndrome_override or SyndromeConfig(
        schedule.zeta_sgn,
        schedule.h_min,
        M_sgn=schedule.M_sgn,
        return_details=config.return_details,
    )
    tomography = config.tomography_override or TomographyConfig(
        schedule.epsilon_tom,
        schedule.zeta_tom,
        return_details=config.return_details,
        max_dense_qubits=config.max_dense_qubits,
    )
    tomography = replace(
        tomography,
        max_dense_qubits=int(config.max_dense_debug_qubits),
        materialize_localized_estimator=config.materialize_dense_estimator,
    )
    return peeling, recovery, grouping, syndrome, tomography


_FIXED_BUDGET_STAGES = ("peeling", "recovery", "grouping", "syndrome", "tomography")


def fixed_budget_nominal_stage_caps(
    total_budget: int, weights: FixedBudgetStageWeights
) -> Tuple[Tuple[str, int], ...]:
    """Normalize weights, floor early caps, and give integer remainder to tomography."""

    total = _validate_positive_integer("total_budget", total_budget)
    fractions = dict(weights.normalized())
    caps = []
    allocated = 0
    for stage in _FIXED_BUDGET_STAGES[:-1]:
        cap = int(math.floor(total * fractions[stage]))
        caps.append((stage, cap))
        allocated += cap
    caps.append(("tomography", total - allocated))
    if sum(cap for _stage, cap in caps) != total:
        raise RuntimeError("Fixed-budget integer caps do not sum to total_budget.")
    return tuple(caps)


def fixed_budget_resolved_stage_caps(
    config: EndToEndConfig,
) -> Tuple[Tuple[str, int], ...]:
    """Resolve immutable fixed-budget caps without cross-stage carry."""

    if not isinstance(config, EndToEndConfig):
        raise TypeError("config must be an EndToEndConfig.")
    if config.execution_policy is not ExecutionPolicy.FIXED_BUDGET_GRACEFUL:
        raise ValueError("Resolved fixed-budget caps require fixed_budget_graceful.")
    if config.max_realized_copies is None:
        raise RuntimeError("Fixed-budget execution is missing its total copy cap.")
    if config.fixed_budget_stage_caps is not None:
        return tuple(config.fixed_budget_stage_caps)
    if config.fixed_budget_stage_weights is None:
        raise RuntimeError("Fixed-budget execution is missing stage weights or caps.")
    return fixed_budget_nominal_stage_caps(
        int(config.max_realized_copies), config.fixed_budget_stage_weights
    )


def _complete_fixed_budget_stage_records(
    config: EndToEndConfig,
    records: Sequence[FixedBudgetStageRecord],
    realized_ledger: CopyLedger,
) -> Tuple[FixedBudgetStageRecord, ...]:
    """Return all immutable stage caps in stable order, including unreached stages."""

    total_budget = int(config.max_realized_copies)
    caps = dict(fixed_budget_resolved_stage_caps(config))
    by_stage = {record.stage: record for record in records}
    if len(by_stage) != len(records) or any(stage not in caps for stage in by_stage):
        raise RuntimeError("Fixed-budget stage records are duplicated or malformed.")
    upstream = records[-1].stage if records else "peeling"
    completed = tuple(
        by_stage.get(
            stage,
            FixedBudgetStageRecord(
                stage=stage,
                assigned_cap=caps[stage],
                realized_copies=0,
                unused_copies=caps[stage],
                budget_exhausted=False,
                stage_complete=False,
                degradation_reason=(
                    f"not_reached_due_to_upstream_failure:{upstream}"
                ),
            ),
        )
        for stage in _FIXED_BUDGET_STAGES
    )
    if sum(record.assigned_cap for record in completed) != total_budget:
        raise RuntimeError("Fixed-budget stage records do not sum to total budget.")
    if sum(record.realized_copies for record in completed) != realized_ledger.total:
        raise RuntimeError("Fixed-budget stage records disagree with the copy ledger.")
    return completed


def _graceful_mixed_fallback(
    schedule: EndToEndSchedule,
    seeds: StageSeedLedger,
    config: EndToEndConfig,
    ledger: CopyLedger,
    reason: str,
    truncated_stages: Sequence[str],
    stage_records: Sequence[FixedBudgetStageRecord],
    *,
    peeling: Optional[PeelingResult] = None,
    recovery: Optional[RecoveryResult] = None,
    grouping: Optional[GroupingResult] = None,
    localization: Optional[LocalizationResult] = None,
    syndrome: Optional[SyndromeResult] = None,
) -> EndToEndResult:
    """Return the conservative global maximally mixed estimator."""

    budget = int(config.max_realized_copies)
    if ledger.total > budget:
        raise RuntimeError("Graceful fallback exceeded its physical-copy budget.")
    complete_stage_records = _complete_fixed_budget_stage_records(
        config, stage_records, ledger
    )
    identity_clifford = SignedClifford.identity(schedule.n)
    compact = CompactCEBPEstimator(
        n=schedule.n,
        t=0,
        m=schedule.n,
        U_stab=None,
        bar_U_rec=None,
        syndrome_bits=(),
        register_estimates=(),
        J_aux=tuple(range(schedule.n)),
        peeling_clifford=identity_clifford,
        recovery_clifford=identity_clifford,
    )
    density = (
        materialize_compact_cebp_estimator(
            compact, max_dense_qubits=int(config.max_dense_debug_qubits)
        )
        if config.materialize_dense_estimator
        else None
    )
    return EndToEndResult(
        success=True,
        failure_stage=None,
        failure_reason=None,
        branch=schedule.branch,
        n=schedule.n,
        d=schedule.d,
        epsilon=schedule.epsilon,
        delta=schedule.delta,
        schedule=schedule,
        seed_ledger=seeds,
        worst_case_reservation=schedule.reservation,
        peeling=peeling,
        syndrome=syndrome,
        recovery=recovery,
        grouping=grouping,
        localization=localization,
        compact_estimator=compact,
        localized_estimator=density,
        decoded_density=density,
        realized_copy_ledger=ledger,
        realized_total=ledger.total,
        reservation_slack=schedule.reservation.total - ledger.total,
        theorem_certified=False,
        estimator_available=True,
        execution_complete=False,
        budget_truncated=True,
        truncated_stages=tuple(dict.fromkeys(str(stage) for stage in truncated_stages)),
        degradation_reason=reason,
        copies_remaining=budget - ledger.total,
        fixed_budget_stage_records=complete_stage_records,
    )


def _full_cebp_tomography_fixed_budget_graceful(
    source: CEBPLearnerView,
    config: EndToEndConfig,
    schedule: EndToEndSchedule,
    seeds: StageSeedLedger,
) -> EndToEndResult:
    """Run the explicitly selected budget-native, always-estimating policy."""

    budget = int(config.max_realized_copies)
    caps = dict(fixed_budget_resolved_stage_caps(config))
    peeling_cfg, recovery_cfg, grouping_cfg, syndrome_cfg, tomography_cfg = _execution_configs(
        schedule, config
    )
    records: list[FixedBudgetStageRecord] = []
    truncated: list[str] = []

    peel_available = caps["peeling"]
    M1 = peel_available // 2
    if M1 <= 0:
        records.append(FixedBudgetStageRecord(
            "peeling", peel_available, 0, peel_available, False, False,
            "insufficient_for_one_bell_round",
        ))
        return _graceful_mixed_fallback(
            schedule, seeds, config, CopyLedger(), "insufficient_peeling_copies",
            _FIXED_BUDGET_STAGES, records,
        )
    peeling_cfg = replace(peeling_cfg, M1=M1)
    peeling = empirical_certified_stabilizer_peeling(
        source, peeling_cfg, seed=seeds.peeling_seed,
        simulation_backend=config.simulation_backend,
    )
    peel_used = peeling.copy_ledger.total
    peel_truncated = not peeling.success
    if peel_truncated:
        truncated.append("peeling")
    records.append(FixedBudgetStageRecord(
        "peeling", peel_available, peel_used, peel_available - peel_used,
        peel_used == peel_available, peeling.success,
        None if peeling.success else "peeling_inconclusive",
    ))
    if not peeling.success:
        return _graceful_mixed_fallback(
            schedule, seeds, config, peeling.copy_ledger,
            "peeling_inconclusive", (*truncated, "recovery", "grouping", "syndrome", "tomography"),
            records, peeling=peeling,
        )

    recovery_available = caps["recovery"]
    M2 = recovery_available // 2
    if M2 <= 0:
        records.append(FixedBudgetStageRecord(
            "recovery", recovery_available, 0, recovery_available, False, False,
            "insufficient_for_one_bell_round",
        ))
        return _graceful_mixed_fallback(
            schedule, seeds, config, peeling.copy_ledger,
            "insufficient_recovery_copies", (*truncated, "recovery", "grouping", "syndrome", "tomography"),
            records, peeling=peeling,
        )
    recovery_cfg = replace(
        recovery_cfg,
        M2=M2,
        allow_uncalibrated_peeling=True,
        allow_margin_failure=True,
    )
    try:
        recovery = empirical_rank_guided_sector_recovery(
            source, peeling, recovery_cfg, seed=seeds.recovery_seed,
            simulation_backend=config.simulation_backend,
        )
    except (RecoveryPreconditionError, ValueError, RuntimeError) as error:
        records.append(FixedBudgetStageRecord(
            "recovery", recovery_available, 0, recovery_available, False, False,
            f"recovery_inconclusive: {error}",
        ))
        return _graceful_mixed_fallback(
            schedule, seeds, config, peeling.copy_ledger,
            f"recovery_inconclusive: {error}", (*truncated, "recovery", "grouping", "syndrome", "tomography"),
            records, peeling=peeling,
        )
    recovery_used = recovery.copy_ledger.total
    recovery_truncated = not recovery.success
    if recovery_truncated:
        truncated.append("recovery")
    records.append(FixedBudgetStageRecord(
        "recovery", recovery_available, recovery_used,
        recovery_available - recovery_used,
        recovery_used == recovery_available, recovery.success,
        None if recovery.success else "recovery_inconclusive",
    ))
    if not recovery.success:
        return _graceful_mixed_fallback(
            schedule, seeds, config, recovery.cumulative_copy_ledger,
            "recovery_inconclusive", (*truncated, "grouping", "syndrome", "tomography"),
            records, peeling=peeling, recovery=recovery,
        )

    before_grouping = recovery.cumulative_copy_ledger.total
    grouping_available = caps["grouping"]
    grouping_cfg = replace(
        grouping_cfg,
        eta_test=max(float(grouping_cfg.eta_test), float(np.finfo(float).eps)),
        allow_uncalibrated_recovery=True,
        allow_no_false_merge_margin_failure=True,
        sampling_policy=GroupingSamplingPolicy.FIXED_BUDGET.value,
    )
    grouping = empirical_hierarchical_cumulant_grouping(
        source, peeling, recovery, grouping_cfg,
        seed=seeds.grouping_seed,
        simulation_backend=config.simulation_backend,
        fixed_budget_local_cap=grouping_available,
    )
    grouping_used = grouping.realized_grouping_copies
    grouping_truncated = bool(grouping.grouping_budget_truncated)
    if grouping_truncated:
        truncated.append("grouping")
    records.append(FixedBudgetStageRecord(
        "grouping", grouping_available, grouping_used,
        grouping_available - grouping_used,
        grouping_used == grouping_available,
        grouping.success and grouping.grouping_complete,
        (
            grouping.failure_reason
            if not grouping.success
            else "grouping_budget_exhausted" if grouping_truncated else None
        ),
    ))
    if not grouping.success:
        ledger = grouping.cumulative_copy_ledger
        return _graceful_mixed_fallback(
            schedule, seeds, config, ledger,
            grouping.failure_reason or "grouping_inconclusive",
            (*truncated, "syndrome", "tomography"), records,
            peeling=peeling, recovery=recovery, grouping=grouping,
        )

    localization = localize_grouped_recovery(
        recovery, grouping, d=source.d,
        config=LocalizationConfig(
            allow_uncertified_grouping=True,
            return_details=config.return_details,
            verify_dense_unitary=False,
            max_dense_qubits=int(config.max_dense_debug_qubits),
            materialize_dense_clifford=False,
            enforce_model_block_bound=False,
        ),
    )
    if not localization.success:
        return _graceful_mixed_fallback(
            schedule, seeds, config, grouping.cumulative_copy_ledger,
            localization.failure_reason or "localization_inconclusive",
            (*truncated, "syndrome", "tomography"), records,
            peeling=peeling, recovery=recovery, grouping=grouping,
            localization=localization,
        )

    before_syndrome = grouping.cumulative_copy_ledger.total
    syndrome_available = caps["syndrome"]
    t = int(peeling.t)
    M_sgn = 0 if t == 0 else syndrome_available // t
    syndrome_truncated = t > 0 and M_sgn == 0
    if t > 0 and M_sgn == 0:
        records.append(FixedBudgetStageRecord(
            "syndrome", syndrome_available, 0, syndrome_available, False, False,
            "insufficient_for_one_shot_per_peeled_generator",
        ))
        return _graceful_mixed_fallback(
            schedule, seeds, config, grouping.cumulative_copy_ledger,
            "unknown_syndrome_signs_demoted_to_global_mixed",
            (*truncated, "syndrome", "tomography"), records,
            peeling=peeling, recovery=recovery, grouping=grouping,
            localization=localization,
        )
    syndrome_run_cfg = syndrome_cfg if t == 0 else replace(syndrome_cfg, M_sgn=M_sgn)
    syndrome = recover_peeling_syndrome(
        source, peeling, syndrome_run_cfg, seed=seeds.syndrome_seed,
        prior_copy_ledger=grouping.cumulative_copy_ledger,
    )
    syndrome_used = syndrome.syndrome_sign_pool
    if syndrome_truncated:
        truncated.append("syndrome")
    records.append(FixedBudgetStageRecord(
        "syndrome", syndrome_available, syndrome_used,
        syndrome_available - syndrome_used,
        syndrome_used == syndrome_available, syndrome.success,
        None if syndrome.success else "syndrome_inconclusive",
    ))
    if not syndrome.success:
        return _graceful_mixed_fallback(
            schedule, seeds, config, syndrome.cumulative_copy_ledger,
            "syndrome_inconclusive_demoted_to_global_mixed",
            (*truncated, "tomography"), records,
            peeling=peeling, recovery=recovery, grouping=grouping,
            localization=localization, syndrome=syndrome,
        )

    prior = syndrome.cumulative_copy_ledger
    tomography_available = caps["tomography"]
    if schedule.branch == "d1_specialized":
        attempts_per_setting = tomography_available // 3
        tomography = _conditional_one_qubit_tomography(
            source,
            peeling,
            localization,
            syndrome,
            schedule,
            seed=seeds.tomography_seed,
            max_dense_qubits=config.max_dense_qubits,
            materialize_localized_estimator=config.materialize_dense_estimator,
            attempts_override=attempts_per_setting,
            budget_native=True,
        )
        tomography_used = tomography.attempted_copies
        tomography_truncated = False
        realized = CopyLedger(
            prior.entries + (("conditional_one_qubit_pool", tomography_used),)
        )
    else:
        tomography = tomograph_localized_registers_fixed_budget(
            source, peeling, localization, tomography_available, tomography_cfg,
            seed=seeds.tomography_seed, prior_copy_ledger=prior,
        )
        tomography_used = tomography.block_tomography_pool
        tomography_truncated = bool(tomography.budget_truncated)
        realized = tomography.cumulative_copy_ledger
    if tomography_truncated:
        truncated.append("tomography")
    records.append(FixedBudgetStageRecord(
        "tomography", tomography_available, tomography_used,
        tomography_available - tomography_used,
        tomography_used == tomography_available,
        tomography.success and not tomography_truncated,
        "incomplete_pauli_coverage" if tomography_truncated else None,
    ))
    if not tomography.success:
        return _graceful_mixed_fallback(
            schedule, seeds, config, realized,
            tomography.failure_reason or "tomography_inconclusive",
            (*truncated, "tomography"), records,
            peeling=peeling, recovery=recovery, grouping=grouping,
            localization=localization, syndrome=syndrome,
        )

    if realized.total > budget:
        raise RuntimeError("Graceful fixed-budget execution exceeded its budget.")
    compact = _compact_estimator(peeling, localization, syndrome, tomography.estimates)
    localized_estimator = (
        _localized_estimator_from_compact(compact)
        if config.materialize_dense_estimator
        else None
    )
    decoded = (
        materialize_compact_cebp_estimator(
            compact, max_dense_qubits=int(config.max_dense_debug_qubits)
        )
        if config.materialize_dense_estimator
        else None
    )
    phase6 = (
        None
        if schedule.branch == "d1_specialized"
        else RecoveredBlockTomographyResult(
            True, None, syndrome, tomography, syndrome.syndrome_bits,
            tuple(
                (estimate.cluster, estimate.J_C, estimate.nu_hat)
                for estimate in tomography.estimates
            ),
            localization.J_aux, realized, False,
        )
    )
    return EndToEndResult(
        success=True,
        failure_stage=None,
        failure_reason=None,
        branch=schedule.branch,
        n=source.n,
        d=source.d,
        epsilon=config.epsilon,
        delta=config.delta,
        schedule=schedule,
        seed_ledger=seeds,
        worst_case_reservation=schedule.reservation,
        peeling=peeling,
        syndrome=syndrome,
        recovery=recovery,
        grouping=grouping,
        localization=localization,
        tomography=tomography,
        phase6_handoff=phase6,
        compact_estimator=compact,
        localized_estimator=localized_estimator,
        decoded_density=decoded,
        realized_copy_ledger=realized,
        realized_total=realized.total,
        reservation_slack=schedule.reservation.total - realized.total,
        theorem_certified=False,
        estimator_available=True,
        execution_complete=not truncated,
        budget_truncated=bool(truncated),
        truncated_stages=tuple(dict.fromkeys(truncated)),
        degradation_reason=("fixed_budget_undersampling" if truncated else None),
        copies_remaining=budget - realized.total,
        fixed_budget_stage_records=tuple(records),
    )


def _full_cebp_tomography_impl(
    source: CEBPLearnerView,
    epsilon: Optional[float] = None,
    delta: Optional[float] = None,
    *,
    seed: RngSeed = None,
    config: Optional[EndToEndConfig] = None,
) -> EndToEndResult:
    """Run the complete learner-facing calibrated CEBP pipeline.

    Schedule construction and worst-case reservation always precede sampling.
    A finite default resource guard returns an explicit operational failure for
    formal schedules too large for the dense simulator.  Explicit stage-budget
    overrides are useful for compact simulator fixtures but can never make a
    theorem-certified result.
    """
    if not isinstance(source, CEBPLearnerView):
        raise TypeError("full_cebp_tomography accepts a CEBPLearnerView only.")
    if source.n != source.measurement_source.n:
        raise ValueError("Learner view and measurement source dimensions differ.")
    if config is None:
        if epsilon is None or delta is None:
            raise ValueError("epsilon and delta are required without EndToEndConfig.")
        config = EndToEndConfig(float(epsilon), float(delta), seed=seed)
    elif not isinstance(config, EndToEndConfig):
        raise TypeError("config must be an EndToEndConfig.")
    else:
        if epsilon is not None and not np.isclose(float(epsilon), config.epsilon):
            raise ValueError("epsilon disagrees with EndToEndConfig.")
        if delta is not None and not np.isclose(float(delta), config.delta):
            raise ValueError("delta disagrees with EndToEndConfig.")
        if seed is not None and config.seed is not None and seed != config.seed:
            raise ValueError("seed disagrees with EndToEndConfig.")
    if source.n > int(config.max_enumeration_qubits):
        raise ValueError(
            "End-to-end exhaustive Pauli processing is outside the supported "
            f"qubit policy: n={source.n} exceeds max_enumeration_qubits="
            f"{config.max_enumeration_qubits}."
        )
    master_seed = config.seed if seed is None else seed
    schedule = calibrated_end_to_end_schedule(
        source.n, source.d, config.epsilon, config.delta
    )
    seeds = _stage_seed_ledger(master_seed)
    if config.execution_policy is ExecutionPolicy.FIXED_BUDGET_GRACEFUL:
        return _full_cebp_tomography_fixed_budget_graceful(
            source, config, schedule, seeds
        )
    if (
        config.max_reserved_copies is not None
        and schedule.reservation.total > config.max_reserved_copies
        and not config.uses_execution_overrides
    ):
        return _end_to_end_failure(
            schedule,
            seeds,
            "reservation",
            "worst_case_reservation_exceeds_resource_guard",
        )
    peeling_config, recovery_config, grouping_config, syndrome_config, tomography_config = _execution_configs(
        schedule, config
    )
    if peeling_config.M1 is not None:
        try:
            _check_execution_copy_cap(
                config.max_realized_copies,
                0,
                2 * int(peeling_config.M1),
                "peeling_bell_pool",
            )
        except CopyBudgetExceeded as error:
            return _end_to_end_failure(schedule, seeds, "copy_budget", str(error))
    peeling = empirical_certified_stabilizer_peeling(
        source,
        peeling_config,
        seed=seeds.peeling_seed,
        simulation_backend=config.simulation_backend,
    )
    if not peeling.success:
        return _end_to_end_failure(
            schedule,
            seeds,
            "peeling",
            peeling.failure_reason or "peeling_failed",
            peeling=peeling,
            ledger=peeling.copy_ledger,
        )
    syndrome_requested = int(peeling.t) * int(
        syndrome_config.M_sgn
        if syndrome_config.M_sgn is not None
        else syndrome_sign_sample_count(source.n, syndrome_config.h_min, syndrome_config.zeta_sgn)
    )
    try:
        _check_execution_copy_cap(
            config.max_realized_copies,
            peeling.copy_ledger.total,
            syndrome_requested,
            "syndrome_sign_pool",
        )
    except CopyBudgetExceeded as error:
        return _end_to_end_failure(
            schedule,
            seeds,
            "copy_budget",
            str(error),
            peeling=peeling,
            ledger=peeling.copy_ledger,
        )
    syndrome = recover_peeling_syndrome(
        source,
        peeling,
        syndrome_config,
        seed=seeds.syndrome_seed,
        prior_copy_ledger=peeling.copy_ledger,
    )
    if not syndrome.success:
        return _end_to_end_failure(
            schedule,
            seeds,
            "syndrome",
            syndrome.failure_reason or "syndrome_failed",
            peeling=peeling,
            syndrome=syndrome,
            ledger=syndrome.cumulative_copy_ledger,
        )
    try:
        if recovery_config.M2 is not None:
            _check_execution_copy_cap(
                config.max_realized_copies,
                syndrome.cumulative_copy_ledger.total,
                2 * int(recovery_config.M2),
                "recovery_bell_pool",
            )
        recovery = empirical_rank_guided_sector_recovery(
            source,
            peeling,
            recovery_config,
            seed=seeds.recovery_seed,
            simulation_backend=config.simulation_backend,
        )
    except CopyBudgetExceeded as error:
        return _end_to_end_failure(
            schedule,
            seeds,
            "copy_budget",
            str(error),
            peeling=peeling,
            syndrome=syndrome,
            ledger=syndrome.cumulative_copy_ledger,
        )
    except RecoveryPreconditionError as error:
        return _end_to_end_failure(
            schedule,
            seeds,
            "recovery",
            str(error),
            peeling=peeling,
            syndrome=syndrome,
            ledger=syndrome.cumulative_copy_ledger,
        )
    if not recovery.success:
        ledger = CopyLedger(
            recovery.cumulative_copy_ledger.entries
            + (("syndrome_sign_pool", syndrome.syndrome_sign_pool),)
        )
        return _end_to_end_failure(
            schedule,
            seeds,
            "recovery",
            recovery.failure_reason or "recovery_failed",
            peeling=peeling,
            syndrome=syndrome,
            recovery=recovery,
            ledger=ledger,
        )
    grouping = empirical_hierarchical_cumulant_grouping(
        source,
        peeling,
        recovery,
        grouping_config,
        seed=seeds.grouping_seed,
        simulation_backend=config.simulation_backend,
        max_realized_copies=config.max_realized_copies,
        realized_before_grouping=(
            recovery.cumulative_copy_ledger.total + syndrome.syndrome_sign_pool
        ),
    )
    if not grouping.success:
        ledger = grouping.cumulative_copy_ledger.with_entry(
            "syndrome_sign_pool", syndrome.syndrome_sign_pool
        )
        return _end_to_end_failure(
            schedule,
            seeds,
            "copy_budget"
            if (grouping.failure_reason or "").startswith("copy_budget:")
            else "grouping",
            grouping.failure_reason or "grouping_failed",
            peeling=peeling,
            syndrome=syndrome,
            recovery=recovery,
            grouping=grouping,
            ledger=ledger,
        )
    localization = localize_grouped_recovery(
        recovery,
        grouping,
        d=source.d,
        config=LocalizationConfig(
            allow_uncertified_grouping=config.allow_uncertified_execution,
            return_details=config.return_details,
            verify_dense_unitary=False,
            max_dense_qubits=int(config.max_dense_debug_qubits),
            materialize_dense_clifford=False,
        ),
    )
    if not localization.success:
        ledger = grouping.cumulative_copy_ledger.with_entry(
            "syndrome_sign_pool", syndrome.syndrome_sign_pool
        )
        return _end_to_end_failure(
            schedule,
            seeds,
            "localization",
            localization.failure_reason or "localization_failed",
            peeling=peeling,
            syndrome=syndrome,
            recovery=recovery,
            grouping=grouping,
            localization=localization,
            ledger=ledger,
        )

    if schedule.branch == "d1_specialized":
        conditional_attempts = int(
            config.d1_attempts_per_setting_override
            if config.d1_attempts_per_setting_override is not None
            else schedule.d1_attempts_per_setting
        )
        try:
            _check_execution_copy_cap(
                config.max_realized_copies,
                grouping.cumulative_copy_ledger.total + syndrome.syndrome_sign_pool,
                3 * conditional_attempts,
                "conditional_one_qubit_pool",
            )
        except CopyBudgetExceeded as error:
            prior = grouping.cumulative_copy_ledger.with_entry(
                "syndrome_sign_pool", syndrome.syndrome_sign_pool
            )
            return _end_to_end_failure(
                schedule,
                seeds,
                "copy_budget",
                str(error),
                peeling=peeling,
                syndrome=syndrome,
                recovery=recovery,
                grouping=grouping,
                localization=localization,
                ledger=prior,
            )
        conditional = _conditional_one_qubit_tomography(
            source,
            peeling,
            localization,
            syndrome,
            schedule,
            seed=seeds.tomography_seed,
            max_dense_qubits=config.max_dense_qubits,
            materialize_localized_estimator=config.materialize_dense_estimator,
            target_accepted_override=config.d1_accepted_per_setting_override,
            attempts_override=config.d1_attempts_per_setting_override,
        )
        realized = CopyLedger(
            (
                ("peeling_bell_pool", peeling.copy_ledger.as_dict()["peeling_bell_pool"]),
                ("recovery_bell_pool", recovery.copy_ledger.as_dict()["recovery_bell_pool"]),
                ("syndrome_sign_pool", syndrome.syndrome_sign_pool),
                ("conditional_one_qubit_pool", conditional.attempted_copies),
            )
        )
        if not conditional.success:
            return _end_to_end_failure(
                schedule,
                seeds,
                "tomography",
                conditional.failure_reason or "tomography_failed",
                peeling=peeling,
                syndrome=syndrome,
                recovery=recovery,
                grouping=grouping,
                localization=localization,
                tomography=conditional,
                ledger=realized,
            )
        estimates = conditional.estimates
        tomography_output: Union[TomographyResult, ConditionalOneQubitTomographyResult] = conditional
        phase6_handoff = None
    else:
        prior = grouping.cumulative_copy_ledger.with_entry(
            "syndrome_sign_pool", syndrome.syndrome_sign_pool
        )
        if config.max_realized_copies is not None:
            try:
                tomography_budgets = _tomography_budgets(localization, tomography_config)
                tomography_requested = max(
                    (budget.L_C for budget in tomography_budgets), default=0
                )
                _check_execution_copy_cap(
                    config.max_realized_copies,
                    prior.total,
                    tomography_requested,
                    "block_tomography_pool",
                )
            except CopyBudgetExceeded as error:
                return _end_to_end_failure(
                    schedule,
                    seeds,
                    "copy_budget",
                    str(error),
                    peeling=peeling,
                    syndrome=syndrome,
                    recovery=recovery,
                    grouping=grouping,
                    localization=localization,
                    ledger=prior,
                )
        tomography = tomograph_localized_registers(
            source,
            peeling,
            localization,
            tomography_config,
            seed=seeds.tomography_seed,
            prior_copy_ledger=prior,
            simulation_backend=config.simulation_backend,
        )
        if not tomography.success:
            return _end_to_end_failure(
                schedule,
                seeds,
                "tomography",
                tomography.failure_reason or "tomography_failed",
                peeling=peeling,
                syndrome=syndrome,
                recovery=recovery,
                grouping=grouping,
                localization=localization,
                tomography=tomography,
                ledger=tomography.cumulative_copy_ledger,
            )
        phase6_handoff = RecoveredBlockTomographyResult(
            True,
            None,
            syndrome,
            tomography,
            syndrome.syndrome_bits,
            tuple(
                (estimate.cluster, estimate.J_C, estimate.nu_hat)
                for estimate in tomography.estimates
            ),
            localization.J_aux,
            tomography.cumulative_copy_ledger,
            bool(
                syndrome.theorem_preconditions_hold
                and tomography.theorem_preconditions_hold
            ),
        )
        try:
            validate_recovered_block_tomography_handoff(
                peeling, localization, phase6_handoff
            )
        except ValueError as error:
            return _end_to_end_failure(
                schedule,
                seeds,
                "handoff_validation",
                str(error),
                peeling=peeling,
                syndrome=syndrome,
                recovery=recovery,
                grouping=grouping,
                localization=localization,
                tomography=tomography,
                ledger=tomography.cumulative_copy_ledger,
            )
        estimates = tomography.estimates
        tomography_output = tomography
        realized = tomography.cumulative_copy_ledger

    structural = compute_structural_certificate(
        schedule, peeling, recovery, grouping, localization
    )
    compact = _compact_estimator(peeling, localization, syndrome, estimates)
    localized_estimator = (
        _localized_estimator_from_compact(compact)
        if config.materialize_dense_estimator
        else None
    )
    decoded = (
        materialize_compact_cebp_estimator(
            compact, max_dense_qubits=int(config.max_dense_debug_qubits)
        )
        if config.materialize_dense_estimator
        else None
    )
    theorem_flags = {
        "calibrated_schedule_used": not config.uses_execution_overrides,
        "peeling": peeling.theorem_preconditions_hold,
        "recovery": recovery.theorem_recovery_preconditions_hold,
        "grouping": (
            True
            if schedule.branch == "d1_specialized"
            else grouping.theorem_grouping_preconditions_hold
        ),
        "localization": localization.theorem_localization_preconditions_hold,
        "syndrome": syndrome.theorem_preconditions_hold,
        "tomography": tomography_output.theorem_preconditions_hold,
    }
    certificate = build_end_to_end_certificate(
        schedule,
        structural,
        theorem_preconditions=theorem_flags,
        realized_copy_ledger=realized,
        operational_success=True,
    )
    return EndToEndResult(
        True,
        None,
        None,
        schedule.branch,
        source.n,
        source.d,
        config.epsilon,
        config.delta,
        schedule,
        seeds,
        schedule.reservation,
        peeling,
        syndrome,
        recovery,
        grouping,
        localization,
        tomography_output,
        phase6_handoff,
        compact,
        localized_estimator,
        decoded,
        structural,
        certificate,
        realized,
        realized.total,
        schedule.reservation.total - realized.total,
        certificate.theorem_certified,
        True,
        True,
        False,
        (),
        None,
        (
            None
            if config.max_realized_copies is None
            else config.max_realized_copies - realized.total
        ),
        (),
    )


def full_cebp_tomography(
    source: CEBPLearnerView,
    epsilon: Optional[float] = None,
    delta: Optional[float] = None,
    *,
    seed: RngSeed = None,
    config: Optional[EndToEndConfig] = None,
) -> EndToEndResult:
    """Run the learner and attach diagnostic-only stage wall times."""

    collector: list[Tuple[str, float]] = []
    token = _STAGE_TIMING_CONTEXT.set(collector)
    overall_started = time.perf_counter()
    try:
        result = _full_cebp_tomography_impl(
            source, epsilon, delta, seed=seed, config=config
        )
    finally:
        overall_elapsed = time.perf_counter() - overall_started
        _STAGE_TIMING_CONTEXT.reset(token)
    detailed: list[Tuple[str, float]] = []
    for prefix, stage_result in (
        ("peeling", result.peeling),
        ("recovery", result.recovery),
    ):
        diagnostics = getattr(stage_result, "enumeration_diagnostics", None)
        if diagnostics is not None:
            detailed.extend(
                (f"{prefix}.{name}", float(seconds))
                for name, seconds in diagnostics.stage_wall_times
            )
    detailed.extend((name, float(seconds)) for name, seconds in collector)
    detailed.append(("overall_end_to_end", overall_elapsed))
    return replace(
        result,
        performance_diagnostics=PerformanceDiagnostics(tuple(detailed)),
    )


def debug_end_to_end_trace_error(
    result: EndToEndResult,
    target: Union[CEBPInstance, StateLike],
    *,
    max_dense_qubits: Optional[int] = None,
) -> float:
    """Oracle/debug trace-norm error under an explicit bounded dense policy."""
    if not isinstance(result, EndToEndResult) or not result.success:
        raise ValueError("A successful EndToEndResult is required.")
    if max_dense_qubits is None:
        if not isinstance(target, CEBPInstance):
            raise ValueError(
                "max_dense_qubits is required when target has no configured debug limit."
            )
        max_dense_qubits = target.max_dense_debug_qubits
    max_dense_qubits = _validate_positive_integer(
        "max_dense_qubits", max_dense_qubits
    )
    if result.n > max_dense_qubits:
        raise ValueError("dense trace-error debug resource guard exceeded.")
    estimate = result.decoded_density
    if estimate is None:
        if result.compact_estimator is None:
            raise ValueError("Successful result is missing its compact estimator.")
        estimate = materialize_compact_cebp_estimator(
            result.compact_estimator,
            max_dense_qubits=max_dense_qubits,
        )
    state = (
        target.materialize_state_debug(max_qubits=max_dense_qubits)
        if isinstance(target, CEBPInstance)
        else target
    )
    qobj = _v1._state_to_qobj(state, result.n)
    density = qobj * qobj.dag() if qobj.isket else qobj
    difference = np.asarray(estimate.full() - density.full(), dtype=complex)
    return float(np.linalg.svd(difference, compute_uv=False).sum())


def validate_cebp_instance(
    instance: CEBPInstance,
    *,
    atol: float = 1e-9,
    validate_dense_reference: bool = False,
) -> None:
    """Raise if a generated instance violates a Phase-1 model invariant."""
    n, d = instance.n, instance.d
    truth = instance.oracle_truth
    flattened = truth.hidden_partition_qubits
    if len(flattened) != n or sorted(flattened) != list(range(n)):
        raise ValueError("The hidden partition must cover [n] exactly once.")
    if any(not block or len(block) > d for block in truth.hidden_partition):
        raise ValueError("Every hidden block must be nonempty and have size at most d.")
    if len(truth.latent_block_states) != len(truth.hidden_partition):
        raise ValueError("Every hidden block must have exactly one latent state.")
    for block, state in zip(truth.hidden_partition, truth.latent_block_states):
        if _infer_state_qubits(state) != len(block):
            raise ValueError("Latent block-state dimension does not match its block.")
        if state.dims[0] != [2] * len(block):
            raise ValueError("Latent block states must expose qubit tensor dimensions.")
        if state.isket:
            if not np.isclose(state.norm(), 1.0, atol=atol):
                raise ValueError("Latent block kets must be normalized.")
        else:
            if not state.isherm or not np.isclose(state.tr(), 1.0, atol=atol):
                raise ValueError("Latent density matrices must be Hermitian and trace one.")
            _validate_density_psd(state, tolerance=atol)

    if not np.array_equal(truth.encoder_tableau, truth._structured_state.encoder.tableau):
        raise ValueError("Encoder tableau and compact signed Clifford disagree.")
    if not _v1.is_symplectic(truth.encoder_tableau):
        raise ValueError("Encoder tableau is not symplectic.")
    if instance.is_ket != all(state.isket for state in truth.latent_block_states):
        raise ValueError("Instance ket metadata disagrees with latent blocks.")
    if validate_dense_reference:
        if not tableau_unitary_convention_holds(
            truth.encoder_tableau, truth.encoder_unitary, atol=atol
        ):
            raise ValueError("Encoder tableau and unitary use inconsistent conventions.")
        unitary_qobj = qt.Qobj(truth.encoder_unitary, dims=[[2] * n, [2] * n])
        if truth.latent_product_state.isket:
            expected_ket = (unitary_qobj * truth.latent_product_state).unit()
            expected = expected_ket * expected_ket.dag()
        else:
            expected = unitary_qobj * truth.latent_product_state * unitary_qobj.dag()
        state = instance.state
        actual = state * state.dag() if state.isket else state
        if not np.allclose(actual.full(), expected.full(), atol=atol):
            raise ValueError("Encoded state does not equal U_c rho_latent U_c^dagger.")
        if not actual.isherm or not np.isclose(actual.tr(), 1.0, atol=atol):
            raise ValueError("Encoded density matrix must be Hermitian and trace one.")
        _validate_density_psd(actual, tolerance=atol)
    if instance.copy_ledger.total != 0:
        raise ValueError("A newly generated Phase-1 instance must consume no measurement copies.")


def _random_compact_clifford(
    n: int, steps: Optional[int], seed: RngSeed
) -> SignedClifford:
    """V1-compatible random walk without constructing its dense unitary."""

    if steps is None:
        steps = 5 * n
    if steps == 0:
        return SignedClifford.identity(n)
    rng = _v1._rng(seed)
    tableau = np.eye(2 * n, dtype=np.uint8)
    gates = []
    for _ in range(steps):
        kind = int(rng.integers(0, 2 if n == 1 else 3))
        if kind == 0:
            qubit = int(rng.integers(0, n))
            _v1.apply_h_right(tableau, qubit)
            gates.append(("H", qubit))
        elif kind == 1:
            qubit = int(rng.integers(0, n))
            _v1.apply_s_right(tableau, qubit)
            gates.append(("S", qubit))
        else:
            control = int(rng.integers(0, n))
            target = int(rng.integers(0, n - 1))
            if target >= control:
                target += 1
            _v1.apply_cnot_right(tableau, control, target)
            gates.append(("CNOT", control, target))
    return SignedClifford(n, tableau, tuple(gates))


def random_cebp_state(
    n: int,
    d: int,
    *,
    block_sizes: Optional[Sequence[int]] = None,
    block_states: Optional[Sequence[StateLike]] = None,
    pure: bool = False,
    clifford_steps: Optional[int] = None,
    seed: RngSeed = None,
    oracle_backend: Literal["structured", "legacy_dense"] = "structured",
    max_dense_debug_qubits: int = 8,
) -> CEBPInstance:
    """Generate a general Clifford-encoded block-product state.

    Random pure blocks are Haar-distributed via normalized complex Gaussian
    vectors.  Random mixed blocks use the Hilbert--Schmidt (Ginibre) ensemble.
    The encoder is V1's explicitly nonuniform random walk over H, S, and CNOT.

    Supplied ``block_states`` determine whether the encoded state remains in
    ket representation.  Setting ``pure=True`` additionally requires every
    supplied block state to be represented as a ket.
    """
    n = _validate_positive_integer("n", n)
    d = _validate_positive_integer("d", d)
    max_dense_debug_qubits = _validate_positive_integer(
        "max_dense_debug_qubits", max_dense_debug_qubits
    )
    if oracle_backend not in ("structured", "legacy_dense"):
        raise ValueError("oracle_backend must be 'structured' or 'legacy_dense'.")
    seed = _validate_seed(seed)
    if clifford_steps is not None:
        if isinstance(clifford_steps, bool) or not isinstance(
            clifford_steps, (int, np.integer)
        ):
            raise TypeError("clifford_steps must be an integer or None.")
        clifford_steps = int(clifford_steps)
        if clifford_steps < 0:
            raise ValueError("clifford_steps must be nonnegative.")

    if seed is None:
        partition_seed = latent_seed = clifford_seed = None
    else:
        partition_seed, latent_seed, clifford_seed = _v1._child_seeds(seed, 3)
    seed_ledger = SeedLedger(seed, partition_seed, latent_seed, clifford_seed)

    partition_rng = _v1._rng(partition_seed)
    latent_rng = _v1._rng(latent_seed)

    supplied_states: Optional[Tuple[StateLike, ...]] = None
    if block_states is not None:
        supplied_states = tuple(block_states)
        if not supplied_states:
            raise ValueError("block_states must contain at least one state.")
        inferred_sizes = tuple(_infer_state_qubits(state) for state in supplied_states)
        if any(size <= 0 for size in inferred_sizes):
            raise ValueError("Zero-qubit latent blocks are not supported.")
        if block_sizes is None:
            sizes = _validate_block_sizes(n, d, inferred_sizes)
        else:
            sizes = _validate_block_sizes(n, d, block_sizes)
            if sizes != inferred_sizes:
                raise ValueError("block_sizes do not match the supplied block-state dimensions.")
    elif block_sizes is None:
        sizes = _random_block_sizes(n, min(d, n), partition_rng)
    else:
        sizes = _validate_block_sizes(n, d, block_sizes)

    if supplied_states is None:
        latent_blocks = tuple(
            _random_pure_block(size, latent_rng)
            if pure
            else _random_mixed_block(size, latent_rng)
            for size in sizes
        )
        latent_source = "haar_pure" if pure else "hilbert_schmidt_mixed"
    else:
        latent_blocks = tuple(
            _normalize_block_state(state, size)
            for state, size in zip(supplied_states, sizes)
        )
        if pure and not all(state.isket for state in latent_blocks):
            raise ValueError("pure=True requires every supplied block state to be a ket.")
        latent_source = "user_supplied"

    is_ket = all(state.isket for state in latent_blocks)
    partition = _contiguous_partition(sizes)
    encoder = _random_compact_clifford(n, clifford_steps, clifford_seed)
    structured_state = StructuredCEBPState(
        n, partition, latent_blocks, encoder
    )

    oracle_truth = CEBPOracleTruth(
        hidden_partition=partition,
        latent_block_states=latent_blocks,
        encoder_gates=tuple(encoder.gates),
        encoder_tableau=np.asarray(encoder.tableau, dtype=np.uint8).copy(),
        encoder_sampling=(
            "identity_zero_step"
            if clifford_steps == 0
            else "nonuniform_h_s_cnot_random_walk"
        ),
        encoder_steps=len(encoder.gates),
        latent_state_source=latent_source,
        _structured_state=structured_state,
        max_dense_debug_qubits=max_dense_debug_qubits,
    )
    measurement_source = (
        SimulatorMeasurementSource(
            n=n,
            is_ket=is_ket,
            _structured_state=structured_state,
        )
        if oracle_backend == "structured"
        else SimulatorMeasurementSource(
            n=n,
            is_ket=is_ket,
            _state=structured_state.materialize_dense_debug(
                max_qubits=max_dense_debug_qubits
            ),
        )
    )
    instance = CEBPInstance(
        n=n,
        d=d,
        measurement_source=measurement_source,
        is_ket=is_ket,
        seed_ledger=seed_ledger,
        copy_ledger=CopyLedger(),
        oracle_truth=oracle_truth,
        max_dense_debug_qubits=max_dense_debug_qubits,
    )
    validate_cebp_instance(instance)
    return instance


__all__ = [
    "BellScoreRecord",
    "CallableResidualCumulantInterface",
    "CEBPInstance",
    "CEBPLearnerView",
    "CEBPOracleTruth",
    "ClusterLocalization",
    "ClusterSymplecticStructure",
    "CompactCEBPEstimator",
    "ConditionalOneQubitTomographyResult",
    "CopyLedger",
    "DebugExactResidualCumulantInterface",
    "DataProvenance",
    "DebugExactPauliScoreSource",
    "EmpiricalOrdinaryCumulantInterface",
    "FixedBudgetEmpiricalCumulantInterface",
    "EndToEndCertificate",
    "EndToEndConfig",
    "EndToEndResult",
    "EndToEndSchedule",
    "EnumerationDiagnostics",
    "EnumerationExecutionConfig",
    "EnumerationWorkspaceEstimate",
    "ExactOperationalDiagnostic",
    "ExecutionPolicy",
    "FixedBudgetStageRecord",
    "FixedBudgetStageWeights",
    "FixedBudgetRegisterTomographyBudget",
    "GeneratedPauliGroup",
    "GroupingConfig",
    "GroupingSamplingPolicy",
    "GroupingResult",
    "GroupingScan",
    "HyperedgeWitness",
    "JointPauliMeasurementRecord",
    "JointPauliCountRecord",
    "LocalizationConfig",
    "LocalizationInvariantError",
    "LocalizationResult",
    "MAX_SUPPORTED_END_TO_END_QUBITS",
    "PauliMeasurementRecord",
    "PauliScoreMap",
    "PauliScoreSource",
    "PeelingConfig",
    "PeelingResult",
    "PeelingThresholdAttempt",
    "PerformanceDiagnostics",
    "RecoveredBlockTomographyConfig",
    "RecoveredBlockTomographyResult",
    "RecoveredSector",
    "RegisterTomographyBudget",
    "RegisterTomographyEstimate",
    "RegisterTomographyRecord",
    "RegisterTomographyCountRecord",
    "RecoveryCandidate",
    "RecoveryConfig",
    "RecoveryPreconditionError",
    "RecoveryResult",
    "RecoveryStep",
    "SeedLedger",
    "SignedPauliProduct",
    "SignedClifford",
    "SimulatorMeasurementSource",
    "SimulationBackend",
    "StageSeedLedger",
    "StructuralCertificate",
    "SyndromeConfig",
    "SyndromeResult",
    "TomographyConfig",
    "TomographyResult",
    "WorstCaseCopyReservation",
    "WorkspacePhaseEstimate",
    "adaptive_cumulant_test_bound",
    "analyze_group_symplectic_structure",
    "assemble_localized_product_estimator",
    "bell_score_uniform_radius",
    "build_end_to_end_certificate",
    "calibrated_end_to_end_schedule",
    "calibrated_asymptotic_summary",
    "canonical_gf2_span_basis",
    "certified_stabilizer_peeling_v2",
    "debug_exact_certified_stabilizer_peeling",
    "debug_exact_pauli_expectation",
    "debug_exact_pauli_score",
    "debug_exact_hierarchical_cumulant_grouping",
    "debug_exact_localized_state",
    "debug_exact_operational_diagnostic",
    "debug_exact_peeled_score_source",
    "debug_exact_rank_guided_sector_recovery",
    "debug_exact_register_marginal",
    "debug_exact_residual_mixed_cumulant",
    "debug_exact_residual_signed_moment",
    "debug_exact_residual_subset_moments",
    "debug_exact_score_source",
    "debug_end_to_end_trace_error",
    "debug_oracle_residual_block_labels",
    "debug_oracle_sector_block_labels",
    "dress_residual_pauli",
    "empirical_certified_stabilizer_peeling",
    "empirical_hierarchical_cumulant_grouping",
    "empirical_mixed_cumulant_from_record",
    "empirical_rank_guided_sector_recovery",
    "estimate_enumeration_workspace",
    "full_cebp_tomography",
    "fixed_budget_nominal_stage_caps",
    "fixed_budget_resolved_stage_caps",
    "gf2_pauli_spans_equal",
    "generated_cluster_pauli_group",
    "grouping_beta_peel",
    "hierarchical_cumulant_grouping",
    "hypergraph_connected_components",
    "hermitian_pauli_product",
    "labeled_set_partitions",
    "linear_inversion_from_pauli_coefficients",
    "localize_grouped_recovery",
    "localized_measurement_source",
    "materialize_compact_cebp_estimator",
    "measure_pauli_expectation",
    "measure_signed_pauli_tuple",
    "mixed_cumulant_from_moments",
    "ordinary_cumulant_sample_count",
    "ordinary_grouping_copy_upper_bound",
    "parse_peeled_full_pauli",
    "pauli_in_recovered_span",
    "phase_free_pauli_product",
    "peeling_clifford_mapping_holds",
    "project_eigenvalues_to_simplex",
    "project_to_density_matrix_hs",
    "random_cebp_state",
    "recover_peeling_syndrome",
    "recovered_block_tomography",
    "register_tomography_budget",
    "rank_guided_sector_recovery",
    "recovered_sector_axes",
    "recovered_sector_span_basis",
    "recovered_sectors_are_symplectically_valid",
    "sample_bell_scores",
    "sample_peeled_bell_scores",
    "signed_pauli_product",
    "simplified_cumulant_test_bound",
    "singleton_anticommutation_set",
    "syndrome_sign_sample_count",
    "tableau_unitary_convention_holds",
    "tomograph_localized_registers",
    "tomograph_localized_registers_fixed_budget",
    "compute_structural_certificate",
    "validate_cebp_instance",
    "validate_grouping_against_recovery",
    "validate_recovered_block_tomography_handoff",
    "cumulant_gamma",
]
