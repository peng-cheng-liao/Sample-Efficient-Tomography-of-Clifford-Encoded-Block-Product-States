"""Compact, deterministic primitives for the scalable CEBP simulator.

This module contains no learner policy.  It provides integer Pauli indexing,
streaming GF(2) reductions, exact signed Clifford conjugation, and an
oracle-private block-product state representation.  Full-system dense arrays
are constructed only by the explicitly named debug materializers.
"""

from __future__ import annotations

import itertools
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import qutip as qt

import main as _v1


PAULI_CHARS = "IXYZ"
_CHAR_TO_DIGIT = {character: index for index, character in enumerate(PAULI_CHARS)}


def pauli_string_to_index(pauli: str) -> int:
    """Encode a Pauli in the fixed lexicographic ``I<X<Y<Z`` order."""

    if not isinstance(pauli, str) or not pauli:
        raise ValueError("pauli must be a nonempty IXYZ string.")
    index = 0
    for character in pauli:
        try:
            digit = _CHAR_TO_DIGIT[character]
        except KeyError as error:
            raise ValueError("pauli must contain only I, X, Y, and Z.") from error
        index = 4 * index + digit
    return index


def pauli_index_to_string(index: int, n: int) -> str:
    """Decode one canonical Pauli index without enumerating its predecessors."""

    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) <= 0:
        raise ValueError("n must be a positive integer.")
    n = int(n)
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise TypeError("index must be an integer.")
    index = int(index)
    if index < 0 or index >= 4**n:
        raise ValueError("Pauli index is outside [0, 4**n).")
    characters = ["I"] * n
    for position in range(n - 1, -1, -1):
        index, digit = divmod(index, 4)
        characters[position] = PAULI_CHARS[digit]
    return "".join(characters)


def pauli_index_to_symplectic_int(index: int, n: int) -> int:
    """Return V1's integer ``[x|z]`` encoding for one canonical index."""

    if n <= 0 or index < 0 or index >= 4**n:
        raise ValueError("Invalid Pauli index or qubit count.")
    x = 0
    z = 0
    value = int(index)
    for qubit in range(n - 1, -1, -1):
        value, digit = divmod(value, 4)
        if digit in (1, 2):
            x |= 1 << qubit
        if digit in (2, 3):
            z |= 1 << qubit
    return x | (z << n)


def symplectic_int_to_pauli_index(vector: int, n: int) -> int:
    """Inverse of :func:`pauli_index_to_symplectic_int`."""

    if n <= 0 or vector < 0 or vector >= 1 << (2 * n):
        raise ValueError("Invalid symplectic vector or qubit count.")
    mask = (1 << n) - 1
    x = vector & mask
    z = (vector >> n) & mask
    index = 0
    for qubit in range(n):
        digit = (1 if (x >> qubit) & 1 else 0) + (3 if (z >> qubit) & 1 else 0)
        if digit == 4:  # Y is digit two, not the arithmetic sum X+Z.
            digit = 2
        index = 4 * index + digit
    return index


class StreamingGF2Basis:
    """Canonical incremental GF(2) basis of rank at most its bit width."""

    __slots__ = ("width", "_rows")

    def __init__(self, width: int):
        if width <= 0:
            raise ValueError("width must be positive.")
        self.width = int(width)
        self._rows: dict[int, int] = {}

    def reduce(self, vector: int) -> int:
        value = int(vector)
        for pivot in sorted(self._rows):
            if (value >> pivot) & 1:
                value ^= self._rows[pivot]
        return value

    def contains(self, vector: int) -> bool:
        return self.reduce(vector) == 0

    def add(self, vector: int) -> bool:
        value = self.reduce(vector)
        if not value:
            return False
        pivot = (value & -value).bit_length() - 1
        for existing, row in tuple(self._rows.items()):
            if (row >> pivot) & 1:
                self._rows[existing] = row ^ value
        self._rows[pivot] = value
        return True

    def merge(self, rows: Iterable[int]) -> None:
        for row in rows:
            self.add(int(row))

    @property
    def rank(self) -> int:
        return len(self._rows)

    @property
    def signature(self) -> Tuple[int, ...]:
        return tuple(self._rows[pivot] for pivot in sorted(self._rows))


try:  # Numba is optional; the deterministic Python fallback remains complete.
    import numba as _numba
except Exception:  # pragma: no cover - exercised only without optional Numba.
    _numba = None


if _numba is not None:

    @_numba.njit(nogil=True, cache=False)
    def _numba_chunk_span(scores, start, stop, cutoff, n):
        rows = np.zeros(2 * n, dtype=np.uint64)
        count = 0
        for score_index in range(start, stop):
            if scores[score_index] < cutoff:
                continue
            count += 1
            value = score_index
            x = np.uint64(0)
            z = np.uint64(0)
            for qubit in range(n - 1, -1, -1):
                digit = value % 4
                value //= 4
                if digit == 1 or digit == 2:
                    x |= np.uint64(1) << np.uint64(qubit)
                if digit == 2 or digit == 3:
                    z |= np.uint64(1) << np.uint64(qubit)
            vector = x | (z << np.uint64(n))
            for pivot in range(2 * n):
                if ((vector >> np.uint64(pivot)) & np.uint64(1)) and rows[pivot]:
                    vector ^= rows[pivot]
            if not vector:
                continue
            pivot = 0
            while not ((vector >> np.uint64(pivot)) & np.uint64(1)):
                pivot += 1
            for existing in range(2 * n):
                if rows[existing] and ((rows[existing] >> np.uint64(pivot)) & np.uint64(1)):
                    rows[existing] ^= vector
            rows[pivot] = vector
        return count, rows


def _python_chunk_span(
    scores: np.ndarray, start: int, stop: int, cutoff: float, n: int
) -> Tuple[int, Tuple[int, ...]]:
    basis = StreamingGF2Basis(2 * n)
    count = 0
    for index in range(start, stop):
        if scores[index] < cutoff:
            continue
        count += 1
        basis.add(pauli_index_to_symplectic_int(index, n))
    return count, basis.signature


def _chunk_span(
    scores: np.ndarray, start: int, stop: int, cutoff: float, n: int
) -> Tuple[int, Tuple[int, ...]]:
    if _numba is None or 2 * n > 64:
        return _python_chunk_span(scores, start, stop, cutoff, n)
    count, rows = _numba_chunk_span(scores, start, stop, cutoff, n)
    return int(count), tuple(int(row) for row in rows if row)


def threshold_span_reduction(
    scores: np.ndarray,
    cutoff: float,
    n: int,
    *,
    workers: int = 1,
    chunk_size: Optional[int] = None,
) -> Tuple[int, Tuple[int, ...]]:
    """Count and span a fixed threshold set using deterministic chunks.

    Worker completion order is irrelevant: chunk results are merged strictly in
    ascending chunk order.  With Numba installed, chunk scans release the GIL.
    """

    values = np.asarray(scores)
    expected = 4**n
    if values.ndim != 1 or values.size != expected:
        raise ValueError(f"scores must be a length-{expected} vector.")
    if workers <= 0:
        raise ValueError("workers must be positive.")
    workers = min(int(workers), expected)
    if chunk_size is None:
        chunk_size = max(1, math.ceil(expected / workers))
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    chunks = tuple(
        (start, min(expected, start + int(chunk_size)))
        for start in range(0, expected, int(chunk_size))
    )
    if workers == 1 or len(chunks) == 1:
        partial = tuple(
            _chunk_span(values, start, stop, float(cutoff), n)
            for start, stop in chunks
        )
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = tuple(
                executor.submit(_chunk_span, values, start, stop, float(cutoff), n)
                for start, stop in chunks
            )
            partial = tuple(future.result() for future in futures)
    merged = StreamingGF2Basis(2 * n)
    count = 0
    for local_count, rows in partial:
        count += local_count
        merged.merge(rows)
    return count, merged.signature


def bell_score_sums_from_outcomes(outcomes: np.ndarray) -> np.ndarray:
    """Return exact integer parity sums for every Pauli in fixed index order."""

    values = np.asarray(outcomes, dtype=np.int8)
    if values.ndim != 3 or values.shape[2] != 3 or values.shape[1] <= 0:
        raise ValueError("Bell outcomes must have shape (n, rounds, 3).")
    n, rounds, _ = values.shape
    local_outcomes = np.asarray(_v1._BELL_EIG_TABLE, dtype=np.int8)
    characters = np.column_stack(
        (np.ones(4, dtype=np.int8), local_outcomes)
    ).astype(np.int64)
    matches = np.all(
        values[:, :, None, :] == local_outcomes[None, None, :, :], axis=3
    )
    if not np.all(matches.sum(axis=2) == 1):
        raise ValueError("Bell outcomes contain an unknown local category.")
    categories = np.argmax(matches, axis=2)
    del matches
    category_indices = np.zeros(rounds, dtype=np.int64)
    multiplier = 1
    for qubit in range(n):
        category_indices += categories[qubit] * multiplier
        multiplier *= 4
    histogram = np.bincount(category_indices, minlength=4**n).astype(np.int64)
    del categories, category_indices
    return bell_score_sums_from_counts(histogram, n)


def bell_score_sums_from_counts(counts: np.ndarray, n: int) -> np.ndarray:
    """Return all exact parity sums from little-endian Bell-category counts."""

    n = int(n)
    histogram = np.asarray(counts, dtype=np.int64)
    if n <= 0 or histogram.ndim != 1 or histogram.size != 4**n:
        raise ValueError("Bell counts must be a length-4**n vector.")
    if np.any(histogram < 0) or int(histogram.sum()) <= 0:
        raise ValueError("Bell counts must be nonnegative with positive total.")
    # Reorder the little-endian category histogram into canonical C-order once,
    # then transform in place. This retains exactly one new length-4**n score
    # array and avoids a full transient per tensor axis.
    result = np.array(
        histogram.reshape((4,) * n, order="F"),
        dtype=np.int64,
        order="C",
        copy=True,
    ).reshape(-1)
    _bell_score_character_transform_inplace(result, n)
    result.setflags(write=False)
    return result


if _numba is not None:

    @_numba.njit(nogil=True, cache=False)
    def _numba_permute_grouped_pauli_table(grouped, qubit_order, n):
        result = np.empty(grouped.size, dtype=np.float64)
        digits = np.empty(n, dtype=np.uint8)
        for natural_index in range(grouped.size):
            value = natural_index
            for qubit in range(n - 1, -1, -1):
                digits[qubit] = value % 4
                value //= 4
            grouped_index = 0
            for position in range(n):
                grouped_index = 4 * grouped_index + digits[qubit_order[position]]
            result[natural_index] = grouped[grouped_index]
        return result


    @_numba.njit(nogil=True, cache=False)
    def _numba_map_clifford_squared_scores(latent_scores, tableau, n):
        """Apply the phase-free physical-to-latent symplectic permutation."""

        result = np.empty(latent_scores.size, dtype=np.float64)
        source = np.empty(2 * n, dtype=np.uint8)
        latent = np.empty(2 * n, dtype=np.uint8)
        for physical_index in range(latent_scores.size):
            for bit in range(2 * n):
                source[bit] = 0
            value = physical_index
            for qubit in range(n - 1, -1, -1):
                digit = value % 4
                value //= 4
                if digit == 1 or digit == 2:
                    source[qubit] = 1
                if digit == 2 or digit == 3:
                    source[n + qubit] = 1
            for row in range(2 * n):
                parity = np.uint8(0)
                for column in range(2 * n):
                    parity ^= tableau[row, column] & source[column]
                latent[row] = parity
            latent_index = 0
            for qubit in range(n):
                x = latent[qubit]
                z = latent[n + qubit]
                digit = 0
                if x and z:
                    digit = 2
                elif x:
                    digit = 1
                elif z:
                    digit = 3
                latent_index = 4 * latent_index + digit
            result[physical_index] = latent_scores[latent_index]
        return result


    @_numba.njit(nogil=True, cache=False)
    def _numba_bell_character_transform_inplace(values, n):
        """Apply the local Bell inverse-character transform in C tensor order."""

        characters = np.array(
            (
                (1.0, 1.0, -1.0, 1.0),
                (1.0, -1.0, 1.0, 1.0),
                (1.0, 1.0, 1.0, -1.0),
                (1.0, -1.0, -1.0, -1.0),
            ),
            dtype=np.float64,
        )
        for axis in range(n):
            stride = 4 ** (n - axis - 1)
            group = 4 * stride
            for base in range(0, values.size, group):
                for offset in range(stride):
                    a0 = values[base + offset]
                    a1 = values[base + stride + offset]
                    a2 = values[base + 2 * stride + offset]
                    a3 = values[base + 3 * stride + offset]
                    for category in range(4):
                        values[base + category * stride + offset] = (
                            characters[category, 0] * a0
                            + characters[category, 1] * a1
                            + characters[category, 2] * a2
                            + characters[category, 3] * a3
                        ) / 4.0


    @_numba.njit(nogil=True, cache=False)
    def _numba_bell_score_character_transform_inplace(values, n):
        characters_t = np.array(
            (
                (1, 1, 1, 1),
                (1, -1, 1, -1),
                (-1, 1, 1, -1),
                (1, 1, -1, -1),
            ),
            dtype=np.int64,
        )
        for axis in range(n):
            stride = 4 ** (n - axis - 1)
            group = 4 * stride
            for base in range(0, values.size, group):
                for offset in range(stride):
                    a0 = values[base + offset]
                    a1 = values[base + stride + offset]
                    a2 = values[base + 2 * stride + offset]
                    a3 = values[base + 3 * stride + offset]
                    for pauli in range(4):
                        values[base + pauli * stride + offset] = (
                            characters_t[pauli, 0] * a0
                            + characters_t[pauli, 1] * a1
                            + characters_t[pauli, 2] * a2
                            + characters_t[pauli, 3] * a3
                        )


def _permute_grouped_pauli_table(
    grouped: np.ndarray, qubit_order: np.ndarray, n: int
) -> np.ndarray:
    if _numba is not None:
        return _numba_permute_grouped_pauli_table(grouped, qubit_order, n)
    result = np.empty_like(grouped)
    for natural_index in range(grouped.size):
        value = natural_index
        digits = [0] * n
        for qubit in range(n - 1, -1, -1):
            value, digits[qubit] = divmod(value, 4)
        grouped_index = 0
        for qubit in qubit_order:
            grouped_index = 4 * grouped_index + digits[int(qubit)]
        result[natural_index] = grouped[grouped_index]
    return result


def _map_clifford_squared_scores(
    latent_scores: np.ndarray, tableau: np.ndarray, n: int
) -> np.ndarray:
    tableau = np.asarray(tableau, dtype=np.uint8)
    if _numba is not None:
        return _numba_map_clifford_squared_scores(latent_scores, tableau, n)
    # Keep the no-Numba debug environment practical without allocating a
    # full ``4**n by 2*n`` symplectic table.  The chunked vectorization is the
    # exact phase-free permutation implemented by the Numba kernel above.
    result = np.empty_like(latent_scores)
    powers = np.asarray([4 ** (n - qubit - 1) for qubit in range(n)], dtype=np.int64)
    chunk_size = min(result.size, 1 << 16)
    for start in range(0, result.size, chunk_size):
        stop = min(start + chunk_size, result.size)
        indices = np.arange(start, stop, dtype=np.int64)
        digits = (indices[:, None] // powers[None, :]) % 4
        source = np.empty((stop - start, 2 * n), dtype=np.uint8)
        source[:, :n] = np.logical_or(digits == 1, digits == 2)
        source[:, n:] = np.logical_or(digits == 2, digits == 3)
        latent = (source @ tableau.T) & 1
        latent_digits = np.where(
            np.logical_and(latent[:, :n] != 0, latent[:, n:] != 0),
            2,
            np.where(latent[:, :n] != 0, 1, np.where(latent[:, n:] != 0, 3, 0)),
        )
        latent_indices = np.zeros(stop - start, dtype=np.int64)
        for qubit in range(n):
            latent_indices *= 4
            latent_indices += latent_digits[:, qubit]
        result[start:stop] = latent_scores[latent_indices]
    return result


def _bell_character_transform_inplace(values: np.ndarray, n: int) -> None:
    if _numba is not None:
        _numba_bell_character_transform_inplace(values, n)
        return
    characters = np.column_stack(
        (np.ones(4, dtype=np.float64), np.asarray(_v1._BELL_EIG_TABLE, dtype=float))
    )
    tensor = values.reshape((4,) * n)
    for axis in range(n):
        moved = np.moveaxis(tensor, axis, -1)
        moved[...] = np.matmul(moved, characters.T) / 4.0


def _bell_score_character_transform_inplace(values: np.ndarray, n: int) -> None:
    if _numba is not None:
        _numba_bell_score_character_transform_inplace(values, n)
        return
    characters_t = np.column_stack(
        (np.ones(4, dtype=np.int64), np.asarray(_v1._BELL_EIG_TABLE, dtype=np.int64))
    ).T
    tensor = values.reshape((4,) * n)
    for axis in range(n):
        moved = np.moveaxis(tensor, axis, -1)
        moved[...] = np.matmul(moved, characters_t.T)


_H_MAP = {"I": (0, "I"), "X": (0, "Z"), "Y": (2, "Y"), "Z": (0, "X")}
_S_MAP = {"I": (0, "I"), "X": (2, "Y"), "Y": (0, "X"), "Z": (0, "Z")}
_X_MAP = {"I": (0, "I"), "X": (0, "X"), "Y": (2, "Y"), "Z": (2, "Z")}
_CNOT_MAP = {
    "II": (0, "II"), "IX": (0, "IX"), "IY": (0, "ZY"), "IZ": (0, "ZZ"),
    "XI": (0, "XX"), "XX": (0, "XI"), "XY": (0, "YZ"), "XZ": (2, "YY"),
    "YI": (0, "YX"), "YX": (0, "YI"), "YY": (2, "XZ"), "YZ": (0, "XY"),
    "ZI": (0, "ZI"), "ZX": (0, "ZX"), "ZY": (0, "IY"), "ZZ": (0, "IZ"),
}


@dataclass(frozen=True)
class SignedPauli:
    """Exact ``i**phase_exponent`` times a canonical phase-free Pauli."""

    phase_exponent: int
    pauli: str

    def __post_init__(self) -> None:
        if not self.pauli or any(character not in PAULI_CHARS for character in self.pauli):
            raise ValueError("SignedPauli requires a nonempty IXYZ string.")
        object.__setattr__(self, "phase_exponent", int(self.phase_exponent) % 4)

    @property
    def coefficient(self) -> complex:
        return (1.0 + 0.0j, 1.0j, -1.0 + 0.0j, -1.0j)[self.phase_exponent]


def multiply_paulis(*paulis: str) -> SignedPauli:
    """Multiply same-size canonical Pauli strings with exact phase."""

    if not paulis:
        raise ValueError("At least one Pauli is required.")
    n = len(paulis[0])
    if n == 0 or any(len(pauli) != n for pauli in paulis):
        raise ValueError("Paulis must have one common positive length.")
    local = {
        ("I", "I"): (0, "I"), ("I", "X"): (0, "X"), ("I", "Y"): (0, "Y"), ("I", "Z"): (0, "Z"),
        ("X", "I"): (0, "X"), ("X", "X"): (0, "I"), ("X", "Y"): (1, "Z"), ("X", "Z"): (3, "Y"),
        ("Y", "I"): (0, "Y"), ("Y", "X"): (3, "Z"), ("Y", "Y"): (0, "I"), ("Y", "Z"): (1, "X"),
        ("Z", "I"): (0, "Z"), ("Z", "X"): (1, "Y"), ("Z", "Y"): (3, "X"), ("Z", "Z"): (0, "I"),
    }
    phase = 0
    result = "I" * n
    for pauli in paulis:
        if any(character not in PAULI_CHARS for character in pauli):
            raise ValueError("Paulis may contain only I, X, Y, and Z.")
        next_result = []
        for left, right in zip(result, pauli):
            local_phase, character = local[(left, right)]
            phase += local_phase
            next_result.append(character)
        result = "".join(next_result)
    return SignedPauli(phase, result)


def _tableau_from_gates(n: int, gates: Sequence[Tuple]) -> np.ndarray:
    columns = []
    for index in range(2 * n):
        pauli = ["I"] * n
        pauli[index if index < n else index - n] = "X" if index < n else "Z"
        columns.append(_conjugate_by_gates("".join(pauli), gates).pauli)
    return np.column_stack([_v1.pauli_to_symplectic_col(pauli) for pauli in columns]).astype(np.uint8)


def _conjugate_by_gates(pauli: str, gates: Sequence[Tuple]) -> SignedPauli:
    characters = list(pauli)
    phase = 0
    n = len(characters)
    for gate in gates:
        if not gate:
            raise ValueError("Empty Clifford gate tuple.")
        kind = gate[0]
        if kind in ("H", "S", "X"):
            qubit = int(gate[1])
            if qubit < 0 or qubit >= n:
                raise ValueError("Clifford gate qubit is out of range.")
            mapping = _H_MAP if kind == "H" else _S_MAP if kind == "S" else _X_MAP
            local_phase, characters[qubit] = mapping[characters[qubit]]
            phase += local_phase
        elif kind == "CNOT":
            control, target = int(gate[1]), int(gate[2])
            if control == target or min(control, target) < 0 or max(control, target) >= n:
                raise ValueError("Invalid CNOT gate tuple.")
            local_phase, pair = _CNOT_MAP[characters[control] + characters[target]]
            characters[control], characters[target] = pair
            phase += local_phase
        else:
            raise ValueError(f"Unsupported compact Clifford gate {kind!r}.")
    return SignedPauli(phase, "".join(characters))


@dataclass(frozen=True)
class SignedClifford:
    """Compact exact Clifford with convention ``U^dagger P U``.

    The tableau stores the phase-free image while ``gates`` retain the exact
    signs.  No dense matrix is cached by this object.
    """

    n: int
    tableau: np.ndarray = field(repr=False, compare=False)
    gates: Tuple[Tuple, ...]

    def __post_init__(self) -> None:
        n = int(self.n)
        tableau = np.asarray(self.tableau, dtype=np.uint8) % 2
        if n <= 0 or tableau.shape != (2 * n, 2 * n) or not _v1.is_symplectic(tableau):
            raise ValueError("SignedClifford requires a valid 2n-by-2n tableau.")
        derived = _tableau_from_gates(n, self.gates)
        if not np.array_equal(derived, tableau):
            raise ValueError("Clifford gates and tableau use inconsistent conventions.")
        frozen = tableau.copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "tableau", frozen)
        object.__setattr__(self, "gates", tuple(tuple(gate) for gate in self.gates))

    @classmethod
    def identity(cls, n: int) -> "SignedClifford":
        return cls(int(n), np.eye(2 * int(n), dtype=np.uint8), ())

    @classmethod
    def from_gates(cls, n: int, gates: Sequence[Tuple]) -> "SignedClifford":
        values = tuple(tuple(gate) for gate in gates)
        return cls(int(n), _tableau_from_gates(int(n), values), values)

    def conjugate(self, pauli: str) -> SignedPauli:
        if len(pauli) != self.n:
            raise ValueError("Pauli length does not match Clifford size.")
        return _conjugate_by_gates(pauli, self.gates)

    def compose(self, right: "SignedClifford") -> "SignedClifford":
        """Return the Clifford with unitary ``self.U @ right.U``."""

        if not isinstance(right, SignedClifford) or right.n != self.n:
            raise ValueError("Clifford composition dimensions differ.")
        return SignedClifford.from_gates(self.n, self.gates + right.gates)

    def inverse(self) -> "SignedClifford":
        gates = []
        for gate in reversed(self.gates):
            if gate[0] == "S":
                gates.extend((gate, gate, gate))
            else:
                gates.append(gate)
        return SignedClifford.from_gates(self.n, tuple(gates))

    def embedded(self, total_qubits: int, offset: int) -> "SignedClifford":
        if total_qubits < self.n or offset < 0 or offset + self.n > total_qubits:
            raise ValueError("Invalid Clifford embedding.")
        gates = []
        for gate in self.gates:
            if gate[0] == "CNOT":
                gates.append((gate[0], gate[1] + offset, gate[2] + offset))
            else:
                gates.append((gate[0], gate[1] + offset))
        return SignedClifford.from_gates(total_qubits, gates)

    def materialize_dense_debug(self, *, max_qubits: int = 8) -> np.ndarray:
        if self.n > int(max_qubits):
            raise ValueError("dense Clifford debug resource guard exceeded.")
        return _v1.gates_to_unitary(list(self.gates), self.n)


@dataclass(frozen=True)
class StructuredCEBPState:
    """Oracle-private bounded-block representation of an encoded CEBP state."""

    n: int
    partition: Tuple[Tuple[int, ...], ...]
    latent_blocks: Tuple[qt.Qobj, ...] = field(repr=False, compare=False)
    encoder: SignedClifford = field(repr=False, compare=False)
    _local_expectations: Tuple[np.ndarray, ...] = field(init=False, repr=False, compare=False)
    _all_squared_scores_cache: Optional[np.ndarray] = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.n <= 0 or self.encoder.n != self.n:
            raise ValueError("Structured state dimensions are inconsistent.")
        flattened = tuple(qubit for block in self.partition for qubit in block)
        if len(flattened) != self.n or sorted(flattened) != list(range(self.n)):
            raise ValueError("Structured partition must cover every qubit exactly once.")
        if len(self.partition) != len(self.latent_blocks):
            raise ValueError("Every structured block needs one latent state.")
        tables = []
        normalized = []
        for block, state in zip(self.partition, self.latent_blocks):
            k = len(block)
            qobj = _v1._state_to_qobj(state, k)
            normalized.append(qobj)
            table = np.empty(4**k, dtype=np.float64)
            for index in range(4**k):
                pauli = pauli_index_to_string(index, k)
                value = complex(qt.expect(_v1._qutip_pauli_op(k, pauli), qobj))
                if abs(value.imag) > 1e-9:
                    raise RuntimeError("Latent Pauli expectation acquired an imaginary part.")
                table[index] = value.real
            table.setflags(write=False)
            tables.append(table)
        object.__setattr__(self, "latent_blocks", tuple(normalized))
        object.__setattr__(self, "_local_expectations", tuple(tables))

    @property
    def local_expectation_bytes(self) -> int:
        return sum(table.nbytes for table in self._local_expectations)

    def expectation(self, physical_pauli: str) -> float:
        if len(physical_pauli) != self.n:
            raise ValueError("Physical Pauli has the wrong length.")
        latent = self.encoder.conjugate(physical_pauli)
        if latent.phase_exponent not in (0, 2):
            raise RuntimeError("Hermitian Clifford conjugation produced a non-real phase.")
        value = 1.0 if latent.phase_exponent == 0 else -1.0
        for block, table in zip(self.partition, self._local_expectations):
            local = "".join(latent.pauli[qubit] for qubit in block)
            value *= float(table[pauli_string_to_index(local)])
        return float(np.clip(value, -1.0, 1.0))

    def all_squared_pauli_scores(self) -> np.ndarray:
        """Return every exact ``|Tr(rho P)|**2`` in canonical Pauli order.

        The table is composed from bounded latent-block expectation tables and
        permuted through the compact Clifford tableau.  No dense global state
        or Clifford unitary is constructed.  The returned array is a cached,
        contiguous, read-only float64 vector ordered exactly like
        :func:`pauli_index_to_string`.
        """

        cached = self._all_squared_scores_cache
        if cached is not None:
            return cached
        latent_scores = np.array([1.0], dtype=np.float64)
        for table in self._local_expectations:
            latent_scores = np.kron(latent_scores, np.square(table))
        qubit_order = np.asarray(
            tuple(qubit for block in self.partition for qubit in block), dtype=np.int64
        )
        if not np.array_equal(qubit_order, np.arange(self.n, dtype=np.int64)):
            latent_scores = _permute_grouped_pauli_table(
                latent_scores, qubit_order, self.n
            )
        scores = np.ascontiguousarray(
            _map_clifford_squared_scores(latent_scores, self.encoder.tableau, self.n),
            dtype=np.float64,
        )
        del latent_scores
        tolerance = 1.0e-12
        if abs(float(scores[0]) - 1.0) > tolerance:
            raise RuntimeError("Structured exact identity score differs from one.")
        if float(scores.min()) < -tolerance or float(scores.max()) > 1.0 + tolerance:
            raise RuntimeError("Structured exact Pauli scores lie outside [0,1].")
        np.clip(scores, 0.0, 1.0, out=scores)
        scores[0] = 1.0
        scores.setflags(write=False)
        object.__setattr__(self, "_all_squared_scores_cache", scores)
        return scores

    def transformed_by_dagger(self, clifford: SignedClifford) -> "StructuredCEBPState":
        """Return the representation of ``clifford.U^dag rho clifford.U``."""

        if clifford.n != self.n:
            raise ValueError("Transformation dimension differs from state dimension.")
        encoder = clifford.inverse().compose(self.encoder)
        return StructuredCEBPState(self.n, self.partition, self.latent_blocks, encoder)

    def commuting_probabilities(self, observables: Sequence[str]) -> np.ndarray:
        """Reconstruct a commuting tuple's joint distribution from moments."""

        values = tuple(observables)
        q = len(values)
        if q <= 0:
            raise ValueError("At least one observable is required.")
        symplectic = _v1.Symplectic(self.n)
        if any(len(pauli) != self.n for pauli in values):
            raise ValueError("Observable length differs from state dimension.")
        if any(not symplectic.commutes(left, right) for left, right in itertools.combinations(values, 2)):
            raise ValueError("Joint observables must commute.")
        moments = np.ones(2**q, dtype=np.float64)
        for mask in range(1, 2**q):
            product = multiply_paulis(
                *(values[position] for position in range(q) if (mask >> position) & 1)
            )
            if product.phase_exponent not in (0, 2):
                raise RuntimeError("Commuting Hermitian product has an imaginary phase.")
            sign = 1.0 if product.phase_exponent == 0 else -1.0
            moments[mask] = sign * self.expectation(product.pauli)
        outcome_values = tuple(itertools.product((-1, 1), repeat=q))
        probabilities = np.empty(2**q, dtype=np.float64)
        for outcome_index, outcome in enumerate(outcome_values):
            total = 0.0
            for mask, moment in enumerate(moments):
                character = 1
                for position in range(q):
                    if (mask >> position) & 1:
                        character *= outcome[position]
                total += character * moment
            probabilities[outcome_index] = total / (2**q)
        if probabilities.min() < -1e-9:
            raise RuntimeError("Structured joint distribution has a negative probability.")
        probabilities = np.clip(probabilities, 0.0, None)
        probabilities /= probabilities.sum()
        return probabilities

    @property
    def bell_workspace_bytes(self) -> int:
        """Known Bell-only peak across construction, sampling, and scoring.

        Every modeled phase holds two full ``4**n`` tables: two float64 work
        tables during distribution construction, float64 probabilities plus
        int64 counts during batched sampling, or retained counts plus int64
        score sums during the later transform. Runtime allocator scratch is
        covered by the higher-level enumeration-workspace safety factor.
        """

        return 2 * (4**self.n) * np.dtype(np.float64).itemsize

    @property
    def bell_workspace_breakdown(self) -> Tuple[Tuple[str, int], ...]:
        """Return auditable known peaks for the three full-table Bell phases."""

        table_bytes = (4**self.n) * np.dtype(np.float64).itemsize
        return (
            ("distribution_two_float64_tables", 2 * table_bytes),
            ("sampling_probability_plus_int64_counts", 2 * table_bytes),
            ("score_retained_counts_plus_int64_sums", 2 * table_bytes),
        )

    def bell_probabilities_with_diagnostics(
        self, *, max_workspace_bytes: Optional[int] = None
    ) -> Tuple[np.ndarray, Tuple[Tuple[str, float], ...]]:
        """Build the exact Bell table through numeric moments and a tableau map."""

        required = self.bell_workspace_bytes
        if max_workspace_bytes is not None and required > int(max_workspace_bytes):
            raise ValueError(
                "structured Bell workspace exceeds max_workspace_bytes "
                f"(required={required}, limit={int(max_workspace_bytes)})."
            )
        score_started = time.perf_counter()
        # Bell construction mutates its score workspace, so retain the cached
        # exact table and copy it once for the character transform.
        scores = np.array(self.all_squared_pauli_scores(), copy=True)
        score_elapsed = time.perf_counter() - score_started

        transform_started = time.perf_counter()
        _bell_character_transform_inplace(scores, self.n)
        # Bell samplers use little-endian base-four outcome indices.
        probabilities = scores.reshape((4,) * self.n).reshape(-1, order="F")
        del scores
        transform_elapsed = time.perf_counter() - transform_started
        if probabilities.min() < -1e-9:
            raise RuntimeError("Structured Bell distribution has negative mass.")
        np.maximum(probabilities, 0.0, out=probabilities)
        probabilities /= probabilities.sum()
        probabilities.setflags(write=False)
        return probabilities, (
            ("exact_squared_score_table", score_elapsed),
            ("bell_probability_transform", transform_elapsed),
        )

    def bell_probabilities(
        self, *, max_workspace_bytes: Optional[int] = None
    ) -> np.ndarray:
        """Exact product-Bell outcome distribution with bounded numeric workspace."""

        probabilities, _diagnostics = self.bell_probabilities_with_diagnostics(
            max_workspace_bytes=max_workspace_bytes
        )
        return probabilities

    def sample_bell(
        self,
        rounds: int,
        seed: Optional[int],
        *,
        max_workspace_bytes: Optional[int] = None,
    ) -> np.ndarray:
        probabilities = self.bell_probabilities(
            max_workspace_bytes=max_workspace_bytes
        )
        sampled = np.random.default_rng(seed).choice(probabilities.size, size=rounds, p=probabilities)
        outcomes = np.empty((self.n, rounds, 3), dtype=np.int8)
        multiplier = 1
        table = np.asarray(_v1._BELL_EIG_TABLE, dtype=np.int8)
        for qubit in range(self.n):
            categories = (sampled // multiplier) % 4
            outcomes[qubit] = table[categories]
            multiplier *= 4
        return outcomes

    def sample_bell_counts(
        self,
        rounds: int,
        seed: Optional[int],
        *,
        max_workspace_bytes: Optional[int] = None,
    ) -> Tuple[np.ndarray, Tuple[Tuple[str, float], ...]]:
        """Sample a sufficient Bell-category histogram without raw shots."""

        probabilities, diagnostics = self.bell_probabilities_with_diagnostics(
            max_workspace_bytes=max_workspace_bytes
        )
        sample_started = time.perf_counter()
        counts = np.random.default_rng(seed).multinomial(int(rounds), probabilities)
        del probabilities
        counts = np.asarray(counts, dtype=np.int64)
        counts.setflags(write=False)
        return counts, diagnostics + (
            ("bell_category_sampling", time.perf_counter() - sample_started),
        )

    def materialize_dense_debug(self, *, max_qubits: int = 8) -> qt.Qobj:
        if self.n > int(max_qubits):
            raise ValueError("dense oracle debug resource guard exceeded.")
        all_kets = all(state.isket for state in self.latent_blocks)
        factors = (
            self.latent_blocks
            if all_kets
            else tuple(state * state.dag() if state.isket else state for state in self.latent_blocks)
        )
        product = qt.tensor(factors) if len(factors) > 1 else factors[0]
        unitary = self.encoder.materialize_dense_debug(max_qubits=max_qubits)
        Uq = qt.Qobj(unitary, dims=[[2] * self.n, [2] * self.n])
        if product.isket:
            result = (Uq * product).unit()
            result.dims = [[2] * self.n, [1] * self.n]
            return result
        result = Uq * product * Uq.dag()
        result = 0.5 * (result + result.dag())
        result = result / result.tr()
        result.dims = [[2] * self.n, [2] * self.n]
        return result


def structured_bell_probabilities_from_dense(state: qt.Qobj, n: int) -> np.ndarray:
    """Small-n reference helper used only by equivalence tests."""

    return np.asarray(_v1._bell_outcome_probabilities(state, n), dtype=float)


__all__ = [
    "SignedClifford",
    "SignedPauli",
    "StreamingGF2Basis",
    "StructuredCEBPState",
    "bell_score_sums_from_counts",
    "bell_score_sums_from_outcomes",
    "multiply_paulis",
    "pauli_index_to_string",
    "pauli_index_to_symplectic_int",
    "pauli_string_to_index",
    "symplectic_int_to_pauli_index",
    "threshold_span_reduction",
]
