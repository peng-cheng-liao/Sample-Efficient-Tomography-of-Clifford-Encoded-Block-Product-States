from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union
import numpy as np
import qutip as qt

# =============================================================================
# Constants / Types
# =============================================================================

# Representation boundary: states and partial traces are QuTiP Qobj objects;
# GF(2) tableaux are uint8 NumPy arrays; Clifford matrices are dense complex
# NumPy arrays; and measurement records are NumPy arrays or small typed tuples.

Axis = Literal["X", "Y", "Z"]
PauliSpec = Union[str, Iterable[Tuple[int, Axis]]]
RngLike = Union[int, np.random.Generator, None]
Ranking = List[Tuple[float, str]]  # [(score, "IXYZ..."), ...] sorted high->low

PAULI_CHARS = "IXYZ"
AXIS_TO_K = {"X": 0, "Y": 1, "Z": 2}

# Single-qubit matrices (NumPy)
I2 = np.eye(2, dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)

# Single-qubit matrices (QuTiP)
_QT_PAULI_1Q = {"I": qt.qeye(2), "X": qt.sigmax(), "Y": qt.sigmay(), "Z": qt.sigmaz()}

# Rows map a vectorized one-qubit density-matrix factor to Tr(rho P),
# with Pauli order I,X,Y,Z and row/column matrix indices interleaved.
_PAULI_TRACE_TRANSFORM = np.stack(
    [I2.T, X2.T, qt.sigmay().full().T, Z2.T]
).reshape(4, 4)

# Rows are local Bell outcomes (Phi+, Phi-, Psi+, Psi-); columns are I,X,Y,Z.
_BELL_PAULI_TRANSFORM = np.array(
    [
        [1, +1, -1, +1],
        [1, -1, +1, +1],
        [1, +1, +1, -1],
        [1, -1, -1, -1],
    ],
    dtype=float,
)
_BELL_EIG_TABLE = _BELL_PAULI_TRANSFORM[:, 1:].astype(np.int8)
_MAX_EXACT_MIXED_BELL_QUBITS = 11


def _rng(seed: RngLike = None) -> np.random.Generator:
    """Return an explicit generator, preserving a supplied generator's stream."""
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def _child_seeds(seed: Optional[int], count: int) -> List[int]:
    """Deterministically split one seed into independent child streams."""
    return [int(s.generate_state(1, dtype=np.uint64)[0]) for s in np.random.SeedSequence(seed).spawn(count)]


# =============================================================================
# QuTiP helpers
# =============================================================================


def _ensure_qubit_dims(obj: qt.Qobj, n: int) -> qt.Qobj:
    """Ensure obj has qubit dims for kets/operators."""
    if not isinstance(obj, qt.Qobj):
        raise TypeError("Expected a qutip.Qobj.")

    if obj.isket:
        if obj.shape != (2 ** n, 1):
            raise ValueError(f"Ket must have shape {(2 ** n, 1)} for n={n}.")
        if obj.dims != [[2] * n, [1] * n]:
            obj = qt.Qobj(obj.full(), dims=[[2] * n, [1] * n])
        return obj

    if obj.shape != (2 ** n, 2 ** n):
        raise ValueError(f"Operator must be {2 ** n}x{2 ** n} for n={n}.")
    if obj.dims != [[2] * n, [2] * n]:
        obj = qt.Qobj(obj.full(), dims=[[2] * n, [2] * n])
    return obj


def _as_density(rho: qt.Qobj, n: int) -> qt.Qobj:
    """Normalize dims + convert ket->density; enforce Hermitian trace-1."""
    rho = _ensure_qubit_dims(rho, n)
    if rho.isket:
        rho = rho * rho.dag()
    if not (rho.isherm and abs(rho.tr() - 1.0) < 1e-8):
        raise ValueError("rho must be Hermitian with trace 1.")
    return rho


def _parse_pauli_spec(n: int, P: PauliSpec) -> List[Tuple[int, Axis]]:
    """Return support list [(i, 'X'/'Y'/'Z'), ...] (no 'I')."""
    if isinstance(P, str):
        if len(P) != n:
            raise ValueError(f"Pauli string length must be n={n}.")
        out: List[Tuple[int, Axis]] = []
        for i, ch in enumerate(P):
            if ch == "I":
                continue
            if ch not in ("X", "Y", "Z"):
                raise ValueError("Pauli string must use only I,X,Y,Z.")
            out.append((i, ch))  # type: ignore[arg-type]
        return out

    out: List[Tuple[int, Axis]] = []
    for i, a in P:
        if not (0 <= int(i) < n):
            raise ValueError("Index out of range in PauliSpec.")
        if a not in ("X", "Y", "Z"):
            raise ValueError('Axis must be one of "X","Y","Z".')
        out.append((int(i), a))
    return out


def _qutip_pauli_op(n: int, P: PauliSpec) -> qt.Qobj:
    sup = dict(_parse_pauli_spec(n, P))
    return qt.tensor([_QT_PAULI_1Q[sup.get(i, "I")] for i in range(n)])


def _single_qubit_pauli_basis_change(axis: str) -> np.ndarray:
    """Return the unitary whose columns are the +/- Pauli eigenvectors."""
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    if axis == "Z":
        return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    if axis == "X":
        return inv_sqrt2 * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex)
    if axis == "Y":
        return inv_sqrt2 * np.array([[1.0, 1.0], [1.0j, -1.0j]], dtype=complex)
    raise ValueError("axis must be one of 'X', 'Y', 'Z'.")


# =============================================================================
# Bell sampling
# =============================================================================


def _bell_outcome_probabilities(rho: qt.Qobj, n: int) -> np.ndarray:
    """Exact product-Bell probabilities using O(4^n) working storage.

    The Pauli-score tensor contains ``Tr(rho P)^2`` in local I,X,Y,Z order.
    Applying the factorized Bell-projector transform on each tensor axis gives
    probabilities in the existing base-4 outcome order, with qubit 0 as the
    least-significant digit.
    """
    rho = _as_density(rho, n)
    rho_tensor = rho.full().reshape([2] * (2 * n))
    interleaved_axes = [axis for i in range(n) for axis in (i, n + i)]
    pauli_tensor = np.transpose(rho_tensor, interleaved_axes).reshape([4] * n)

    for axis in range(n):
        pauli_tensor = np.tensordot(
            _PAULI_TRACE_TRANSFORM, pauli_tensor, axes=([1], [axis])
        )
        pauli_tensor = np.moveaxis(pauli_tensor, 0, axis)

    if np.max(np.abs(pauli_tensor.imag)) > 1e-8:
        raise RuntimeError("Numerical issue: Pauli expectations have a non-negligible imaginary part.")
    probabilities = np.square(pauli_tensor.real)
    for axis in range(n):
        probabilities = np.tensordot(
            _BELL_PAULI_TRANSFORM, probabilities, axes=([1], [axis])
        )
        probabilities = np.moveaxis(probabilities, 0, axis)

    probabilities = probabilities.reshape(-1, order="F") / float(4 ** n)
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(probabilities.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("Numerical issue: total Bell probability is non-positive.")
    return probabilities / total


def bell_sampling(n: int, M: int, rho: qt.Qobj, seed: RngLike = None) -> np.ndarray:
    """
    Joint product-Bell measurement on (rho ⊗ rho) with correlations preserved.
    Output B has shape (n, M, 3) with columns [X,Y,Z] eigenvalues in {-1,+1}.

    The exact factorized Pauli transform stores O(4^n) scores/probabilities;
    it never constructs the 4^n-by-4^n two-copy density matrix. Runtime and
    memory are still exponential, so this remains a small-system simulator.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if M <= 0:
        raise ValueError("M must be positive.")
    if n > _MAX_EXACT_MIXED_BELL_QUBITS:
        raise ValueError(
            "This exact mixed-state Bell sampler stores O(4^n) arrays; "
            f"n>{_MAX_EXACT_MIXED_BELL_QUBITS} exceeds its simulation guard."
        )
    rng = _rng(seed)
    probs = _bell_outcome_probabilities(rho, n)
    num_out = probs.size
    pow4 = 4 ** np.arange(n, dtype=np.int64)
    sampled = rng.choice(num_out, size=M, p=probs)

    B = np.empty((n, M, 3), dtype=np.int8)
    for i in range(n):
        s_i = (sampled // int(pow4[i])) % 4
        B[i, :, :] = _BELL_EIG_TABLE[s_i, :]
    return B


def _pure_bell_outcome_probabilities(n: int, psi: qt.Qobj) -> np.ndarray:
    """Exact product-Bell probabilities for a pure ket in base-4 outcome order."""
    if n <= 0:
        raise ValueError("n must be positive.")

    # --- validate pure ket ---
    if not isinstance(psi, qt.Qobj):
        raise TypeError("psi must be a qutip.Qobj.")
    if not psi.isket:
        raise ValueError("psi must be a ket (pure state).")
    dim = psi.shape[0]
    if dim != 2 ** n:
        raise ValueError(f"psi dimension mismatch: got {dim}, expected {2 ** n} for n={n}.")
    # normalize if needed (cheap)
    norm = float((psi.dag() * psi).real)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("psi has non-finite or non-positive norm.")
    if abs(norm - 1.0) > 1e-10:
        psi = psi / np.sqrt(norm)

    # --- build two-copy ket |psi> ⊗ |psi> as a numpy vector ---
    v = psi.full().reshape(-1)  # (2^n,)
    v2 = np.kron(v, v)  # (4^n,) corresponds to qubit order [0..n-1, n..2n-1]

    # --- interleave copies so pairs (i, i+n) become contiguous: [0,n,1,n+1,...] ---
    v2 = v2.reshape([2] * (2 * n))
    perm = [p for i in range(n) for p in (i, n + i)]
    v2 = np.transpose(v2, axes=perm).reshape(-1)  # still length 4^n

    # --- Bell basis transform on each 2-qubit pair ---
    # Bell kets in computational basis order |00>,|01>,|10>,|11>:
    # Φ+ = (|00>+|11>)/√2, Φ- = (|00>-|11>)/√2, Ψ+ = (|01>+|10>)/√2, Ψ- = (|01>-|10>)/√2
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    U = np.array(
        [
            [inv_sqrt2, inv_sqrt2, 0.0, 0.0],
            [0.0, 0.0, inv_sqrt2, inv_sqrt2],
            [0.0, 0.0, inv_sqrt2, -inv_sqrt2],
            [inv_sqrt2, -inv_sqrt2, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    # Columns are Bell kets in computational coordinates, so computational->Bell is U^\dagger
    Udag = U.conj().T

    # Treat each interleaved pair as a qudit of dimension 4: reshape to (4,)*n
    state = v2.reshape([4] * n)
    for ax in range(n):
        # Apply Udag on axis ax: new_state[i_ax,...] = sum_j Udag[i_ax,j] * state[j,...]
        state = np.tensordot(Udag, state, axes=([1], [ax]))  # puts new axis at front
        state = np.moveaxis(state, 0, ax)  # move it back to position ax

    amps = state.reshape(-1, order="F")  # length 4^n, in Bell-outcome ordering (base-4 digits per pair)
    probs = (amps.conj() * amps).real  # |amp|^2

    probs = np.clip(probs, 0.0, None)
    ssum = probs.sum()
    if not np.isfinite(ssum) or ssum <= 0:
        raise RuntimeError("Numerical issue: total probability is non-positive.")
    probs /= ssum
    return probs


def bell_sampling_pure(n: int, M: int, psi: qt.Qobj, seed: RngLike = None) -> np.ndarray:
    """
    Joint product-Bell measurement on (|psi><psi| ⊗ |psi><psi|), assuming input is PURE.
    Returns B with shape (n, M, 3), columns [X, Y, Z] eigenvalues in {-1, +1}.

    Memory improvement vs bell_sampling():
      - avoids building rho (2^n x 2^n) and the two-copy density matrix,
      - works directly with the 2-copy ket (length 4^n) and a Bell-basis change.

    NOTE: Still exponential in n due to 4^n Bell outcomes (state length 4^n).
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if M <= 0:
        raise ValueError("M must be positive.")
    rng = _rng(seed)
    probs = _pure_bell_outcome_probabilities(n, psi)

    # Sample Bell outcome index in [0, 4^n)
    sampled = rng.choice(probs.size, size=M, p=probs)

    pow4 = 4 ** np.arange(n, dtype=np.int64)
    B = np.empty((n, M, 3), dtype=np.int8)
    for i in range(n):
        s_i = (sampled // int(pow4[i])) % 4
        B[i, :, :] = _BELL_EIG_TABLE[s_i, :]
    return B


# =============================================================================
# Ranking utilities
# =============================================================================


def rank_all_sP_from_B(B: np.ndarray, weight: Optional[int] = None, include_identity: bool = False) -> Ranking:
    """
    Estimate s(P)=Tr(rho P)^2 for all Pauli strings (or exact weight),
    using Bell-sampling outcomes B, then sort high->low.
    """
    if not isinstance(B, np.ndarray) or B.ndim != 3 or B.shape[2] != 3:
        raise ValueError("B must have shape (n, M, 3).")
    n, M, _ = B.shape
    if M <= 0:
        raise ValueError("M must be positive.")
    if weight is not None and not (0 <= weight <= n):
        raise ValueError("weight must satisfy 0 <= weight <= n.")

    ones = np.ones(M, dtype=np.int8)
    per_qubit = [{"I": ones, "X": B[i, :, 0], "Y": B[i, :, 1], "Z": B[i, :, 2]} for i in range(n)]

    results: Ranking = []

    def add_string(tup: Tuple[str, ...], support: Optional[Tuple[int, ...]] = None) -> None:
        P = "".join(tup)
        if not include_identity and all(ch == "I" for ch in P):
            return
        prod = ones.astype(np.int16, copy=True)
        if support is None:
            for i, ch in enumerate(tup):
                if ch != "I":
                    prod *= per_qubit[i][ch]
        else:
            for i in support:
                prod *= per_qubit[i][tup[i]]
        results.append((float(prod.mean()), P))

    if weight is None:
        for tup in itertools.product(("I", "X", "Y", "Z"), repeat=n):
            add_string(tup)
    else:
        if weight == 0:
            if include_identity:
                results.append((1.0, "I" * n))
        else:
            for support in itertools.combinations(range(n), weight):
                for letters in itertools.product(("X", "Y", "Z"), repeat=weight):
                    P_list = ["I"] * n
                    for idx, ch in zip(support, letters):
                        P_list[idx] = ch
                    add_string(tuple(P_list), support=tuple(support))

    results.sort(key=lambda x: x[0], reverse=True)
    return results


def paulis_above_threshold_from_B(
        B: np.ndarray,
        threshold: float,
        *,
        include_identity: bool = False,
) -> Ranking:
    """Return only empirical Pauli scores at least ``threshold``.

    Enumeration remains exponential, but the active recovery path avoids
    materializing and sorting the below-threshold portion of the ranking.
    Stable sorting preserves the order used by ``rank_all_sP_from_B`` on ties.
    """
    if not isinstance(B, np.ndarray) or B.ndim != 3 or B.shape[2] != 3:
        raise ValueError("B must have shape (n, M, 3).")
    n, M, _ = B.shape
    if M <= 0:
        raise ValueError("M must be positive.")
    ones = np.ones(M, dtype=np.int8)
    results: Ranking = []
    for tup in itertools.product(PAULI_CHARS, repeat=n):
        if not include_identity and all(ch == "I" for ch in tup):
            continue
        prod = ones.astype(np.int16, copy=True)
        for i, ch in enumerate(tup):
            if ch != "I":
                prod *= B[i, :, AXIS_TO_K[ch]]
        score = float(prod.mean())
        if score >= threshold:
            results.append((score, "".join(tup)))
    results.sort(key=lambda x: x[0], reverse=True)
    return results


# =============================================================================
# Symplectic (phase-free) Pauli utilities
# =============================================================================


@dataclass(frozen=True)
class Symplectic:
    n: int

    def to_int(self, P: str) -> int:
        x = 0
        z = 0
        for i, ch in enumerate(P):
            if ch == "X":
                x |= 1 << i
            elif ch == "Z":
                z |= 1 << i
            elif ch == "Y":
                x |= 1 << i
                z |= 1 << i
        return x | (z << self.n)

    def to_str(self, v: int) -> str:
        x = v & ((1 << self.n) - 1)
        z = v >> self.n
        out: List[str] = []
        for i in range(self.n):
            xi = (x >> i) & 1
            zi = (z >> i) & 1
            if (xi, zi) == (0, 0):
                out.append("I")
            elif (xi, zi) == (1, 0):
                out.append("X")
            elif (xi, zi) == (0, 1):
                out.append("Z")
            else:
                out.append("Y")
        return "".join(out)

    def commutes_int(self, v1: int, v2: int) -> bool:
        mask = (1 << self.n) - 1
        x1, z1 = v1 & mask, v1 >> self.n
        x2, z2 = v2 & mask, v2 >> self.n
        parity = ((x1 & z2).bit_count() + (z1 & x2).bit_count()) & 1
        return parity == 0

    def anticommutes_int(self, v1: int, v2: int) -> bool:
        return not self.commutes_int(v1, v2)

    def commutes(self, P: str, Q: str) -> bool:
        return self.commutes_int(self.to_int(P), self.to_int(Q))

    def anticommutes(self, P: str, Q: str) -> bool:
        return not self.commutes(P, Q)


class GF2Basis:
    """Maintain a GF(2) span basis for integers interpreted as bit-vectors."""

    def __init__(self) -> None:
        self.pivots: Dict[int, int] = {}

    @staticmethod
    def _msb(v: int) -> int:
        return v.bit_length() - 1

    def reduce(self, v: int) -> int:
        while v:
            p = self._msb(v)
            row = self.pivots.get(p)
            if row is None:
                break
            v ^= row
        return v

    def contains(self, v: int) -> bool:
        return self.reduce(v) == 0

    def add(self, v: int) -> None:
        v = self.reduce(v)
        if v:
            self.pivots[self._msb(v)] = v


def pauli_to_symplectic_col(P: str) -> np.ndarray:
    """
    P in {'I','X','Y','Z'}^n -> [a_1..a_n | b_1..b_n]^T in GF(2)^(2n), where Y -> (1,1).
    """
    n = len(P)
    a = np.zeros(n, dtype=np.uint8)
    b = np.zeros(n, dtype=np.uint8)

    for i, ch in enumerate(P):
        if ch == "I":
            continue
        if ch == "X":
            a[i] = 1
        elif ch == "Z":
            b[i] = 1
        elif ch == "Y":
            a[i] = 1
            b[i] = 1
        else:
            raise ValueError(f"Invalid Pauli character '{ch}' in P='{P}'.")
    return np.concatenate([a, b])  # shape (2n,)


def is_symplectic(F: np.ndarray) -> bool:
    """Check F^T J F = J over GF(2), with J = [[0,I],[I,0]]."""
    F = (F % 2).astype(np.uint8)
    n = F.shape[0] // 2
    if F.shape != (2 * n, 2 * n):
        return False
    J = np.block(
        [
            [np.zeros((n, n), dtype=np.uint8), np.eye(n, dtype=np.uint8)],
            [np.eye(n, dtype=np.uint8), np.zeros((n, n), dtype=np.uint8)],
        ]
    )
    return np.array_equal((F.T @ J @ F) % 2, J)


# =============================================================================
# Tableau synthesis (Clifford from symplectic tableau)
# =============================================================================

# Convention: Pauli vectors are columns ``a=[x|z]`` and tableau columns are
# ``[X_0,...,X_{n-1}|Z_0,...,Z_{n-1}]``.  ``apply_*_right`` updates a tableau
# by right composition.  Thus a forward random walk returns ``(F,U)`` with
# ``U† P(a) U = ±P(Fa)``.  In contrast, synthesis emits gates that reduce a
# prescribed F to identity; its dense U obeys ``U P(a) U† = ±P(Fa)``.  Recovery
# consequently uses ``U† g U`` to map a prescribed stabilizer to canonical Z.


def apply_cnot_right(F: np.ndarray, c: int, t: int) -> np.ndarray:
    """Right-multiply tableau by CNOT(c->t) update rule on columns."""
    n = F.shape[0] // 2
    x, z = F[:n, :], F[n:, :]
    x[t, :] ^= x[c, :]
    z[c, :] ^= z[t, :]
    return F


def apply_h_right(F: np.ndarray, j: int) -> np.ndarray:
    """Right-multiply tableau by H(j) update rule on columns."""
    n = F.shape[0] // 2
    F[j, :], F[n + j, :] = F[n + j, :].copy(), F[j, :].copy()
    return F


def apply_s_right(F: np.ndarray, j: int) -> np.ndarray:
    """Right-multiply tableau by S(j) update rule on columns."""
    n = F.shape[0] // 2
    F[n + j, :] ^= F[j, :]
    return F


def synthesize_clifford_from_tableau(F_in: np.ndarray) -> List[Tuple]:
    """
    Input  : F_in (2n x 2n, uint8 in {0,1}), columns [X1..Xn | Z1..Zn]
    Output : gate list [('H',j), ('S',j), ('CNOT',c,t)] in the order they should be applied.
    """
    F = (F_in.copy() % 2).astype(np.uint8)
    n = F.shape[0] // 2
    if F.shape != (2 * n, 2 * n):
        raise ValueError("F must be 2n x 2n.")
    if not is_symplectic(F):
        raise ValueError("Input is not symplectic (F^T J F != J).")

    gates: List[Tuple] = []

    def H(j: int, record: bool = True) -> None:
        apply_h_right(F, j)
        if record:
            gates.append(("H", j))

    def S(j: int) -> None:
        apply_s_right(F, j)
        gates.append(("S", j))

    def CNOT(c: int, t: int) -> None:
        if c == t:
            return
        apply_cnot_right(F, c, t)
        gates.append(("CNOT", c, t))

    def rank_gf2(M: np.ndarray) -> int:
        M = (M % 2).copy().astype(np.uint8)
        r, i, j = 0, 0, 0
        m, k = M.shape
        while i < m and j < k:
            p = next((t for t in range(i, m) if M[t, j]), None)
            if p is None:
                j += 1
                continue
            if p != i:
                M[[i, p]] = M[[p, i]]
            for t in range(i + 1, m):
                if M[t, j]:
                    M[t, :] ^= M[i, :]
            i += 1
            j += 1
            r += 1
        return r

    # Phase 0: make X-block full rank using H flips as needed
    flipped: set[int] = set()
    while rank_gf2(F[:n, :n]) < n:
        improved = False
        tried: set[int] = set()

        for j in range(n):
            if j in tried:
                continue
            F_backup = F.copy()
            prev_rank = rank_gf2(F_backup[:n, :n])

            H(j, record=False)
            new_rank = rank_gf2(F[:n, :n])

            if new_rank > prev_rank:
                gates.append(("H", j))
                flipped.add(j)
                improved = True
                break

            F[...] = F_backup
            tried.add(j)

        if not improved:
            for j in range(n):
                if j not in flipped:
                    H(j, record=True)
                    flipped.add(j)
                    if rank_gf2(F[:n, :n]) == n:
                        improved = True
                        break

        if not improved and rank_gf2(F[:n, :n]) < n:
            raise RuntimeError("Could not make X-block full rank via H flips.")

    # Phase 1: reduce A (X-part of X-columns) to identity with CNOTs
    for i in range(n):
        A = F[:n, :n]
        if A[i, i] == 0:
            found = False
            for j in range(i + 1, n):
                if A[j, i] == 1:
                    CNOT(j, i)
                    found = True
                    break
            if not found:
                raise RuntimeError(f"No pivot to set A[{i},{i}]=1.")
        for j in range(n):
            if j != i and F[:n, :n][j, i] == 1:
                CNOT(i, j)

    # Phase 2: clean Z-leakage in X-columns (make each colX(i) = (e_i | 0))
    for i in range(n):
        cx = i
        if F[n + i, cx] == 1:
            S(i)
        for j in range(n):
            if j != i and F[n + j, cx] == 1:
                H(j)
                CNOT(i, j)
                H(j)

    # Phase 3: zero x-part in Z columns
    for i in range(n):
        cz = n + i
        for j in range(n):
            if j == i and F[j, cz] == 1:
                H(i)
                S(i)
                H(i)
            elif j != i and F[j, cz] == 1:
                H(j)
                CNOT(j, i)
                H(j)

    return gates


def _kron_n(*ops: np.ndarray) -> np.ndarray:
    out = np.array([[1]], dtype=complex)
    for op in ops:
        out = np.kron(out, op)
    return out


def _single_qubit_op(n: int, j: int, M: np.ndarray) -> np.ndarray:
    ops = [M if q == j else I2 for q in range(n)]
    return _kron_n(*ops)


def _two_qubit_cnot(n: int, c: int, t: int) -> np.ndarray:
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)

    term0_ops, term1_ops = [], []
    for q in range(n):
        if q == c:
            term0_ops.append(P0)
            term1_ops.append(P1)
        elif q == t:
            term0_ops.append(I2)
            term1_ops.append(X2)
        else:
            term0_ops.append(I2)
            term1_ops.append(I2)
    return _kron_n(*term0_ops) + _kron_n(*term1_ops)


def random_clifford_gate(
        n: int,
        steps: int | None = None,
        seed: int | None = None,
) -> tuple[list[tuple], np.ndarray, np.ndarray]:
    """
    Generate a Clifford random walk over {H, S, CNOT}. It returns the circuit
    unitary U, not U†. This is not
    a uniform sampler from the n-qubit Clifford group.

    Returns:
      - gates: list of gate tuples [('H', j), ('S', j), ('CNOT', c, t), ...]
      - F:     resulting symplectic matrix (2n x 2n) over GF(2), column tableau convention
      - U:     unitary (2^n x 2^n) built from `gates`

    Notes:
      - This uses the SAME update rules as apply_h_right/apply_s_right/apply_cnot_right.
      - For n=1 only H and S are sampled, since CNOT is not admissible.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if steps is None:
        steps = 5 * n
    if steps <= 0:
        raise ValueError("steps must be positive.")

    rng = np.random.default_rng(seed)

    F = np.eye(2 * n, dtype=np.uint8)
    gates: list[tuple] = []

    for _ in range(steps):
        g = int(rng.integers(0, 2 if n == 1 else 3))
        if g == 0:
            j = int(rng.integers(0, n))
            apply_h_right(F, j)
            gates.append(("H", j))
        elif g == 1:
            j = int(rng.integers(0, n))
            apply_s_right(F, j)
            gates.append(("S", j))
        else:
            c = int(rng.integers(0, n))
            t = int(rng.integers(0, n - 1))
            if t >= c:
                t += 1
            apply_cnot_right(F, c, t)
            gates.append(("CNOT", c, t))

    U = gates_to_unitary(gates, n)
    return gates, F, U


# =============================================================================
# Single-qubit tomography
# =============================================================================


def pauli_shot_counts_from_state(
        rho: np.ndarray,
        Nx: int,
        Ny: int,
        Nz: int,
        *,
        seed: RngLike = None,
) -> Tuple[int, int, int, int, int, int]:
    """
    Simulate Pauli (X,Y,Z) projective measurement counts on a single-qubit state rho.

    Args:
        rho: 2x2 density matrix (complex). Should be Hermitian, PSD, trace 1 (approximately).
        Nx, Ny, Nz: number of shots along X, Y, Z axes.
        seed: RNG seed.

    Returns:
        ``(n_xp, n_xm, n_yp, n_ym, n_zp, n_zm)`` in that exact order.
    """
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (2, 2):
        raise ValueError("rho must be a 2x2 matrix.")
    if any(N <= 0 for N in (Nx, Ny, Nz)):
        raise ValueError("Nx, Ny, Nz must be positive integers.")

    # Pauli matrices
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Expectation values <sigma_m> = Tr(rho sigma_m) are real for physical rho
    ex = float(np.real_if_close(np.trace(rho @ X)))
    ey = float(np.real_if_close(np.trace(rho @ Y)))
    ez = float(np.real_if_close(np.trace(rho @ Z)))

    # Probabilities p_{m,+} = (1 + <sigma_m>)/2
    px = 0.5 * (1.0 + ex)
    py = 0.5 * (1.0 + ey)
    pz = 0.5 * (1.0 + ez)

    # Clip for numerical safety
    px = float(np.clip(px, 0.0, 1.0))
    py = float(np.clip(py, 0.0, 1.0))
    pz = float(np.clip(pz, 0.0, 1.0))

    rng = _rng(seed)

    n_xp = int(rng.binomial(Nx, px))
    n_xm = Nx - n_xp
    n_yp = int(rng.binomial(Ny, py))
    n_ym = Ny - n_yp
    n_zp = int(rng.binomial(Nz, pz))
    n_zm = Nz - n_zp

    return n_xp, n_xm, n_yp, n_ym, n_zp, n_zm


def linear_inversion_qubit_from_pauli_counts(
        counts,
        *,
        enforce_physical: bool = True,
) -> qt.Qobj:
    """
    Direct (non-MLE) reconstruction via linear inversion, returning a QuTiP Qobj.

        r_m = (n_{m,+} - n_{m,-}) / (n_{m,+} + n_{m,-})
        rho = 1/2 (I + r_x X + r_y Y + r_z Z)

    If enforce_physical=True, projects Bloch vector onto the Bloch ball (||r||<=1).
    This is a simple physicality fix, not MLE.

    Returns:
        qt.Qobj density matrix with dims=[[2],[2]].
    """
    [n_xp, n_xm, n_yp, n_ym, n_zp, n_zm] = counts
    if any((not isinstance(c, (int, np.integer)) or c < 0) for c in counts):
        raise ValueError("All counts must be non-negative integers.")

    Nx, Ny, Nz = n_xp + n_xm, n_yp + n_ym, n_zp + n_zm
    if Nx == 0 or Ny == 0 or Nz == 0:
        raise ValueError("Each axis must have at least one shot (Nx,Ny,Nz > 0).")

    rx = (n_xp - n_xm) / Nx
    ry = (n_yp - n_ym) / Ny
    rz = (n_zp - n_zm) / Nz
    r = np.array([rx, ry, rz], dtype=float)

    if enforce_physical:
        nr = float(np.linalg.norm(r))
        if nr > 1.0:
            r /= nr

    rx, ry, rz = map(float, r)

    I = qt.qeye(2)
    X = qt.sigmax()
    Y = qt.sigmay()
    Z = qt.sigmaz()

    rho = 0.5 * (I + rx * X + ry * Y + rz * Z)

    # Numerical cleanup
    rho = 0.5 * (rho + rho.dag())
    rho = rho / rho.tr()

    # Ensure dims are explicitly single-qubit operator dims
    rho.dims = [[2], [2]]
    return rho


def single_qubit_tomography(
        rho: qt.Qobj,
        Nx: int,
        Ny: int,
        Nz: int,
        seed: RngLike = None,
) -> qt.Qobj:
    """Linear-inversion single-qubit tomography with an explicit RNG stream."""
    rho = rho.full()
    counts = pauli_shot_counts_from_state(rho, Nx, Ny, Nz, seed=seed)
    return linear_inversion_qubit_from_pauli_counts(counts)


# =============================================================================
# State helpers / high-level tomography pipeline additions
# =============================================================================


def _state_to_qobj(state: Union[np.ndarray, qt.Qobj], n: int) -> qt.Qobj:
    """
    Accept either a NumPy ket/density matrix or a QuTiP Qobj and return a
    normalized n-qubit Qobj with explicit qubit dims.
    """
    if isinstance(state, qt.Qobj):
        obj = _ensure_qubit_dims(state, n)
        if obj.isket:
            norm = complex(obj.norm())
            if abs(norm) == 0:
                raise ValueError("State ket must have nonzero norm.")
            if abs(abs(norm) - 1.0) > 1e-10:
                obj = obj.unit()
            return obj
        return _as_density(obj, n)

    arr = np.asarray(state, dtype=complex)
    dim = 2 ** n
    if arr.shape == (dim,):
        obj = qt.Qobj(arr.reshape((dim, 1)), dims=[[2] * n, [1] * n]).unit()
        return obj
    if arr.shape == (dim, 1):
        obj = qt.Qobj(arr, dims=[[2] * n, [1] * n]).unit()
        return obj
    if arr.shape == (dim, dim):
        return _as_density(qt.Qobj(arr, dims=[[2] * n, [2] * n]), n)
    raise ValueError(f"State must have shape ({dim},), ({dim},1), or ({dim},{dim}) for n={n}.")


def _local_product_pauli_measurement_probabilities(
        state: Union[np.ndarray, qt.Qobj],
        axes: Tuple[str, ...],
) -> np.ndarray:
    """Exact probabilities for a local product-Pauli measurement on a Qobj state."""
    n = len(axes)
    if n <= 0 or any(axis not in ("X", "Y", "Z") for axis in axes):
        raise ValueError("axes must be a nonempty tuple containing only X, Y, and Z.")
    state_q = _state_to_qobj(state, n)
    if state_q.isket:
        return _measurement_probs_in_local_pauli_basis(
            state_q.full().reshape(-1), axes
        )

    rotated = state_q.full().reshape([2] * (2 * n))
    for qubit, axis in enumerate(axes):
        basis = _single_qubit_pauli_basis_change(axis)
        rotated = np.tensordot(basis.conj().T, rotated, axes=([1], [qubit]))
        rotated = np.moveaxis(rotated, 0, qubit)
        rotated = np.tensordot(basis.T, rotated, axes=([1], [n + qubit]))
        rotated = np.moveaxis(rotated, 0, n + qubit)

    probabilities = np.diag(rotated.reshape(2 ** n, 2 ** n)).real
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(probabilities.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("Numerical issue: invalid product-Pauli measurement probabilities.")
    return probabilities / total


def _shared_single_qubit_tomography(
        state: Union[np.ndarray, qt.Qobj],
        n: int,
        M3: int,
        seed: Optional[int] = None,
        return_details: bool = False,
) -> Union[List[qt.Qobj], Dict[str, object]]:
    """Estimate every one-qubit marginal from three shared global record sets.

    ``M3`` copies are measured in each of the global X, Y, and Z product
    bases. The same M3 n-bit outcomes for an axis supply that axis's counts for
    every qubit, so this stage consumes 3*M3 physical copies in total.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if M3 <= 0:
        raise ValueError("M3 must be positive.")
    state_q = _state_to_qobj(state, n)
    axis_names = ("X", "Y", "Z")
    axis_seeds = _child_seeds(seed, 3) if seed is not None else [None] * 3
    records: Dict[str, np.ndarray] = {}

    bit_shifts = np.arange(n - 1, -1, -1, dtype=np.int64)
    for axis, axis_seed in zip(axis_names, axis_seeds):
        probabilities = _local_product_pauli_measurement_probabilities(
            state_q, (axis,) * n
        )
        outcomes = _rng(axis_seed).choice(2 ** n, size=M3, p=probabilities)
        records[axis] = (
            (outcomes[:, np.newaxis] >> bit_shifts[np.newaxis, :]) & 1
        ).astype(np.uint8)

    counts: List[Tuple[int, int, int, int, int, int]] = []
    estimates: List[qt.Qobj] = []
    for qubit in range(n):
        axis_counts: List[int] = []
        for axis in axis_names:
            minus = int(records[axis][:, qubit].sum())
            axis_counts.extend((M3 - minus, minus))
        count_tuple = (
            axis_counts[0], axis_counts[1], axis_counts[2],
            axis_counts[3], axis_counts[4], axis_counts[5],
        )
        counts.append(count_tuple)
        estimates.append(linear_inversion_qubit_from_pauli_counts(count_tuple))

    if return_details:
        return {
            "rho_estimates": estimates,
            "counts": counts,
            "records": records,
            "axis_seeds": dict(zip(axis_names, axis_seeds)),
            "shots_per_axis": M3,
            "physical_copies": 3 * M3,
        }
    return estimates


def _bell_sampling_state(
        n: int,
        M: int,
        state: Union[np.ndarray, qt.Qobj],
        seed: RngLike = None,
) -> Tuple[np.ndarray, bool]:
    """
    Dispatch Bell sampling based on whether the input state is pure (ket) or mixed.
    Returns (B, is_pure).
    """
    state_q = _state_to_qobj(state, n)
    if state_q.isket:
        return bell_sampling_pure(n=n, M=M, psi=state_q, seed=seed), True
    return bell_sampling(n=n, M=M, rho=state_q, seed=seed), False


# =============================================================================
# GF(2) helpers for tableau completion
# =============================================================================


def _gf2_rref(A: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Reduced row-echelon form over GF(2).
    Returns (R, pivots), where pivots is the list of pivot columns.
    """
    A = (A % 2).copy().astype(np.uint8)
    m, n = A.shape
    pivots: List[int] = []
    r = 0

    for c in range(n):
        pivot = None
        for i in range(r, m):
            if A[i, c]:
                pivot = i
                break
        if pivot is None:
            continue

        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]

        for i in range(m):
            if i != r and A[i, c]:
                A[i, :] ^= A[r, :]

        pivots.append(c)
        r += 1
        if r == m:
            break

    return A, pivots


def _gf2_rank(A: np.ndarray) -> int:
    return len(_gf2_rref(A)[1])


def _gf2_solve_one(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve A x = b over GF(2), returning one solution.
    Raises ValueError if inconsistent.
    """
    A = (A % 2).astype(np.uint8)
    b = (np.asarray(b, dtype=np.uint8).reshape(-1, 1) % 2)

    if A.shape[0] != b.shape[0]:
        raise ValueError("Dimension mismatch in GF(2) solve.")

    aug = np.concatenate([A, b], axis=1)
    R, pivots = _gf2_rref(aug)

    m, n1 = aug.shape
    n = n1 - 1

    for i in range(m):
        if not R[i, :n].any() and R[i, n]:
            raise ValueError("Linear system over GF(2) is inconsistent.")

    x = np.zeros(n, dtype=np.uint8)
    for r, c in enumerate(pivots):
        if c < n:
            x[c] = R[r, n]

    return x


def _gf2_nullspace_basis(A: np.ndarray) -> List[np.ndarray]:
    """
    Basis of nullspace {x : A x = 0} over GF(2).
    Returned as a list of 1D uint8 vectors.
    """
    A = (A % 2).astype(np.uint8)
    m, n = A.shape
    R, pivots = _gf2_rref(A)
    free_cols = [j for j in range(n) if j not in pivots]

    basis: List[np.ndarray] = []
    for f in free_cols:
        x = np.zeros(n, dtype=np.uint8)
        x[f] = 1
        for r, c in enumerate(pivots):
            x[c] = R[r, f]
        basis.append(x)

    return basis


# =============================================================================
# Symplectic completion helpers
# =============================================================================


def complete_isotropic_to_symplectic(Zcols: np.ndarray) -> np.ndarray:
    """
    Complete an isotropic set of commuting independent Pauli columns into a full
    symplectic tableau [X | Z], preserving the prescribed Z columns.
    """
    Zcols = np.asarray(Zcols, dtype=np.uint8) % 2

    if Zcols.ndim != 2:
        raise ValueError("Zcols must be a 2D array of shape (2n, t).")

    two_n, t = Zcols.shape
    if two_n % 2 != 0:
        raise ValueError("Zcols must have an even number of rows, i.e. shape (2n, t).")

    n = two_n // 2
    if t > n:
        raise ValueError(f"Need t <= n, but got t={t}, n={n}.")

    if t == 0:
        return np.eye(2 * n, dtype=np.uint8)

    if _gf2_rank(Zcols) != t:
        raise ValueError("Columns of Zcols must be linearly independent over GF(2).")

    J = np.block(
        [
            [np.zeros((n, n), dtype=np.uint8), np.eye(n, dtype=np.uint8)],
            [np.eye(n, dtype=np.uint8), np.zeros((n, n), dtype=np.uint8)],
        ]
    )

    if np.any((Zcols.T @ J @ Zcols) % 2):
        raise ValueError("Columns of Zcols must pairwise commute (form an isotropic set).")

    Z_list: List[np.ndarray] = [Zcols[:, j].copy() for j in range(t)]
    X_list: List[np.ndarray] = []

    def solve_pairing_constraints(targets: List[np.ndarray], rhs: List[int]) -> np.ndarray:
        A = np.vstack([(J @ w) % 2 for w in targets]).astype(np.uint8)
        b = np.array(rhs, dtype=np.uint8)
        return _gf2_solve_one(A, b)

    for j in range(t):
        targets = Z_list[:t] + X_list
        rhs = [1 if k == j else 0 for k in range(t)] + [0] * len(X_list)
        xj = solve_pairing_constraints(targets, rhs)
        X_list.append(xj)

    while len(Z_list) < n:
        W = X_list + Z_list
        A_perp = np.vstack([(J @ w) % 2 for w in W]).astype(np.uint8)
        ns_basis = _gf2_nullspace_basis(A_perp)

        z_new = None
        for v in ns_basis:
            if np.any(v):
                z_new = v
                break
        if z_new is None:
            raise RuntimeError("Failed to find a nonzero vector in the symplectic complement.")

        targets = W + [z_new]
        rhs = [0] * len(W) + [1]
        x_new = solve_pairing_constraints(targets, rhs)

        X_list.append(x_new)
        Z_list.append(z_new)

    X = np.column_stack(X_list).astype(np.uint8) % 2
    Z = np.column_stack(Z_list).astype(np.uint8) % 2
    F = np.concatenate([X, Z], axis=1).astype(np.uint8) % 2

    if not is_symplectic(F):
        raise RuntimeError("Internal error: completion failed to produce a symplectic tableau.")
    if not np.array_equal(F[:, n:n+t], Zcols):
        raise RuntimeError("Internal error: prescribed Z columns were not preserved.")

    return F


def gates_to_unitary(gates: List[Tuple], n: int) -> np.ndarray:
    """
    Convert an ordered gate list into its dense unitary.

    Gate lists in this module are composed on the right, so this routine uses
    ``U = U @ G``.  The caller determines whether a list came from a forward
    walk or tableau reduction; see the authoritative convention note above.
    """
    Hm = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    Sm = np.array([[1, 0], [0, 1j]], dtype=complex)

    U = np.eye(2 ** n, dtype=complex)
    for g in gates:
        if g[0] == "H":
            G = _single_qubit_op(n, g[1], Hm)
        elif g[0] == "S":
            G = _single_qubit_op(n, g[1], Sm)
        elif g[0] == "X":
            G = _single_qubit_op(n, g[1], X2)
        elif g[0] == "CNOT":
            G = _two_qubit_cnot(n, g[1], g[2])
        else:
            raise ValueError(f"Unknown gate: {g}")
        U = U @ G
    return U


def _fix_stabilizer_signs(
        U: np.ndarray,
        g_list: List[str],
        n: int,
        tol: float = 1e-8,
) -> Tuple[np.ndarray, List[Tuple]]:
    """
    Given a unitary U that satisfies U^dagger g_j U = ± Z_j, fix the signs exactly.
    """
    Uq = qt.Qobj(U, dims=[[2] * n, [2] * n])
    extra_gates: List[Tuple] = []

    for j, g in enumerate(g_list):
        g_op = _qutip_pauli_op(n, g)
        z_string = ["I"] * n
        z_string[j] = "Z"
        Zj = _qutip_pauli_op(n, "".join(z_string))

        lhs = Uq.dag() * g_op * Uq
        overlap = complex((Zj * lhs).tr() / (2 ** n))

        if abs(overlap.imag) > tol:
            raise RuntimeError(
                f"Unexpected non-real phase in sign check for g_{j + 1}; overlap={overlap}."
            )
        if abs(abs(overlap.real) - 1.0) > 1e-5:
            raise RuntimeError(
                f"Sign check failed for g_{j + 1}; overlap={overlap}. Expected approximately ±1."
            )

        if overlap.real < 0:
            U = U @ _single_qubit_op(n, j, X2)
            Uq = qt.Qobj(U, dims=[[2] * n, [2] * n])
            extra_gates.append(("X", j))

    return U, extra_gates


def _greedy_maximal_independent_commuting_subset(
        candidates: Ranking,
        n: int,
) -> List[str]:
    """
    Given candidate Paulis [(score, P), ...], return a greedy maximal
    independent commuting subset.
    """
    symp = Symplectic(n)
    span = GF2Basis()

    selected: List[str] = []
    selected_ints: List[int] = []

    for _, P in candidates:
        v = symp.to_int(P)
        if v == 0:
            continue
        if span.contains(v):
            continue
        if not all(symp.commutes_int(v, w) for w in selected_ints):
            continue

        selected.append(P)
        selected_ints.append(v)
        span.add(v)

    return selected


def empirical_stabilizer_peeling(
        rho: Union[np.ndarray, qt.Qobj],
        n: int,
        M1: int,
        lam: float,
        seed: Optional[int] = None,
        return_details: bool = False,
) -> Union[Tuple[List[str], np.ndarray], Dict[str, object]]:
    """
    Implement Algorithm 1: Empirical stabilizer peeling.

    Accepts either a mixed-state density matrix or a pure-state ket. When the
    input is pure, Bell sampling uses bell_sampling_pure to avoid forming rho⊗rho.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if M1 <= 0:
        raise ValueError("M1 must be positive.")
    if not (0.0 < lam < 1.0):
        raise ValueError("lam must lie in (0,1).")

    B, is_pure = _bell_sampling_state(n=n, M=M1, state=rho, seed=seed)
    threshold = 1.0 - lam
    if return_details:
        ranking = rank_all_sP_from_B(B, include_identity=False)
        candidates = [(score, P) for score, P in ranking if score >= threshold]
    else:
        ranking = None
        candidates = paulis_above_threshold_from_B(B, threshold, include_identity=False)
    g_list = _greedy_maximal_independent_commuting_subset(candidates, n)
    t = len(g_list)

    if t == 0:
        U_stab = np.eye(2 ** n, dtype=complex)
        if return_details:
            return {
                "g_list": g_list,
                "t": 0,
                "U_stab": U_stab,
                "gates": [],
                "tableau": np.eye(2 * n, dtype=np.uint8),
                "B": B,
                "ranking": ranking,
                "candidate_set": candidates,
                "is_pure_input": is_pure,
            }
        return g_list, U_stab

    Zcols = np.column_stack([pauli_to_symplectic_col(g) for g in g_list]).astype(np.uint8)
    F = complete_isotropic_to_symplectic(Zcols)

    gates = synthesize_clifford_from_tableau(F)
    U_stab = gates_to_unitary(gates, n)
    U_stab, extra_gates = _fix_stabilizer_signs(U_stab, g_list, n)
    gates = gates + extra_gates

    if return_details:
        return {
            "g_list": g_list,
            "t": t,
            "U_stab": U_stab,
            "gates": gates,
            "tableau": F,
            "B": B,
            "ranking": ranking,
            "candidate_set": candidates,
            "is_pure_input": is_pure,
        }

    return g_list, U_stab


def _embed_residual_pauli(P_res: str, n: int, t: int) -> str:
    m = n - t
    if len(P_res) != m:
        raise ValueError(f"P_res must have length m={m}.")
    return "I" * t + P_res


def _embed_residual_unitary(U_res: np.ndarray, n: int, t: int) -> np.ndarray:
    m = n - t
    if U_res.shape != (2 ** m, 2 ** m):
        raise ValueError(f"U_res must have shape {(2 ** m, 2 ** m)}.")
    return np.kron(np.eye(2 ** t, dtype=complex), U_res)


def _complete_partial_symplectic_basis(
        Xcols: np.ndarray,
        Zcols_paired: np.ndarray,
) -> np.ndarray:
    """
    Complete a partial symplectic specification on m qubits into a full tableau.
    """
    Xcols = np.asarray(Xcols, dtype=np.uint8) % 2
    Zcols_paired = np.asarray(Zcols_paired, dtype=np.uint8) % 2

    if Xcols.ndim != 2 or Zcols_paired.ndim != 2:
        raise ValueError("Xcols and Zcols_paired must be 2D arrays.")

    two_m, L = Xcols.shape
    two_m2, a = Zcols_paired.shape
    if two_m != two_m2:
        raise ValueError("Xcols and Zcols_paired must have the same row count.")
    if two_m % 2 != 0:
        raise ValueError("Row count must be even, i.e. 2m.")
    if a > L:
        raise ValueError("Need a <= L.")

    m = two_m // 2
    if L > m:
        raise ValueError(f"Need L <= m, but got L={L}, m={m}.")
    if L == 0:
        return np.eye(2 * m, dtype=np.uint8)

    J = np.block(
        [
            [np.zeros((m, m), dtype=np.uint8), np.eye(m, dtype=np.uint8)],
            [np.eye(m, dtype=np.uint8), np.zeros((m, m), dtype=np.uint8)],
        ]
    )

    if _gf2_rank(Xcols) != L:
        raise ValueError("Prescribed X columns must be linearly independent.")
    if a > 0 and _gf2_rank(Zcols_paired) != a:
        raise ValueError("Prescribed paired Z columns must be linearly independent.")
    if np.any((Xcols.T @ J @ Xcols) % 2):
        raise ValueError("Prescribed X columns must pairwise commute.")
    if a > 0 and np.any((Zcols_paired.T @ J @ Zcols_paired) % 2):
        raise ValueError("Prescribed paired Z columns must pairwise commute.")
    if a > 0:
        X_paired = Xcols[:, :a]
        pairing = (X_paired.T @ J @ Zcols_paired) % 2
        if not np.array_equal(pairing, np.eye(a, dtype=np.uint8)):
            raise ValueError("Paired X/Z columns must satisfy <X_i, Z_j> = delta_{ij}.")
        if L > a:
            X_single = Xcols[:, a:L]
            if np.any((X_single.T @ J @ Zcols_paired) % 2):
                raise ValueError("Singleton X columns must commute with all prescribed paired Z columns.")

    X_list: List[np.ndarray] = [Xcols[:, j].copy() for j in range(L)]
    Z_list: List[np.ndarray] = [Zcols_paired[:, j].copy() for j in range(a)]

    def solve_pairing_constraints(targets: List[np.ndarray], rhs: List[int]) -> np.ndarray:
        A = np.vstack([(J @ w) % 2 for w in targets]).astype(np.uint8)
        b = np.array(rhs, dtype=np.uint8)
        return _gf2_solve_one(A, b)

    for j in range(a, L):
        targets = X_list + Z_list
        rhs = [1 if k == j else 0 for k in range(L)] + [0] * len(Z_list)
        Z_list.append(solve_pairing_constraints(targets, rhs))

    while len(X_list) < m:
        W = X_list + Z_list
        A_perp = np.vstack([(J @ w) % 2 for w in W]).astype(np.uint8)
        ns_basis = _gf2_nullspace_basis(A_perp)

        x_new = None
        for v in ns_basis:
            if np.any(v):
                x_new = v
                break
        if x_new is None:
            raise RuntimeError("Failed to find a nonzero vector in the symplectic complement.")

        targets = W + [x_new]
        rhs = [0] * len(W) + [1]
        z_new = solve_pairing_constraints(targets, rhs)

        X_list.append(x_new)
        Z_list.append(z_new)

    X = np.column_stack(X_list).astype(np.uint8) % 2
    Z = np.column_stack(Z_list).astype(np.uint8) % 2
    F = np.concatenate([X, Z], axis=1).astype(np.uint8) % 2

    if not is_symplectic(F):
        raise RuntimeError("Internal error: completion failed to produce a symplectic tableau.")
    if not np.array_equal(F[:, :L], Xcols):
        raise RuntimeError("Internal error: prescribed X columns were not preserved.")
    if a > 0 and not np.array_equal(F[:, m:m+a], Zcols_paired):
        raise RuntimeError("Internal error: prescribed Z columns were not preserved.")

    return F


def _fix_recovery_axis_signs(
        U_res: np.ndarray,
        A1_axes: List[Dict[str, str]],
        A2_axes: List[Dict[str, str]],
        m: int,
        tol: float = 1e-8,
) -> np.ndarray:
    """
    Fix signs so that on the residual m-qubit space the recovered axes map exactly
    to the canonical X/Z logical axes.
    """
    Uq = qt.Qobj(U_res, dims=[[2] * m, [2] * m])

    for j, axes in enumerate(A1_axes):
        Rx = _qutip_pauli_op(m, axes["x"])
        Rz = _qutip_pauli_op(m, axes["z"])

        x_string = ["I"] * m
        x_string[j] = "X"
        Xj = _qutip_pauli_op(m, "".join(x_string))

        z_string = ["I"] * m
        z_string[j] = "Z"
        Zj = _qutip_pauli_op(m, "".join(z_string))

        lhs_x = Uq.dag() * Rx * Uq
        overlap_x = complex((Xj * lhs_x).tr() / (2 ** m))
        if abs(overlap_x.imag) > tol or abs(abs(overlap_x.real) - 1.0) > 1e-5:
            raise RuntimeError(f"Failed X-sign check on paired qubit {j + 1}.")
        if overlap_x.real < 0:
            U_res = U_res @ _single_qubit_op(m, j, Z2)
            Uq = qt.Qobj(U_res, dims=[[2] * m, [2] * m])

        lhs_z = Uq.dag() * Rz * Uq
        overlap_z = complex((Zj * lhs_z).tr() / (2 ** m))
        if abs(overlap_z.imag) > tol or abs(abs(overlap_z.real) - 1.0) > 1e-5:
            raise RuntimeError(f"Failed Z-sign check on paired qubit {j + 1}.")
        if overlap_z.real < 0:
            U_res = U_res @ _single_qubit_op(m, j, X2)
            Uq = qt.Qobj(U_res, dims=[[2] * m, [2] * m])

    offset = len(A1_axes)
    for s, axes in enumerate(A2_axes):
        j = offset + s
        Rx = _qutip_pauli_op(m, axes["x"])
        x_string = ["I"] * m
        x_string[j] = "X"
        Xj = _qutip_pauli_op(m, "".join(x_string))

        lhs_x = Uq.dag() * Rx * Uq
        overlap_x = complex((Xj * lhs_x).tr() / (2 ** m))
        if abs(overlap_x.imag) > tol or abs(abs(overlap_x.real) - 1.0) > 1e-5:
            raise RuntimeError(f"Failed X-sign check on singleton qubit {s + 1}.")
        if overlap_x.real < 0:
            U_res = U_res @ _single_qubit_op(m, j, Z2)
            Uq = qt.Qobj(U_res, dims=[[2] * m, [2] * m])

    return U_res


def rank_guided_symplectic_recovery(
        tilde_rho: Union[np.ndarray, qt.Qobj],
        n: int,
        t: int,
        M2: int,
        kappa: float,
        seed: Optional[int] = None,
        return_details: bool = False,
) -> Union[
    Tuple[Dict[str, object], np.ndarray],
    Dict[str, object]
]:
    """
    Implement Algorithm 2: Rank-Guided Symplectic Recovery.

    Accepts either a mixed-state density matrix or a pure-state ket. When the
    input is pure, Bell sampling uses bell_sampling_pure and all state updates in
    the higher-level pipeline may stay in ket form.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if not (0 <= t <= n):
        raise ValueError("Need 0 <= t <= n.")
    if not (0.0 < kappa < 0.5):
        raise ValueError("kappa must lie in (0, 1/2).")

    m = n - t
    if m == 0:
        is_pure = _state_to_qobj(tilde_rho, n).isket
        recovered_axes = {
            "A1_axes_res": [],
            "A2_axes_res": [],
            "A1_axes_full": [],
            "A2_axes_full": [],
            "A1": [],
            "A2": [],
            "m": 0,
        }
        U_rec = np.eye(2 ** n, dtype=complex)
        if return_details:
            return {
                "recovered_axes": recovered_axes,
                "U_rec": U_rec,
                "B_full": None,
                "B_res": None,
                "ranking_res": [],
                "candidates_res": [],
                "tableau_res": np.eye(0, dtype=np.uint8),
                "is_pure_input": is_pure,
            }
        return recovered_axes, U_rec

    if M2 <= 0:
        raise ValueError("M2 must be positive when residual qubits remain.")

    B_full, is_pure = _bell_sampling_state(n=n, M=M2, state=tilde_rho, seed=seed)
    B_res = B_full[t:, :, :]
    if return_details:
        ranking_res = rank_all_sP_from_B(B_res, include_identity=False)
        candidates_res = [(score, P) for score, P in ranking_res if score >= kappa]
    else:
        ranking_res = None
        candidates_res = paulis_above_threshold_from_B(B_res, kappa, include_identity=False)

    symp = Symplectic(m)
    span = GF2Basis()
    A1_axes_res: List[Dict[str, str]] = []
    A2_axes_res: List[Dict[str, str]] = []
    assigned_vecs: List[int] = []

    def commutes_with_completed_qubits(v: int) -> bool:
        return all(symp.commutes_int(v, w) for w in assigned_vecs)

    for _, Px in candidates_res:
        vx = symp.to_int(Px)
        if span.contains(vx):
            continue
        if not commutes_with_completed_qubits(vx):
            continue

        partner = None
        for _, Pz in candidates_res:
            vz = symp.to_int(Pz)
            if not symp.anticommutes_int(vz, vx):
                continue
            if not commutes_with_completed_qubits(vz):
                continue
            partner = Pz
            break

        if partner is not None:
            vz = symp.to_int(partner)
            Py = symp.to_str(vx ^ vz)
            A1_axes_res.append({"x": Px, "z": partner, "y": Py})
            for v in (vx, vz):
                assigned_vecs.append(v)
                span.add(v)
        else:
            A2_axes_res.append({"x": Px})
            assigned_vecs.append(vx)
            span.add(vx)

    a = len(A1_axes_res)
    b = len(A2_axes_res)
    L = a + b

    if L == 0:
        U_res = np.eye(2 ** m, dtype=complex)
        F_res = np.eye(2 * m, dtype=np.uint8)
    else:
        Xcols_list: List[np.ndarray] = []
        Zcols_pair_list: List[np.ndarray] = []

        for axes in A1_axes_res:
            Xcols_list.append(pauli_to_symplectic_col(axes["x"]))
            Zcols_pair_list.append(pauli_to_symplectic_col(axes["z"]))
        for axes in A2_axes_res:
            Xcols_list.append(pauli_to_symplectic_col(axes["x"]))

        Xcols = np.column_stack(Xcols_list).astype(np.uint8)
        Zcols_pairs = (
            np.column_stack(Zcols_pair_list).astype(np.uint8)
            if a > 0 else np.zeros((2 * m, 0), dtype=np.uint8)
        )

        F_res = _complete_partial_symplectic_basis(Xcols, Zcols_pairs)
        gates_res = synthesize_clifford_from_tableau(F_res)
        U_res = gates_to_unitary(gates_res, m)
        U_res = _fix_recovery_axis_signs(U_res, A1_axes_res, A2_axes_res, m)

    U_rec = _embed_residual_unitary(U_res, n, t)

    A1_axes_full = [
        {
            "x": _embed_residual_pauli(axes["x"], n, t),
            "z": _embed_residual_pauli(axes["z"], n, t),
            "y": _embed_residual_pauli(axes["y"], n, t),
        }
        for axes in A1_axes_res
    ]
    A2_axes_full = [
        {"x": _embed_residual_pauli(axes["x"], n, t)}
        for axes in A2_axes_res
    ]

    recovered_axes = {
        "A1_axes_res": A1_axes_res,
        "A2_axes_res": A2_axes_res,
        "A1_axes_full": A1_axes_full,
        "A2_axes_full": A2_axes_full,
        "A1": list(range(1, a + 1)),
        "A2": list(range(a + 1, a + b + 1)),
        "m": m,
    }

    if return_details:
        return {
            "recovered_axes": recovered_axes,
            "U_rec": U_rec,
            "B_full": B_full,
            "B_res": B_res,
            "ranking_res": ranking_res,
            "candidates_res": candidates_res,
            "tableau_res": F_res,
            "is_pure_input": is_pure,
        }

    return recovered_axes, U_rec


def full_recovery_infidelity(
        rho: Union[np.ndarray, qt.Qobj],
        n: int,
        lam: float,
        kappa: float,
        M1: int,
        M2: int,
        M3: int,
        seed1: Optional[int] = None,
        seed2: Optional[int] = None,
        seed3: Optional[int] = None,
        master_seed: Optional[int] = None,
        return_details: bool = False,
) -> Union[Tuple[float, float], Dict[str, object]]:
    """
    Full tomography pipeline.

    Mixed-state input:
      keep the original density-matrix workflow.

    Pure-state input:
      keep state manipulations in ket form as long as possible, using
      bell_sampling_pure in the peeling and recovery steps.

    Returns
    -------
    If return_details=False:
        (infidelity, trace_distance)

    Physical-copy accounting
    ------------------------
    Each Bell record consumes two copies. Stage 3 uses M3 shared global records
    in each of the X, Y, and Z product bases, so all one-qubit marginals together
    consume 3*M3 copies, not 3*n*M3. Thus the total is 2*M1 + 2*M2 + 3*M3
    when t<n, and 2*M1 + 3*M3 when peeling removes all qubits (t=n).
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if M1 <= 0 or M3 <= 0:
        raise ValueError("M1 and M3 must be positive.")
    if not (0.0 < lam < 1.0):
        raise ValueError("lam must lie in (0,1).")
    if not (0.0 < kappa < 0.5):
        raise ValueError("kappa must lie in (0, 1/2).")

    # A master seed yields independent Bell-1, Bell-2, and tomography streams.
    # Explicit stage seeds retain backwards-compatible control of the first two.
    if master_seed is not None:
        spawned = _child_seeds(master_seed, 3)
        stage_seed1 = seed1 if seed1 is not None else spawned[0]
        stage_seed2 = seed2 if seed2 is not None else spawned[1]
        stage_seed3 = seed3 if seed3 is not None else spawned[2]
    elif seed1 is not None or seed2 is not None or seed3 is not None:
        entropy = [0xFFFFFFFF if value is None else int(value) for value in (seed1, seed2, seed3)]
        fallback = _child_seeds(
            int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint64)[0]), 3
        )
        stage_seed1 = seed1 if seed1 is not None else fallback[0]
        stage_seed2 = seed2 if seed2 is not None else fallback[1]
        stage_seed3 = seed3 if seed3 is not None else fallback[2]
    else:
        stage_seed1 = stage_seed2 = stage_seed3 = None

    state_q = _state_to_qobj(rho, n)
    is_pure_input = state_q.isket

    # Step 1: empirical stabilizer peeling
    peel_out = empirical_stabilizer_peeling(
        rho=state_q,
        n=n,
        M1=M1,
        lam=lam,
        seed=stage_seed1,
        return_details=return_details,
    )
    if return_details:
        U_stab_np = peel_out["U_stab"]
        g_list = peel_out["g_list"]
        t = peel_out["t"]
    else:
        g_list, U_stab_np = peel_out
        t = len(g_list)

    U_stab = qt.Qobj(U_stab_np, dims=[[2] * n, [2] * n])

    if is_pure_input:
        psi_q = state_q.unit()
        tilde_state = U_stab.dag() * psi_q
    else:
        rho_q = _as_density(state_q, n)
        tilde_state = U_stab.dag() * rho_q * U_stab

    # Step 2: rank-guided symplectic recovery
    if t == n:
        rec_out = None
        U_rec_np = np.eye(2 ** n, dtype=complex)
    else:
        if M2 <= 0:
            raise ValueError("M2 must be positive unless empirical peeling removes all qubits.")
        rec_out = rank_guided_symplectic_recovery(
            tilde_rho=tilde_state,
            n=n,
            t=t,
            M2=M2,
            kappa=kappa,
            seed=stage_seed2,
            return_details=return_details,
        )
        U_rec_np = rec_out["U_rec"] if return_details else rec_out[1]
    U_rec = qt.Qobj(U_rec_np, dims=[[2] * n, [2] * n])

    # Step 3: tomography in the doubly rotated frame. Each global product-Pauli
    # record is shared across all one-qubit marginals.
    if is_pure_input:
        sigma_state = U_rec.dag() * tilde_state
    else:
        sigma_state = U_rec.dag() * tilde_state * U_rec

    shared_tomography = _shared_single_qubit_tomography(
        sigma_state,
        n,
        M3,
        seed=stage_seed3,
        return_details=return_details,
    )
    rho_prod_prime = (
        shared_tomography["rho_estimates"] if return_details else shared_tomography
    )
    rho_prod_prime_q = qt.tensor(rho_prod_prime)
    rho_prod_prime_q.dims = [[2] * n, [2] * n]

    # Step 4: rotate back to obtain rho_recovered
    U_total = U_stab * U_rec
    rho_recovered = U_total * rho_prod_prime_q * U_total.dag()
    rho_recovered = 0.5 * (rho_recovered + rho_recovered.dag())
    rho_recovered = rho_recovered / rho_recovered.tr()

    if is_pure_input:
        psi_q = state_q.unit()
        fidelity = float(np.real((psi_q.dag() * (rho_recovered * psi_q))))
        fidelity = float(np.clip(fidelity, 0.0, 1.0))
        F_amp = float(np.sqrt(fidelity))
        rho_ref = psi_q * psi_q.dag()
    else:
        rho_ref = _as_density(state_q, n)
        F_amp = float(qt.fidelity(rho_recovered, rho_ref))
        fidelity = F_amp ** 2

    infidelity = 1.0 - fidelity
    trace_distance = 0.5 * float((rho_recovered - rho_ref).norm("tr"))

    copies_bell_1 = 2 * M1
    copies_bell_2 = 0 if t == n else 2 * M2
    copies_tomography = 3 * M3
    total_physical_copies = copies_bell_1 + copies_bell_2 + copies_tomography

    if return_details:
        return {
            "infidelity": infidelity,
            "trace_distance": trace_distance,
            "fidelity": fidelity,
            "fidelity_amplitude": F_amp,
            "rho_recovered": rho_recovered,
            "rho_prod_prime": rho_prod_prime,
            "rho_prod_prime_tensor": rho_prod_prime_q,
            "sigma": sigma_state,
            "tilde_state": tilde_state,
            "U_stab": U_stab,
            "U_rec": U_rec,
            "U_total": U_total,
            "peeling_output": peel_out,
            "recovery_output": rec_out,
            "t": t,
            "g_list": g_list,
            "is_pure_input": is_pure_input,
            "reference_state": rho_ref,
            "stage_seeds": {"bell_1": stage_seed1, "bell_2": stage_seed2, "tomography": stage_seed3},
            "shared_tomography": shared_tomography,
            "copies_bell_1": copies_bell_1,
            "copies_bell_2": copies_bell_2,
            "copies_tomography": copies_tomography,
            "total_physical_copies": total_physical_copies,
            "copy_budget": {
                "bell_1": copies_bell_1,
                "bell_2": copies_bell_2,
                "tomography": copies_tomography,
                "total": total_physical_copies,
            },
        }

    return infidelity, trace_distance


# =============================================================================
# Random Clifford-encoded product-state generators
# =============================================================================


def random_clifford_encoded_product_state(
        steps: int,
        delta: float,
        n: int,
        S: int,
        A1: int,
        A2: int,
        B: int,
        seed: Optional[int] = None,
        return_details: bool = False,
) -> Union[qt.Qobj, Dict[str, object]]:
    r"""
    Generate a random Clifford-encoded product state
        rho = U_c (⊗_i rho_i) U_c^\dagger
    with the class constraints requested by the user.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if any(x < 0 for x in (S, A1, A2, B)):
        raise ValueError("S, A1, A2, B must be nonnegative.")
    if S + A1 + A2 + B != n:
        raise ValueError("Need S + A1 + A2 + B = n.")
    if not (0.0 <= delta <= 1.0):
        raise ValueError("delta must lie in [0,1].")

    local_seed, clifford_seed = _child_seeds(seed, 2) if seed is not None else (None, None)
    rng = _rng(local_seed)

    def random_sign() -> float:
        return float(rng.choice([-1.0, 1.0]))

    def build_local_from_sq(vals_sq: np.ndarray) -> Tuple[qt.Qobj, np.ndarray]:
        if np.any(vals_sq < -1e-12) or float(vals_sq.sum()) > 1.0 + 1e-12:
            raise ValueError("Requested Bloch-score constraints are not physically feasible.")
        mags = np.sqrt(np.clip(vals_sq, 0.0, None))
        r = np.array([
            random_sign() * mags[0],
            random_sign() * mags[1],
            random_sign() * mags[2],
        ], dtype=float)

        rho_i = 0.5 * (qt.qeye(2) + r[0] * qt.sigmax() + r[1] * qt.sigmay() + r[2] * qt.sigmaz())
        rho_i = 0.5 * (rho_i + rho_i.dag())
        rho_i = rho_i / rho_i.tr()
        rho_i.dims = [[2], [2]]
        return rho_i, r

    def sample_S() -> Tuple[qt.Qobj, np.ndarray]:
        vals_sq = np.zeros(3, dtype=float)
        axis = int(rng.integers(0, 3))
        vals_sq[axis] = float(rng.uniform(max(0.0, 1.0 - delta), 1.0))
        return build_local_from_sq(vals_sq)

    def sample_A1() -> Tuple[qt.Qobj, np.ndarray]:
        boundary = 1.0 / 3.0
        if delta > boundary + 1e-12:
            raise ValueError("No feasible A1 Bloch region: need delta <= 1/3.")
        if abs(delta - boundary) <= 1e-12:
            vals_sq = np.full(3, boundary, dtype=float)
        else:
            # s_i = delta + x_i with a fourth Dirichlet component providing
            # nonnegative slack in sum_i x_i <= 1-3*delta. Since x_i <= R,
            # s_i <= 1-2*delta <= 1-delta automatically.
            radius = 1.0 - 3.0 * delta
            vals_sq = delta + radius * rng.dirichlet(np.ones(4))[:3]
        return build_local_from_sq(vals_sq)

    def sample_A2() -> Tuple[qt.Qobj, np.ndarray]:
        lo, hi = float(delta), float(1.0 - delta)
        if lo > hi:
            raise ValueError("No feasible A2 Bloch region: need delta <= 1/2.")
        big_axis = int(rng.integers(0, 3))
        other_axes = [a for a in range(3) if a != big_axis]
        for _ in range(10_000):
            vals_sq = np.zeros(3, dtype=float)
            vals_sq[big_axis] = float(rng.uniform(lo, hi))
            vals_sq[other_axes[0]] = float(rng.uniform(0.0, delta))
            vals_sq[other_axes[1]] = float(rng.uniform(0.0, delta))
            if float(vals_sq.sum()) <= 1.0:
                return build_local_from_sq(vals_sq)
        raise ValueError("Failed to sample the feasible A2 Bloch region.")

    def sample_B() -> Tuple[qt.Qobj, np.ndarray]:
        # Sampling directly from the interval box can leave the Bloch ball
        # when delta > 1/sqrt(3); rejection preserves the documented ranges.
        for _ in range(10_000):
            vals_sq = rng.uniform(0.0, delta, size=3)
            if float(vals_sq.sum()) <= 1.0:
                return build_local_from_sq(vals_sq)
        raise ValueError("Failed to sample the feasible B Bloch region.")

    labels = (["S"] * S) + (["A1"] * A1) + (["A2"] * A2) + (["B"] * B)
    rng.shuffle(labels)

    local_states: List[qt.Qobj] = []
    bloch_vectors: List[np.ndarray] = []
    for label in labels:
        if label == "S":
            rho_i, r = sample_S()
        elif label == "A1":
            rho_i, r = sample_A1()
        elif label == "A2":
            rho_i, r = sample_A2()
        elif label == "B":
            rho_i, r = sample_B()
        else:
            raise RuntimeError(f"Unexpected label: {label}")
        local_states.append(rho_i)
        bloch_vectors.append(r)

    rho_product = qt.tensor(local_states)
    rho_product.dims = [[2] * n, [2] * n]

    Uc = random_clifford_gate(n, steps, seed=clifford_seed)[2]
    Uc_qobj = qt.Qobj(Uc, dims=[[2] * n, [2] * n])
    rho_encoded = Uc_qobj * rho_product * Uc_qobj.dag()
    rho_encoded = 0.5 * (rho_encoded + rho_encoded.dag())
    rho_encoded = rho_encoded / rho_encoded.tr()
    rho_encoded.dims = [[2] * n, [2] * n]

    if return_details:
        return {
            "rho_encoded": rho_encoded,
            "rho_product": rho_product,
            "Uc": Uc,
            "Uc_qobj": Uc_qobj,
            "local_states": local_states,
            "bloch_vectors": bloch_vectors,
            "class_assignment": labels,
        }

    return rho_encoded


def random_clifford_encoded_product_pure_state(
        steps: int,
        delta: float,
        n: int,
        S: int,
        A1: int,
        seed: Optional[int] = None,
        return_details: bool = False,
) -> Union[qt.Qobj, Dict[str, object]]:
    """
    Generate a random Clifford-encoded product pure state as a ket.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if S < 0 or A1 < 0:
        raise ValueError("S and A1 must be nonnegative.")
    if S + A1 != n:
        raise ValueError("Need S + A1 = n.")
    if not (0.0 <= delta <= 1.0):
        raise ValueError("delta must lie in [0,1].")
    if delta > 1.0 / 3.0 + 1e-12:
        raise ValueError("For pure A1 states with all three |r_alpha|^2 >= delta, need delta <= 1/3.")

    local_seed, clifford_seed = _child_seeds(seed, 2) if seed is not None else (None, None)
    rng = _rng(local_seed)

    def random_sign() -> float:
        return float(rng.choice([-1.0, 1.0]))

    def ket_from_bloch(r: np.ndarray) -> qt.Qobj:
        rx, ry, rz = map(float, r)
        norm_r = float(np.linalg.norm(r))
        if not np.isclose(norm_r, 1.0, atol=1e-10):
            raise ValueError("Bloch vector for a pure state must have norm 1.")
        rz_clip = float(np.clip(rz, -1.0, 1.0))
        theta = float(np.arccos(rz_clip))
        phi = 0.0 if abs(np.sin(theta)) < 1e-12 else float(np.angle(rx + 1j * ry))
        psi = np.array([
            np.cos(theta / 2.0),
            np.exp(1j * phi) * np.sin(theta / 2.0)
        ], dtype=complex)
        return qt.Qobj(psi.reshape((2, 1)), dims=[[2], [1]]).unit()

    def sample_S_pure() -> Tuple[qt.Qobj, np.ndarray]:
        vals_sq = np.zeros(3, dtype=float)
        big_axis = int(rng.integers(0, 3))
        other = [a for a in range(3) if a != big_axis]
        big_val = float(rng.uniform(max(0.0, 1.0 - delta), 1.0))
        rem = 1.0 - big_val
        u = float(rng.uniform(0.0, 1.0))
        vals_sq[big_axis] = big_val
        vals_sq[other[0]] = u * rem
        vals_sq[other[1]] = (1.0 - u) * rem
        mags = np.sqrt(np.clip(vals_sq, 0.0, None))
        r = np.array([random_sign() * mags[0], random_sign() * mags[1], random_sign() * mags[2]], dtype=float)
        r = r / np.linalg.norm(r)
        return ket_from_bloch(r), r

    def sample_A1_pure() -> Tuple[qt.Qobj, np.ndarray]:
        rem = 1.0 - 3.0 * delta
        if rem < -1e-12:
            raise ValueError("No feasible pure A1 state for this delta; need delta <= 1/3.")
        if rem <= 1e-12:
            vals_sq = np.array([delta, delta, delta], dtype=float)
        else:
            vals_sq = delta + rng.dirichlet(np.ones(3)) * rem
        mags = np.sqrt(np.clip(vals_sq, 0.0, None))
        r = np.array([random_sign() * mags[0], random_sign() * mags[1], random_sign() * mags[2]], dtype=float)
        r = r / np.linalg.norm(r)
        return ket_from_bloch(r), r

    labels = (["S"] * S) + (["A1"] * A1)
    rng.shuffle(labels)

    local_kets: List[qt.Qobj] = []
    bloch_vectors: List[np.ndarray] = []
    for label in labels:
        if label == "S":
            ket_i, r = sample_S_pure()
        elif label == "A1":
            ket_i, r = sample_A1_pure()
        else:
            raise RuntimeError(f"Unexpected label: {label}")
        local_kets.append(ket_i)
        bloch_vectors.append(r)

    psi_product = qt.tensor(local_kets)
    psi_product.dims = [[2] * n, [1] * n]
    psi_product = psi_product.unit()

    Uc = random_clifford_gate(n, steps, seed=clifford_seed)[2]
    Uc_qobj = qt.Qobj(Uc, dims=[[2] * n, [2] * n])
    psi_encoded = (Uc_qobj * psi_product).unit()
    psi_encoded.dims = [[2] * n, [1] * n]

    if return_details:
        rho_product = psi_product * psi_product.dag()
        rho_product.dims = [[2] * n, [2] * n]
        rho_encoded = psi_encoded * psi_encoded.dag()
        rho_encoded.dims = [[2] * n, [2] * n]
        return {
            "psi_encoded": psi_encoded,
            "psi_product": psi_product,
            "rho_encoded": rho_encoded,
            "rho_product": rho_product,
            "Uc": Uc,
            "Uc_qobj": Uc_qobj,
            "local_kets": local_kets,
            "bloch_vectors": bloch_vectors,
            "class_assignment": labels,
        }

    return psi_encoded


# =============================================================================
# Generic pure-state local Pauli tomography baseline
# =============================================================================


def _measurement_probs_in_local_pauli_basis(psi_vec: np.ndarray, axes: Tuple[str, ...]) -> np.ndarray:
    """
    Exact Born probabilities for measuring |psi> in the local Pauli basis `axes`.

    The returned array has length 2^n in standard computational bit order:
    bit 0 at qubit i means the +1 eigenvector of axes[i], and bit 1 means
    the -1 eigenvector of axes[i].
    """
    n = len(axes)
    state = np.asarray(psi_vec, dtype=complex).reshape([2] * n)

    for ax, basis_axis in enumerate(axes):
        U = _single_qubit_pauli_basis_change(basis_axis)
        Udag = U.conj().T
        state = np.tensordot(Udag, state, axes=([1], [ax]))
        state = np.moveaxis(state, 0, ax)

    probs = np.abs(state.reshape(-1)) ** 2
    probs = np.clip(probs.real, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("Numerical issue: invalid measurement probabilities.")
    return probs / total


def _fwht_probabilities_to_pauli_expectations(prob: np.ndarray) -> np.ndarray:
    """
    Walsh-Hadamard transform of an empirical probability vector.

    For outcome bits o and subset mask s, output[s] = sum_o prob[o] (-1)^(s·o).
    Thus output[s] is the empirical expectation of the product of local outcomes
    over the subset s.
    """
    h = np.asarray(prob, dtype=float).copy()
    n_out = h.size
    if n_out <= 0 or (n_out & (n_out - 1)) != 0:
        raise ValueError("prob length must be a positive power of two.")

    step = 1
    while step < n_out:
        for start in range(0, n_out, 2 * step):
            a = h[start:start + step].copy()
            b = h[start + step:start + 2 * step].copy()
            h[start:start + step] = a + b
            h[start + step:start + 2 * step] = a - b
        step *= 2
    return h


def _pauli_from_setting_and_subset(axes: Tuple[str, ...], subset_mask: int) -> str:
    """Map a local setting and subset mask to an n-qubit Pauli string."""
    n = len(axes)
    return "".join(
        axes[i] if ((subset_mask >> (n - 1 - i)) & 1) else "I"
        for i in range(n)
    )


def generic_pure_state_local_pauli_tomography(
        n: int,
        psi: Union[np.ndarray, qt.Qobj],
        M: int,
        seed: Optional[int] = None,
        return_details: bool = False,
) -> Union[Tuple[qt.Qobj, float, float], Dict[str, object]]:
    """
    Generic pure-state tomography baseline using all local Pauli bases.

    Method:
      1. Measure every local Pauli setting B in {X,Y,Z}^n.
      2. Use M shots per setting, for total copies M * 3^n.
      3. Estimate all Pauli expectations mu(P)=Tr(P |psi><psi|) by averaging
         over all compatible local settings.
      4. Reconstruct by linear inversion:
             rho_lin = 2^{-n} sum_P mu_hat(P) P.
      5. Project to a pure state by taking the top eigenvector of rho_lin.

    Args:
        n: number of qubits.
        psi: target pure state, either a QuTiP ket or a NumPy ket.
        M: shots per local Pauli measurement setting.
        seed: optional RNG seed.
        return_details: if True, return a dictionary with rho_lin, mu_hat, etc.

    Returns:
        If return_details=False:
            (hat_rho, infidelity, trace_distance)
        where hat_rho = |hat_psi><hat_psi| is a QuTiP density matrix.

        If return_details=True:
            dictionary containing the above plus rho_lin, hat_psi, mu_hat,
            total_samples, and settings.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if M <= 0:
        raise ValueError("M must be positive.")

    psi_q = _state_to_qobj(psi, n)
    if not psi_q.isket:
        raise ValueError("psi must be a pure state ket for this pure-state tomography baseline.")
    psi_q = psi_q.unit()
    psi_vec = psi_q.full().reshape(-1)
    d = 2 ** n

    rng = np.random.default_rng(seed)

    # Accumulate Pauli expectation estimates. A weight-w Pauli is compatible
    # with 3^(n-w) local settings, and we average all such estimates.
    mu_sum: Dict[str, float] = {"I" * n: 0.0}
    mu_count: Dict[str, int] = {"I" * n: 0}

    settings_iter = itertools.product(("X", "Y", "Z"), repeat=n)
    settings = [] if return_details else None
    for axes in settings_iter:
        if settings is not None:
            settings.append(axes)
        probs = _measurement_probs_in_local_pauli_basis(psi_vec, axes)
        outcomes = rng.choice(d, size=M, p=probs)
        empirical_prob = np.bincount(outcomes, minlength=d).astype(float) / float(M)

        # All subset-product expectations for this basis at once.
        subset_expectations = _fwht_probabilities_to_pauli_expectations(empirical_prob)

        for subset_mask, est in enumerate(subset_expectations):
            P = _pauli_from_setting_and_subset(axes, subset_mask)
            mu_sum[P] = mu_sum.get(P, 0.0) + float(est)
            mu_count[P] = mu_count.get(P, 0) + 1

    mu_hat: Dict[str, float] = {}
    for P, total in mu_sum.items():
        cnt = mu_count[P]
        if cnt <= 0:
            raise RuntimeError(f"Internal error: no estimates accumulated for Pauli string {P}.")
        mu_hat[P] = total / float(cnt)
    mu_hat["I" * n] = 1.0  # exact normalization convention

    # Linear inversion rho_lin = 2^{-n} sum_P mu_hat(P) P.
    rho_lin = 0 * _qutip_pauli_op(n, "I" * n)
    for P in itertools.product(PAULI_CHARS, repeat=n):
        P_str = "".join(P)
        rho_lin = rho_lin + mu_hat.get(P_str, 0.0) * _qutip_pauli_op(n, P_str)
    rho_lin = rho_lin / d
    rho_lin = 0.5 * (rho_lin + rho_lin.dag())
    rho_lin.dims = [[2] * n, [2] * n]

    # Pure-state projection by top eigenvector.
    evals, evecs = np.linalg.eigh(rho_lin.full())
    top_idx = int(np.argmax(evals.real))
    hat_vec = evecs[:, top_idx]
    hat_vec = hat_vec / np.linalg.norm(hat_vec)
    hat_psi = qt.Qobj(hat_vec.reshape((d, 1)), dims=[[2] * n, [1] * n])
    hat_rho = hat_psi * hat_psi.dag()
    hat_rho.dims = [[2] * n, [2] * n]

    fidelity = float(np.abs(np.vdot(hat_vec, psi_vec)) ** 2)
    fidelity = float(np.clip(fidelity, 0.0, 1.0))
    infidelity = 1.0 - fidelity
    trace_distance = float(np.sqrt(max(0.0, infidelity)))

    if return_details:
        return {
            "hat_rho": hat_rho,
            "hat_psi": hat_psi,
            "rho_lin": rho_lin,
            "infidelity": infidelity,
            "trace_distance": trace_distance,
            "fidelity": fidelity,
            "mu_hat": mu_hat,
            "total_samples": M * (3 ** n),
            "shots_per_setting": M,
            "num_settings": 3 ** n,
            "settings": settings,
            "top_eigenvalue": float(evals[top_idx].real),
            "all_eigenvalues": evals,
        }

    return hat_rho, infidelity, trace_distance
