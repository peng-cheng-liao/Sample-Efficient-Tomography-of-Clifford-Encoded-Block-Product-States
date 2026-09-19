from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import qutip as qt
import re


# ---------- Single-qubit helpers ----------

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

# ----------------------------
# Types / constants
# ----------------------------
Axis = Literal["X", "Y", "Z"]
PauliSpec = Union[str, Iterable[Tuple[int, Axis]]]
Ranking = List[Tuple[float, str]]  # [(score, "IXYZ..."), ...] sorted high->low

PAULI_CHARS = "IXYZ"
AXIS_TO_K = {"X": 0, "Y": 1, "Z": 2}


# ----------------------------
# Basic helpers
# ----------------------------
def _ensure_qubit_dims(obj: qt.Qobj, n: int) -> qt.Qobj:
    """Ensure obj has qubit dims ([[2]*n,[2]*n]) for operators or ([[2]*n,[1]*n]) for kets."""
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
    """Normalize dims + convert ket->density."""
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

    out = []
    for i, a in P:
        if not (0 <= i < n):
            raise ValueError("Index out of range in PauliSpec.")
        if a not in ("X", "Y", "Z"):
            raise ValueError('Axis must be one of "X","Y","Z".')
        out.append((int(i), a))
    return out


def _qutip_pauli_op(n: int, P: PauliSpec) -> qt.Qobj:
    sup = dict(_parse_pauli_spec(n, P))
    op1 = {"I": qt.qeye(2), "X": qt.sigmax(), "Y": qt.sigmay(), "Z": qt.sigmaz()}
    return qt.tensor([op1[sup.get(i, "I")] for i in range(n)])


# ----------------------------
# Bell sampling
# ----------------------------
def bell_sampling(n: int, M: int, rho: qt.Qobj, seed: Optional[int] = None) -> np.ndarray:
    """
    Joint product-Bell measurement on (rho ⊗ rho) with correlations preserved.
    Output B has shape (n, M, 3) with columns [X,Y,Z] eigenvalues in {-1,+1}.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if M <= 0:
        raise ValueError("M must be positive.")
    rng = np.random.default_rng(seed)

    rho = _as_density(rho, n)
    rho2 = qt.tensor(rho, rho)

    # Interleave copies to make each (i, i+n) pair contiguous: (0,n,1,n+1,...)
    perm = [p for i in range(n) for p in (i, n + i)]
    try:
        rho2 = rho2.permute(perm)
    except Exception as e:
        raise RuntimeError("Failed to permute rho⊗rho; ensure qubit dims and QuTiP supports Qobj.permute.") from e

    # Bell projectors on 2 qubits
    zero, one = qt.basis(2, 0), qt.basis(2, 1)
    bell_kets = [
        (qt.tensor(zero, zero) + qt.tensor(one, one)).unit(),  # Φ+
        (qt.tensor(zero, zero) - qt.tensor(one, one)).unit(),  # Φ-
        (qt.tensor(zero, one) + qt.tensor(one, zero)).unit(),  # Ψ+
        (qt.tensor(zero, one) - qt.tensor(one, zero)).unit(),  # Ψ-
    ]
    bell_projs = [k * k.dag() for k in bell_kets]

    # Outcome -> (XX,YY,ZZ) eigenvalues, order [X,Y,Z]
    eig_table = np.array(
        [
            [+1, -1, +1],  # Φ+
            [-1, +1, +1],  # Φ-
            [+1, +1, -1],  # Ψ+
            [-1, -1, -1],  # Ψ-
        ],
        dtype=np.int8,
    )

    # Exact enumeration over 4^n outcomes (exponential)
    if n > 10:
        raise ValueError("This exact sampler enumerates 4^n outcomes; n is too large for this method.")
    num_out = 4 ** n
    pow4 = 4 ** np.arange(n, dtype=np.int64)

    probs = np.empty(num_out, dtype=float)
    for idx in range(num_out):
        digits = [(idx // int(pow4[i])) % 4 for i in range(n)]
        proj = qt.tensor([bell_projs[d] for d in digits])
        probs[idx] = (proj * rho2).tr().real

    probs = np.clip(probs, 0.0, None)
    ssum = probs.sum()
    if ssum <= 0:
        raise RuntimeError("Numerical issue: total probability is non-positive.")
    probs /= ssum

    sampled = rng.choice(num_out, size=M, p=probs)

    B = np.empty((n, M, 3), dtype=np.int8)
    for i in range(n):
        s_i = (sampled // int(pow4[i])) % 4
        B[i, :, :] = eig_table[s_i, :]
    return B


def tr2_true_est_error(B: np.ndarray, rho: qt.Qobj, P: PauliSpec) -> Tuple[float, float, float]:
    """Return (true Tr(rho P)^2, estimate from B, absolute error)."""
    if not isinstance(B, np.ndarray) or B.ndim != 3 or B.shape[2] != 3:
        raise ValueError("B must have shape (n, M, 3).")
    n, M, _ = B.shape
    if M <= 0:
        raise ValueError("B must have M>0 samples.")

    rho = _as_density(rho, n)
    sup = _parse_pauli_spec(n, P)

    if not sup:
        est = 1.0
    else:
        prod = np.ones(M, dtype=np.int8)
        for i, a in sup:
            prod *= B[i, :, AXIS_TO_K[a]]
        est = float(prod.mean())

    P_op = _qutip_pauli_op(n, P)
    t = (P_op * rho).tr().real
    true_val = float(t * t)
    return true_val, est, abs(est - true_val)


# ----------------------------
# Random generators
# ----------------------------
def random_pauli_strings(
        n: int,
        k: int,
        w: Optional[int] = None,
        seed: Optional[int] = None,
        allow_identity: bool = True,
) -> List[str]:
    """Generate k random length-n Pauli strings over {I,X,Y,Z}. Optionally fix weight w."""
    if n <= 0 or k < 0:
        raise ValueError("n must be > 0 and k must be >= 0.")
    if w is not None and not (0 <= w <= n):
        raise ValueError("w must satisfy 0 <= w <= n.")

    rng = np.random.default_rng(seed)
    out: List[str] = []

    if w is None:
        alphabet = np.array(list(PAULI_CHARS))
        for _ in range(k):
            s = rng.choice(alphabet, size=n, replace=True)
            if not allow_identity:
                while np.all(s == "I"):
                    s = rng.choice(alphabet, size=n, replace=True)
            out.append("".join(s.tolist()))
        return out

    # exact weight
    xyz = np.array(["X", "Y", "Z"])
    for _ in range(k):
        s = np.full(n, "I", dtype="<U1")
        if w > 0:
            idx = rng.choice(n, size=w, replace=False)
            s[idx] = rng.choice(xyz, size=w, replace=True)
        out.append("".join(s.tolist()))
    return out


def random_product_state(
        n: int,
        pure: bool = False,
        purity: float = 0.5,  # kept for API compatibility; unused in uniform-ball sampling
        seed: Optional[int] = None,
        return_ket_if_pure: bool = False,
        epsilon: float = 0.01,
) -> qt.Qobj:
    """
    Generate a random n-qubit product state  ρ = ⊗_i ρ_i.

    When pure=False:
      - Sample Bloch vectors uniformly from the 3D unit ball (Bloch ball),
        then reject unless epsilon < |r_x|,|r_y|,|r_z| < 1-epsilon.

    where ρ_i = (1/2)(I + r_i^x σ^x + r_i^y σ^y + r_i^z σ^z).

    Args
      n: number of qubits
      pure: if True, each ρ_i is a random pure state |ψ_i><ψ_i|.
      purity: unused when pure=False in this uniform-ball version (kept for compatibility).
      seed: RNG seed
      return_ket_if_pure: if pure=True, return ⊗_i |ψ_i> as a ket instead of density matrix.
      epsilon: constraint parameter. Used only when pure=False.

    Returns
      QuTiP Qobj with dims:
        - if return_ket_if_pure: [[2]*n, [1]*n]
        - else: [[2]*n, [2]*n]
    """
    if n <= 0:
        raise ValueError("n must be > 0.")
    if not (0.0 <= purity <= 1.0):
        raise ValueError("purity must be in [0,1].")
    if not (0.0 <= epsilon < 0.5):
        raise ValueError("epsilon must satisfy 0 <= epsilon < 1/2.")
    # Geometric feasibility (not strictly required, but gives a clearer error early)
    if epsilon >= 1.0 / np.sqrt(3.0):
        raise ValueError("epsilon is too large: need epsilon < 1/sqrt(3) for feasibility.")

    rng = np.random.default_rng(seed)

    def random_qubit_ket() -> qt.Qobj:
        z = rng.normal(size=2) + 1j * rng.normal(size=2)
        z /= np.linalg.norm(z)
        return qt.Qobj(z.reshape((2, 1)), dims=[[2], [1]])

    def random_unit_vector_3() -> np.ndarray:
        v = rng.normal(size=3)
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            return np.array([1.0, 0.0, 0.0])
        return v / v_norm

    def random_bloch_vector_uniform_ball() -> np.ndarray:
        # Uniform in 3D ball: direction uniform on sphere, radius r = U^(1/3)
        n_hat = random_unit_vector_3()
        r = float(rng.random()) ** (1.0 / 3.0)
        return r * n_hat

    def random_qubit_rho_mixed_uniform_ball() -> qt.Qobj:

        sx, sy, sz = qt.sigmax(), qt.sigmay(), qt.sigmaz()
        #r = np.random.uniform(epsilon, 1 - epsilon)
        #rx, ry, rz = (r ** (1.0 / 3.0)) * random_unit_vector_3()

        max_tries = 200000
        for _ in range(max_tries):
            r_vec = np.random.uniform(epsilon, 1 - epsilon, size=3)
            r_norm = np.linalg.norm(r_vec)
            if r_norm > 1:
                r_vec = r_vec / r_norm
            if np.logical_and(r_vec >= epsilon, r_vec <= 1 - epsilon).all():
                rx, ry, rz = r_vec
                rho_i = (qt.qeye(2) + rx * sx + ry * sy + rz * sz) / 2.0
                rho_i.dims = [[2], [2]]
                return rho_i
        raise RuntimeError(
            "Failed to sample a mixed qubit state satisfying the epsilon constraints "
            "with uniform-in-ball proposals. Try reducing epsilon.")

    """
            max_tries = 200_000
            for _ in range(max_tries):
                rx, ry, rz = random_bloch_vector_uniform_ball()

                if (
                        (abs(rx) > epsilon) and (abs(ry) > epsilon) and (abs(rz) > epsilon)
                        and (abs(rx) < 1.0 - epsilon) and (abs(ry) < 1.0 - epsilon) and (abs(rz) < 1.0 - epsilon)
                ):
                    rho_i = (qt.qeye(2) + rx * sx + ry * sy + rz * sz) / 2.0
                    rho_i.dims = [[2], [2]]
                    return rho_i

            raise RuntimeError(
                "Failed to sample a mixed qubit state satisfying the epsilon constraints "
                "with uniform-in-ball proposals. Try reducing epsilon."
            )
    """

    if pure:
        kets = [random_qubit_ket() for _ in range(n)]
        ket = qt.tensor(kets)
        ket.dims = [[2] * n, [1] * n]
        if return_ket_if_pure:
            return ket
        rho = ket * ket.dag()
        rho.dims = [[2] * n, [2] * n]
        return rho

    rhos = [random_qubit_rho_mixed_uniform_ball() for _ in range(n)]
    rho = qt.tensor(rhos)
    rho.dims = [[2] * n, [2] * n]
    return rho


# ----------------------------
# Ranking utilities
# ----------------------------
def rank_all_sP_from_B(
        B: np.ndarray,
        weight: Optional[int] = None,
        include_identity: bool = False,
) -> Ranking:
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
                ch = tup[i]
                prod *= per_qubit[i][ch]
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


def rank_all_true_sP(
        rho: qt.Qobj,
        n: int,
        weight: Optional[int] = None,
        include_identity: bool = False,
) -> Ranking:
    """Exact s(P)=Tr(rho P)^2 ranking over all Paulis (or exact weight)."""
    rho = _as_density(rho, n)
    if weight is not None and not (0 <= weight <= n):
        raise ValueError("weight must satisfy 0 <= weight <= n.")

    op1 = {"I": qt.qeye(2), "X": qt.sigmax(), "Y": qt.sigmay(), "Z": qt.sigmaz()}
    results: Ranking = []

    def s_of(P: str) -> float:
        P_op = qt.tensor([op1[ch] for ch in P])
        t = (P_op * rho).tr().real
        return float(t * t)

    if weight is None:
        for tup in itertools.product(("I", "X", "Y", "Z"), repeat=n):
            P = "".join(tup)
            if not include_identity and all(ch == "I" for ch in P):
                continue
            results.append((s_of(P), P))
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
                    P = "".join(P_list)
                    results.append((s_of(P), P))

    results.sort(key=lambda x: x[0], reverse=True)
    return results


# ----------------------------
# Symplectic (phase-free) Pauli utilities
# ----------------------------
@dataclass
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
        out = []
        for i in range(self.n):
            xi = (x >> i) & 1
            zi = (z >> i) & 1
            out.append("I" if (xi, zi) == (0, 0) else "X" if (xi, zi) == (1, 0) else "Z" if (xi, zi) == (0, 1) else "Y")
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

    def __init__(self):
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


# ----------------------------
# Recover Q from ranking
# ----------------------------
def recover_Q_from_rankingv1(
        ranking: Ranking,
        n: Optional[int] = None,
        max_steps: Optional[int] = None,
) -> Dict[str, str]:
    """Recover {Q_i^x,Q_i^y,Q_i^z} following the described algorithm (phase ignored)."""
    if not ranking:
        raise ValueError("ranking is empty.")
    if n is None:
        n = len(ranking[0][1])
    if n <= 0:
        raise ValueError("n must be positive.")
    for _, P in ranking:
        if len(P) != n or any(ch not in PAULI_CHARS for ch in P):
            raise ValueError("All Paulis must be length-n strings over {I,X,Y,Z}.")

    symp = Symplectic(n)
    S = GF2Basis()

    Qx: List[Optional[int]] = [None] * n
    Qy: List[Optional[int]] = [None] * n
    Qz: List[Optional[int]] = [None] * n

    def all_none(i: int) -> bool:
        return Qx[i] is None and Qy[i] is None and Qz[i] is None

    def all_assigned() -> bool:
        return all((Qx[i] is not None and Qy[i] is not None and Qz[i] is not None) for i in range(n))

    def assigned_except(i_ex: int) -> List[int]:
        ops: List[int] = []
        for k in range(n):
            if k == i_ex:
                continue
            for v in (Qx[k], Qy[k], Qz[k]):
                if v is not None:
                    ops.append(v)
        return ops

    j = 0
    steps = 0
    max_steps = max_steps if max_steps is not None else len(ranking) * (n + 2)

    while j < len(ranking) and not all_assigned() and steps < max_steps:
        steps += 1
        Pj = symp.to_int(ranking[j][1])

        # Step 2: skip if in span S
        if S.contains(Pj):
            j += 1
            continue

        i = 0
        while True:
            # 4(a)
            if all_none(i):
                Qx[i] = Pj
                S.add(Pj)
                j += 1
                break

            # 4(b) (only if we still need z at site i)
            if Qx[i] is not None and Qz[i] is None:
                if symp.anticommutes_int(Pj, Qx[i]) and all(symp.commutes_int(Pj, v) for v in assigned_except(i)):
                    Qz[i] = Pj
                    Qy[i] = Qx[i] ^ Qz[i]
                    S.add(Pj)
                    j += 1
                    break

            # 4(c)
            if Qx[i] is not None and symp.commutes_int(Pj, Qx[i]):
                if i == n - 1:
                    j += 1
                    break
                i += 1
                continue

            # unspecified case -> advance j to avoid stalling
            j += 1
            break

    out: Dict[str, str] = {}
    for i in range(n):
        if Qx[i] is not None:
            out[f"Q_{i + 1}^x"] = symp.to_str(Qx[i])
        if Qy[i] is not None:
            out[f"Q_{i + 1}^y"] = symp.to_str(Qy[i])
        if Qz[i] is not None:
            out[f"Q_{i + 1}^z"] = symp.to_str(Qz[i])
    return out


def recover_Q_from_ranking_primes(
        ranking: Ranking,
        n: Optional[int] = None,
        max_steps: Optional[int] = None,
) -> Dict[str, str]:
    """Recover {Q_i^x,Q_i^y,Q_i^z} following the described algorithm (phase ignored).

    Fix implemented:
      - Any candidate P assigned to Q_i^α must commute with ALL already-assigned Q_k^β for k != i.
        (In particular, this prevents picking something like IXX for Q_3^x when Q_1 has IYI/IZI.)
    """
    if not ranking:
        raise ValueError("ranking is empty.")
    if n is None:
        n = len(ranking[0][1])
    if n <= 0:
        raise ValueError("n must be positive.")
    for _, P in ranking:
        if len(P) != n or any(ch not in PAULI_CHARS for ch in P):
            raise ValueError("All Paulis must be length-n strings over {I,X,Y,Z}.")

    symp = Symplectic(n)
    S = GF2Basis()

    Qx: List[Optional[int]] = [None] * n
    Qy: List[Optional[int]] = [None] * n
    Qz: List[Optional[int]] = [None] * n

    def all_none(i: int) -> bool:
        return Qx[i] is None and Qy[i] is None and Qz[i] is None

    def all_assigned() -> bool:
        return all((Qx[i] is not None and Qy[i] is not None and Qz[i] is not None) for i in range(n))

    def assigned_all_except(i_ex: int) -> List[int]:
        """All currently assigned operators except those at site i_ex."""
        ops: List[int] = []
        for k in range(n):
            if k == i_ex:
                continue
            for v in (Qx[k], Qy[k], Qz[k]):
                if v is not None:
                    ops.append(v)
        return ops

    def commutes_with_all_except_site(P: int, i_ex: int) -> bool:
        """Cross-commutation constraint: P must commute with every already-assigned Q_k^β for k != i_ex."""
        return all(symp.commutes_int(P, v) for v in assigned_all_except(i_ex))

    j = 0
    steps = 0
    max_steps = max_steps if max_steps is not None else len(ranking) * (n + 2)

    while j < len(ranking) and not all_assigned() and steps < max_steps:
        steps += 1
        Pj = symp.to_int(ranking[j][1])

        # Step 2: skip if in span S
        if S.contains(Pj):
            j += 1
            continue

        i = 0
        while True:
            # 4(a): assign Q_i^x
            # FIX: require cross-commutation with all already-assigned operators on other sites.
            if all_none(i):
                if commutes_with_all_except_site(Pj, i):
                    Qx[i] = Pj
                    S.add(Pj)
                    j += 1
                else:
                    j += 1
                break

            # 4(b): assign Q_i^z (and then Q_i^y = Q_i^x ^ Q_i^z)
            # FIX: also require cross-commutation with all other sites (not just "assigned_except(i)" ad hoc).
            if Qx[i] is not None and Qz[i] is None:
                if (
                        symp.anticommutes_int(Pj, Qx[i])
                        and commutes_with_all_except_site(Pj, i)
                ):
                    Qz[i] = Pj
                    Qy[i] = Qx[i] ^ Qz[i]
                    S.add(Pj)
                    j += 1
                    break

            # 4(c): if commutes with Q_i^x, move to next i
            if Qx[i] is not None and symp.commutes_int(Pj, Qx[i]):
                if i == n - 1:
                    j += 1
                    break
                i += 1
                continue

            # unspecified case -> advance j to avoid stalling
            j += 1
            break

    out: Dict[str, str] = {}
    for i in range(n):
        if Qx[i] is not None:
            out[f"Q_{i + 1}^x"] = symp.to_str(Qx[i])
        if Qy[i] is not None:
            out[f"Q_{i + 1}^y"] = symp.to_str(Qy[i])
        if Qz[i] is not None:
            out[f"Q_{i + 1}^z"] = symp.to_str(Qz[i])
    return out


def recover_Q_from_ranking(
        ranking: Ranking,
        n: Optional[int] = None,
        max_steps: Optional[int] = None,
        disp: bool = False,
) -> Dict[str, str]:
    """Recover {Q_i^x,Q_i^y,Q_i^z} following the described algorithm (phase ignored).

    Args:
        ranking: Ranked list of Pauli operators.
        n: Number of qubits. If None, inferred from the first Pauli string.
        max_steps: Maximum number of loop iterations.
        disp: If True, print the action taken on each operator during recovery.

    Fix implemented:
      - Any candidate P assigned to Q_i^α must commute with ALL already-assigned
        Q_k^β for k != i. In particular, this prevents picking something like IXX
        for Q_3^x when Q_1 has IYI/IZI.
    """
    if not ranking:
        raise ValueError("ranking is empty.")
    if n is None:
        n = len(ranking[0][1])
    if n <= 0:
        raise ValueError("n must be positive.")
    for _, P in ranking:
        if len(P) != n or any(ch not in PAULI_CHARS for ch in P):
            raise ValueError("All Paulis must be length-n strings over {I,X,Y,Z}.")

    symp = Symplectic(n)
    S = GF2Basis()

    Qx: List[Optional[int]] = [None] * n
    Qy: List[Optional[int]] = [None] * n
    Qz: List[Optional[int]] = [None] * n

    def show(msg: str) -> None:
        if disp:
            print(msg)

    def pstr(P: int) -> str:
        return symp.to_str(P)

    def all_none(i: int) -> bool:
        return Qx[i] is None and Qy[i] is None and Qz[i] is None

    def all_assigned() -> bool:
        return all((Qx[i] is not None and Qy[i] is not None and Qz[i] is not None) for i in range(n))

    def assigned_all_except(i_ex: int) -> List[int]:
        """All currently assigned operators except those at site i_ex."""
        ops: List[int] = []
        for k in range(n):
            if k == i_ex:
                continue
            for v in (Qx[k], Qy[k], Qz[k]):
                if v is not None:
                    ops.append(v)
        return ops

    def commutes_with_all_except_site(P: int, i_ex: int) -> bool:
        """Cross-commutation constraint: P must commute with every already-assigned Q_k^β for k != i_ex."""
        return all(symp.commutes_int(P, v) for v in assigned_all_except(i_ex))

    j = 0
    steps = 0
    max_steps = max_steps if max_steps is not None else len(ranking) * (n + 2)

    while j < len(ranking) and not all_assigned() and steps < max_steps:
        steps += 1
        score_j, Pj_str = ranking[j]
        Pj = symp.to_int(Pj_str)

        show(f"Considering ranking[{j}]: {Pj_str} (score={score_j})")

        # Step 2: skip if in span S
        if S.contains(Pj):
            show(f"  Discarded {Pj_str} because it is already in the span S.")
            j += 1
            continue

        i = 0
        while True:
            # 4(a): assign Q_i^x
            if all_none(i):
                if commutes_with_all_except_site(Pj, i):
                    Qx[i] = Pj
                    S.add(Pj)
                    show(f"  Assigned {Pj_str} as Q_{i + 1}^x.")
                    j += 1
                else:
                    show(
                        f"  Discarded {Pj_str} for Q_{i + 1}^x because it does not commute "
                        f"with all already-assigned operators on other sites."
                    )
                    j += 1
                break

            # 4(b): assign Q_i^z and Q_i^y = Q_i^x ^ Q_i^z
            if Qx[i] is not None and Qz[i] is None:
                if not symp.anticommutes_int(Pj, Qx[i]):
                    show(
                        f"  Cannot assign {Pj_str} as Q_{i + 1}^z because it does not "
                        f"anticommute with Q_{i + 1}^x={pstr(Qx[i])}."
                    )
                elif not commutes_with_all_except_site(Pj, i):
                    show(
                        f"  Discarded {Pj_str} for Q_{i + 1}^z because it fails the "
                        f"cross-commutation constraint with other sites."
                    )
                else:
                    Qz[i] = Pj
                    Qy[i] = Qx[i] ^ Qz[i]
                    S.add(Pj)
                    show(f"  Assigned {Pj_str} as Q_{i + 1}^z.")
                    show(f"  Inferred {pstr(Qy[i])} as Q_{i + 1}^y = Q_{i + 1}^x * Q_{i + 1}^z (phase ignored).")
                    j += 1
                    break

            # 4(c): if commutes with Q_i^x, move to next i
            if Qx[i] is not None and symp.commutes_int(Pj, Qx[i]):
                if i == n - 1:
                    show(
                        f"  Discarded {Pj_str} because it commutes with Q_{i + 1}^x={pstr(Qx[i])} "
                        f"and no later site is available."
                    )
                    j += 1
                    break
                show(
                    f"  {Pj_str} commutes with Q_{i + 1}^x={pstr(Qx[i])}; "
                    f"moving to site {i + 2}."
                )
                i += 1
                continue

            # unspecified case -> advance j to avoid stalling
            if Qx[i] is not None:
                show(
                    f"  Discarded {Pj_str} at site {i + 1} because it does not fit the "
                    f"assignment rules."
                )
            else:
                show(f"  Discarded {Pj_str} because it does not fit the assignment rules.")
            j += 1
            break

    if disp:
        if all_assigned():
            print("Recovery finished: all Q_i^x, Q_i^y, Q_i^z assigned.")
        elif steps >= max_steps:
            print("Recovery stopped because max_steps was reached.")
        else:
            print("Recovery stopped because the ranking was exhausted.")

    out: Dict[str, str] = {}
    for i in range(n):
        if Qx[i] is not None:
            out[f"Q_{i + 1}^x"] = symp.to_str(Qx[i])
        if Qy[i] is not None:
            out[f"Q_{i + 1}^y"] = symp.to_str(Qy[i])
        if Qz[i] is not None:
            out[f"Q_{i + 1}^z"] = symp.to_str(Qz[i])
    return out

# ----------------------------
# Check recovered Q
# ----------------------------
def check_recovered_Q(
        Q: Dict[str, str],
        n: int,
        require_all_assigned: bool = True,
) -> Dict[str, object]:
    """
    Checks:
      (1) each Q_i^alpha is a valid Pauli string
      (2) within each i: x,y,z pairwise anticommute
      (3) across i!=j: all commute
    """
    required = [f"Q_{i}^{a}" for i in range(1, n + 1) for a in ("^x", "^y", "^z")]  # not used
    required_keys = [f"Q_{i}^x" for i in range(1, n + 1)] + [f"Q_{i}^y" for i in range(1, n + 1)] + [f"Q_{i}^z" for i in
                                                                                                     range(1, n + 1)]

    missing_keys = [k for k in required_keys if k not in Q]
    if require_all_assigned and missing_keys:
        return {"ok": False, "missing_keys": missing_keys, "invalid_paulis": [], "anticommute_violations": [],
                "commute_violations": []}

    def valid_pauli(s: str) -> bool:
        return isinstance(s, str) and len(s) == n and all(ch in PAULI_CHARS for ch in s)

    invalid_paulis = [(k, v) for k, v in Q.items() if not valid_pauli(v)]
    if invalid_paulis:
        return {"ok": False, "missing_keys": missing_keys, "invalid_paulis": invalid_paulis,
                "anticommute_violations": [], "commute_violations": []}

    symp = Symplectic(n)

    anticommute_violations: List[Tuple[int, str, str]] = []
    for i in range(1, n + 1):
        Px, Py, Pz = Q.get(f"Q_{i}^x"), Q.get(f"Q_{i}^y"), Q.get(f"Q_{i}^z")
        if Px is None or Py is None or Pz is None:
            continue
        if not symp.anticommutes(Px, Py):
            anticommute_violations.append((i, "x", "y"))
        if not symp.anticommutes(Py, Pz):
            anticommute_violations.append((i, "y", "z"))
        if not symp.anticommutes(Pz, Px):
            anticommute_violations.append((i, "z", "x"))

    commute_violations: List[Tuple[Tuple[int, str], Tuple[int, str]]] = []
    labels = ("x", "y", "z")
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            for a in labels:
                Pa = Q.get(f"Q_{i}^{a}")
                if Pa is None:
                    continue
                for b in labels:
                    Pb = Q.get(f"Q_{j}^{b}")
                    if Pb is None:
                        continue
                    if not symp.commutes(Pa, Pb):
                        commute_violations.append(((i, a), (j, b)))

    ok = (
                 not missing_keys or not require_all_assigned) and not invalid_paulis and not anticommute_violations and not commute_violations
    return {
        "ok": ok,
        "missing_keys": missing_keys,
        "invalid_paulis": invalid_paulis,
        "anticommute_violations": anticommute_violations,
        "commute_violations": commute_violations,
    }


def check_recovered_Q_product_state(
        Q: Dict[str, str],
        n: int,
        require_all_assigned: bool = True,
) -> Dict[str, object]:
    """
    Specialized checker for the *product-state* structure:

    Requirements:
      (1) Every Q_i^alpha is a *single-site* Pauli: exactly one of {X,Y,Z} and the rest 'I'.
      (2) For each i, the triple (Q_i^x, Q_i^y, Q_i^z) acts on the same site and equals
          {X_j, Y_j, Z_j} for some j (order may be permuted across alpha).
      (3) Sites are a permutation: different i map to different j (bijection i -> site j).

    Args:
      Q: dict like {"Q_1^x": "IXII...", "Q_1^y": "...", ...}
      n: number of qubits (length of Pauli strings)
      require_all_assigned: if True, require all 3n keys present

    Returns:
      report dict with:
        - ok: bool
        - missing_keys: list[str]
        - invalid_paulis: list[(key, value)]
        - non_single_site: list[(key, value)]
        - triple_site_mismatch: list[int]              # i where x/y/z act on different sites
        - triple_not_xyz: list[(i, site, letters)]     # letters not exactly {X,Y,Z}
        - site_collisions: list[(i1, i2, site)]        # two i's mapped to same site
        - site_map: dict[int,int]                      # i -> site (1-indexed)
        - alpha_permutation: dict[int, dict[str,str]]  # i -> mapping alpha -> letter at that site
    """

    def expected_keys(n_: int) -> List[str]:
        return [f"Q_{i}^x" for i in range(1, n_ + 1)] + \
            [f"Q_{i}^y" for i in range(1, n_ + 1)] + \
            [f"Q_{i}^z" for i in range(1, n_ + 1)]

    req = expected_keys(n)
    missing_keys = [k for k in req if k not in Q]
    if require_all_assigned and missing_keys:
        return {
            "ok": False,
            "missing_keys": missing_keys,
            "invalid_paulis": [],
            "non_single_site": [],
            "triple_site_mismatch": [],
            "triple_not_xyz": [],
            "site_collisions": [],
            "site_map": {},
            "alpha_permutation": {},
        }

    # --- basic validity ---
    def is_valid_pauli(s: str) -> bool:
        return isinstance(s, str) and len(s) == n and all(ch in "IXYZ" for ch in s)

    invalid_paulis: List[Tuple[str, str]] = []
    for k, v in Q.items():
        if not is_valid_pauli(v):
            invalid_paulis.append((k, v))

    if invalid_paulis:
        return {
            "ok": False,
            "missing_keys": missing_keys,
            "invalid_paulis": invalid_paulis,
            "non_single_site": [],
            "triple_site_mismatch": [],
            "triple_not_xyz": [],
            "site_collisions": [],
            "site_map": {},
            "alpha_permutation": {},
        }

    # --- single-site extraction ---
    def single_site_info(P: str) -> Tuple[bool, int, str]:
        """
        Returns (is_single_site, site_index_1based, letter_at_site).
        If not single-site, site_index_1based=-1, letter=""
        """
        nonI = [(idx, ch) for idx, ch in enumerate(P) if ch != "I"]
        if len(nonI) != 1:
            return False, -1, ""
        idx0, ch0 = nonI[0]
        if ch0 not in ("X", "Y", "Z"):
            return False, -1, ""
        return True, idx0 + 1, ch0  # 1-based

    non_single_site: List[Tuple[str, str]] = []

    # For each i, gather x/y/z single-site info
    triple_site_mismatch: List[int] = []
    triple_not_xyz: List[Tuple[int, int, str]] = []  # (i, site, letters_str)
    site_map: Dict[int, int] = {}  # i -> site (1-based)
    alpha_perm: Dict[int, Dict[str, str]] = {}  # i -> {'x': 'X', 'y':'Z', 'z':'Y'} etc.

    for i in range(1, n + 1):
        Px = Q.get(f"Q_{i}^x")
        Py = Q.get(f"Q_{i}^y")
        Pz = Q.get(f"Q_{i}^z")

        # If not required, allow missing triples and just skip
        if Px is None or Py is None or Pz is None:
            if require_all_assigned:
                triple_site_mismatch.append(i)
            continue

        okx, sx, lx = single_site_info(Px)
        oky, sy, ly = single_site_info(Py)
        okz, sz, lz = single_site_info(Pz)

        if not okx:
            non_single_site.append((f"Q_{i}^x", Px))
        if not oky:
            non_single_site.append((f"Q_{i}^y", Py))
        if not okz:
            non_single_site.append((f"Q_{i}^z", Pz))

        # If any are not single-site, can't proceed reliably for this i
        if not (okx and oky and okz):
            continue

        # Must act on same site
        if not (sx == sy == sz):
            triple_site_mismatch.append(i)
            continue

        letters = (lx, ly, lz)
        if set(letters) != {"X", "Y", "Z"}:
            triple_not_xyz.append((i, sx, "".join(letters)))
            continue

        site_map[i] = sx
        alpha_perm[i] = {"x": lx, "y": ly, "z": lz}

    # --- site permutation (bijection) ---
    site_collisions: List[Tuple[int, int, int]] = []
    inv: Dict[int, int] = {}
    for i, s in site_map.items():
        if s in inv:
            site_collisions.append((inv[s], i, s))
        else:
            inv[s] = i

    ok = (
            (not missing_keys or not require_all_assigned)
            and not invalid_paulis
            and not non_single_site
            and not triple_site_mismatch
            and not triple_not_xyz
            and not site_collisions
            and (len(site_map) == n if require_all_assigned else True)
    )

    return {
        "ok": ok,
        "missing_keys": missing_keys,
        "invalid_paulis": invalid_paulis,
        "non_single_site": non_single_site,
        "triple_site_mismatch": triple_site_mismatch,
        "triple_not_xyz": triple_not_xyz,
        "site_collisions": site_collisions,
        "site_map": site_map,
        "alpha_permutation": alpha_perm,
    }




def pauli_to_symplectic_col(P: str) -> np.ndarray:
    """
    P in {'I','X','Y','Z'}^n  ->  [a_1..a_n | b_1..b_n]^T in GF(2)^(2n),
    where Y -> (a_i,b_i)=(1,1).
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


def symplectic_matrix_from_recover_output(Qhat: dict) -> np.ndarray:
    """
    Build 2n x 2n binary symplectic matrix S with columns:
      [Q_1^x ... Q_n^x | Q_1^z ... Q_n^z]
    from recover_Q_from_ranking output like:
      {'Q_1^x': 'IXI', 'Q_1^y': 'IYI', 'Q_1^z': 'IZI', ... }.
    """
    if not isinstance(Qhat, dict):
        raise TypeError("Qhat must be a dict.")

    # infer n from any value
    try:
        any_pauli = next(iter(Qhat.values()))
    except StopIteration:
        raise ValueError("Empty dict.")

    if not isinstance(any_pauli, str):
        raise ValueError("Dict values must be Pauli strings.")
    n = len(any_pauli)

    # collect Q_i^x and Q_i^z
    pat = re.compile(r"^Q_(\d+)\^([xyz])$")
    Qx = {}
    Qz = {}

    for k, P in Qhat.items():
        m = pat.match(k)
        if m is None:
            continue
        i = int(m.group(1))
        alpha = m.group(2)

        if not isinstance(P, str) or len(P) != n or (set(P) - set("IXYZ")):
            raise ValueError(f"Invalid Pauli for key '{k}': '{P}' (expected length {n} over IXYZ).")

        if alpha == "x":
            Qx[i] = P
        elif alpha == "z":
            Qz[i] = P

    if not Qx or not Qz:
        raise ValueError("Did not find any Q_i^x / Q_i^z keys.")

    n_from_keys = max(max(Qx.keys()), max(Qz.keys()))
    if len(Qx) != n_from_keys or len(Qz) != n_from_keys:
        missing_x = [i for i in range(1, n_from_keys + 1) if i not in Qx]
        missing_z = [i for i in range(1, n_from_keys + 1) if i not in Qz]
        raise ValueError(f"Missing assignments: x-missing={missing_x}, z-missing={missing_z}")

    if n_from_keys != n:
        # Your Pauli strings should be length-n qubits. This catches inconsistency.
        raise ValueError(f"Inconsistent n: inferred from strings n={n}, but keys go up to {n_from_keys}.")

    S = np.zeros((2 * n, 2 * n), dtype=np.uint8)

    # first n columns: Q_1^x..Q_n^x
    for i in range(1, n + 1):
        S[:, i - 1] = pauli_to_symplectic_col(Qx[i])

    # last n columns: Q_1^z..Q_n^z
    for i in range(1, n + 1):
        S[:, n + (i - 1)] = pauli_to_symplectic_col(Qz[i])

    return S


def is_symplectic(F: np.ndarray) -> bool:
    F = F % 2
    n = F.shape[0] // 2
    J = np.block([[np.zeros((n, n), dtype=np.uint8), np.eye(n, dtype=np.uint8)],
                  [np.eye(n, dtype=np.uint8), np.zeros((n, n), dtype=np.uint8)]])
    return np.array_equal((F.T @ J @ F) % 2, J)

import numpy as np
from itertools import product
from typing import Iterable, Optional, Tuple, List


# ===========================
#  Basic symplectic helpers
# ===========================
def is_symplectic(F: np.ndarray) -> bool:
    F = F % 2
    n = F.shape[0] // 2
    J = np.block([[np.zeros((n, n), dtype=np.uint8), np.eye(n, dtype=np.uint8)],
                  [np.eye(n, dtype=np.uint8), np.zeros((n, n), dtype=np.uint8)]])
    return np.array_equal((F.T @ J @ F) % 2, J)


def apply_cnot_right(F: np.ndarray, c: int, t: int) -> np.ndarray:
    n = F.shape[0] // 2
    x, z = F[:n, :], F[n:, :]
    x[t, :] ^= x[c, :]
    z[c, :] ^= z[t, :]
    return F


def apply_h_right(F: np.ndarray, j: int) -> np.ndarray:
    n = F.shape[0] // 2
    F[j, :], F[n + j, :] = F[n + j, :].copy(), F[j, :].copy()
    return F


def apply_s_right(F: np.ndarray, j: int) -> np.ndarray:
    n = F.shape[0] // 2
    F[n + j, :] ^= F[j, :]
    return F


# ===========================
#  Tableau synthesis (gates)
# ===========================
def synthesize_clifford_from_tableau(F_in: np.ndarray) -> List[Tuple]:
    """
    Input  : F_in (2n x 2n, uint8 in {0,1}), columns [X1..Xn | Z1..Zn]
    Output : gate list [('H',j), ('S',j), ('CNOT',c,t)] in the order they should be applied.
    """
    F = (F_in.copy() % 2).astype(np.uint8)
    n = F.shape[0] // 2
    assert F.shape == (2 * n, 2 * n), "F must be 2n x 2n"
    if not is_symplectic(F):
        raise ValueError("Input is not symplectic (F^T J F != J).")

    gates: List[Tuple] = []

    # record=False prevents appending to `gates` (used for trials)
    def H(j: int, record: bool = True):
        apply_h_right(F, j)
        if record:
            gates.append(("H", j))

    def S(j: int):
        apply_s_right(F, j)
        gates.append(("S", j))

    def CNOT(c: int, t: int):
        if c == t:
            return
        apply_cnot_right(F, c, t)
        gates.append(("CNOT", c, t))

    # Phase 0: ensure X-block (first n columns, upper n rows) is full rank using H flips if needed
    def rank_gf2(M):
        M = (M % 2).copy()
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

    flipped: set[int] = set()  # qubits that have been COMMITTED (H recorded) in Phase 0
    target_rank = n

    while rank_gf2(F[:n, :n]) < target_rank:
        improved = False
        tried: set[int] = set()  # only for this greedy pass

        # Greedy: find a single H(j) that immediately increases rank
        for j in range(n):
            if j in tried:
                continue
            F_backup = F.copy()
            prev_rank = rank_gf2(F_backup[:n, :n])

            H(j, record=False)  # trial: mutate F only, do not record
            new_rank = rank_gf2(F[:n, :n])

            if new_rank > prev_rank:
                # commit this H: record gate and remember parity
                gates.append(("H", j))
                flipped.add(j)
                improved = True
                break
            else:
                # rollback in place; mark j as tried for THIS pass
                F[...] = F_backup
                tried.add(j)

        if not improved:
            # Fallback: flip any qubit that hasn't been COMMITTED yet.
            # (We purposely ignore `tried` here; combinations may be needed.)
            for j in range(n):
                if j not in flipped:
                    H(j, record=True)
                    flipped.add(j)
                    if rank_gf2(F[:n, :n]) == target_rank:
                        improved = True
                        break

        if not improved and rank_gf2(F[:n, :n]) < target_rank:
            raise RuntimeError("Could not make X-block full rank via H flips.")

    # Phase 1: reduce A (X-part of X-columns) to identity with CNOTs (column-wise elimination)
    for i in range(n):
        A = F[:n, :n]
        if A[i, i] == 0:
            found = False
            for j in range(n):
                if j == i:
                    continue
                if A[j, i] == 1 and j > i:
                    CNOT(j, i)
                    found = True
                    break
            if not found:
                raise RuntimeError(f"No pivot to set A[{i},{i}]=1.")
        for j in range(n):
            if j == i:
                continue
            if F[:n, :n][j, i] == 1:
                CNOT(i, j)  # zero A[i,j]
    # print("Phase 1: ", F[:n, :n])

    # Phase 2: clean Z-leakage in X-columns (make each colX(i) = (e_i | 0))
    for i in range(n):
        cx = i
        if F[n + i, cx] == 1:  # z_i bit
            S(i)
        for j in range(n):
            if j == i:
                continue
            if F[n + j, cx] == 1:  # other z_j
                H(j)
                CNOT(i, j)
                H(j)

    # Phase 3: make Z-columns canonical (x=0, z=e_i)

    """
        for i in range(n):
        cz = n + i
        if F[n + i, cz] == 0:
            if F[i, cz] == 1:
                H(i)  # move x_i->z_i in this column
            else:
                # borrow from some z_j
                got = False
                for j in range(n):
                    if j == i:
                        continue
                    if F[n + j, cz] == 1:
                        H(j)
                        CNOT(i, j)
                        H(j)
                        got = True
                        break
                if not got:
                    # minimal local toggle (valid F rarely reaches here)
                    S(i)
        # zero other z_j
        for j in range(n):
            if j == i:
                continue
            if F[n + j, cz] == 1:
                H(j)
                CNOT(i, j)
                H(j)

        for j in range(n):
        cz = n+j
        for i in range(n):
            if F[i,cz]==1:
                CNOT()

        if F[j, cz] == 1:
            H(j)
            S(j)
            H(j)
    if F[n + i, cz] == 0:
        S(i)
    """
    # zero any x-part in this Z column
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
    # print("Phase 3: ", F)

    # (Optional) Peephole: cancel adjacent HH/SS or duplicate CNOTs here if desired.
    return gates


# ===========================
#  Build the unitary matrix
# ===========================
def kron_n(*ops):
    """Kronecker product of a list of single/two-qubit operators in given order (left to right)."""
    out = np.array([[1]], dtype=complex)
    for op in ops:
        out = np.kron(out, op)
    return out


def single_qubit_op(n: int, j: int, M: np.ndarray) -> np.ndarray:
    """Embed 2x2 gate M on line j (0-based, leftmost = qubit 0) into 2^n space."""
    ops = []
    for q in range(n):
        ops.append(M if q == j else np.eye(2, dtype=complex))
    return kron_n(*ops)


def two_qubit_cnot(n: int, c: int, t: int) -> np.ndarray:
    """Embed CNOT_{c->t} on n qubits (0-based indexing, leftmost = qubit 0)."""
    I2 = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)
    term0_ops, term1_ops = [], []
    for q in range(n):
        if q == c:
            term0_ops.append(P0);
            term1_ops.append(P1)
        elif q == t:
            term0_ops.append(I2);
            term1_ops.append(X)
        else:
            term0_ops.append(I2);
            term1_ops.append(I2)
    return kron_n(*term0_ops) + kron_n(*term1_ops)


def gates_to_unitary(gates: List[Tuple], n: int) -> np.ndarray:
    """Build the 2^n x 2^n unitary by multiplying gates in the given order."""
    Hm = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    Sm = np.array([[1, 0], [0, 1j]], dtype=complex)
    U = np.eye(2 ** n, dtype=complex)
    for g in gates:
        if g[0] == "H":
            U = single_qubit_op(n, g[1], Hm) @ U
        elif g[0] == "S":
            U = single_qubit_op(n, g[1], Sm) @ U
        elif g[0] == "CNOT":
            U = two_qubit_cnot(n, g[1], g[2]) @ U
        else:
            raise ValueError(f"Unknown gate: {g}")
    return U


# ===========================
#  Main convenience wrapper
# ===========================
def synthesize_and_unitary(F: np.ndarray):
    """
    Given a symplectic tableau F (2n x 2n), return:
      - gates: synthesis sequence over {H, S, CNOT}
      - U    : 2^n x 2^n unitary matrix implementing those gates
    """
    n = F.shape[0] // 2
    gates = synthesize_clifford_from_tableau(F)
    U = gates_to_unitary(gates, n)
    return gates, np.conj(U.T)


# ===========================
#  Pretty print (optional)
# ===========================
def format_gates(gates: List[Tuple]) -> str:
    lines = []
    for g in gates:
        if g[0] == "H":
            lines.append(f"H {g[1] + 1}")
        elif g[0] == "S":
            lines.append(f"S {g[1] + 1}")
        elif g[0] == "CNOT":
            lines.append(f"CNOT {g[1] + 1}->{g[2] + 1}")
    return "\n".join(lines)


# ---------- Basic Pauli machinery ----------

I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_all(mats):
    """Kronecker product over a list of matrices, left-to-right."""
    out = np.array([[1]], dtype=complex)
    for M in mats:
        out = np.kron(out, M)
    return out


def pauli_from_xz(x, z, *, qubit_order="MSB"):
    """
    Build the Hermitian Pauli matrix σ(x,z) = i^{x·z} ⊗_j (X^x_j Z^z_j),
    where the single-qubit factor is (X then Z) on each site.
    - x, z: 1D arrays of 0/1 of length n
    - qubit_order:
        "MSB": x[0] is leftmost tensor factor (most significant qubit)
        "LSB": x[0] is rightmost tensor factor (least significant qubit)
    """
    x = np.asarray(x, dtype=int)
    z = np.asarray(z, dtype=int)
    assert x.shape == z.shape
    n = x.size

    # Choose iteration order for tensoring
    idxs = range(n) if qubit_order == "MSB" else range(n - 1, -1, -1)

    mats = []
    for j in idxs:
        Mj = I
        if x[j] == 1:
            Mj = X @ Mj
        if z[j] == 1:
            Mj = Mj @ Z  # XZ if both=1
        mats.append(Mj)
    phase = (1j) ** int(np.dot(x, z))  # i^{x·z}
    return phase * kron_all(mats)


# ---------- GF(2) helpers & symplectic form ----------

def mod2(a):  # elementwise mod 2
    return np.mod(a, 2).astype(int)


# ---------- Conjugation & phase extraction ----------

def best_global_phase(A, B):
    """
    Return alpha minimizing ||A - alpha B||_F under Hilbert-Schmidt inner product.
    alpha = tr(B^† A)/d, where d = 2^n.
    """
    d = A.shape[0]
    alpha = np.trace(B.conj().T @ A) / d
    if np.abs(alpha) > 0:
        alpha /= np.abs(alpha)  # put it on the unit circle
    return alpha


def is_unitary(U, tol=1e-10):
    d = U.shape[0]
    return np.allclose(U.conj().T @ U, np.eye(d), atol=tol) and np.allclose(U @ U.conj().T, np.eye(d), atol=tol)


# ---------- Main verification ----------

def verify_clifford_action(F, U, *, side="left", qubit_order="MSB",
                           check_generators=True, random_trials=0, check_all=False,
                           tol=1e-8, verbose=True):
    """
    Verify that conjugation by U maps Hermitian Paulis according to F over GF(2).

    Conventions:
      - Pauli vectors are concatenated as a = [x; z] (x then z), each in {0,1}^n.
      - 'side' chooses whether we compute a' = F @ a  (side='left', COLUMN convention)
                            or a' = a @ F  (side='right', ROW convention), all mod 2.
      - 'qubit_order' governs tensor order in σ(x,z):
            'MSB' -> x[0] is leftmost factor; 'LSB' -> x[0] is rightmost factor.
      - We check equality up to a global phase (±1, ±i) via HS inner product.

    Args:
      F: int ndarray shape (2n, 2n) over GF(2)
      U: complex ndarray shape (2^n, 2^n), the Clifford unitary
      check_generators: verify on the 2n generators {X_i, Z_i}
      random_trials: additional random Paulis to test (useful for larger n)
      check_all: if True and n<=4, test all 4^n Paulis
      tol: numerical tolerance
      verbose: print a short report

    Returns dict with success flag and details.
    """
    F = mod2(F)
    m = F.shape[0]
    assert F.shape[0] == F.shape[1] and m % 2 == 0, "F must be 2n x 2n over GF(2)."
    n = m // 2
    d = 2 ** n
    assert U.shape == (d, d), "U must be 2^n x 2^n."

    if verbose:
        if not is_unitary(U, tol=max(1e-10, tol / 10)):
            print("Warning: U is not perfectly unitary within tolerance.")
        if not is_symplectic(F):
            print("Warning: F is not symplectic over GF(2) (F^T Ω F ≠ Ω).")

    def apply_F(a):
        if side == "left":
            return mod2(F @ a)
        elif side == "right":
            return mod2(a @ F)
        else:
            raise ValueError("side must be 'left' or 'right'")

    def U_conj(P):
        return U @ P @ U.conj().T

    failures = []
    checks = []

    # Build the set of (x,z) to test
    tests = []

    if check_generators:
        for i in range(n):
            x = np.zeros(n, dtype=int);
            x[i] = 1
            z = np.zeros(n, dtype=int)
            tests.append((x, z, f"X_{i}"))
        for i in range(n):
            x = np.zeros(n, dtype=int)
            z = np.zeros(n, dtype=int);
            z[i] = 1
            tests.append((x, z, f"Z_{i}"))

    if check_all:
        if n > 4:
            raise ValueError("check_all is only feasible for n<=4 (4^n Paulis).")
        for x in product([0, 1], repeat=n):
            for z in product([0, 1], repeat=n):
                tests.append((np.array(x, int), np.array(z, int), "ALL"))

    if random_trials > 0:
        rng = np.random.default_rng(1234)
        for _ in range(random_trials):
            x = rng.integers(0, 2, size=n, dtype=int)
            z = rng.integers(0, 2, size=n, dtype=int)
            tests.append((x, z, "RAND"))

    # Deduplicate while preserving order
    seen = set()
    uniq_tests = []
    for x, z, tag in tests:
        key = (tuple(x.tolist()), tuple(z.tolist()))
        if key not in seen:
            seen.add(key)
            uniq_tests.append((x, z, tag))
    tests = uniq_tests

    for (x, z, tag) in tests:
        a = np.concatenate([x, z])  # [x; z], column convention in memory
        a_prime = apply_F(a)
        x_p = a_prime[:n]
        z_p = a_prime[n:]

        P = pauli_from_xz(x, z, qubit_order=qubit_order)
        P_target = pauli_from_xz(x_p, z_p, qubit_order=qubit_order)
        P_conj = U_conj(P)

        alpha = best_global_phase(P_conj, P_target)

        ok = np.allclose(P_conj, alpha * P_target, atol=tol)
        if not ok:
            # Try relaxing to allow tiny magnitude error on alpha computation
            num = np.trace(P_target.conj().T @ P_conj) / (2 ** n)
            if np.abs(num) > 1e-12:
                alpha2 = num / np.abs(num)
                ok = np.allclose(P_conj, alpha2 * P_target, atol=tol)
                alpha = alpha2
        checks.append((tag, (x.copy(), z.copy()), (x_p.copy(), z_p.copy()), alpha, ok))
        if not ok:
            failures.append((x.copy(), z.copy(), x_p.copy(), z_p.copy(), alpha))

    success = (len(failures) == 0)
    if verbose:
        total = len(checks)
        passed = sum(int(ok) for *_, ok in checks)
        print(f"Verification summary: {passed}/{total} cases passed "
              f"(n={n}, side='{side}', qubit_order='{qubit_order}', tol={tol}).")
        if failures:
            print("Example failure (up to global phase):")
            fx, fz, fxp, fzp, alph = failures[0]
            print(f"  Input (x,z): {fx}, {fz}")
            print(f"  Expected F·(x,z)->(x',z'): {fxp}, {fzp}")
            print(f"  Best phase α ≈ {alph}")

    return {
        "success": success,
        "n": n,
        "side": side,
        "qubit_order": qubit_order,
        "tol": tol,
        "num_checks": len(checks),
        "num_failures": len(failures),
        "failures": failures,  # list of tuples for debugging
        "details": checks  # per-test details
    }


# ---------- (Optional) tiny helper to build U from a gate list ----------

def one_qubit_on_n(gate2x2, n, q, *, qubit_order="MSB"):
    """
    Embed a 1-qubit 2x2 gate on qubit index q into n-qubit space.
    q is 0..n-1 in the same 'qubit_order' convention as pauli_from_xz.
    """
    mats = []
    for j in (range(n) if qubit_order == "MSB" else range(n - 1, -1, -1)):
        mats.append(gate2x2 if j == q else I)
    return kron_all(mats)


def cnot_on_n(n, control, target, *, qubit_order="MSB"):
    """
    Embed CNOT(control -> target) into n-qubit space.
    Uses projector decomposition: |0><0|⊗I + |1><1|⊗X acting on (control,target),
    lifted to n qubits via sparse-like composition.
    """
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)

    # Build projectors on control
    P0c = one_qubit_on_n(P0, n, control, qubit_order=qubit_order)
    P1c = one_qubit_on_n(P1, n, control, qubit_order=qubit_order)
    Xt = one_qubit_on_n(X, n, target, qubit_order=qubit_order)

    return P0c + P1c @ Xt


def random_symplectic(n: int, steps: int | None = None, seed: int | None = None) -> np.ndarray:
    """
    Return a random symplectic matrix F ∈ F2^{2n×2n} by composing random Clifford generators.
    - steps: number of random gate applications; default ~5n (can increase for more mixing)
    - seed : RNG seed for reproducibility
    """
    if steps is None:
        steps = 5 * n

    rng = np.random.default_rng(seed)
    F = np.eye(2 * n, dtype=np.uint8)

    for _ in range(max(1, steps)):
        g = rng.integers(0, 3)
        if g == 0:
            # H on a random qubit
            j = rng.integers(0, n)
            apply_h_right(F, j)
        elif g == 1:
            # S on a random qubit
            j = rng.integers(0, n)
            apply_s_right(F, j)
        else:
            # CNOT on two distinct qubits
            c = rng.integers(0, n)
            t = rng.integers(0, n - 1)
            if t >= c:
                t += 1
            apply_cnot_right(F, c, t)

    # (Optional) assertions
    # assert is_symplectic(F), "Generated matrix is not symplectic"
    # assert np.linalg.matrix_rank(F % 2) == 2*n, "Rank is not full over reals; use GF(2) rank if you prefer"
    return F


def apply_qubit_permutation(F: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Relabel qubits by permutation M (n×n). Returns F' = B F B^T over GF(2)."""
    F = (F % 2).astype(np.uint8)
    M = (M % 2).astype(np.uint8)
    n2 = F.shape[0];
    n = n2 // 2
    assert F.shape == (2 * n, 2 * n) and M.shape == (n, n)
    B = np.block([[M, np.zeros((n, n), dtype=np.uint8)],
                  [np.zeros((n, n), dtype=np.uint8), M]])
    Fp = (F @ B) % 2
    return Fp


def random_permutation_matrix(n: int, *, seed: int | None = None, dtype=np.uint8):
    """
    Return (M, p) where:
      - p is a random permutation of 0..n-1 sampled uniformly
      - M is the corresponding n×n permutation matrix with M[i, p[i]] = 1
    """
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)  # permutation vector
    M = np.zeros((n, n), dtype=dtype)
    M[p, np.arange(n)] = 1  # place ones
    return M, p





def _haar_pure_qubit(rng: np.random.Generator) -> np.ndarray:
    """Haar-random pure |psi> on C^2 as a normalized ket (2,)."""
    # Uniform on Bloch sphere: cosθ ~ U[-1,1], φ ~ U[0,2π)
    u = rng.uniform(-1.0, 1.0)
    phi = rng.uniform(0.0, 2.0 * np.pi)
    theta = np.arccos(u)
    return np.array([np.cos(theta / 2.0), np.exp(1j * phi) * np.sin(theta / 2.0)], dtype=complex)


def _rho_from_ket(psi: np.ndarray) -> np.ndarray:
    psi = psi.reshape(2, 1)
    return psi @ psi.conj().T


def _rho_from_bloch_vector(r: np.ndarray) -> np.ndarray:
    """ρ = 1/2 (I + r·σ), with ||r|| <= 1."""
    return 0.5 * (I2 + r[0] * X + r[1] * Y + r[2] * Z)


def _random_bloch_vector(rng: np.random.Generator, radius: Optional[float]) -> np.ndarray:
    """Random vector in the Bloch ball. If radius is None, sample uniformly in volume."""
    # Random direction
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    # Radius: uniform in volume => radius = U[0,1]^(1/3)
    if radius is None:
        r = rng.uniform() ** (1.0 / 3.0)
    else:
        r = float(np.clip(radius, 0.0, 1.0))
    return r * v


# ---------- n-qubit product builder ----------

def kron_all(mats: Iterable[np.ndarray]) -> np.ndarray:
    out = np.array([[1]], dtype=complex)
    for M in mats:
        out = np.kron(out, M)
    return out


def product_state(
        n: int,
        *,
        pure: bool = True,
        qubits: Optional[List[np.ndarray]] = None,
        mixed_radius: Optional[float] = None,
        seed: Optional[int] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Build an n-qubit product state ρ = ⊗_i ρ_i.

    Args:
      n: number of qubits.
      pure: if True, return a pure product (each ρ_i = |ψ_i><ψ_i|).
            if False, return a mixed product (each ρ_i has Bloch radius <= 1).
      qubits: optional list length n specifying each qubit.
              - If pure=True: list of kets |ψ_i> (shape (2,)).
              - If pure=False: list of 2x2 density matrices (each valid qubit state).
              If provided, these override random generation for the entries given.
      mixed_radius: for pure=False, set Bloch radius r in [0,1] for all random qubits.
                    If None, sample uniformly in Bloch-ball volume (r ~ U[0,1]^(1/3)).
      seed: RNG seed.

    Returns:
      (rho, psi) where:
        - rho: (2^n x 2^n) density matrix of the product state.
        - psi: (2^n,) statevector if pure=True and all inputs were pure; otherwise None.
    """
    rng = np.random.default_rng(seed)

    # Build single-qubit density matrices
    rhos: List[np.ndarray] = []
    kets: List[np.ndarray] = [] if pure else []

    if qubits is not None and len(qubits) != n:
        raise ValueError("`qubits` must be None or a list of length n.")

    for i in range(n):
        if pure:
            if qubits is not None and qubits[i] is not None:
                psi = np.asarray(qubits[i], dtype=complex).reshape(2)
            else:
                psi = _haar_pure_qubit(rng)
            kets.append(psi)
            rhos.append(_rho_from_ket(psi))
        else:
            if qubits is not None and qubits[i] is not None:
                rho_i = np.asarray(qubits[i], dtype=complex).reshape(2, 2)
            else:
                rvec = _random_bloch_vector(rng, mixed_radius)
                rho_i = _rho_from_bloch_vector(rvec)
            # small numerical cleanup to keep Hermitian/pos/trace
            rho_i = 0.5 * (rho_i + rho_i.conj().T)
            tr = np.trace(rho_i)
            if not np.allclose(tr, 1.0):
                rho_i = rho_i / tr
            rhos.append(rho_i)

    rho = kron_all(rhos)

    if pure:
        # Build global product ket and return it as well
        psi = np.array([1.0 + 0.0j])
        for v in kets:
            psi = np.kron(psi, v)
        # Ensure normalization (numerical guard)
        psi = psi / np.linalg.norm(psi)
        # Recompute rho from psi to avoid accumulation errors across kron
        rho = psi[:, None] @ psi.conj()[None, :]
        return rho, psi
    else:
        return rho, None



