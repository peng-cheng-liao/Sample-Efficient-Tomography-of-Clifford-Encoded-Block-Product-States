from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union
import numpy as np
import qutip as qt

# =============================================================================
# Constants / Types
# =============================================================================

Axis = Literal["X", "Y", "Z"]
PauliSpec = Union[str, Iterable[Tuple[int, Axis]]]
Ranking = List[Tuple[float, str]]  # [(score, "IXYZ..."), ...] sorted high->low

PAULI_CHARS = "IXYZ"
AXIS_TO_K = {"X": 0, "Y": 1, "Z": 2}

# Single-qubit matrices (NumPy)
I2 = np.eye(2, dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)

# Single-qubit matrices (QuTiP)
_QT_PAULI_1Q = {"I": qt.qeye(2), "X": qt.sigmax(), "Y": qt.sigmay(), "Z": qt.sigmaz()}


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


# =============================================================================
# Bell sampling
# =============================================================================


def bell_sampling(n: int, M: int, rho: qt.Qobj, seed: Optional[int] = None) -> np.ndarray:
    """
    Joint product-Bell measurement on (rho ⊗ rho) with correlations preserved.
    Output B has shape (n, M, 3) with columns [X,Y,Z] eigenvalues in {-1,+1}.

    NOTE: Exact enumeration over 4^n Bell outcomes (exponential).
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
        raise RuntimeError("Failed to permute rho⊗rho; check dims and QuTiP support.") from e

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


def bell_sampling_pure(n: int, M: int, psi: qt.Qobj, seed: Optional[int] = None) -> np.ndarray:
    """
    Joint product-Bell measurement on (|psi><psi| ⊗ |psi><psi|), assuming input is PURE.
    Returns B with shape (n, M, 3), columns [X, Y, Z] eigenvalues in {-1, +1}.

    Memory improvement vs bell_sampling():
      - avoids building rho (2^n x 2^n) and rho2 (4^n x 4^n),
      - works directly with the 2-copy ket (length 4^n) and a Bell-basis change.

    NOTE: Still exponential in n due to 4^n Bell outcomes (state length 4^n).
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if M <= 0:
        raise ValueError("M must be positive.")
    rng = np.random.default_rng(seed)

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

    # Sample Bell outcome index in [0, 4^n)
    sampled = rng.choice(probs.size, size=M, p=probs)

    # Outcome -> (XX, YY, ZZ) eigenvalues, order [X,Y,Z]
    eig_table = np.array(
        [
            [+1, -1, +1],  # Φ+
            [-1, +1, +1],  # Φ-
            [+1, +1, -1],  # Ψ+
            [-1, -1, -1],  # Ψ-
        ],
        dtype=np.int8,
    )

    pow4 = 4 ** np.arange(n, dtype=np.int64)
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


# =============================================================================
# Random generators (Paulis, product states)
# =============================================================================


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
        return_ket_if_pure: bool = True,
        epsilon: float = 0.05,
) -> qt.Qobj:
    """
    Generate a random n-qubit product state ρ = ⊗_i ρ_i.

    When pure=False:
      - Sample Bloch vectors uniformly from the 3D unit ball and reject unless
        epsilon < |r_x|,|r_y|,|r_z| < 1-epsilon.

    Returns QuTiP Qobj with dims:
      - if return_ket_if_pure: [[2]*n, [1]*n]
      - else: [[2]*n, [2]*n]
    """
    if n <= 0:
        raise ValueError("n must be > 0.")
    if not (0.0 <= purity <= 1.0):
        raise ValueError("purity must be in [0,1].")
    if not (0.0 <= epsilon < 0.5):
        raise ValueError("epsilon must satisfy 0 <= epsilon < 1/2.")
    if epsilon >= 1.0 / np.sqrt(3.0):
        raise ValueError("epsilon is too large: need epsilon < 1/sqrt(3) for feasibility.")

    rng = np.random.default_rng(seed)
    sx, sy, sz = qt.sigmax(), qt.sigmay(), qt.sigmaz()

    def random_qubit_ket() -> qt.Qobj:
        z = rng.normal(size=2) + 1j * rng.normal(size=2)
        z /= np.linalg.norm(z)
        return qt.Qobj(z.reshape((2, 1)), dims=[[2], [1]])

    def random_unit_vector_3() -> np.ndarray:
        v = rng.normal(size=3)
        v_norm = np.linalg.norm(v)
        return np.array([1.0, 0.0, 0.0]) if v_norm == 0 else (v / v_norm)

    def random_bloch_vector_uniform_ball() -> np.ndarray:
        # uniform-in-volume ball: r = U^(1/3), direction uniform on sphere
        direction = random_unit_vector_3()
        radius = float(rng.random()) ** (1.0 / 3.0)
        return radius * direction

    def accept_eps(r: np.ndarray) -> bool:
        ar = np.abs(r)
        return bool(np.all((ar > epsilon) & (ar < 1.0 - epsilon)) and (np.linalg.norm(r) <= 1.0 + 1e-12))

    def random_qubit_rho_mixed_uniform_ball() -> qt.Qobj:
        max_tries = 200_000
        for _ in range(max_tries):
            r = random_bloch_vector_uniform_ball()
            if accept_eps(r):
                rx, ry, rz = map(float, r)
                rho_i = (qt.qeye(2) + rx * sx + ry * sy + rz * sz) / 2.0
                rho_i.dims = [[2], [2]]
                return rho_i
        raise RuntimeError("Failed to sample a mixed qubit state; try reducing epsilon.")

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





def random_product_state_with_bloch_constraints(n, delta, seed=None, return_bloch_vectors=False):
    """
    Build an n-qubit product state (as a QuTiP density matrix) such that for each qubit i:

        - one Bloch-component magnitude is exactly (1 - delta),
        - one Bloch-component magnitude is exactly delta,
        - one Bloch-component magnitude is a random number < delta,

    and the assignment to x/y/z is randomly permuted for each qubit.

    The state is returned as an n-qubit density matrix:
        rho = rho_1 ⊗ rho_2 ⊗ ... ⊗ rho_n

    Parameters
    ----------
    n : int
        Number of qubits.
    delta : float
        Must satisfy 0 <= delta <= 1.
    seed : int or None
        Random seed for reproducibility.
    return_bloch_vectors : bool
        If True, also return the list of local Bloch vectors.

    Returns
    -------
    rho : qutip.Qobj
        n-qubit product density matrix.
    bloch_vectors : list[np.ndarray], optional
        Returned only if return_bloch_vectors=True.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")
    if not (0 <= delta <= 1):
        raise ValueError("delta must satisfy 0 <= delta <= 1.")

    rng = np.random.default_rng(seed)

    I = qt.qeye(2)
    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()

    single_qubit_states = []
    bloch_vectors = []

    # For fixed magnitudes (1-delta) and delta, the third component must satisfy
    # y^2 <= 1 - (1-delta)^2 - delta^2
    max_allowed_by_norm = np.sqrt(max(0.0, 1.0 - (1.0 - delta) ** 2 - delta ** 2))

    for _ in range(n):
        # Random magnitude for the third component, strictly less than delta
        # and small enough to keep ||r|| <= 1
        upper = min(delta, max_allowed_by_norm)
        rand_mag = rng.uniform(0.0, upper) if upper > 0 else 0.0

        # The three magnitudes to place on x, y, z in random order
        mags = np.array([1.0 - delta, delta, rand_mag], dtype=float)

        # Randomly permute which axis gets which magnitude
        perm = rng.permutation(3)

        # Random signs for each component
        signs = rng.choice([-1.0, 1.0], size=3)

        r = np.empty(3, dtype=float)
        r[perm] = signs * mags

        rx, ry, rz = r

        # rho_i = (I + r·sigma)/2
        rho_i = 0.5 * (I + rx * sx + ry * sy + rz * sz)

        single_qubit_states.append(rho_i)
        bloch_vectors.append(r)

    rho = qt.tensor(single_qubit_states)

    if return_bloch_vectors:
        return rho, bloch_vectors
    return rho

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


def rank_all_true_sP(rho: qt.Qobj, n: int, weight: Optional[int] = None, include_identity: bool = False) -> Ranking:
    """Exact s(P)=Tr(rho P)^2 ranking over all Paulis (or exact weight)."""
    rho = _as_density(rho, n)
    if weight is not None and not (0 <= weight <= n):
        raise ValueError("weight must satisfy 0 <= weight <= n.")

    results: Ranking = []

    def s_of(P: str) -> float:
        P_op = qt.tensor([_QT_PAULI_1Q[ch] for ch in P])
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
                    results.append((s_of("".join(P_list)), "".join(P_list)))

    results.sort(key=lambda x: x[0], reverse=True)
    return results


def rank_all_true_sP_v2(
        rho: qt.Qobj,
        n: int,
        weight: Optional[int] = None,
        include_identity: bool = False,
) -> "Ranking":
    """
    Exact ranking of s(P) = Tr(rho P)^2 over all n-qubit Paulis (or fixed weight).

    Accepts either:
      - pure state ket |psi> (Qobj.isket == True), or
      - mixed state density matrix rho (Qobj.isoper / isherm, shape 2^n x 2^n).

    If input is a ket, computes Tr(|psi><psi| P) as <psi|P|psi> without
    forming the density matrix explicitly.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if weight is not None and not (0 <= weight <= n):
        raise ValueError("weight must satisfy 0 <= weight <= n.")

    # Decide pure vs mixed path
    is_ket = isinstance(rho, qt.Qobj) and rho.isket

    if is_ket:
        psi = rho
        dim = psi.shape[0]
        if dim != 2 ** n or psi.shape[1] != 1:
            raise ValueError(f"Ket dimension mismatch: got {psi.shape}, expected ({2 ** n}, 1).")

        # normalize if needed (cheap)
        norm = (psi.dag() * psi)
        norm = float(np.real(norm))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Ket has non-finite or non-positive norm.")
        if abs(norm - 1.0) > 1e-10:
            psi = psi / np.sqrt(norm)

        def s_of(P: str) -> float:
            P_op = qt.tensor([_QT_PAULI_1Q[ch] for ch in P])
            # t = <psi|P|psi> (real for Hermitian P, but guard numerics)
            t = (psi.dag() * (P_op * psi))
            t = float(np.real(t))
            return t * t

    else:
        # mixed / operator path (keeps your existing validation)
        rho_dm = _as_density(rho, n)

        def s_of(P: str) -> float:
            P_op = qt.tensor([_QT_PAULI_1Q[ch] for ch in P])
            t = (P_op * rho_dm).tr()
            t = float(np.real(t))
            return t * t

    results: "Ranking" = []

    if weight is None:
        for tup in itertools.product(("I", "X", "Y", "Z"), repeat=n):
            P = "".join(tup)
            if not include_identity and all(ch == "I" for ch in P):
                continue
            results.append((s_of(P), P))
    else:
        if weight == 0:
            if include_identity:
                # s(I...I)=Tr(rho)^2 = 1 for normalized states
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


# =============================================================================
# Recover Q from ranking
# =============================================================================


def recover_Q_from_ranking(
        ranking: Ranking,
        n: Optional[int] = None,
        max_steps: Optional[int] = None,
) -> Dict[str, str]:
    """
    Recover {Q_i^x,Q_i^y,Q_i^z} following the described algorithm (phase ignored).

    Constraint enforced:
      - Any candidate P assigned to Q_i^α must commute with ALL already-assigned Q_k^β for k != i.
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
    span = GF2Basis()

    Qx: List[Optional[int]] = [None] * n
    Qy: List[Optional[int]] = [None] * n
    Qz: List[Optional[int]] = [None] * n

    def all_none(i: int) -> bool:
        return Qx[i] is None and Qy[i] is None and Qz[i] is None

    def all_assigned() -> bool:
        return all(Qx[i] is not None and Qy[i] is not None and Qz[i] is not None for i in range(n))

    def assigned_all_except(i_ex: int) -> List[int]:
        ops: List[int] = []
        for k in range(n):
            if k == i_ex:
                continue
            for v in (Qx[k], Qy[k], Qz[k]):
                if v is not None:
                    ops.append(v)
        return ops

    def commutes_with_all_except_site(P: int, i_ex: int) -> bool:
        return all(symp.commutes_int(P, v) for v in assigned_all_except(i_ex))

    j = 0
    steps = 0
    max_steps = max_steps if max_steps is not None else len(ranking) * (n + 2)

    while j < len(ranking) and not all_assigned() and steps < max_steps:
        steps += 1
        Pj = symp.to_int(ranking[j][1])

        # Skip if already in span
        if span.contains(Pj):
            j += 1
            continue

        i = 0
        while True:
            # Assign Q_i^x
            if all_none(i):
                if commutes_with_all_except_site(Pj, i):
                    Qx[i] = Pj
                    span.add(Pj)
                j += 1
                break

            # Assign Q_i^z (and Q_i^y)
            if Qx[i] is not None and Qz[i] is None:
                if symp.anticommutes_int(Pj, Qx[i]) and commutes_with_all_except_site(Pj, i):
                    Qz[i] = Pj
                    Qy[i] = Qx[i] ^ Qz[i]
                    span.add(Pj)
                    j += 1
                    break

            # If commutes with Q_i^x, move to next i
            if Qx[i] is not None and symp.commutes_int(Pj, Qx[i]):
                if i == n - 1:
                    j += 1
                    break
                i += 1
                continue

            # Otherwise advance j to avoid stalling
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


# =============================================================================
# Check recovered Q
# =============================================================================


def check_recovered_Q(Q: Dict[str, str], n: int, require_all_assigned: bool = True) -> Dict[str, object]:
    """
    Checks:
      (1) each Q_i^alpha is a valid Pauli string
      (2) within each i: x,y,z pairwise anticommute
      (3) across i!=j: all commute
    """
    required_keys = (
            [f"Q_{i}^x" for i in range(1, n + 1)]
            + [f"Q_{i}^y" for i in range(1, n + 1)]
            + [f"Q_{i}^z" for i in range(1, n + 1)]
    )

    missing_keys = [k for k in required_keys if k not in Q]
    if require_all_assigned and missing_keys:
        return {
            "ok": False,
            "missing_keys": missing_keys,
            "invalid_paulis": [],
            "anticommute_violations": [],
            "commute_violations": [],
        }

    def valid_pauli(s: str) -> bool:
        return isinstance(s, str) and len(s) == n and all(ch in PAULI_CHARS for ch in s)

    invalid_paulis = [(k, v) for k, v in Q.items() if not valid_pauli(v)]
    if invalid_paulis:
        return {
            "ok": False,
            "missing_keys": missing_keys,
            "invalid_paulis": invalid_paulis,
            "anticommute_violations": [],
            "commute_violations": [],
        }

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
            (not missing_keys or not require_all_assigned)
            and not invalid_paulis
            and not anticommute_violations
            and not commute_violations
    )
    return {
        "ok": ok,
        "missing_keys": missing_keys,
        "invalid_paulis": invalid_paulis,
        "anticommute_violations": anticommute_violations,
        "commute_violations": commute_violations,
    }


def check_recovered_Q_product_state(Q: Dict[str, str], n: int, require_all_assigned: bool = True) -> Dict[str, object]:
    """
    Specialized checker for product-state structure:

    (1) Every Q_i^alpha is single-site Pauli (exactly one of X/Y/Z, rest I).
    (2) For each i, the triple acts on the same site and equals {X_j,Y_j,Z_j} (up to alpha permutation).
    (3) Sites are a permutation (bijection i -> site j).
    """
    req = (
            [f"Q_{i}^x" for i in range(1, n + 1)]
            + [f"Q_{i}^y" for i in range(1, n + 1)]
            + [f"Q_{i}^z" for i in range(1, n + 1)]
    )
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

    def is_valid_pauli(s: str) -> bool:
        return isinstance(s, str) and len(s) == n and all(ch in PAULI_CHARS for ch in s)

    invalid_paulis: List[Tuple[str, str]] = [(k, v) for k, v in Q.items() if not is_valid_pauli(v)]
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

    def single_site_info(P: str) -> Tuple[bool, int, str]:
        nonI = [(idx, ch) for idx, ch in enumerate(P) if ch != "I"]
        if len(nonI) != 1:
            return False, -1, ""
        idx0, ch0 = nonI[0]
        if ch0 not in ("X", "Y", "Z"):
            return False, -1, ""
        return True, idx0 + 1, ch0  # 1-based index

    non_single_site: List[Tuple[str, str]] = []
    triple_site_mismatch: List[int] = []
    triple_not_xyz: List[Tuple[int, int, str]] = []
    site_map: Dict[int, int] = {}
    alpha_perm: Dict[int, Dict[str, str]] = {}

    for i in range(1, n + 1):
        Px = Q.get(f"Q_{i}^x")
        Py = Q.get(f"Q_{i}^y")
        Pz = Q.get(f"Q_{i}^z")

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
        if not (okx and oky and okz):
            continue

        if not (sx == sy == sz):
            triple_site_mismatch.append(i)
            continue

        letters = (lx, ly, lz)
        if set(letters) != {"X", "Y", "Z"}:
            triple_not_xyz.append((i, sx, "".join(letters)))
            continue

        site_map[i] = sx
        alpha_perm[i] = {"x": lx, "y": ly, "z": lz}

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


# =============================================================================
# Symplectic matrix I/O (from recovered Paulis)
# =============================================================================


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


def symplectic_matrix_from_recover_output(Qhat: Dict[str, str]) -> np.ndarray:
    """
    Build 2n x 2n binary symplectic matrix S with columns:
      [Q_1^x ... Q_n^x | Q_1^z ... Q_n^z]
    from recover_Q_from_ranking output dict.
    """
    if not isinstance(Qhat, dict):
        raise TypeError("Qhat must be a dict.")
    if not Qhat:
        raise ValueError("Empty dict.")

    any_pauli = next(iter(Qhat.values()))
    if not isinstance(any_pauli, str):
        raise ValueError("Dict values must be Pauli strings.")
    n = len(any_pauli)

    pat = re.compile(r"^Q_(\d+)\^([xyz])$")
    Qx: Dict[int, str] = {}
    Qz: Dict[int, str] = {}

    for k, P in Qhat.items():
        m = pat.match(k)
        if m is None:
            continue
        i = int(m.group(1))
        alpha = m.group(2)

        if not isinstance(P, str) or len(P) != n or (set(P) - set(PAULI_CHARS)):
            raise ValueError(f"Invalid Pauli for key '{k}': '{P}' (expected length {n} over IXYZ).")

        if alpha == "x":
            Qx[i] = P
        elif alpha == "z":
            Qz[i] = P

    if not Qx or not Qz:
        raise ValueError("Did not find any Q_i^x / Q_i^z keys.")

    n_from_keys = max(max(Qx.keys()), max(Qz.keys()))
    missing_x = [i for i in range(1, n_from_keys + 1) if i not in Qx]
    missing_z = [i for i in range(1, n_from_keys + 1) if i not in Qz]
    if missing_x or missing_z:
        raise ValueError(f"Missing assignments: x-missing={missing_x}, z-missing={missing_z}")
    if n_from_keys != n:
        raise ValueError(f"Inconsistent n: inferred from strings n={n}, but keys go up to {n_from_keys}.")

    S = np.zeros((2 * n, 2 * n), dtype=np.uint8)
    for i in range(1, n + 1):
        S[:, i - 1] = pauli_to_symplectic_col(Qx[i])
    for i in range(1, n + 1):
        S[:, n + (i - 1)] = pauli_to_symplectic_col(Qz[i])
    return S


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


def gates_to_unitary(gates: List[Tuple], n: int) -> np.ndarray:
    """Build the 2^n x 2^n unitary by multiplying gates in the given order."""
    Hm = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    Sm = np.array([[1, 0], [0, 1j]], dtype=complex)

    U = np.eye(2 ** n, dtype=complex)
    for g in gates:
        if g[0] == "H":
            U = _single_qubit_op(n, g[1], Hm) @ U
        elif g[0] == "S":
            U = _single_qubit_op(n, g[1], Sm) @ U
        elif g[0] == "CNOT":
            U = _two_qubit_cnot(n, g[1], g[2]) @ U
        else:
            raise ValueError(f"Unknown gate: {g}")
    return U


def synthesize_and_unitary(F: np.ndarray) -> Tuple[List[Tuple], np.ndarray]:
    """
    Given a symplectic tableau F (2n x 2n), return:
      - gates: synthesis sequence over {H, S, CNOT}
      - Udag : conjugate transpose of the synthesized unitary
    """
    n = F.shape[0] // 2
    gates = synthesize_clifford_from_tableau(F)
    U = gates_to_unitary(gates, n)
    return gates, np.conj(U.T)


def format_gates(gates: List[Tuple]) -> str:
    lines: List[str] = []
    for g in gates:
        if g[0] == "H":
            lines.append(f"H {g[1] + 1}")
        elif g[0] == "S":
            lines.append(f"S {g[1] + 1}")
        elif g[0] == "CNOT":
            lines.append(f"CNOT {g[1] + 1}->{g[2] + 1}")
    return "\n".join(lines)


# =============================================================================
# Clifford verification utilities (NumPy)
# =============================================================================


def mod2(a: np.ndarray) -> np.ndarray:
    return (a % 2).astype(int)


def _kron_all(mats: Iterable[np.ndarray]) -> np.ndarray:
    out = np.array([[1]], dtype=complex)
    for M in mats:
        out = np.kron(out, M)
    return out


def pauli_from_xz(x: np.ndarray, z: np.ndarray, *, qubit_order: str = "MSB") -> np.ndarray:
    """
    Build Hermitian Pauli σ(x,z) = i^{x·z} ⊗_j (X^{x_j} Z^{z_j}), with X then Z on each site.
    """
    x = np.asarray(x, dtype=int)
    z = np.asarray(z, dtype=int)
    if x.shape != z.shape:
        raise ValueError("x and z must have the same shape.")
    n = x.size

    idxs = range(n) if qubit_order == "MSB" else range(n - 1, -1, -1)
    mats: List[np.ndarray] = []
    for j in idxs:
        Mj = I2
        if x[j] == 1:
            Mj = X2 @ Mj
        if z[j] == 1:
            Mj = Mj @ Z2
        mats.append(Mj)

    phase = (1j) ** int(np.dot(x, z))
    return phase * _kron_all(mats)


def best_global_phase(A: np.ndarray, B: np.ndarray) -> complex:
    """
    Return alpha minimizing ||A - alpha B||_F under Hilbert-Schmidt inner product.
    alpha = tr(B^† A)/d, where d = 2^n.
    """
    d = A.shape[0]
    alpha = np.trace(B.conj().T @ A) / d
    if np.abs(alpha) > 0:
        alpha /= np.abs(alpha)
    return alpha


def is_unitary(U: np.ndarray, tol: float = 1e-10) -> bool:
    d = U.shape[0]
    return np.allclose(U.conj().T @ U, np.eye(d), atol=tol) and np.allclose(U @ U.conj().T, np.eye(d), atol=tol)


def verify_clifford_action(
        F: np.ndarray,
        U: np.ndarray,
        *,
        side: str = "left",
        qubit_order: str = "MSB",
        check_generators: bool = True,
        random_trials: int = 0,
        check_all: bool = False,
        tol: float = 1e-8,
        verbose: bool = True,
) -> Dict[str, object]:
    """
    Verify that conjugation by U maps Hermitian Paulis according to F over GF(2).

    Conventions:
      - Pauli vectors a = [x; z] (x then z), each in {0,1}^n.
      - side='left'  uses a' = F @ a (column convention)
        side='right' uses a' = a @ F (row convention)
      - Equality checked up to a global phase (±1, ±i).
    """
    F = mod2(F)
    m = F.shape[0]
    if F.shape[0] != F.shape[1] or m % 2 != 0:
        raise ValueError("F must be 2n x 2n.")
    n = m // 2
    d = 2 ** n
    if U.shape != (d, d):
        raise ValueError("U must be 2^n x 2^n.")

    if verbose:
        if not is_unitary(U, tol=max(1e-10, tol / 10)):
            print("Warning: U is not perfectly unitary within tolerance.")
        if not is_symplectic(F.astype(np.uint8)):
            print("Warning: F is not symplectic over GF(2).")

    def apply_F(a: np.ndarray) -> np.ndarray:
        if side == "left":
            return mod2(F @ a)
        if side == "right":
            return mod2(a @ F)
        raise ValueError("side must be 'left' or 'right'")

    def U_conj(P: np.ndarray) -> np.ndarray:
        return U @ P @ U.conj().T

    tests: List[Tuple[np.ndarray, np.ndarray, str]] = []

    if check_generators:
        for i in range(n):
            x = np.zeros(n, dtype=int)
            x[i] = 1
            z = np.zeros(n, dtype=int)
            tests.append((x, z, f"X_{i}"))
        for i in range(n):
            x = np.zeros(n, dtype=int)
            z = np.zeros(n, dtype=int)
            z[i] = 1
            tests.append((x, z, f"Z_{i}"))

    if check_all:
        if n > 4:
            raise ValueError("check_all is only feasible for n<=4 (4^n Paulis).")
        for x in itertools.product([0, 1], repeat=n):
            for z in itertools.product([0, 1], repeat=n):
                tests.append((np.array(x, int), np.array(z, int), "ALL"))

    if random_trials > 0:
        rng = np.random.default_rng(1234)
        for _ in range(random_trials):
            x = rng.integers(0, 2, size=n, dtype=int)
            z = rng.integers(0, 2, size=n, dtype=int)
            tests.append((x, z, "RAND"))

    # Deduplicate while preserving order
    seen = set()
    uniq_tests: List[Tuple[np.ndarray, np.ndarray, str]] = []
    for x, z, tag in tests:
        key = (tuple(x.tolist()), tuple(z.tolist()))
        if key not in seen:
            seen.add(key)
            uniq_tests.append((x, z, tag))
    tests = uniq_tests

    checks: List[Tuple[str, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], complex, bool]] = []
    failures: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, complex]] = []

    for x, z, tag in tests:
        a = np.concatenate([x, z])
        a_prime = apply_F(a)
        x_p, z_p = a_prime[:n], a_prime[n:]

        P = pauli_from_xz(x, z, qubit_order=qubit_order)
        P_target = pauli_from_xz(x_p, z_p, qubit_order=qubit_order)
        P_conj = U_conj(P)

        alpha = best_global_phase(P_conj, P_target)
        ok = np.allclose(P_conj, alpha * P_target, atol=tol)

        if not ok:
            num = np.trace(P_target.conj().T @ P_conj) / (2 ** n)
            if np.abs(num) > 1e-12:
                alpha2 = num / np.abs(num)
                ok = np.allclose(P_conj, alpha2 * P_target, atol=tol)
                alpha = alpha2

        checks.append((tag, (x.copy(), z.copy()), (x_p.copy(), z_p.copy()), alpha, ok))
        if not ok:
            failures.append((x.copy(), z.copy(), x_p.copy(), z_p.copy(), alpha))

    success = len(failures) == 0
    if verbose:
        total = len(checks)
        passed = sum(int(ok) for *_, ok in checks)
        print(
            f"Verification summary: {passed}/{total} cases passed "
            f"(n={n}, side='{side}', qubit_order='{qubit_order}', tol={tol})."
        )
        if failures:
            fx, fz, fxp, fzp, alph = failures[0]
            print("Example failure (up to global phase):")
            print(f"  Input (x,z): {fx}, {fz}")
            print(f"  Expected (x',z'): {fxp}, {fzp}")
            print(f"  Best phase α ≈ {alph}")

    return {
        "success": success,
        "n": n,
        "side": side,
        "qubit_order": qubit_order,
        "tol": tol,
        "num_checks": len(checks),
        "num_failures": len(failures),
        "failures": failures,
        "details": checks,
    }


# =============================================================================
# Random symplectic / permutations
# =============================================================================


def random_symplectic(n: int, steps: int | None = None, seed: int | None = None) -> np.ndarray:
    """Return random symplectic F ∈ GF(2)^{2n×2n} by composing random Clifford generators."""
    if steps is None:
        steps = 5 * n

    rng = np.random.default_rng(seed)
    F = np.eye(2 * n, dtype=np.uint8)

    for _ in range(max(1, steps)):
        g = int(rng.integers(0, 3))
        if g == 0:
            apply_h_right(F, int(rng.integers(0, n)))
        elif g == 1:
            apply_s_right(F, int(rng.integers(0, n)))
        else:
            c = int(rng.integers(0, n))
            t = int(rng.integers(0, n - 1))
            if t >= c:
                t += 1
            apply_cnot_right(F, c, t)

    return F


def random_clifford_gate(
        n: int,
        steps: int | None = None,
        seed: int | None = None,
) -> tuple[list[tuple], np.ndarray, np.ndarray]:
    """
    Generate a random Clifford (restricted to {H, S, CNOT}) using the same logic as random_symplectic.

    Returns:
      - gates: list of gate tuples [('H', j), ('S', j), ('CNOT', c, t), ...]
      - F:     resulting symplectic matrix (2n x 2n) over GF(2), column tableau convention
      - Udag:  conjugate-transpose of the unitary (2^n x 2^n) built from `gates`

    Notes:
      - This uses the SAME update rules as apply_h_right/apply_s_right/apply_cnot_right.
      - If you prefer U (not Udag), replace the last line with `U = gates_to_unitary(...); return gates, F, U`.
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
        g = int(rng.integers(0, 3))
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

    # Optional sanity check
    # if not is_symplectic(F):
    #     raise RuntimeError("Generated tableau is not symplectic; check update rules/conventions.")

    U = gates_to_unitary(gates, n)
    return gates, F, U


def apply_qubit_permutation(F: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Relabel qubits by permutation matrix M (n×n). Returns F' = F @ B over GF(2) with B=diag(M,M)."""
    F = (F % 2).astype(np.uint8)
    M = (M % 2).astype(np.uint8)
    n2 = F.shape[0]
    n = n2 // 2
    if F.shape != (2 * n, 2 * n) or M.shape != (n, n):
        raise ValueError("Bad shapes: F must be 2n×2n and M must be n×n.")
    B = np.block([[M, np.zeros((n, n), dtype=np.uint8)], [np.zeros((n, n), dtype=np.uint8), M]])
    return (F @ B) % 2


def random_permutation_matrix(n: int, *, seed: int | None = None, dtype=np.uint8) -> Tuple[np.ndarray, np.ndarray]:
    """Return (M, p): permutation matrix M and permutation vector p (0..n-1)."""
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    M = np.zeros((n, n), dtype=dtype)
    M[p, np.arange(n)] = 1
    return M, p


# =============================================================================
# Pure/mixed product state builder (NumPy version)
# =============================================================================


def _haar_pure_qubit(rng: np.random.Generator) -> np.ndarray:
    """Haar-random pure |psi> on C^2 as a normalized ket (2,)."""
    u = rng.uniform(-1.0, 1.0)  # cos(theta) uniform
    phi = rng.uniform(0.0, 2.0 * np.pi)
    theta = np.arccos(u)
    return np.array([np.cos(theta / 2.0), np.exp(1j * phi) * np.sin(theta / 2.0)], dtype=complex)


def _rho_from_ket(psi: np.ndarray) -> np.ndarray:
    psi = psi.reshape(2, 1)
    return psi @ psi.conj().T


def _rho_from_bloch_vector(r: np.ndarray) -> np.ndarray:
    """ρ = 1/2 (I + r·σ), with ||r|| <= 1."""
    return 0.5 * (I2 + r[0] * X2 + r[1] * Y2 + r[2] * Z2)


def _random_bloch_vector(rng: np.random.Generator, radius: Optional[float]) -> np.ndarray:
    """Random vector in the Bloch ball. If radius is None, sample uniformly in volume."""
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    if radius is None:
        r = rng.uniform() ** (1.0 / 3.0)
    else:
        r = float(np.clip(radius, 0.0, 1.0))
    return r * v


def product_state(
        n: int,
        *,
        pure: bool = True,
        qubits: Optional[List[np.ndarray]] = None,
        mixed_radius: Optional[float] = None,
        seed: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Build an n-qubit product state ρ = ⊗_i ρ_i (NumPy).

    Args:
      pure: if True, each qubit is |ψ_i><ψ_i| (Haar random unless provided)
      qubits:
        - if pure=True: list of kets (2,)
        - if pure=False: list of density matrices (2,2)
      mixed_radius: if pure=False and qubit not provided, set Bloch radius; None => uniform-in-volume.
    Returns:
      (rho, psi) where psi is only returned if pure=True.
    """
    rng = np.random.default_rng(seed)
    if qubits is not None and len(qubits) != n:
        raise ValueError("`qubits` must be None or a list of length n.")

    rhos: List[np.ndarray] = []
    kets: List[np.ndarray] = []

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
            rho_i = 0.5 * (rho_i + rho_i.conj().T)
            tr = np.trace(rho_i)
            if not np.allclose(tr, 1.0):
                rho_i = rho_i / tr
            rhos.append(rho_i)

    rho = _kron_all(rhos)

    if pure:
        psi = np.array([1.0 + 0.0j])
        for v in kets:
            psi = np.kron(psi, v)
        psi = psi / np.linalg.norm(psi)
        rho = psi[:, None] @ psi.conj()[None, :]
        return rho, psi

    return rho, None


def pauli_shot_counts_from_state(
        rho: np.ndarray,
        Nx: int,
        Ny: int,
        Nz: int,
        *,
        seed: int | None = None,
) -> dict[str, int]:
    """
    Simulate Pauli (X,Y,Z) projective measurement counts on a single-qubit state rho.

    Args:
        rho: 2x2 density matrix (complex). Should be Hermitian, PSD, trace 1 (approximately).
        Nx, Ny, Nz: number of shots along X, Y, Z axes.
        seed: RNG seed.

    Returns:
        dict with keys:
          n_xp, n_xm, n_yp, n_ym, n_zp, n_zm
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

    rng = np.random.default_rng(seed)

    n_xp = int(rng.binomial(Nx, px))
    n_xm = Nx - n_xp
    n_yp = int(rng.binomial(Ny, py))
    n_ym = Ny - n_yp
    n_zp = int(rng.binomial(Nz, pz))
    n_zm = Nz - n_zp

    counts = [n_xp, n_xm, n_yp, n_ym, n_zp, n_zm]
    return counts


def mle_qubit_from_pauli_counts(
        counts,
        *,
        max_iter: int = 5000,
        tol: float = 1e-12,
        step0: float = 0.25,
        seed: int | None = 0,
) -> np.ndarray:
    """
    Return 2x2 density matrix (complex) reconstructed by MLE.

    The optimization is done over the Bloch vector r=(rx,ry,rz),
    with constraint ||r|| <= 1 (physicality). We do projected
    gradient ascent with simple backtracking.

    Notes:
      - If counts are extreme (all + or all -), MLE may lie on
        the boundary (pure state).
      - Handles zero counts safely via epsilon clipping.
    """
    # Validate counts
    [n_xp, n_xm, n_yp, n_ym, n_zp, n_zm] = counts

    if any((not isinstance(c, (int, np.integer)) or c < 0) for c in counts):
        raise ValueError("All counts must be non-negative integers.")

    N_x, N_y, N_z = n_xp + n_xm, n_yp + n_ym, n_zp + n_zm
    if N_x == 0 or N_y == 0 or N_z == 0:
        raise ValueError("Each axis must have at least one shot (N_x,N_y,N_z > 0).")

    n_plus = np.array([n_xp, n_yp, n_zp], dtype=float)
    n_minus = np.array([n_xm, n_ym, n_zm], dtype=float)

    # Linear inversion starting point
    r = (n_plus - n_minus) / np.array([N_x, N_y, N_z], dtype=float)

    # Project to Bloch ball if needed
    nr = np.linalg.norm(r)
    if nr > 1.0:
        r = r / nr  # boundary
    # Add tiny jitter if exactly on boundary to avoid log singularities in early steps
    if np.linalg.norm(r) >= 1.0:
        r = 0.999999 * r

    rng = np.random.default_rng(seed)
    r = r + 1e-6 * rng.standard_normal(3)  # tiny perturbation
    # Ensure physical
    nr = np.linalg.norm(r)
    if nr > 0.999999:
        r = 0.999999 * r / nr

    eps = 1e-15  # for probability clipping

    def loglike(rr: np.ndarray) -> float:
        # p_{m,+}=(1+r_m)/2, p_{m,-}=(1-r_m)/2
        p_plus = 0.5 * (1.0 + rr)
        p_minus = 0.5 * (1.0 - rr)
        p_plus = np.clip(p_plus, eps, 1.0 - eps)
        p_minus = np.clip(p_minus, eps, 1.0 - eps)
        return float(np.sum(n_plus * np.log(p_plus) + n_minus * np.log(p_minus)))

    def grad_loglike(rr: np.ndarray) -> np.ndarray:
        # d/d r_m: n_{m,+}/(1+r_m) - n_{m,-}/(1-r_m)
        denom_plus = np.clip(1.0 + rr, eps, None)
        denom_minus = np.clip(1.0 - rr, eps, None)
        return n_plus / denom_plus - n_minus / denom_minus

    ll = loglike(r)
    step = step0

    for _ in range(max_iter):
        g = grad_loglike(r)

        # Proposed step
        r_new = r + step * g

        # Project to Bloch ball (allow boundary)
        norm_new = np.linalg.norm(r_new)
        if norm_new > 1.0:
            r_new = r_new / norm_new  # project to boundary
            # pull slightly inside to avoid log(0)
            r_new = 0.999999 * r_new

        ll_new = loglike(r_new)

        # Backtracking if likelihood decreases
        if ll_new < ll:
            step *= 0.5
            if step < 1e-16:
                break
            continue

        # Accept
        if abs(ll_new - ll) < tol:
            r = r_new
            ll = ll_new
            break

        r = r_new
        ll = ll_new

        # Mild step growth if doing well
        step *= 1.02

    # Build density matrix rho = 1/2 (I + rx X + ry Y + rz Z)
    rx, ry, rz = r
    rho = 0.5 * np.array(
        [[1.0 + rz, rx - 1j * ry],
         [rx + 1j * ry, 1.0 - rz]],
        dtype=complex
    )

    # Numerical cleanup: enforce Hermiticity and trace 1
    rho = 0.5 * (rho + rho.conj().T)
    rho = rho / np.trace(rho)

    # Optional: clip tiny negative eigenvalues from numerical noise
    evals, evecs = np.linalg.eigh(rho)
    evals = np.clip(evals, 0.0, None)
    rho = (evecs * evals) @ evecs.conj().T
    rho = rho / np.trace(rho)

    return rho


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


def single_qubit_tomography(rho, Nx, Ny, Nz):
    rho = rho.full()
    counts = pauli_shot_counts_from_state(rho, Nx, Ny, Nz)
    rho_prime = linear_inversion_qubit_from_pauli_counts(counts)

    rho_prime = qt.Qobj(rho_prime)
    return rho_prime


# =============================================================================
# State helpers / high-level tomography pipeline additions
# =============================================================================


def qutip_pauli_op(n: int, P: PauliSpec) -> qt.Qobj:
    """Public alias matching later helper usage."""
    return _qutip_pauli_op(n, P)


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


def _density_to_qobj(rho: Union[np.ndarray, qt.Qobj], n: int) -> qt.Qobj:
    """Compatibility helper: always return a density operator Qobj."""
    return _as_density(_state_to_qobj(rho, n), n)


def _bell_sampling_state(n: int, M: int, state: Union[np.ndarray, qt.Qobj], seed: Optional[int] = None) -> Tuple[np.ndarray, bool, qt.Qobj]:
    """
    Dispatch Bell sampling based on whether the input state is pure (ket) or mixed.
    Returns (B, is_pure, state_qobj).
    """
    state_q = _state_to_qobj(state, n)
    if state_q.isket:
        return bell_sampling_pure(n=n, M=M, psi=state_q, seed=seed), True, state_q
    return bell_sampling(n=n, M=M, rho=state_q, seed=seed), False, state_q


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
    Build the 2^n x 2^n unitary matching the tableau-synthesis convention.
    The gate list returned by synthesize_clifford_from_tableau is a
    right-composition list for the tableau, so we accumulate on the right.
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


def format_gates(gates: List[Tuple]) -> str:
    lines: List[str] = []
    for g in gates:
        if g[0] == "H":
            lines.append(f"H {g[1] + 1}")
        elif g[0] == "S":
            lines.append(f"S {g[1] + 1}")
        elif g[0] == "X":
            lines.append(f"X {g[1] + 1}")
        elif g[0] == "CNOT":
            lines.append(f"CNOT {g[1] + 1}->{g[2] + 1}")
    return "\n".join(lines)


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
        g_op = qutip_pauli_op(n, g)
        z_string = ["I"] * n
        z_string[j] = "Z"
        Zj = qutip_pauli_op(n, "".join(z_string))

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

    B, is_pure, state_q = _bell_sampling_state(n=n, M=M1, state=rho, seed=seed)
    ranking = rank_all_sP_from_B(B, include_identity=False)
    candidates: Ranking = [(score, P) for score, P in ranking if score >= 1.0 - lam]
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
        Rx = qutip_pauli_op(m, axes["x"])
        Rz = qutip_pauli_op(m, axes["z"])

        x_string = ["I"] * m
        x_string[j] = "X"
        Xj = qutip_pauli_op(m, "".join(x_string))

        z_string = ["I"] * m
        z_string[j] = "Z"
        Zj = qutip_pauli_op(m, "".join(z_string))

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
        Rx = qutip_pauli_op(m, axes["x"])
        x_string = ["I"] * m
        x_string[j] = "X"
        Xj = qutip_pauli_op(m, "".join(x_string))

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
    if M2 <= 0:
        raise ValueError("M2 must be positive.")
    if not (0.0 < kappa < 0.5):
        raise ValueError("kappa must lie in (0, 1/2).")

    m = n - t
    if m == 0:
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
                "is_pure_input": False,
            }
        return recovered_axes, U_rec

    B_full, is_pure, state_q = _bell_sampling_state(n=n, M=M2, state=tilde_rho, seed=seed)
    B_res = B_full[t:, :, :]
    ranking_res = rank_all_sP_from_B(B_res, include_identity=False)
    candidates_res: Ranking = [(score, P) for score, P in ranking_res if score >= kappa]

    symp = Symplectic(m)
    span = GF2Basis()
    A1_axes_res: List[Dict[str, str]] = []
    A2_axes_res: List[Dict[str, str]] = []
    assigned_vecs: List[int] = []

    def commutes_with_completed_qubits(v: int) -> bool:
        return all(symp.commutes_int(v, w) for w in assigned_vecs)

    for score_x, Px in candidates_res:
        vx = symp.to_int(Px)
        if span.contains(vx):
            continue
        if not commutes_with_completed_qubits(vx):
            continue

        partner = None
        for score_z, Pz in candidates_res:
            if score_z < kappa:
                continue
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
            for v in (vx, vz, vx ^ vz):
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
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if M1 <= 0 or M2 <= 0 or M3 <= 0:
        raise ValueError("M1, M2, M3 must be positive.")
    if not (0.0 < lam < 1.0):
        raise ValueError("lam must lie in (0,1).")
    if not (0.0 < kappa < 0.5):
        raise ValueError("kappa must lie in (0, 1/2).")

    state_q = _state_to_qobj(rho, n)
    is_pure_input = state_q.isket

    # Step 1: empirical stabilizer peeling
    peel_out = empirical_stabilizer_peeling(
        rho=state_q,
        n=n,
        M1=M1,
        lam=lam,
        seed=seed1,
        return_details=True,
    )
    U_stab_np = peel_out["U_stab"]
    g_list = peel_out["g_list"]
    t = peel_out["t"]

    U_stab = qt.Qobj(U_stab_np, dims=[[2] * n, [2] * n])

    if is_pure_input:
        psi_q = state_q.unit()
        tilde_state = U_stab.dag() * psi_q
    else:
        rho_q = _as_density(state_q, n)
        tilde_state = U_stab.dag() * rho_q * U_stab

    # Step 2: rank-guided symplectic recovery
    rec_out = rank_guided_symplectic_recovery(
        tilde_rho=tilde_state,
        n=n,
        t=t,
        M2=M2,
        kappa=kappa,
        seed=seed2,
        return_details=True,
    )
    U_rec_np = rec_out["U_rec"]
    U_rec = qt.Qobj(U_rec_np, dims=[[2] * n, [2] * n])

    # Step 3: tomography in the doubly rotated frame
    if is_pure_input:
        sigma_state = U_rec.dag() * tilde_state
        local_marginals = [qt.ptrace(sigma_state, [j]) for j in range(n)]
    else:
        sigma_state = U_rec.dag() * tilde_state * U_rec
        local_marginals = [qt.ptrace(sigma_state, [j]) for j in range(n)]

    rho_prod_prime = [single_qubit_tomography(marg, M3, M3, M3) for marg in local_marginals]
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
    """
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

    rng = np.random.default_rng(seed)

    def random_sign() -> float:
        return float(rng.choice([-1.0, 1.0]))

    def shrink_until_feasible(vals_sq: np.ndarray) -> np.ndarray:
        vals_sq = vals_sq.astype(float).copy()
        while float(vals_sq.sum()) > 1.0 + 1e-12:
            idx = int(np.argmax(vals_sq))
            vals_sq[idx] *= 0.5
        return vals_sq

    def build_local_from_sq(vals_sq: np.ndarray) -> Tuple[qt.Qobj, np.ndarray]:
        vals_sq = shrink_until_feasible(vals_sq)
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
        vals_sq = rng.uniform(float(delta), float(max(delta, 1.0 - delta)), size=3).astype(float)
        return build_local_from_sq(vals_sq)

    def sample_A2() -> Tuple[qt.Qobj, np.ndarray]:
        vals_sq = np.zeros(3, dtype=float)
        big_axis = int(rng.integers(0, 3))
        other_axes = [a for a in range(3) if a != big_axis]
        vals_sq[big_axis] = float(rng.uniform(delta, max(delta, 1.0 - delta)))
        vals_sq[other_axes[0]] = float(rng.uniform(0.0, delta))
        vals_sq[other_axes[1]] = float(rng.uniform(0.0, delta))
        return build_local_from_sq(vals_sq)

    def sample_B() -> Tuple[qt.Qobj, np.ndarray]:
        vals_sq = rng.uniform(0.0, delta, size=3).astype(float)
        return build_local_from_sq(vals_sq)

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

    Uc = random_clifford_gate(n, steps, seed=seed)[2]
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

    rng = np.random.default_rng(seed)

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

    Uc = random_clifford_gate(n, steps, seed=seed)[2]
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
