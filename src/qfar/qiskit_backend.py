"""Qiskit circuit compilation, ASCII/OpenQASM2 export, depolarising + readout noise,
and Zero-Noise Extrapolation (ZNE) for Q-FAR QAOA circuits.

Qiskit is kept an optional runtime dependency so core ``.[dev]`` CI tests without
Qiskit continue to pass; functions raise a clear ImportError only when invoked
without Qiskit installed.
"""
from __future__ import annotations

import numpy as np

from .qubo import AllocationQUBO
from .solvers import finalize


def _require_qiskit():
    try:
        import qiskit
        import qiskit.qasm2 as qasm2
        import qiskit.quantum_info as qi
        from qiskit import QuantumCircuit, transpile
    except ImportError as exc:
        raise ImportError(
            "Qiskit >= 1.0 is required for qfar.qiskit_backend. "
            "Install with: pip install -e '.[full]'"
        ) from exc
    return qiskit, qasm2, qi, QuantumCircuit, transpile


def build_qaoa_circuit(
    q: AllocationQUBO,
    params: np.ndarray,
    p: int = 1,
    measure: bool = False,
    warm_start_thetas: np.ndarray | None = None,
):
    """Build an exact Qiskit QuantumCircuit matching ``StateVec`` QAOA / WS-QAOA.

    Sign & qubit ordering conventions:
    - Qubit ``i`` corresponds to bit ``i`` (LSB) of computational basis index ``k``.
    - Ising Hamiltonian ``H(z) = C + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j``
      normalised by ``||H||_1 = sum|h_i| + sum|J_ij|`` matches ``StateVec.Etil``.
    - ``RZ(2 * gamma * h_i / norm)`` and ``RZZ(2 * gamma * J_ij / norm)`` implement
      ``exp(-i * gamma * H_tilde)`` up to a global phase.
    - When ``warm_start_thetas`` is None: prepares ``|+>^n`` and mixes with ``RX(2 * beta)``.
    - When ``warm_start_thetas`` is given: prepares ``prod_i RY(theta_i)|0>`` and applies
      the warm-start mixer ``RY(theta_i) RZ(-2 * beta) RY(-theta_i) = exp(+i*beta*(cos*Z + sin*X))``.
    """
    _, _, _, QuantumCircuit, _ = _require_qiskit()
    params = np.asarray(params, float)
    if len(params) != 2 * p:
        raise ValueError(f"Expected 2*p={2 * p} parameters, got {len(params)}")
    if warm_start_thetas is not None and len(warm_start_thetas) != q.n:
        raise ValueError(f"Expected {q.n} warm-start angles, got {len(warm_start_thetas)}")

    h, J, _ = q.ising()
    norm = float(np.abs(h).sum() + np.abs(J).sum())
    n = q.n

    qc = QuantumCircuit(n, n if measure else 0, name=f"qfar_qaoa_p{p}")
    if warm_start_thetas is None:
        qc.h(range(n))
    else:
        for i, th in enumerate(warm_start_thetas):
            qc.ry(float(th), i)

    for layer in range(p):
        gamma = float(params[layer])
        beta = float(params[p + layer])
        for i in range(n):
            if abs(h[i]) > 1e-12:
                qc.rz(2.0 * gamma * float(h[i]) / norm, i)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-12:
                    qc.rzz(2.0 * gamma * float(J[i, j]) / norm, i, j)
        qc.barrier()
        if warm_start_thetas is None:
            for i in range(n):
                qc.rx(2.0 * beta, i)
        else:
            for i, th in enumerate(warm_start_thetas):
                qc.ry(-float(th), i)
                qc.rz(-2.0 * beta, i)
                qc.ry(float(th), i)

    if measure:
        qc.measure(range(n), range(n))
    return qc


def export_circuit_artifacts(
    q: AllocationQUBO,
    params: np.ndarray,
    p: int = 1,
    warm_start_thetas: np.ndarray | None = None,
) -> dict:
    """Return high-level circuit text, OpenQASM 2.0 string, and transpiled gate statistics."""
    _, qasm2, _, _, transpile = _require_qiskit()
    qc = build_qaoa_circuit(q, params, p=p, measure=False, warm_start_thetas=warm_start_thetas)
    qc_t = transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=0)
    ops = dict(qc_t.count_ops())
    return {
        "qubits": q.n,
        "logical_depth": int(qc.depth()),
        "logical_ops": {str(k): int(v) for k, v in qc.count_ops().items()},
        "transpiled_depth": int(qc_t.depth()),
        "transpiled_cx": int(ops.get("cx", 0)),
        "transpiled_single_qubit": int(ops.get("rz", 0) + ops.get("sx", 0) + ops.get("x", 0)),
        "circuit_text": str(qc.draw(output="text", fold=120)),
        "qasm2": str(qasm2.dumps(qc)),
    }


def fold_qaoa_circuit(qc, scale_factor: int = 1):
    """Local unitary gate folding ``G -> G (G^dagger G)^k`` for odd integer ``scale_factor = 2k + 1``.

    Barriers and measurements are preserved in position (unitary identity is exact
    in the noise-free limit, while amplifying physical gate counts by ``scale_factor``).
    """
    _, _, _, QuantumCircuit, _ = _require_qiskit()
    if scale_factor < 1 or scale_factor % 2 == 0:
        raise ValueError(f"scale_factor must be a positive odd integer, got {scale_factor}")
    if scale_factor == 1:
        return qc.copy()

    k = (scale_factor - 1) // 2
    folded = QuantumCircuit(*qc.qregs, *qc.cregs, name=f"{qc.name}_fold{scale_factor}")
    for inst in qc.data:
        op, qargs, cargs = inst.operation, inst.qubits, inst.clbits
        folded.append(op, qargs, cargs)
        if op.name not in ("barrier", "measure", "reset", "delay"):
            inv = op.inverse()
            for _ in range(k):
                folded.append(inv, qargs, cargs)
                folded.append(op, qargs, cargs)
    return folded


def apply_readout_noise(probs: np.ndarray, n: int, p_readout: float) -> np.ndarray:
    """Apply symmetric independent single-qubit bit-flip readout noise to ``probs``."""
    probs = np.asarray(probs, float)
    if p_readout <= 0.0:
        return probs / probs.sum()
    p_ro = float(p_readout)
    M = np.array([[1.0 - p_ro, p_ro], [p_ro, 1.0 - p_ro]], dtype=float)
    t = probs.reshape([2] * n)
    for ax in range(n):
        t = np.moveaxis(np.tensordot(M, t, axes=(1, ax)), 0, ax)
    out = np.clip(t.reshape(-1), 0.0, None)
    return out / out.sum()


def _depolarizing_kraus_1q(qi, p: float):
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    w0 = np.sqrt(1.0 - p)
    wp = np.sqrt(p / 3.0)
    return qi.Kraus([w0 * I, wp * X, wp * Y, wp * Z])


def _depolarizing_kraus_2q(qi, p: float):
    paulis = [
        np.eye(2, dtype=complex),
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    ]
    w0 = np.sqrt(1.0 - p)
    wp = np.sqrt(p / 15.0)
    ops = []
    for a in range(4):
        for b in range(4):
            w = w0 if (a == 0 and b == 0) else wp
            ops.append(w * np.kron(paulis[a], paulis[b]))
    return qi.Kraus(ops)


def _simulate_density_matrix(qc, p1_depol: float, p2_depol: float) -> np.ndarray:
    _, _, qi, _, _ = _require_qiskit()
    n = qc.num_qubits
    rho = qi.DensityMatrix.from_label("0" * n)
    ch1 = _depolarizing_kraus_1q(qi, p1_depol) if p1_depol > 0 else None
    ch2 = _depolarizing_kraus_2q(qi, p2_depol) if p2_depol > 0 else None

    for inst in qc.data:
        op = inst.operation
        if op.name in ("barrier", "measure"):
            continue
        qidx = [qc.find_bit(qb).index for qb in inst.qubits]
        rho = rho.evolve(qi.Operator(op), qargs=qidx)
        if len(qidx) == 1 and ch1 is not None:
            rho = rho.evolve(ch1, qargs=qidx)
        elif len(qidx) == 2 and ch2 is not None:
            rho = rho.evolve(ch2, qargs=qidx)
    return np.clip(np.real(rho.probabilities()), 0.0, None)


def _simulate_trajectories(
    qc, p1_depol: float, p2_depol: float, n_traj: int, rng: np.random.Generator
) -> np.ndarray:
    """Exact unbiased Kraus Monte-Carlo trajectory simulation for depolarising noise."""
    _, _, qi, _, _ = _require_qiskit()
    n = qc.num_qubits
    dim = 1 << n
    probs_1q = np.array([1.0 - p1_depol, p1_depol / 3.0, p1_depol / 3.0, p1_depol / 3.0])
    probs_2q = np.array([1.0 - p2_depol] + [p2_depol / 15.0] * 15)

    qmap = {qb: idx for idx, qb in enumerate(qc.qubits)}
    inv_s2 = 1.0 / np.sqrt(2.0)
    h_mat = np.array([[inv_s2, inv_s2], [inv_s2, -inv_s2]], dtype=complex)
    idx_arr = np.arange(dim)
    bit_masks = [((idx_arr >> q) & 1).astype(bool) for q in range(n)]

    # Pre-compile circuit ops once outside the L3-cache-sized trajectory chunk loop
    ops: list[tuple] = []
    zz_cache: dict[tuple[int, int], np.ndarray] = {}
    for inst in qc.data:
        op = inst.operation
        name = op.name
        if name in ("barrier", "measure"):
            continue
        qidx = [qmap[qb] for qb in inst.qubits]
        if len(qidx) == 1:
            q0 = qidx[0]
            if name == "rz":
                th = float(op.params[0])
                ph = np.where(bit_masks[q0], np.exp(0.5j * th), np.exp(-0.5j * th))
                ops.append(("diag1", q0, ph))
            elif name == "rx":
                th = float(op.params[0]) / 2.0
                c, s = np.cos(th), -1j * np.sin(th)
                ops.append(("1q", q0, np.array([[c, s], [s, c]], dtype=complex)))
            elif name == "ry":
                th = float(op.params[0]) / 2.0
                c, s = np.cos(th), np.sin(th)
                ops.append(("1q", q0, np.array([[c, -s], [s, c]], dtype=complex)))
            elif name == "h":
                ops.append(("1q", q0, h_mat))
            else:
                ops.append(("1q", q0, qi.Operator(op).data))
        elif len(qidx) == 2:
            q0, q1 = qidx[0], qidx[1]
            if name == "rzz":
                key = (min(q0, q1), max(q0, q1))
                if key not in zz_cache:
                    zz_cache[key] = bit_masks[q0] ^ bit_masks[q1]
                phi = float(op.params[0])
                ph = np.where(zz_cache[key], np.exp(0.5j * phi), np.exp(-0.5j * phi))
                ops.append(("diag2", q0, q1, ph))
            else:
                ops.append(("2q", q0, q1, qi.Operator(op).data.reshape(2, 2, 2, 2)))

    chunk_size = 16
    probs_acc = np.zeros(dim, dtype=float)

    for start in range(0, n_traj, chunk_size):
        bsz = min(chunk_size, n_traj - start)
        st = np.zeros((bsz, dim), dtype=complex)
        st[:, 0] = 1.0

        def apply_pauli_rows(rows: np.ndarray, code: int, q: int) -> None:
            v = st.reshape(bsz, 1 << (n - 1 - q), 2, 1 << q)
            if code == 3:
                v[rows, :, 1, :] *= -1.0
            elif code == 1:
                s0 = v[rows, :, 0, :].copy()
                v[rows, :, 0, :] = v[rows, :, 1, :]
                v[rows, :, 1, :] = s0
            elif code == 2:
                s0 = v[rows, :, 0, :].copy()
                v[rows, :, 0, :] = -1j * v[rows, :, 1, :]
                v[rows, :, 1, :] = 1j * s0

        for item in ops:
            kind = item[0]
            if kind in ("diag1", "1q"):
                q0 = item[1]
                if kind == "diag1":
                    np.multiply(st, item[2], out=st)
                else:
                    mat = item[2]
                    v = st.reshape(bsz, 1 << (n - 1 - q0), 2, 1 << q0)
                    s0 = v[:, :, 0, :].copy()
                    s1 = v[:, :, 1, :].copy()
                    v[:, :, 0, :] = mat[0, 0] * s0 + mat[0, 1] * s1
                    v[:, :, 1, :] = mat[1, 0] * s0 + mat[1, 1] * s1
                if p1_depol > 0:
                    choices = rng.choice(4, size=bsz, p=probs_1q)
                    if np.any(choices):
                        for code in (1, 2, 3):
                            rows = np.flatnonzero(choices == code)
                            if len(rows):
                                apply_pauli_rows(rows, code, q0)
            else:
                q0, q1 = item[1], item[2]
                if kind == "diag2":
                    np.multiply(st, item[3], out=st)
                else:
                    ax0, ax1 = 1 + (n - 1 - q0), 1 + (n - 1 - q1)
                    t = np.moveaxis(st.reshape((bsz,) + (2,) * n), (ax1, ax0), (1, 2))
                    st = np.moveaxis(np.einsum("abij,tij...->tab...", item[3], t), (1, 2), (ax1, ax0)).reshape(bsz, dim)
                if p2_depol > 0:
                    choices = rng.choice(16, size=bsz, p=probs_2q)
                    nz = np.flatnonzero(choices)
                    if len(nz):
                        c_nz = choices[nz]
                        a_nz, b_nz = c_nz // 4, c_nz % 4
                        for code in (1, 2, 3):
                            ra = nz[a_nz == code]
                            if len(ra):
                                apply_pauli_rows(ra, code, q1)
                            rb = nz[b_nz == code]
                            if len(rb):
                                apply_pauli_rows(rb, code, q0)

        probs_acc += np.sum(np.abs(st) ** 2, axis=0)
    return probs_acc / float(n_traj)


def simulate_noisy_qaoa(
    q: AllocationQUBO,
    params: np.ndarray,
    p: int = 1,
    p1_depol: float = 0.002,
    p2_depol: float = 0.01,
    p_readout: float = 0.02,
    shots: int = 1024,
    scale_factor: int = 1,
    seed: int = 0,
    warm_start_thetas: np.ndarray | None = None,
    method: str = "auto",
    n_trajectories: int = 256,
) -> dict:
    """Simulate noisy QAOA (depolarising 1q/2q + symmetric readout bit-flip) via Qiskit."""
    qc_base = build_qaoa_circuit(q, params, p=p, measure=False, warm_start_thetas=warm_start_thetas)
    qc = fold_qaoa_circuit(qc_base, scale_factor=scale_factor)

    use_method = method
    if use_method == "auto":
        use_method = "density_matrix" if q.n <= 8 else "trajectory"

    rng = np.random.default_rng(seed)
    if use_method == "density_matrix":
        probs_pre = _simulate_density_matrix(qc, p1_depol=p1_depol, p2_depol=p2_depol)
    elif use_method == "trajectory":
        probs_pre = _simulate_trajectories(qc, p1_depol=p1_depol, p2_depol=p2_depol,
                                           n_traj=n_trajectories, rng=rng)
    else:
        raise ValueError(f"Unknown noise simulation method: {use_method}")

    probs = apply_readout_noise(probs_pre, q.n, p_readout=p_readout)
    E = q.all_energies()
    bits = q.all_bits.astype(int)
    feasible_mask = (bits[:, : q.N] @ q.w) <= q.B

    samp = rng.choice(len(probs), size=shots, p=probs)
    best_k = samp[np.argmin(E[samp])]
    name = f"{'ws_' if warm_start_thetas is not None else ''}qaoa_p{p}_noisy_c{scale_factor}"
    return finalize(
        name, "P1-QUBO", q, bits[best_k][: q.N],
        selection_rule="lowest-energy noisy sample",
        probs=probs,
        expectation=float(probs @ E),
        expectation_samples=float(E[samp].mean()),
        p_feasible_exact=float(probs[feasible_mask].sum()),
        p_feasible_shots=float(np.mean((bits[samp][:, : q.N] @ q.w) <= q.B)),
        scale_factor=int(scale_factor),
        simulation_method=use_method,
        depth_folded=int(qc.depth()),
    )


def zero_noise_extrapolation(
    scales: list[int] | np.ndarray,
    energies: list[float] | np.ndarray,
    degree: int | None = None,
) -> dict:
    """Richardson / polynomial Zero-Noise Extrapolation evaluated at scale ``c = 0``.

    For scales ``[1, 3, 5]`` and ``degree=None`` (2), reproduces the exact
    Richardson weights ``[1.875, -1.25, 0.375]``.
    """
    x = np.asarray(scales, dtype=float)
    y = np.asarray(energies, dtype=float)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("scales and energies must have the same length >= 2")
    deg = (len(x) - 1) if degree is None else int(degree)
    if deg >= len(x):
        deg = len(x) - 1
    coefs = np.polyfit(x, y, deg)
    e_zne = float(np.polyval(coefs, 0.0))
    return {
        "scales": [float(v) for v in x],
        "energies": [float(v) for v in y],
        "degree": deg,
        "zne_energy": e_zne,
        "coeffs": [float(v) for v in coefs],
    }


def run_noisy_and_zne(
    q: AllocationQUBO,
    params: np.ndarray,
    p: int = 1,
    scales: tuple[int, ...] = (1, 3, 5),
    p1_depol: float = 0.002,
    p2_depol: float = 0.01,
    p_readout: float = 0.02,
    shots: int = 1024,
    seed: int = 0,
    warm_start_thetas: np.ndarray | None = None,
    method: str = "auto",
    n_trajectories: int = 256,
) -> dict:
    """Evaluate noisy QAOA across noise scale factors and compute ZNE expectation."""
    per_scale = []
    for idx, c in enumerate(scales):
        res = simulate_noisy_qaoa(
            q, params, p=p, p1_depol=p1_depol, p2_depol=p2_depol,
            p_readout=p_readout, shots=shots, scale_factor=int(c),
            seed=seed + idx, warm_start_thetas=warm_start_thetas,
            method=method, n_trajectories=n_trajectories,
        )
        per_scale.append(res)
    zne = zero_noise_extrapolation(list(scales), [r["expectation"] for r in per_scale])
    return {
        "base": per_scale[0],
        "per_scale": per_scale,
        "zne": zne,
    }
