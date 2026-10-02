"""
Quantum Variational Engine for Fair Microfinance Allocation.
Implements:
1. QAOA (Quantum Approximate Optimization Algorithm) with parameter optimization
2. VQE (Variational Quantum Eigensolver) with hardware-efficient RealAmplitudes ansatz
3. Qiskit Circuit Exporter (creates authentic QuantumCircuit and exports ASCII diagram/QASM)
"""

from typing import Dict, Tuple, Any, List, Optional
import numpy as np
from scipy.optimize import minimize
from src.qubo_builder import QUBOMapper

# Optional Qiskit import for circuit generation
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class QuantumSimulator:
    """
    Statevector Quantum Simulator for Variational Combinatorial Optimization.
    Operates on full 2^N Hilbert space with analytical precision and zero simulator overhead.
    """

    def __init__(self, qubo_mapper: QUBOMapper):
        self.mapper = qubo_mapper
        self.N = qubo_mapper.N
        self.dim = 2 ** self.N
        # Precompute diagonal energies for all 2^N basis states
        self.diag_energies = self.mapper.get_diagonal_hamiltonian()

    def get_uniform_superposition(self) -> np.ndarray:
        r"""Initial state |+>^{\otimes N} = (1 / sqrt(2^N)) * [1, 1, ..., 1]^T."""
        return np.ones(self.dim, dtype=complex) / np.sqrt(self.dim)

    def apply_cost_unitary(self, state: np.ndarray, gamma: float) -> np.ndarray:
        """
        Applies problem unitary U_C(gamma) = exp(-i * gamma * H_C).
        Since H_C is diagonal, this is an elementwise phase rotation:
        state[k] -> state[k] * exp(-i * gamma * E_k)
        """
        phase_factors = np.exp(-1j * gamma * self.diag_energies)
        return state * phase_factors

    def apply_mixer_unitary(self, state: np.ndarray, beta: float) -> np.ndarray:
        """
        Applies transverse mixer unitary U_M(beta) = prod_{i=0}^{N-1} exp(-i * beta * X_i).
        Exp(-i * beta * X) = cos(beta) * I - i * sin(beta) * X.
        Applies single-qubit X rotations sequentially across all N qubits.
        """
        c = np.cos(beta)
        s = -1j * np.sin(beta)
        rx_gate = np.array([[c, s], [s, c]], dtype=complex)

        current_state = state.copy()
        # Reshape to tensor of shape (2, 2, ..., 2) for clean single-qubit tensor contractions
        shape = [2] * self.N
        current_state = current_state.reshape(shape)

        for qubit in range(self.N):
            # Contract along axis `qubit`
            current_state = np.tensordot(rx_gate, current_state, axes=(1, qubit))
            # Move the newly transformed axis back to position `qubit`
            current_state = np.moveaxis(current_state, 0, qubit)

        return current_state.reshape(self.dim)

    def apply_ry_rotations(self, state: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """Applies single-qubit Ry(theta_i) rotations across all N qubits."""
        current_state = state.copy().reshape([2] * self.N)

        for qubit in range(self.N):
            theta = angles[qubit]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            ry_gate = np.array([[c, -s], [s, c]], dtype=complex)

            current_state = np.tensordot(ry_gate, current_state, axes=(1, qubit))
            current_state = np.moveaxis(current_state, 0, qubit)

        return current_state.reshape(self.dim)

    def apply_entangling_cz_chain(self, state: np.ndarray) -> np.ndarray:
        """Applies circular Controlled-Z (CZ) entangling gates between neighboring qubits."""
        current_state = state.copy()
        for k in range(self.dim):
            # If both qubit i and (i+1)%N are 1, phase is flipped (-1)
            flip_count = 0
            for i in range(self.N):
                q1 = (k >> i) & 1
                q2 = (k >> ((i + 1) % self.N)) & 1
                if q1 == 1 and q2 == 1:
                    flip_count += 1
            if flip_count % 2 == 1:
                current_state[k] = - current_state[k]
        return current_state

    def compute_energy_expectation(self, state: np.ndarray) -> float:
        """
        Computes <psi | H_C | psi> = sum_{k=0}^{2^N - 1} |psi_k|^2 * E_k.
        """
        probs = np.abs(state) ** 2
        return float(np.sum(probs * self.diag_energies))

    def compute_cvar_energy(self, state: np.ndarray, alpha: float = 1.0) -> float:
        """
        Computes CVaR_alpha over the diagonal Hamiltonian energy distribution.
        Follows Barkoutsos et al. (2020), 'Improving Variational Quantum Optimization using CVaR'.
        
        When alpha = 1.0, exactly equals standard expectation value <psi|H_C|psi>.
        When alpha in (0, 1), evaluates the conditional expectation over the lowest alpha-quantile.
        """
        if alpha >= 1.0:
            return self.compute_energy_expectation(state)

        alpha = max(1e-4, float(alpha))
        probs = np.abs(state) ** 2

        # Sort energies in ascending order
        order = np.argsort(self.diag_energies)
        sorted_e = self.diag_energies[order]
        sorted_p = probs[order]

        cum_p = np.cumsum(sorted_p)
        k_alpha = int(np.searchsorted(cum_p, alpha))
        if k_alpha >= len(sorted_e):
            k_alpha = len(sorted_e) - 1

        p_head = sorted_p[:k_alpha]
        e_head = sorted_e[:k_alpha]
        rem_p = alpha - float(np.sum(p_head))

        cvar = (np.sum(p_head * e_head) + max(0.0, rem_p) * sorted_e[k_alpha]) / alpha
        return float(cvar)


def solve_qaoa(
    qubo_mapper: QUBOMapper,
    p_layers: int = 1,
    cvar_alpha: float = 1.0,
    max_iter: int = 120,
    seed: Optional[int] = 42
) -> Dict[str, Any]:
    """
    Executes QAOA on the microfinance QUBO instance with optional CVaR objective.

    Parameters:
    -----------
    qubo_mapper : QUBOMapper
        The mapped problem instance.
    p_layers : int
        Number of alternating (cost, mixer) layers.
    cvar_alpha : float
        CVaR confidence quantile in (0, 1]. Defaults to 1.0 (standard QAOA).
    max_iter : int
        Maximum iterations for classical COBYLA optimizer.
    seed : int, optional
        Seed for variational angle initialization.

    Returns:
    --------
    dict containing solver results, optimal allocation, and convergence history.
    """
    if seed is not None:
        np.random.seed(seed)

    sim = QuantumSimulator(qubo_mapper)
    history: List[float] = []

    # Initial random angles: gamma in [0, 2*pi], beta in [0, pi]
    init_gamma = np.random.uniform(0.1, np.pi, size=p_layers)
    init_beta = np.random.uniform(0.1, np.pi / 2.0, size=p_layers)
    init_params = np.concatenate([init_gamma, init_beta])

    def qaoa_objective(params: np.ndarray) -> float:
        gamma_vec = params[:p_layers]
        beta_vec = params[p_layers:]

        # Evolve state
        state = sim.get_uniform_superposition()
        for k in range(p_layers):
            state = sim.apply_cost_unitary(state, gamma_vec[k])
            state = sim.apply_mixer_unitary(state, beta_vec[k])

        if cvar_alpha < 1.0:
            obj_val = sim.compute_cvar_energy(state, alpha=cvar_alpha)
        else:
            obj_val = sim.compute_energy_expectation(state)

        history.append(obj_val)
        return obj_val

    # Classical parameter optimization via COBYLA
    opt_res = minimize(
        qaoa_objective,
        init_params,
        method="COBYLA",
        options={"maxiter": max_iter, "tol": 1e-4}
    )

    # Evaluate final statevector with optimal parameters
    opt_gamma = opt_res.x[:p_layers]
    opt_beta = opt_res.x[p_layers:]

    final_state = sim.get_uniform_superposition()
    for k in range(p_layers):
        final_state = sim.apply_cost_unitary(final_state, opt_gamma[k])
        final_state = sim.apply_mixer_unitary(final_state, opt_beta[k])

    probs = np.abs(final_state) ** 2
    best_state_idx = int(np.argmax(probs))

    # Bitstring of the maximum likelihood state
    best_alloc = np.array([(best_state_idx >> i) & 1 for i in range(qubo_mapper.N)], dtype=int)
    best_alloc_energy = qubo_mapper.evaluate_bitstring_energy(best_alloc)

    solver_label = f"QAOA_p{p_layers}" if cvar_alpha >= 1.0 else f"CVaR_QAOA_p{p_layers}_a{int(cvar_alpha*100)}"

    return {
        "solver": solver_label,
        "p_layers": p_layers,
        "cvar_alpha": cvar_alpha,
        "allocation": best_alloc,
        "energy": float(opt_res.fun),
        "best_state_energy": float(best_alloc_energy),
        "probabilities": probs,
        "convergence_history": history,
        "optimal_gamma": opt_gamma.tolist(),
        "optimal_beta": opt_beta.tolist(),
        "best_state_index": best_state_idx,
        "optimizer_evals": len(history)
    }


def solve_cvar_qaoa(
    qubo_mapper: QUBOMapper,
    p_layers: int = 1,
    cvar_alpha: float = 0.25,
    max_iter: int = 120,
    seed: Optional[int] = 42
) -> Dict[str, Any]:
    """
    Executes CVaR-QAOA (Conditional Value-at-Risk QAOA, Barkoutsos et al. 2020)
    for tail-risk-aware quantum combinatorial optimization.
    """
    return solve_qaoa(
        qubo_mapper=qubo_mapper,
        p_layers=p_layers,
        cvar_alpha=cvar_alpha,
        max_iter=max_iter,
        seed=seed
    )


def solve_vqe(
    qubo_mapper: QUBOMapper,
    ansatz_layers: int = 2,
    max_iter: int = 150,
    seed: Optional[int] = 42
) -> Dict[str, Any]:
    """
    Executes VQE using a Hardware-Efficient RealAmplitudes ansatz (Ry + CZ chain).
    """
    if seed is not None:
        np.random.seed(seed)

    sim = QuantumSimulator(qubo_mapper)
    N = qubo_mapper.N
    n_params = N * (ansatz_layers + 1)
    init_params = np.random.normal(loc=0.0, scale=0.1, size=n_params)

    history: List[float] = []

    def vqe_objective(params: np.ndarray) -> float:
        # Start in |0>^{\otimes N} state
        state = np.zeros(sim.dim, dtype=complex)
        state[0] = 1.0

        for layer in range(ansatz_layers):
            angles = params[layer * N : (layer + 1) * N]
            state = sim.apply_ry_rotations(state, angles)
            state = sim.apply_entangling_cz_chain(state)

        # Final rotation layer
        final_angles = params[ansatz_layers * N : (ansatz_layers + 1) * N]
        state = sim.apply_ry_rotations(state, final_angles)

        energy = sim.compute_energy_expectation(state)
        history.append(energy)
        return energy

    opt_res = minimize(
        vqe_objective,
        init_params,
        method="COBYLA",
        options={"maxiter": max_iter, "tol": 1e-4}
    )

    # Reconstruct final state
    state = np.zeros(sim.dim, dtype=complex)
    state[0] = 1.0
    for layer in range(ansatz_layers):
        angles = opt_res.x[layer * N : (layer + 1) * N]
        state = sim.apply_ry_rotations(state, angles)
        state = sim.apply_entangling_cz_chain(state)
    final_angles = opt_res.x[ansatz_layers * N : (ansatz_layers + 1) * N]
    final_state = sim.apply_ry_rotations(state, final_angles)

    probs = np.abs(final_state) ** 2
    best_state_idx = int(np.argmax(probs))
    best_alloc = np.array([(best_state_idx >> i) & 1 for i in range(N)], dtype=int)
    best_alloc_energy = qubo_mapper.evaluate_bitstring_energy(best_alloc)

    return {
        "solver": f"VQE_L{ansatz_layers}",
        "ansatz_layers": ansatz_layers,
        "allocation": best_alloc,
        "energy": float(opt_res.fun),
        "best_state_energy": float(best_alloc_energy),
        "probabilities": probs,
        "convergence_history": history,
        "best_state_index": best_state_idx,
        "optimizer_evals": len(history)
    }


def generate_qiskit_qaoa_circuit(
    qubo_mapper: QUBOMapper,
    p_layers: int = 1
) -> Tuple[Optional[Any], str]:
    """
    Constructs a native Qiskit QuantumCircuit representing the QAOA circuit for the problem.
    Exports the ASCII circuit diagram string for slide presentations.
    """
    if not QISKIT_AVAILABLE:
        return None, "Qiskit not installed in current environment."

    N = qubo_mapper.N
    qc = QuantumCircuit(N)

    # 1. Uniform superposition initialization
    qc.h(range(N))

    # 2. Alternating layers
    for layer in range(1, p_layers + 1):
        gamma = Parameter(f"gamma_{layer}")
        beta = Parameter(f"beta_{layer}")

        # Problem Unitary: single-qubit Rz for longitudinal fields h_i
        for i in range(N):
            if not np.isclose(qubo_mapper.h[i], 0.0):
                qc.rz(2.0 * gamma * float(qubo_mapper.h[i]), i)

        # Problem Unitary: two-qubit Rzz for exchange couplings J_ij
        for i in range(N):
            for j in range(i + 1, N):
                if not np.isclose(qubo_mapper.J[i, j], 0.0):
                    qc.rzz(2.0 * gamma * float(qubo_mapper.J[i, j]), i, j)

        # Transverse Mixer Unitary: Rx(2 * beta) across all qubits
        for i in range(N):
            qc.rx(2.0 * beta, i)

    # Measure all qubits
    qc.measure_all()

    try:
        diagram_str = qc.draw(output="text").single_string()
    except Exception:
        diagram_str = str(qc)

    return qc, diagram_str


if __name__ == "__main__":
    from src.data_generator import generate_microfinance_cohort
    from src.scoring import compute_multiobjective_scores

    df = generate_microfinance_cohort(n_applicants=6, seed=42)
    _, arr = compute_multiobjective_scores(df)

    mapper = QUBOMapper(
        composite_scores=arr["composite_scores"],
        loan_amounts=arr["loan_amounts"],
        group_indicators=arr["group_indicators"],
        total_budget=80000.0
    )

    qaoa_res = solve_qaoa(mapper, p_layers=1, max_iter=80)
    vqe_res = solve_vqe(mapper, ansatz_layers=1, max_iter=80)

    print("QAOA allocation:", qaoa_res["allocation"], "Energy:", round(qaoa_res["energy"], 3))
    print("VQE  allocation:", vqe_res["allocation"], "Energy:", round(vqe_res["energy"], 3))

    _, diagram = generate_qiskit_qaoa_circuit(mapper, p_layers=1)
    print("\nQiskit Circuit ASCII Preview:")
    print(diagram[:400] + "...")
