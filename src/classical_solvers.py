"""
Classical Baseline Solvers for Microfinance Capital Allocation.
Implements:
1. Greedy Ratio Knapsack (Value/Cost heuristic)
2. Exact Combinatorial Solver (Global Ground Truth)
3. Simulated Annealing (Classical Metaheuristic)
"""

from typing import Dict, Tuple, Any, Optional
import numpy as np
from src.qubo_builder import QUBOMapper


def solve_greedy_knapsack(
    composite_scores: np.ndarray,
    loan_amounts: np.ndarray,
    budget: float,
    qubo_mapper: Optional[QUBOMapper] = None
) -> Dict[str, Any]:
    """
    Greedy Knapsack Baseline.
    Ranks applicants by ratio = composite_score / loan_amount,
    and greedily packs items until budget is reached.
    Notice: Completely blind to demographic fairness and cross-penalties!
    """
    N = len(composite_scores)
    ratios = composite_scores / np.maximum(1e-5, loan_amounts)
    sorted_indices = np.argsort(ratios)[::-1]

    x_alloc = np.zeros(N, dtype=int)
    current_cost = 0.0

    for idx in sorted_indices:
        if current_cost + loan_amounts[idx] <= budget:
            x_alloc[idx] = 1
            current_cost += loan_amounts[idx]

    energy = None
    if qubo_mapper is not None:
        energy = qubo_mapper.evaluate_bitstring_energy(x_alloc)

    return {
        "solver": "Greedy_Ratio_Knapsack",
        "allocation": x_alloc,
        "total_cost": float(current_cost),
        "energy": energy,
        "is_feasible": bool(current_cost <= budget)
    }


def solve_exact_qubo(qubo_mapper: QUBOMapper) -> Dict[str, Any]:
    """
    Exact Combinatorial Ground-State Solver.
    Evaluates all 2^N bitstrings to determine the true theoretical minimum
    of the QUBO Hamiltonian landscape.
    """
    N = qubo_mapper.N
    dim = 2 ** N
    diag_energies = qubo_mapper.get_diagonal_hamiltonian()

    best_idx = int(np.argmin(diag_energies))
    min_energy = float(diag_energies[best_idx])

    # Convert best_idx to binary allocation vector
    best_allocation = np.array([(best_idx >> i) & 1 for i in range(N)], dtype=int)

    return {
        "solver": "Classical_Exact_Optimal",
        "allocation": best_allocation,
        "energy": min_energy,
        "ground_state_index": best_idx,
        "total_states_evaluated": dim
    }


def solve_simulated_annealing(
    qubo_mapper: QUBOMapper,
    initial_temp: float = 10.0,
    min_temp: float = 0.01,
    cooling_rate: float = 0.985,
    max_steps: int = 1500,
    seed: Optional[int] = 42
) -> Dict[str, Any]:
    """
    Classical Simulated Annealing Solver (Metropolis-Hastings algorithm).
    Iteratively explores the QUBO energy landscape through single-bit flips
    with thermal acceptance probability exp(-delta_E / T).
    """
    if seed is not None:
        np.random.seed(seed)

    N = qubo_mapper.N
    # Initialize with random binary vector
    current_x = np.random.choice([0, 1], size=N)
    current_energy = qubo_mapper.evaluate_bitstring_energy(current_x)

    best_x = current_x.copy()
    best_energy = current_energy

    T = float(initial_temp)
    step = 0
    energy_history = [best_energy]

    while T > min_temp and step < max_steps:
        # Propose bit-flip neighbor
        flip_idx = np.random.randint(0, N)
        candidate_x = current_x.copy()
        candidate_x[flip_idx] = 1 - candidate_x[flip_idx]

        candidate_energy = qubo_mapper.evaluate_bitstring_energy(candidate_x)
        delta_E = candidate_energy - current_energy

        # Metropolis acceptance criterion
        if delta_E < 0 or np.random.uniform(0, 1) < np.exp(-delta_E / max(1e-6, T)):
            current_x = candidate_x
            current_energy = candidate_energy

            if current_energy < best_energy:
                best_x = current_x.copy()
                best_energy = current_energy

        energy_history.append(best_energy)
        T *= cooling_rate
        step += 1

    return {
        "solver": "Simulated_Annealing",
        "allocation": best_x,
        "energy": float(best_energy),
        "steps_taken": step,
        "energy_history": energy_history
    }


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

    greedy_res = solve_greedy_knapsack(arr["composite_scores"], arr["loan_amounts"], 80000.0, mapper)
    exact_res = solve_exact_qubo(mapper)
    sa_res = solve_simulated_annealing(mapper)

    print("Greedy allocation:", greedy_res["allocation"], "Energy:", round(greedy_res["energy"], 3))
    print("Exact  allocation:", exact_res["allocation"], "Energy:", round(exact_res["energy"], 3))
    print("SA     allocation:", sa_res["allocation"], "Energy:", round(sa_res["energy"], 3))
