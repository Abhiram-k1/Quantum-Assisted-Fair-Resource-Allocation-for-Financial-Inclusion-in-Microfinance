"""Tests for Q-FAR extensions: Warm-Start QAOA, Qiskit noise + ZNE, unified pipeline,
visuals, and interactive dashboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qfar.pipeline import run_allocation
from qfar.qubo import AllocationQUBO
from qfar.quantum import StateVec, _ws_mixer_gates, solve_warm_start_qaoa


def _tiny_qubo():
    return AllocationQUBO(
        U=np.array([0.85, 0.72, 0.64, 0.50]),
        w=np.array([2, 3, 2, 1]),
        B=4,
        groups=np.array([0, 0, 1, 1]),
        eta=0.05,
        lam_F=0.25,
    )


def test_warm_start_mixer_eigenstate_and_unitarity():
    thetas = np.array([0.4, 1.1, np.pi / 2, 2.3])
    beta = 0.63
    gates = _ws_mixer_gates(thetas, beta)
    phase = np.exp(1j * beta)
    for th, g in zip(thetas, gates):
        assert np.allclose(g @ g.conj().T, np.eye(2), atol=1e-12)
        ket = np.array([np.cos(th / 2.0), np.sin(th / 2.0)], dtype=complex)
        assert np.allclose(g @ ket, phase * ket, atol=1e-12)

    # At theta = pi/2, warm-start mixer with -beta equals standard exp(-i beta X)
    std_g = np.array([[np.cos(beta), -1j * np.sin(beta)], [-1j * np.sin(beta), np.cos(beta)]])
    ws_pi2 = _ws_mixer_gates(np.array([np.pi / 2]), -beta)[0]
    assert np.allclose(ws_pi2, std_g, atol=1e-12)


def test_warm_start_qaoa_feasible_and_bounded():
    q = _tiny_qubo()
    res = solve_warm_start_qaoa(q, p=1, eps=0.05, restarts=3, max_iter=150, shots=512, seed=7)
    assert res["feasible"] is True
    assert res["solver"] == "warm_start_qaoa_p1"
    assert np.isclose(res["probs"].sum(), 1.0, atol=1e-9)
    assert np.all((res["c_star"] >= 0.05 - 1e-12) & (res["c_star"] <= 0.95 + 1e-12))
    mx = q.marginal_x(res["probs"])
    assert mx.max() > 1.0 / (2 ** q.N)


def test_qiskit_circuit_parity_and_folding():
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator, Statevector
    from qfar.qiskit_backend import build_qaoa_circuit, export_circuit_artifacts, fold_qaoa_circuit

    q = _tiny_qubo()
    sv = StateVec(q)
    params = np.array([0.55, 0.31, 0.42, 0.19])

    # Standard p=2 StateVec vs Qiskit Statevector parity
    st = sv.plus()
    for l in range(2):
        st = sv.mixer(st * np.exp(-1j * params[l] * sv.Etil), params[2 + l])
    p_sv = np.abs(st) ** 2
    qc = build_qaoa_circuit(q, params, p=2)
    p_qk = Statevector.from_instruction(qc).probabilities()
    assert np.allclose(p_sv, p_qk, atol=1e-10)

    # Warm-start p=1 parity
    ws = solve_warm_start_qaoa(q, p=1, restarts=2, max_iter=80, seed=0, sv=sv)
    thetas = 2.0 * np.arcsin(np.sqrt(ws["c_star"]))
    qc_ws = build_qaoa_circuit(q, ws["params"], p=1, warm_start_thetas=thetas)
    assert np.allclose(ws["probs"], Statevector.from_instruction(qc_ws).probabilities(), atol=1e-10)

    # Unitary folding preserves operator and distribution cleanly
    qc3 = fold_qaoa_circuit(qc, scale_factor=3)
    qc5 = fold_qaoa_circuit(qc, scale_factor=5)
    assert np.allclose(Statevector.from_instruction(qc3).probabilities(), p_qk, atol=1e-10)
    assert np.allclose(Statevector.from_instruction(qc5).probabilities(), p_qk, atol=1e-10)
    assert qc3.size() > qc.size()

    art = export_circuit_artifacts(q, ws["params"], p=1, warm_start_thetas=thetas)
    assert "OPENQASM 2.0;" in art["qasm2"]
    assert art["transpiled_depth"] > 0 and len(art["circuit_text"]) > 10


def test_noisy_simulation_and_zne():
    pytest.importorskip("qiskit")
    from qfar.qiskit_backend import run_noisy_and_zne, simulate_noisy_qaoa, zero_noise_extrapolation

    # Analytical polynomial recovery at c = 0
    poly = [1.0 + 0.4 * c + 0.12 * (c ** 2) for c in (1, 3, 5)]
    zne = zero_noise_extrapolation([1, 3, 5], poly)
    assert np.isclose(zne["zne_energy"], 1.0, atol=1e-11)

    q = AllocationQUBO(
        U=np.array([0.8, 0.7, 0.6]),
        w=np.array([2, 2, 1]),
        B=3,
        groups=np.array([0, 1, 0]),
        eta=0.05,
        lam_F=0.2,
    )
    sv = StateVec(q)
    params = np.array([0.65, 0.35])
    st = sv.mixer(sv.plus() * np.exp(-1j * params[0] * sv.Etil), params[1])
    ideal_probs = np.abs(st) ** 2

    clean_dm = simulate_noisy_qaoa(q, params, p=1, p1_depol=0.0, p2_depol=0.0, p_readout=0.0,
                                   method="density_matrix", seed=0)
    assert np.allclose(clean_dm["probs"], ideal_probs, atol=1e-10)

    dm = simulate_noisy_qaoa(q, params, p=1, p1_depol=0.04, p2_depol=0.12, p_readout=0.03,
                             method="density_matrix", seed=0)
    tr = simulate_noisy_qaoa(q, params, p=1, p1_depol=0.04, p2_depol=0.12, p_readout=0.03,
                             method="trajectory", n_trajectories=6000, seed=0)
    assert np.max(np.abs(dm["probs"] - tr["probs"])) < 0.015

    res_zne = run_noisy_and_zne(q, params, p=1, scales=(1, 3, 5), method="density_matrix", seed=0)
    assert len(res_zne["per_scale"]) == 3
    assert np.isfinite(res_zne["zne"]["zne_energy"])


def test_pipeline_synthetic_and_legacy_csv(tmp_path):
    # Synthetic cohort 0
    res = run_allocation(
        cohort_idx=0,
        primary_solver="warm_start_qaoa_p1",
        solvers=("exact_p1", "milp_p2", "greedy", "sa", "warm_start_qaoa_p1"),
        include_qiskit_circuit=False,
        include_noisy_zne=False,
    )
    assert set(["approved", "F", "N", "S", "U", "w_units"]).issubset(res.cohort.columns)
    assert len(res.selected_ids) == int(res.cohort["approved"].sum())
    assert (res.comparison["feasible"]).all()
    assert (res.comparison["budget_utilisation"] <= 1.0 + 1e-9).all()

    # Legacy v1 schema CSV (demographic_group, no village, no repaid, no inclusion columns)
    df_legacy = pd.DataFrame({
        "applicant_id": [f"L{i}" for i in range(6)],
        "demographic_group": [
            "Group_A_Marginalized", "Group_A_Marginalized", "Group_A_Marginalized",
            "Group_B_General", "Group_B_General", "Group_B_General",
        ],
        "monthly_income": [12000, 15000, 9500, 11000, 14000, 8500],
        "dti_ratio": [0.28, 0.35, 0.42, 0.25, 0.31, 0.45],
        "repayment_history": [0.92, 0.88, 0.78, 0.90, 0.85, 0.72],
        "poverty_index": [0.75, 0.60, 0.85, 0.70, 0.55, 0.88],
        "dependents": [3, 2, 4, 2, 1, 5],
        "is_women_led": ["True", "False", "True", "True", "False", "True"],
        "requested_amount": [10000, 15000, 10000, 10000, 15000, 10000],
    })
    csv_p = tmp_path / "legacy_v1.csv"
    df_legacy.to_csv(csv_p, index=False)

    res_leg = run_allocation(
        csv_path=csv_p,
        primary_solver="milp_p2",
        solvers=("exact_p1", "milp_p2", "sa", "warm_start_qaoa_p1"),
    )
    assert res_leg.metadata["social_kind"] == "priority"
    assert (res_leg.comparison["feasible"]).all()
    assert len(res_leg.cohort) == 6


def test_visuals_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    from qfar.visuals import plot_qubo_heatmap

    out = plot_qubo_heatmap(tmp_path, cohort_idx=0)
    assert out.exists() and out.stat().st_size > 1000


def test_dashboard_smoke():
    pytest.importorskip("dash")
    pytest.importorskip("plotly")
    from qfar.dashboard import create_app

    app = create_app()
    assert app.layout is not None
    client = app.server.test_client()
    assert client.get("/").status_code == 200

