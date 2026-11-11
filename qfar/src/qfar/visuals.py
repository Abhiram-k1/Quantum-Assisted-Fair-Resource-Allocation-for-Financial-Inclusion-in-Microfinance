"""Publication-grade Matplotlib figures for Q-FAR (non-interactive Agg backend)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ROOT, load_config
from .experiment import build_instance, cohort_risk
from .pipeline import load_cohort_input


def _save_fig(fig: plt.Figure, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pd_calibration(results_dir: Path, fig_dir: Path) -> Path | None:
    rel_p = results_dir / "01_pd_model/test_reliability.csv"
    sp_p = results_dir / "01_pd_model/score_feature_spearman.csv"
    if not rel_p.exists():
        return None
    rel = pd.read_csv(rel_p)
    fig, axes = plt.subplots(1, 2 if sp_p.exists() else 1, figsize=(11.5 if sp_p.exists() else 5.8, 4.4))
    ax0 = axes[0] if sp_p.exists() else axes
    ax0.plot([0, 1], [0, 1], "--", color="gray", lw=1.2, label="Perfect calibration")
    ax0.plot(rel["mean_pred"], rel["observed"], "o-", color="#1f77b4", lw=2, ms=6, label="Test reliability")
    ax0.set_title("Out-of-Sample PD Calibration (Test Split)")
    ax0.set_xlabel("Mean predicted repayment probability")
    ax0.set_ylabel("Observed repayment rate")
    ax0.grid(alpha=0.25)
    ax0.legend(frameon=False)

    if sp_p.exists():
        sp = pd.read_csv(sp_p)
        ax1 = axes[1]
        num_cols = [c for c in ("F", "N", "S", "U") if c in sp.columns]
        feat_col = [c for c in sp.columns if c not in num_cols][0]
        x = np.arange(len(sp))
        w = 0.2
        for k, col in enumerate(num_cols):
            ax1.bar(x + (k - 1.5) * w, sp[col], width=w, label=col)
        ax1.axhline(0, color="black", lw=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(sp[feat_col], rotation=28, ha="right", fontsize=8)
        ax1.set_ylabel("Spearman correlation")
        ax1.set_title("Score-Component Independence (Test Split)")
        ax1.grid(axis="y", alpha=0.25)
        ax1.legend(ncol=4, frameon=False, fontsize=8)

    return _save_fig(fig, fig_dir / "01_pd_calibration.png")


def plot_qubo_heatmap(fig_dir: Path, cfg_path: str | Path | None = None, cohort_idx: int = 0) -> Path:
    cfg = load_config(cfg_path)
    coh, _ = load_cohort_input(cfg, cohort_idx=cohort_idx)
    risk = cohort_risk(coh, cfg, scenario_seed=int(cfg["seed"]) + 1000 + cohort_idx)
    inst = build_instance(coh, cfg, risk)
    q = inst.q

    labels = [f"x{i}" for i in range(q.N)] + [f"s{k}" for k in range(q.K)]
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    vmax = float(np.max(np.abs(q.M)))
    im = ax.imshow(q.M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.axvline(q.N - 0.5, color="black", ls="--", lw=1.2)
    ax.axhline(q.N - 0.5, color="black", ls="--", lw=1.2)
    ax.set_xticks(range(q.n))
    ax.set_xticklabels(labels, fontsize=8, rotation=45)
    ax.set_yticks(range(q.n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(
        f"P1 QUBO Matrix M (N={q.N} applicants, K={q.K} slack qubits, "
        f"lambda_B={q.lam_B:.1f})"
    )
    fig.colorbar(im, ax=ax, shrink=0.82, label="Coupling coefficient")
    return _save_fig(fig, fig_dir / "02_qubo_matrix_heatmap.png")


def plot_solver_comparison(results_dir: Path, fig_dir: Path) -> Path | None:
    p = results_dir / "03_allocation/allocation_summary.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p).sort_values("J1")
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.0, 4.8))

    y = np.arange(len(df))
    ax0.barh(y - 0.18, df["J1"], xerr=df["J1_sd"], height=0.35, color="#1f77b4", label="P1 Obj (J1)")
    ax0.barh(y + 0.18, df["J2"], xerr=df["J2_sd"], height=0.35, color="#ff7f0e", label="P2 Obj (J2)")
    ax0.set_yticks(y)
    ax0.set_yticklabels(df["solver"], fontsize=8)
    ax0.invert_yaxis()
    ax0.set_xlabel("Objective value (lower is better)")
    ax0.set_title("Classical vs Quantum Objective (30 Test Cohorts)")
    ax0.grid(axis="x", alpha=0.25)
    ax0.legend(frameon=False, fontsize=8)

    ax1.scatter(df["abs_dpd"], df["cvar_inr"] / 1000.0, s=90, c=np.arange(len(df)), cmap="viridis", zorder=3)
    df_lbl = df.assign(xr=df["abs_dpd"].round(4), yr=(df["cvar_inr"] / 1000.0).round(2))
    for (xv, yv), grp in df_lbl.groupby(["xr", "yr"], sort=False):
        label = " / ".join(grp["solver"].tolist())
        dx, ha = (-8, "right") if xv > 0.5 else (6, "left")
        ax1.annotate(label, (xv, yv), textcoords="offset points", xytext=(dx, 4), ha=ha, fontsize=7.2)
    ax1.set_xlim(max(0.0, df["abs_dpd"].min() - 0.05), df["abs_dpd"].max() + 0.08)
    ax1.set_xlabel("|DPD| = |AR_A - AR_B| (lower is fairer)")
    ax1.set_ylabel("95% Portfolio Loss CVaR (k INR, lower is safer)")
    ax1.set_title("Fairness vs Tail-Risk Across Solvers")
    ax1.grid(alpha=0.25)
    return _save_fig(fig, fig_dir / "03_solver_comparison.png")


def plot_quantum_vs_uniform(results_dir: Path, fig_dir: Path) -> Path | None:
    p = results_dir / "03_allocation/quantum_summary.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    x = np.arange(len(df))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.2, 4.5))

    w = 0.28
    popt_err = np.vstack([
        np.minimum(df["P_opt_exact_sd"], df["P_opt_exact"]),
        np.minimum(df["P_opt_exact_sd"], 1.0 - df["P_opt_exact"]),
    ])
    ax0.bar(x - w / 2, df["P_opt_exact"], yerr=popt_err, width=w, color="#2ca02c", label="Trained circuit")
    ax0.bar(x + w / 2, df["P_opt_uniform"], width=w, color="#7f7f7f", label="Uniform 1/2^N")
    ax0.set_xticks(x)
    ax0.set_xticklabels(df["solver"], rotation=20, ha="right", fontsize=8)
    ax0.set_ylim(0, None)
    ax0.set_ylabel("P(x = x*_P1)")
    ax0.set_title("Ground-State Amplification vs Uniform Baseline")
    ax0.grid(axis="y", alpha=0.25)
    ax0.legend(frameon=False, fontsize=8)

    pfeas_err = np.vstack([
        np.minimum(df["P_feasible_exact_sd"], df["P_feasible_exact"]),
        np.minimum(df["P_feasible_exact_sd"], 1.0 - df["P_feasible_exact"]),
    ])
    ax1.bar(x - w / 2, df["P_feasible_exact"], yerr=pfeas_err, width=w, color="#1f77b4", label="Exact P(feasible)")
    ax1.bar(x + w / 2, df["P_feasible_uniform"], width=w, color="#bcbd22", label="Uniform P(feasible)")
    ax1.plot(x, df["hit_opt_P1"], "D--", color="#d62728", lw=1.6, label="1024-shot P1 hit rate")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["solver"], rotation=20, ha="right", fontsize=8)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Probability / hit rate")
    ax1.set_title("Feasibility Mass & Multi-Shot Optimal Hit Rate")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(frameon=False, fontsize=8)

    return _save_fig(fig, fig_dir / "04_quantum_vs_uniform.png")


def plot_ablation_and_weights(results_dir: Path, fig_dir: Path) -> list[Path]:
    saved: list[Path] = []
    pen_p = results_dir / "04_component_ablation/penalties_summary.csv"
    if pen_p.exists():
        df = pd.read_csv(pen_p)
        x = np.arange(len(df))
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.2, 4.4))
        ax0.bar(x, df["abs_dpd"], yerr=df["abs_dpd_sd"], color="#9467bd")
        ax0.set_xticks(x)
        ax0.set_xticklabels(df["setting"], rotation=22, ha="right", fontsize=8)
        ax0.set_ylabel("|DPD|")
        ax0.set_title("Component Ablation (E1-E7): Demographic Parity Gap")
        ax0.grid(axis="y", alpha=0.25)

        ax1.bar(x, df["cvar_inr"] / 1000.0, yerr=df["cvar_inr_sd"] / 1000.0, color="#d62728")
        ax1.set_xticks(x)
        ax1.set_xticklabels(df["setting"], rotation=22, ha="right", fontsize=8)
        ax1.set_ylabel("95% Portfolio CVaR (k INR)")
        ax1.set_title("Component Ablation (E1-E7): Tail Loss Reduction")
        ax1.grid(axis="y", alpha=0.25)
        saved.append(_save_fig(fig, fig_dir / "05_ablation_risk_fairness.png"))

    wt_p = results_dir / "02_weight_sensitivity/weight_summary.csv"
    if wt_p.exists():
        wdf = pd.read_csv(wt_p)
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.0, 4.6))
        sc0 = ax0.scatter(wdf["wF"], wdf["wS"], c=wdf["utility"], cmap="plasma", s=95, edgecolors="k", lw=0.4)
        ax0.set_xlabel("w_F (Financial weight)")
        ax0.set_ylabel("w_S (Social / Inclusion weight)")
        ax0.set_title("Simplex Weight Grid: Portfolio Utility (w_N = 1 - w_F - w_S)")
        ax0.grid(alpha=0.25)
        fig.colorbar(sc0, ax=ax0, label="Mean utility U.x")

        sc1 = ax1.scatter(wdf["wF"], wdf["wS"], c=wdf["jaccard_vs_baseline"], cmap="viridis", s=95, edgecolors="k", lw=0.4)
        ax1.set_xlabel("w_F (Financial weight)")
        ax1.set_ylabel("w_S (Social / Inclusion weight)")
        ax1.set_title("Selection Stability: Jaccard vs Equal Weights (1/3, 1/3, 1/3)")
        ax1.grid(alpha=0.25)
        fig.colorbar(sc1, ax=ax1, label="Mean Jaccard similarity")
        saved.append(_save_fig(fig, fig_dir / "06_weight_sensitivity.png"))
    return saved


def plot_scaling_and_noise(results_dir: Path, fig_dir: Path) -> list[Path]:
    saved: list[Path] = []
    sc_p = results_dir / "05_scaling_and_noise/scaling_per_cohort.csv"
    if sc_p.exists():
        df = pd.read_csv(sc_p)
        qsols = [s for s in ("qaoa_p1", "qaoa_p2", "warm_start_qaoa_p1", "warm_start_qaoa_p2") if s in df["solver"].unique()]
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.4, 4.6))
        for sol in qsols:
            g = df[df["solver"] == sol].groupby("N")[["P_opt_exact", "P_feasible_exact"]].mean().reset_index()
            ax0.plot(g["N"], g["P_opt_exact"], "o-", lw=2, label=sol)
            ax1.plot(g["N"], g["P_feasible_exact"], "s-", lw=2, label=sol)
        ax0.set_xlabel("Cohort applicants N")
        ax0.set_ylabel("Exact optimal ground-state probability P(x*)")
        ax0.set_title("Scaling: Standard QAOA vs Warm-Start QAOA")
        ax0.grid(alpha=0.25)
        ax0.legend(frameon=False, fontsize=8)

        ax1.set_xlabel("Cohort applicants N")
        ax1.set_ylabel("Exact feasible probability mass")
        ax1.set_title("Feasible Subspace Concentration vs Problem Size N")
        ax1.set_ylim(0, 1.05)
        ax1.grid(alpha=0.25)
        ax1.legend(frameon=False, fontsize=8)
        saved.append(_save_fig(fig, fig_dir / "07_scaling_warmstart.png"))

    nz_p = results_dir / "05_scaling_and_noise/noisy_zne_per_cohort.csv"
    if nz_p.exists():
        ndf = pd.read_csv(nz_p)
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.4, 4.6))
        sizes = sorted(ndf["N"].unique())
        for label, ls in (("qaoa_p1", "--"), ("warm_start_qaoa_p1", "-")):
            sub = ndf[ndf["solver"] == label]
            if sub.empty:
                continue
            g = sub.groupby("N")[["ideal_expectation", "noisy_expectation_c1", "zne_expectation", "err_c1", "err_zne"]].mean().reset_index()
            ax0.plot(g["N"], g["noisy_expectation_c1"], f"o{ls}", label=f"{label} noisy (c=1)")
            ax0.plot(g["N"], g["zne_expectation"], f"^{ls}", label=f"{label} ZNE (0)")
            ax0.plot(g["N"], g["ideal_expectation"], f"s:", alpha=0.8, label=f"{label} ideal")

            x = np.arange(len(g))
            off = -0.18 if label == "qaoa_p1" else 0.18
            ax1.bar(x + off - 0.08, g["err_c1"], width=0.15, label=f"{label} |E_c1 - E_ideal|")
            ax1.bar(x + off + 0.08, g["err_zne"], width=0.15, label=f"{label} |E_ZNE - E_ideal|")
        ax0.set_xticks(sizes)
        ax0.set_xlabel("Cohort size N")
        ax0.set_ylabel("QUBO energy expectation <H>")
        ax0.set_title("Qiskit Depolarising + Readout Noise & ZNE Recovery")
        ax0.grid(alpha=0.25)
        ax0.legend(frameon=False, fontsize=7.5)

        ax1.set_xticks(np.arange(len(sizes)))
        ax1.set_xticklabels([f"N={n}" for n in sizes])
        ax1.set_ylabel("Absolute energy bias vs ideal")
        ax1.set_title("Zero-Noise Extrapolation Bias Reduction")
        ax1.grid(axis="y", alpha=0.25)
        ax1.legend(frameon=False, fontsize=7.5)
        saved.append(_save_fig(fig, fig_dir / "08_noisy_zne_recovery.png"))
    return saved


def generate_all_visuals(
    results_dir: Path | None = None,
    fig_dir: Path | None = None,
    cfg_path: str | Path | None = None,
) -> list[Path]:
    """Generate all available Q-FAR figures from saved experiment CSVs."""
    rdir = Path(results_dir) if results_dir is not None else (ROOT / "results")
    fdir = Path(fig_dir) if fig_dir is not None else (rdir / "figures")
    generated: list[Path] = []
    for res in (
        plot_pd_calibration(rdir, fdir),
        plot_qubo_heatmap(fdir, cfg_path=cfg_path),
        plot_solver_comparison(rdir, fdir),
        plot_quantum_vs_uniform(rdir, fdir),
    ):
        if res is not None:
            generated.append(res)
    generated.extend(plot_ablation_and_weights(rdir, fdir))
    generated.extend(plot_scaling_and_noise(rdir, fdir))
    return generated

