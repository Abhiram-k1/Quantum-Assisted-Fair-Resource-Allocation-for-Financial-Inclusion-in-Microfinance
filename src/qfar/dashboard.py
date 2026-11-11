"""Interactive Dash + Plotly dashboard for Q-FAR allocation decisions."""
from __future__ import annotations

from pathlib import Path

from .config import ROOT
from .pipeline import run_allocation

FAST_UI_SOLVERS = (
    "exact_p1",
    "exact_p2",
    "milp_p2",
    "milp_hard_fair",
    "greedy",
    "sa",
    "qaoa_p1",
    "cvar_qaoa_p1",
    "warm_start_qaoa_p1",
)


def _require_dash():
    try:
        from dash import Dash, Input, Output, dcc, html
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Dash and Plotly are required for qfar.dashboard. "
            "Install with: pip install -e '.[full]'"
        ) from exc
    return Dash, Input, Output, dcc, html, go


def _kpi_card(html, title: str, value: str, subtitle: str = ""):
    return html.Div(
        [
            html.Div(title, style={"fontSize": "12px", "color": "#555", "textTransform": "uppercase"}),
            html.Div(value, style={"fontSize": "22px", "fontWeight": "700", "color": "#162447", "marginTop": "4px"}),
            html.Div(subtitle, style={"fontSize": "11px", "color": "#777", "marginTop": "2px"}),
        ],
        style={
            "background": "#f8fafc",
            "border": "1px solid #dbe2ea",
            "borderRadius": "8px",
            "padding": "12px 14px",
            "minWidth": "145px",
            "flex": "1",
        },
    )


def _html_table(html, df, cols):
    header = html.Tr([html.Th(c, style={"padding": "6px 10px", "borderBottom": "2px solid #cbd5e1", "textAlign": "left"}) for c in cols])
    rows = []
    for _, r in df.iterrows():
        bg = "#eefaf1" if bool(r.get("approved", False)) else "#ffffff"
        cells = []
        for c in cols:
            v = r[c]
            txt = f"{v:.3f}" if isinstance(v, float) else str(v)
            cells.append(html.Td(txt, style={"padding": "6px 10px", "borderBottom": "1px solid #e2e8f0", "fontSize": "13px"}))
        rows.append(html.Tr(cells, style={"backgroundColor": bg}))
    return html.Table(
        [html.Thead(header), html.Tbody(rows)],
        style={"width": "100%", "borderCollapse": "collapse", "background": "#fff", "border": "1px solid #dbe2ea"},
    )


def create_app(cfg_path: str | Path | None = None):
    """Build and configure the interactive Q-FAR Dash web application."""
    Dash, Input, Output, dcc, html, go = _require_dash()
    legacy_csv = ROOT.parent / "data/sample_applicants.csv"

    app = Dash(__name__, title="Q-FAR Decision Dashboard")
    source_opts = [{"label": "Synthetic Test Pool (Q-FAR v2 Cohort)", "value": "test_pool"}]
    if legacy_csv.exists():
        source_opts.append({"label": "v1 Sample Cohort (sample_applicants.csv)", "value": "v1_sample"})

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H2("Q-FAR: Quantum Fairness- & Risk-Aware Microfinance Allocation", style={"margin": "0 0 4px 0"}),
                    html.Div("Interactive multi-objective portfolio selector (P1 QUBO vs P2 MILP + Warm-Start QAOA)",
                             style={"color": "#475569", "fontSize": "14px"}),
                ],
                style={"marginBottom": "16px", "borderBottom": "2px solid #e2e8f0", "paddingBottom": "10px"},
            ),
            html.Div(
                [
                    html.Div([
                        html.Label("Cohort Source", style={"fontWeight": "600", "fontSize": "13px"}),
                        dcc.Dropdown(id="cohort-source", options=source_opts, value="test_pool", clearable=False),
                    ], style={"flex": "1.3", "minWidth": "210px"}),
                    html.Div([
                        html.Label("Cohort Index (0-29)", style={"fontWeight": "600", "fontSize": "13px"}),
                        dcc.Slider(id="cohort-idx", min=0, max=29, step=1, value=0,
                                   marks={0: "0", 10: "10", 20: "20", 29: "29"}),
                    ], style={"flex": "1", "minWidth": "170px"}),
                    html.Div([
                        html.Label("Primary Solver", style={"fontWeight": "600", "fontSize": "13px"}),
                        dcc.Dropdown(
                            id="primary-solver",
                            options=[{"label": s, "value": s} for s in FAST_UI_SOLVERS],
                            value="warm_start_qaoa_p1",
                            clearable=False,
                        ),
                    ], style={"flex": "1.2", "minWidth": "190px"}),
                ],
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "14px"},
            ),
            html.Div(
                [
                    html.Div([
                        html.Label("Financial w_F", style={"fontSize": "12px", "fontWeight": "600"}),
                        dcc.Slider(id="w-F", min=0.0, max=1.0, step=0.05, value=0.35),
                    ], style={"flex": "1", "minWidth": "140px"}),
                    html.Div([
                        html.Label("Need w_N", style={"fontSize": "12px", "fontWeight": "600"}),
                        dcc.Slider(id="w-N", min=0.0, max=1.0, step=0.05, value=0.35),
                    ], style={"flex": "1", "minWidth": "140px"}),
                    html.Div([
                        html.Label("Social w_S", style={"fontSize": "12px", "fontWeight": "600"}),
                        dcc.Slider(id="w-S", min=0.0, max=1.0, step=0.05, value=0.30),
                    ], style={"flex": "1", "minWidth": "140px"}),
                    html.Div([
                        html.Label("Budget Fraction", style={"fontSize": "12px", "fontWeight": "600"}),
                        dcc.Slider(id="budget-frac", min=0.25, max=0.85, step=0.05, value=0.50),
                    ], style={"flex": "1", "minWidth": "140px"}),
                    html.Div([
                        html.Label("Fairness lambda_F", style={"fontSize": "12px", "fontWeight": "600"}),
                        dcc.Slider(id="lam-F", min=0.0, max=5.0, step=0.5, value=2.0),
                    ], style={"flex": "1", "minWidth": "140px"}),
                    html.Div([
                        html.Label("Variance eta", style={"fontSize": "12px", "fontWeight": "600"}),
                        dcc.Slider(id="eta", min=0.0, max=3.0, step=0.25, value=1.0),
                    ], style={"flex": "1", "minWidth": "140px"}),
                ],
                style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "background": "#f8fafc",
                       "padding": "12px", "borderRadius": "8px", "border": "1px solid #e2e8f0", "marginBottom": "14px"},
            ),
            html.Div(id="kpi-row", style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "16px"}),
            html.Div(
                [
                    dcc.Graph(id="obj-chart", style={"flex": "1", "minWidth": "340px"}),
                    dcc.Graph(id="risk-fairness-chart", style={"flex": "1", "minWidth": "340px"}),
                ],
                style={"display": "flex", "gap": "14px", "flexWrap": "wrap", "marginBottom": "16px"},
            ),
            html.H4("Applicant Decisions (green rows approved by primary solver)", style={"margin": "8px 0"}),
            html.Div(id="applicant-table", style={"overflowX": "auto", "marginBottom": "16px"}),
            html.Details([
                html.Summary("Show Qiskit QAOA / Warm-Start Circuit Diagram", style={"cursor": "pointer", "fontWeight": "600"}),
                html.Pre(id="circuit-pre", style={"background": "#0f172a", "color": "#e2e8f0", "padding": "12px",
                                                 "borderRadius": "6px", "overflowX": "auto", "fontSize": "12px"}),
            ]),
        ],
        style={"fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
               "maxWidth": "1260px", "margin": "0 auto", "padding": "18px"},
    )

    @app.callback(
        Output("kpi-row", "children"),
        Output("obj-chart", "figure"),
        Output("risk-fairness-chart", "figure"),
        Output("applicant-table", "children"),
        Output("circuit-pre", "children"),
        Input("cohort-source", "value"),
        Input("cohort-idx", "value"),
        Input("primary-solver", "value"),
        Input("w-F", "value"),
        Input("w-N", "value"),
        Input("w-S", "value"),
        Input("budget-frac", "value"),
        Input("lam-F", "value"),
        Input("eta", "value"),
    )
    def update_view(source, cohort_idx, primary_solver, wF, wN, wS, b_frac, lam_F, eta):
        s_w = max(float(wF) + float(wN) + float(wS), 1e-6)
        weights = {"F": float(wF) / s_w, "N": float(wN) / s_w, "S": float(wS) / s_w}
        overrides = {
            "scores": {"weights": weights},
            "allocation": {"budget_fraction": float(b_frac), "lambda_F": float(lam_F), "eta": float(eta)},
            "solvers": {"qaoa": {"restarts": 2, "max_iter": 140, "shots": 1024},
                        "warm_start_qaoa": {"restarts": 2, "max_iter": 140, "shots": 1024},
                        "cvar_qaoa": {"restarts": 2, "max_iter": 140, "shots": 1024}},
        }
        csv_p = str(legacy_csv) if source == "v1_sample" and legacy_csv.exists() else None
        res = run_allocation(
            cfg_path=cfg_path,
            overrides=overrides,
            csv_path=csv_p,
            cohort_idx=int(cohort_idx or 0),
            primary_solver=primary_solver,
            solvers=FAST_UI_SOLVERS,
            include_qiskit_circuit=True,
            include_noisy_zne=False,
        )
        comp = res.comparison
        prow = comp[comp["solver"] == res.primary_solver].iloc[0]
        m = res.metadata

        kpis = [
            _kpi_card(html, "Approved / Total", f"{int(prow['approved'])} / {m['n_applicants']}",
                      f"Groups AR_A={prow['AR_A']:.2f}, AR_B={prow['AR_B']:.2f}"),
            _kpi_card(html, "Budget Used", f"{prow['lent_inr']:,.0f} INR",
                      f"{prow['budget_pct']:.1f}% of {m['budget_inr']:,.0f} INR"),
            _kpi_card(html, "95% Tail CVaR", f"{prow['cvar_inr']:,.0f} INR",
                      f"Exp. Return: {prow['expected_return_inr']:,.0f} INR"),
            _kpi_card(html, "Demographic Gap |DPD|", f"{prow['abs_dpd']:.3f}",
                      f"Signed d(x)={prow['dpd']:+.3f}"),
            _kpi_card(html, "QUBO Space", f"{m['total_qubits']} qubits",
                      f"N={m['n_applicants']} + K={m['slack_qubits']} slack ({m['social_kind']})"),
        ]

        comp_s = comp.sort_values("J1")
        fig_obj = go.Figure(data=[
            go.Bar(name="P1 Objective (J1)", x=comp_s["solver"], y=comp_s["J1"]),
            go.Bar(name="P2 Objective (J2)", x=comp_s["solver"], y=comp_s["J2"]),
        ])
        fig_obj.update_layout(barmode="group", title="Objective Comparison (Lower is Better)",
                              margin=dict(l=35, r=15, t=42, b=60), height=340)

        fig_rf = go.Figure(data=[
            go.Scatter(
                x=comp["abs_dpd"], y=comp["cvar_inr"],
                mode="markers+text", text=comp["solver"], textposition="top center",
                marker=dict(size=12, color=["#d62728" if s == res.primary_solver else "#1f77b4" for s in comp["solver"]]),
            )
        ])
        fig_rf.update_layout(title="Fairness (|DPD|) vs 95% CVaR (INR)",
                             xaxis_title="|DPD|", yaxis_title="CVaR_95 (INR)",
                             margin=dict(l=45, r=15, t=42, b=45), height=340)

        cols = [c for c in ("applicant_id", "group", "requested_amount", "w_units", "p_repay", "F", "N", "S", "U", "approved")
                if c in res.cohort.columns]
        table = _html_table(html, res.cohort, cols)
        circ_txt = res.circuit_artifacts["circuit_text"] if res.circuit_artifacts else "Circuit unavailable."
        return kpis, fig_obj, fig_rf, table, circ_txt

    return app

