"""
Multi-Strategy Research Dashboard
===================================
50+ metrics across strategies. Dark theme, Dash + Bootstrap.
Call launch(prices, results, summary) or run standalone.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Palette ───────────────────────────────────────────────────────────────────
BG     = "#0d1117"
CARD   = "#161b22"
BORDER = "#30363d"
GREEN  = "#3fb950"
RED    = "#f85149"
BLUE   = "#58a6ff"
AMBER  = "#d29922"
GREY   = "#8b949e"
WHITE  = "#e6edf3"
PURPLE = "#bc8cff"
FONT   = "Inter, sans-serif"

BASE_LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=CARD,
    font=dict(family=FONT, color=WHITE, size=11),
    margin=dict(l=55, r=20, t=40, b=40),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, font=dict(size=10)),
)

STRATEGY_COLORS = px.colors.qualitative.Plotly


def kpi(label, value, color=WHITE, sub=""):
    return dbc.Col(
        dbc.Card(dbc.CardBody([
            html.P(label, style={"fontSize": "0.68rem", "color": GREY,
                                  "textTransform": "uppercase", "letterSpacing": "0.06em",
                                  "marginBottom": 2}),
            html.H4(value, style={"color": color, "fontWeight": 700, "margin": 0}),
            html.P(sub, style={"fontSize": "0.7rem", "color": GREY, "margin": 0}) if sub else None,
        ]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": 8}),
        xs=6, sm=4, md=3, lg=2, className="mb-2",
    )


def empty_fig(msg="No data"):
    fig = go.Figure(layout={**BASE_LAYOUT})
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False, font=dict(color=GREY, size=13))
    return fig


# ── Global state (populated by launch()) ─────────────────────────────────────
_PRICES   = None
_RESULTS  = None
_SUMMARY  = None


def launch(prices: pd.DataFrame, results: dict, summary: pd.DataFrame,
           port: int = 8052):
    global _PRICES, _RESULTS, _SUMMARY
    _PRICES  = prices
    _RESULTS = results
    _SUMMARY = summary
    app = _build_app()
    import os
    app.run(host="0.0.0.0", port=port, debug=False)


def _build_app():
    app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],
               title="Quant Research Dashboard", suppress_callback_exceptions=True)

    strat_names = list(_RESULTS.keys()) if _RESULTS else []

    app.layout = dbc.Container(fluid=True,
        style={"backgroundColor": BG, "minHeight": "100vh", "padding": "24px"},
        children=[
            # Header
            dbc.Row([
                dbc.Col(html.H3("📊 Quantitative Research Framework",
                                style={"color": WHITE, "fontWeight": 700})),
                dbc.Col(html.Small(
                    f"{len(strat_names)} strategies  |  "
                    f"{len(_PRICES) if _PRICES is not None else 0} trading days",
                    style={"color": GREY}),
                    width="auto", className="d-flex align-items-center"),
            ], className="mb-4"),

            # Strategy selector
            dbc.Row([
                dbc.Col([
                    html.Label("Select Strategies", style={"color": GREY, "fontSize": "0.8rem"}),
                    dcc.Dropdown(
                        id="strat-select",
                        options=[{"label": s, "value": s} for s in strat_names],
                        value=strat_names[:6] if strat_names else [],
                        multi=True,
                        style={"backgroundColor": CARD, "color": WHITE},
                    ),
                ]),
            ], className="mb-3"),

            # KPI strip (best strategy)
            dbc.Row(id="kpi-row", className="mb-3"),

            # Row 1: Equity curves | Drawdowns
            dbc.Row([
                dbc.Col(dcc.Graph(id="fig-equity",   config={"displayModeBar": False}), md=8),
                dbc.Col(dcc.Graph(id="fig-dd",       config={"displayModeBar": False}), md=4),
            ], className="mb-3"),

            # Row 2: Strategy ranking | Rolling Sharpe
            dbc.Row([
                dbc.Col(dcc.Graph(id="fig-ranking",  config={"displayModeBar": False}), md=5),
                dbc.Col(dcc.Graph(id="fig-sharpe",   config={"displayModeBar": False}), md=7),
            ], className="mb-3"),

            # Row 3: Return distribution | Monthly heatmap of best strategy
            dbc.Row([
                dbc.Col(dcc.Graph(id="fig-dist",     config={"displayModeBar": False}), md=5),
                dbc.Col(dcc.Graph(id="fig-heat",     config={"displayModeBar": False}), md=7),
            ], className="mb-3"),

            # Row 4: Correlation heatmap | Trade scatter
            dbc.Row([
                dbc.Col(dcc.Graph(id="fig-corr",     config={"displayModeBar": False}), md=5),
                dbc.Col(dcc.Graph(id="fig-trades",   config={"displayModeBar": False}), md=7),
            ], className="mb-3"),

            # Metrics table
            dbc.Row([dbc.Col(html.Div(id="metrics-table"))], className="mb-3"),
        ],
    )

    @app.callback(
        Output("kpi-row",      "children"),
        Output("fig-equity",   "figure"),
        Output("fig-dd",       "figure"),
        Output("fig-ranking",  "figure"),
        Output("fig-sharpe",   "figure"),
        Output("fig-dist",     "figure"),
        Output("fig-heat",     "figure"),
        Output("fig-corr",     "figure"),
        Output("fig-trades",   "figure"),
        Output("metrics-table","children"),
        Input("strat-select",  "value"),
    )
    def update(selected):
        if not selected or _RESULTS is None:
            ef = empty_fig()
            return [], ef, ef, ef, ef, ef, ef, ef, ef, html.P("No data")

        sel_results = {k: _RESULTS[k] for k in selected if k in _RESULTS}

        # Best strategy by Sharpe
        best_name = max(sel_results, key=lambda k: sel_results[k].metrics["Sharpe"])
        best      = sel_results[best_name]
        bm        = best.metrics

        kpis = dbc.Row([
            kpi("Best Strategy", best_name[:22], BLUE),
            kpi("CAGR",          f"{bm['CAGR']:+.1f}%",    GREEN if bm["CAGR"] > 0 else RED),
            kpi("Sharpe",        f"{bm['Sharpe']:.2f}",     GREEN if bm["Sharpe"] > 0.5 else AMBER),
            kpi("Max DD",        f"{bm['Max Drawdown']:.1f}%", RED),
            kpi("Sortino",       f"{bm['Sortino']:.2f}",    WHITE),
            kpi("Calmar",        f"{bm['Calmar']:.2f}",     GREEN if bm["Calmar"] > 0.5 else AMBER),
            kpi("Win Rate",      f"{bm['Win Rate']:.0f}%",  GREEN if bm["Win Rate"] > 50 else AMBER),
            kpi("N Trades",      str(bm["N Trades"]),       WHITE),
        ])

        colors = {name: STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
                  for i, name in enumerate(selected)}

        # ── Equity curves ─────────────────────────────────────────────────
        fig_eq = go.Figure(layout={**BASE_LAYOUT, "title": {"text": "Equity Curves"}})
        for name, res in sel_results.items():
            eq = res.equity / res.equity.iloc[0]
            fig_eq.add_trace(go.Scatter(x=eq.index, y=(eq - 1) * 100,
                                         name=name, line=dict(color=colors[name], width=1.8)))
        fig_eq.add_hline(y=0, line_color=BORDER, line_width=1)
        fig_eq.update_yaxes(ticksuffix="%")

        # ── Drawdowns ─────────────────────────────────────────────────────
        fig_dd = go.Figure(layout={**BASE_LAYOUT, "title": {"text": "Drawdowns"}})
        for name, res in sel_results.items():
            dd = (res.equity / res.equity.cummax() - 1) * 100
            fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values,
                                         fill="tozeroy", name=name,
                                         line=dict(color=colors[name], width=1),
                                         fillcolor=f"{colors[name]}22"))
        fig_dd.update_yaxes(ticksuffix="%")

        # ── Ranking bar chart ──────────────────────────────────────────────
        if _SUMMARY is not None:
            sub = _SUMMARY[_SUMMARY["Strategy"].isin(selected)].sort_values("Sharpe")
            fig_rank = go.Figure(layout={**BASE_LAYOUT,
                                          "title": {"text": "Strategy Ranking — Sharpe"}})
            fig_rank.add_trace(go.Bar(
                y=sub["Strategy"],
                x=sub["Sharpe"],
                orientation="h",
                marker_color=[GREEN if v > 0 else RED for v in sub["Sharpe"]],
                opacity=0.85,
            ))
            fig_rank.add_vline(x=0, line_color=GREY, line_width=1)
        else:
            fig_rank = empty_fig()

        # ── Rolling Sharpe (63d) ──────────────────────────────────────────
        fig_sharpe = go.Figure(layout={**BASE_LAYOUT,
                                        "title": {"text": "Rolling Sharpe (63d)"}})
        for name, res in sel_results.items():
            r    = res.returns
            rs   = r.rolling(63).mean() / r.rolling(63).std() * np.sqrt(252)
            fig_sharpe.add_trace(go.Scatter(x=rs.index, y=rs.values,
                                             name=name, line=dict(color=colors[name], width=1.5)))
        fig_sharpe.add_hline(y=0,  line_color=GREY,  line_width=1)
        fig_sharpe.add_hline(y=1,  line_color=GREEN, line_dash="dot", line_width=1)
        fig_sharpe.add_hline(y=-1, line_color=RED,   line_dash="dot", line_width=1)

        # ── Return distribution ───────────────────────────────────────────
        fig_dist = go.Figure(layout={**BASE_LAYOUT,
                                      "title": {"text": "Daily Return Distribution"}})
        for name, res in sel_results.items():
            fig_dist.add_trace(go.Histogram(x=res.returns.values * 100,
                                             nbinsx=60, name=name,
                                             marker_color=colors[name], opacity=0.5))
        fig_dist.update_layout(barmode="overlay")
        fig_dist.add_vline(x=0, line_color=GREY, line_width=1)
        fig_dist.update_xaxes(title="Daily Return (%)", ticksuffix="%")

        # ── Monthly heatmap (best strategy) ──────────────────────────────
        fig_heat = empty_fig("Monthly Returns")
        best_eq = best.equity
        if not best_eq.empty:
            monthly = best_eq.resample("ME").last().pct_change().dropna() * 100
            if not monthly.empty:
                df_m   = pd.DataFrame({
                    "Y": monthly.index.year,
                    "M": monthly.index.strftime("%b"),
                    "R": monthly.values,
                })
                months = ["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"]
                piv = df_m.pivot_table(index="Y", columns="M", values="R")
                piv = piv.reindex(columns=[m for m in months if m in piv.columns])
                z   = piv.values
                txt = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in z]
                fig_heat = go.Figure(
                    go.Heatmap(z=z, x=piv.columns.tolist(), y=piv.index.tolist(),
                               colorscale=[[0, RED], [0.5, CARD], [1, GREEN]],
                               zmid=0, text=txt, texttemplate="%{text}",
                               colorbar=dict(title="%")),
                    layout={**BASE_LAYOUT,
                            "title": {"text": f"Monthly Returns — {best_name}"}},
                )

        # ── Correlation heatmap ────────────────────────────────────────────
        fig_corr = empty_fig("Correlation")
        if len(sel_results) > 1:
            ret_df = pd.concat({n: r.returns for n, r in sel_results.items()}, axis=1).dropna()
            corr   = ret_df.corr()
            fig_corr = go.Figure(
                go.Heatmap(z=corr.values,
                           x=corr.columns.tolist(), y=corr.index.tolist(),
                           colorscale=[[0, RED], [0.5, CARD], [1, GREEN]],
                           zmid=0, text=np.round(corr.values, 2),
                           texttemplate="%{text}",
                           colorbar=dict(title="ρ")),
                layout={**BASE_LAYOUT, "title": {"text": "Strategy Return Correlations"}},
            )

        # ── Trade scatter ─────────────────────────────────────────────────
        fig_trades = go.Figure(layout={**BASE_LAYOUT,
                                        "title": {"text": "Trade Duration vs PnL"}})
        for name, res in sel_results.items():
            if not res.trades.empty and "pnl" in res.trades.columns:
                t  = res.trades.copy()
                if "entry_date" in t.columns and "exit_date" in t.columns:
                    t["entry_date"] = pd.to_datetime(t["entry_date"])
                    t["exit_date"]  = pd.to_datetime(t["exit_date"])
                    t["duration"]   = (t["exit_date"] - t["entry_date"]).dt.days
                    fig_trades.add_trace(go.Scatter(
                        x=t["duration"], y=t["pnl"],
                        mode="markers", name=name,
                        marker=dict(color=colors[name], size=4, opacity=0.5),
                    ))
        fig_trades.add_hline(y=0, line_color=BORDER, line_width=1)
        fig_trades.update_xaxes(title="Trade Duration (days)")
        fig_trades.update_yaxes(title="PnL (%)", ticksuffix="%")

        # ── Metrics table ─────────────────────────────────────────────────
        if _SUMMARY is not None:
            sub_tbl = _SUMMARY[_SUMMARY["Strategy"].isin(selected)].sort_values("Sharpe",
                                                                                 ascending=False)
            rows = []
            for _, row in sub_tbl.iterrows():
                sharpe_c = GREEN if row["Sharpe"] > 0.5 else (AMBER if row["Sharpe"] > 0 else RED)
                cagr_c   = GREEN if row["CAGR (%)"] > 0 else RED
                rows.append(html.Tr([
                    html.Td(html.Strong(row["Strategy"], style={"color": BLUE}),
                            style={"padding": "5px 10px"}),
                    html.Td(f"{row['CAGR (%)']:+.1f}%",   style={"color": cagr_c,   "padding": "5px 10px"}),
                    html.Td(f"{row['Sharpe']:+.2f}",        style={"color": sharpe_c, "padding": "5px 10px"}),
                    html.Td(f"{row['Sortino']:+.2f}",       style={"padding": "5px 10px"}),
                    html.Td(f"{row['MaxDD (%)']:+.1f}%",    style={"color": RED,       "padding": "5px 10px"}),
                    html.Td(f"{row['Calmar']:+.2f}",        style={"padding": "5px 10px"}),
                    html.Td(f"{row['WinRate (%)']:.0f}%",  style={"padding": "5px 10px"}),
                    html.Td(str(int(row["Trades"])),        style={"padding": "5px 10px"}),
                ], style={"borderBottom": f"1px solid {BORDER}"}))

            hdr = ["Strategy", "CAGR", "Sharpe", "Sortino", "MaxDD", "Calmar",
                   "WinRate", "Trades"]
            table = dbc.Card(
                dbc.CardBody(
                    dbc.Table([
                        html.Thead(html.Tr([
                            html.Th(h, style={"color": GREY, "fontSize": "0.72rem",
                                              "padding": "5px 10px"}) for h in hdr
                        ])),
                        html.Tbody(rows),
                    ], bordered=False, hover=True, size="sm",
                       style={"color": WHITE, "fontSize": "0.82rem"}),
                    style={"padding": "0 8px 8px 8px"},
                ),
                style={"background": CARD, "border": f"1px solid {BORDER}"},
            )
            metrics_div = html.Div([
                html.H6("Full Metrics Table", style={"color": WHITE, "fontWeight": 700,
                                                       "marginBottom": 8}),
                table,
            ])
        else:
            metrics_div = html.Div()

        return (kpis, fig_eq, fig_dd, fig_rank, fig_sharpe,
                fig_dist, fig_heat, fig_corr, fig_trades, metrics_div)

    return app


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, warnings
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("  Quant Research Dashboard — loading data…")
    print("=" * 60)

    # Run simulation inline
    from run_simulation import load_prices, run_all, print_summary, BacktestConfig

    tickers = ["SPY", "QQQ", "GLD", "TLT", "EEM"]
    prices  = load_prices(tickers, years=5)
    cfg     = BacktestConfig(initial_capital=100_000, commission_bps=5)
    results = run_all(prices, cfg)
    summary = print_summary(results)

    print(f"\n  Dashboard → http://127.0.0.1:8052\n")
    launch(prices, results, summary, port=8052)
