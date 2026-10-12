"""Plotly figures. Every figure goes through `dark()` so the dark theme never depends on the browser."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .theme import (AMBER, BORDER, CARD, CYAN, EMERALD, FONT, INDIGO, MUTED, PANEL, RISK_COLORS, ROSE, TEXT)

GRID = "rgba(41,53,72,0.75)"
RATE_SCALE = [[0, INDIGO], [0.5, "#A855F7"], [1, ROSE]]
HEAT_SCALE = [[0, CARD], [0.35, "#312E81"], [0.7, INDIGO], [1, ROSE]]


def dark(fig: go.Figure, height: int = 340, legend: bool = True, margin: dict | None = None) -> go.Figure:
    """Apply the dark layout: transparent backgrounds, light text, subtle grid, styled tooltips."""
    fig.update_layout(
        template="plotly_dark", height=height, autosize=True,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=TEXT, size=12),
        margin=margin or dict(l=8, r=8, t=40 if legend else 12, b=8),
        showlegend=legend,
        legend=dict(font=dict(color=TEXT, size=12), bgcolor="rgba(0,0,0,0)", orientation="h",
                    yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=BORDER, font=dict(color=TEXT, family=FONT, size=12)),
        colorway=[INDIGO, CYAN, EMERALD, AMBER, ROSE],
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=MUTED, activecolor=CYAN),
    )
    axis = dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER, tickfont=dict(color=MUTED, size=11),
                title_font=dict(color=TEXT, size=12), automargin=True)
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def empty_fig(message: str = "No data to display", height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=MUTED, size=14), x=0.5, y=0.5,
                       xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return dark(fig, height=height, legend=False)


# --------------------------------------------------------------------------- churn overview
def churn_donut(df: pd.DataFrame, target: str) -> go.Figure:
    y = df[target].dropna()
    churned, kept = int((y == 1).sum()), int((y == 0).sum())
    fig = go.Figure(go.Pie(labels=["Retained", "Churned"], values=[kept, churned], hole=0.68, sort=False,
                           marker=dict(colors=[EMERALD, ROSE], line=dict(color=CARD, width=3)),
                           textinfo="none", hovertemplate="%{label}: %{value:,} customers (%{percent})<extra></extra>"))
    rate = churned / max(1, churned + kept)
    fig.add_annotation(text=f"<b>{rate:.1%}</b><br><span style='font-size:12px;color:{MUTED}'>churn rate</span>",
                       showarrow=False, font=dict(size=26, color=TEXT))
    return dark(fig, height=300)


def risk_bars(view: pd.DataFrame) -> go.Figure:
    counts = view["risk_category"].astype(str).value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
    fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=[RISK_COLORS[k] for k in counts.index],
                           text=[f"{int(v):,}" for v in counts.values], textposition="outside",
                           textfont=dict(color=TEXT), hovertemplate="%{x} risk: %{y:,} customers<extra></extra>"))
    fig.update_yaxes(title="Customers", range=[0, max(1, counts.max()) * 1.18])
    fig.update_layout(bargap=0.35)
    return dark(fig, height=300, legend=False)


def probability_histogram(proba: pd.Series, high: float, medium: float) -> go.Figure:
    p = proba.dropna().to_numpy()
    counts, edges = np.histogram(p, bins=30, range=(0, 1))
    centers = (edges[:-1] + edges[1:]) / 2
    colors = [ROSE if c >= high else AMBER if c >= medium else EMERALD for c in centers]
    fig = go.Figure(go.Bar(x=centers, y=counts, marker_color=colors, width=(edges[1] - edges[0]) * 0.92,
                           hovertemplate="Estimated probability ≈ %{x:.0%}: %{y} customers<extra></extra>"))
    for thr, col, label in ((medium, AMBER, "Medium"), (high, ROSE, "High")):
        fig.add_vline(x=thr, line=dict(color=col, dash="dot", width=1.5), annotation_text=label,
                      annotation_font=dict(color=col, size=11), annotation_position="top")
    fig.update_xaxes(title="Estimated churn probability", tickformat=".0%", range=[0, 1])
    fig.update_yaxes(title="Customers")
    return dark(fig, height=300, legend=False, margin=dict(l=8, r=8, t=28, b=8))


def churn_by_category(g: pd.DataFrame, col: str, overall: float | None) -> go.Figure:
    g = g.sort_values("churn_rate", ascending=False)
    fig = go.Figure(go.Bar(
        x=g[col].astype(str), y=g["churn_rate"], customdata=np.c_[g["customers"], g["churned"]],
        marker=dict(color=g["churn_rate"], colorscale=RATE_SCALE, cmin=0, cmax=max(0.01, g["churn_rate"].max())),
        text=[f"{v:.0%}" for v in g["churn_rate"]], textposition="outside", textfont=dict(color=TEXT),
        hovertemplate="<b>%{x}</b><br>Churn rate: %{y:.1%}<br>Customers: %{customdata[0]:,}<br>"
                      "Churned: %{customdata[1]:,}<extra></extra>"))
    if overall is not None:
        fig.add_hline(y=overall, line=dict(color=CYAN, dash="dash", width=1.5), annotation_text=f"Overall {overall:.0%}",
                      annotation_font=dict(color=CYAN, size=11), annotation_position="top right")
    fig.update_yaxes(title="Churn rate", tickformat=".0%", range=[0, min(1.05, g["churn_rate"].max() * 1.25 + 0.02)])
    fig.update_xaxes(title=None)
    fig.update_layout(bargap=0.35)
    return dark(fig, height=300, legend=False)


def churn_by_tenure(g: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=g["bin"].astype(str), y=g["customers"], name="Customers", marker_color="rgba(99,102,241,0.38)",
                         hovertemplate="%{x} months: %{y:,} customers<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=g["bin"].astype(str), y=g["churn_rate"], name="Churn rate", mode="lines+markers",
                             line=dict(color=ROSE, width=3), marker=dict(size=8, color=ROSE, line=dict(color=CARD, width=2)),
                             hovertemplate="%{x} months: %{y:.1%} churn<extra></extra>"), secondary_y=True)
    fig.update_yaxes(title="Customers", secondary_y=False)
    fig.update_yaxes(title="Churn rate", tickformat=".0%", secondary_y=True, showgrid=False, rangemode="tozero")
    fig.update_xaxes(title="Tenure (months)")
    return dark(fig, height=320)


def charges_vs_churn(df: pd.DataFrame, col: str, target: str) -> go.Figure:
    d = df[[col, target]].dropna()
    fig = go.Figure()
    for val, name, color in ((0, "Retained", EMERALD), (1, "Churned", ROSE)):
        s = d.loc[d[target] == val, col]
        fig.add_trace(go.Violin(y=s, name=name, line_color=color, fillcolor=color, opacity=0.55, box_visible=True,
                                meanline_visible=True, points=False, spanmode="hard",
                                hovertemplate=f"{name}: %{{y:,.1f}}<extra></extra>"))
    fig.update_yaxes(title=col)
    fig.update_layout(violingap=0.25)
    return dark(fig, height=320)


def numeric_hist(df: pd.DataFrame, col: str, target: str | None) -> go.Figure:
    fig = go.Figure()
    if target and target in df:
        for val, name, color in ((0, "Retained", EMERALD), (1, "Churned", ROSE)):
            fig.add_trace(go.Histogram(x=df.loc[df[target] == val, col], name=name, marker_color=color, opacity=0.65,
                                       nbinsx=30, hovertemplate=f"{name}<br>%{{x}}: %{{y}} customers<extra></extra>"))
        fig.update_layout(barmode="overlay")
    else:
        fig.add_trace(go.Histogram(x=df[col], name=col, marker_color=INDIGO, opacity=0.85, nbinsx=30))
    fig.update_xaxes(title=col)
    fig.update_yaxes(title="Customers")
    return dark(fig, height=320, legend=bool(target))


def scatter_churn(df: pd.DataFrame, x: str, y: str, target: str, max_points: int = 1500) -> go.Figure:
    d = df[[x, y, target]].dropna()
    if len(d) > max_points:
        d = d.sample(max_points, random_state=1)
    fig = go.Figure()
    for val, name, color in ((0, "Retained", EMERALD), (1, "Churned", ROSE)):
        s = d[d[target] == val]
        fig.add_trace(go.Scattergl(x=s[x], y=s[y], mode="markers", name=name,
                                   marker=dict(color=color, size=6, opacity=0.55),
                                   hovertemplate=f"{name}<br>{x}: %{{x:,.1f}}<br>{y}: %{{y:,.1f}}<extra></extra>"))
    fig.update_xaxes(title=x)
    fig.update_yaxes(title=y)
    return dark(fig, height=340)


def segment_heatmap(df: pd.DataFrame, row: str, col: str, target: str, min_n: int = 10) -> go.Figure:
    d = df[[row, col, target]].dropna()
    rate = d.pivot_table(index=row, columns=col, values=target, aggfunc="mean", observed=True)
    cnt = d.pivot_table(index=row, columns=col, values=target, aggfunc="size", observed=True)
    rate = rate.where(cnt >= min_n)
    text = rate.map(lambda v: "" if pd.isna(v) else f"{v:.0%}")
    fig = go.Figure(go.Heatmap(z=rate.to_numpy(), x=[str(c) for c in rate.columns], y=[str(r) for r in rate.index],
                               colorscale=HEAT_SCALE, zmin=0, zmax=max(0.01, float(np.nanmax(rate.to_numpy())) if rate.notna().to_numpy().any() else 1),
                               text=text.to_numpy(), texttemplate="%{text}", textfont=dict(color=TEXT, size=13),
                               customdata=cnt.reindex_like(rate).to_numpy(), xgap=3, ygap=3,
                               colorbar=dict(tickformat=".0%", tickfont=dict(color=MUTED), outlinewidth=0, thickness=10),
                               hovertemplate=f"{row}: %{{y}}<br>{col}: %{{x}}<br>Churn rate: %{{z:.1%}}"
                                             "<br>Customers: %{customdata:,.0f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return dark(fig, height=320, legend=False)


def segment_risk_bars(seg: pd.DataFrame, col: str) -> go.Figure:
    s = seg.sort_values("avg_predicted_risk", ascending=True)
    fig = go.Figure(go.Bar(y=s[col].astype(str), x=s["avg_predicted_risk"], orientation="h",
                           marker=dict(color=s["avg_predicted_risk"], colorscale=RATE_SCALE),
                           text=[f"{v:.0%}" for v in s["avg_predicted_risk"]], textposition="outside",
                           textfont=dict(color=TEXT), customdata=s["customers"],
                           hovertemplate="<b>%{y}</b><br>Avg estimated risk: %{x:.1%}<br>Customers: %{customdata:,}<extra></extra>"))
    fig.update_xaxes(title="Average estimated churn probability", tickformat=".0%",
                     range=[0, max(0.1, s["avg_predicted_risk"].max() * 1.2)])
    return dark(fig, height=max(240, 60 + 44 * len(s)), legend=False)


def missing_bars(missing: pd.DataFrame) -> go.Figure:
    m = missing[missing["missing"] > 0].sort_values("missing_pct")
    if m.empty:
        return empty_fig("No missing values detected", 240)
    fig = go.Figure(go.Bar(y=m["column"], x=m["missing_pct"], orientation="h", marker_color=AMBER,
                           text=[f"{v:.1f}%" for v in m["missing_pct"]], textposition="outside", textfont=dict(color=TEXT),
                           customdata=m["missing"], hovertemplate="<b>%{y}</b><br>%{x:.2f}% missing (%{customdata:,} rows)<extra></extra>"))
    fig.update_xaxes(title="% missing", range=[0, max(1.0, m["missing_pct"].max() * 1.3)])
    return dark(fig, height=max(220, 70 + 38 * len(m)), legend=False)


# --------------------------------------------------------------------------- customer + model
def risk_gauge(p: float, high: float, medium: float) -> go.Figure:
    color = ROSE if p >= high else AMBER if p >= medium else EMERALD
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number=dict(suffix="%", font=dict(color=TEXT, size=40), valueformat=".0f"),
        gauge=dict(axis=dict(range=[0, 100], tickcolor=MUTED, tickfont=dict(color=MUTED, size=11)),
                   bar=dict(color=color, thickness=0.28), bgcolor=PANEL, borderwidth=0,
                   steps=[dict(range=[0, medium * 100], color="rgba(16,185,129,0.18)"),
                          dict(range=[medium * 100, high * 100], color="rgba(245,158,11,0.18)"),
                          dict(range=[high * 100, 100], color="rgba(244,63,94,0.18)")])))
    return dark(fig, height=230, legend=False, margin=dict(l=24, r=24, t=20, b=0))


def factor_effects(effects: pd.DataFrame) -> go.Figure:
    e = effects.iloc[::-1]
    colors = [ROSE if v > 0 else EMERALD for v in e["effect_pp"]]
    labels = [f"{f} = {v if not isinstance(v, float) else f'{v:,.1f}'}" for f, v in zip(e["feature"], e["customer_value"])]
    fig = go.Figure(go.Bar(y=labels, x=e["effect_pp"], orientation="h", marker_color=colors,
                           text=[f"{v:+.1f} pp" for v in e["effect_pp"]], textposition="outside", textfont=dict(color=TEXT),
                           hovertemplate="%{y}<br>Effect on estimate: %{x:+.1f} percentage points<extra></extra>"))
    lim = max(1.0, float(e["effect_pp"].abs().max()) * 1.6)
    fig.update_xaxes(title="Change in estimated risk (percentage points)", range=[-lim, lim])
    return dark(fig, height=max(240, 50 + 46 * len(e)), legend=False)


def roc_curve_fig(curves: dict[str, dict], aucs: dict[str, float | None], selected: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                             line=dict(color=MUTED, dash="dash", width=1), hoverinfo="skip"))
    palette = {selected: INDIGO}
    for name, c in curves.items():
        col = palette.get(name, CYAN)
        a = aucs.get(name)
        fig.add_trace(go.Scatter(x=c["fpr"], y=c["tpr"], mode="lines", name=f"{name} (AUC {a:.3f})" if a is not None else name,
                                 line=dict(color=col, width=3 if name == selected else 2),
                                 hovertemplate="False positive rate %{x:.2f}<br>True positive rate %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1.02])
    return dark(fig, height=340)


def confusion_fig(cm: list[list[int]]) -> go.Figure:
    z = np.array(cm)
    labels = ["Retained", "Churned"]
    pct = z / np.clip(z.sum(axis=1, keepdims=True), 1, None)
    text = [[f"<b>{z[i][j]:,}</b><br>{pct[i][j]:.0%} of row" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(z=pct, x=[f"Predicted {l}" for l in labels], y=[f"Actual {l}" for l in labels],
                               colorscale=[[0, CARD], [1, INDIGO]], zmin=0, zmax=1, showscale=False, xgap=4, ygap=4,
                               text=text, texttemplate="%{text}", textfont=dict(color=TEXT, size=14), hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    return dark(fig, height=320, legend=False)


def importance_bars(imp: pd.DataFrame, title_x: str = "Drop in ROC-AUC when shuffled", top: int = 10) -> go.Figure:
    d = imp.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(y=d["feature"], x=d["importance"], orientation="h", marker_color=CYAN,
                           error_x=dict(type="data", array=d["std"], color=MUTED, thickness=1.2),
                           hovertemplate="<b>%{y}</b><br>AUC drop: %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title=title_x)
    return dark(fig, height=max(260, 60 + 34 * len(d)), legend=False)


def coefficient_bars(coef: pd.DataFrame, top: int = 12) -> go.Figure:
    d = coef.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(y=d["feature"], x=d["coefficient"], orientation="h",
                           marker_color=[ROSE if v > 0 else EMERALD for v in d["coefficient"]],
                           customdata=d["odds_ratio"],
                           hovertemplate="<b>%{y}</b><br>Coefficient %{x:+.3f}<br>Odds ratio %{customdata:.2f}<extra></extra>"))
    fig.update_xaxes(title="Logistic regression coefficient (positive = higher estimated risk)")
    return dark(fig, height=max(280, 60 + 34 * len(d)), legend=False)


def model_comparison(comp: pd.DataFrame) -> go.Figure:
    metrics = [("CV ROC-AUC", INDIGO), ("Test ROC-AUC", CYAN), ("Test F1", EMERALD)]
    fig = go.Figure()
    for m, c in metrics:
        fig.add_trace(go.Bar(x=comp["Model"], y=comp[m], name=m, marker_color=c, text=[f"{v:.2f}" for v in comp[m]],
                             textposition="outside", textfont=dict(color=TEXT)))
    fig.update_yaxes(range=[0, 1.05], title="Score")
    fig.update_layout(barmode="group", bargap=0.3)
    return dark(fig, height=320)
