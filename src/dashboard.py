"""Streamlit pages. Each page reads the shared AppContext prepared in app.py."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st

from . import charts as C
from . import config
from .ai_assistant import SUGGESTED_QUESTIONS, answer_question, build_facts, llm_config
from .analysis import (bin_numeric, build_view, export_scored_csv, generate_recommendations, group_churn,
                       numeric_summary, overall_churn_rate, revenue_estimate, segment_summary, tenure_bins)
from .database import list_runs, save_run
from .model import explain_customer
from .report import build_report_markdown
from .theme import (set_active_nav, AMBER, CYAN, EMERALD, INDIGO, ROSE, app_header, callout, chart_card, empty_state, esc, html_table,
                    kpi_card, kpi_row, md, page_heading, panel, pill, risk_pill, style_dataframe)
from .workspace import Workspace


@dataclass
class AppContext:
    ws: Workspace
    high: float
    medium: float
    sig: str
    view: pd.DataFrame = field(init=False)
    revenue: dict | None = field(init=False)
    recs: list = field(init=False)

    def __post_init__(self):
        self.view = build_view(self.ws.clean, self.ws.result, self.high, self.medium)
        self.revenue = revenue_estimate(self.view, self.ws.schema.roles.get("monthly"))
        self.recs = generate_recommendations(self.view, self.ws.schema, self.ws.result)

    @property
    def target(self) -> str | None:
        return self.ws.schema.target

    @property
    def roles(self) -> dict:
        return self.ws.schema.roles


def get_ctx() -> AppContext:
    return st.session_state["ctx"]


# --------------------------------------------------------------------------- formatting
def fmt_money(v: float) -> str:
    if v >= 1e6:
        return f"${v / 1e6:.2f}M"
    if v >= 1e4:
        return f"${v / 1e3:.1f}K"
    return f"${v:,.0f}"


def fmt_val(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}"
    return str(v)


def header(ctx: AppContext, active: str = "overview") -> None:
    set_active_nav(active)
    ws, sch = ctx.ws, ctx.ws.schema
    pills = [pill(ws.name, "info", "database"), pill(f"{len(ws.clean):,} rows", "muted", "columns")]
    pills.append(pill(f"Label: {sch.target}", "pos", "check") if sch.has_labels else pill("No churn labels", "warn", "alert"))
    if ws.has_model:
        pills.append(pill(f"Model: {ws.result.selected_name}", "brand", "activity"))
    app_header("Customer Churn & Retention Predictor",
               "Estimate churn risk, see what is associated with it, and plan retention experiments.", pills)


def cat_options(ctx: AppContext, max_unique: int = 12) -> list[str]:
    return [c for c in ctx.ws.schema.categorical_features if ctx.view[c].nunique(dropna=True) <= max_unique]


def primary_category(ctx: AppContext) -> str | None:
    opts = cat_options(ctx)
    c = ctx.roles.get("contract")
    return c if c in opts else (opts[0] if opts else None)


def no_model_state(ctx: AppContext, what: str) -> None:
    reason = ctx.ws.training_error or "No churn label column was found."
    empty_state(f"{what} needs a trained model",
                f"{reason} Supervised training and risk scores are unavailable, and nothing has been estimated. "
                "Upload data that includes a churn column (Yes/No or 1/0), or choose the label column in the sidebar.",
                icon="alert")


# --------------------------------------------------------------------------- filters
def filter_view(ctx: AppContext, prefix: str, include_risk: bool = False) -> pd.DataFrame:
    view, sch = ctx.view, ctx.ws.schema
    cats: list[str] = []
    for r in ("contract", "payment"):
        c = sch.roles.get(r)
        if c in sch.categorical_features and c not in cats:
            cats.append(c)
    for c in cat_options(ctx):
        if len(cats) >= 3:
            break
        if c not in cats:
            cats.append(c)
    nums = [sch.roles[r] for r in ("tenure", "monthly") if sch.roles.get(r) in sch.numeric_features]
    for c in sch.numeric_features:
        if len(nums) >= 2:
            break
        if c not in nums:
            nums.append(c)
    items = [("risk", "risk_category")] if include_risk and "risk_category" in view else []
    items += [("cat", c) for c in cats] + [("num", c) for c in nums[:2]]

    mask = pd.Series(True, index=view.index)
    with panel(f"filters-{prefix}"):
        md('<div class="card-head"><div class="card-title">Filters</div>'
           '<div class="card-sub">Charts and tables on this page update with your selection.</div></div>')
        for i in range(0, len(items), 3):
            cols = st.columns(3)
            for col, (kind, name) in zip(cols, items[i:i + 3]):
                with col:
                    key = f"{prefix}-{ctx.sig}-{name}"
                    if kind == "risk":
                        chosen = st.multiselect("Risk level", ["Low", "Medium", "High"], default=["Low", "Medium", "High"], key=key)
                        mask &= view["risk_category"].astype(str).isin(chosen)
                    elif kind == "cat":
                        options = sorted(view[name].dropna().astype(str).unique())
                        chosen = st.multiselect(name, options, default=options, key=key)
                        if len(chosen) != len(options):
                            mask &= view[name].astype(str).isin(chosen)
                    else:
                        s = pd.to_numeric(view[name], errors="coerce")
                        lo, hi = float(np.floor(s.min())), float(np.ceil(s.max()))
                        if lo < hi:
                            sel = st.slider(name, lo, hi, (lo, hi), key=key)
                            if sel != (lo, hi):
                                mask &= s.between(sel[0], sel[1])
    out = view[mask]
    st.caption(f"Showing {len(out):,} of {len(view):,} customers")
    return out


# --------------------------------------------------------------------------- Overview
def overview_page() -> None:
    ctx = get_ctx()
    header(ctx, "overview")
    page_heading("Overview", "A snapshot of customers, churn and estimated risk.")
    ws, view, sch, t = ctx.ws, ctx.view, ctx.ws.schema, ctx.target

    if not sch.has_labels:
        callout(ctx.ws.training_error or "No churn column was found.", "warn", "Descriptive mode.")
        callout("Without churn labels the app cannot train a model or estimate risk, so no predictions are shown. "
                "The charts below describe your customers only.", "info")
        q = ws.quality
        num = sch.numeric_features[:1]
        kpi_row([kpi_card("Customers", f"{len(view):,}", None, "muted", "users"),
                 kpi_card("Columns", f"{view.shape[1]}", f"{len(sch.features)} usable as features", "info", "columns", CYAN),
                 kpi_card("Missing values", f"{q['missing_pct']}%", f"{q['missing_cells']:,} cells", "warn", "alert", AMBER),
                 kpi_card("Data-quality score", f"{q['quality_score']}/100", "Heuristic, see Data Explorer", "pos", "check", EMERALD)])
        c1, c2 = st.columns(2)
        with c1:
            if num:
                chart_card(f"Distribution of {num[0]}", C.numeric_hist(view, num[0], None), key="ov-desc-num")
        with c2:
            cat = primary_category(ctx)
            if cat:
                vc = view[cat].value_counts().reset_index()
                vc.columns = [cat, "customers"]
                import plotly.graph_objects as go
                fig = go.Figure(go.Bar(x=vc[cat].astype(str), y=vc["customers"], marker_color=INDIGO))
                chart_card(f"Customers by {cat}", C.dark(fig, 320, False), key="ov-desc-cat")
        return

    overall = overall_churn_rate(view, t)
    labelled = int(view[t].notna().sum())
    if ws.has_model:
        v = view["churn_probability"]
        high_n = int((view["risk_category"].astype(str) == "High").sum())
        rev = ctx.revenue
        kpi_row([
            kpi_card("Total customers", f"{len(view):,}", f"{labelled:,} with churn labels", "muted", "users"),
            kpi_card("Actual churn rate", f"{overall:.1%}", f"Avg. estimated risk {v.mean():.1%}", "info", "trend", ROSE),
            kpi_card("Predicted high-risk customers", f"{high_n:,}", f"{high_n / len(view):.1%} of customers", "neg", "alert", ROSE),
            kpi_card("Est. monthly revenue at risk", fmt_money(rev["high_risk_monthly_revenue"]) if rev else "n/a",
                     f"{rev['high_risk_share']:.1%} of monthly charges (estimate)" if rev else "No monthly charge column found",
                     "warn", "dollar", AMBER),
        ])
    else:
        callout(ws.training_error or "Model unavailable.", "warn", "Model not trained.")
        kpi_row([kpi_card("Total customers", f"{len(view):,}", None, "muted", "users"),
                 kpi_card("Actual churn rate", f"{overall:.1%}", f"{labelled:,} labelled", "info", "trend", ROSE),
                 kpi_card("Churned customers", f"{int((view[t] == 1).sum()):,}", None, "neg", "alert", ROSE),
                 kpi_card("Retained customers", f"{int((view[t] == 0).sum()):,}", None, "pos", "check", EMERALD)])
    st.write("")

    cat = primary_category(ctx)
    c1, c2, c3 = st.columns(3)
    with c1:
        chart_card("Churn distribution", C.churn_donut(view, t), "Share of labelled customers", key="ov-donut")
    with c2:
        if ws.has_model:
            chart_card("Churn risk distribution", C.risk_bars(view), "Estimated risk band per customer", key="ov-risk")
        else:
            empty_state("Risk bands unavailable", "A trained model is needed to estimate risk.", "alert")
    with c3:
        if cat:
            chart_card(f"Churn by {cat}", C.churn_by_category(group_churn(view, cat, t), cat, overall),
                       "Observed churn rate, dashed line = overall", key="ov-cat")

    c1, c2 = st.columns([3, 2])
    tcol, mcol = ctx.roles.get("tenure"), ctx.roles.get("monthly")
    with c1:
        if tcol in sch.numeric_features:
            tb = view.assign(bin=tenure_bins(view[tcol]))
            g = group_churn(tb, "bin", t)
            chart_card("Churn by tenure", C.churn_by_tenure(g), "Customers (bars) and observed churn rate (line)", key="ov-tenure")
    with c2:
        if mcol in sch.numeric_features:
            chart_card(f"{mcol} vs churn", C.charges_vs_churn(view, mcol, t), "Distribution by outcome", key="ov-charges")

    c1, c2 = st.columns([3, 2])
    with c1:
        with panel("ov-watchlist"):
            md('<div class="card-head"><div class="card-title">Highest-risk customers</div>'
               '<div class="card-sub">Top 10 by estimated probability. Estimates, not certainties.</div></div>')
            if ws.has_model:
                html_table(watchlist(ctx, view.sort_values("churn_probability", ascending=False).head(10)),
                           risk_cols=("Risk",), prob_cols=("Probability",))
            else:
                empty_state("No risk scores", "Train a model to see a watchlist.", "alert")
    with c2:
        with panel("ov-findings"):
            md('<div class="card-head"><div class="card-title">Key findings</div>'
               '<div class="card-sub">Associations in the observed data</div></div>')
            md("".join(f'<div class="finding"><b>{esc(f["title"])}</b><br><span>{esc(f["detail"])}</span></div>'
                       for f in ws.findings[:6]))


def watchlist(ctx: AppContext, rows: pd.DataFrame) -> pd.DataFrame:
    sch = ctx.ws.schema
    out = pd.DataFrame(index=rows.index)
    out["Customer"] = rows[sch.id_col] if sch.id_col else rows.index
    out["Risk"] = rows["risk_category"].astype(str)
    out["Probability"] = rows["churn_probability"]
    for r, label in (("tenure", "Tenure"), ("contract", "Contract"), ("monthly", "Monthly charge"), ("support", "Support calls")):
        c = ctx.roles.get(r)
        if c in rows:
            out[label] = rows[c].map(fmt_val)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- Customer Analytics
def analytics_page() -> None:
    ctx = get_ctx()
    header(ctx, "customer-analytics")
    page_heading("Customer Analytics", "Compare segments and explore how customer attributes relate to churn.")
    sch, t = ctx.ws.schema, ctx.target
    view = filter_view(ctx, "an")
    if view.empty:
        empty_state("No customers match these filters", "Widen the filters above to see charts.")
        return
    d = view.copy()
    tcol, mcol = ctx.roles.get("tenure"), ctx.roles.get("monthly")
    segs = {c: c for c in cat_options(ctx)}
    if tcol in sch.numeric_features:
        d["Tenure band"] = tenure_bins(d[tcol]).astype(str).where(d[tcol].notna())
        segs["Tenure band"] = "Tenure band"
    if mcol in sch.numeric_features:
        d["Monthly charge band"] = bin_numeric(d[mcol])
        segs["Monthly charge band"] = "Monthly charge band"
    if not segs:
        empty_state("No segment columns", "No categorical columns with a manageable number of values were found.")
        return

    c1, c2 = st.columns([2, 3])
    with c1:
        seg = st.selectbox("Compare customers by", list(segs), index=list(segs).index(primary_category(ctx)) if primary_category(ctx) in segs else 0, key=f"seg-{ctx.sig}")
    st.write("")
    if t:
        g = group_churn(d, seg, t)
        if seg in ("Tenure band", "Monthly charge band"):
            order = list(dict.fromkeys(d[seg].dropna().astype(str)))
            g[seg] = g[seg].astype(str)
            g = g.set_index(seg).reindex(sorted(g[seg], key=lambda s: order.index(s) if s in order else 0)).reset_index()
        a, b = st.columns([3, 2])
        with a:
            chart_card(f"Churn rate by {seg}", C.churn_by_category(g, seg, overall_churn_rate(d, t)),
                       "Observed churn; hover for customer counts", key="an-seg")
        with b:
            with panel("an-segtable"):
                md(f'<div class="card-head"><div class="card-title">Segment comparison</div>'
                   f'<div class="card-sub">By {esc(seg)}</div></div>')
                s = segment_summary(d, seg, t, mcol)
                disp = pd.DataFrame({seg: s[seg].astype(str), "Customers": s["customers"],
                                     "Churn rate": s["churn_rate"].map(lambda v: f"{v:.1%}")})
                if "avg_predicted_risk" in s:
                    disp["Avg est. risk"] = s["avg_predicted_risk"].map(lambda v: f"{v:.1%}")
                    disp["High-risk"] = s["high_risk_customers"]
                if "avg_monthly_charge" in s:
                    disp["Avg charge"] = s["avg_monthly_charge"].map(lambda v: f"{v:,.2f}")
                html_table(disp)
    else:
        vc = d[seg].value_counts().reset_index()
        vc.columns = [seg, "customers"]
        import plotly.graph_objects as go
        chart_card(f"Customers by {seg}", C.dark(go.Figure(go.Bar(x=vc[seg].astype(str), y=vc["customers"], marker_color=INDIGO)), 320, False), key="an-seg-desc")

    if t:
        opts = list(segs)
        a, b = st.columns(2)
        with a:
            r_sel = st.selectbox("Heatmap rows", opts, index=0, key=f"hm-r-{ctx.sig}")
        with b:
            c_sel = st.selectbox("Heatmap columns", opts, index=min(1, len(opts) - 1), key=f"hm-c-{ctx.sig}")
        a, b = st.columns(2)
        with a:
            if r_sel != c_sel:
                chart_card(f"Churn rate: {r_sel} × {c_sel}", C.segment_heatmap(d, r_sel, c_sel, t),
                           "Cells with fewer than 10 customers are blank", key="an-heat")
            else:
                empty_state("Pick two different fields", "Choose different row and column fields for the heatmap.")
        with b:
            if tcol in sch.numeric_features and mcol in sch.numeric_features:
                chart_card(f"{tcol} vs {mcol}", C.scatter_churn(d, tcol, mcol, t), "Each dot is a customer", key="an-scatter")
            elif len(sch.numeric_features) >= 2:
                x, y = sch.numeric_features[:2]
                chart_card(f"{x} vs {y}", C.scatter_churn(d, x, y, t), "Each dot is a customer", key="an-scatter")

    if sch.numeric_features:
        col = st.selectbox("Numeric distribution", sch.numeric_features,
                           index=sch.numeric_features.index(mcol) if mcol in sch.numeric_features else 0, key=f"num-{ctx.sig}")
        a, b = st.columns([3, 2])
        with a:
            chart_card(f"Distribution of {col}", C.numeric_hist(d, col, t), "Churned vs retained" if t else "", key="an-hist")
        with b:
            with panel("an-numsummary"):
                md('<div class="card-head"><div class="card-title">Numeric summary</div>'
                   '<div class="card-sub">Filtered customers</div></div>')
                ns = numeric_summary(d, sch.numeric_features).round(2).reset_index().rename(columns={"index": "column"})
                html_table(ns[["column", "mean", "median", "missing_%"]])


# --------------------------------------------------------------------------- Churn Predictions
def predictions_page() -> None:
    ctx = get_ctx()
    header(ctx, "churn-predictions")
    page_heading("Churn Predictions", "Estimated churn probability for every customer.")
    ws = ctx.ws
    if not ws.has_model:
        no_model_state(ctx, "Predictions")
        return
    callout("Scores are statistical estimates, not certainties. Training-portion customers are scored with "
            "cross-validated predictions so no customer is scored by a model that saw their own label.", "info", "Read as estimates.")
    view_all, sch = ctx.view, ws.schema
    counts = view_all["risk_category"].astype(str).value_counts()
    n = len(view_all)
    rev = ctx.revenue
    kpi_row([
        kpi_card("High risk", f"{counts.get('High', 0):,}", f"{counts.get('High', 0) / n:.1%} of customers", "neg", "alert", ROSE),
        kpi_card("Medium risk", f"{counts.get('Medium', 0):,}", f"{counts.get('Medium', 0) / n:.1%} of customers", "warn", "activity", AMBER),
        kpi_card("Low risk", f"{counts.get('Low', 0):,}", f"{counts.get('Low', 0) / n:.1%} of customers", "pos", "check", EMERALD),
        kpi_card("Est. monthly revenue at risk", fmt_money(rev["high_risk_monthly_revenue"]) if rev else "n/a",
                 f"≈ {fmt_money(rev['high_risk_annualised'])} annualised (estimate)" if rev else "No charge column", "warn", "dollar", AMBER),
    ])
    st.write("")
    a, b = st.columns(2)
    with a:
        chart_card("Risk distribution", C.risk_bars(view_all), f"High ≥ {ctx.high:.0%}, Medium ≥ {ctx.medium:.0%}", key="pr-risk")
    with b:
        chart_card("Estimated probability histogram", C.probability_histogram(view_all["churn_probability"], ctx.high, ctx.medium),
                   "Dotted lines mark the risk bands", key="pr-hist")

    view = filter_view(ctx, "pr", include_risk=True)
    q = st.text_input("Find a customer ID", placeholder="e.g. CUST-10384", key=f"pr-search-{ctx.sig}") if sch.id_col else ""
    if q and sch.id_col:
        view = view[view[sch.id_col].astype(str).str.contains(q.strip(), case=False, regex=False)]
    view = view.sort_values("churn_probability", ascending=False)

    with panel("pr-table"):
        md('<div class="card-head"><div class="card-title">Scored customers</div>'
           '<div class="card-sub">Sorted by estimated churn probability</div></div>')
        if view.empty:
            empty_state("No customers match", "Adjust the filters or search above.")
        else:
            cols = ([sch.id_col] if sch.id_col else []) + ["risk_category", "churn_probability"] + sch.features
            cols += [sch.target] if sch.target else []
            show = view[cols].copy()
            show["risk_category"] = show["risk_category"].astype(str)
            st.dataframe(style_dataframe(show.head(2000)), width="stretch", hide_index=True, height=420,
                         column_config={"churn_probability": st.column_config.NumberColumn("Churn probability", format="percent"),
                                        "risk_category": "Risk"})
            if len(view) > 2000:
                st.caption("Showing the top 2,000 rows. Download the CSV for all rows.")
        a, b, _ = st.columns([1, 1, 2])
        a.download_button("Download all scored customers", export_scored_csv(view_all), "scored_customers.csv", "text/csv",
                          key="dl-all", type="primary")
        if len(view) != len(view_all):
            b.download_button("Download filtered view", export_scored_csv(view), "scored_customers_filtered.csv", "text/csv", key="dl-filt")

    inspector(ctx, view if not view.empty else view_all)


def inspector(ctx: AppContext, pool: pd.DataFrame) -> None:
    ws, sch = ctx.ws, ctx.ws.schema
    md('<h3 class="sub">Customer risk inspector</h3>')
    pool = pool.sort_values("churn_probability", ascending=False).head(1500)

    def label(i):
        r = pool.loc[i]
        ident = r[sch.id_col] if sch.id_col else f"Row {i}"
        return f"{ident} · {r['churn_probability']:.0%} · {r['risk_category']}"

    idx = st.selectbox("Select a customer (type to search)", pool.index.tolist(), format_func=label, key=f"insp-{ctx.sig}")
    row = pool.loc[idx]
    p = float(row["churn_probability"])
    effects = explain_customer(ws.result, ws.clean.loc[[idx]])
    left, right = st.columns([2, 3])
    with left:
        with panel("insp-gauge"):
            md('<div class="card-head"><div class="card-title">Estimated churn probability</div></div>')
            st.plotly_chart(C.risk_gauge(p, ctx.high, ctx.medium), theme=None, width="stretch", key="plot-insp-gauge",
                            config={"displayModeBar": False})
            md(f'<div style="text-align:center;margin-bottom:8px">{risk_pill(row["risk_category"])}</div>'
               f'<div class="card-sub" style="text-align:center">Score source: {esc(row["score_source"])}</div>')
        with panel("insp-attrs"):
            md('<div class="card-head"><div class="card-title">Customer attributes</div></div>')
            attrs = {c_: fmt_val(row[c_]) for c_ in sch.features}
            if sch.target and pd.notna(row[sch.target]):
                attrs["Actual outcome"] = "Churned" if row[sch.target] == 1 else "Retained"
            html_table(pd.DataFrame({"Attribute": list(attrs), "Value": list(attrs.values())}))
    with right:
        chart_card("What the model is reacting to", C.factor_effects(effects),
                   "Change in this customer's estimate if the value were typical (red = raises risk)", key="insp-factors")
    st.caption("These effects describe how the model behaves for this customer. They are associations learned from "
               "data, not proof of what causes a customer to churn.")


# --------------------------------------------------------------------------- AI Insights
def ai_page() -> None:
    ctx = get_ctx()
    header(ctx, "ai-insights")
    page_heading("AI Insights", "Ask questions about your results. Answers use only figures computed by this app.")
    cfg = llm_config()
    if cfg:
        callout(f"LLM-assisted mode ({cfg['provider']}, {cfg['model']}). Only aggregate statistics are sent, never customer rows. "
                "If the call fails, the app falls back to local insights.", "success", "LLM enabled.")
    else:
        callout("No API key found, so answers are generated locally from your computed results. Add a key to .env to enable "
                "LLM-written explanations (optional).", "info", "Local insights mode.")

    facts_key = f"facts-{ctx.sig}-{ctx.high}-{ctx.medium}"
    if facts_key not in st.session_state:
        ws = ctx.ws
        st.session_state[facts_key] = build_facts(ctx.view, ws.schema, ws.result, ws.findings, ctx.recs, ws.quality, ctx.high, ctx.medium)
    facts = st.session_state[facts_key]

    chat_key = f"chat-{ctx.sig}"
    chat = st.session_state.setdefault(chat_key, [])
    md('<div class="side-label" style="padding-left:0">Try asking</div>')
    cols = st.columns(len(SUGGESTED_QUESTIONS))
    clicked = None
    for i, (col, q) in enumerate(zip(cols, SUGGESTED_QUESTIONS)):
        if col.button(q, key=f"sq-{i}", width="stretch"):
            clicked = q
    typed = st.chat_input("Ask about risk, factors, segments or retention actions")
    question = clicked or typed
    if question:
        with st.spinner("Preparing an answer from your computed results…"):
            text, mode = answer_question(question, facts, chat)
        chat.append({"role": "user", "content": question})
        chat.append({"role": "assistant", "content": text, "mode": mode})
    if not chat:
        empty_state("Ask your first question", "Pick a suggestion above or type your own. Answers cite the numbers computed from your data.", "info")
    for m in chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                st.caption("Generated by an LLM from aggregate facts" if m.get("mode") == "llm" else "Generated locally from computed results")

    md('<h3 class="sub">Retention hypotheses to test</h3>')
    if not ctx.recs:
        empty_state("No hypotheses yet", "Hypotheses are generated from churn labels and recognised columns such as tenure, contract or charges.")
    else:
        callout("These are hypotheses, not proven solutions. Each shows the observed evidence and a way to test it, "
                "ideally with a controlled experiment.", "warn", "Hypotheses to test.")
        a, b = st.columns(2)
        for i, r in enumerate(ctx.recs):
            colr = EMERALD if r["supported"] else AMBER
            badge = pill("Supported by this data" if r["supported"] else "Not supported by this data", "pos" if r["supported"] else "warn")
            html = (f'<div class="rec" style="--c:{colr}"><div class="rec-title">{esc(r["title"])}</div>'
                    f'<div style="margin:6px 0 8px">{badge}{pill("Hypothesis to test", "brand")}</div>'
                    f'<div class="rec-meta"><b>Segment:</b> {esc(r["segment"])}<br><b>Evidence:</b> {esc(r["evidence"])}<br>'
                    f'<b>Why it might matter:</b> {esc(r["hypothesis"])}<br><b>How to test:</b> {esc(r["test"])}</div></div>')
            (a if i % 2 == 0 else b).markdown(html, unsafe_allow_html=True)


# --------------------------------------------------------------------------- Model Performance
def model_page() -> None:
    ctx = get_ctx()
    header(ctx, "model-performance")
    page_heading("Model Performance", "Held-out evaluation, model comparison and what drives the predictions.")
    ws = ctx.ws
    if not ws.has_model:
        no_model_state(ctx, "Model evaluation")
        return
    r, si, m = ws.result, ws.result.split_info, ws.result.metrics
    callout("Predictions are estimates, not certainties. Metrics describe this dataset and this test split; performance on "
            "new customers or later periods may be lower.", "warn", "Interpret with care.")
    kpi_row([
        kpi_card("ROC-AUC (test)", f"{m['roc_auc']:.3f}" if m["roc_auc"] is not None else "n/a", f"Held-out, n={si['n_test']:,}", "info", "activity", INDIGO),
        kpi_card("Precision (test)", f"{m['precision']:.2f}", f"At {m['threshold']:.2f} cut-off", "muted", "check", CYAN),
        kpi_card("Recall (test)", f"{m['recall']:.2f}", f"{m['confusion_matrix'][1][1]} of {m['positives']} churners caught", "muted", "trend", EMERALD),
        kpi_card("F1-score (test)", f"{m['f1']:.2f}", f"Model: {r.selected_name}", "brand", "activity", INDIGO),
    ])
    st.write("")
    with panel("mp-protocol"):
        md('<div class="card-head"><div class="card-title">Evaluation protocol</div></div>')
        md(f'<div class="rec-meta"><b>Split:</b> {esc(si["method"])} · {si["n_train"]:,} training / {si["n_test"]:,} test customers '
           f'(test_size={si["test_size"]}, random_state={si["random_state"]})<br>'
           f'<b>Churn rate:</b> train {si["train_churn_rate"]:.1%}, test {si["test_churn_rate"]:.1%}<br>'
           f'<b>Selection:</b> {esc(si["selection_rule"])} ({si["cv_folds"]}-fold CV)<br>'
           f'<b>Cut-off:</b> precision/recall/F1 use a {m["threshold"]:.3f} probability cut-off chosen on cross-validated '
           f'training predictions, never on the test set. Risk bands on other pages are separate and adjustable.</div>')

    a, b = st.columns([3, 2])
    with a:
        chart_card("Model comparison", C.model_comparison(r.comparison), "CV = validation on training portion; Test = held-out", key="mp-compare")
    with b:
        with panel("mp-table"):
            md('<div class="card-head"><div class="card-title">Comparison table</div></div>')
            comp = r.comparison.copy()
            comp["Selected"] = comp["Selected"].map({True: "✓", False: ""})
            for c in comp.columns:
                if comp[c].dtype.kind == "f":
                    comp[c] = comp[c].map(lambda v: "—" if pd.isna(v) else f"{v:.3f}")
            html_table(comp.rename(columns={"CV ROC-AUC": "CV AUC", "Test ROC-AUC": "Test AUC", "Selected": "Pick"})[["Model", "CV AUC", "Test AUC", "Pick"]])
            st.caption("Training-set performance is deliberately not reported.")

    a, b = st.columns(2)
    with a:
        chart_card("Confusion matrix (test set)", C.confusion_fig(m["confusion_matrix"]), "Rows = actual, columns = predicted", key="mp-cm")
    with b:
        chart_card("ROC curve (test set)", C.roc_curve_fig(r.roc_curves, {k: v["roc_auc"] for k, v in r.test_metrics.items()}, r.selected_name),
                   "Closer to the top-left is better", key="mp-roc")

    md('<h3 class="sub">What the models rely on</h3>')
    names = list(r.importance)
    tabs = st.tabs([f"{n} importance" for n in names] + (["Logistic coefficients"] if r.coefficients is not None else []))
    for tab, n in zip(tabs, names):
        with tab:
            chart_card(f"Permutation importance: {n}", C.importance_bars(r.importance[n]),
                       "Drop in test ROC-AUC when a feature is shuffled. Predictive usefulness, not causation.", key=f"mp-imp-{n}")
    if r.coefficients is not None:
        with tabs[-1]:
            chart_card("Logistic regression coefficients", C.coefficient_bars(r.coefficients),
                       "Numeric features are standardised; categories are one-hot. Associations, not causes.", key="mp-coef")

    with st.expander("Features used and columns excluded"):
        st.write("**Used as features:**", ", ".join(ws.schema.features) or "none")
        if ws.schema.excluded:
            ex = pd.DataFrame({"column": list(ws.schema.excluded), "reason": list(ws.schema.excluded.values())})
            html_table(ex)

    md('<h3 class="sub">Export and history</h3>')
    risk_counts = ctx.view["risk_category"].astype(str).value_counts().reindex(["High", "Medium", "Low"]).fillna(0).astype(int).to_dict()
    report = build_report_markdown(ws.name, ws.quality, ws.schema, r, ws.findings, ctx.recs, risk_counts, ctx.revenue, ctx.high, ctx.medium)
    metrics_json = json.dumps({"model": r.selected_name, "split": si, "test_metrics": r.metrics,
                               "cv_metrics": {k: {kk: vv for kk, vv in v.items() if kk != "confusion_matrix"} for k, v in r.cv_metrics.items()}}, indent=2, default=float)
    a, b, c, _ = st.columns([1, 1, 1, 1])
    a.download_button("Download report (.md)", report, "churn_report.md", "text/markdown", type="primary", key="dl-report")
    b.download_button("Download metrics (.json)", metrics_json, "model_metrics.json", "application/json", key="dl-metrics")
    if c.button("Save run to SQLite", key="save-run"):
        try:
            rid = save_run(config.DB_PATH, ws.name, r.selected_name, r.metrics, si,
                           ctx.view[["churn_probability", "risk_category", "score_source"]],
                           ctx.view[ws.schema.id_col] if ws.schema.id_col else None)
            st.success(f"Saved run #{rid} to {config.DB_PATH.name}")
        except Exception as exc:
            st.error(f"Could not save the run: {exc}")
    if config.DB_PATH.exists():
        runs = list_runs(config.DB_PATH, 8)
        if not runs.empty:
            with panel("mp-runs"):
                md('<div class="card-head"><div class="card-title">Saved runs</div><div class="card-sub">Most recent first</div></div>')
                runs["roc_auc"] = runs["roc_auc"].map(lambda v: "—" if pd.isna(v) else f"{v:.3f}")
                for c in ("precision", "recall", "f1"):
                    runs[c] = runs[c].map(lambda v: "—" if pd.isna(v) else f"{v:.2f}")
                html_table(runs)


# --------------------------------------------------------------------------- Data Explorer
def explorer_page() -> None:
    ctx = get_ctx()
    header(ctx, "data-explorer")
    page_heading("Data Explorer", "Data quality, cleaning steps and the underlying records. Your original upload is never modified.")
    ws, q, sch = ctx.ws, ctx.ws.quality, ctx.ws.schema
    kpi_row([
        kpi_card("Rows", f"{q['rows']:,}", f"{len(ws.clean):,} after cleaning", "muted", "database"),
        kpi_card("Columns", f"{q['columns']}", f"{len(sch.features)} model features", "info", "columns", CYAN),
        kpi_card("Missing cells", f"{q['missing_pct']}%", f"{q['missing_cells']:,} cells", "warn" if q["missing_cells"] else "pos", "alert", AMBER),
        kpi_card("Duplicate rows", f"{q['duplicate_rows']:,}", f"{q['duplicate_ids']:,} repeated IDs" if q["id_column"] else "No ID column", "warn" if q["duplicate_rows"] else "pos", "check", EMERALD),
    ])
    st.write("")
    t1, t2, t3, t4 = st.tabs(["Data quality", "Cleaned data", "Column profile", "Original upload"])
    with t1:
        a, b = st.columns([3, 2])
        with a:
            chart_card("Missing values by column", C.missing_bars(q["missing_by_column"]), "Share of rows with a missing or blank value", key="ex-missing")
        with b:
            with panel("ex-score"):
                md(f'<div class="card-head"><div class="card-title">Data-quality score: {q["quality_score"]}/100</div>'
                   '<div class="card-sub">A heuristic combining missing data, duplicates, type issues and outliers</div></div>')
                md("".join(f'<div class="finding">{esc(l)}</div>' for l in ws.clean_log))
        a, b = st.columns(2)
        with a:
            with panel("ex-types"):
                md('<div class="card-head"><div class="card-title">Data-type issues</div></div>')
                html_table(pd.DataFrame(q["type_issues"])) if q["type_issues"] else md('<div class="card-sub">None detected.</div>')
        with b:
            with panel("ex-unusual"):
                md('<div class="card-head"><div class="card-title">Unusual values</div></div>')
                html_table(pd.DataFrame(q["unusual_values"])) if q["unusual_values"] else md('<div class="card-sub">None detected.</div>')
        with panel("ex-schema"):
            md('<div class="card-head"><div class="card-title">Detected schema</div><div class="card-sub">How columns are used</div></div>')
            roles = ", ".join(f"{k}: {v}" for k, v in sch.roles.items()) or "none recognised"
            md(f'<div class="rec-meta"><b>Churn label:</b> {esc(sch.target or "none")} · <b>ID:</b> {esc(sch.id_col or "none")}<br>'
               f'<b>Recognised roles:</b> {esc(roles)}<br><b>Numeric features:</b> {esc(", ".join(sch.numeric_features) or "none")}<br>'
               f'<b>Categorical features:</b> {esc(", ".join(sch.categorical_features) or "none")}</div>')
    with t2:
        st.caption("Cleaned copy plus scores. Missing values are kept as blanks; imputation happens inside the model pipeline using training data only.")
        st.dataframe(style_dataframe(ctx.view.head(3000).assign(**({"risk_category": ctx.view.head(3000)["risk_category"].astype(str)} if "risk_category" in ctx.view else {}))),
                     width="stretch", height=460, hide_index=True)
        st.download_button("Download cleaned data (CSV)", export_scored_csv(ctx.view), "cleaned_customers.csv", "text/csv", key="dl-clean")
    with t3:
        if sch.numeric_features:
            with panel("ex-numprofile"):
                md('<div class="card-head"><div class="card-title">Numeric summary</div></div>')
                html_table(numeric_summary(ws.clean, sch.numeric_features).round(2).reset_index().rename(columns={"index": "column"}))
        if sch.categorical_features:
            col = st.selectbox("Categorical column", sch.categorical_features, key=f"prof-{ctx.sig}")
            vc = ws.clean[col].value_counts(dropna=False).reset_index()
            vc.columns = [col, "customers"]
            vc["share"] = (vc["customers"] / vc["customers"].sum()).map(lambda v: f"{v:.1%}")
            with panel("ex-catprofile"):
                html_table(vc.head(25))
    with t4:
        st.caption("Exactly as uploaded (first 500 rows).")
        st.dataframe(ws.raw.head(500), width="stretch", height=460, hide_index=True)
