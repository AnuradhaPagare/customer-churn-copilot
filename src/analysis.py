"""Descriptive analytics, key findings and retention hypotheses.

Everything here is computed from the data. Findings describe associations in the
observed data; recommendations are hypotheses to test, never proven solutions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .model import ModelResult, assign_risk
from .preprocessing import Schema

TENURE_BINS = [0, 6, 12, 24, 36, 48, 60, np.inf]
SMALL_SAMPLE = 30


# --------------------------------------------------------------------------- core frames
def build_view(clean: pd.DataFrame, result: ModelResult | None,
               high: float = config.DEFAULT_HIGH_THRESHOLD,
               medium: float = config.DEFAULT_MEDIUM_THRESHOLD) -> pd.DataFrame:
    """Cleaned data + churn probability + risk band (if a model exists). Original is untouched."""
    view = clean.copy()
    if result is not None:
        view["churn_probability"] = result.scored["churn_probability"].reindex(view.index)
        view["risk_category"] = assign_risk(view["churn_probability"].fillna(0).to_numpy(), high, medium)
        view.loc[view["churn_probability"].isna(), "risk_category"] = np.nan
        view["score_source"] = result.scored["score_source"].reindex(view.index)
    return view


def overall_churn_rate(df: pd.DataFrame, target: str | None) -> float | None:
    if not target or target not in df:
        return None
    y = df[target].dropna()
    return float(y.mean()) if len(y) else None


def group_churn(df: pd.DataFrame, by: str, target: str) -> pd.DataFrame:
    d = df[[by, target]].dropna()
    g = d.groupby(by, observed=True)[target].agg(customers="size", churned="sum", churn_rate="mean").reset_index()
    return g


def tenure_bins(s: pd.Series) -> pd.Series:
    """Readable tenure buckets (months). Falls back to quantiles for non-month scales."""
    s = pd.to_numeric(s, errors="coerce")
    if s.max() <= 120:
        labels = ["0–6", "7–12", "13–24", "25–36", "37–48", "49–60", "61+"]
        return pd.cut(s, bins=TENURE_BINS, labels=labels, include_lowest=True, ordered=True)
    return pd.qcut(s, q=6, duplicates="drop").astype(str)


def bin_numeric(s: pd.Series, q: int = 5) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    try:
        return pd.qcut(s, q=q, duplicates="drop").astype(str).where(s.notna())
    except ValueError:
        return pd.Series(np.nan, index=s.index)


def segment_summary(view: pd.DataFrame, by: str, target: str | None, monthly_col: str | None) -> pd.DataFrame:
    """One row per segment: size, actual churn, average predicted risk, high-risk count, avg charge."""
    d = view.copy()
    keyed = d.assign(_seg=d[by]).dropna(subset=["_seg"])
    agg = {"customers": ("_seg", "size")}
    if target and target in keyed:
        agg["churn_rate"] = (target, "mean")
    if "churn_probability" in keyed:
        agg["avg_predicted_risk"] = ("churn_probability", "mean")
        keyed["_hi"] = (keyed["risk_category"].astype(str) == "High").astype(int)
        agg["high_risk_customers"] = ("_hi", "sum")
    if monthly_col and monthly_col in keyed:
        agg["avg_monthly_charge"] = (monthly_col, "mean")
    out = keyed.groupby("_seg", observed=True).agg(**agg).reset_index().rename(columns={"_seg": by})
    return out


def revenue_estimate(view: pd.DataFrame, monthly_col: str | None) -> dict | None:
    """Revenue associated with high-risk customers. An ESTIMATE, not a forecast."""
    if not monthly_col or monthly_col not in view or "risk_category" not in view:
        return None
    rev = pd.to_numeric(view[monthly_col], errors="coerce")
    high = view["risk_category"].astype(str) == "High"
    total = float(rev.sum())
    high_rev = float(rev[high].sum())
    expected = float((rev * view["churn_probability"]).sum())
    return {
        "total_monthly_revenue": total,
        "high_risk_monthly_revenue": high_rev,
        "high_risk_annualised": high_rev * 12,
        "high_risk_share": high_rev / total if total else 0.0,
        "probability_weighted_monthly": expected,
        "customers_without_charge_data": int(rev.isna().sum()),
    }


def numeric_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame()
    d = df[cols].apply(pd.to_numeric, errors="coerce")
    out = d.describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    out["missing_%"] = (100 * d.isna().mean()).round(2)
    return out.rename(columns={"50%": "median"})


# --------------------------------------------------------------------------- segment comparison
def compare_segment(df: pd.DataFrame, mask: pd.Series, target: str) -> dict | None:
    d = df[df[target].notna()]
    m = mask.reindex(d.index).fillna(False).astype(bool)
    seg, rest = d[m], d[~m]
    if len(seg) == 0 or len(rest) == 0:
        return None
    sr, rr = float(seg[target].mean()), float(rest[target].mean())
    return {"seg_n": int(len(seg)), "seg_rate": sr, "rest_n": int(len(rest)), "rest_rate": rr,
            "diff_pp": (sr - rr) * 100, "small": len(seg) < SMALL_SAMPLE}


def _fmt_cmp(c: dict) -> str:
    note = " Small sample, interpret cautiously." if c["small"] else ""
    return (f"Observed churn is {c['seg_rate']:.1%} in this segment (n={c['seg_n']:,}) versus "
            f"{c['rest_rate']:.1%} for all other customers (n={c['rest_n']:,}).{note}")


# --------------------------------------------------------------------------- findings
def generate_findings(clean: pd.DataFrame, schema: Schema, result: ModelResult | None = None) -> list[dict]:
    """Evidence-based observations. Wording is associative, never causal."""
    target = schema.target
    findings: list[dict] = []
    n = len(clean)
    findings.append({"title": "Dataset", "category": "data",
                     "detail": f"{n:,} customers and {clean.shape[1]} columns; "
                               f"{len(schema.features)} columns are usable as model features."})
    if not target:
        return findings
    overall = overall_churn_rate(clean, target)
    labelled = int(clean[target].notna().sum())
    findings.append({"title": "Overall churn", "category": "metric",
                     "detail": f"{overall:.1%} of {labelled:,} labelled customers churned."})

    seg_rows = []
    for col in schema.categorical_features:
        if clean[col].nunique(dropna=True) > 12:
            continue
        g = group_churn(clean, col, target)
        g = g[g["customers"] >= max(20, int(0.02 * labelled))]
        for _, r in g.iterrows():
            seg_rows.append((r["churn_rate"] - overall, col, r[col], int(r["customers"]), r["churn_rate"]))
    seg_rows.sort(key=lambda t: t[0], reverse=True)
    for diff, col, val, cnt, rate in seg_rows[:3]:
        if diff > 0.02:
            findings.append({"title": f"Higher churn: {col} = {val}", "category": "segment",
                             "detail": f"{rate:.1%} observed churn (n={cnt:,}), {diff * 100:.1f} percentage points "
                                       f"above the overall rate."})
    if seg_rows and seg_rows[-1][0] < -0.02:
        diff, col, val, cnt, rate = seg_rows[-1]
        findings.append({"title": f"Lower churn: {col} = {val}", "category": "segment",
                         "detail": f"{rate:.1%} observed churn (n={cnt:,}), {abs(diff) * 100:.1f} percentage points "
                                   f"below the overall rate."})

    num_rows = []
    d = clean[clean[target].notna()]
    for col in schema.numeric_features:
        a = pd.to_numeric(d.loc[d[target] == 1, col], errors="coerce").dropna()
        b = pd.to_numeric(d.loc[d[target] == 0, col], errors="coerce").dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        pooled = np.sqrt((a.var() + b.var()) / 2)
        if pooled > 0:
            num_rows.append((abs(a.mean() - b.mean()) / pooled, col, a.mean(), b.mean()))
    num_rows.sort(reverse=True)
    for eff, col, ma, mb in num_rows[:3]:
        if eff >= 0.15:
            direction = "higher" if ma > mb else "lower"
            findings.append({"title": f"{col}: {direction} among churned customers", "category": "numeric",
                             "detail": f"Average {ma:,.1f} for churned vs {mb:,.1f} for retained customers "
                                       f"(an association, not proof of cause)."})
    if result is not None:
        top = result.importance[result.selected_name].head(3)
        names = ", ".join(top["feature"])
        findings.append({"title": "Features the model relies on most", "category": "model",
                         "detail": f"{names} (permutation importance on the held-out test set). "
                                   "Importance shows predictive usefulness, not causation."})
    return findings


# --------------------------------------------------------------------------- hypotheses
def generate_recommendations(view: pd.DataFrame, schema: Schema, result: ModelResult | None = None) -> list[dict]:
    """Retention hypotheses with the evidence behind each. All are things to TEST."""
    target, roles = schema.target, schema.roles
    if not target:
        return []
    recs: list[dict] = []

    def add(title, segment, comp, hypothesis, test):
        if comp is None:
            return
        recs.append({"title": title, "segment": segment, "evidence": _fmt_cmp(comp),
                     "supported": comp["diff_pp"] > 2.0, "hypothesis": hypothesis, "test": test,
                     "customers": comp["seg_n"], "diff_pp": comp["diff_pp"]})

    t, m = roles.get("tenure"), roles.get("monthly")
    if t and m:
        short = pd.to_numeric(view[t], errors="coerce") <= 12
        pricey = pd.to_numeric(view[m], errors="coerce") >= pd.to_numeric(view[m], errors="coerce").median()
        add("Review short-tenure customers with high monthly charges",
            f"{t} ≤ 12 and {m} ≥ median", compare_segment(view, short & pricey, target),
            "New customers paying above-median prices may feel a value gap early on.",
            "Pilot an onboarding check-in or plan-fit review for a random half of this group and compare retention.")

    c = roles.get("contract")
    if c and view[c].nunique() > 1:
        g = group_churn(view, c, target)
        g = g[g["customers"] >= 20].sort_values("churn_rate", ascending=False)
        if len(g):
            top = g.iloc[0][c]
            add(f"Investigate whether '{top}' customers are more likely to churn", f"{c} = {top}",
                compare_segment(view, view[c] == top, target),
                "Flexible or short commitments may make leaving easier; the plan itself may not be the issue.",
                "Test a term-commitment incentive or loyalty benefit against a control group before any pricing change.")

    s = roles.get("support")
    if s:
        sup = pd.to_numeric(view[s], errors="coerce")
        thr = max(2, float(sup.quantile(0.75)))
        add(f"Prioritise customers with frequent support contact ({s} ≥ {thr:g})", f"{s} ≥ {thr:g}",
            compare_segment(view, sup >= thr, target),
            "Repeated support contact may signal unresolved problems that precede cancellation.",
            "Review these cases for root causes and test proactive follow-up on a sample.")

    p = roles.get("payment")
    if p and view[p].nunique() > 1:
        g = group_churn(view, p, target)
        g = g[g["customers"] >= 20].sort_values("churn_rate", ascending=False)
        if len(g):
            top = g.iloc[0][p]
            add(f"Look into the '{top}' payment method", f"{p} = {top}", compare_segment(view, view[p] == top, target),
                "Payment friction (failed payments, manual steps) may contribute; other factors may also differ.",
                "Test an automatic-payment prompt for this group and track churn against a control.")

    u = roles.get("usage")
    if u:
        use = pd.to_numeric(view[u], errors="coerce")
        add("Re-engage low-usage customers", f"{u} in the lowest quartile",
            compare_segment(view, use <= use.quantile(0.25), target),
            "Customers who barely use the service may not see its value and may be easier to lose.",
            "Try a usage tip or feature-adoption campaign on a sample and compare churn.")

    if m:
        mon = pd.to_numeric(view[m], errors="coerce")
        add("Evaluate pricing or contract options for the highest-charge customers", f"{m} in the top quartile",
            compare_segment(view, mon >= mon.quantile(0.75), target),
            "Higher bills may raise price sensitivity; a cheaper or bundled plan might help, but that is untested.",
            "Run a controlled offer (e.g. plan review or bundle) for a subset and measure retention and margin.")

    if "risk_category" in view:
        high = view["risk_category"].astype(str) == "High"
        n_high = int(high.sum())
        if n_high:
            est = revenue_estimate(view, m)
            rev = f" They account for an estimated ${est['high_risk_monthly_revenue']:,.0f} of monthly charges." if est else ""
            recs.append({
                "title": "Prioritise high-risk customers for appropriate support",
                "segment": "Risk category = High", "supported": True, "customers": n_high, "diff_pp": float("nan"),
                "evidence": f"{n_high:,} customers ({n_high / len(view):.1%}) have an estimated churn probability at or above "
                            f"the high-risk threshold.{rev}",
                "hypothesis": "Early, tailored outreach to the highest-risk customers may reduce churn, "
                              "but the model only estimates risk; it does not show what would change a decision.",
                "test": "Contact a random half of the high-risk list, hold out the rest, and compare 90-day retention."})
    recs.sort(key=lambda r: (not r["supported"], -(r["diff_pp"] if r["diff_pp"] == r["diff_pp"] else 99)))
    return recs


# --------------------------------------------------------------------------- export helpers
def _sanitize_cell(v):
    """Neutralise spreadsheet formula injection in exported text cells."""
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + v
    return v


def export_scored_csv(view: pd.DataFrame) -> bytes:
    """CSV bytes for scored customers (risk columns included). Text cells are sanitised."""
    out = view.copy()
    for c in out.columns:
        if out[c].dtype == object or pd.api.types.is_string_dtype(out[c]):
            out[c] = out[c].map(_sanitize_cell)
    if "churn_probability" in out:
        out["churn_probability"] = out["churn_probability"].round(4)
        out = out.sort_values("churn_probability", ascending=False)
    return out.to_csv(index=False).encode("utf-8")
