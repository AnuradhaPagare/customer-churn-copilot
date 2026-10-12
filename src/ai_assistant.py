"""Retention assistant.

Two modes, same evidence:
- Local (default): deterministic answers generated from computed results. No network.
- LLM (optional): if an API key is set in .env, a language model PHRASES an answer, but it only
  receives aggregate facts computed by this app. Model output is treated as plain text and is
  never executed. Any failure falls back to the local answer.
"""
from __future__ import annotations

import os
import re

import pandas as pd
import requests

from .analysis import revenue_estimate, segment_summary
from .model import ModelResult
from .preprocessing import Schema

DISCLAIMER = ("_These are estimates from observed data. Associations are not proof of cause, and no retention "
              "action is guaranteed to prevent churn. Treat suggestions as hypotheses to test._")

SYSTEM_PROMPT = (
    "You are a retention analytics assistant inside a churn-prediction app. Answer ONLY from the JSON facts "
    "provided. If the facts do not contain the answer, say so. Never invent numbers or customers. Never say a "
    "model factor causes churn; describe it as an association. Never promise or guarantee that an action will "
    "prevent churn; frame actions as hypotheses to test (e.g. with an A/B test). Predictions are probability "
    "estimates, not certainties. Be concise (under 220 words), use short bullet points, and cite figures from "
    "the facts. Do not output code."
)

SUGGESTED_QUESTIONS = [
    "Which customers are most at risk?",
    "What factors are associated with churn?",
    "Which customer segment has the highest churn rate?",
    "What retention actions should the business investigate?",
    "Summarize the model evaluation.",
]


# --------------------------------------------------------------------------- facts
def build_facts(view: pd.DataFrame, schema: Schema, result: ModelResult | None, findings: list[dict],
                recs: list[dict], quality: dict, high: float, medium: float) -> dict:
    """Aggregate, JSON-safe evidence. `_local` holds row-level details that are never sent to an LLM."""
    target = schema.target
    facts: dict = {
        "dataset": {"rows": int(len(view)), "columns": int(view.shape[1]),
                    "data_quality_score": quality["quality_score"], "missing_pct": quality["missing_pct"],
                    "has_churn_labels": bool(target)},
        "findings": [{"title": f["title"], "detail": f["detail"]} for f in findings],
        "hypotheses": [{"title": r["title"], "evidence": r["evidence"], "supported_by_data": r["supported"],
                        "how_to_test": r["test"]} for r in recs],
        "_local": {},
    }
    if not target:
        facts["note"] = "No churn labels: supervised training unavailable; analysis is descriptive."
        return facts
    labelled = view[target].dropna()
    facts["dataset"]["actual_churn_rate"] = round(float(labelled.mean()), 4)
    if result is not None:
        m, si = result.metrics, result.split_info
        facts["model"] = {
            "name": result.selected_name,
            "evaluation": f"{si['method']}: {si['n_train']} train / {si['n_test']} test; selected on cross-validated "
                          "ROC-AUC of the training portion",
            "test_roc_auc": None if m["roc_auc"] is None else round(m["roc_auc"], 3),
            "test_precision": round(m["precision"], 3), "test_recall": round(m["recall"], 3),
            "test_f1": round(m["f1"], 3), "decision_threshold": round(m["threshold"], 3),
            "confusion_matrix_[[TN,FP],[FN,TP]]": m["confusion_matrix"],
            "cv_roc_auc": round(result.cv_metrics[result.selected_name]["cv_auc_mean"], 3),
            "alternatives": {r["Model"]: {"cv_roc_auc": round(r["CV ROC-AUC"], 3), "test_roc_auc": None
                             if pd.isna(r["Test ROC-AUC"]) else round(r["Test ROC-AUC"], 3)}
                             for _, r in result.comparison.iterrows()},
            "top_permutation_importance": [
                {"feature": r.feature, "auc_drop": round(float(r.importance), 4)}
                for r in result.importance[result.selected_name].head(5).itertuples()],
        }
        counts = view["risk_category"].astype(str).value_counts().to_dict()
        facts["risk"] = {"bands": f"High >= {high:.0%}, Medium >= {medium:.0%} estimated probability",
                         "counts": {k: int(counts.get(k, 0)) for k in ["Low", "Medium", "High"]}}
        rev = revenue_estimate(view, schema.roles.get("monthly"))
        if rev:
            facts["revenue_estimate_monthly"] = {k: round(v, 2) if isinstance(v, float) else v for k, v in rev.items()}
        top = view.sort_values("churn_probability", ascending=False).head(10)
        idc = schema.id_col
        facts["_local"]["top_customers"] = [
            {"id": str(r[idc]) if idc else str(i), "probability": float(r["churn_probability"]),
             "risk": str(r["risk_category"])} for i, r in top.iterrows()]
    segs = []
    for col in (schema.categorical_features or [])[:6]:
        if view[col].nunique(dropna=True) <= 12:
            g = segment_summary(view, col, target, schema.roles.get("monthly"))
            g = g[g["customers"] >= 20]
            for _, r in g.iterrows():
                segs.append({"column": col, "segment": str(r[col]), "customers": int(r["customers"]),
                             "actual_churn_rate": round(float(r["churn_rate"]), 4)})
    segs.sort(key=lambda d: d["actual_churn_rate"], reverse=True)
    facts["segments_by_churn_rate"] = segs[:8]
    return facts


# --------------------------------------------------------------------------- local answers
def _intent(q: str) -> str:
    q = q.lower()
    rules = [
        ("model", r"model|evaluat|accura|auc|precision|recall|f1|metric|performance|confusion|summar"),
        ("risk", r"at risk|most at|who|which customers|high[- ]risk|top customers|likely to (leave|churn)"),
        ("segment", r"segment|group|highest churn|contract|plan|category|cohort"),
        ("factors", r"factor|driver|associat|why|feature|importan|reason|cause|correlat"),
        ("actions", r"retention|action|strateg|recommend|reduce|prevent|intervention|offer|what (should|can)"),
        ("revenue", r"revenue|money|financial|exposure|dollar|charges"),
    ]
    for name, pat in rules:
        if re.search(pat, q):
            return name
    return "overview"


def local_answer(question: str, facts: dict) -> str:
    """Deterministic, evidence-only answer. Every number comes from `facts`."""
    ds, intent = facts["dataset"], _intent(question)
    if not ds["has_churn_labels"]:
        return (f"This dataset has no usable churn label, so supervised training and risk predictions are "
                f"unavailable. What I can say descriptively: {ds['rows']:,} rows, {ds['columns']} columns, a "
                f"data-quality score of {ds['data_quality_score']}/100.\n\nTo get predictions, upload data with a "
                f"column such as `churn` (Yes/No or 1/0).")
    model, risk = facts.get("model"), facts.get("risk")

    if intent == "risk":
        if not risk:
            return "No model is available, so I can't rank customers by risk."
        c = risk["counts"]
        parts = [f"**Risk distribution** ({risk['bands']}): High {c['High']:,}, Medium {c['Medium']:,}, Low {c['Low']:,}."]
        top = facts["_local"].get("top_customers", [])[:5]
        if top:
            parts.append("**Highest estimated churn probability:**\n" +
                         "\n".join(f"- {t['id']}: {t['probability']:.0%} ({t['risk']})" for t in top))
        parts.append("Probabilities are estimates; use them to prioritise review, not as certain outcomes.")
        return "\n\n".join(parts) + "\n\n" + DISCLAIMER

    if intent == "factors":
        out = []
        if model:
            imp = ", ".join(f"{d['feature']}" for d in model["top_permutation_importance"][:4])
            out.append(f"**Model view:** the features that most improved held-out ROC-AUC (permutation importance) "
                       f"were {imp}.")
        obs = [f for f in facts["findings"] if f["title"].startswith(("Higher", "Lower")) or "among churned" in f["title"]]
        if obs:
            out.append("**Observed in the data:**\n" + "\n".join(f"- {f['title']}: {f['detail']}" for f in obs[:5]))
        return "\n\n".join(out or ["No factor information is available yet."]) + "\n\n" + DISCLAIMER

    if intent == "segment":
        segs = facts.get("segments_by_churn_rate", [])
        if not segs:
            return "No segment with enough customers was found."
        lines = [f"- {s['column']} = {s['segment']}: {s['actual_churn_rate']:.1%} churn (n={s['customers']:,})"
                 for s in segs[:5]]
        top = segs[0]
        return (f"The highest observed churn rate is **{top['column']} = {top['segment']}** at "
                f"{top['actual_churn_rate']:.1%} (overall {ds['actual_churn_rate']:.1%}).\n\n" + "\n".join(lines)
                + "\n\n" + DISCLAIMER)

    if intent == "actions":
        recs = facts.get("hypotheses", [])
        if not recs:
            return "No hypotheses could be generated from the available columns."
        out = ["**Hypotheses to investigate** (not proven solutions):"]
        for r in recs[:4]:
            flag = "supported by this data" if r["supported_by_data"] else "NOT supported by this data"
            out.append(f"- **{r['title']}** ({flag}). {r['evidence']} Test: {r['how_to_test']}")
        return "\n".join(out) + "\n\n" + DISCLAIMER

    if intent == "model":
        if not model:
            return "No model was trained."
        cm = model["confusion_matrix_[[TN,FP],[FN,TP]]"]
        auc = "n/a" if model["test_roc_auc"] is None else f"{model['test_roc_auc']:.3f}"
        return (f"**{model['name']}** was selected using cross-validated ROC-AUC ({model['cv_roc_auc']:.3f}) on the "
                f"training portion. Evaluation: {model['evaluation']}.\n\n"
                f"**Held-out test set:** ROC-AUC {auc}, precision {model['test_precision']:.2f}, recall "
                f"{model['test_recall']:.2f}, F1 {model['test_f1']:.2f} at a {model['decision_threshold']:.2f} cut-off. "
                f"Of the test customers, {cm[1][1]} churners were caught, {cm[1][0]} missed, and {cm[0][1]} "
                f"retained customers were flagged.\n\nThe model is an estimate; performance on new data may differ.")

    if intent == "revenue":
        rev = facts.get("revenue_estimate_monthly")
        if not rev:
            return "No monthly charge column was detected, so I can't estimate revenue exposure."
        return (f"High-risk customers are associated with roughly **${rev['high_risk_monthly_revenue']:,.0f}** of "
                f"monthly charges ({rev['high_risk_share']:.1%} of the total; about "
                f"${rev['high_risk_annualised']:,.0f} annualised). This is an exposure estimate, not a forecast of "
                f"revenue that will be lost.")

    parts = [f"**Overview:** {ds['rows']:,} customers, actual churn {ds['actual_churn_rate']:.1%}."]
    if risk:
        parts.append(f"The model flags {risk['counts']['High']:,} customers as high risk.")
    parts.append("Try asking about at-risk customers, churn factors, segments, retention actions, or model evaluation.")
    return " ".join(parts)


# --------------------------------------------------------------------------- optional LLM
def llm_config() -> dict | None:
    """Return provider settings if a key is configured, else None (local mode)."""
    provider = (os.getenv("LLM_PROVIDER") or "anthropic").strip().lower()
    placeholders = {"", "your-api-key-here", "changeme"}
    if provider == "openai":
        key = (os.getenv("OPENAI_API_KEY") or "").strip()
        model = os.getenv("LLM_MODEL") or "gpt-4o-mini"
    else:
        provider = "anthropic"
        key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
        model = os.getenv("LLM_MODEL") or "claude-sonnet-5-5"
    if key in placeholders:
        return None
    return {"provider": provider, "key": key, "model": model}


def _strip_private(facts: dict) -> dict:
    return {k: v for k, v in facts.items() if not k.startswith("_")}


def _call_llm(cfg: dict, question: str, facts: dict, history: list[dict]) -> str:
    import json

    context = "FACTS (JSON):\n" + json.dumps(_strip_private(facts), default=str)
    msgs = [{"role": m["role"], "content": m["content"]} for m in history[-6:] if m["role"] in ("user", "assistant")]
    msgs.append({"role": "user", "content": f"{context}\n\nQUESTION: {question}"})
    if cfg["provider"] == "openai":
        r = requests.post("https://api.openai.com/v1/chat/completions", timeout=30,
                          headers={"Authorization": f"Bearer {cfg['key']}"},
                          json={"model": cfg["model"], "max_tokens": 700,
                                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *msgs]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    r = requests.post("https://api.anthropic.com/v1/messages", timeout=30,
                      headers={"x-api-key": cfg["key"], "anthropic-version": "2023-06-01"},
                      json={"model": cfg["model"], "max_tokens": 700, "system": SYSTEM_PROMPT, "messages": msgs})
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text").strip()


def answer_question(question: str, facts: dict, history: list[dict] | None = None) -> tuple[str, str]:
    """Return (answer_markdown, mode) where mode is 'llm' or 'local'. Never raises."""
    question = (question or "").strip()[:1000]
    if not question:
        return "Please type a question.", "local"
    cfg = llm_config()
    if cfg and facts["dataset"]["has_churn_labels"]:
        try:
            text = _call_llm(cfg, question, facts, history or [])
            if text:
                return text + "\n\n" + DISCLAIMER, "llm"
        except Exception as exc:  # network, auth, rate limit, bad JSON ...
            note = f"_LLM unavailable ({type(exc).__name__}); showing local insights instead._\n\n"
            return note + local_answer(question, facts), "local"
    return local_answer(question, facts), "local"
