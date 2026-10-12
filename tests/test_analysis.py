import io
import json

import pandas as pd

from src import config
from src.ai_assistant import SUGGESTED_QUESTIONS, answer_question, build_facts, llm_config, local_answer
from src.analysis import (build_view, export_scored_csv, generate_findings, generate_recommendations, group_churn,
                          revenue_estimate, tenure_bins)
from src.database import list_runs, load_predictions, save_run
from src.report import build_report_markdown


def test_group_churn_and_tenure_bins(ws):
    g = group_churn(ws.clean, "contract_type", "churn")
    assert g["customers"].sum() == len(ws.clean)
    assert g.set_index("contract_type").loc["Month-to-month", "churn_rate"] > g.set_index("contract_type").loc["Two year", "churn_rate"]
    assert tenure_bins(ws.clean["tenure_months"]).notna().all()


def test_view_adds_scores_without_mutating_clean(ws):
    before = ws.clean.copy()
    view = build_view(ws.clean, ws.result, 0.5, 0.25)
    assert {"churn_probability", "risk_category"} <= set(view.columns)
    pd.testing.assert_frame_equal(ws.clean, before)


def test_revenue_estimate_is_labelled_estimate(ws):
    view = build_view(ws.clean, ws.result)
    rev = revenue_estimate(view, ws.schema.roles["monthly"])
    assert 0 <= rev["high_risk_share"] <= 1
    assert rev["high_risk_annualised"] == rev["high_risk_monthly_revenue"] * 12


def test_csv_export_roundtrip_and_injection_safety(ws):
    view = build_view(ws.clean, ws.result)
    view.loc[view.index[0], "contract_type"] = "=HYPERLINK(\"http://evil\")"
    out = pd.read_csv(io.BytesIO(export_scored_csv(view)))
    assert len(out) == len(view) and {"churn_probability", "risk_category"} <= set(out.columns)
    assert not out["contract_type"].astype(str).str.startswith("=").any()
    assert out["churn_probability"].is_monotonic_decreasing


def test_findings_and_recommendations_are_hedged(ws):
    view = build_view(ws.clean, ws.result)
    recs = generate_recommendations(view, ws.schema, ws.result)
    assert recs and all({"evidence", "hypothesis", "test", "supported"} <= set(r) for r in recs)
    text = " ".join(r["hypothesis"] + r["test"] for r in recs).lower()
    assert "guarantee" not in text and "will reduce" not in text
    assert generate_findings(ws.clean, ws.schema, ws.result)


def test_report_contains_split_and_model(ws):
    view = build_view(ws.clean, ws.result)
    recs = generate_recommendations(view, ws.schema, ws.result)
    rep = build_report_markdown("sample", ws.quality, ws.schema, ws.result, ws.findings, recs)
    assert ws.result.selected_name in rep
    assert "Evaluation split" in rep and "test_size=0.2" in rep and "random_state=42" in rep
    assert "estimates, not certainties" in rep


def test_database_roundtrip(tmp_path, ws):
    view = build_view(ws.clean, ws.result)
    db = tmp_path / "t.db"
    rid = save_run(db, "sample", ws.result.selected_name, ws.result.metrics, ws.result.split_info,
                   view[["churn_probability", "risk_category", "score_source"]], view["customer_id"])
    runs = list_runs(db)
    assert runs.loc[0, "id"] == rid and runs.loc[0, "model_name"] == ws.result.selected_name
    assert len(load_predictions(db, rid)) == len(view)


def test_assistant_works_without_api_key(ws, monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert llm_config() is None
    view = build_view(ws.clean, ws.result)
    recs = generate_recommendations(view, ws.schema, ws.result)
    facts = build_facts(view, ws.schema, ws.result, ws.findings, recs, ws.quality, 0.5, 0.25)
    for q in SUGGESTED_QUESTIONS:
        text, mode = answer_question(q, facts)
        assert mode == "local" and len(text) > 40
    seg = local_answer("Which customer segment has the highest churn rate?", facts)
    assert "Month-to-month" in seg
    assert "_local" not in json.dumps({k: v for k, v in facts.items() if not k.startswith("_")})


def test_assistant_falls_back_when_llm_fails(ws, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    monkeypatch.setattr("src.ai_assistant._call_llm", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    view = build_view(ws.clean, ws.result)
    facts = build_facts(view, ws.schema, ws.result, ws.findings, [], ws.quality, 0.5, 0.25)
    text, mode = answer_question("Summarize the model evaluation.", facts)
    assert mode == "local" and "unavailable" in text and "ROC-AUC" in text


def test_assistant_without_labels(sample_df):
    from src.workspace import build_workspace
    w = build_workspace(sample_df.drop(columns=["churn"]), "x", "auto")
    view = build_view(w.clean, None)
    facts = build_facts(view, w.schema, None, w.findings, [], w.quality, 0.5, 0.25)
    text, _ = answer_question("Which customers are most at risk?", facts)
    assert "unavailable" in text
