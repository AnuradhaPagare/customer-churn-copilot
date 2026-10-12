"""Markdown report: evaluation split, model, metrics, findings and hypotheses."""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from .model import ModelResult


def _pct(v) -> str:
    return "n/a" if v is None else f"{v:.1%}"


def build_report_markdown(dataset_name: str, quality: dict, schema, result: ModelResult | None,
                          findings: list[dict], recommendations: list[dict], risk_counts: dict | None = None,
                          revenue: dict | None = None, high: float = 0.5, medium: float = 0.25) -> str:
    L: list[str] = []
    L += ["# Customer Churn & Retention Report", "",
          f"- Dataset: **{dataset_name}**",
          f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
          f"- Rows: {quality['rows']:,} | Columns: {quality['columns']} | Data-quality score: {quality['quality_score']}/100 "
          "(heuristic)", ""]
    L += ["> Predictions are statistical estimates, not certainties. Feature importance and segment differences "
          "describe associations in this dataset and do not prove causation. Recommendations are hypotheses to test.", ""]

    L += ["## Data quality", "",
          f"- Missing cells: {quality['missing_cells']:,} ({quality['missing_pct']}%)",
          f"- Duplicate rows: {quality['duplicate_rows']:,}",
          f"- Type issues: {len(quality['type_issues'])} | Unusual-value flags: {len(quality['unusual_values'])}", ""]

    if result is None:
        L += ["## Model", "", "Supervised training was not available for this dataset "
              + (f"({schema.label_message})" if schema.label_message else "(no churn label column)")
              + ". The findings below are descriptive.", ""]
    else:
        si, m = result.split_info, result.metrics
        L += ["## Model and evaluation split", "",
              f"- **Selected model:** {result.selected_name}",
              f"- **Evaluation split:** {si['method']} - {si['n_train']:,} training / {si['n_test']:,} test customers "
              f"(test_size={si['test_size']}, random_state={si['random_state']})",
              f"- Churn rate: train {_pct(si['train_churn_rate'])}, test {_pct(si['test_churn_rate'])}",
              f"- Model selection: {si['selection_rule']} ({si['cv_folds']}-fold CV)",
              f"- Decision threshold for precision/recall/F1: {m['threshold']:.3f} (chosen on cross-validated "
              "training predictions, not on the test set)", "",
              "### Held-out test-set metrics (selected model)", "",
              "| Metric | Value |", "|---|---|",
              f"| ROC-AUC | {m['roc_auc']:.3f} |" if m["roc_auc"] is not None else "| ROC-AUC | n/a |",
              f"| Precision | {m['precision']:.3f} |", f"| Recall | {m['recall']:.3f} |", f"| F1-score | {m['f1']:.3f} |",
              f"| Accuracy | {m['accuracy']:.3f} |", "",
              "Confusion matrix (rows = actual, columns = predicted; order: retained, churned):", "",
              f"|  | Pred. retained | Pred. churned |", "|---|---|---|",
              f"| Actual retained | {m['confusion_matrix'][0][0]} | {m['confusion_matrix'][0][1]} |",
              f"| Actual churned | {m['confusion_matrix'][1][0]} | {m['confusion_matrix'][1][1]} |", "",
              "### Model comparison", "", _md_table(result.comparison.drop(columns=["Selected"]).round(3)), ""]
        L += [f"Risk bands: High ≥ {high:.0%} estimated probability, Medium ≥ {medium:.0%}, otherwise Low "
              "(adjustable, not calibrated business thresholds).", ""]
        if risk_counts:
            L += ["Risk distribution: " + ", ".join(f"{k}: {v:,}" for k, v in risk_counts.items()), ""]
        if revenue:
            L += [f"Estimated monthly charges associated with high-risk customers: ${revenue['high_risk_monthly_revenue']:,.0f} "
                  f"({revenue['high_risk_share']:.1%} of total). This is an estimate of exposure, not a forecast of lost revenue.", ""]

    L += ["## Key findings", ""] + [f"- **{f['title']}** - {f['detail']}" for f in findings] + [""]
    if recommendations:
        L += ["## Retention hypotheses to test", ""]
        for r in recommendations:
            L += [f"### {r['title']}", f"- Segment: {r['segment']}", f"- Evidence: {r['evidence']}",
                  f"- Hypothesis: {r['hypothesis']}", f"- How to test: {r['test']}", ""]
    L += ["## Limitations", "",
          "- Models were trained on one dataset snapshot; performance may differ on new customers or periods.",
          "- Probabilities are not guaranteed to be calibrated.",
          "- No intervention outcome is guaranteed; use controlled experiments to evaluate retention actions.", ""]
    return "\n".join(L)


def _md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.to_numpy()]
    return "\n".join([head, sep, *body])
