# AI-Powered Customer Churn & Retention Predictor

An interactive analytics app that cleans customer data, predicts churn with machine learning, shows which
attributes are associated with churn, and turns the results into **retention hypotheses to test**.
Dark-themed Streamlit dashboard; works fully offline with a built-in synthetic dataset.

> **Predictions are estimates, not certainties.** Feature importance and segment differences describe
> *associations*, not causes. Retention recommendations are *hypotheses*, never guaranteed fixes.

## Problem

Winning back a churned customer costs more than keeping one, but teams rarely know *who* is at risk or
*what to try first*. This project scores every customer, explains the model's behaviour, estimates revenue
exposure, and proposes experiments, with the evidence shown next to each suggestion.

## Quick start (Windows)

```
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

macOS / Linux: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && streamlit run app.py`

The app opens on a generated 1,500-customer sample (`data/sample_customers.csv`, created automatically if missing).
Upload your own CSV/Excel from the sidebar; include a churn column (`Yes/No`, `1/0`, `True/False`).

Run the tests: `python -m pytest -q`

## Features

| Area | What it does |
|---|---|
| Data input | CSV / .xlsx upload, friendly errors, synthetic sample with realistic churn drivers |
| Cleaning & EDA | Missing values, duplicates, text-as-number columns, negatives, outliers, quality score. Works on a copy; the upload is never modified |
| Modelling | Logistic Regression + Random Forest, stratified split, sklearn pipelines, cross-validated model selection |
| Dashboard | KPIs, churn by contract / tenure / charges, segment comparison, heatmaps, filters, customer risk table |
| Risk inspector | Per-customer probability, Low / Medium / High band, top model factors |
| AI assistant | Local deterministic answers by default; optional LLM phrasing from aggregate facts only |
| Hypotheses | Evidence-based segments (short tenure + high charges, month-to-month, frequent support calls, ...) each with a test plan |
| Export | Scored customers CSV, Markdown report (model, split, metrics, findings), metrics JSON, SQLite run history |

Pages: **Overview, Customer Analytics, Churn Predictions, AI Insights, Model Performance, Data Explorer**.
It adapts to your columns. Without a churn label it shows descriptive analysis only and explains that
supervised training is unavailable. It never fabricates predictions.

## Architecture

```
app.py                 entry: page config, navigation, sidebar, dataset + cached workspace
assets/style.css       dark theme CSS (palette injected from src/theme.py)
.streamlit/config.toml native dark theme + dataframe colours
src/config.py          paths, constants, risk-band defaults
src/data_loader.py     sample generator, CSV/Excel loading
src/preprocessing.py   quality report, cleaning, schema + leakage guards, sklearn preprocessor
src/model.py           training, CV selection, metrics, scoring, explanations
src/analysis.py        KPIs, segments, revenue estimate, findings, hypotheses, CSV export
src/ai_assistant.py    local insights engine + optional LLM
src/workspace.py       raw -> quality -> clean -> schema -> model
src/database.py        SQLite run history (parameterised queries)
src/report.py          Markdown report
src/theme.py, charts.py, dashboard.py   UI components, Plotly figures, pages
tests/                 pytest suite (models, analysis, assistant, DB, UI smoke tests)
```

## Model evaluation and leakage protection

1. The label, ID-like columns, date columns, columns whose names suggest the outcome (`churn_reason`,
   `cancel_date`, ...) and any column that almost perfectly predicts the label are excluded from features.
2. A **stratified 80/20 split** is made first. The 20% test set is untouched until a model is chosen.
3. Imputation, scaling and one-hot encoding live **inside** the sklearn pipeline, so they are fit on training data only.
   Unseen categories and missing values are handled at prediction time.
4. Models are compared with **5-fold stratified cross-validation on the training portion**; the model is selected
   from validation ROC-AUC (ties within 0.005 favour Logistic Regression for interpretability).
5. The precision/recall/F1 cut-off is chosen from cross-validated training predictions, never from the test set.
6. Reported metrics (precision, recall, F1, ROC-AUC, confusion matrix) are **test-set** values. Training-set
   performance is not reported.
7. Customers in the training portion are scored with out-of-fold probabilities, so nobody is scored by a model that
   saw their own label.

On the bundled synthetic sample the selected model typically reaches a test ROC-AUC of roughly 0.75 to 0.80
(values vary with data and seed; the app shows exact numbers). Because the sample is synthetic, this demonstrates
the workflow, not real-world performance.

## Optional LLM assistant

Copy `.env.example` to `.env` and set a key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`, plus `LLM_PROVIDER`).
Only **aggregate statistics** are sent, never customer rows. The model is instructed to use only those facts,
avoid causal claims and guarantees. Its output is rendered as text and **never executed**. If the call fails, the
app falls back to local insights. Without a key, everything still works.

## Limitations

- Synthetic sample data: relationships are built in; real data will be noisier.
- One snapshot, no time dimension: no guarantee a model generalises to later periods.
- Probabilities are not calibrated; risk bands (High ≥ 50%, Medium ≥ 25% by default) are adjustable conventions.
- Revenue "at risk" is the monthly charges of high-risk customers: an exposure estimate, not a forecast of loss.
- Column-role detection is name-based (e.g. `tenure`, `contract`, `monthly_charges`); unusual names still get
  modelled but may lose role-specific charts and hypotheses. Currency is displayed as `$`.
- The final model is trained on the 80% training portion only (not refit on all data).
- Hypotheses reflect observed association; test them with controlled experiments before changing pricing or contracts.

## Future improvements

Probability calibration, SHAP explanations, time-based validation, gradient boosting, uplift modelling for
intervention targeting, user-defined column mapping, authentication and multi-user storage.
