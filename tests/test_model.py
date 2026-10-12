import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_loader import DataLoadError, generate_sample_data, load_dataframe
from src.model import ModelTrainingError, assign_risk, compute_metrics, explain_customer, score_customers, train_models
from src.preprocessing import clean_data, infer_schema, resolve_target
from src.workspace import build_workspace


# ---- sample dataset generation
def test_sample_dataset_shape_and_columns():
    df = generate_sample_data(600, seed=1)
    assert len(df) >= 500
    for col in ["tenure_months", "monthly_charges", "contract_type", "payment_method", "support_calls",
                "service_usage_hours", "total_charges", "churn"]:
        assert col in df.columns
    assert df["customer_id"].is_unique
    assert 0.15 < (df["churn"] == "Yes").mean() < 0.40


def test_sample_has_plausible_relationships():
    df = generate_sample_data(3000, seed=3)
    rate = df.assign(y=df.churn.eq("Yes")).groupby("contract_type")["y"].mean()
    assert rate["Month-to-month"] > rate["Two year"] + 0.10
    corr = np.corrcoef(df.tenure_months, df.churn.eq("Yes"))[0, 1]
    assert corr < 0


def test_sample_is_deterministic():
    pd.testing.assert_frame_equal(generate_sample_data(200, seed=5), generate_sample_data(200, seed=5))


# ---- data loading
def test_load_csv_and_excel(tmp_path, sample_df):
    csv = tmp_path / "d.csv"
    sample_df.head(50).to_csv(csv, index=False)
    assert len(load_dataframe(csv)) == 50
    xlsx = tmp_path / "d.xlsx"
    sample_df.head(30).to_excel(xlsx, index=False)
    assert len(load_dataframe(xlsx)) == 30


def test_load_errors_are_friendly(tmp_path):
    bad = tmp_path / "x.pdf"
    bad.write_bytes(b"123")
    with pytest.raises(DataLoadError, match="Unsupported"):
        load_dataframe(bad)
    empty = tmp_path / "e.csv"
    empty.write_text("")
    with pytest.raises(DataLoadError):
        load_dataframe(empty)
    with pytest.raises(DataLoadError, match="not found"):
        load_dataframe(tmp_path / "missing.csv")


# ---- cleaning / missing values
def test_cleaning_does_not_modify_original(sample_df):
    messy = sample_df.head(100).copy()
    messy["total_charges"] = messy["total_charges"].astype(str)   # numbers stored as text
    messy.loc[0, "total_charges"] = " "
    messy["monthly_charges"] = messy["monthly_charges"].astype(str)
    messy.loc[1, "support_calls"] = -5
    messy = pd.concat([messy, messy.iloc[[2]]], ignore_index=True)  # duplicate row
    before = messy.copy(deep=True)
    clean, log = clean_data(messy, "churn")
    pd.testing.assert_frame_equal(messy, before)           # original untouched
    assert pd.api.types.is_numeric_dtype(clean["monthly_charges"])
    assert pd.api.types.is_numeric_dtype(clean["total_charges"]) and pd.isna(clean.loc[0, "total_charges"])
    assert pd.isna(clean.loc[1, "support_calls"])           # negative -> missing
    assert len(clean) == len(messy) - 1                     # duplicate removed
    assert set(clean["churn"].dropna().unique()) <= {0.0, 1.0}
    assert any("duplicate" in l.lower() for l in log)


def test_missing_values_and_unseen_categories_are_handled(ws):
    new = ws.clean.head(5).copy()
    new["contract_type"] = "Totally New Plan"
    new.loc[new.index[0], ["monthly_charges", "tenure_months"]] = np.nan
    new = new.drop(columns=["support_calls"])               # even a missing column
    p = score_customers(ws.result, new)
    assert p.notna().all() and p.between(0, 1).all()


# ---- training, evaluation, leakage
def test_training_selects_model_and_metrics(ws):
    r = ws.result
    assert r.selected_name in r.pipelines
    m = r.metrics
    for k in ["precision", "recall", "f1", "roc_auc", "confusion_matrix"]:
        assert m[k] is not None
    assert 0.6 < m["roc_auc"] <= 1.0
    assert np.array(m["confusion_matrix"]).sum() == r.split_info["n_test"]
    assert {"Logistic Regression", "Random Forest"} <= set(r.pipelines)


def test_split_is_stratified(ws):
    si = ws.result.split_info
    assert si["n_train"] + si["n_test"] == si["n_labelled"]
    assert abs(si["train_churn_rate"] - si["test_churn_rate"]) < 0.02


def test_no_leakage_columns_in_features(ws):
    feats = ws.schema.features
    assert "churn" not in feats and "customer_id" not in feats
    assert "churn" not in ws.result.numeric_features + ws.result.categorical_features


def test_leaky_columns_are_excluded(sample_df):
    df = sample_df.copy()
    df["churn_reason"] = np.where(df.churn == "Yes", "Price", "None")   # name-based leak
    df["retained_flag"] = (df.churn == "No").astype(int)               # perfect predictor
    clean, _ = clean_data(df, "churn")
    schema = infer_schema(clean, "churn")
    assert "churn_reason" in schema.excluded and "retained_flag" in schema.excluded
    assert "retained_flag" not in schema.features


def test_prediction_probabilities_valid(ws):
    s = ws.result.scored["churn_probability"]
    assert s.notna().all() and s.between(0, 1).all()
    assert set(ws.result.scored["score_source"]) == {"cross-validated (training portion)", "held-out test set"}


def test_test_metrics_not_training_metrics(ws):
    r = ws.result
    assert r.metrics["n"] == r.split_info["n_test"]
    assert "Train ROC-AUC" not in r.comparison.columns


def test_compute_metrics_perfect_and_single_class():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
    assert m["precision"] == m["recall"] == m["f1"] == m["roc_auc"] == 1.0
    assert compute_metrics([1, 1], [0.6, 0.7], 0.5)["roc_auc"] is None


def test_risk_categories():
    cat = assign_risk([0.1, 0.3, 0.7], high=0.5, medium=0.25)
    assert list(cat) == ["Low", "Medium", "High"]


def test_explain_customer(ws):
    e = explain_customer(ws.result, ws.clean.head(1))
    assert {"feature", "effect_pp"} <= set(e.columns) and len(e) > 0


# ---- missing churn labels
def test_no_labels_means_no_model(sample_df):
    ws2 = build_workspace(sample_df.drop(columns=["churn"]), "nolabel", "auto")
    assert ws2.result is None and not ws2.schema.has_labels
    assert "no churn label" in ws2.training_error.lower()
    with pytest.raises(ModelTrainingError):
        train_models(ws2.clean, ws2.schema)


def test_too_few_rows_refuses_training(sample_df):
    ws2 = build_workspace(sample_df.head(30), "tiny", "auto")
    assert ws2.result is None and "labelled rows" in ws2.training_error


def test_explicit_none_target_is_descriptive(sample_df):
    assert resolve_target(sample_df, None) is None
