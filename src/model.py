"""Model training, validation-based selection, evaluation and explanations.

Evaluation protocol (no leakage, no inflated numbers)
1. Stratified train/test split. The test set is not touched until a model is chosen.
2. Stratified K-fold cross-validation on the TRAINING portion only produces
   validation metrics, out-of-fold probabilities and a decision threshold.
3. The model is selected from validation ROC-AUC (ties go to the simpler
   Logistic Regression).
4. Test metrics are computed once, on held-out customers the model never saw.
5. Customers in the training portion are scored with out-of-fold probabilities,
   so no customer is scored by a model that was trained on their own label.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from . import config
from .preprocessing import Schema, build_preprocessor, prepare_features

LR_NAME = "Logistic Regression"
RF_NAME = "Random Forest"
AUC_TIE_TOLERANCE = 0.005


class ModelTrainingError(Exception):
    """Raised with a user-friendly message when supervised training is not possible."""


@dataclass
class ModelResult:
    selected_name: str
    pipelines: dict[str, Pipeline]
    comparison: pd.DataFrame
    test_metrics: dict[str, dict]
    cv_metrics: dict[str, dict]
    roc_curves: dict[str, dict]
    importance: dict[str, pd.DataFrame]
    coefficients: pd.DataFrame | None
    scored: pd.DataFrame
    split_info: dict
    numeric_features: list[str]
    categorical_features: list[str]
    reference_values: dict
    thresholds: dict[str, float] = field(default_factory=dict)

    @property
    def selected(self) -> Pipeline:
        return self.pipelines[self.selected_name]

    @property
    def metrics(self) -> dict:
        return self.test_metrics[self.selected_name]


# --------------------------------------------------------------------------- helpers
def candidate_models() -> dict[str, object]:
    return {
        LR_NAME: LogisticRegression(max_iter=2000, C=1.0, random_state=config.RANDOM_STATE),
        RF_NAME: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                        random_state=config.RANDOM_STATE, n_jobs=1),
    }


def make_pipeline(estimator, numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(numeric, categorical)), ("model", clone(estimator))])


def best_f1_threshold(y_true, proba) -> float:
    """Probability cut-off that maximises F1 on the given (out-of-fold) predictions."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(np.clip(thresholds[int(np.nanargmax(f1))], 0.05, 0.95))


def compute_metrics(y_true, proba, threshold: float) -> dict:
    """Classification metrics at `threshold` plus threshold-free ROC-AUC / average precision."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    two_classes = len(np.unique(y_true)) == 2
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if two_classes else None,
        "avg_precision": float(average_precision_score(y_true, proba)) if two_classes else None,
        "confusion_matrix": cm.tolist(),   # [[TN, FP], [FN, TP]]
    }


def assign_risk(proba, high: float = config.DEFAULT_HIGH_THRESHOLD,
                medium: float = config.DEFAULT_MEDIUM_THRESHOLD) -> pd.Categorical:
    """Label probabilities Low / Medium / High using the given bands."""
    p = np.asarray(proba, dtype=float)
    labels = np.where(p >= high, "High", np.where(p >= medium, "Medium", "Low"))
    return pd.Categorical(labels, categories=config.RISK_LEVELS, ordered=True)


def _feature_names(pipe: Pipeline) -> list[str]:
    return [n.split("__", 1)[1] for n in pipe.named_steps["prep"].get_feature_names_out()]


def _group_of(feature: str, categorical: list[str]) -> str:
    for c in sorted(categorical, key=len, reverse=True):
        if feature.startswith(c + "_"):
            return c
    return feature


def _coefficient_table(pipe: Pipeline, categorical: list[str]) -> pd.DataFrame:
    names = _feature_names(pipe)
    coef = pipe.named_steps["model"].coef_.ravel()
    return pd.DataFrame({
        "feature": names,
        "group": [_group_of(n, categorical) for n in names],
        "coefficient": coef,
        "odds_ratio": np.exp(coef),
    }).sort_values("coefficient", key=np.abs, ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- training
def train_models(df: pd.DataFrame, schema: Schema) -> ModelResult:
    """Train, compare and evaluate candidate models on the cleaned labelled data."""
    if not schema.has_labels:
        raise ModelTrainingError(schema.label_message or "No churn label column was found, so supervised "
                                 "training is unavailable.")
    if not schema.features:
        raise ModelTrainingError("No usable feature columns remain after excluding identifiers and target.")
    target = schema.target
    labelled = df[target].notna()
    n_lab = int(labelled.sum())
    y_all = df.loc[labelled, target].astype(int)
    counts = y_all.value_counts()
    if n_lab < config.MIN_LABELLED_ROWS:
        raise ModelTrainingError(f"Only {n_lab} labelled rows; at least {config.MIN_LABELLED_ROWS} are needed.")
    if len(counts) < 2 or counts.min() < config.MIN_CLASS_COUNT:
        raise ModelTrainingError(f"Each class needs at least {config.MIN_CLASS_COUNT} labelled customers "
                                 f"(found {counts.to_dict()}).")

    num, cat = schema.numeric_features, schema.categorical_features
    X_all = prepare_features(df.loc[labelled], num, cat)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=config.TEST_SIZE, stratify=y_all, random_state=config.RANDOM_STATE)

    n_splits = int(max(2, min(config.CV_FOLDS, y_train.value_counts().min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)

    pipelines, cv_metrics, test_metrics, oof_store, roc_curves, importance = {}, {}, {}, {}, {}, {}
    test_proba = {}
    for name, est in candidate_models().items():
        base = make_pipeline(est, num, cat)
        oof = np.zeros(len(X_train))
        fold_auc = []
        for tr, va in cv.split(X_train, y_train):
            fold = clone(base).fit(X_train.iloc[tr], y_train.iloc[tr])
            p = fold.predict_proba(X_train.iloc[va])[:, 1]
            oof[va] = p
            fold_auc.append(roc_auc_score(y_train.iloc[va], p))
        thr = best_f1_threshold(y_train, oof)
        cv_m = compute_metrics(y_train, oof, thr)
        cv_m.update(cv_auc_mean=float(np.mean(fold_auc)), cv_auc_std=float(np.std(fold_auc)), folds=n_splits)
        cv_metrics[name], oof_store[name] = cv_m, oof

        fitted = clone(base).fit(X_train, y_train)          # fit on training data only
        pipelines[name] = fitted
        proba = fitted.predict_proba(X_test)[:, 1]
        test_proba[name] = proba
        test_metrics[name] = compute_metrics(y_test, proba, thr)  # threshold chosen WITHOUT test data
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_curves[name] = {"fpr": fpr, "tpr": tpr}
        perm = permutation_importance(fitted, X_test, y_test, scoring="roc_auc", n_repeats=5,
                                      random_state=config.RANDOM_STATE, n_jobs=1)
        importance[name] = pd.DataFrame({
            "feature": X_test.columns, "importance": perm.importances_mean, "std": perm.importances_std,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    # --- model selection on VALIDATION (cross-validation) results only
    best = max(cv_metrics, key=lambda k: cv_metrics[k]["cv_auc_mean"])
    if LR_NAME in cv_metrics and cv_metrics[best]["cv_auc_mean"] - cv_metrics[LR_NAME]["cv_auc_mean"] <= AUC_TIE_TOLERANCE:
        best = LR_NAME
    selected = best

    rows = []
    for name in pipelines:
        rows.append({
            "Model": name,
            "CV ROC-AUC": cv_metrics[name]["cv_auc_mean"],
            "CV ROC-AUC std": cv_metrics[name]["cv_auc_std"],
            "CV F1": cv_metrics[name]["f1"],
            "Test ROC-AUC": test_metrics[name]["roc_auc"],
            "Test precision": test_metrics[name]["precision"],
            "Test recall": test_metrics[name]["recall"],
            "Test F1": test_metrics[name]["f1"],
            "Threshold": test_metrics[name]["threshold"],
            "Selected": name == selected,
        })
    comparison = pd.DataFrame(rows)

    # --- score every customer without in-sample leakage
    scored = pd.DataFrame(index=df.index, columns=["churn_probability", "score_source"], dtype=object)
    scored.loc[X_train.index, "churn_probability"] = oof_store[selected]
    scored.loc[X_train.index, "score_source"] = "cross-validated (training portion)"
    scored.loc[X_test.index, "churn_probability"] = test_proba[selected]
    scored.loc[X_test.index, "score_source"] = "held-out test set"
    if (~labelled).any():
        unl = df.loc[~labelled]
        scored.loc[unl.index, "churn_probability"] = pipelines[selected].predict_proba(
            prepare_features(unl, num, cat))[:, 1]
        scored.loc[unl.index, "score_source"] = "model (row had no churn label)"
    scored["churn_probability"] = scored["churn_probability"].astype(float)

    reference = {c: float(X_train[c].median()) for c in num}
    reference.update({c: (X_train[c].mode().iloc[0] if X_train[c].notna().any() else "Missing") for c in cat})

    coefficients = _coefficient_table(pipelines[LR_NAME], cat) if LR_NAME in pipelines else None
    split_info = {
        "method": "Stratified hold-out split",
        "test_size": config.TEST_SIZE, "random_state": config.RANDOM_STATE,
        "n_labelled": n_lab, "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "train_churn_rate": float(y_train.mean()), "test_churn_rate": float(y_test.mean()),
        "cv_folds": n_splits,
        "selection_rule": f"Highest cross-validated ROC-AUC on the training portion "
                          f"(ties within {AUC_TIE_TOLERANCE} favour {LR_NAME}).",
    }
    return ModelResult(selected_name=selected, pipelines=pipelines, comparison=comparison,
                       test_metrics=test_metrics, cv_metrics=cv_metrics, roc_curves=roc_curves,
                       importance=importance, coefficients=coefficients, scored=scored,
                       split_info=split_info, numeric_features=num, categorical_features=cat,
                       reference_values=reference,
                       thresholds={m: test_metrics[m]["threshold"] for m in test_metrics})


# --------------------------------------------------------------------------- inference
def score_customers(result: ModelResult, df: pd.DataFrame) -> pd.Series:
    """Predict churn probability for NEW rows. Missing values and unseen categories are handled safely."""
    X = prepare_features(df, result.numeric_features, result.categorical_features)
    return pd.Series(result.selected.predict_proba(X)[:, 1], index=df.index, name="churn_probability")


def explain_customer(result: ModelResult, row: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """Model-agnostic 'what-if' explanation for one customer.

    For each feature, replace the customer's value with the typical value (median /
    most common) and measure how the predicted probability changes. Positive effect =
    this feature's value pushes the estimate UP relative to a typical customer.
    This describes how the MODEL behaves; it does not prove a cause of churn.
    """
    num, cat = result.numeric_features, result.categorical_features
    x = prepare_features(row.iloc[[0]], num, cat)
    feats = num + cat
    variants = pd.concat([x] * (len(feats) + 1), ignore_index=True)
    for i, f in enumerate(feats, start=1):
        variants.loc[i, f] = result.reference_values[f]
    proba = result.selected.predict_proba(variants)[:, 1]
    base = proba[0]
    out = pd.DataFrame({
        "feature": feats,
        "customer_value": [x.iloc[0][f] for f in feats],
        "typical_value": [result.reference_values[f] for f in feats],
        "effect_pp": (base - proba[1:]) * 100,
    })
    out = out.reindex(out["effect_pp"].abs().sort_values(ascending=False).index).head(top_n)
    return out.reset_index(drop=True)
