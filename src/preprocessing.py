"""Schema detection, data-quality assessment, cleaning and sklearn preprocessing.

Design rules
- The uploaded DataFrame is never modified; cleaning works on a copy.
- Imputation and scaling happen *inside* the sklearn pipeline, so they are fit on
  training data only (no leakage from the test set).
- The target, identifiers and columns that look derived from the outcome are
  never used as model features.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "?", "#n/a", "--"}
TARGET_NAMES = (
    "churn", "churned", "is_churn", "is_churned", "churn_label", "churn_flag",
    "customer_churn", "exited", "attrition", "left", "cancelled", "canceled",
)
POSITIVE_LABELS = {"yes", "y", "true", "t", "1", "churned", "churn", "left", "exited",
                   "cancelled", "canceled", "attrited", "attrited customer", "lost"}
NEGATIVE_LABELS = {"no", "n", "false", "f", "0", "retained", "stayed", "active",
                   "existing customer", "current", "kept"}
ID_NAMES = {"id", "customer_id", "customerid", "cust_id", "client_id", "user_id",
            "account_id", "account_number", "subscriber_id"}
LEAKAGE_NAME_RE = re.compile(r"(churn|cancel|terminat|attrit|deactivat|retention_offer|win_?back|exit)")
NONNEG_RE = re.compile(r"(tenure|charge|fee|amount|call|ticket|usage|hours|minutes|num_|count|months|mrr|revenue|spend|bill)")
ROLE_PATTERNS = {
    "tenure": ["tenure", "months_active", "account_age", "months_as_customer", "lifetime_months"],
    "monthly": ["monthly_charge", "monthlycharge", "monthly_fee", "monthly_revenue", "mrr", "monthly_bill",
                "monthly_spend", "monthly_cost"],
    "total": ["total_charge", "totalcharge", "total_revenue", "total_spend", "total_paid", "lifetime_value"],
    "contract": ["contract"],
    "payment": ["payment"],
    "support": ["support", "complaint", "ticket", "service_call"],
    "usage": ["usage", "hours", "minutes", "data_gb", "logins"],
}
NUMERIC_ROLES = {"tenure", "monthly", "total", "support", "usage"}
MAX_CATEGORIES = 50


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _is_text(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return False
    return s.dtype == object or pd.api.types.is_string_dtype(s)


def _clean_text(s: pd.Series) -> pd.Series:
    """Strip whitespace and map blank / NA-like tokens to NaN (object dtype)."""
    t = s.astype("string").str.strip()
    blank = t.str.lower().isin(MISSING_TOKENS).fillna(True)
    return t.astype(object).where(~blank.to_numpy(dtype=bool), np.nan)


def _to_numeric_text(s: pd.Series) -> pd.Series:
    t = s.astype("string").str.replace(r"[,$€£%\s]", "", regex=True)
    return pd.to_numeric(t, errors="coerce").astype("float64")


def _numeric_like_ratio(s: pd.Series) -> float:
    cleaned = _clean_text(s)
    non_null = cleaned.notna().sum()
    if non_null == 0:
        return 0.0
    return float(_to_numeric_text(cleaned).notna().sum() / non_null)


def _missing_mask(s: pd.Series) -> pd.Series:
    if _is_text(s):
        return _clean_text(s).isna()
    return s.isna()


# --------------------------------------------------------------------------- target
def normalize_target(series: pd.Series) -> tuple[pd.Series | None, str | None]:
    """Map a churn column to 0/1 floats (NaN where unknown). Returns (series, error)."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float), None
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce").astype(float)
        if set(vals.dropna().unique()) <= {0.0, 1.0}:
            return vals, None
        return None, "The column is numeric but not a 0/1 flag."

    def conv(v):
        if not isinstance(v, str):
            return np.nan
        k = v.strip().lower()
        if k in POSITIVE_LABELS:
            return 1.0
        if k in NEGATIVE_LABELS:
            return 0.0
        return -1.0  # unmappable marker

    mapped = _clean_text(series).map(conv).astype(float)
    non_missing = int(_clean_text(series).notna().sum())
    unmappable = int((mapped == -1.0).sum())
    if non_missing == 0:
        return None, "The column is empty."
    if unmappable / non_missing > 0.05:
        examples = sorted({str(v) for v in series[mapped == -1.0].dropna().unique()})[:4]
        return None, f"Values such as {examples} cannot be read as Yes/No churn labels."
    return mapped.where(mapped != -1.0, np.nan), None


def detect_target_candidates(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if _norm(c) in TARGET_NAMES]


def resolve_target(df: pd.DataFrame, choice: str | None = "auto") -> str | None:
    """choice: 'auto' (name-based detection), None / 'none' (descriptive only) or a column name."""
    if choice is None or str(choice).lower() == "none":
        return None
    if choice != "auto":
        return choice if choice in df.columns else None
    for c in detect_target_candidates(df):
        converted, _ = normalize_target(df[c])
        if converted is not None and converted.nunique() == 2:
            return c
    return None


# --------------------------------------------------------------------------- quality
def detect_id_column(df: pd.DataFrame) -> str | None:
    n = len(df)
    if n == 0:
        return None
    for c in df.columns:
        key = _norm(c)
        if (key in ID_NAMES or key.endswith("_id")) and df[c].nunique(dropna=True) >= 0.9 * n:
            return c
    return None


def assess_quality(df: pd.DataFrame) -> dict:
    """Data-quality metrics for the *raw* upload (read-only)."""
    n_rows, n_cols = df.shape
    rows = []
    type_issues: list[dict] = []
    unusual: list[dict] = []
    for c in df.columns:
        s = df[c]
        miss = int(_missing_mask(s).sum())
        rows.append({"column": c, "dtype": str(s.dtype), "missing": miss,
                     "missing_pct": round(100 * miss / n_rows, 2) if n_rows else 0.0,
                     "unique": int(s.nunique(dropna=True))})
        if _is_text(s):
            ratio = _numeric_like_ratio(s)
            if ratio >= 0.9:
                type_issues.append({"column": c, "issue": "Numbers stored as text",
                                    "detail": f"{ratio:.0%} of values parse as numbers"})
            elif 0.1 <= ratio < 0.9:
                type_issues.append({"column": c, "issue": "Mixed numbers and text",
                                    "detail": f"{ratio:.0%} of values parse as numbers"})
            if "date" in _norm(c):
                type_issues.append({"column": c, "issue": "Dates stored as text", "detail": "Not used as a feature"})
            if s.nunique(dropna=True) > MAX_CATEGORIES and detect_id_column(df[[c]]) is None:
                unusual.append({"column": c, "issue": "Very many distinct text values",
                                "count": int(s.nunique(dropna=True))})
        else:
            num = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
            if NONNEG_RE.search(_norm(c)) and not pd.api.types.is_bool_dtype(s):
                neg = int((num < 0).sum())
                if neg:
                    unusual.append({"column": c, "issue": "Negative values where not expected", "count": neg})
            vals = num.dropna()
            if len(vals) >= 20:
                q1, q3 = vals.quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr > 0:
                    out = int(((vals < q1 - 3 * iqr) | (vals > q3 + 3 * iqr)).sum())
                    if out:
                        unusual.append({"column": c, "issue": "Extreme outliers (beyond 3×IQR)", "count": out})
        if s.nunique(dropna=True) <= 1 and n_rows > 1:
            unusual.append({"column": c, "issue": "Constant column", "count": n_rows})

    id_col = detect_id_column(df)
    dup_rows = int(df.duplicated().sum())
    dup_ids = int(df[id_col].duplicated().sum()) if id_col else 0
    missing_cells = int(sum(r["missing"] for r in rows))
    missing_pct = 100 * missing_cells / max(1, n_rows * n_cols)
    unusual_total = sum(u["count"] for u in unusual if "outliers" in u["issue"] or "Negative" in u["issue"])
    score = 100 - 2 * missing_pct - 2 * (100 * dup_rows / max(1, n_rows)) - min(20, 4 * len(type_issues)) \
        - min(15, 100 * unusual_total / max(1, n_rows))
    return {
        "rows": n_rows, "columns": n_cols,
        "missing_cells": missing_cells, "missing_pct": round(missing_pct, 2),
        "missing_by_column": pd.DataFrame(rows),
        "duplicate_rows": dup_rows, "duplicate_ids": dup_ids, "id_column": id_col,
        "type_issues": type_issues, "unusual_values": unusual,
        "quality_score": int(max(0, min(100, round(score)))),
    }


# --------------------------------------------------------------------------- cleaning
def clean_data(raw: pd.DataFrame, target: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Return (cleaned copy, log). The input frame is never modified.

    No imputation happens here (that is done inside the model pipeline, on training
    data only). Cleaning = strip text, map NA tokens to NaN, convert numbers stored
    as text, blank impossible negatives, drop exact duplicate rows, encode target 0/1.
    """
    clean = raw.copy(deep=True)
    log: list[str] = []

    for c in list(clean.columns):
        s = clean[c]
        if c == target:
            continue
        if _is_text(s):
            text = _clean_text(s)
            blanks = int(text.isna().sum() - s.isna().sum())
            if _numeric_like_ratio(text) >= 0.9:
                num = _to_numeric_text(text)
                bad = int(num.isna().sum() - text.isna().sum())
                clean[c] = num
                log.append(f"'{c}': converted from text to numeric"
                           + (f"; {bad} unparseable value(s) set to missing" if bad > 0 else ""))
            else:
                clean[c] = text
                if blanks > 0:
                    log.append(f"'{c}': {blanks} blank / NA-like text value(s) set to missing")
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            num = s.astype(float).replace([np.inf, -np.inf], np.nan)
            n_inf = int(np.isinf(s.astype(float)).sum())
            if n_inf:
                log.append(f"'{c}': {n_inf} infinite value(s) set to missing")
            if NONNEG_RE.search(_norm(c)):
                neg = int((num < 0).sum())
                if neg:
                    num = num.mask(num < 0)
                    log.append(f"'{c}': {neg} negative value(s) set to missing")
            clean[c] = num

    before = len(clean)
    clean = clean.drop_duplicates().reset_index(drop=True)
    if before - len(clean):
        log.append(f"Removed {before - len(clean)} exact duplicate row(s)")

    if target and target in clean.columns:
        y, err = normalize_target(clean[target])
        if y is not None:
            clean[target] = y
            log.append(f"Target '{target}' encoded as 1 = churned, 0 = retained "
                       f"({int(y.isna().sum())} row(s) without a usable label)")
        else:
            log.append(f"Target '{target}' could not be used: {err}")
    if not log:
        log.append("No cleaning changes were needed.")
    return clean, log


# --------------------------------------------------------------------------- schema
@dataclass
class Schema:
    target: str | None
    id_col: str | None
    numeric_features: list[str]
    categorical_features: list[str]
    excluded: dict[str, str]
    roles: dict[str, str]
    label_message: str | None = None
    requested_target: str | None = None

    @property
    def has_labels(self) -> bool:
        return self.target is not None

    @property
    def features(self) -> list[str]:
        return self.numeric_features + self.categorical_features


def infer_roles(df: pd.DataFrame) -> dict[str, str]:
    roles: dict[str, str] = {}
    for role, patterns in ROLE_PATTERNS.items():
        for c in df.columns:
            if role in roles.values():
                break
            if _norm(c) in (_norm(v) for v in roles.values()):
                continue
            name = _norm(c)
            if any(p in name for p in patterns):
                if role in NUMERIC_ROLES and not pd.api.types.is_numeric_dtype(df[c]):
                    continue
                roles[role] = c
                break
    return roles


def _perfect_predictor(s: pd.Series, y: pd.Series) -> bool:
    """True if a single column (almost) determines the label by itself -> likely leakage."""
    mask = s.notna() & y.notna()
    if mask.sum() < 30 or y[mask].nunique() < 2:
        return False
    try:
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            auc = roc_auc_score(y[mask], s[mask])
            return bool(auc >= 0.99 or auc <= 0.01)
        if s[mask].nunique() <= 30 and s[mask].nunique() < 0.1 * mask.sum():
            purity = y[mask].groupby(s[mask].astype(str)).agg(lambda v: max(v.mean(), 1 - v.mean()))
            counts = s[mask].astype(str).value_counts()
            return bool((purity * counts.reindex(purity.index)).sum() / mask.sum() >= 0.995)
    except Exception:
        return False
    return False


def infer_schema(df: pd.DataFrame, target: str | None = None, requested_target: str | None = None) -> Schema:
    """Decide which columns are features. Excluded columns come with a reason."""
    excluded: dict[str, str] = {}
    label_message = None
    y = None
    if target is not None:
        if target in df.columns and df[target].dropna().isin([0.0, 1.0]).all() and df[target].dropna().nunique() == 2:
            y = df[target]
            excluded[target] = "Target label (never used as a feature)"
        else:
            label_message = (f"Column '{target}' is not a usable binary churn label "
                             "(it needs both churned and retained customers).")
            target = None

    id_col = detect_id_column(df)
    if id_col:
        excluded[id_col] = "Customer identifier (not predictive)"

    n = len(df)
    numeric, categorical = [], []
    for c in df.columns:
        if c in excluded:
            continue
        s = df[c]
        nun = s.nunique(dropna=True)
        name = _norm(c)
        if s.isna().all():
            excluded[c] = "All values missing"
        elif nun <= 1:
            excluded[c] = "Constant column"
        elif pd.api.types.is_datetime64_any_dtype(s) or "date" in name:
            excluded[c] = "Date column (excluded to avoid using future information)"
        elif LEAKAGE_NAME_RE.search(name):
            excluded[c] = "Name suggests it is derived from the churn outcome (possible leakage)"
        elif y is not None and _perfect_predictor(s, y):
            excluded[c] = "Almost perfectly predicts the label on its own (possible leakage)"
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            numeric.append(c)
        elif nun > MAX_CATEGORIES or (n >= 50 and nun > 0.5 * n):
            excluded[c] = "Too many distinct text values / identifier-like"
        else:
            categorical.append(c)

    return Schema(target=target, id_col=id_col, numeric_features=numeric, categorical_features=categorical,
                  excluded=excluded, roles=infer_roles(df), label_message=label_message,
                  requested_target=requested_target)


# --------------------------------------------------------------------------- model inputs
def prepare_features(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """Build a model-ready frame with exactly the training columns.

    Missing columns become NaN, numerics are coerced, categoricals become object
    dtype with NaN for missing. Safe to call on new, unseen data.
    """
    out = pd.DataFrame(index=df.index)
    for c in numeric:
        col = pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)
        out[c] = col.astype(float).replace([np.inf, -np.inf], np.nan)
    for c in categorical:
        if c in df.columns:
            col = df[c]
            col = col.astype("string").str.strip().astype(object) if col.notna().any() else col.astype(object)
            col = col.where(col.notna(), np.nan)
        else:
            col = pd.Series(np.nan, index=df.index, dtype=object)
        out[c] = col
    return out


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Median-impute + scale numerics; constant-impute + one-hot categoricals.

    handle_unknown='ignore' means categories unseen in training become all-zero
    rows instead of raising errors at prediction time.
    """
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    if not transformers:
        raise ValueError("No usable feature columns were found.")
    return ColumnTransformer(transformers, remainder="drop")
