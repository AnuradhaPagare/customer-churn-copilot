"""SQLite persistence for model runs and their customer-level predictions."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    n_train INTEGER, n_test INTEGER, test_size REAL,
    roc_auc REAL, precision_score REAL, recall REAL, f1 REAL, threshold REAL,
    metrics_json TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    run_id INTEGER NOT NULL REFERENCES model_runs(id) ON DELETE CASCADE,
    customer_ref TEXT NOT NULL,
    churn_probability REAL,
    risk_category TEXT,
    score_source TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_run ON predictions(run_id);
"""


def _connect(path: Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def save_run(path: Path, dataset_name: str, model_name: str, metrics: dict, split_info: dict,
             scored: pd.DataFrame, id_series: pd.Series | None = None) -> int:
    """Store one run (metrics + predictions). All values go through bound parameters."""
    refs = id_series.astype(str) if id_series is not None else pd.Series(scored.index.astype(str), index=scored.index)
    rows = [
        (str(refs.loc[i]), None if pd.isna(p) else float(p), None if pd.isna(r) else str(r), str(s))
        for i, p, r, s in zip(scored.index, scored["churn_probability"], scored["risk_category"], scored["score_source"])
    ]
    with closing(_connect(path)) as conn:
        with conn:
            cur = conn.execute(
                "INSERT INTO model_runs (created_at, dataset_name, model_name, n_train, n_test, test_size, roc_auc, "
                "precision_score, recall, f1, threshold, metrics_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), dataset_name, model_name,
                 split_info.get("n_train"), split_info.get("n_test"), split_info.get("test_size"),
                 metrics.get("roc_auc"), metrics.get("precision"), metrics.get("recall"), metrics.get("f1"),
                 metrics.get("threshold"), json.dumps(metrics)))
            run_id = int(cur.lastrowid)
            conn.executemany("INSERT INTO predictions VALUES (?,?,?,?,?)", [(run_id, *r) for r in rows])
    return run_id


def list_runs(path: Path, limit: int = 20) -> pd.DataFrame:
    with closing(_connect(path)) as conn:
        return pd.read_sql_query(
            "SELECT id, created_at, dataset_name, model_name, n_train, n_test, roc_auc, precision_score AS precision, "
            "recall, f1 FROM model_runs ORDER BY id DESC LIMIT ?", conn, params=(int(limit),))


def load_predictions(path: Path, run_id: int) -> pd.DataFrame:
    with closing(_connect(path)) as conn:
        return pd.read_sql_query("SELECT customer_ref, churn_probability, risk_category, score_source "
                                 "FROM predictions WHERE run_id = ? ORDER BY churn_probability DESC", conn,
                                 params=(int(run_id),))
