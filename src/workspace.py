"""Builds everything the UI needs from one uploaded (or sample) DataFrame."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .analysis import generate_findings
from .model import ModelResult, ModelTrainingError, train_models
from .preprocessing import Schema, assess_quality, clean_data, infer_schema, resolve_target


@dataclass
class Workspace:
    name: str
    raw: pd.DataFrame
    clean: pd.DataFrame
    clean_log: list[str]
    quality: dict
    schema: Schema
    result: ModelResult | None
    training_error: str | None
    findings: list[dict] = field(default_factory=list)

    @property
    def has_model(self) -> bool:
        return self.result is not None


def build_workspace(raw: pd.DataFrame, name: str, target_choice: str | None = "auto") -> Workspace:
    """Quality check -> clean (copy) -> schema -> train (if labelled). The raw upload is never changed."""
    quality = assess_quality(raw)
    target = resolve_target(raw, target_choice)
    clean, log = clean_data(raw, target)
    schema = infer_schema(clean, target, requested_target=None if target_choice in ("auto", None) else target_choice)
    result, error = None, None
    if schema.has_labels:
        try:
            result = train_models(clean, schema)
        except ModelTrainingError as exc:
            error = str(exc)
    else:
        error = schema.label_message or "No churn label column was found, so supervised training is unavailable."
    ws = Workspace(name=name, raw=raw, clean=clean, clean_log=log, quality=quality, schema=schema,
                   result=result, training_error=error)
    ws.findings = generate_findings(clean, schema, result)
    return ws
