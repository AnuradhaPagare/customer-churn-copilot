"""Data input: realistic sample-data generation and safe CSV / Excel loading."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


class DataLoadError(Exception):
    """Raised with a user-friendly message when a file cannot be loaded."""


CONTRACTS = ["Month-to-month", "One year", "Two year"]
PAYMENTS = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
INTERNET = ["Fiber optic", "DSL", "No internet"]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def generate_sample_data(n_rows: int = config.SAMPLE_ROWS, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic telecom-style customer table with plausible churn drivers.

    Churn is drawn from a logistic model of *customer attributes only*
    (contract, tenure, price, support calls, payment method, usage, ...).
    No future information is included as a column. A small amount of missing
    data is injected so the cleaning workflow has something real to do.
    The data is synthetic: it demonstrates the workflow, not real market behaviour.
    """
    if n_rows < 50:
        raise ValueError("n_rows must be at least 50")
    rng = np.random.default_rng(seed)
    n = int(n_rows)

    contract = rng.choice(CONTRACTS, size=n, p=[0.55, 0.25, 0.20])
    mean_tenure = np.select([contract == "Month-to-month", contract == "One year"], [14, 30], default=44)
    tenure = np.clip(np.round(rng.gamma(2.0, mean_tenure / 2.0)), 1, 72).astype(int)

    internet = rng.choice(INTERNET, size=n, p=[0.42, 0.38, 0.20])
    add_ons = np.where(internet == "No internet", 0, rng.integers(0, 6, size=n))
    base_price = np.select([internet == "Fiber optic", internet == "DSL"], [72.0, 46.0], default=22.0)
    monthly = np.clip(base_price + 4.5 * add_ons + rng.normal(0, 6, n), 18, 125).round(2)

    payment = rng.choice(PAYMENTS, size=n, p=[0.34, 0.16, 0.25, 0.25])
    paperless = rng.choice(["Yes", "No"], size=n, p=[0.6, 0.4])

    support_lambda = 0.6 + 0.5 * (internet == "Fiber optic") + 0.4 * (contract == "Month-to-month")
    support_calls = np.clip(rng.poisson(support_lambda), 0, 10)

    usage_mean = np.select([internet == "Fiber optic", internet == "DSL"], [120.0, 90.0], default=25.0)
    usage = np.clip(rng.gamma(3.0, usage_mean / 3.0), 0, 400).round(1)

    total = (monthly * tenure * rng.uniform(0.94, 1.06, n)).round(2)

    short_and_pricey = ((tenure <= 12) & (monthly >= 80)).astype(float)
    z = (
        1.0 * (contract == "Month-to-month")
        - 0.8 * (contract == "Two year")
        - 0.035 * tenure
        + 0.022 * (monthly - 65)
        + 0.30 * support_calls
        + 0.55 * (payment == "Electronic check")
        - 0.12 * add_ons
        - 0.006 * (usage - 90)
        + 0.25 * (internet == "Fiber optic")
        + 0.15 * (paperless == "Yes")
        + 0.60 * short_and_pricey
        + rng.normal(0, 0.4, n)
    )
    # Calibrate the intercept so overall churn is roughly 27%.
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(z + mid).mean() > 0.27:
            hi = mid
        else:
            lo = mid
    churn = rng.random(n) < _sigmoid(z + (lo + hi) / 2)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{10001 + i}" for i in range(n)],
            "tenure_months": tenure,
            "contract_type": contract,
            "payment_method": payment,
            "internet_service": internet,
            "paperless_billing": paperless,
            "num_add_on_services": add_ons,
            "monthly_charges": monthly,
            "total_charges": total,
            "support_calls": support_calls,
            "service_usage_hours": usage,
            "churn": np.where(churn, "Yes", "No"),
        }
    )
    # Light, realistic missingness (features only; never the label).
    df.loc[rng.random(n) < 0.015, "service_usage_hours"] = np.nan
    df.loc[rng.random(n) < 0.007, "total_charges"] = np.nan
    return df


def ensure_sample_dataset(path: Path = config.SAMPLE_PATH) -> pd.DataFrame:
    """Return the sample dataset, generating and saving it if the CSV is missing."""
    path = Path(path)
    if path.exists():
        try:
            return load_dataframe(path)
        except DataLoadError:
            pass  # fall through and regenerate a corrupt file
    df = generate_sample_data()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _read_csv(data: bytes) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=encoding, low_memory=False)
            if df.shape[1] == 1:  # maybe a ; or tab delimited file
                df = pd.read_csv(io.BytesIO(data), encoding=encoding, sep=None, engine="python")
            return df
        except pd.errors.EmptyDataError:
            raise DataLoadError("The file is empty. Please upload a CSV with a header row and data.")
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    raise DataLoadError(f"Could not parse the CSV file ({last_error}). Check the delimiter and encoding.")


def load_dataframe(source, filename: str | None = None) -> pd.DataFrame:
    """Load a CSV or Excel file from a path or an uploaded file-like object.

    Raises DataLoadError with a helpful message instead of a raw traceback.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise DataLoadError(f"File not found: {path.name}")
        filename = filename or path.name
        data = path.read_bytes()
    elif hasattr(source, "getvalue"):
        data = source.getvalue()
        filename = filename or getattr(source, "name", "upload")
    elif hasattr(source, "read"):
        data = source.read()
        filename = filename or getattr(source, "name", "upload")
    else:
        raise DataLoadError("Unsupported input. Provide a file path or an uploaded file.")

    ext = Path(str(filename)).suffix.lower()
    if ext == ".csv" or ext == ".txt":
        df = _read_csv(data)
    elif ext in (".xlsx", ".xlsm"):
        try:
            df = pd.read_excel(io.BytesIO(data), engine="openpyxl")
        except Exception as exc:
            raise DataLoadError(f"Could not read the Excel file ({type(exc).__name__}). Is it a valid .xlsx workbook?")
    elif ext == ".xls":
        raise DataLoadError("Legacy .xls files are not supported. Please save the file as .xlsx or .csv.")
    else:
        raise DataLoadError(f"Unsupported file type '{ext or 'unknown'}'. Please upload a .csv or .xlsx file.")

    return _validate_frame(df)


def _validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Drop fully empty "Unnamed" export artefacts and fully empty rows.
    junk = [c for c in df.columns if str(c).startswith("Unnamed:") and df[c].isna().all()]
    df = df.drop(columns=junk).dropna(how="all").reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise DataLoadError("The file has no data rows after removing empty rows.")
    if df.shape[1] < 2:
        raise DataLoadError("The file needs at least two columns (e.g. a customer id and one attribute).")
    if df.columns.duplicated().any():
        dupes = sorted(set(df.columns[df.columns.duplicated()]))
        raise DataLoadError(f"Duplicate column names found: {', '.join(dupes)}. Please rename them and re-upload.")
    if len(df) > config.MAX_UPLOAD_ROWS:
        raise DataLoadError(f"The file has {len(df):,} rows; the limit is {config.MAX_UPLOAD_ROWS:,}.")
    return df
