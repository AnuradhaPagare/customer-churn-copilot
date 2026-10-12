"""AI-Powered Customer Churn & Retention Predictor: Streamlit entry point.

Run with:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from src import config

config.load_env()

st.set_page_config(page_title="Churn & Retention Predictor", page_icon=":material/monitoring:",
                   layout="wide", initial_sidebar_state="expanded")

from src import dashboard as D  # noqa: E402  (imported after set_page_config on purpose)
from src.ai_assistant import llm_config  # noqa: E402
from src.data_loader import DataLoadError, ensure_sample_dataset, load_dataframe  # noqa: E402
from src.theme import inject_css, md, svg, pill  # noqa: E402
from src.workspace import build_workspace  # noqa: E402

SAMPLE_NAME = "sample_customers.csv"


@st.cache_resource(show_spinner=False, max_entries=4)
def cached_workspace(fingerprint: str, name: str, target_choice: str, _raw: pd.DataFrame):
    """Clean + train once per (dataset, label choice). `_raw` is excluded from hashing."""
    return build_workspace(_raw, name, target_choice)


def fingerprint(df: pd.DataFrame) -> str:
    try:
        h = int(pd.util.hash_pandas_object(df, index=False).sum())
    except Exception:
        h = 0
    return f"{df.shape}-{h}-{'|'.join(map(str, df.columns))}"


def init_state() -> None:
    if "raw" not in st.session_state:
        st.session_state.update(raw=ensure_sample_dataset(), name=SAMPLE_NAME, upload_sig=None, load_error=None)


def handle_upload(uploaded) -> None:
    """Load a new upload once; fall back to the sample when the file is removed."""
    if uploaded is not None:
        sig = (uploaded.name, uploaded.size)
        if st.session_state.get("upload_sig") != sig:
            st.session_state["upload_sig"] = sig
            try:
                st.session_state.update(raw=load_dataframe(uploaded), name=uploaded.name, load_error=None)
            except DataLoadError as exc:
                st.session_state["load_error"] = str(exc)
            except Exception as exc:  # unexpected parser problems
                st.session_state["load_error"] = f"Could not read this file ({type(exc).__name__}). Check that it is a valid CSV or .xlsx."
    elif st.session_state.get("upload_sig") is not None:
        st.session_state.update(raw=ensure_sample_dataset(), name=SAMPLE_NAME, upload_sig=None, load_error=None)


def sidebar(pages: list) -> tuple[str, float, float]:
    with st.sidebar:
        md(f'<div class="brand"><div class="brand-mark">{svg("trend", 20, "#fff")}</div>'
           '<div><div class="brand-name">Churn Copilot</div><div class="brand-tag">Retention analytics</div></div></div>')
        for p in pages:
            st.page_link(p)
        md('<div class="side-label">Dataset</div>')
        uploaded = st.file_uploader("Upload customer data", type=["csv", "xlsx"], key="uploader",
                                    help="CSV or Excel (.xlsx). Include a churn column (Yes/No or 1/0) to enable predictions.")
        handle_upload(uploaded)
        if st.session_state.get("load_error"):
            st.error(st.session_state["load_error"])
        if st.session_state["name"] == SAMPLE_NAME:
            st.caption("Using the built-in synthetic sample (1,500 customers).")
        raw = st.session_state["raw"]
        options = ["Auto-detect", "None (descriptive only)"] + list(raw.columns)
        dataset_key = fingerprint(raw)
        choice = st.selectbox("Churn label column", options, key=f"target-{dataset_key}",
                              help="Auto-detect looks for columns named churn, exited, attrition, etc.")
        target_choice = "auto" if choice == "Auto-detect" else None if choice.startswith("None") else choice
        with st.expander("Risk bands"):
            high = st.slider("High risk at or above", 0.30, 0.95, config.DEFAULT_HIGH_THRESHOLD, 0.05, key="thr-high")
            medium = st.slider("Medium risk at or above", 0.05, 0.60, config.DEFAULT_MEDIUM_THRESHOLD, 0.05, key="thr-med")
            medium = min(medium, round(high - 0.05, 2))
            st.caption("Bands apply to estimated probabilities. They are adjustable, not calibrated business thresholds.")
        md('<div class="side-label">AI assistant</div>')
        cfg = llm_config()
        md(pill(f"LLM: {cfg['provider']}", "pos", "check") if cfg else pill("Local insights", "info", "info"))
    return target_choice, high, medium


init_state()
pages = [
    st.Page(D.overview_page, title="Overview", icon=":material/space_dashboard:", url_path="overview", default=True),
    st.Page(D.analytics_page, title="Customer Analytics", icon=":material/groups:", url_path="customer-analytics"),
    st.Page(D.predictions_page, title="Churn Predictions", icon=":material/crisis_alert:", url_path="churn-predictions"),
    st.Page(D.ai_page, title="AI Insights", icon=":material/auto_awesome:", url_path="ai-insights"),
    st.Page(D.model_page, title="Model Performance", icon=":material/model_training:", url_path="model-performance"),
    st.Page(D.explorer_page, title="Data Explorer", icon=":material/table_view:", url_path="data-explorer"),
]
nav = st.navigation(pages, position="hidden")
inject_css()
target_choice, high, medium = sidebar(pages)

raw, name = st.session_state["raw"], st.session_state["name"]
fp = fingerprint(raw)
with st.spinner("Cleaning data and training models…"):
    ws = cached_workspace(fp, name, str(target_choice), raw)
st.session_state["ctx"] = D.AppContext(ws=ws, high=high, medium=medium, sig=f"{abs(hash(fp)) % 10**8}-{target_choice}")
nav.run()
