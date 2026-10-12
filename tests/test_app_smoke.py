"""Headless UI smoke tests: every page renders without exceptions for several dataset shapes."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
PAGES = ["overview_page", "analytics_page", "predictions_page", "ai_page", "model_page", "explorer_page"]

SCRIPT = '''
import sys; sys.path.insert(0, r"__ROOT__")
import streamlit as st
from src import dashboard as D
from src.theme import inject_css
from src.data_loader import generate_sample_data
from src.workspace import build_workspace
inject_css()
@st.cache_resource
def ws_(kind):
    df = generate_sample_data(600, seed=2)
    if kind == "nolabel": df = df.drop(columns=["churn"])
    if kind == "tiny": df = df.head(40)
    return build_workspace(df, kind + ".csv", "auto")
st.session_state["ctx"] = D.AppContext(ws=ws_("{kind}"), high=0.5, medium=0.25, sig="t")
D.{page}()
'''


@pytest.mark.parametrize("kind", ["full", "nolabel", "tiny"])
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(kind, page):
    at = AppTest.from_string(SCRIPT.replace("__ROOT__", str(ROOT)).format(kind=kind, page=page), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]


def test_full_app_launches():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception
