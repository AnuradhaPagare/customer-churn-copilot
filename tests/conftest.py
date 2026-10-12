import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import generate_sample_data  # noqa: E402
from src.workspace import build_workspace  # noqa: E402


@pytest.fixture(scope="session")
def sample_df():
    return generate_sample_data(1000, seed=7)


@pytest.fixture(scope="session")
def ws(sample_df):
    return build_workspace(sample_df, "sample", "auto")
