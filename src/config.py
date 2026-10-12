"""Central configuration: paths, modelling constants and risk-band defaults."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"
SAMPLE_PATH = DATA_DIR / "sample_customers.csv"
DB_PATH = DATA_DIR / "churn_app.db"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
MIN_LABELLED_ROWS = 60      # below this, supervised training is refused
MIN_CLASS_COUNT = 10        # each class needs at least this many labelled rows
MAX_UPLOAD_ROWS = 200_000
SAMPLE_ROWS = 1500

# Probability bands used to label customers Low / Medium / High risk.
# They are adjustable in the UI; they are NOT calibrated business thresholds.
DEFAULT_HIGH_THRESHOLD = 0.50
DEFAULT_MEDIUM_THRESHOLD = 0.25
RISK_LEVELS = ["Low", "Medium", "High"]


def load_env() -> None:
    """Load variables from a local .env file if python-dotenv is available."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except Exception:  # pragma: no cover - dotenv is optional
        pass
