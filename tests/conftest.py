"""Shared fixtures: quick models trained on a small real-data fixture in a temp directory."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIXTURE_CSV = ROOT / "tests" / "fixtures" / "sdss_fixture.csv"


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Run the real training pipeline (quick mode) on the fixture. Returns the paths used."""
    from src.train import run_training

    base = tmp_path_factory.mktemp("trained")
    paths = {
        "raw": FIXTURE_CSV,
        "processed": base / "processed.csv",
        "models": base / "models",
        "reports": base / "reports",
    }
    metrics = run_training(raw_path=paths["raw"], processed_path=paths["processed"], model_dir=paths["models"],
                           reports_dir=paths["reports"], quick=True, download=False)
    paths["metrics"] = metrics
    return paths


@pytest.fixture()
def app(trained, tmp_path):
    from app import create_app

    flask_app = create_app({
        "TESTING": True,
        "MODEL_DIR": trained["models"],
        "REPORTS_DIR": trained["reports"],
        "PROCESSED_DATA_PATH": trained["processed"],
        "HISTORY_DB_PATH": tmp_path / "history.db",
        "CSRF_ENABLED": False,
    })
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()
