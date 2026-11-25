"""Central configuration for StellarClass AI.

Every path, constant and dataset setting lives here so that the training
pipeline, the evaluation script, the notebook and the Flask app all agree.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "star_classification.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "sdss_star_galaxy.csv"
MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
INSTANCE_DIR = BASE_DIR / "instance"
HISTORY_DB_PATH = INSTANCE_DIR / "history.db"

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
# "Stellar Classification Dataset - SDSS17" (fedesoriano, Kaggle, Jan 2022):
# 100,000 spectroscopically classified objects exported from SDSS Data
# Release 17. The automatic download uses a public GitHub mirror pinned to
# an exact commit and verified with SHA-256, so every run uses the same bytes.
DATASET_NAME = "Stellar Classification Dataset - SDSS17"
DATASET_RELEASE = "SDSS Data Release 17 (DR17)"
DATASET_KAGGLE_URL = "https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17"
DATASET_MIRROR_URL = (
    "https://raw.githubusercontent.com/rhthm/sdss-object-classification/"
    "6f6e95a86cb3d58520716cadefb077d3ac50c107/data/star_classification.csv"
)
DATASET_SHA256 = "9628c95bf0b132f5a965d9bd775e6d7e1304d31ace2ab0527ed0d0bdd9fb3858"

TARGET_COLUMN = "class"
# Binary task: the raw file also contains QSO (quasar) rows, which are removed.
KEEP_CLASSES = ("STAR", "GALAXY")
# STAR is the positive class (label 1) for precision, recall, F1 and ROC-AUC.
POSITIVE_CLASS = "STAR"
NEGATIVE_CLASS = "GALAXY"
LABEL_MAP = {NEGATIVE_CLASS: 0, POSITIVE_CLASS: 1}
INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

MAGNITUDE_COLUMNS = ["u", "g", "r", "i", "z"]
REDSHIFT_COLUMN = "redshift"

# Two feature sets are trained:
#   full        -> magnitudes + spectroscopic redshift (the course feature set)
#   photometric -> magnitudes only (usable when no spectrum/redshift exists)
FEATURE_SETS = {
    "full": MAGNITUDE_COLUMNS + [REDSHIFT_COLUMN],
    "photometric": list(MAGNITUDE_COLUMNS),
}
PRIMARY_FEATURE_SET = "full"

# Identifiers and survey bookkeeping columns. They are never model inputs:
# they identify the object or the observation rather than describe it, and
# some (plate, MJD, fiber_ID) correlate with target-selection programmes,
# which would leak information about the label.
EXCLUDED_COLUMNS = [
    "obj_ID", "alpha", "delta", "run_ID", "rerun_ID", "cam_col", "field_ID",
    "spec_obj_ID", "plate", "MJD", "fiber_ID",
]

# SDSS uses -9999 as a "no measurement" sentinel value.
SENTINEL_VALUE = -9999.0

FEATURE_DESCRIPTIONS = {
    "u": "Apparent magnitude through the ultraviolet filter (~355 nm). Larger values mean a fainter object.",
    "g": "Apparent magnitude through the green filter (~469 nm).",
    "r": "Apparent magnitude through the red filter (~617 nm).",
    "i": "Apparent magnitude through the near-infrared filter (~748 nm).",
    "z": "Apparent magnitude through the infrared filter (~893 nm).",
    "redshift": "Spectroscopic redshift: how much the spectrum is shifted to longer wavelengths. "
                "Requires a spectrum, so it is not available for photometry-only objects.",
    "u_g": "Colour index u - g (derived). Colours compare brightness between bands and do not depend on distance.",
    "g_r": "Colour index g - r (derived).",
    "r_i": "Colour index r - i (derived).",
    "i_z": "Colour index i - z (derived).",
}

# Accepted ranges for user input in the web app. These are physical sanity
# limits, not the training range (the app warns separately about that).
INPUT_LIMITS = {
    "u": (0.0, 40.0), "g": (0.0, 40.0), "r": (0.0, 40.0),
    "i": (0.0, 40.0), "z": (0.0, 40.0),
    "redshift": (-0.05, 10.0),
}

# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


def model_path(feature_set: str, model_dir: Path = MODEL_DIR) -> Path:
    """Location of the saved inference pipeline for a feature set."""
    return Path(model_dir) / f"stellarclass_rf_{feature_set}.joblib"


def metrics_path(reports_dir: Path = REPORTS_DIR) -> Path:
    return Path(reports_dir) / "metrics.json"
