"""Dataset acquisition and loading.

Usage:
    python -m src.data_loader            # download (if needed) and verify the raw CSV
    python -m src.data_loader --force    # re-download even if the file exists
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

import pandas as pd

import config

REQUIRED_COLUMNS = config.MAGNITUDE_COLUMNS + [config.REDSHIFT_COLUMN, config.TARGET_COLUMN]

MANUAL_DOWNLOAD_HELP = f"""
Manual download instructions
----------------------------
1. Open {config.DATASET_KAGGLE_URL}
   (free Kaggle account required) and download 'star_classification.csv'.
   Alternative mirror: {config.DATASET_MIRROR_URL}
2. Save the file as:
   {config.RAW_DATA_PATH}
3. Run:  python -m src.data_loader   (to verify it), then  python -m src.train
"""


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the raw or processed dataset is missing."""


class DatasetValidationError(ValueError):
    """Raised when a dataset file does not have the expected structure."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(dest: Path = config.RAW_DATA_PATH, url: str = config.DATASET_MIRROR_URL,
                     expected_sha256: str | None = config.DATASET_SHA256, force: bool = False) -> Path:
    """Download the SDSS DR17 CSV and verify its checksum.

    The file is written to a temporary name first so that an interrupted
    download never leaves a half-written CSV behind.
    """
    dest = Path(dest)
    if dest.exists() and not force:
        print(f"Dataset already present at {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"Downloading {config.DATASET_NAME}\n  from {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as out:
            shutil.copyfileobj(response, out)
    except OSError as exc:  # URLError, timeouts and HTTP errors are all OSError subclasses
        tmp.unlink(missing_ok=True)
        raise DatasetNotFoundError(f"Automatic download failed: {exc}\n{MANUAL_DOWNLOAD_HELP}") from exc

    if expected_sha256:
        actual = sha256_of(tmp)
        if actual != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise DatasetValidationError(
                f"Checksum mismatch for downloaded file.\n  expected {expected_sha256}\n  got      {actual}"
            )
    tmp.replace(dest)
    print(f"Saved to {dest} (checksum verified)")
    return dest


def load_raw_data(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw SDSS CSV, raising an actionable error if it is missing."""
    path = Path(path)
    if not path.exists():
        raise DatasetNotFoundError(
            f"Raw dataset not found at {path}.\n"
            f"Run 'python -m src.data_loader' to download it automatically.\n{MANUAL_DOWNLOAD_HELP}"
        )
    # The ID columns are 64-bit integers; read them as text so no precision is lost.
    df = pd.read_csv(path, dtype={"obj_ID": "string", "spec_obj_ID": "string"})
    validate_columns(df, REQUIRED_COLUMNS, source=str(path))
    return df


def load_processed_data(path: Path = config.PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the cleaned binary STAR/GALAXY dataset produced by preprocessing."""
    path = Path(path)
    if not path.exists():
        raise DatasetNotFoundError(
            f"Processed dataset not found at {path}.\n"
            "Run 'python -m src.train' (it downloads and cleans the data first)."
        )
    df = pd.read_csv(path)
    validate_columns(df, REQUIRED_COLUMNS, source=str(path))
    return df


def validate_columns(df: pd.DataFrame, required: list[str], source: str = "dataframe") -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DatasetValidationError(f"{source} is missing required columns: {missing}")
    if df.empty:
        raise DatasetValidationError(f"{source} contains no rows")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and verify the SDSS DR17 dataset.")
    parser.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = parser.parse_args(argv)
    try:
        path = download_dataset(force=args.force)
        df = load_raw_data(path)
    except (DatasetNotFoundError, DatasetValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns")
    print(df[config.TARGET_COLUMN].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
