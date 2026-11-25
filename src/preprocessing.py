"""Data cleaning and feature preprocessing.

Cleaning (done once, before the train/test split) only removes rows that are
invalid for *any* model: other classes, sentinel values, non-finite numbers
and exact duplicates. Anything learned from data (scaling statistics) lives
inside a scikit-learn Pipeline so it is fitted on the training fold only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import config

COLOUR_PAIRS = [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]
COLOUR_COLUMNS = [f"{a}_{b}" for a, b in COLOUR_PAIRS]


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return the cleaned STAR/GALAXY dataframe and a report of every step."""
    numeric = config.MAGNITUDE_COLUMNS + [config.REDSHIFT_COLUMN]
    report: dict = {"raw_rows": int(len(raw)),
                    "raw_class_counts": raw[config.TARGET_COLUMN].value_counts().to_dict()}

    df = raw[numeric + [config.TARGET_COLUMN]].copy()
    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].astype(str).str.strip().str.upper()

    before = len(df)
    df = df[df[config.TARGET_COLUMN].isin(config.KEEP_CLASSES)]
    report["removed_other_classes"] = before - len(df)

    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    before = len(df)
    df = df.dropna(subset=numeric)
    report["removed_missing_values"] = before - len(df)

    before = len(df)
    sentinel = (df[numeric] <= config.SENTINEL_VALUE).any(axis=1)
    non_finite = ~np.isfinite(df[numeric]).all(axis=1)
    df = df[~(sentinel | non_finite)]
    report["removed_sentinel_or_non_finite"] = before - len(df)

    before = len(df)
    df = df.drop_duplicates()
    report["removed_duplicates"] = before - len(df)

    df = df.reset_index(drop=True)
    report["clean_rows"] = int(len(df))
    report["clean_class_counts"] = df[config.TARGET_COLUMN].value_counts().to_dict()
    report["features_kept"] = numeric
    report["columns_excluded_as_identifiers_or_leakage"] = config.EXCLUDED_COLUMNS
    return df, report


def prepare_dataset(raw_path: Path = config.RAW_DATA_PATH,
                    processed_path: Path = config.PROCESSED_DATA_PATH,
                    report_path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Load the raw CSV, clean it, and write the processed CSV plus a JSON report."""
    from src.data_loader import load_raw_data  # local import avoids a cycle

    clean, report = clean_data(load_raw_data(raw_path))
    processed_path = Path(processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(processed_path, index=False)
    if report_path is not None:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, indent=2))
    return clean, report


def encode_target(labels: pd.Series) -> np.ndarray:
    """Map class names to integers: GALAXY -> 0, STAR -> 1 (positive class)."""
    unknown = set(labels.unique()) - set(config.LABEL_MAP)
    if unknown:
        raise ValueError(f"Unknown class labels: {sorted(unknown)}")
    return labels.map(config.LABEL_MAP).to_numpy(dtype=int)


def add_colour_features(X: pd.DataFrame) -> pd.DataFrame:
    """Append colour indices (differences between adjacent magnitudes).

    Colours measure the shape of an object's spectrum independently of how far
    away it is, which makes them more informative than raw magnitudes. This is
    a pure function of each row, so it cannot leak information between rows.
    """
    X = pd.DataFrame(X).copy()
    for a, b in COLOUR_PAIRS:
        X[f"{a}_{b}"] = X[a] - X[b]
    return X


def engineered_feature_names(input_features: list[str]) -> list[str]:
    return list(input_features) + COLOUR_COLUMNS


def build_preprocessor(scale: bool) -> list[tuple[str, object]]:
    """Pipeline steps shared by every model.

    Tree models do not need scaling; linear models do. Either way the colour
    features are added inside the pipeline so inference matches training.
    """
    steps: list[tuple[str, object]] = [
        ("colours", FunctionTransformer(add_colour_features, feature_names_out=_colour_names_out)),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return steps


def _colour_names_out(transformer, input_features):
    return np.asarray(engineered_feature_names(list(input_features)), dtype=object)


def make_pipeline(estimator, scale: bool) -> Pipeline:
    return Pipeline(build_preprocessor(scale) + [("model", estimator)])
