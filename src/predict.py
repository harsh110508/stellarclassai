"""Inference: load saved pipelines, validate inputs, predict and explain.

Usage (command line):
    python -m src.predict --u 19.1 --g 17.9 --r 17.4 --i 17.2 --z 17.1 --redshift 0.0001
    python -m src.predict --u 19.1 --g 17.9 --r 17.4 --i 17.2 --z 17.1      # photometric model
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

import config
from src.preprocessing import add_colour_features

REQUIRED_BUNDLE_KEYS = {"pipeline", "features", "feature_set", "label_map", "class_medians", "training_ranges"}


class ModelNotFoundError(FileNotFoundError):
    """Raised when a trained model file is missing or unusable."""


class InputValidationError(ValueError):
    """Raised when user input is invalid. `errors` maps field name -> message."""

    def __init__(self, errors: dict[str, str]):
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


def load_model(path: Path) -> dict:
    """Load a saved model bundle, checking that it is complete."""
    path = Path(path)
    if not path.exists():
        raise ModelNotFoundError(f"Model file not found at {path}. Train the models first with 'python -m src.train'.")
    try:
        bundle = joblib.load(path)
    except Exception as exc:  # corrupted / incompatible pickle
        raise ModelNotFoundError(f"Could not load model file {path}: {exc}. Re-run 'python -m src.train'.") from exc
    if not isinstance(bundle, dict) or not REQUIRED_BUNDLE_KEYS <= set(bundle):
        raise ModelNotFoundError(f"{path} is not a valid StellarClass model bundle. Re-run 'python -m src.train'.")
    return bundle


@lru_cache(maxsize=None)
def _cached_bundle(path_str: str, mtime: float) -> dict:
    return load_model(Path(path_str))


def get_model(feature_set: str, model_dir: Path = config.MODEL_DIR) -> dict:
    """Load a model once and reuse it (reloaded automatically if the file changes)."""
    path = config.model_path(feature_set, model_dir)
    if not path.exists():
        raise ModelNotFoundError(f"Model file not found at {path}. Train the models first with 'python -m src.train'.")
    return _cached_bundle(str(path), path.stat().st_mtime)


def _parse_number(value) -> float:
    if isinstance(value, bool):
        raise ValueError("must be a number")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        number = float(value.strip())  # raises ValueError for non-numeric text
    else:
        raise ValueError("must be a number")
    if not math.isfinite(number):
        raise ValueError("must be a finite number")
    return number


def validate_input(data) -> tuple[dict[str, float], str]:
    """Validate raw input (form fields or JSON) and choose the feature set.

    The five magnitudes are required. Redshift is optional: when supplied the
    'full' model is used, otherwise the photometry-only model.
    Returns (clean values, feature set name).
    """
    if not isinstance(data, dict):
        raise InputValidationError({"request": "expected an object with fields u, g, r, i, z and optional redshift"})
    errors: dict[str, str] = {}
    clean: dict[str, float] = {}

    fields = list(config.MAGNITUDE_COLUMNS)
    redshift_raw = data.get(config.REDSHIFT_COLUMN)
    has_redshift = redshift_raw is not None and str(redshift_raw).strip() != ""
    if has_redshift:
        fields.append(config.REDSHIFT_COLUMN)

    for field in fields:
        raw = data.get(field)
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            errors[field] = "is required"
            continue
        try:
            value = _parse_number(raw)
        except ValueError as exc:
            errors[field] = str(exc) if "finite" in str(exc) else "must be a number"
            continue
        low, high = config.INPUT_LIMITS[field]
        if not low <= value <= high:
            errors[field] = f"must be between {low:g} and {high:g}"
            continue
        clean[field] = value

    if errors:
        raise InputValidationError(errors)
    return clean, ("full" if has_redshift else "photometric")


def format_probability(p: float) -> str:
    """Percent string that never claims certainty: 0.99999 -> '> 99.9%'."""
    if p >= 0.9995:
        return "> 99.9%"
    if p <= 0.0005:
        return "< 0.1%"
    return f"{p:.1%}"


def _confidence_label(p: float) -> str:
    if p >= 0.9:
        return "high"
    if p >= 0.7:
        return "moderate"
    return "low"


def explain(values: dict[str, float], bundle: dict, label: str, probability: float) -> dict:
    """Plain-language explanation built from the training data's class medians.

    This compares the input to typical stars and galaxies. It describes the
    evidence the model had available; it is not a causal account of how each
    tree voted.
    """
    medians = bundle["class_medians"]
    derived = add_colour_features(pd.DataFrame([values]))[list(medians)].iloc[0]
    lines = []
    if config.REDSHIFT_COLUMN in values:
        z = values[config.REDSHIFT_COLUMN]
        star_z = medians["redshift"]["STAR"]
        gal_z = medians["redshift"]["GALAXY"]
        closer = "STAR" if abs(z - star_z) < abs(z - gal_z) else "GALAXY"
        lines.append(f"Redshift {z:.4f} is closer to the typical star value ({star_z:.4f}) than to the typical "
                     f"galaxy value ({gal_z:.3f})." if closer == "STAR" else
                     f"Redshift {z:.4f} is closer to the typical galaxy value ({gal_z:.3f}) than to the typical "
                     f"star value ({star_z:.4f}). Stars in our galaxy have redshift close to zero.")
    for colour in ("u_g", "g_r"):
        v = float(derived[colour])
        s, g = medians[colour]["STAR"], medians[colour]["GALAXY"]
        closer = "stars" if abs(v - s) < abs(v - g) else "galaxies"
        lines.append(f"Colour {colour.replace('_', ' − ')} = {v:.2f} is nearer the typical value for {closer} "
                     f"(stars {s:.2f}, galaxies {g:.2f}).")

    warnings = []
    for field, value in values.items():
        low, high = bundle["training_ranges"][field]
        if not low <= value <= high:
            warnings.append(f"{field} = {value:g} is outside the training range [{low:.3g}, {high:.3g}]; "
                            "the model is extrapolating and the result is less reliable.")

    confidence = _confidence_label(probability)
    summary = (f"The model estimates this object is a {label} with {format_probability(probability)} probability "
               f"({confidence} confidence).")
    if probability >= 0.9995:
        summary += (" Almost all trees in the forest voted the same way. That shows agreement between the trees, "
                    "not certainty about the object.")
    if confidence == "low":
        summary += " The two classes are close; treat this result as uncertain."
    return {"summary": summary, "evidence": lines, "warnings": warnings, "confidence": confidence}


def predict_one(data, model_dir: Path = config.MODEL_DIR) -> dict:
    """Validate input, run the saved pipeline and return a JSON-serialisable result."""
    values, feature_set = validate_input(data)
    bundle = get_model(feature_set, model_dir)
    X = pd.DataFrame([values])[bundle["features"]]
    proba = bundle["pipeline"].predict_proba(X)[0]
    classes = list(bundle["pipeline"].classes_)  # encoded labels, e.g. [0, 1]
    inverse = {v: k for k, v in bundle["label_map"].items()}
    probabilities = {inverse[c]: float(p) for c, p in zip(classes, proba, strict=True)}
    label = max(probabilities, key=probabilities.get)
    return {
        "prediction": label,
        "probabilities": probabilities,
        "feature_set": feature_set,
        "features_used": bundle["features"],
        "inputs": values,
        "model": "Random Forest",
        "explanation": explain(values, bundle, label, probabilities[label]),
        "disclaimer": "This is a statistical model estimate, not a definitive astronomical identification.",
    }


def predict_dataframe(df: pd.DataFrame, model_dir: Path = config.MODEL_DIR) -> pd.DataFrame:
    """Batch prediction. Rows with a redshift use the full model, others the photometric model.

    Invalid rows are kept and given an error message instead of a prediction.
    """
    out = df.copy()
    out["prediction"], out["p_star"], out["p_galaxy"] = None, None, None
    out["feature_set"], out["error"] = None, None
    valid: dict[str, list[tuple[int, dict]]] = {"full": [], "photometric": []}
    for idx, row in df.iterrows():
        record = {k: row.get(k) for k in config.MAGNITUDE_COLUMNS + [config.REDSHIFT_COLUMN]}
        record = {k: (None if pd.isna(v) else v) for k, v in record.items()}
        try:
            values, feature_set = validate_input(record)
            valid[feature_set].append((idx, values))
        except InputValidationError as exc:
            out.at[idx, "error"] = str(exc)
    for feature_set, rows in valid.items():
        if not rows:
            continue
        bundle = get_model(feature_set, model_dir)
        X = pd.DataFrame([v for _, v in rows])[bundle["features"]]
        star_col = list(bundle["pipeline"].classes_).index(bundle["label_map"][config.POSITIVE_CLASS])
        p_star = bundle["pipeline"].predict_proba(X)[:, star_col]
        for (idx, _), p in zip(rows, p_star, strict=True):
            out.at[idx, "p_star"] = round(float(p), 4)
            out.at[idx, "p_galaxy"] = round(float(1 - p), 4)
            out.at[idx, "prediction"] = config.POSITIVE_CLASS if p >= 0.5 else config.NEGATIVE_CLASS
            out.at[idx, "feature_set"] = feature_set
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify one object as STAR or GALAXY.")
    for band in config.MAGNITUDE_COLUMNS:
        parser.add_argument(f"--{band}", required=True, help=f"{band}-band magnitude")
    parser.add_argument("--redshift", default=None, help="optional spectroscopic redshift")
    args = vars(parser.parse_args(argv))
    try:
        result = predict_one(args)
    except (InputValidationError, ModelNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
