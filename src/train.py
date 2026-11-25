"""Train, tune, evaluate and save the star-galaxy classifiers.

Usage:
    python -m src.train           # full run (a few minutes on a laptop)
    python -m src.train --quick   # small search for a fast smoke test

For each feature set ('full' = u,g,r,i,z,redshift; 'photometric' = u,g,r,i,z):
  1. stratified 80/20 train/test split (fixed seed)
  2. baselines: majority-class Dummy and Logistic Regression
  3. Random Forest tuned with RandomizedSearchCV (stratified k-fold, F1 for STAR)
  4. 5-fold cross-validation of the tuned Random Forest on the training set
  5. single final evaluation on the untouched test set
  6. the whole fitted Pipeline is saved with joblib
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split

import config
from src import evaluate
from src.data_loader import DatasetNotFoundError, DatasetValidationError, download_dataset, load_processed_data
from src.preprocessing import add_colour_features, encode_target, make_pipeline, prepare_dataset

RF_PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 10, 20, 30],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2", None],
    "model__class_weight": [None, "balanced"],
}
CV_SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
              "f1": "f1", "roc_auc": "roc_auc", "balanced_accuracy": "balanced_accuracy"}


def get_data(raw_path: Path, processed_path: Path, reports_dir: Path, download: bool) -> tuple[pd.DataFrame, dict]:
    """Clean the raw CSV (downloading it first if allowed) and return data + cleaning report."""
    report_path = Path(reports_dir) / "preprocessing_report.json"
    if not Path(raw_path).exists():
        if Path(processed_path).exists() and report_path.exists():
            return load_processed_data(processed_path), json.loads(report_path.read_text())
        if not download:
            raise DatasetNotFoundError(f"Raw dataset not found at {raw_path}. Run 'python -m src.data_loader'.")
        download_dataset(raw_path)
    return prepare_dataset(raw_path, processed_path, report_path)


def class_summaries(X: pd.DataFrame, y: np.ndarray) -> tuple[dict, dict]:
    """Per-class medians and overall training ranges, used to explain predictions."""
    full = add_colour_features(X)
    medians = {col: {config.INVERSE_LABEL_MAP[k]: float(full.loc[y == k, col].median()) for k in (0, 1)}
               for col in full.columns}
    ranges = {col: [float(X[col].min()), float(X[col].max())] for col in X.columns}
    return medians, ranges


def train_feature_set(feature_set: str, df_train: pd.DataFrame, df_test: pd.DataFrame,
                      y_train: np.ndarray, y_test: np.ndarray, quick: bool, random_state: int,
                      model_dir: Path, figures_dir: Path) -> dict:
    features = config.FEATURE_SETS[feature_set]
    X_train, X_test = df_train[features], df_test[features]
    n_folds = 3 if quick else config.CV_FOLDS
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    print(f"\n=== Feature set '{feature_set}': {features} ===")

    # --- Baselines ------------------------------------------------------
    candidates = {
        "Majority-class baseline": make_pipeline(DummyClassifier(strategy="most_frequent"), scale=False),
        "Logistic Regression": make_pipeline(LogisticRegression(max_iter=2000, random_state=random_state),
                                             scale=True),
    }
    results, fitted = {}, {}
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        results[name] = evaluate.compute_metrics(y_test, pipe.predict(X_test), proba)
        fitted[name] = (pipe, proba)
        print(f"{name:<26} test F1={results[name]['f1']:.4f}  ROC-AUC={results[name].get('roc_auc', float('nan')):.4f}")

    # --- Random Forest with randomized hyper-parameter search -------------
    rf = make_pipeline(RandomForestClassifier(random_state=random_state, n_jobs=-1), scale=False)
    search = RandomizedSearchCV(
        rf, RF_PARAM_DISTRIBUTIONS, n_iter=3 if quick else 12, scoring="f1", cv=cv,
        random_state=random_state, n_jobs=1, refit=True, return_train_score=False,
    )
    if quick:
        search.set_params(param_distributions={**RF_PARAM_DISTRIBUTIONS, "model__n_estimators": [30, 50]})
    t0 = time.time()
    search.fit(X_train, y_train)
    search_seconds = time.time() - t0
    best = search.best_estimator_
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    print(f"Random Forest search: {search_seconds:.0f}s, best CV F1={search.best_score_:.4f}, params={best_params}")

    # Cross-validation of the chosen configuration (training data only).
    cv_raw = cross_validate(best, X_train, y_train, cv=cv, scoring=CV_SCORING, n_jobs=1)
    cv_results = {m: {"mean": float(cv_raw[f"test_{m}"].mean()), "std": float(cv_raw[f"test_{m}"].std()),
                      "folds": [round(float(v), 6) for v in cv_raw[f"test_{m}"]]} for m in CV_SCORING}
    print(f"{n_folds}-fold CV F1 = {cv_results['f1']['mean']:.4f} +/- {cv_results['f1']['std']:.4f}")

    # Final, single evaluation on the held-out test set.
    rf_proba = best.predict_proba(X_test)[:, 1]
    rf_pred = (rf_proba >= 0.5).astype(int)
    results["Random Forest"] = evaluate.compute_metrics(y_test, rf_pred, rf_proba)
    fitted["Random Forest"] = (best, rf_proba)
    print(f"{'Random Forest':<26} test F1={results['Random Forest']['f1']:.4f}  "
          f"ROC-AUC={results['Random Forest']['roc_auc']:.4f}")
    print(evaluate.text_report(y_test, rf_pred))

    impurity = evaluate.impurity_importance(best)
    permutation = evaluate.permutation_importances(best, X_test, y_test, random_state=random_state)

    # --- Save the complete inference pipeline -----------------------------
    medians, ranges = class_summaries(X_train, y_train)
    bundle = {
        "pipeline": best,
        "feature_set": feature_set,
        "features": features,
        "label_map": config.LABEL_MAP,
        "positive_class": config.POSITIVE_CLASS,
        "class_medians": medians,
        "training_ranges": ranges,
        "best_params": best_params,
        "test_metrics": {k: v for k, v in results["Random Forest"].items() if k != "confusion_matrix"},
        "n_train": int(len(y_train)),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }
    path = config.model_path(feature_set, model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    print(f"Saved pipeline to {path} ({path.stat().st_size / 1e6:.1f} MB)")

    # --- Figures ----------------------------------------------------------
    evaluate.plot_confusion_matrix(results["Random Forest"]["confusion_matrix"],
                                   f"Random Forest ({feature_set}) - test set",
                                   figures_dir / f"confusion_matrix_{feature_set}.png")
    curves = {}
    for name in ("Logistic Regression", "Random Forest"):
        pts = evaluate.roc_points(y_test, fitted[name][1], max_points=10_000)
        curves[name] = (pts["fpr"], pts["tpr"], results[name]["roc_auc"])
    evaluate.plot_roc_curves(curves, f"ROC curves ({feature_set} features)", figures_dir / f"roc_{feature_set}.png")
    evaluate.plot_importance(impurity, f"Random Forest impurity importance ({feature_set})",
                             "Mean decrease in impurity", figures_dir / f"importance_impurity_{feature_set}.png")
    evaluate.plot_importance(permutation, f"Permutation importance on test set ({feature_set})",
                             "Drop in ROC-AUC when shuffled", figures_dir / f"importance_permutation_{feature_set}.png")
    table = pd.DataFrame({name: {m: r[m] for m in ("accuracy", "balanced_accuracy", "precision", "recall", "f1")}
                          for name, r in results.items()}).T
    evaluate.plot_model_comparison(table, f"Model comparison ({feature_set} features)",
                                   figures_dir / f"model_comparison_{feature_set}.png")

    return {
        "features": features,
        "models": results,
        "best_params": best_params,
        "search": {"n_iter": int(search.n_iter), "cv_folds": n_folds, "scoring": "f1 (STAR positive)",
                   "best_cv_f1": float(search.best_score_), "seconds": round(search_seconds, 1)},
        "cross_validation": cv_results,
        "roc_curve": evaluate.roc_points(y_test, rf_proba),
        "roc_curve_logreg": evaluate.roc_points(y_test, fitted["Logistic Regression"][1]),
        "feature_importance": {"impurity": impurity, "permutation": permutation},
        "classification_report": evaluate.text_report(y_test, rf_pred),
        "model_file": path.name,
    }


def run_training(raw_path: Path = config.RAW_DATA_PATH, processed_path: Path = config.PROCESSED_DATA_PATH,
                 model_dir: Path = config.MODEL_DIR, reports_dir: Path = config.REPORTS_DIR,
                 quick: bool = False, random_state: int = config.RANDOM_STATE, download: bool = True) -> dict:
    model_dir, reports_dir = Path(model_dir), Path(reports_dir)
    figures_dir = reports_dir / "figures"
    df, cleaning = get_data(raw_path, processed_path, reports_dir, download)
    y = encode_target(df[config.TARGET_COLUMN])
    df_train, df_test, y_train, y_test = train_test_split(
        df, y, test_size=config.TEST_SIZE, stratify=y, random_state=random_state)
    print(f"Clean rows: {len(df):,}  train: {len(df_train):,}  test: {len(df_test):,}  "
          f"STAR share: {y.mean():.3f}")

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quick_mode": quick,
        "random_state": random_state,
        "positive_class": config.POSITIVE_CLASS,
        "primary_feature_set": config.PRIMARY_FEATURE_SET,
        "dataset": {
            "name": config.DATASET_NAME, "release": config.DATASET_RELEASE,
            "cleaning": cleaning, "n_total": int(len(df)),
            "n_train": int(len(df_train)), "n_test": int(len(df_test)),
            "test_class_counts": {config.INVERSE_LABEL_MAP[k]: int((y_test == k).sum()) for k in (0, 1)},
        },
        "feature_sets": {},
    }
    for feature_set in config.FEATURE_SETS:
        metrics["feature_sets"][feature_set] = train_feature_set(
            feature_set, df_train, df_test, y_train, y_test, quick, random_state, model_dir, figures_dir)

    out = config.metrics_path(reports_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(f"\nMetrics written to {out}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the StellarClass AI models.")
    parser.add_argument("--quick", action="store_true", help="small hyper-parameter search (smoke test)")
    parser.add_argument("--no-download", action="store_true", help="fail instead of downloading the dataset")
    args = parser.parse_args(argv)
    try:
        run_training(quick=args.quick, download=not args.no_download)
    except (DatasetNotFoundError, DatasetValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
