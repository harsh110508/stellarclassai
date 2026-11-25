"""Model evaluation: metrics, figures, and a CLI to re-evaluate saved models.

Usage:
    python -m src.evaluate        # reload the saved models and re-score the held-out test set
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split

import config

CLASS_NAMES = [config.NEGATIVE_CLASS, config.POSITIVE_CLASS]  # index = encoded label
PALETTE = {"GALAXY": "#4C72B0", "STAR": "#DD8452"}


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Binary metrics with STAR (label 1) as the positive class."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    metrics = {k: round(float(v), 6) for k, v in metrics.items()}
    metrics["confusion_matrix"] = [[int(tn), int(fp)], [int(fn), int(tp)]]
    return metrics


def roc_points(y_true, y_proba, max_points: int = 200) -> dict:
    """ROC curve thinned to at most `max_points` points for the web charts."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    if len(fpr) > max_points:
        idx = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
        fpr, tpr = fpr[idx], tpr[idx]
    return {"fpr": [round(float(v), 5) for v in fpr], "tpr": [round(float(v), 5) for v in tpr]}


def impurity_importance(pipeline) -> list[dict]:
    """Mean decrease in impurity from the Random Forest, per engineered feature."""
    model = pipeline.named_steps["model"]
    names = pipeline[:-1].get_feature_names_out()
    per_tree = np.array([t.feature_importances_ for t in model.estimators_])
    rows = [{"feature": str(n), "importance": float(m), "std": float(s)}
            for n, m, s in zip(names, model.feature_importances_, per_tree.std(axis=0), strict=True)]
    return sorted(rows, key=lambda r: r["importance"], reverse=True)


def permutation_importances(pipeline, X: pd.DataFrame, y, random_state: int = config.RANDOM_STATE,
                            max_rows: int = 5000) -> list[dict]:
    """Drop in ROC-AUC when one *input* column is shuffled (held-out data).

    Shuffling a magnitude also changes the colours derived from it, so this
    measures the total contribution of each measured input.
    """
    if len(X) > max_rows:
        X, _, y, _ = train_test_split(X, y, train_size=max_rows, stratify=y, random_state=random_state)
    result = permutation_importance(pipeline, X, y, scoring="roc_auc", n_repeats=5,
                                    random_state=random_state, n_jobs=-1)
    rows = [{"feature": c, "importance": float(m), "std": float(s)}
            for c, m, s in zip(X.columns, result.importances_mean, result.importances_std, strict=True)]
    return sorted(rows, key=lambda r: r["importance"], reverse=True)


def text_report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES, digits=4,
                                 zero_division=0)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _save(fig, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion_matrix(cm, title: str, path: Path) -> None:
    cm = np.asarray(cm)
    row_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    labels = np.array([[f"{cm[i, j]:,}\n({row_pct[i, j]:.1f}%)" for j in range(2)] for i in range(2)])
    fig, ax = plt.subplots(figsize=(5, 4.2))
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, annot_kws={"size": 11})
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    _save(fig, path)


def plot_roc_curves(curves: dict[str, tuple], title: str, path: Path) -> None:
    """curves: label -> (fpr, tpr, auc)."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for label, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="Random guess (AUC = 0.5)")
    ax.set_xlabel("False positive rate (galaxies called stars)")
    ax.set_ylabel("True positive rate (stars recovered)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def plot_importance(rows: list[dict], title: str, xlabel: str, path: Path) -> None:
    df = pd.DataFrame(rows).sort_values("importance")
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(df) + 1.2))
    ax.barh(df["feature"], df["importance"], xerr=df["std"], color="#4C72B0", ecolor="#555", capsize=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Feature")
    ax.set_title(title)
    _save(fig, path)


def plot_model_comparison(table: pd.DataFrame, title: str, path: Path) -> None:
    """table: rows = models, columns = metrics."""
    long = table.reset_index(names="model").melt(id_vars="model", var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.barplot(data=long, x="metric", y="score", hue="model", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Metric (STAR = positive class)")
    ax.set_ylabel("Test-set score")
    ax.set_title(title)
    ax.legend(title="Model", fontsize=8, loc="lower left")
    _save(fig, path)


# --------------------------------------------------------------------------
# CLI: re-evaluate saved models on the same held-out split used in training
# --------------------------------------------------------------------------
def main() -> int:
    from src.data_loader import DatasetNotFoundError, load_processed_data
    from src.predict import ModelNotFoundError, load_model
    from src.preprocessing import encode_target

    try:
        df = load_processed_data()
    except DatasetNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    y = encode_target(df[config.TARGET_COLUMN])
    # Same seed, same data, same stratification -> the identical test set used in training.
    _, df_test, _, y_test = train_test_split(df, y, test_size=config.TEST_SIZE, stratify=y,
                                             random_state=config.RANDOM_STATE)
    for feature_set in config.FEATURE_SETS:
        try:
            bundle = load_model(config.model_path(feature_set))
        except ModelNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        X_test = df_test[bundle["features"]]
        proba = bundle["pipeline"].predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        m = compute_metrics(y_test, pred, proba)
        print(f"\n=== Random Forest, feature set '{feature_set}' ({', '.join(bundle['features'])}) ===")
        print(f"Test rows: {len(y_test):,}   positive class: {config.POSITIVE_CLASS}")
        print(text_report(y_test, pred))
        print(f"ROC-AUC: {m['roc_auc']:.4f}   balanced accuracy: {m['balanced_accuracy']:.4f}")
        print(f"Confusion matrix [[TN, FP], [FN, TP]]: {m['confusion_matrix']}")
        plot_confusion_matrix(m["confusion_matrix"], f"Random Forest ({feature_set}) - test set",
                              config.FIGURES_DIR / f"confusion_matrix_{feature_set}.png")
    print(f"\nFigures written to {config.FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
