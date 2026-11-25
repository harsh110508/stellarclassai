"""StellarClass AI - Flask web application.

Run with:  python app.py   then open http://127.0.0.1:5000
The models are trained separately (python -m src.train) and only loaded here.
"""
from __future__ import annotations

import io
import json
import os
import secrets
from functools import lru_cache
from pathlib import Path

import pandas as pd
from flask import (Flask, Response, abort, flash, jsonify, redirect, render_template, request,
                   send_from_directory, session, url_for)

import config
from src import history
from src.data_loader import DatasetNotFoundError, load_processed_data
from src.predict import (InputValidationError, ModelNotFoundError, format_probability, get_model, predict_dataframe,
                         predict_one)
from src.preprocessing import add_colour_features

MAX_BATCH_ROWS = 5000
FEATURE_SET_LABELS = {"full": "Magnitudes + redshift", "photometric": "Magnitudes only"}
METRIC_LABELS = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "precision": "Precision",
                 "recall": "Recall", "f1": "F1-score", "roc_auc": "ROC-AUC", "specificity": "Specificity"}


def create_app(overrides: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.environ.get("STELLARCLASS_SECRET_KEY") or secrets.token_hex(32),
        MAX_CONTENT_LENGTH=2 * 1024 * 1024,  # 2 MB upload limit
        MODEL_DIR=config.MODEL_DIR,
        REPORTS_DIR=config.REPORTS_DIR,
        PROCESSED_DATA_PATH=config.PROCESSED_DATA_PATH,
        HISTORY_DB_PATH=config.HISTORY_DB_PATH,
        CSRF_ENABLED=True,
    )
    if overrides:
        app.config.update(overrides)
    history.init_db(app.config["HISTORY_DB_PATH"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def load_metrics() -> dict | None:
        path = config.metrics_path(app.config["REPORTS_DIR"])
        if not path.exists():
            return None
        return _read_json(str(path), path.stat().st_mtime)

    def model_status() -> dict:
        status = {}
        for fs in config.FEATURE_SETS:
            try:
                get_model(fs, app.config["MODEL_DIR"])
                status[fs] = None
            except ModelNotFoundError as exc:
                status[fs] = str(exc)
        return status

    def dataset() -> pd.DataFrame:
        path = Path(app.config["PROCESSED_DATA_PATH"])
        if not path.exists():
            raise DatasetNotFoundError(f"Processed dataset not found at {path}. Run 'python -m src.train' first.")
        return _read_dataset(str(path), path.stat().st_mtime)

    def csrf_token() -> str:
        if "csrf_token" not in session:
            session["csrf_token"] = secrets.token_urlsafe(32)
        return session["csrf_token"]

    def check_csrf() -> None:
        if app.config["CSRF_ENABLED"]:
            sent = request.form.get("csrf_token", "")
            if not sent or not secrets.compare_digest(sent, session.get("csrf_token", "")):
                abort(400, description="The form expired or was submitted from another site. "
                                       "Reload the page and try again.")

    app.add_template_filter(format_probability, "pct")

    @app.context_processor
    def inject_globals():
        return {"csrf_token": csrf_token, "FEATURE_SET_LABELS": FEATURE_SET_LABELS, "METRIC_LABELS": METRIC_LABELS,
                "config": config}

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------
    @app.get("/")
    def index():
        metrics = load_metrics()
        primary = metrics["feature_sets"][config.PRIMARY_FEATURE_SET] if metrics else None
        return render_template("index.html", metrics=metrics, primary=primary, status=model_status())

    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        result, errors, values = None, {}, {}
        if request.method == "POST":
            check_csrf()
            fields = config.MAGNITUDE_COLUMNS + [config.REDSHIFT_COLUMN]
            values = {f: request.form.get(f, "").strip() for f in fields}
            try:
                result = predict_one(values, app.config["MODEL_DIR"])
                history.add_prediction(app.config["HISTORY_DB_PATH"], result)
            except InputValidationError as exc:
                errors = exc.errors
            except ModelNotFoundError as exc:
                errors = {"model": str(exc)}
        examples = _examples(dataset_or_none(dataset))
        return render_template("predict.html", result=result, errors=errors, values=values,
                               examples=examples, status=model_status())

    @app.get("/explore")
    def explore():
        try:
            df = dataset()
        except DatasetNotFoundError as exc:
            return render_template("explore.html", error=str(exc))
        sample = df.sample(n=min(12, len(df)), random_state=config.RANDOM_STATE)
        features = add_colour_features(df[config.MAGNITUDE_COLUMNS])
        features["redshift"] = df["redshift"]
        features["class"] = df["class"]
        stats = features.groupby("class").agg(["median", "mean", "std"]).round(4)
        return render_template("explore.html", error=None, sample=sample.to_dict("records"),
                               counts=df["class"].value_counts().to_dict(), n_rows=len(df),
                               stats=stats, stat_features=[c for c in features.columns if c != "class"],
                               metrics=load_metrics())

    @app.get("/performance")
    def performance():
        metrics = load_metrics()
        feature_set = request.args.get("set", config.PRIMARY_FEATURE_SET)
        if feature_set not in config.FEATURE_SETS:
            abort(404)
        fs = metrics["feature_sets"].get(feature_set) if metrics else None
        figures = {}
        for key in ("confusion_matrix", "roc", "importance_impurity", "importance_permutation", "model_comparison"):
            name = f"{key}_{feature_set}.png"
            if (Path(app.config["REPORTS_DIR"]) / "figures" / name).exists():
                figures[key] = name
        return render_template("model_performance.html", metrics=metrics, fs=fs, feature_set=feature_set,
                               figures=figures)

    @app.get("/history")
    def history_page():
        rows = history.recent_predictions(app.config["HISTORY_DB_PATH"], limit=50)
        return render_template("history.html", rows=rows)

    @app.post("/history/clear")
    def history_clear():
        check_csrf()
        removed = history.clear_history(app.config["HISTORY_DB_PATH"])
        flash(f"Cleared {removed} saved prediction{'s' if removed != 1 else ''}.")
        return redirect(url_for("history_page"))

    @app.route("/batch", methods=["GET", "POST"])
    def batch():
        if request.method == "GET":
            return render_template("batch.html", error=None, status=model_status(), max_rows=MAX_BATCH_ROWS)
        check_csrf()
        upload = request.files.get("file")
        error = None
        if upload is None or upload.filename == "":
            error = "Choose a CSV file to upload."
        elif not upload.filename.lower().endswith(".csv"):
            error = "The file must be a .csv file."
        else:
            try:
                df = pd.read_csv(upload.stream)
            except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
                error = f"Could not read the CSV file: {exc}"
            else:
                missing = [c for c in config.MAGNITUDE_COLUMNS if c not in df.columns]
                if missing:
                    error = f"The CSV is missing required columns: {', '.join(missing)}."
                elif len(df) == 0:
                    error = "The CSV has no data rows."
                elif len(df) > MAX_BATCH_ROWS:
                    error = f"The CSV has {len(df):,} rows; the limit is {MAX_BATCH_ROWS:,}."
                else:
                    try:
                        result = predict_dataframe(df, app.config["MODEL_DIR"])
                    except ModelNotFoundError as exc:
                        error = str(exc)
                    else:
                        buf = io.StringIO()
                        result.to_csv(buf, index=False)
                        return Response(buf.getvalue(), mimetype="text/csv", headers={
                            "Content-Disposition": "attachment; filename=stellarclass_predictions.csv"})
        return render_template("batch.html", error=error, status=model_status(), max_rows=MAX_BATCH_ROWS), 400

    @app.get("/batch/template.csv")
    def batch_template():
        csv = "u,g,r,i,z,redshift\n19.4,17.9,17.3,17.1,17.0,0.0001\n22.1,20.6,19.1,18.5,18.1,\n"
        return Response(csv, mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=stellarclass_template.csv"})

    @app.get("/figures/<path:name>")
    def figure(name):
        return send_from_directory(Path(app.config["REPORTS_DIR"]) / "figures", name)

    # ------------------------------------------------------------------
    # JSON API
    # ------------------------------------------------------------------
    @app.post("/api/predict")
    def api_predict():
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify(error="Send a JSON object with u, g, r, i, z and optional redshift."), 400
        try:
            return jsonify(predict_one(payload, app.config["MODEL_DIR"]))
        except InputValidationError as exc:
            return jsonify(error="Invalid input", fields=exc.errors), 422
        except ModelNotFoundError as exc:
            return jsonify(error=str(exc)), 503

    @app.get("/api/sample")
    def api_sample():
        """Small stratified sample for the charts (never the whole dataset)."""
        try:
            df = dataset()
        except DatasetNotFoundError as exc:
            return jsonify(error=str(exc)), 503
        per_class = min(request.args.get("n", 1500, type=int) or 1500, 3000)
        parts = [g.sample(n=min(per_class, len(g)), random_state=config.RANDOM_STATE) for _, g in df.groupby("class")]
        sample = pd.concat(parts)
        out = add_colour_features(sample[config.MAGNITUDE_COLUMNS]).round(4)
        out["redshift"] = sample["redshift"].round(5)
        out["class"] = sample["class"]
        return jsonify(columns=list(out.columns), rows=out.to_numpy().tolist())

    @app.get("/api/metrics")
    def api_metrics():
        metrics = load_metrics()
        if metrics is None:
            return jsonify(error="No metrics found. Run 'python -m src.train'."), 503
        return jsonify(metrics)

    @app.get("/api/health")
    def api_health():
        status = model_status()
        return jsonify(status="ok" if not any(status.values()) else "degraded",
                       models={k: v is None for k, v in status.items()})

    # ------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------
    @app.errorhandler(400)
    @app.errorhandler(404)
    @app.errorhandler(405)
    @app.errorhandler(413)
    @app.errorhandler(500)
    def handle_error(err):
        code = getattr(err, "code", 500)
        messages = {404: "That page does not exist.", 405: "That request method is not allowed here.",
                    413: "The upload is larger than the 2 MB limit.", 500: "The server hit an unexpected error."}
        message = messages.get(code) if code != 400 else getattr(err, "description", "The request was malformed.")
        if request.path.startswith("/api/"):
            return jsonify(error=message), code
        return render_template("error.html", code=code, message=message), code

    return app


@lru_cache(maxsize=2)
def _read_json(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=2)
def _read_dataset(path: str, mtime: float) -> pd.DataFrame:
    return load_processed_data(Path(path))


def dataset_or_none(loader):
    try:
        return loader()
    except DatasetNotFoundError:
        return None


def _examples(df: pd.DataFrame | None) -> list[dict]:
    """One real star and one real galaxy from the dataset to pre-fill the form."""
    if df is None:
        return []
    out = []
    for cls in ("STAR", "GALAXY"):
        row = df[df["class"] == cls].sample(n=1, random_state=7).iloc[0]
        out.append({"label": f"Example {cls.lower()}", "class": cls,
                    "values": {c: round(float(row[c]), 5) for c in config.MAGNITUDE_COLUMNS + ["redshift"]}})
    return out


app = create_app()

if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG") == "1"
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=debug)
