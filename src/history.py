"""Prediction history stored in a local SQLite database.

Only the numeric inputs, the predicted class and its probability are stored.
No personal information (names, IP addresses, cookies) is recorded.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    inputs TEXT NOT NULL,
    feature_set TEXT NOT NULL,
    prediction TEXT NOT NULL CHECK (prediction IN ('STAR', 'GALAXY')),
    p_star REAL NOT NULL CHECK (p_star BETWEEN 0 AND 1)
)
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(SCHEMA)


def add_prediction(db_path: Path, result: dict) -> None:
    """Store one prediction result as returned by src.predict.predict_one."""
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            "INSERT INTO predictions (created_at, inputs, feature_set, prediction, p_star) VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                json.dumps(result["inputs"]),
                result["feature_set"],
                result["prediction"],
                float(result["probabilities"]["STAR"]),
            ),
        )


def recent_predictions(db_path: Path, limit: int = 50) -> list[dict]:
    with closing(_connect(db_path)) as conn:
        rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
    return [{**dict(r), "inputs": json.loads(r["inputs"])} for r in rows]


def clear_history(db_path: Path) -> int:
    with closing(_connect(db_path)) as conn, conn:
        return conn.execute("DELETE FROM predictions").rowcount
