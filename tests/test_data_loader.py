import hashlib

import pandas as pd
import pytest

from src.data_loader import (DatasetNotFoundError, DatasetValidationError, download_dataset, load_processed_data,
                             load_raw_data)
from tests.conftest import FIXTURE_CSV


def test_load_raw_data_reads_fixture():
    df = load_raw_data(FIXTURE_CSV)
    assert len(df) == 792
    assert {"u", "g", "r", "i", "z", "redshift", "class"} <= set(df.columns)
    assert df["spec_obj_ID"].dtype == "string"  # 64-bit IDs are kept as text, no precision loss


def test_missing_raw_dataset_gives_actionable_error(tmp_path):
    with pytest.raises(DatasetNotFoundError) as exc:
        load_raw_data(tmp_path / "nope.csv")
    message = str(exc.value)
    assert "python -m src.data_loader" in message
    assert "kaggle.com" in message


def test_missing_processed_dataset_mentions_training(tmp_path):
    with pytest.raises(DatasetNotFoundError, match="python -m src.train"):
        load_processed_data(tmp_path / "nope.csv")


def test_missing_columns_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"u": [1.0], "class": ["STAR"]}).to_csv(path, index=False)
    with pytest.raises(DatasetValidationError, match="missing required columns"):
        load_raw_data(path)


def test_empty_file_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("u,g,r,i,z,redshift,class\n")
    with pytest.raises(DatasetValidationError, match="no rows"):
        load_raw_data(path)


def test_download_verifies_checksum(tmp_path):
    source = FIXTURE_CSV.as_uri()  # file:// URL, so no network is needed
    good = hashlib.sha256(FIXTURE_CSV.read_bytes()).hexdigest()
    dest = download_dataset(tmp_path / "ok.csv", url=source, expected_sha256=good)
    assert dest.read_bytes() == FIXTURE_CSV.read_bytes()

    with pytest.raises(DatasetValidationError, match="Checksum mismatch"):
        download_dataset(tmp_path / "bad.csv", url=source, expected_sha256="0" * 64)
    assert not (tmp_path / "bad.csv").exists()
    assert not (tmp_path / "bad.part").exists()


def test_download_failure_explains_manual_steps(tmp_path):
    with pytest.raises(DatasetNotFoundError, match="Manual download"):
        download_dataset(tmp_path / "x.csv", url=(tmp_path / "missing.csv").as_uri())


def test_download_skips_existing_file(tmp_path):
    dest = tmp_path / "exists.csv"
    dest.write_text("already here")
    assert download_dataset(dest, url="file:///does/not/matter") == dest
    assert dest.read_text() == "already here"
