import numpy as np
import pandas as pd
import pytest

import config
from src.data_loader import load_raw_data
from src.preprocessing import add_colour_features, clean_data, encode_target, make_pipeline
from tests.conftest import FIXTURE_CSV


@pytest.fixture(scope="module")
def cleaned():
    return clean_data(load_raw_data(FIXTURE_CSV))


def test_clean_keeps_only_star_and_galaxy(cleaned):
    df, report = cleaned
    assert set(df["class"]) == {"STAR", "GALAXY"}
    assert report["removed_other_classes"] == 40


def test_clean_removes_sentinel_and_duplicates(cleaned):
    df, report = cleaned
    assert report["removed_sentinel_or_non_finite"] == 1
    assert report["removed_duplicates"] == 1
    assert (df[config.MAGNITUDE_COLUMNS] > config.SENTINEL_VALUE).all().all()
    assert not df.duplicated().any()
    assert report["clean_rows"] == len(df) == 750


def test_clean_drops_identifier_columns(cleaned):
    df, _ = cleaned
    for col in config.EXCLUDED_COLUMNS:
        assert col not in df.columns


def test_clean_handles_missing_and_non_numeric_values():
    raw = pd.DataFrame({
        "u": [20.0, None, "abc", 21.0], "g": [19.0] * 4, "r": [18.0] * 4, "i": [17.5] * 4, "z": [17.0] * 4,
        "redshift": [0.0, 0.1, 0.2, np.inf], "class": ["star", "GALAXY", "GALAXY", "GALAXY"],
    })
    df, report = clean_data(raw)
    assert len(df) == 1 and df["class"].iloc[0] == "STAR"  # labels are normalised to upper case
    assert report["removed_missing_values"] == 2
    assert report["removed_sentinel_or_non_finite"] == 1


def test_encode_target_mapping():
    y = encode_target(pd.Series(["STAR", "GALAXY", "STAR"]))
    assert y.tolist() == [1, 0, 1]
    assert config.LABEL_MAP[config.POSITIVE_CLASS] == 1


def test_encode_target_rejects_unknown_label():
    with pytest.raises(ValueError, match="Unknown class"):
        encode_target(pd.Series(["STAR", "QSO"]))


def test_colour_features_are_band_differences():
    X = pd.DataFrame({"u": [20.0], "g": [18.5], "r": [18.0], "i": [17.75], "z": [17.7]})
    out = add_colour_features(X)
    assert out[["u_g", "g_r", "r_i", "i_z"]].iloc[0].tolist() == pytest.approx([1.5, 0.5, 0.25, 0.05])
    assert "u_g" not in X.columns  # input not modified


def test_pipeline_scaler_is_fitted_on_training_data_only():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(20, 1, size=(40, 5)), columns=config.MAGNITUDE_COLUMNS)
    y = np.array([0, 1] * 20)
    pipe = make_pipeline(LogisticRegression(), scale=True).fit(X.iloc[:30], y[:30])
    expected_mean = add_colour_features(X.iloc[:30]).mean().to_numpy()
    assert pipe.named_steps["scaler"].mean_ == pytest.approx(expected_mean)
    assert list(pipe[:-1].get_feature_names_out()) == config.MAGNITUDE_COLUMNS + ["u_g", "g_r", "r_i", "i_z"]
