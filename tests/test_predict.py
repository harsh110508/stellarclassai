import json
import math

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

import config
from src.predict import (InputValidationError, ModelNotFoundError, load_model, predict_dataframe, predict_one,
                         validate_input)
from src.preprocessing import encode_target

STAR_LIKE = {"u": 19.4, "g": 17.9, "r": 17.3, "i": 17.1, "z": 17.0, "redshift": 0.0001}


# --- input validation ---------------------------------------------------
def test_validate_accepts_numbers_and_numeric_strings():
    values, fs = validate_input({"u": "19.4", "g": 17.9, "r": 17, "i": " 17.1 ", "z": "17.0", "redshift": "0.0001"})
    assert fs == "full"
    assert values == {"u": 19.4, "g": 17.9, "r": 17.0, "i": 17.1, "z": 17.0, "redshift": 0.0001}


@pytest.mark.parametrize("redshift", [None, "", "   "])
def test_missing_redshift_selects_photometric_model(redshift):
    data = {k: v for k, v in STAR_LIKE.items() if k != "redshift"}
    data["redshift"] = redshift
    values, fs = validate_input(data)
    assert fs == "photometric"
    assert "redshift" not in values


@pytest.mark.parametrize("bad, field, message", [
    ({"g": 1, "r": 1, "i": 1, "z": 1}, "u", "is required"),
    ({**STAR_LIKE, "u": ""}, "u", "is required"),
    ({**STAR_LIKE, "g": "seventeen"}, "g", "must be a number"),
    ({**STAR_LIKE, "r": True}, "r", "must be a number"),
    ({**STAR_LIKE, "r": [1, 2]}, "r", "must be a number"),
    ({**STAR_LIKE, "i": "nan"}, "i", "must be a finite number"),
    ({**STAR_LIKE, "i": float("inf")}, "i", "must be a finite number"),
    ({**STAR_LIKE, "z": -9999}, "z", "between"),
    ({**STAR_LIKE, "z": 41}, "z", "between"),
    ({**STAR_LIKE, "redshift": 12}, "redshift", "between"),
])
def test_validate_rejects_bad_values(bad, field, message):
    with pytest.raises(InputValidationError) as exc:
        validate_input(bad)
    assert message in exc.value.errors[field]


def test_validate_rejects_non_object_payload():
    for payload in (None, [], "u=1", 5):
        with pytest.raises(InputValidationError):
            validate_input(payload)


# --- model files --------------------------------------------------------
def test_missing_model_file_raises(tmp_path):
    with pytest.raises(ModelNotFoundError, match="python -m src.train"):
        load_model(tmp_path / "missing.joblib")


def test_corrupt_model_file_raises(tmp_path):
    path = tmp_path / "corrupt.joblib"
    path.write_bytes(b"not a pickle")
    with pytest.raises(ModelNotFoundError, match="Could not load"):
        load_model(path)


def test_wrong_object_in_model_file_raises(tmp_path):
    path = tmp_path / "wrong.joblib"
    joblib.dump({"pipeline": None}, path)
    with pytest.raises(ModelNotFoundError, match="not a valid"):
        load_model(path)


def test_predict_without_models_raises(tmp_path):
    with pytest.raises(ModelNotFoundError):
        predict_one(STAR_LIKE, model_dir=tmp_path)


# --- trained models -----------------------------------------------------
def test_trained_models_load(trained):
    for fs, features in config.FEATURE_SETS.items():
        bundle = load_model(config.model_path(fs, trained["models"]))
        assert bundle["features"] == features
        assert bundle["label_map"] == {"GALAXY": 0, "STAR": 1}
        assert list(bundle["pipeline"].classes_) == [0, 1]


def test_prediction_response_structure(trained):
    result = predict_one(STAR_LIKE, model_dir=trained["models"])
    assert set(result) >= {"prediction", "probabilities", "feature_set", "features_used", "inputs",
                           "explanation", "disclaimer", "model"}
    assert result["prediction"] in ("STAR", "GALAXY")
    assert set(result["probabilities"]) == {"STAR", "GALAXY"}
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert all(0 <= p <= 1 for p in result["probabilities"].values())
    assert result["prediction"] == max(result["probabilities"], key=result["probabilities"].get)
    assert result["explanation"]["summary"]
    json.dumps(result)  # must be JSON-serialisable


def test_label_mapping_matches_pipeline(trained):
    """predict_one's STAR probability must be the pipeline's probability for encoded label 1."""
    bundle = load_model(config.model_path("full", trained["models"]))
    X = pd.DataFrame([STAR_LIKE])[bundle["features"]]
    p_label_1 = bundle["pipeline"].predict_proba(X)[0][list(bundle["pipeline"].classes_).index(1)]
    result = predict_one(STAR_LIKE, model_dir=trained["models"])
    assert result["probabilities"]["STAR"] == pytest.approx(p_label_1)


def test_predictions_on_unseen_records(trained):
    """Score the saved model on the held-out split it never saw during training."""
    df = pd.read_csv(trained["processed"])
    y = encode_target(df["class"])
    _, df_test, _, y_test = train_test_split(df, y, test_size=config.TEST_SIZE, stratify=y,
                                             random_state=config.RANDOM_STATE)
    out = predict_dataframe(df_test[config.MAGNITUDE_COLUMNS + ["redshift"]], model_dir=trained["models"])
    assert out["error"].isna().all()
    assert set(out["feature_set"]) == {"full"}
    predicted = encode_target(out["prediction"])
    assert (predicted == y_test).mean() > 0.95  # redshift makes this an easy problem
    assert np.allclose(out["p_star"].astype(float) + out["p_galaxy"].astype(float), 1.0, atol=1e-3)


def test_obvious_cases(trained):
    assert predict_one(STAR_LIKE, model_dir=trained["models"])["prediction"] == "STAR"
    galaxy = {"u": 23.0, "g": 21.6, "r": 20.1, "i": 19.2, "z": 18.8, "redshift": 0.45}
    assert predict_one(galaxy, model_dir=trained["models"])["prediction"] == "GALAXY"


def test_extrapolation_warning(trained):
    far = {**STAR_LIKE, "redshift": 9.0}
    warnings = predict_one(far, model_dir=trained["models"])["explanation"]["warnings"]
    assert any("outside the training range" in w for w in warnings)


def test_batch_prediction_keeps_invalid_rows(trained):
    df = pd.DataFrame([
        {"u": 19.4, "g": 17.9, "r": 17.3, "i": 17.1, "z": 17.0, "redshift": 0.0001},
        {"u": 22.1, "g": 20.6, "r": 19.1, "i": 18.5, "z": 18.1, "redshift": math.nan},
        {"u": "bad", "g": 20.6, "r": 19.1, "i": 18.5, "z": 18.1, "redshift": 0.3},
    ])
    out = predict_dataframe(df, model_dir=trained["models"])
    assert out.loc[0, "feature_set"] == "full"
    assert out.loc[1, "feature_set"] == "photometric"
    assert out.loc[2, "prediction"] is None and "u" in out.loc[2, "error"]


def test_probability_formatting_never_claims_certainty():
    from src.predict import format_probability

    assert format_probability(1.0) == "> 99.9%"
    assert format_probability(0.0) == "< 0.1%"
    assert format_probability(0.8766) == "87.7%"


def test_warns_when_one_magnitude_is_copied_into_every_box(trained):
    result = predict_one({b: 18.5 for b in "ugriz"}, model_dir=trained["models"])
    assert any("copied into every box" in w for w in result["explanation"]["warnings"])


def test_warns_about_bands_in_reverse_order(trained):
    reversed_bands = {"u": 18.8, "g": 19.2, "r": 20.1, "i": 21.6, "z": 22.8}
    result = predict_one(reversed_bands, model_dir=trained["models"])
    assert any("reverse order" in w for w in result["explanation"]["warnings"])


def test_warns_about_colours_outside_the_real_range(trained):
    result = predict_one({"u": 25.0, "g": 18.0, "r": 18.0, "i": 18.0, "z": 18.0}, model_dir=trained["models"])
    assert any("right box" in w for w in result["explanation"]["warnings"])


def test_near_zero_redshift_note_only_when_colours_look_like_a_galaxy(trained):
    galaxy_colours = {"u": 23.0, "g": 21.6, "r": 20.1, "i": 19.2, "z": 18.8, "redshift": -0.001}
    result = predict_one(galaxy_colours, model_dir=trained["models"])
    assert any("Andromeda" in w for w in result["explanation"]["warnings"])
    star = predict_one(STAR_LIKE, model_dir=trained["models"])
    assert not any("Andromeda" in w for w in star["explanation"]["warnings"])


def test_no_input_warnings_for_a_typical_galaxy(trained):
    galaxy = {"u": 23.0, "g": 21.6, "r": 20.1, "i": 19.2, "z": 18.8, "redshift": 0.45}
    assert predict_one(galaxy, model_dir=trained["models"])["explanation"]["warnings"] == []


def test_batch_marks_suspicious_rows(trained):
    df = pd.DataFrame([{b: 18.5 for b in "ugriz"}, {"u": 23.0, "g": 21.6, "r": 20.1, "i": 19.2, "z": 18.8}])
    out = predict_dataframe(df, model_dir=trained["models"])
    assert "copied" in out.loc[0, "warning"]
    assert out.loc[1, "warning"] is None


ANDROMEDA = {"u": 5.31, "g": 3.82, "r": 3.10, "i": 2.71, "z": 2.46, "redshift": -0.001}


def test_size_rule_classifies_andromeda_as_galaxy(trained):
    result = predict_one({**ANDROMEDA, "size_arcsec": 10800}, model_dir=trained["models"])
    assert result["prediction"] == "GALAXY"
    assert result["decided_by"] == "size rule"
    assert result["inputs"]["size_arcsec"] == 10800
    assert "cannot be a star" in result["explanation"]["summary"]


def test_point_sized_object_is_left_to_the_model(trained):
    result = predict_one({**STAR_LIKE, "size_arcsec": 1.5}, model_dir=trained["models"])
    assert result["decided_by"] == "model"
    assert result["prediction"] == result["model_prediction"]


def test_very_bright_input_gets_one_clear_warning(trained):
    warnings = predict_one(ANDROMEDA, model_dir=trained["models"])["explanation"]["warnings"]
    assert sum("much brighter than anything" in w for w in warnings) == 1
    assert not any("outside the training range" in w for w in warnings)


@pytest.mark.parametrize("size, message", [("big", "must be a number"), (-5, "between"), ("inf", "finite")])
def test_invalid_size_is_rejected(size, message):
    with pytest.raises(InputValidationError) as exc:
        predict_one({**STAR_LIKE, "size_arcsec": size})
    assert message in exc.value.errors["size_arcsec"]


def test_batch_applies_size_rule(trained):
    df = pd.DataFrame([{**ANDROMEDA, "size_arcsec": 10800}, {**STAR_LIKE, "size_arcsec": math.nan}])
    out = predict_dataframe(df, model_dir=trained["models"])
    assert out.loc[0, "prediction"] == "GALAXY" and out.loc[0, "decided_by"] == "size rule"
    assert out.loc[1, "decided_by"] == "model"
