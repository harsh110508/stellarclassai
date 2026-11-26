import io
import re

import pytest

from app import create_app

VALID_FORM = {"u": "19.4", "g": "17.9", "r": "17.3", "i": "17.1", "z": "17.0", "redshift": "0.0001"}


@pytest.mark.parametrize("url", ["/", "/predict", "/explore", "/performance", "/performance?set=photometric",
                                 "/history", "/batch", "/api/health", "/api/metrics", "/batch/template.csv"])
def test_pages_load(client, url):
    assert client.get(url).status_code == 200


def test_dashboard_shows_real_metrics(client, trained):
    html = client.get("/").get_data(as_text=True)
    n_train = trained["metrics"]["dataset"]["n_train"]
    acc = trained["metrics"]["feature_sets"]["full"]["models"]["Random Forest"]["accuracy"]
    assert f"{n_train:,}" in html
    assert f"{acc * 100:.2f}%" in html


def test_unknown_routes_return_404(client):
    assert client.get("/does-not-exist").status_code == 404
    assert client.get("/performance?set=bogus").status_code == 404
    assert client.get("/api/nothing").get_json()["error"]


def test_predict_form_shows_result_and_saves_history(client):
    html = client.post("/predict", data=VALID_FORM).get_data(as_text=True)
    assert re.search(r'class="verdict verdict--(star|galaxy)"', html)
    assert "not a definitive astronomical identification" in html
    history = client.get("/history").get_data(as_text=True)
    assert "Clear history" in history and "0.0001" in history


def test_predict_form_reports_errors(client):
    html = client.post("/predict", data={**VALID_FORM, "u": "abc", "g": ""}).get_data(as_text=True)
    assert "u must be a number" in html
    assert "g is required" in html
    assert "verdict" not in html


def test_clear_history(client):
    client.post("/predict", data=VALID_FORM)
    response = client.post("/history/clear", follow_redirects=True)
    assert "Cleared 1 saved prediction" in response.get_data(as_text=True)
    assert "No predictions saved yet" in client.get("/history").get_data(as_text=True)


def test_api_predict_success(client):
    data = client.post("/api/predict", json={k: float(v) for k, v in VALID_FORM.items()}).get_json()
    assert data["prediction"] in ("STAR", "GALAXY")
    assert sum(data["probabilities"].values()) == pytest.approx(1.0)


def test_api_predict_rejects_malformed_requests(client):
    assert client.post("/api/predict", data="not json", content_type="application/json").status_code == 400
    assert client.post("/api/predict", data="u=1").status_code == 400
    response = client.post("/api/predict", json={"u": "x"})
    assert response.status_code == 422
    assert set(response.get_json()["fields"]) == {"u", "g", "r", "i", "z"}
    assert client.post("/api/predict", json=[1, 2, 3]).status_code == 422
    assert client.get("/api/predict").status_code == 405


def test_api_sample_is_limited(client):
    data = client.get("/api/sample?n=100000").get_json()
    assert len(data["rows"]) <= 2 * 3000
    assert "class" in data["columns"]


def test_batch_upload_returns_csv(client):
    csv = b"u,g,r,i,z,redshift\n19.4,17.9,17.3,17.1,17.0,0.0001\n22.1,20.6,19.1,18.5,18.1,\nx,1,1,1,1,\n"
    response = client.post("/batch", data={"file": (io.BytesIO(csv), "objects.csv")},
                           content_type="multipart/form-data")
    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    lines = response.get_data(as_text=True).strip().splitlines()
    assert lines[0].endswith("prediction,p_star,p_galaxy,feature_set,decided_by,error,warning")
    assert len(lines) == 4


@pytest.mark.parametrize("payload, message", [
    ({}, "Choose a CSV file"),
    ({"file": (io.BytesIO(b"a,b\n1,2\n"), "x.txt")}, "must be a .csv"),
    ({"file": (io.BytesIO(b"a,b\n1,2\n"), "x.csv")}, "missing required columns"),
    ({"file": (io.BytesIO(b"u,g,r,i,z\n"), "x.csv")}, "no data rows"),
])
def test_batch_upload_errors(client, payload, message):
    response = client.post("/batch", data=payload, content_type="multipart/form-data")
    assert response.status_code == 400
    assert message in response.get_data(as_text=True)


def test_csrf_protection(trained, tmp_path):
    app = create_app({"TESTING": True, "MODEL_DIR": trained["models"], "REPORTS_DIR": trained["reports"],
                      "PROCESSED_DATA_PATH": trained["processed"], "HISTORY_DB_PATH": tmp_path / "h.db"})
    client = app.test_client()
    assert client.post("/predict", data=VALID_FORM).status_code == 400
    assert client.post("/history/clear").status_code == 400
    page = client.get("/predict").get_data(as_text=True)
    token = re.search(r'name="csrf_token" value="([^"]+)"', page).group(1)
    assert client.post("/predict", data={**VALID_FORM, "csrf_token": token}).status_code == 200


def test_app_without_models_or_data(tmp_path):
    app = create_app({"TESTING": True, "CSRF_ENABLED": False, "MODEL_DIR": tmp_path / "models",
                      "REPORTS_DIR": tmp_path / "reports", "PROCESSED_DATA_PATH": tmp_path / "none.csv",
                      "HISTORY_DB_PATH": tmp_path / "h.db"})
    client = app.test_client()
    assert "python -m src.train" in client.get("/").get_data(as_text=True)
    assert "python -m src.train" in client.get("/explore").get_data(as_text=True)
    assert "No evaluation results yet" in client.get("/performance").get_data(as_text=True)
    assert "python -m src.train" in client.post("/predict", data=VALID_FORM).get_data(as_text=True)
    assert client.post("/api/predict", json={k: float(v) for k, v in VALID_FORM.items()}).status_code == 503
    assert client.get("/api/health").get_json()["status"] == "degraded"
    assert client.get("/api/sample").status_code == 503


def test_form_shows_size_rule_for_andromeda(client):
    form = {"u": "5.31", "g": "3.82", "r": "3.10", "i": "2.71", "z": "2.46",
            "redshift": "-0.001", "size_arcsec": "10800"}
    html = client.post("/predict", data=form).get_data(as_text=True)
    assert 'class="verdict verdict--galaxy"' in html
    assert "Decided by the size rule" in html
    assert "10800" in client.get("/history").get_data(as_text=True)
