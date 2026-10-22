import json

import pytest
from fastapi.testclient import TestClient

from codeformer.api.app import create_app


@pytest.fixture
def client(predictor, tmp_path):
    app = create_app(predictor=predictor, feedback_path=str(tmp_path / "fb.jsonl"))
    with TestClient(app) as c:
        c.feedback_file = tmp_path / "fb.jsonl"
        yield c


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["model_loaded"] and r["n_levels"] == 6


def test_predict_shape(client):
    r = client.post("/predict", json={"texts": ["l6k01 l5k01 parcel", "foo"], "top_k": 3})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 2 and len(preds[0]) == 3
    first = preds[0][0]
    assert len(first["codes"]) == 6 and first["code_string"].count("-") == 5
    assert 0 <= first["score"] <= 1


def test_predict_parallel_mode(client):
    r = client.post("/predict", json={"texts": ["l6k01 l5k01"], "mode": "parallel"})
    assert r.status_code == 200 and len(r.json()["predictions"][0]) == 1


@pytest.mark.parametrize(
    "body",
    [{}, {"texts": []}, {"texts": ["x"], "top_k": 0}, {"texts": ["x"], "mode": "bad"}],
)
def test_predict_validation(client, body):
    assert client.post("/predict", json=body).status_code == 422


def test_feedback_appends_jsonl(client):
    body = {"predictor_text": "some text", "output_text": "01-01-01-01-01-01"}
    assert client.post("/feedback", json=body).status_code == 202
    assert client.post("/feedback", json=body).status_code == 202
    lines = client.feedback_file.read_text().strip().splitlines()
    assert len(lines) == 2 and json.loads(lines[0]) == body


def test_feedback_rejects_bad_codes(client):
    r = client.post("/feedback", json={"predictor_text": "t", "output_text": "01-02"})
    assert r.status_code == 422
    assert not client.feedback_file.exists()


def test_no_model_returns_503(tmp_path):
    with TestClient(create_app(checkpoint=str(tmp_path / "missing.pt"))) as c:
        assert c.get("/health").json() == {
            "status": "degraded",
            "model_loaded": False,
            "n_levels": None,
            "vocab_size": None,
            "n_paths": None,
        }
        assert c.post("/predict", json={"texts": ["x"]}).status_code == 503
