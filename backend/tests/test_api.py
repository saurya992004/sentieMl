from fastapi.testclient import TestClient
from types import SimpleNamespace
from unittest.mock import patch
from app.main import app


def test_health_and_csv_upload_round_trip():
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        response = client.post(
            "/datasets/upload",
            files={"file": ("sample.csv", b"age,city,label\n25,Delhi,yes\n31,Pune,no\n", "text/csv")},
        )
        assert response.status_code == 201
        dataset = response.json()
        assert dataset["row_count"] == 2
        assert dataset["columns"]["age"] == "int64"
        detail = client.get(f"/datasets/{dataset['id']}")
        assert detail.status_code == 200
        assert detail.json()["preview"][0]["city"] == "Delhi"


def test_training_is_enqueued_in_celery():
    with TestClient(app) as client, patch("app.routers.training.train_model.apply_async") as enqueue:
        enqueue.return_value = SimpleNamespace(id="celery-test-id")
        dataset = client.post(
            "/datasets/upload",
            files={"file": ("train.csv", b"feature,label\n1,no\n2,no\n3,yes\n4,yes\n", "text/csv")},
        ).json()
        response = client.post("/training/start", json={
            "dataset_id": dataset["id"],
            "target_column": "label",
            "problem_type": "classification",
            "algorithms": ["logistic_regression"],
            "primary_metric": "f1",
        })
        assert response.status_code == 202
        assert response.json()["status"] == "QUEUED"
        assert response.json()["celery_task_id"] == "celery-test-id"
        enqueue.assert_called_once()
