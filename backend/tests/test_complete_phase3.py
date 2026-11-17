from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app
from app.services.retraining import promote_model_version, rollback_model_version


def test_complete_phase3_workflow():
    with TestClient(app) as client:
        # TEST A: Intelligent Dataset Analysis
        csv_content = (
            b"feature1,feature2,feature3,constant_feat,target\n"
            b"10,0.5,A,1,yes\n"
            b"20,1.5,B,1,no\n"
            b"30,2.5,A,1,yes\n"
            b"40,3.5,B,1,no\n"
            b"50,4.5,A,1,yes\n"
            b"60,5.5,B,1,no\n"
            b"70,6.5,A,1,yes\n"
            b"80,7.5,B,1,no\n"
            b"90,8.5,A,1,yes\n"
            b"100,9.5,B,1,no\n"
        )
        upload_resp = client.post("/datasets/upload", files={"file": ("profile_test.csv", csv_content, "text/csv")})
        assert upload_resp.status_code == 201
        dataset_id = upload_resp.json()["id"]

        # Retrieve profile report
        profile_resp = client.get(f"/datasets/{dataset_id}/profile")
        assert profile_resp.status_code == 200
        profile = profile_resp.json()
        assert profile["dataset_id"] == dataset_id
        assert profile["quality_score"] > 0
        assert len(profile["suggested_operations"]) > 0

        # Model Recommendations
        rec_resp = client.post("/models/recommend", json={
            "dataset_id": dataset_id,
            "target_column": "target",
            "problem_type": "classification"
        })
        assert rec_resp.status_code == 200
        recs = rec_resp.json()["recommended_algorithms"]
        assert len(recs) >= 3

        # TEST B: Intelligent Model Selection & Ranking
        ranking_resp = client.get("/models/compare?problem_type=classification&primary_metric=f1")
        assert ranking_resp.status_code == 200
        assert isinstance(ranking_resp.json(), list)

        # TEST D: Monitoring and Drift
        drift_resp = client.post("/monitoring/drift", json={
            "model_name": "test_model",
            "version": "1",
            "reference_dataset_id": dataset_id,
            "production_records": [
                {"feature1": 150, "feature2": 25.0, "feature3": "Z", "constant_feat": 2},
                {"feature1": 160, "feature2": 30.0, "feature3": "Z", "constant_feat": 2},
                {"feature1": 170, "feature2": 35.0, "feature3": "Z", "constant_feat": 2},
            ]
        })
        assert drift_resp.status_code == 200
        drift_data = drift_resp.json()
        assert drift_data["sample_count"] == 3
        assert drift_data["drift_detected"] is True

        # Performance Monitoring & Ground Truth
        gt_resp = client.post("/monitoring/ground-truth", json={
            "model_name": "test_model",
            "version": "1",
            "predictions_with_actuals": [
                {"input": {"f1": 10}, "prediction": "yes", "ground_truth": "no"},
                {"input": {"f1": 20}, "prediction": "yes", "ground_truth": "no"},
            ]
        })
        assert gt_resp.status_code == 200

        perf_resp = client.get("/monitoring/performance?model_name=test_model")
        assert perf_resp.status_code == 200
        assert "degradation_detected" in perf_resp.json()

        # TEST E, F, G: Governance, Approval, Promotion & Rollback
        # 1. Promote v1
        prom1 = client.post("/models/test_pipeline_model/promote", json={
            "version": "1",
            "promoter": "Lead ML Engineer",
            "comments": "Initial production baseline"
        })
        assert prom1.status_code == 200
        assert prom1.json()["stage"] == "PRODUCTION"

        # 2. Promote v2
        prom2 = client.post("/models/test_pipeline_model/promote", json={
            "version": "2",
            "promoter": "Lead ML Engineer",
            "comments": "Upgraded model"
        })
        assert prom2.status_code == 200
        assert prom2.json()["stage"] == "PRODUCTION"

        # 3. Rollback to previous version v1
        rb_resp = client.post("/models/test_pipeline_model/rollback", json={
            "target_version": "1",
            "reason": "Degradation observed in v2",
            "initiated_by": "Ops Manager"
        })
        assert rb_resp.status_code == 200
        rb_data = rb_resp.json()
        assert rb_data["version"] == "1"
        assert rb_data["action"] == "ROLLBACK"

        # Deployment History Check
        deps = client.get("/deployments/history").json()
        assert len(deps) >= 2

        # TEST H: Security and Governance Audit Records
        audit_logs = client.get("/audit-logs").json()
        assert len(audit_logs) >= 3
        actions = [a["action"] for a in audit_logs]
        assert "DATASET_UPLOAD" in actions
        assert "MODEL_PROMOTED" in actions
        assert "MODEL_ROLLBACK" in actions
