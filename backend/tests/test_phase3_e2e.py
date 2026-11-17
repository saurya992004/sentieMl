import pandas as pd
from fastapi.testclient import TestClient
from app.main import app
from app.services.comparison import rank_models
from app.services.explainability import explain_pipeline
from app.services.monitoring import analyze_drift, evaluate_degradation
from sklearn.ensemble import RandomForestClassifier


def test_model_recommendations_endpoint():
    with TestClient(app) as client:
        # Upload a dataset
        csv_data = b"f1,f2,f3,label\n1,2,3,yes\n4,5,6,no\n7,8,9,yes\n10,11,12,no\n13,14,15,yes\n"
        ds = client.post("/datasets/upload", files={"file": ("test.csv", csv_data, "text/csv")}).json()
        
        # Test recommendation endpoint
        rec = client.post("/models/recommend", json={
            "dataset_id": ds["id"],
            "target_column": "label",
            "problem_type": "classification"
        })
        assert rec.status_code == 200
        data = rec.json()
        assert data["problem_type"] == "classification"
        assert len(data["recommended_algorithms"]) > 0
        assert data["recommended_algorithms"][0]["algorithm"] in ["logistic_regression", "random_forest", "gradient_boosting"]


def test_multi_objective_ranking():
    evals = [
        {
            "model_name": "model_fast",
            "algorithm": "logistic_regression",
            "metrics": {"f1": 0.85},
            "inference_latency_ms": 0.5,
            "training_time_seconds": 0.2,
            "model_size_kb": 10.0,
        },
        {
            "model_name": "model_accurate",
            "algorithm": "gradient_boosting",
            "metrics": {"f1": 0.92},
            "inference_latency_ms": 8.0,
            "training_time_seconds": 4.5,
            "model_size_kb": 250.0,
        }
    ]
    ranked = rank_models(evals, problem_type="classification", primary_metric="f1")
    assert len(ranked) == 2
    assert ranked[0]["rank"] == 1
    assert ranked[1]["rank"] == 2
    assert "selection_reasoning" in ranked[0]


def test_drift_analysis():
    ref_df = pd.DataFrame({"age": [20, 22, 25, 28, 30] * 10, "city": ["NY", "SF"] * 25})
    # Drastic shift in age
    prod_df = pd.DataFrame({"age": [65, 70, 75, 80, 85] * 10, "city": ["LA", "CHI"] * 25})
    drift = analyze_drift(ref_df, prod_df)
    assert drift["drift_detected"] is True
    assert drift["drift_severity"] in ["MODERATE", "CRITICAL"]
    assert "age" in drift["feature_drift"]
    assert drift["feature_drift"]["age"]["drift_detected"] is True


def test_performance_degradation():
    baseline = {"accuracy": 0.95, "f1": 0.94}
    # Current predictions with severe drop in accuracy
    prod_preds = [
        {"input": {}, "prediction": 0, "ground_truth": 1},
        {"input": {}, "prediction": 0, "ground_truth": 1},
        {"input": {}, "prediction": 0, "ground_truth": 1},
        {"input": {}, "prediction": 1, "ground_truth": 1},
    ]
    deg = evaluate_degradation(baseline, prod_preds, problem_type="classification")
    assert deg["degradation_detected"] is True
    assert deg["status"] == "DEGRADATION_DETECTED"
    assert len(deg["degradation_details"]) > 0


def test_governance_and_audit_endpoints():
    with TestClient(app) as client:
        # Create automation rule
        rule_resp = client.post("/automation/rules", json={
            "name": "Auto-retrain on drift",
            "model_name": "churn_model",
            "trigger_type": "DRIFT",
            "threshold": 0.3,
            "cooldown_minutes": 30
        })
        assert rule_resp.status_code == 200
        assert rule_resp.json()["name"] == "Auto-retrain on drift"

        # Check rules list
        rules = client.get("/automation/rules").json()
        assert len(rules) >= 1

        # Check audit logs
        logs = client.get("/audit-logs").json()
        assert isinstance(logs, list)
        assert len(logs) >= 1
        actions = [log["action"] for log in logs]
        assert "AUTOMATION_RULE_CREATED" in actions
