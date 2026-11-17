from datetime import datetime, timezone
from typing import Any
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import AuditLog, Dataset, DriftReport, ModelLifecycle, ProductionPrediction
from ..schemas import DriftAnalysisRequest, DriftReportOut, GroundTruthSubmission
from ..services.datasets import read_dataset
from ..services.monitoring import analyze_drift, evaluate_degradation

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.post("/drift", response_model=DriftReportOut)
def detect_drift(payload: DriftAnalysisRequest, db: Session = Depends(get_db)):
    # Fetch reference dataset
    ref_ds = None
    if payload.reference_dataset_id:
        ref_ds = db.get(Dataset, payload.reference_dataset_id)
    if not ref_ds:
        ref_ds = db.scalars(select(Dataset).order_by(Dataset.created_at.desc())).first()
    if not ref_ds:
        raise HTTPException(status_code=404, detail="No reference dataset found for drift comparison")

    ref_df = read_dataset(ref_ds.file_path)
    prod_df = pd.DataFrame(payload.production_records)
    if prod_df.empty:
        raise HTTPException(status_code=422, detail="Production records cannot be empty")

    res = analyze_drift(ref_df, prod_df)
    report = DriftReport(
        model_name=payload.model_name,
        model_version=payload.version,
        reference_dataset_id=ref_ds.id,
        sample_count=res["sample_count"],
        feature_drift=res["feature_drift"],
        overall_drift_score=res["overall_drift_score"],
        drift_severity=res["drift_severity"],
        drift_detected=res["drift_detected"],
        recommended_action=res["recommended_action"],
    )
    db.add(report)

    db.add(
        AuditLog(
            action="DRIFT_ANALYSIS",
            entity_type="MODEL",
            entity_id=payload.model_name,
            actor="monitoring_agent",
            details={
                "model_version": payload.version,
                "drift_detected": res["drift_detected"],
                "severity": res["drift_severity"],
                "drifted_features": res["drifted_features_count"],
            },
        )
    )
    db.commit()
    db.refresh(report)
    return report


@router.get("/drift", response_model=list[DriftReportOut])
def list_drift_reports(
    model_name: str | None = Query(default=None),
    limit: int = Query(default=20),
    db: Session = Depends(get_db),
):
    query = select(DriftReport).order_by(DriftReport.created_at.desc()).limit(limit)
    if model_name:
        query = query.where(DriftReport.model_name == model_name)
    return db.scalars(query).all()


@router.post("/ground-truth")
def submit_ground_truth(payload: GroundTruthSubmission, db: Session = Depends(get_db)):
    # Record or update production predictions
    count = 0
    for item in payload.predictions_with_actuals:
        pred_val = item.get("prediction")
        actual_val = item.get("ground_truth")
        inp = item.get("input", {})
        db.add(
            ProductionPrediction(
                model_name=payload.model_name,
                model_version=payload.version,
                input_data=inp,
                prediction=pred_val,
                ground_truth=actual_val,
            )
        )
        count += 1
    db.commit()
    return {"status": "recorded", "records_count": count}


@router.get("/performance")
def get_performance_monitoring(
    model_name: str,
    version: str = Query(default="latest"),
    db: Session = Depends(get_db),
):
    # Retrieve baseline metrics from ModelLifecycle
    query = select(ModelLifecycle).where(ModelLifecycle.model_name == model_name)
    if version != "latest":
        query = query.where(ModelLifecycle.version == version)
    else:
        query = query.order_by(ModelLifecycle.created_at.desc())

    lifecycle = db.scalars(query).first()
    baseline = lifecycle.metrics if lifecycle and lifecycle.metrics else {"accuracy": 0.90, "f1": 0.88}

    # Fetch production predictions with ground truth
    preds = db.scalars(
        select(ProductionPrediction)
        .where(ProductionPrediction.model_name == model_name)
        .order_by(ProductionPrediction.created_at.desc())
        .limit(200)
    ).all()

    pred_dicts = [
        {"input": p.input_data, "prediction": p.prediction, "ground_truth": p.ground_truth}
        for p in preds
    ]

    prob_type = "classification"
    if lifecycle and lifecycle.parameters:
        prob_type = lifecycle.parameters.get("problem_type", "classification")

    return evaluate_degradation(baseline, pred_dicts, problem_type=prob_type)
