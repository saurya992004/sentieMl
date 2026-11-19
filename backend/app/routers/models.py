from datetime import datetime, timezone
from typing import Any
import mlflow.pyfunc
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import AuditLog, Dataset, DeploymentRecord, ModelApproval, ModelEvaluation, ModelLifecycle, TrainingJob
from ..schemas import (
    ApprovalDecision,
    DeploymentRecordOut,
    ExplanationResponse,
    ModelApprovalOut,
    ModelRankingItem,
    PromotionRequest,
    RecommendationRequest,
    RecommendationResponse,
    RollbackRequest,
)
from ..services.comparison import rank_models
from ..services.datasets import read_dataset
from ..services.explainability import explain_pipeline
from ..services.recommender import recommend_models
from ..services.retraining import promote_model_version, rollback_model_version

router = APIRouter(tags=["models"])


@router.post("/models/recommend", response_model=RecommendationResponse)
def get_model_recommendations(payload: RecommendationRequest, db: Session = Depends(get_db)):
    dataset = db.get(Dataset, payload.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        frame = read_dataset(dataset.file_path)
        res = recommend_models(frame, target_column=payload.target_column, problem_type=payload.problem_type)
        res["dataset_id"] = payload.dataset_id
        db.add(
            AuditLog(
                action="MODEL_RECOMMENDATION",
                entity_type="DATASET",
                entity_id=str(dataset.id),
                actor="user",
                details={"target_column": payload.target_column, "top_algorithm": res["recommended_algorithms"][0]["algorithm"]},
            )
        )
        db.commit()
        return res
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/models/compare", response_model=list[ModelRankingItem])
def compare_models(
    problem_type: str = Query(default="classification"),
    primary_metric: str = Query(default="f1"),
    db: Session = Depends(get_db),
):
    evaluations = db.scalars(
        select(ModelEvaluation)
        .where(ModelEvaluation.problem_type == problem_type)
        .order_by(ModelEvaluation.created_at.desc())
        .limit(20)
    ).all()

    eval_dicts = [
        {
            "model_name": e.model_name,
            "version": e.version,
            "algorithm": e.algorithm,
            "problem_type": e.problem_type,
            "metrics": e.metrics,
            "training_time_seconds": e.training_time_seconds,
            "inference_latency_ms": e.inference_latency_ms,
            "model_size_kb": e.model_size_kb,
        }
        for e in evaluations
    ]

    return rank_models(eval_dicts, problem_type=problem_type, primary_metric=primary_metric)


@router.get("/models/{model_name}/explanations", response_model=ExplanationResponse)
def get_model_explanations(
    model_name: str,
    version: str = Query(default="latest"),
    dataset_id: int | None = Query(default=None),
    db: Session = Depends(get_db),
):
    try:
        from .mlflow import client
        resolved_version = version
        if resolved_version == "latest":
            versions = client().search_model_versions(f"name='{model_name}'")
            if not versions:
                raise ValueError(f"No versions found for model '{model_name}'")
            resolved_version = max(versions, key=lambda item: int(item.version)).version

        # Load pipeline from MLflow (prefer native sklearn pipeline, fallback to pyfunc)
        try:
            import mlflow.sklearn
            sklearn_pipeline = mlflow.sklearn.load_model(f"models:/{model_name}/{resolved_version}")
        except Exception:
            model = mlflow.pyfunc.load_model(f"models:/{model_name}/{resolved_version}")
            sklearn_pipeline = getattr(model, "_model_impl", model)
            if hasattr(sklearn_pipeline, "sklearn_model"):
                sklearn_pipeline = sklearn_pipeline.sklearn_model
            elif hasattr(sklearn_pipeline, "python_model"):
                sklearn_pipeline = sklearn_pipeline.python_model

        # Get sample data from dataset
        if dataset_id:
            ds = db.get(Dataset, dataset_id)
            sample_df = read_dataset(ds.file_path).head(50) if ds else None
        else:
            # Pick latest dataset
            latest_ds = db.scalars(select(Dataset).order_by(Dataset.created_at.desc())).first()
            sample_df = read_dataset(latest_ds.file_path).head(50) if latest_ds else None

        if sample_df is None:
            raise ValueError("No sample dataset available for computing feature explainability")

        # Exclude common target names if present
        features_df = sample_df.drop(columns=[col for col in ["target", "label", "price", "income"] if col in sample_df.columns], errors="ignore")
        local_sample = features_df.iloc[0].to_dict() if len(features_df) > 0 else None

        res = explain_pipeline(sklearn_pipeline, features_df, local_sample=local_sample)
        return ExplanationResponse(
            model_name=model_name,
            version=resolved_version,
            algorithm=res["algorithm"],
            feature_names=res["feature_names"],
            global_importance=res["global_importance"],
            local_explanation=res["local_explanation"],
            summary=res["summary"],
            ai_business_translation=res.get("ai_business_translation"),
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to generate explanations: {exc}") from exc


@router.get("/approvals", response_model=list[ModelApprovalOut])
def list_approvals(db: Session = Depends(get_db)):
    return db.scalars(select(ModelApproval).order_by(ModelApproval.created_at.desc())).all()


@router.post("/approvals/{approval_id}/approve", response_model=ModelApprovalOut)
def approve_candidate(approval_id: int, payload: ApprovalDecision, db: Session = Depends(get_db)):
    approval = db.get(ModelApproval, approval_id)
    if not approval:
        raise HTTPException(status_code=404, detail="Approval request not found")
    if approval.status != "PENDING":
        raise HTTPException(status_code=400, detail=f"Request is already {approval.status}")

    approval.status = "APPROVED"
    approval.reviewer_comments = payload.comments
    approval.reviewed_by = payload.reviewer
    approval.resolved_at = datetime.now(timezone.utc)

    # Automatically promote to production if approved
    promote_model_version(
        model_name=approval.candidate_model_name,
        version=approval.candidate_version,
        promoter=payload.reviewer,
        comments=payload.comments,
    )

    db.add(
        AuditLog(
            action="MODEL_APPROVED",
            entity_type="APPROVAL",
            entity_id=str(approval.id),
            actor=payload.reviewer,
            details={"model": approval.candidate_model_name, "version": approval.candidate_version, "comments": payload.comments},
        )
    )
    db.commit()
    db.refresh(approval)
    return approval


@router.post("/approvals/{approval_id}/reject", response_model=ModelApprovalOut)
def reject_candidate(approval_id: int, payload: ApprovalDecision, db: Session = Depends(get_db)):
    approval = db.get(ModelApproval, approval_id)
    if not approval:
        raise HTTPException(status_code=404, detail="Approval request not found")
    if approval.status != "PENDING":
        raise HTTPException(status_code=400, detail=f"Request is already {approval.status}")

    approval.status = "REJECTED"
    approval.reviewer_comments = payload.comments
    approval.reviewed_by = payload.reviewer
    approval.resolved_at = datetime.now(timezone.utc)

    db.add(
        AuditLog(
            action="MODEL_REJECTED",
            entity_type="APPROVAL",
            entity_id=str(approval.id),
            actor=payload.reviewer,
            details={"model": approval.candidate_model_name, "version": approval.candidate_version, "comments": payload.comments},
        )
    )
    db.commit()
    db.refresh(approval)
    return approval


@router.post("/models/{model_name}/promote")
def promote_model(model_name: str, payload: PromotionRequest):
    try:
        updated = promote_model_version(
            model_name=model_name,
            version=payload.version,
            promoter=payload.promoter,
            comments=payload.comments,
        )
        return {"status": "promoted", "model_name": model_name, "version": payload.version, "stage": updated.stage}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/models/{model_name}/rollback", response_model=DeploymentRecordOut)
def rollback_model(model_name: str, payload: RollbackRequest):
    try:
        record = rollback_model_version(
            model_name=model_name,
            target_version=payload.target_version,
            initiated_by=payload.initiated_by,
            reason=payload.reason,
        )
        return record
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/deployments/history", response_model=list[DeploymentRecordOut])
def get_deployment_history(db: Session = Depends(get_db)):
    return db.scalars(select(DeploymentRecord).order_by(DeploymentRecord.created_at.desc())).all()
