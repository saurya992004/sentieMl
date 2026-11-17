from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import AuditLog, RetrainingJob, RetrainingRule
from ..schemas import RetrainingJobOut, RetrainingRuleCreate, RetrainingRuleOut, RetrainingTriggerRequest
from ..services.retraining import trigger_retraining

router = APIRouter(prefix="/retraining", tags=["retraining"])


@router.post("/configure", response_model=RetrainingRuleOut)
def configure_rule(payload: RetrainingRuleCreate, db: Session = Depends(get_db)):
    rule = RetrainingRule(
        name=payload.name,
        model_name=payload.model_name,
        trigger_type=payload.trigger_type,
        threshold=payload.threshold,
        metric_name=payload.metric_name,
        cooldown_minutes=payload.cooldown_minutes,
    )
    db.add(rule)
    db.add(
        AuditLog(
            action="RETRAINING_RULE_CREATED",
            entity_type="RETRAINING_RULE",
            entity_id=payload.model_name,
            actor="admin",
            details=payload.model_dump(),
        )
    )
    db.commit()
    db.refresh(rule)
    return rule


@router.get("/rules", response_model=list[RetrainingRuleOut])
def list_rules(model_name: str | None = Query(default=None), db: Session = Depends(get_db)):
    query = select(RetrainingRule).order_by(RetrainingRule.created_at.desc())
    if model_name:
        query = query.where(RetrainingRule.model_name == model_name)
    return db.scalars(query).all()


@router.post("/trigger", response_model=RetrainingJobOut)
def trigger_retraining_job(payload: RetrainingTriggerRequest):
    try:
        job = trigger_retraining(
            model_name=payload.model_name,
            trigger_reason=payload.trigger_reason,
            dataset_id=payload.dataset_id,
            actor="user",
        )
        return job
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/history", response_model=list[RetrainingJobOut])
def get_retraining_history(model_name: str | None = Query(default=None), db: Session = Depends(get_db)):
    query = select(RetrainingJob).order_by(RetrainingJob.created_at.desc())
    if model_name:
        query = query.where(RetrainingJob.model_name == model_name)
    return db.scalars(query).all()
