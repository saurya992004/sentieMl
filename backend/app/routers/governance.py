from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import AuditLog, RetrainingRule
from ..schemas import AuditLogOut, RetrainingRuleCreate, RetrainingRuleOut

router = APIRouter(tags=["governance"])


@router.get("/automation/rules", response_model=list[RetrainingRuleOut])
def get_automation_rules(db: Session = Depends(get_db)):
    return db.scalars(select(RetrainingRule).order_by(RetrainingRule.created_at.desc())).all()


@router.post("/automation/rules", response_model=RetrainingRuleOut)
def create_automation_rule(payload: RetrainingRuleCreate, db: Session = Depends(get_db)):
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
            action="AUTOMATION_RULE_CREATED",
            entity_type="RULE",
            entity_id=payload.name,
            actor="admin",
            details=payload.model_dump(),
        )
    )
    db.commit()
    db.refresh(rule)
    return rule


@router.get("/audit-logs", response_model=list[AuditLogOut])
def get_audit_logs(
    action: str | None = Query(default=None),
    limit: int = Query(default=50),
    db: Session = Depends(get_db),
):
    query = select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)
    if action:
        query = query.where(AuditLog.action == action)
    return db.scalars(query).all()
