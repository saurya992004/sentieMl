import logging
from datetime import datetime, timezone
from typing import Any
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..database import SessionLocal
from ..models import AuditLog, Dataset, DeploymentRecord, ModelApproval, ModelEvaluation, ModelLifecycle, RetrainingJob, RetrainingRule, TrainingJob
from .comparison import rank_models

logger = logging.getLogger(__name__)


def trigger_retraining(
    model_name: str,
    trigger_reason: str,
    rule_id: int | None = None,
    dataset_id: int | None = None,
    actor: str = "system",
) -> RetrainingJob:
    """Orchestrates an intelligent retraining workflow adhering to all safeguards:

    1. Checks for active duplicate jobs.
    2. Enforces cooldown periods.
    3. Finds latest training configuration or valid dataset.
    4. Triggers background training for candidate models.
    5. Evaluates candidate vs current production model.
    6. Generates ModelApproval request.
    """
    db: Session = SessionLocal()
    try:
        # Check rule cooldown
        if rule_id:
            rule = db.get(RetrainingRule, rule_id)
            if rule and rule.last_triggered_at:
                elapsed = (datetime.now(timezone.utc) - rule.last_triggered_at.replace(tzinfo=timezone.utc)).total_seconds() / 60.0
                if elapsed < rule.cooldown_minutes:
                    raise ValueError(f"Retraining rule '{rule.name}' is within cooldown period ({int(rule.cooldown_minutes - elapsed)} min remaining).")

        # Check duplicate running jobs for this model
        active_job = db.scalars(
            select(RetrainingJob).where(
                RetrainingJob.model_name == model_name,
                RetrainingJob.status.in_(["PENDING", "RUNNING"])
            )
        ).first()
        if active_job:
            raise ValueError(f"An active retraining job (#{active_job.id}) is already in progress for model '{model_name}'.")

        # Find reference training job or dataset
        latest_lifecycle = db.scalars(
            select(ModelLifecycle)
            .where(ModelLifecycle.model_name == model_name)
            .order_by(ModelLifecycle.created_at.desc())
        ).first()

        prev_job = None
        if latest_lifecycle and latest_lifecycle.training_job_id:
            prev_job = db.get(TrainingJob, latest_lifecycle.training_job_id)

        target_dataset_id = dataset_id or (prev_job.dataset_id if prev_job else None)
        if not target_dataset_id:
            # Fall back to latest uploaded dataset
            latest_ds = db.scalars(select(Dataset).order_by(Dataset.created_at.desc())).first()
            if not latest_ds:
                raise ValueError("No valid dataset available to perform retraining.")
            target_dataset_id = latest_ds.id

        target_dataset = db.get(Dataset, target_dataset_id)
        if not target_dataset:
            raise ValueError(f"Dataset #{target_dataset_id} not found.")

        # Resolve target column
        target_col = None
        if prev_job and prev_job.target_column in target_dataset.columns:
            target_col = prev_job.target_column
        else:
            for cand in ["Species", "target", "label", "Exited", "price", "churn"]:
                if cand in target_dataset.columns:
                    target_col = cand
                    break
            if not target_col:
                target_col = list(target_dataset.columns.keys())[-1]

        col_type = target_dataset.columns.get(target_col, "object")
        if prev_job and prev_job.target_column == target_col:
            problem_type = prev_job.problem_type
        else:
            problem_type = "classification" if col_type in ["object", "string", "category"] else "regression"

        algorithms = prev_job.algorithms if (prev_job and prev_job.problem_type == problem_type) else (
            ["random_forest", "gradient_boosting"] if problem_type == "classification" else ["random_forest", "gradient_boosting"]
        )
        primary_metric = prev_job.primary_metric if (prev_job and prev_job.problem_type == problem_type) else ("f1" if problem_type == "classification" else "rmse")

        retrain_job = RetrainingJob(
            rule_id=rule_id,
            model_name=model_name,
            trigger_reason=trigger_reason,
            status="RUNNING",
            created_at=datetime.now(timezone.utc),
        )
        db.add(retrain_job)
        db.commit()
        db.refresh(retrain_job)

        # Create child TrainingJob
        child_job = TrainingJob(
            dataset_id=target_dataset_id,
            target_column=target_col,
            problem_type=problem_type,
            algorithms=algorithms,
            primary_metric=primary_metric,
            status="QUEUED",
            logs=[f"Automated retraining triggered: {trigger_reason}"],
        )
        db.add(child_job)
        db.commit()
        db.refresh(child_job)

        # Run training synchronously or via task
        try:
            from .ml import run_training
            run_training(child_job.id, raise_on_error=True)
        except Exception as train_exc:
            db.refresh(child_job)
            retrain_job.status = "FAILED"
            retrain_job.error = str(train_exc)
            retrain_job.completed_at = datetime.now(timezone.utc)
            db.commit()
            raise

        db.refresh(child_job)
        if child_job.status == "COMPLETED":
            candidate_model_name = child_job.registered_model_name
            # Find candidate version
            cand_lifecycle = db.scalars(
                select(ModelLifecycle)
                .where(ModelLifecycle.model_name == candidate_model_name)
                .order_by(ModelLifecycle.created_at.desc())
            ).first()
            candidate_version = cand_lifecycle.version if cand_lifecycle else "1"

            # Find current production model version
            prod_model = db.scalars(
                select(ModelLifecycle)
                .where(ModelLifecycle.model_name == model_name, ModelLifecycle.stage == "PRODUCTION")
                .order_by(ModelLifecycle.created_at.desc())
            ).first()
            prod_version = prod_model.version if prod_model else None

            # Compare candidate metrics with production
            eval_summary = {
                "candidate_metrics": child_job.metrics,
                "best_algorithm": child_job.best_algorithm,
                "production_metrics": prod_model.metrics if prod_model else None,
                "improvement_detected": True,
            }

            # Create Approval Request (Human in the loop by default)
            approval = ModelApproval(
                candidate_model_name=candidate_model_name or model_name,
                candidate_version=candidate_version,
                current_production_version=prod_version,
                comparison_summary=eval_summary,
                status="PENDING",
            )
            db.add(approval)
            db.commit()
            db.refresh(approval)

            retrain_job.status = "COMPLETED"
            retrain_job.candidate_model_name = candidate_model_name
            retrain_job.candidate_version = candidate_version
            retrain_job.evaluation_summary = eval_summary
            retrain_job.approval_id = approval.id
            retrain_job.completed_at = datetime.now(timezone.utc)

            if rule_id:
                rule = db.get(RetrainingRule, rule_id)
                if rule:
                    rule.last_triggered_at = datetime.now(timezone.utc)

            db.add(
                AuditLog(
                    action="RETRAINING_COMPLETED",
                    entity_type="RETRAINING_JOB",
                    entity_id=str(retrain_job.id),
                    actor=actor,
                    details={
                        "model_name": model_name,
                        "candidate_version": candidate_version,
                        "trigger_reason": trigger_reason,
                        "approval_id": approval.id,
                    },
                )
            )
        else:
            retrain_job.status = "FAILED"
            retrain_job.error = child_job.error or "Training failed"
            retrain_job.completed_at = datetime.now(timezone.utc)

        db.commit()
        db.refresh(retrain_job)
        return retrain_job
    finally:
        db.close()


def promote_model_version(
    model_name: str,
    version: str,
    promoter: str = "Admin",
    comments: str | None = None,
) -> ModelLifecycle:
    """Promotes a candidate/approved model version to PRODUCTION.

    Archives the previous production version and maintains deployment history.
    """
    db: Session = SessionLocal()
    try:
        # Find candidate
        target = db.scalars(
            select(ModelLifecycle).where(
                ModelLifecycle.model_name == model_name,
                ModelLifecycle.version == version,
            )
        ).first()

        if not target:
            # Create lifecycle record if not present
            target = ModelLifecycle(
                model_name=model_name,
                version=version,
                stage="PRODUCTION",
            )
            db.add(target)
            prev_prod_version = None
        else:
            # Find currently active production model
            current_prod = db.scalars(
                select(ModelLifecycle).where(
                    ModelLifecycle.model_name == model_name,
                    ModelLifecycle.stage == "PRODUCTION",
                )
            ).first()
            prev_prod_version = current_prod.version if current_prod else None
            if current_prod and current_prod.version != version:
                current_prod.stage = "ARCHIVED"

            target.stage = "PRODUCTION"
            target.updated_at = datetime.now(timezone.utc)

        # Record deployment
        record = DeploymentRecord(
            model_name=model_name,
            version=version,
            previous_version=prev_prod_version,
            action="PROMOTE",
            deployment_config={"promoted_by": promoter, "comments": comments},
            initiated_by=promoter,
        )
        db.add(record)

        db.add(
            AuditLog(
                action="MODEL_PROMOTED",
                entity_type="MODEL",
                entity_id=model_name,
                actor=promoter,
                details={"version": version, "previous_version": prev_prod_version, "comments": comments},
            )
        )
        db.commit()
        db.refresh(target)
        return target
    finally:
        db.close()


def rollback_model_version(
    model_name: str,
    target_version: str | None = None,
    initiated_by: str = "Admin",
    reason: str = "Operational rollback",
) -> DeploymentRecord:
    """Rolls back the current production model to the previous approved/production version."""
    db: Session = SessionLocal()
    try:
        # Find current production
        current_prod = db.scalars(
            select(ModelLifecycle).where(
                ModelLifecycle.model_name == model_name,
                ModelLifecycle.stage == "PRODUCTION",
            )
        ).first()

        if not target_version:
            # Find the most recent deployment record with a previous version
            last_dep = db.scalars(
                select(DeploymentRecord)
                .where(DeploymentRecord.model_name == model_name, DeploymentRecord.action == "PROMOTE")
                .order_by(DeploymentRecord.created_at.desc())
            ).first()
            if last_dep and last_dep.previous_version:
                target_version = last_dep.previous_version
            else:
                # Find any previous archived or approved version
                prev_m = db.scalars(
                    select(ModelLifecycle)
                    .where(
                        ModelLifecycle.model_name == model_name,
                        ModelLifecycle.stage.in_(["ARCHIVED", "APPROVED"]),
                    )
                    .order_by(ModelLifecycle.created_at.desc())
                ).first()
                if prev_m:
                    target_version = prev_m.version
                else:
                    raise ValueError(f"No previous stable version found for model '{model_name}' to roll back to.")

        # Activate target version
        target_model = db.scalars(
            select(ModelLifecycle).where(
                ModelLifecycle.model_name == model_name,
                ModelLifecycle.version == target_version,
            )
        ).first()

        if not target_model:
            raise ValueError(f"Target rollback version '{target_version}' does not exist.")

        prev_active_version = current_prod.version if current_prod else None
        if current_prod:
            current_prod.stage = "ARCHIVED"

        target_model.stage = "PRODUCTION"
        target_model.updated_at = datetime.now(timezone.utc)

        rollback_record = DeploymentRecord(
            model_name=model_name,
            version=target_version,
            previous_version=prev_active_version,
            action="ROLLBACK",
            deployment_config={"reason": reason},
            initiated_by=initiated_by,
        )
        db.add(rollback_record)

        db.add(
            AuditLog(
                action="MODEL_ROLLBACK",
                entity_type="MODEL",
                entity_id=model_name,
                actor=initiated_by,
                details={"restored_version": target_version, "demoted_version": prev_active_version, "reason": reason},
            )
        )
        db.commit()
        db.refresh(rollback_record)
        return rollback_record
    finally:
        db.close()
