from datetime import datetime
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from .database import Base


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    file_path: Mapped[str] = mapped_column(String(1024), unique=True)
    row_count: Mapped[int] = mapped_column(Integer)
    column_count: Mapped[int] = mapped_column(Integer)
    columns: Mapped[dict] = mapped_column(JSON)
    missing_values: Mapped[dict] = mapped_column(JSON)
    duplicate_rows: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DatasetProfile(Base):
    __tablename__ = "dataset_profiles"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    quality_score: Mapped[float] = mapped_column(Float, default=100.0)
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    statistics: Mapped[dict] = mapped_column(JSON, default=dict)
    outliers: Mapped[dict] = mapped_column(JSON, default=dict)
    correlations: Mapped[dict] = mapped_column(JSON, default=dict)
    target_analysis: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    suggested_operations: Mapped[list] = mapped_column(JSON, default=list)
    risks: Mapped[list] = mapped_column(JSON, default=list)
    ai_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class TrainingJob(Base):
    __tablename__ = "training_jobs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    target_column: Mapped[str] = mapped_column(String(255))
    problem_type: Mapped[str] = mapped_column(String(32))
    algorithms: Mapped[list] = mapped_column(JSON)
    primary_metric: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="QUEUED")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    logs: Mapped[list] = mapped_column(JSON, default=list)
    celery_task_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    cancel_requested: Mapped[bool] = mapped_column(default=False)
    hyperparameters: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=dict)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    best_algorithm: Mapped[str | None] = mapped_column(String(100), nullable=True)
    mlflow_experiment_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    registered_model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ModelLifecycle(Base):
    __tablename__ = "model_lifecycles"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(255))
    version: Mapped[str] = mapped_column(String(64))
    stage: Mapped[str] = mapped_column(String(32), default="DEVELOPMENT")  # DEVELOPMENT, CANDIDATE, APPROVED, PRODUCTION, ARCHIVED
    run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    training_job_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    parameters: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ModelEvaluation(Base):
    __tablename__ = "model_evaluations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(255))
    version: Mapped[str] = mapped_column(String(64))
    algorithm: Mapped[str] = mapped_column(String(100))
    problem_type: Mapped[str] = mapped_column(String(32))
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    training_time_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    inference_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    model_size_kb: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DriftReport(Base):
    __tablename__ = "drift_reports"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(255))
    model_version: Mapped[str] = mapped_column(String(64))
    reference_dataset_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    feature_drift: Mapped[dict] = mapped_column(JSON, default=dict)
    overall_drift_score: Mapped[float] = mapped_column(Float, default=0.0)
    drift_severity: Mapped[str] = mapped_column(String(32), default="NONE")  # NONE, LOW, MODERATE, CRITICAL
    drift_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    recommended_action: Mapped[str] = mapped_column(Text, default="No action needed.")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ProductionPrediction(Base):
    __tablename__ = "production_predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(255))
    model_version: Mapped[str] = mapped_column(String(64))
    input_data: Mapped[dict] = mapped_column(JSON)
    prediction: Mapped[dict | list | float | int | str] = mapped_column(JSON)
    ground_truth: Mapped[dict | list | float | int | str | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RetrainingRule(Base):
    __tablename__ = "retraining_rules"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    model_name: Mapped[str] = mapped_column(String(255))
    trigger_type: Mapped[str] = mapped_column(String(32))  # DRIFT, DEGRADATION, SCHEDULED, MANUAL
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    metric_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    cooldown_minutes: Mapped[int] = mapped_column(Integer, default=60)
    last_triggered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RetrainingJob(Base):
    __tablename__ = "retraining_jobs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rule_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_name: Mapped[str] = mapped_column(String(255))
    trigger_reason: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="PENDING")  # PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
    candidate_model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    candidate_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    evaluation_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    approval_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ModelApproval(Base):
    __tablename__ = "model_approvals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_model_name: Mapped[str] = mapped_column(String(255))
    candidate_version: Mapped[str] = mapped_column(String(64))
    current_production_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    comparison_summary: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="PENDING")  # PENDING, APPROVED, REJECTED
    reviewer_comments: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class DeploymentRecord(Base):
    __tablename__ = "deployment_records"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(255))
    version: Mapped[str] = mapped_column(String(64))
    previous_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    action: Mapped[str] = mapped_column(String(32))  # PROMOTE, ROLLBACK
    deployment_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    initiated_by: Mapped[str] = mapped_column(String(255), default="system")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    action: Mapped[str] = mapped_column(String(64))
    entity_type: Mapped[str] = mapped_column(String(64))
    entity_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    actor: Mapped[str] = mapped_column(String(255), default="system")
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
