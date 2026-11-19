from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field


class DatasetOut(BaseModel):
    id: int
    name: str
    row_count: int
    column_count: int
    columns: dict[str, str]
    missing_values: dict[str, int]
    duplicate_rows: int
    created_at: datetime
    model_config = {"from_attributes": True}


class DatasetDetail(DatasetOut):
    preview: list[dict[str, Any]]


class DatasetProfileOut(BaseModel):
    dataset_id: int
    quality_score: float
    warnings: list[dict[str, Any]]
    statistics: dict[str, Any]
    outliers: dict[str, Any]
    correlations: dict[str, Any]
    target_analysis: dict[str, Any] | None
    suggested_operations: list[str]
    risks: list[str]
    ai_summary: str | None = None
    created_at: datetime | None = None
    model_config = {"from_attributes": True}


class RecommendationRequest(BaseModel):
    dataset_id: int
    target_column: str
    problem_type: Literal["classification", "regression"] | None = None


class AlgorithmRecommendation(BaseModel):
    algorithm: str
    display_name: str
    score: float  # suitability score 0-100
    recommended: bool
    reasons: list[str]
    estimated_speed: str  # Fast, Moderate, Slow
    computational_requirement: str  # Low, Medium, High


class RecommendationResponse(BaseModel):
    dataset_id: int
    target_column: str
    problem_type: str
    recommended_algorithms: list[AlgorithmRecommendation]
    explanation: str
    ai_insights: str | None = None
    dataset_summary: dict[str, Any]


class TrainingStart(BaseModel):
    dataset_id: int
    target_column: str = Field(min_length=1)
    problem_type: Literal["classification", "regression"]
    algorithms: list[str] = Field(min_length=1, max_length=10)
    primary_metric: str | None = None
    hyperparameters: dict[str, Any] | None = None


class JobOut(BaseModel):
    id: int
    dataset_id: int
    target_column: str
    problem_type: str
    algorithms: list[str]
    primary_metric: str
    status: str
    progress: int
    logs: list[str]
    celery_task_id: str | None
    attempt_count: int
    cancel_requested: bool
    hyperparameters: dict[str, Any] | None = None
    metrics: dict[str, Any] | None
    best_algorithm: str | None
    mlflow_experiment_id: str | None
    mlflow_run_id: str | None
    registered_model_name: str | None
    error: str | None
    created_at: datetime
    completed_at: datetime | None
    model_config = {"from_attributes": True}


class PredictionRequest(BaseModel):
    model_name: str
    version: str = "latest"
    records: list[dict[str, Any]] = Field(min_length=1)


class PredictionResponse(BaseModel):
    model_name: str
    version: str
    predictions: list[Any]


# --- Phase 3 Additional Schemas ---

class ModelComparisonRequest(BaseModel):
    model_names: list[str] | None = None
    problem_type: Literal["classification", "regression"] | None = None
    primary_metric: str = "f1"
    weights: dict[str, float] | None = None  # weights for multi-objective scoring (e.g. {"f1": 0.6, "latency": 0.2, "training_time": 0.2})


class ModelRankingItem(BaseModel):
    model_name: str
    version: str
    algorithm: str
    problem_type: str
    dataset_name: str | None = None
    job_id: int | None = None
    metrics: dict[str, Any]
    training_time_seconds: float
    inference_latency_ms: float
    model_size_kb: float
    overall_score: float
    rank: int
    selection_reasoning: str


class ExplanationResponse(BaseModel):
    model_name: str
    version: str
    algorithm: str
    feature_names: list[str]
    global_importance: list[dict[str, Any]]
    local_explanation: list[dict[str, Any]] | None = None
    summary: str
    ai_business_translation: str | None = None


class DriftAnalysisRequest(BaseModel):
    model_name: str
    version: str = "latest"
    reference_dataset_id: int | None = None
    production_records: list[dict[str, Any]] = Field(min_length=1)


class DriftReportOut(BaseModel):
    id: int | None = None
    model_name: str
    model_version: str
    reference_dataset_id: int | None
    sample_count: int
    feature_drift: dict[str, Any]
    overall_drift_score: float
    drift_severity: str
    drift_detected: bool
    recommended_action: str
    created_at: datetime
    model_config = {"from_attributes": True}


class GroundTruthSubmission(BaseModel):
    prediction_id: int | None = None
    model_name: str
    version: str = "latest"
    predictions_with_actuals: list[dict[str, Any]]  # [{"input": {...}, "prediction": ..., "ground_truth": ...}]


class RetrainingRuleCreate(BaseModel):
    name: str
    model_name: str
    trigger_type: Literal["DRIFT", "DEGRADATION", "SCHEDULED", "MANUAL"]
    threshold: float | None = None
    metric_name: str | None = None
    cooldown_minutes: int = 60


class RetrainingRuleOut(BaseModel):
    id: int
    name: str
    model_name: str
    trigger_type: str
    threshold: float | None
    metric_name: str | None
    is_active: bool
    cooldown_minutes: int
    last_triggered_at: datetime | None
    created_at: datetime
    model_config = {"from_attributes": True}


class RetrainingTriggerRequest(BaseModel):
    model_name: str
    trigger_reason: str = "Manual trigger by user"
    dataset_id: int | None = None


class RetrainingJobOut(BaseModel):
    id: int
    rule_id: int | None
    model_name: str
    trigger_reason: str
    status: str
    candidate_model_name: str | None
    candidate_version: str | None
    evaluation_summary: dict[str, Any] | None
    approval_id: int | None
    error: str | None
    created_at: datetime
    completed_at: datetime | None
    model_config = {"from_attributes": True}


class ModelApprovalOut(BaseModel):
    id: int
    candidate_model_name: str
    candidate_version: str
    current_production_version: str | None
    comparison_summary: dict[str, Any]
    status: str
    reviewer_comments: str | None
    reviewed_by: str | None
    created_at: datetime
    resolved_at: datetime | None
    model_config = {"from_attributes": True}


class ApprovalDecision(BaseModel):
    comments: str | None = None
    reviewer: str = "Admin"


class PromotionRequest(BaseModel):
    version: str
    comments: str | None = None
    promoter: str = "Admin"


class RollbackRequest(BaseModel):
    target_version: str | None = None  # if None, roll back to previous production version
    reason: str = "Model degradation or operational issue"
    initiated_by: str = "Admin"


class DeploymentRecordOut(BaseModel):
    id: int
    model_name: str
    version: str
    previous_version: str | None
    action: str
    deployment_config: dict[str, Any] | None
    initiated_by: str
    created_at: datetime
    model_config = {"from_attributes": True}


class AuditLogOut(BaseModel):
    id: int
    action: str
    entity_type: str
    entity_id: str | None
    actor: str
    details: dict[str, Any] | None
    created_at: datetime
    model_config = {"from_attributes": True}
