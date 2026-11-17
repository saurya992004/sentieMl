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


class TrainingStart(BaseModel):
    dataset_id: int
    target_column: str = Field(min_length=1)
    problem_type: Literal["classification", "regression"]
    algorithms: list[str] = Field(min_length=1, max_length=3)
    primary_metric: str | None = None


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

