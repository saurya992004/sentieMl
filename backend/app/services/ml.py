import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy.orm import Session
from ..config import settings
from ..database import SessionLocal
from ..models import Dataset, TrainingJob
from .datasets import read_dataset

logger = logging.getLogger(__name__)
CLASSIFIERS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
}
REGRESSORS = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "gradient_boosting": GradientBoostingRegressor(random_state=42),
}


def _serial(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numerical = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [col for col in features.columns if col not in numerical]
    transformers = []
    if numerical:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numerical))
    if categorical:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    if not transformers:
        raise ValueError("Dataset has no usable feature columns")
    return ColumnTransformer(transformers=transformers)


def _metric(problem: str, y_true, y_pred) -> dict[str, Any]:
    if problem == "classification":
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }
    return {"mae": mean_absolute_error(y_true, y_pred), "rmse": mean_squared_error(y_true, y_pred) ** 0.5, "r2": r2_score(y_true, y_pred)}


def _append_log(job: TrainingJob, message: str) -> None:
    job.logs = [*(job.logs or []), message]


def run_training(job_id: int, raise_on_error: bool = False) -> None:
    """Runs in FastAPI's background executor. Job state remains queryable after restarts."""
    db: Session = SessionLocal()
    try:
        job = db.get(TrainingJob, job_id)
        if not job:
            return
        if job.cancel_requested:
            job.status, job.progress = "CANCELLED", 0
            _append_log(job, "Cancellation acknowledged before training started.")
            db.commit()
            return
        job.status, job.progress = "RUNNING", 5
        _append_log(job, "Worker started training.")
        db.commit()
        dataset = db.get(Dataset, job.dataset_id)
        if not dataset:
            raise ValueError("Dataset no longer exists")
        frame = read_dataset(dataset.file_path)
        if job.target_column not in frame.columns:
            raise ValueError("Selected target column no longer exists")
        data = frame.dropna(subset=[job.target_column]).copy()
        X, y = data.drop(columns=[job.target_column]), data[job.target_column]
        if len(X) < 10 or X.shape[1] == 0:
            raise ValueError("At least 10 rows and one feature column are required")
        if job.problem_type == "classification":
            if y.nunique() < 2 or y.value_counts().min() < 2:
                raise ValueError("Classification needs two classes with at least two samples each")
            stratify = y
        else:
            y = pd.to_numeric(y, errors="raise")
            stratify = None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
        candidates = CLASSIFIERS if job.problem_type == "classification" else REGRESSORS
        invalid = set(job.algorithms) - set(candidates)
        if invalid:
            raise ValueError(f"Unsupported algorithms for {job.problem_type}: {', '.join(sorted(invalid))}")
        settings.upload_path  # ensure paths are present
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        experiment = mlflow.set_experiment(f"sentinelml-{job.problem_type}")
        results: dict[str, Any] = {}
        best_name, best_pipeline, best_score, best_run_id = None, None, None, None
        higher_is_better = job.primary_metric not in {"mae", "rmse"}
        for name in job.algorithms:
            db.refresh(job)
            if job.cancel_requested:
                job.status = "CANCELLED"
                _append_log(job, "Cancellation acknowledged between model runs.")
                db.commit()
                return
            estimator = candidates[name]
            pipeline = Pipeline([("preprocessor", _preprocessor(X)), ("model", estimator)])
            with mlflow.start_run(run_name=f"job-{job.id}-{name}") as run:
                mlflow.log_params({"algorithm": name, "problem_type": job.problem_type, "target_column": job.target_column, "train_rows": len(X_train), "test_rows": len(X_test)})
                pipeline.fit(X_train, y_train)
                values = _metric(job.problem_type, y_test, pipeline.predict(X_test))
                mlflow.log_metrics({key: float(value) for key, value in values.items() if not isinstance(value, list)})
                # MLflow 3 defaults to the restrictive ``skops`` serializer when
                # available. sklearn's fitted pipelines legitimately contain NumPy
                # dtypes that skops flags as untrusted, so use MLflow's established
                # cloudpickle format for our server-created, internal artifacts.
                mlflow.sklearn.log_model(
                    pipeline,
                    name="model",
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                )
                results[name] = {key: _serial(value) for key, value in values.items()}
                job.progress = min(90, 10 + int(80 * len(results) / len(job.algorithms)))
                _append_log(job, f"Completed {name}.")
                db.commit()
                score = float(values[job.primary_metric])
                if best_score is None or (score > best_score if higher_is_better else score < best_score):
                    best_name, best_pipeline, best_score, best_run_id = name, pipeline, score, run.info.run_id
        assert best_pipeline is not None and best_name is not None
        model_name = f"sentinelml_job_{job.id}_best"
        # Create an explicit registered version from the winning run artifact.
        mlflow.register_model(f"runs:/{best_run_id}/model", model_name)
        job.status, job.progress, job.metrics, job.best_algorithm = "COMPLETED", 100, results, best_name
        job.mlflow_experiment_id, job.mlflow_run_id, job.registered_model_name = experiment.experiment_id, best_run_id, model_name
        job.completed_at = datetime.now(timezone.utc)
        _append_log(job, f"Registered winning model {model_name}.")
        db.commit()
    except Exception as exc:
        logger.exception("Training job %s failed", job_id)
        job = db.get(TrainingJob, job_id)
        if job:
            job.status, job.error, job.completed_at = "FAILED", str(exc), datetime.now(timezone.utc)
            _append_log(job, f"Training failed: {exc}")
            db.commit()
        if raise_on_error:
            raise
    finally:
        db.close()

