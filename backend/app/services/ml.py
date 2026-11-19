import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import cloudpickle
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy.orm import Session
from ..config import settings
from ..database import SessionLocal
from ..models import AuditLog, Dataset, ModelEvaluation, ModelLifecycle, TrainingJob
from .datasets import read_dataset

logger = logging.getLogger(__name__)

SUPPORTED_CLASSIFIERS = {
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "decision_tree",
    "extra_trees",
}

SUPPORTED_REGRESSORS = {
    "linear_regression",
    "ridge",
    "lasso",
    "random_forest",
    "gradient_boosting",
    "decision_tree",
    "extra_trees",
}


def build_estimator(name: str, problem_type: str, user_hyperparams: dict[str, Any] | None = None) -> Any:
    hp = dict(user_hyperparams or {})
    algo_hp = hp.get(name, {}) if isinstance(hp.get(name), dict) else hp

    n_estimators = int(algo_hp.get("n_estimators", 200))
    raw_depth = algo_hp.get("max_depth")
    max_depth = int(raw_depth) if raw_depth not in (None, "", "None", "null") else None
    learning_rate = float(algo_hp.get("learning_rate", 0.1))
    c_val = float(algo_hp.get("C", 1.0))
    alpha_val = float(algo_hp.get("alpha", 1.0))

    if problem_type == "classification":
        if name == "logistic_regression":
            return LogisticRegression(C=c_val, max_iter=1000, random_state=42)
        if name == "random_forest":
            return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        if name == "gradient_boosting":
            return GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth or 3, learning_rate=learning_rate, random_state=42)
        if name == "decision_tree":
            return DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        if name == "extra_trees":
            return ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        raise ValueError(f"Unsupported classification algorithm: {name}")

    if name == "linear_regression":
        return LinearRegression()
    if name == "ridge":
        return Ridge(alpha=alpha_val, random_state=42)
    if name == "lasso":
        return Lasso(alpha=alpha_val, random_state=42)
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
    if name == "gradient_boosting":
        return GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth or 3, learning_rate=learning_rate, random_state=42)
    if name == "decision_tree":
        return DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    if name == "extra_trees":
        return ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
    raise ValueError(f"Unsupported regression algorithm: {name}")


CLASSIFIERS = {name: build_estimator(name, "classification") for name in SUPPORTED_CLASSIFIERS}
REGRESSORS = {name: build_estimator(name, "regression") for name in SUPPORTED_REGRESSORS}


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


def _metric(problem: str, y_true, y_pred, y_prob=None) -> dict[str, Any]:
    if problem == "classification":
        res = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }
        if y_prob is not None:
            try:
                if y_prob.ndim == 1 or y_prob.shape[1] == 2:
                    prob_col = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
                    res["roc_auc"] = float(roc_auc_score(y_true, prob_col))
                elif y_prob.shape[1] > 2:
                    res["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
            except Exception:
                pass
        return res
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "r2": r2_score(y_true, y_pred),
    }


def _append_log(job: TrainingJob, message: str) -> None:
    job.logs = [*(job.logs or []), message]


def run_training(job_id: int, raise_on_error: bool = False) -> None:
    """Runs training, logs multi-objective evaluations and registers winning candidate."""
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
            estimator = build_estimator(name, job.problem_type, job.hyperparameters)
            pipeline = Pipeline([("preprocessor", _preprocessor(X)), ("model", estimator)])
            with mlflow.start_run(run_name=f"job-{job.id}-{name}") as run:
                base_params = {
                    "algorithm": name,
                    "problem_type": job.problem_type,
                    "target_column": job.target_column,
                    "train_rows": len(X_train),
                    "test_rows": len(X_test),
                }
                # Log tuned hyperparameters to MLflow
                for k, v in estimator.get_params().items():
                    if k in {"n_estimators", "max_depth", "learning_rate", "C", "alpha", "max_iter"}:
                        base_params[f"hp_{k}"] = str(v)
                mlflow.log_params(base_params)
                
                # Multi-objective measurement
                t0 = time.perf_counter()
                pipeline.fit(X_train, y_train)
                train_time = round(time.perf_counter() - t0, 4)

                t1 = time.perf_counter()
                y_pred = pipeline.predict(X_test)
                latency_ms = round((time.perf_counter() - t1) / max(1, len(X_test)) * 1000.0, 4)

                y_prob = None
                if job.problem_type == "classification" and hasattr(pipeline, "predict_proba"):
                    try:
                        y_prob = pipeline.predict_proba(X_test)
                    except Exception:
                        y_prob = None

                values = _metric(job.problem_type, y_test, y_pred, y_prob)

                size_kb = 0.0
                try:
                    size_kb = round(len(cloudpickle.dumps(pipeline)) / 1024.0, 2)
                except Exception:
                    pass

                values["training_time_seconds"] = train_time
                values["inference_latency_ms"] = latency_ms
                values["model_size_kb"] = size_kb

                mlflow.log_metrics({key: float(value) for key, value in values.items() if not isinstance(value, list)})
                mlflow.sklearn.log_model(
                    pipeline,
                    name="model",
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                )
                results[name] = {key: _serial(value) for key, value in values.items()}
                
                # Record ModelEvaluation
                db.add(
                    ModelEvaluation(
                        model_name=f"sentinelml_job_{job.id}_{name}",
                        version="1",
                        algorithm=name,
                        problem_type=job.problem_type,
                        metrics={k: _serial(v) for k, v in values.items() if k not in {"training_time_seconds", "inference_latency_ms", "model_size_kb"}},
                        training_time_seconds=train_time,
                        inference_latency_ms=latency_ms,
                        model_size_kb=size_kb,
                    )
                )

                job.progress = min(90, 10 + int(80 * len(results) / len(job.algorithms)))
                _append_log(job, f"Completed {name} (train: {train_time}s, latency: {latency_ms}ms, size: {size_kb}KB).")
                db.commit()
                score = float(values[job.primary_metric])
                if best_score is None or (score > best_score if higher_is_better else score < best_score):
                    best_name, best_pipeline, best_score, best_run_id = name, pipeline, score, run.info.run_id
        assert best_pipeline is not None and best_name is not None
        model_name = f"sentinelml_job_{job.id}_best"
        reg_model = mlflow.register_model(f"runs:/{best_run_id}/model", model_name)
        v_str = str(getattr(reg_model, "version", "1"))

        job.status, job.progress, job.metrics, job.best_algorithm = "COMPLETED", 100, results, best_name
        job.mlflow_experiment_id, job.mlflow_run_id, job.registered_model_name = experiment.experiment_id, best_run_id, model_name
        job.completed_at = datetime.now(timezone.utc)
        _append_log(job, f"Registered winning candidate {model_name} v{v_str}.")

        # Add to ModelLifecycle
        db.add(
            ModelLifecycle(
                model_name=model_name,
                version=v_str,
                stage="CANDIDATE",
                run_id=best_run_id,
                training_job_id=job.id,
                metrics=results[best_name],
                parameters={"algorithm": best_name, "problem_type": job.problem_type, "primary_metric": job.primary_metric},
            )
        )
        # Audit log
        db.add(
            AuditLog(
                action="MODEL_TRAINED",
                entity_type="MODEL",
                entity_id=model_name,
                actor="system",
                details={"job_id": job.id, "best_algorithm": best_name, "version": v_str, "primary_metric": job.primary_metric, "score": best_score},
            )
        )
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
