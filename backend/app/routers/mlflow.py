from typing import Any
import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import APIRouter, HTTPException
from mlflow.tracking import MlflowClient
from ..config import settings
from ..schemas import PredictionRequest, PredictionResponse

router = APIRouter(tags=["mlflow"])


def client() -> MlflowClient:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return MlflowClient()


@router.get("/experiments")
def experiments() -> list[dict[str, Any]]:
    return [{"experiment_id": x.experiment_id, "name": x.name, "lifecycle_stage": x.lifecycle_stage} for x in client().search_experiments()]


@router.get("/models")
def models() -> list[dict[str, Any]]:
    return [{"name": x.name, "description": x.description, "latest_versions": [{"version": v.version, "run_id": v.run_id, "status": v.status} for v in x.latest_versions]} for x in client().search_registered_models()]


@router.get("/models/{model_name}")
def model_detail(model_name: str) -> dict[str, Any]:
    try:
        registered = client().get_registered_model(model_name)
        versions = client().search_model_versions(f"name='{model_name}'")
        return {"name": registered.name, "description": registered.description, "versions": [{"version": x.version, "run_id": x.run_id, "status": x.status, "creation_timestamp": x.creation_timestamp} for x in versions]}
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Registered model not found") from exc


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    try:
        resolved_version = payload.version
        if resolved_version == "latest":
            versions = client().search_model_versions(f"name='{payload.model_name}'")
            if not versions:
                raise ValueError("No versions are registered for this model")
            resolved_version = max(versions, key=lambda item: int(item.version)).version
        model = mlflow.pyfunc.load_model(f"models:/{payload.model_name}/{resolved_version}")
        values = model.predict(pd.DataFrame(payload.records))
        return PredictionResponse(model_name=payload.model_name, version=resolved_version, predictions=[x.item() if hasattr(x, "item") else x for x in values])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc

