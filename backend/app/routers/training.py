from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import Dataset, TrainingJob
from ..schemas import JobOut, TrainingStart
from ..tasks import train_model

router = APIRouter(prefix="/training", tags=["training"])


@router.post("/start", response_model=JobOut, status_code=202)
def start_training(payload: TrainingStart, db: Session = Depends(get_db)):
    dataset = db.get(Dataset, payload.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if payload.target_column not in dataset.columns:
        raise HTTPException(status_code=422, detail="Target column is not in the selected dataset")
    metric = payload.primary_metric or ("f1" if payload.problem_type == "classification" else "rmse")
    permitted = {"accuracy", "precision", "recall", "f1"} if payload.problem_type == "classification" else {"mae", "rmse", "r2"}
    if metric not in permitted:
        raise HTTPException(status_code=422, detail=f"Primary metric must be one of: {', '.join(sorted(permitted))}")
    job = TrainingJob(**payload.model_dump(exclude={"primary_metric"}), primary_metric=metric)
    db.add(job); db.commit(); db.refresh(job)
    task = train_model.apply_async(args=[job.id])
    job.celery_task_id = task.id
    job.logs = ["Queued for a Celery training worker."]
    db.commit(); db.refresh(job)
    return job


@router.get("", response_model=list[JobOut])
def list_training(db: Session = Depends(get_db)):
    return db.scalars(select(TrainingJob).order_by(TrainingJob.created_at.desc())).all()


@router.get("/{job_id}", response_model=JobOut)
def get_training(job_id: int, db: Session = Depends(get_db)):
    job = db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return job


@router.post("/{job_id}/cancel", response_model=JobOut)
def cancel_training(job_id: int, db: Session = Depends(get_db)):
    job = db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    if job.status in {"COMPLETED", "FAILED", "CANCELLED"}:
        raise HTTPException(status_code=409, detail="Completed jobs cannot be cancelled")
    job.cancel_requested = True
    job.logs = [*(job.logs or []), "Cancellation requested."]
    db.commit(); db.refresh(job)
    return job

