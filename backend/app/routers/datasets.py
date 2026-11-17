import shutil
from pathlib import Path
from uuid import uuid4
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..config import settings
from ..database import get_db
from ..models import Dataset
from ..schemas import DatasetDetail, DatasetOut
from ..services.datasets import describe, preview, read_dataset

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/upload", response_model=DatasetOut, status_code=201)
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Only CSV files are supported")
    destination = settings.upload_path / f"{uuid4()}_{Path(file.filename).name}"
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        if destination.stat().st_size > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB limit")
        frame = read_dataset(destination)
        columns, missing, duplicates = describe(frame)
        item = Dataset(name=Path(file.filename).name, file_path=str(destination), row_count=len(frame), column_count=len(frame.columns), columns=columns, missing_values=missing, duplicate_rows=duplicates)
        db.add(item); db.commit(); db.refresh(item)
        return item
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


@router.get("", response_model=list[DatasetOut])
def list_datasets(db: Session = Depends(get_db)):
    return db.scalars(select(Dataset).order_by(Dataset.created_at.desc())).all()


@router.get("/{dataset_id}", response_model=DatasetDetail)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    item = db.get(Dataset, dataset_id)
    if not item:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {**DatasetOut.model_validate(item).model_dump(), "preview": preview(read_dataset(item.file_path))}

