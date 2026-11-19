import shutil
from pathlib import Path
from uuid import uuid4
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..config import settings
from ..database import get_db
from ..models import AuditLog, Dataset, DatasetProfile
from ..schemas import DatasetDetail, DatasetOut, DatasetProfileOut
from ..services.datasets import describe, preview, read_dataset
from ..services.profiler import profile_dataset

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
        item = Dataset(
            name=Path(file.filename).name,
            file_path=str(destination),
            row_count=len(frame),
            column_count=len(frame.columns),
            columns=columns,
            missing_values=missing,
            duplicate_rows=duplicates,
        )
        db.add(item)
        db.commit()
        db.refresh(item)

        # Automated dataset profiling on upload
        profile_res = profile_dataset(frame)
        profile_record = DatasetProfile(
            dataset_id=item.id,
            quality_score=profile_res["quality_score"],
            warnings=profile_res["warnings"],
            statistics=profile_res["statistics"],
            outliers=profile_res["outliers"],
            correlations=profile_res["correlations"],
            target_analysis=profile_res["target_analysis"],
            suggested_operations=profile_res["suggested_operations"],
            risks=profile_res["risks"],
            ai_summary=profile_res.get("ai_summary"),
        )
        db.add(profile_record)

        # Audit log
        db.add(
            AuditLog(
                action="DATASET_UPLOAD",
                entity_type="DATASET",
                entity_id=str(item.id),
                actor="user",
                details={"name": item.name, "rows": item.row_count, "cols": item.column_count, "quality_score": profile_res["quality_score"]},
            )
        )
        db.commit()
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


@router.post("/{dataset_id}/profile", response_model=DatasetProfileOut)
def run_profile(
    dataset_id: int,
    target_column: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    item = db.get(Dataset, dataset_id)
    if not item:
        raise HTTPException(status_code=404, detail="Dataset not found")
    frame = read_dataset(item.file_path)
    if target_column and target_column not in frame.columns:
        raise HTTPException(status_code=422, detail=f"Target column '{target_column}' not found in dataset")
    profile_res = profile_dataset(frame, target_column=target_column)

    profile_record = DatasetProfile(
        dataset_id=item.id,
        quality_score=profile_res["quality_score"],
        warnings=profile_res["warnings"],
        statistics=profile_res["statistics"],
        outliers=profile_res["outliers"],
        correlations=profile_res["correlations"],
        target_analysis=profile_res["target_analysis"],
        suggested_operations=profile_res["suggested_operations"],
        risks=profile_res["risks"],
    )
    db.add(profile_record)
    db.add(
        AuditLog(
            action="DATASET_PROFILED",
            entity_type="DATASET",
            entity_id=str(item.id),
            actor="user",
            details={"target_column": target_column, "quality_score": profile_res["quality_score"]},
        )
    )
    db.commit()
    db.refresh(profile_record)
    return profile_record


@router.get("/{dataset_id}/profile", response_model=DatasetProfileOut)
def get_profile(dataset_id: int, db: Session = Depends(get_db)):
    item = db.get(Dataset, dataset_id)
    if not item:
        raise HTTPException(status_code=404, detail="Dataset not found")
    profile_record = db.scalars(
        select(DatasetProfile)
        .where(DatasetProfile.dataset_id == dataset_id)
        .order_by(DatasetProfile.created_at.desc())
    ).first()
    if not profile_record:
        # Generate profiling on-the-fly if not found
        frame = read_dataset(item.file_path)
        profile_res = profile_dataset(frame)
        profile_record = DatasetProfile(
            dataset_id=item.id,
            quality_score=profile_res["quality_score"],
            warnings=profile_res["warnings"],
            statistics=profile_res["statistics"],
            outliers=profile_res["outliers"],
            correlations=profile_res["correlations"],
            target_analysis=profile_res["target_analysis"],
            suggested_operations=profile_res["suggested_operations"],
            risks=profile_res["risks"],
            ai_summary=profile_res.get("ai_summary"),
        )
        db.add(profile_record)
        db.commit()
        db.refresh(profile_record)
    return profile_record
