from pathlib import Path
import pandas as pd
from fastapi import HTTPException
from ..models import Dataset


def read_dataset(path: str | Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Unable to parse CSV: {exc}") from exc
    if frame.empty or len(frame.columns) == 0:
        raise HTTPException(status_code=422, detail="CSV must contain at least one row and one column")
    if frame.columns.duplicated().any():
        raise HTTPException(status_code=422, detail="CSV has duplicate column names")
    return frame


def describe(frame: pd.DataFrame) -> tuple[dict[str, str], dict[str, int], int]:
    return (
        {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        {str(column): int(count) for column, count in frame.isna().sum().items()},
        int(frame.duplicated().sum()),
    )


def preview(frame: pd.DataFrame, rows: int = 20) -> list[dict]:
    return frame.head(rows).where(pd.notnull(frame), None).to_dict(orient="records")

