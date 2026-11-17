from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from .config import settings


connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def apply_phase2_schema() -> None:
    """Small additive migration for installations created before Phase 2.

    A full migration framework can be introduced later; this keeps existing MVP
    SQLite/PostgreSQL deployments working without dropping their job history.
    """
    additions = {
        "progress": "INTEGER",
        "logs": "JSON",
        "celery_task_id": "VARCHAR(100)",
        "attempt_count": "INTEGER",
        "cancel_requested": "BOOLEAN",
    }
    inspector = inspect(engine)
    if "training_jobs" not in inspector.get_table_names():
        return
    existing = {item["name"] for item in inspector.get_columns("training_jobs")}
    with engine.begin() as connection:
        for name, sql_type in additions.items():
            if name not in existing:
                connection.execute(text(f"ALTER TABLE training_jobs ADD COLUMN {name} {sql_type}"))
        connection.execute(text("UPDATE training_jobs SET status = UPPER(status) WHERE status IS NOT NULL"))
        connection.execute(text("UPDATE training_jobs SET progress = 0 WHERE progress IS NULL"))
        connection.execute(text("UPDATE training_jobs SET attempt_count = 0 WHERE attempt_count IS NULL"))
        connection.execute(text("UPDATE training_jobs SET cancel_requested = FALSE WHERE cancel_requested IS NULL"))
        # JSON literal works for both supported database engines.
        connection.execute(text("UPDATE training_jobs SET logs = '[]' WHERE logs IS NULL"))


def apply_phase3_schema() -> None:
    """Additive migration ensuring all Phase 3 tables exist without dropping existing data."""
    Base.metadata.create_all(bind=engine)
