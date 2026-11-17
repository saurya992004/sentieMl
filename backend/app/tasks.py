import logging
from celery import Task
from .celery_app import celery_app
from .config import settings
from .database import SessionLocal
from .models import TrainingJob
from .services.ml import run_training

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=settings.training_max_retries, default_retry_delay=15)
def train_model(self: Task, job_id: int) -> dict:
    """Execute one persisted training job, retrying unexpected infrastructure failures."""
    db = SessionLocal()
    try:
        job = db.get(TrainingJob, job_id)
        if not job:
            return {"job_id": job_id, "status": "MISSING"}
        if job.cancel_requested:
            job.status, job.progress = "CANCELLED", 0
            job.logs = [*job.logs, "Cancellation acknowledged before worker start."]
            db.commit()
            return {"job_id": job_id, "status": "CANCELLED"}
        job.attempt_count = self.request.retries + 1
        job.celery_task_id = self.request.id
        db.commit()
    finally:
        db.close()
    try:
        run_training(job_id, raise_on_error=True)
        return {"job_id": job_id, "status": "COMPLETED"}
    except Exception as exc:
        if self.request.retries < self.max_retries:
            logger.warning("Retrying training job %s after failure: %s", job_id, exc)
            raise self.retry(exc=exc)
        raise
