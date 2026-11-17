# SentinelML — MLOps Platform

SentinelML lets users upload CSV datasets, train and compare scikit-learn models, track runs in MLflow, register winners, and serve predictions. Phase 2 makes the Phase 1 MVP deployable with containers, PostgreSQL, Redis/Celery, and persistent MLflow artifacts.

## Included

- Dataset upload, validation, profiling, missing-value/duplicate analysis, and preview.
- Reproducible classification and regression pipelines with leakage-safe preprocessing.
- MLflow experiments, metrics, artifacts, Model Registry versions, and prediction API.
- Redis/Celery-backed training jobs with queue status, progress, logs, controlled retries, and cancellation requests.
- React/Vite dashboard, Docker Compose stack, persistent volumes, health checks, and GitHub Actions CI.

## Architecture

`Frontend → FastAPI → Redis/Celery worker → PostgreSQL + MLflow → S3-compatible artifacts`

SentinelML metadata and job state are stored in the `sentinelml` PostgreSQL database. MLflow uses its separate `mlflow` PostgreSQL database for tracking/registry metadata and SeaweedFS locally (or S3 in cloud) for artifacts. The API and worker share a persistent upload volume.

## Docker quick start

```powershell
Copy-Item .env.example .env
# Change the placeholder passwords in .env
docker compose up --build
```

Open the dashboard at `http://localhost:5173`, API docs at `http://localhost:8000/docs`, MLflow at `http://localhost:5000`, and the local S3-compatible endpoint at `http://localhost:8333`.

Compose runs PostgreSQL, Redis, SeaweedFS, MLflow, API, Celery worker, and frontend. Named volumes preserve databases, artifacts, Redis state, and uploads. Stop with `docker compose down`; do not use `-v` unless you intend to delete all persistent data.

## MLflow configuration

**Local Docker:** configured automatically against PostgreSQL and SeaweedFS. API and workers use `MLFLOW_TRACKING_URI=http://mlflow:5000`.

**AWS/S3:** run MLflow using a PostgreSQL `--backend-store-uri` and `--artifacts-destination s3://YOUR_BUCKET`. Set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_DEFAULT_REGION` for MLflow, API, and worker. Set `MLFLOW_S3_ENDPOINT_URL` only for non-AWS S3-compatible storage. Never commit credentials.

## Background training

`POST /training/start` persists a `QUEUED` job and sends it to a Celery worker. `GET /training/{job_id}` returns status (`QUEUED`, `RUNNING`, `COMPLETED`, `FAILED`, or `CANCELLED`), progress, logs, retry attempts, and MLflow IDs. `POST /training/{job_id}/cancel` requests safe cancellation between candidate runs. Unexpected infrastructure failures retry up to `TRAINING_MAX_RETRIES` (default 2).

## Local development

```powershell
Copy-Item .env.example .env
python -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
uvicorn app.main:app --app-dir backend --reload
```

For background training outside Docker, start Redis and then in another terminal:

```powershell
backend\.venv\Scripts\celery.exe -A app.celery_app.celery_app worker --workdir backend --loglevel=INFO --queues=training --pool=solo
```

Start the frontend separately:

```powershell
cd frontend
npm install
npm run dev
```

## API

- `POST /datasets/upload`, `GET /datasets`, `GET /datasets/{id}`
- `POST /training/start`, `GET /training`, `GET /training/{job_id}`, `POST /training/{job_id}/cancel`
- `GET /experiments`, `GET /models`, `GET /models/{model_name}`, `POST /predict`

## Tests and CI

```powershell
cd backend; .\.venv\Scripts\python.exe -m pytest -q
cd frontend; npm run build
docker compose config
```

GitHub Actions runs backend tests, frontend build, Compose validation, and service-image builds on pushes and pull requests to `main`.

## Current limitations

Cancellation is cooperative: the worker observes requests between candidate model runs. Authentication, role controls, dataset deletion, and automated retraining are intentionally not part of this phase.
