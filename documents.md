# SentinelML — Project Handoff

This document is the current implementation handoff for contributors.

## Project goal

SentinelML is a web application that accepts CSV datasets, trains selected scikit-learn models, compares true holdout metrics, records experiments and artifacts in MLflow, registers the winning model, and serves predictions.

## Completed: Phase 1 core MVP

- React + TypeScript + Vite dashboard.
- FastAPI backend with SQLAlchemy metadata persistence.
- CSV upload, validation, profiling, missing-value/duplicate detection, and preview.
- Classification: Logistic Regression, Random Forest, Gradient Boosting.
- Regression: Linear Regression, Random Forest, Gradient Boosting.
- Leakage-safe sklearn pipelines: split before fitting preprocessing, median/most-frequent imputation, scaling, and one-hot encoding.
- Real classification and regression metrics, experiment comparison, primary metric selection.
- MLflow experiment runs, metric/parameter/artifact logging, model registry registration, and prediction API.
- Phase 1 MLflow serialization failure was fixed by using MLflow CloudPickle serialization for application-created sklearn pipelines.

## Completed: Phase 2 MLOps foundation

- `docker-compose.yml` defines PostgreSQL, Redis, SeaweedFS (S3-compatible storage), one-time bucket creation, MLflow, FastAPI, Celery worker, and frontend services.
- Persistent Docker volumes: PostgreSQL, Redis, object-storage artifacts, and dataset uploads.
- Health checks and `restart: unless-stopped` for long-running services.
- PostgreSQL is configured for both SentinelML metadata and MLflow tracking/model registry metadata.
- SeaweedFS provides local S3-compatible MLflow artifact storage. AWS S3 deployment settings are documented in `README.md`.
- Celery with Redis replaced FastAPI in-process background tasks.
- Training jobs have durable `QUEUED`, `RUNNING`, `COMPLETED`, `FAILED`, and `CANCELLED` states, progress, logs, task ID, retry count, and cooperative cancellation.
- Existing Phase 1 databases receive an additive startup migration for the new job columns. No data is dropped.
- Experiment view now displays job state, progress, and retry attempt.
- GitHub Actions workflow in `.github/workflows/ci.yml` runs backend tests, frontend build, Compose validation, and image-build preparation.

## Important files

| File | Responsibility |
| --- | --- |
| `backend/app/services/ml.py` | Model training, evaluation, MLflow logging/registration, job progress/log updates. |
| `backend/app/tasks.py` | Celery task with controlled retry behavior. |
| `backend/app/celery_app.py` | Redis broker/result configuration. |
| `backend/app/routers/training.py` | Queue submission, job read APIs, cancellation endpoint. |
| `backend/app/database.py` | Database engine/session plus safe Phase 2 additive schema migration. |
| `docker-compose.yml` | Local deployment topology. |
| `backend/Dockerfile`, `frontend/Dockerfile` | API/worker and dashboard images. |
| `.env.example` | Required configuration keys; no production secrets. |
| `README.md` | Setup and operating guide. |

## Verification performed

On the current codebase:

- Backend tests: `3 passed`.
- Frontend: `npm run build` passed.
- Compose syntax: `docker compose --env-file .env.example config --quiet` passed.
- API/worker Docker image build could not be completed earlier because Docker Desktop's Linux engine was unavailable to the agent.

## Current Docker fix

Docker Hub denied both upstream `minio/mc` and `minio/minio` images on this machine. The local S3-compatible service now uses the verified `chrislusf/seaweedfs:4.05` image. A pinned `amazon/aws-cli:2.40.3` one-shot container creates the required artifact bucket through SeaweedFS's S3 API.

## How to run locally with Docker

1. Ensure Docker Desktop is running and set to Linux containers.
2. Update `.env` with all values from `.env.example`; change placeholder passwords.
3. Start the stack:

```powershell
docker compose --env-file .env.example up --build
```

For normal development use an actual `.env` file instead of `.env.example`:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

4. Confirm:

```powershell
docker compose ps
curl http://localhost:8000/health
```

Dashboard: `http://localhost:5173`; API docs: `http://localhost:8000/docs`; MLflow: `http://localhost:5000`; local S3 endpoint: `http://localhost:8333`.

## Work still needed

### Required validation before calling Phase 2 finished

- Run `docker compose up --build` successfully after the MinIO helper change.
- Upload a CSV through the containerized dashboard.
- Start a training job and confirm it becomes `QUEUED → RUNNING → COMPLETED` in the dashboard.
- Confirm the worker logs include training progress and the model appears in MLflow Model Registry.
- Submit a prediction using the registered model.
- Restart the stack (`docker compose down`, then `up`) and verify datasets, MLflow runs, and registered model versions persist.

### Suggested next implementation tasks

- Add Alembic migrations and replace the small additive startup migration once schema changes become frequent.
- Add authentication, authorization, and per-user/project dataset ownership.
- Add structured task log streaming (WebSocket/SSE) rather than the current dashboard polling.
- Add rate limits, file scanning/object storage for datasets, and production secret management.
- Pin image digests and dependency versions for a hardened production deployment.
- Add Docker-based end-to-end tests to CI.
- Add monitoring/alerts and worker autoscaling for production.

### Explicitly out of scope for now

- Intelligent automated model selection.
- Automated retraining and drift-triggered retraining.
- Phase 3 capabilities.

## GitHub sharing checklist

- Do not commit `.env`, local SQLite databases, `mlruns/`, `uploads/`, virtual environments, or `node_modules/`.
- Commit `.env.example`, `README.md`, this `documents.md`, source, Docker files, and `.github/workflows/ci.yml`.
- Ask collaborators to create their own `.env` from `.env.example` and choose unique passwords.
