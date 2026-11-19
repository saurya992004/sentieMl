# SentinelML — Intelligent MLOps & Autonomous ML Lifecycle Platform (Phase 3)

SentinelML is an enterprise-grade, intelligent MLOps platform built around **MLflow**, **FastAPI**, **Celery/Redis**, **PostgreSQL**, and **React**.

While Phase 1 established the Core ML training MVP and Phase 2 built the production container infrastructure, **Phase 3 transforms SentinelML into an intelligent, partially autonomous MLOps platform** featuring automated dataset profiling, algorithmic recommendation, multi-objective model selection, SHAP explainability, drift detection, automated retraining orchestration, human-in-the-loop approvals, model version promotion, and safe rollback.

---

## ◈ Core Architecture & MLflow Foundation

```
                                  ┌───────────────────────────┐
                                  │      React Dashboard      │
                                  │   (Vite + TypeScript)     │
                                  └─────────────┬─────────────┘
                                                │ REST API
                                                ▼
                                  ┌───────────────────────────┐
                                  │      FastAPI Backend      │
                                  │   (Orchestrator & APIs)   │
                                  └──────┬──────────────┬─────┘
                                         │              │
                   Dispatches Jobs       │              │ Tracks Metadata
                          ▼              │              ▼
             ┌─────────────────────────┐ │   ┌───────────────────────────┐
             │      Redis Broker       │ │   │    PostgreSQL Database    │
             │   (Celery Task Queue)   │ │   │  - sentinelml (App data)  │
             └────────────┬────────────┘ │   │  - mlflow (Tracking db)   │
                          │              │   └──────────────▲────────────┘
                          ▼              │                  │
             ┌─────────────────────────┐ │                  │
             │   Celery Worker Nodes   │ │                  │
             │ (Scikit-Learn Training) │ │                  │
             └────────────┬────────────┘ │                  │
                          │              │                  │
                          ▼              ▼                  │
             ┌──────────────────────────────────────────────┴───────────┐
             │                      MLflow Server                       │
             │  - Experiment Tracking (Parameters, Metrics, Tags)       │
             │  - Model Registry (Versioned CloudPickle Pipelines)      │
             │  - S3 / SeaweedFS Object Storage (Model Artifacts)       │
             └──────────────────────────────────────────────────────────┘
```

### Is this an MLflow project?
**Yes, 100%! MLflow is the central backbone of SentinelML:**
1. **MLflow Tracking**: Every model training run logs hyperparameters (`algorithm`, `train_rows`, `test_rows`, etc.), multi-objective metrics (`accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `mae`, `rmse`, `r2`, `training_time_seconds`, `inference_latency_ms`, `model_size_kb`), and tags.
2. **MLflow Artifact Storage**: Fitted pipelines are serialized with CloudPickle and stored in an S3-compatible artifact repository (SeaweedFS locally, Amazon S3 in production).
3. **MLflow Model Registry**: Winning candidate models are registered under formal versioned names (`sentinelml_job_{id}_best`), supporting lifecycle states (`DEVELOPMENT`, `CANDIDATE`, `APPROVED`, `PRODUCTION`, `ARCHIVED`).
4. **MLflow Model Serving & Inference**: The live prediction service (`/predict`) dynamically loads registered models directly from MLflow (`models:/{model_name}/{version}`) via `mlflow.pyfunc.load_model`.
5. **Explainability with MLflow**: SHAP and feature attribution inspect models loaded directly from the MLflow registry to ensure parity with the exact deployed artifact.

---

## ◈ Implemented Capabilities (Phase 1, 2, & 3)

### 1. Automated Dataset Profiling & Quality Intelligence (Step 1)
- Automatic profiling on upload: row count, feature types, missingness, duplicate rows, and unique value counts.
- Descriptive statistics (mean, std, min, 25%, median, 75%, max, skewness).
- Outlier analysis using interquartile range (IQR bounds and counts).
- Pearson feature correlation matrix and collinearity warning flags.
- Target variable distribution analysis and class imbalance ratio calculation.
- Deterministic Dataset Quality Score (0–100) with rule-based deductions.
- Preprocessing suggestions and potential risk identification before training.

### 2. Intelligent Model Recommendation Engine (Step 2)
- Transparent, rule-based recommendation engine evaluating sample size, dimensional ratios, categorical cardinality, outliers, and class imbalance.
- Covers Classification (Logistic Regression, Random Forest, Gradient Boosting) and Regression (Linear Regression, Random Forest Regressor, Gradient Boosting Regressor).
- Provides suitability scores (0–100), estimated training speed, computational footprint, and clear textual reasoning for each algorithm.

### 3. Multi-Objective Model Selection & Comparison (Step 3)
- Evaluates models beyond a single accuracy number: Precision, Recall, F1, ROC-AUC, MAE, RMSE, R², training duration, inference latency (ms), and artifact size (KB).
- Normalizes and ranks models according to user-configurable optimization priorities (e.g., latency-first for real-time vs. recall-first for fraud/medical diagnosis).

### 4. Model Explainability & SHAP Integration (Step 4)
- Global feature importance: computes Shapley value attributions or model-intrinsic attributions.
- Local prediction explanations: explains individual predictions and feature contributions.
- Transparent disclaimer: feature importance reflects predictive association, not causality.

### 5. Advanced Data Drift & Performance Monitoring (Step 5)
- Numerical drift detection using two-sample Kolmogorov-Smirnov test and Population Stability Index (PSI).
- Categorical drift detection using distribution divergence (Total Variation Distance).
- Overall dataset drift severity calculation (`NONE`, `LOW`, `MODERATE`, `CRITICAL`) with recommended action.
- Ground truth submission and baseline degradation tracking.

### 6. Intelligent Retraining Orchestration (Step 6)
- Configurable retraining triggers: feature drift, metric degradation, scheduled, or manual.
- Automated pipeline execution in background workers, candidate evaluation, comparison against current production baseline, and approval request generation.
- Safeguards: duplicate run prevention, cooldown enforcement, and lineage preservation.

### 7. Model Governance, Approval & Promotion Workflow (Step 7)
- Formal lifecycle stages: `DEVELOPMENT` → `CANDIDATE` → `APPROVED` → `PRODUCTION` → `ARCHIVED`.
- Human-in-the-loop review interface with candidate vs. production comparison and audit comments.

### 8. Model Versioning & Safe Rollback (Step 8)
- Zero-downtime rollback mechanism: instantly restores the previous approved production version.
- Preserves complete immutable version history in MLflow and database audit logs.

### 9. Automation Center & Audit Logging (Step 9 & 12)
- Centralized rule management and audit log recording for uploads, training, predictions, approvals, promotions, and rollbacks.

### 10. Advanced React Dashboard (Step 10)
- Unified dashboard covering all 11 views: Overview, Dataset Intelligence, Model Recommendations, Training & Experiments, Model Comparison, Explainability, Drift Monitoring, Automated Retraining, Governance, Deployments, and Automation.

---

## ◈ Quick Start with Docker

```powershell
# 1. Create your local environment configuration
Copy-Item .env.example .env

# 2. Build and launch all services with Docker Compose
docker compose up --build
```

### Services Port Map
| Service | URL | Function |
|---|---|---|
| Dashboard | `http://localhost:5173` | React TypeScript Dashboard |
| FastAPI | `http://localhost:8000/docs` | Swagger / OpenAPI Documentation |
| MLflow UI | `http://localhost:5000` | MLflow Experiment Tracking & Model Registry |
| Object Storage | `http://localhost:8333` | SeaweedFS (S3-compatible MLflow Artifacts) |
| PostgreSQL | `localhost:5432` | Databases: `sentinelml` and `mlflow` |
| Redis | `localhost:6379` | Celery Task Queue Broker |

---

## ◈ Local Development (Outside Docker)

### 1. Backend Setup
```powershell
python -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
uvicorn app.main:app --app-dir backend --reload
```

### 2. Celery Worker Setup
```powershell
celery -A app.celery_app.celery_app worker --workdir backend --loglevel=INFO --queues=training --pool=solo
```

### 3. Frontend Setup
```powershell
cd frontend
npm install
npm run dev
```

---

## ◈ Continuous Integration (CI / GitHub Actions)

The repository includes a comprehensive GitHub Actions CI pipeline in `.github/workflows/ci.yml`:
- **Python test suite**: runs all 13 backend test suites (`pytest backend/tests`).
- **Frontend build**: verifies TypeScript compilation and Vite production build (`npm run build`).
- **Docker Compose validation**: verifies compose configuration syntax.
- **Image build check**: builds deployable Docker images for API, Worker, and Dashboard.
- **Triggers**: runs on all pushes and pull requests to both `main` and `arjun-agrawal`.

---

## ◈ Verification & Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-8.4.2, pluggy-1.7.0
rootdir: D:\sentieMl\backend
configfile: pytest.ini
collected 13 items

backend\tests\test_api.py ..                                             [ 15%]
backend\tests\test_complete_phase3.py .                                  [ 23%]
backend\tests\test_datasets.py .                                         [ 30%]
backend\tests\test_phase3_e2e.py .....                                   [ 69%]
backend\tests\test_profiler.py ..                                        [ 84%]
backend\tests\test_recommender.py ..                                     [100%]

============================= 13 passed in 18.24s =============================
```
