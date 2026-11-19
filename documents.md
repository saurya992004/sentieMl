# SentinelML — Phase 3 Implementation & Project Handoff

This document details the completed Phase 3 architecture, features, and verification for contributors and examiners.

## 1. Project Overview & MLOps Identity

SentinelML is a full-stack, enterprise-grade MLOps platform. It uses **MLflow** as its foundational experiment tracking, model artifact storage, and model registry engine, orchestrated by **FastAPI** and **Celery/Redis** behind a modern **React + TypeScript** dashboard.

## 2. Phase 3 Implemented Capabilities

### Step 1: Automated Dataset Profiling
- **Service**: `backend/app/services/profiler.py`
- Automatically inspects shape, numerical/categorical features, missing percentages, duplicate counts, and unique cardinality.
- Computes comprehensive descriptive statistics and IQR outlier bounds.
- Computes Pearson correlation matrix to flag multicollinear pairs.
- Calculates transparent **Dataset Quality Score (0–100)** with documented penalty rules.
- Identifies class imbalances and data leakage indicators.

### Step 2: Intelligent Model Recommendation Engine
- **Service**: `backend/app/services/recommender.py`
- Transparent, rule-based algorithmic evaluation based on dataset characteristics, dimensional ratios, outlier presence, and class imbalance.
- Recommends Classification (Logistic Regression, Random Forest, Gradient Boosting) and Regression (Linear Regression, Random Forest, Gradient Boosting) models.
- Provides suitability score, estimated training duration, computational complexity, and human-readable explanation.

### Step 3: Multi-Objective Model Selection
- **Service**: `backend/app/services/comparison.py`
- Evaluates models across predictive metrics (Accuracy, Precision, Recall, F1, ROC-AUC, MAE, RMSE, R²), training duration, inference latency (ms), and model artifact size (KB).
- Features configurable multi-objective weighting and ranking.

### Step 4: Model Explainability (SHAP)
- **Service**: `backend/app/services/explainability.py`
- Computes global feature importance using SHAP Shapley values (with fallback to intrinsic attributions).
- Provides local sample-level feature contributions.
- Emphasizes statistical association vs. causality distinction.

### Step 5: Advanced Data Drift & Performance Monitoring
- **Service**: `backend/app/services/monitoring.py`
- Evaluates feature-level distribution drift using the two-sample Kolmogorov-Smirnov test and Population Stability Index (PSI) for numerical features, and Total Variation Distance for categorical features.
- Computes overall dataset drift severity (`NONE`, `LOW`, `MODERATE`, `CRITICAL`).
- Supports production ground-truth submission and metric degradation tracking.

### Step 6: Intelligent Retraining Orchestration
- **Service**: `backend/app/services/retraining.py`
- Supports configurable triggers (drift threshold, performance degradation, scheduled, manual).
- Executes retraining in background Celery workers, compares candidate against production baseline, and creates an approval request.
- Enforces duplicate prevention and cooldown periods.

### Step 7 & 8: Model Governance, Promotion & Rollback
- State transitions: `DEVELOPMENT` → `CANDIDATE` → `APPROVED` → `PRODUCTION` → `ARCHIVED`.
- Human-in-the-loop review interface with candidate vs. production comparison.
- Safe zero-downtime rollback to previous approved version with full audit trail.

### Step 9 & 12: Centralized Automation & Security
- **Service**: `backend/app/routers/governance.py`
- Centralized rule registration and comprehensive audit log recording.

### Step 10: Advanced React Dashboard
- **Frontend**: `frontend/src/App.tsx`
- Complete dashboard with 11 views: Overview, Dataset Intelligence, Recommendations, Training, Comparison, Explainability, Monitoring, Retraining, Governance, Deployments, and Automation.

## 3. MLflow Integration Architecture

1. **Tracking Server**: Pinned PostgreSQL backend store (`postgresql://.../mlflow`) and S3-compatible artifact store (`s3://mlflow` via SeaweedFS or AWS S3).
2. **Model Logging**: Models logged with `mlflow.sklearn.log_model` using CloudPickle serialization to support custom pipelines.
3. **Model Registry**: Models registered via `mlflow.register_model` and tracked across lifecycle stages.
4. **Model Serving**: Live predictions loaded dynamically via `mlflow.pyfunc.load_model`.
5. **Model Parity**: Explainability and drift evaluations operate on models loaded directly from the MLflow registry.

## 4. CI/CD & Testing Verification

- **GitHub Actions CI** (`.github/workflows/ci.yml`):
  - Runs on `main` and `arjun-agrawal`.
  - Backend tests: `13 passed in 18.24s`.
  - Frontend build: `tsc -b && vite build` passed.
  - Docker Compose specification validated.
  - Docker image build validated.
