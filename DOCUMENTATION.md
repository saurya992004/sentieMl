# SentinelML — Comprehensive Architecture, Implementation & MLOps Guide

---

## 📑 Table of Contents
1. [Executive Summary: What is SentinelML?](#1-executive-summary-what-is-sentinelml)
2. [Why is this a True MLOps Project? (ML vs. MLOps)](#2-why-is-this-a-true-mlops-project-ml-vs-mlops)
3. [System Architecture & Data Flow](#3-system-architecture--data-flow)
4. [Deep Dive into Core Modules & Capabilities](#4-deep-dive-into-core-modules--capabilities)
   - [4.1 Automated Dataset Profiling & Quality Scoring](#41-automated-dataset-profiling--quality-scoring)
   - [4.2 Intelligent Algorithm Recommender](#42-intelligent-algorithm-recommender)
   - [4.3 Google Gemini 2.5 Flash Generative Intelligence](#43-google-gemini-25-flash-generative-intelligence)
   - [4.4 Distributed Training & Celery Worker Execution](#44-distributed-training--celery-worker-execution)
   - [4.5 Custom Hyperparameter Tuning & Scikit-Learn Code Generator](#45-custom-hyperparameter-tuning--scikit-learn-code-generator)
   - [4.6 MLflow Experiment Tracking & Model Registry](#46-mlflow-experiment-tracking--model-registry)
   - [4.7 Multi-Objective Pareto Comparison Engine](#47-multi-objective-pareto-comparison-engine)
   - [4.8 SHAP Game-Theoretic Explainability](#48-shap-game-theoretic-explainability)
   - [4.9 Production Prediction Serving & Ground Truth Feedback](#49-production-prediction-serving--ground-truth-feedback)
   - [4.10 Data Drift (KS-Test & PSI) & Performance Monitoring](#410-data-drift-ks-test--psi--performance-monitoring)
   - [4.11 Automated Retraining Orchestrator](#411-automated-retraining-orchestrator)
   - [4.12 Enterprise Governance, Model Promotion & 1-Click Rollback](#412-enterprise-governance-model-promotion--1-click-rollback)
5. [The AWS Architecture vs. Local Infrastructure (Why Not AWS by Default?)](#5-the-aws-architecture-vs-local-infrastructure-why-not-aws-by-default)
6. [Complete Setup Guide from Scratch (Zero to Hero)](#6-complete-setup-guide-from-scratch-zero-to-hero)
7. [Running Tests & Verifying Integrity](#7-running-tests--verifying-integrity)
8. [Live Exam Demo Script (Step-by-Step Walkthrough)](#8-live-exam-demo-script-step-by-step-walkthrough)

---

## 1. Executive Summary: What is SentinelML?

**SentinelML** is an enterprise-grade, partially autonomous Machine Learning Operations (**MLOps**) platform. 

In traditional machine learning workflows, data scientists train models manually in isolated Jupyter notebooks, generate static `.pkl` files, and hand them off to software engineers with little tracking, zero automated monitoring, and no safety mechanisms when data changes in production.

SentinelML eliminates this fragmentation by providing an end-to-end, containerized platform that orchestrates the **entire ML lifecycle**:
- **Ingests and profiles** tabular datasets with automated data quality scoring.
- **Analyzes data distributions** and recommends the optimal candidate algorithms.
- **Trains models asynchronously** across distributed workers while logging parameters, metrics, and pipeline artifacts into **MLflow**.
- **Evaluates models across multiple real-world criteria** (accuracy/F1 vs. inference latency vs. training time vs. artifact storage size).
- **Explains model decisions** using game-theoretic **SHAP (Shapley Additive Explanations)**.
- **Monitors live production inference** to detect distribution shifts (**data drift**) and metric degradation.
- **Triggers automated retraining** when drift thresholds are breached.
- **Enforces strict human-in-the-loop governance** with model staging, approval gates, zero-downtime rollback, and immutable audit logs.
- **Augments MLOps workflows with Generative AI** using **Google Gemini 2.5 Flash** for executive data briefs, architecture advice, and business decision playbooks.

---

## 2. Why is this a True MLOps Project? (ML vs. MLOps)

A common misconception among students and evaluators is that training a Scikit-Learn or PyTorch model makes a project an "MLOps" project. **It does not.**

### The Core Difference:
| Dimension | Traditional Machine Learning (Modeling) | SentinelML (True MLOps) |
| :--- | :--- | :--- |
| **Execution Environment** | Interactive Jupyter Notebooks (`.ipynb`) | Distributed background workers (Celery + Redis) |
| **Code Structure** | Ad-hoc Python scripts; manual feature prep | Dynamic `Pipeline` & `ColumnTransformer` |
| **Model Versioning** | Saving `model_v1_final_really.pkl` to disk | **MLflow Model Registry** with immutable run IDs & tags |
| **Artifact Storage** | Local folders / hard drives | S3-compliant Object Storage (SeaweedFS / Amazon S3) |
| **Artifact Tracking** | None; hyperparameters forgotten | Automatic tracking of hyperparameters, git commits, metrics |
| **Model Evaluation** | Single metric (e.g. Test Accuracy) | **Multi-Objective Pareto trade-offs** (Score vs. Latency vs. Model Size) |
| **Production Serving** | Rerunning notebook or raw pickle loads | Versioned REST API (`/predict`) loading models dynamically from MLflow |
| **Production Drift** | Silent model failure when world changes | Automated statistical tests (**KS-test & PSI**) on live inferences |
| **Retraining** | Manually retraining from scratch weeks later | **Automated retraining orchestrator** triggered by drift alarms |
| **Lifecycle Governance** | Overwriting files in production | Strict Lifecycle Stages (`Staging` $\to$ `Production` $\to$ `Rollback`) |
| **Accountability** | Untracked changes | Append-only **Audit Logs** recording every actor & action |

SentinelML covers all three pillars of Google's MLOps Level 2 maturity framework: **Automated Pipeline Orchestration**, **Continuous Experiment Tracking**, and **Production Monitoring with Feedback Loops**.

---

## 3. System Architecture & Data Flow

```
                                  ┌───────────────────────────┐
                                  │      React Dashboard      │
                                  │   (Vite + TypeScript)     │
                                  │   Dark Enterprise Cockpit │
                                  └─────────────┬─────────────┘
                                                │ REST API (HTTP 8000)
                                                ▼
                                  ┌───────────────────────────┐
                                  │      FastAPI Backend      │
                                  │   (Orchestrator & APIs)   │
                                  └──────┬──────────────┬─────┘
                                         │              │
                   Dispatches Jobs       │              │ Tracks Database Records
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
             │  - Experiment Tracking (Hyperparams, Metrics, Runs)      │
             │  - Model Registry (Registered Versions & Stages)         │
             │  - S3 Object Storage (CloudPickle Pipelines)             │
             └───────────────────────────┬──────────────────────────────┘
                                         │
                                         ▼
                             ┌───────────────────────┐
                             │ S3 Object Store       │
                             │ (SeaweedFS / AWS S3)  │
                             └───────────────────────┘
```

---

## 4. Deep Dive into Core Modules & Capabilities

### 4.1 Automated Dataset Profiling & Quality Scoring
- **Dynamic Inspection**: When a CSV is uploaded, SentinelML computes descriptive statistics (mean, std, median, min, max, unique counts) for every column.
- **Interquartile Range (IQR) Outlier Detection**: Identifies extreme values using $[\text{Q1} - 1.5 \times \text{IQR}, \text{Q3} + 1.5 \times \text{IQR}]$.
- **Multicollinearity Matrix**: Computes Pearson correlation coefficients ($r$) between numerical features to flag collinear feature pairs ($|r| \ge 0.85$).
- **Target Imbalance Analysis**: Detects severe class distributions (e.g. 90:10 ratios) that risk classifier bias.
- **Data Quality Score (0–100)**: Formulates an empirical hygiene score by deducting penalties for missing values, duplicate rows, zero-variance columns, and outliers.

### 4.2 Intelligent Algorithm Recommender
- Detects whether the problem is **Binary Classification**, **Multi-Class Classification**, or **Regression** based on target column datatype and cardinality.
- Evaluates dataset dimensions ($N$ rows, $P$ features), sparsity, and non-linearity.
- Generates suitability scores ($0\text{--}100$) and explicit human-readable rationales for candidate algorithms.

### 4.3 Google Gemini 2.5 Flash Generative Intelligence
- Powered by Google DeepMind's **Gemini 2.5 Flash** (`gemini-2.5-flash`).
- **Dynamic Executive Data Briefs**: Transforms raw statistics and missingness into an authoritative data hygiene analysis.
- **Dynamic MLOps Architectural Guidance**: Explains why certain algorithms suit the dataset's specific variance profile.
- **Dynamic Strategic Business Translations**: Translates complex game-theoretic SHAP attributions into actionable management decisions (e.g. retention strategies).
- **Resilient Fallback**: If an API key is missing or quota is reached, SentinelML automatically falls back to mathematical heuristics without throwing errors.

### 4.4 Distributed Training & Celery Worker Execution
- Training is decoupled from the web server. When a training run is requested:
  1. The API creates a `TrainingJob` record in PostgreSQL.
  2. The task is pushed to a **Redis** message queue.
  3. A **Celery Worker** picks up the task, executes training, streams live progress logs ($0\%\text{--}100\%$), and logs artifacts to MLflow.
  4. The web server remains 100% responsive and non-blocking.

### 4.5 Custom Hyperparameter Tuning & Scikit-Learn Code Generator
- Supports full hyperparameter control from the UI:
  - `n_estimators`: 50 to 500 trees.
  - `max_depth`: Unconstrained or bounded (3 to 20).
  - `learning_rate`: 0.01 to 0.4.
  - `Regularization C` (Logistic) & `Alpha` (Ridge/Lasso).
- **Interactive Code Generator**: Allows users to inspect and copy the exact Scikit-Learn `ColumnTransformer` and `Pipeline` Python code executed by the workers.

### 4.6 MLflow Experiment Tracking & Model Registry
- **Tracking**: Logs all runs under experiment namespaces (`sentinelml-classification`, `sentinelml-regression`).
- **Artifacts**: Serializes fitted Scikit-Learn pipelines using `cloudpickle` and logs them via `mlflow.sklearn.log_model` into S3 object storage.
- **Model Registry**: Automatically registers winning models under versioned identifiers (e.g. `sentinelml_job_2_best` v1).

### 4.7 Multi-Objective Pareto Comparison Engine
In production, accuracy alone is insufficient. SentinelML calculates an empirical Pareto efficiency score ($0\text{--}100$) balancing:
1. **Primary Metric Performance** (Accuracy / F1 / RMSE) (50% weight).
2. **Inference Latency** in milliseconds per sample (20% weight).
3. **Training Duration** in seconds (15% weight).
4. **Model Artifact Size** in KB (15% weight).

### 4.8 SHAP Game-Theoretic Explainability
- Implements **SHAP (Shapley Additive Explanations)** based on cooperative game theory.
- Computes each feature's marginal contribution across all feature subsets.
- Provides **Global Feature Importance** (which features matter most overall) and **Local Instance Explanations** (why the model made a specific prediction on a given customer/record).

### 4.9 Production Prediction Serving & Ground Truth Feedback
- Exposes a dedicated `/predict` endpoint that dynamically loads registered models directly from MLflow (`mlflow.pyfunc.load_model`).
- Logs every production inference call into PostgreSQL (`production_predictions`).
- Exposes a `/monitoring/ground-truth` endpoint where real-world labels can be submitted later to measure actual live model degradation.

### 4.10 Data Drift (KS-Test & PSI) & Performance Monitoring
- **Kolmogorov-Smirnov Test (KS-Test)**: Non-parametric statistical test comparing the cumulative distribution functions (CDF) of baseline training features against production data ($p < 0.05$ indicates drift).
- **Population Stability Index (PSI)**: Measures distributional shift across binned frequencies:
  $$\text{PSI} = \sum \left( \text{Actual}\% - \text{Expected}\% \right) \times \ln\left( \frac{\text{Actual}\%}{\text{Expected}\%} \right)$$
  - $\text{PSI} < 0.1$: No significant change.
  - $0.1 \le \text{PSI} < 0.25$: Moderate drift.
  - $\text{PSI} \ge 0.25$: Critical drift requiring retraining.
- **Categorical Drift**: Total Variation Distance (TVD) across category frequencies.

### 4.11 Automated Retraining Orchestrator
- Define configurable rules: trigger retraining when **Drift Severity == CRITICAL**, or when **Metric Degradation > 10%**.
- Retraining jobs automatically re-profile the latest data, fit new models, compare candidates against the incumbent production model, and create a pending promotion approval if superior.

### 4.12 Enterprise Governance, Model Promotion & 1-Click Rollback
- Models progress through formal lifecycle stages:
  $$\text{DEVELOPMENT} \longrightarrow \text{CANDIDATE} \longrightarrow \text{APPROVED} \longrightarrow \text{PRODUCTION} \longrightarrow \text{ARCHIVED}$$
- **1-Click Zero-Downtime Rollback**: If a production model degrades, operators can roll back to any previous version with a single click. The `/predict` service switches targets immediately without server restarts.
- **Append-Only Audit Trail**: Every dataset upload, training run, promotion, rollback, and approval is immutably logged in PostgreSQL with actor timestamps and metadata.

---

## 5. The AWS Architecture vs. Local Infrastructure (Why Not AWS by Default?)

### Why Not Use AWS Directly for Local Testing & Exams?
1. **Cost & Billing Surprises**: Running AWS RDS PostgreSQL, AWS ElastiCache Redis, Amazon S3, and AWS EC2 24/7 quickly exhausts free credits and risks billing charges.
2. **Network Dependency & Latency**: Running experiments against remote AWS services creates upload/download network bottlenecks when transferring large CSVs and model artifacts.
3. **Friction for Evaluators**: Requiring exam evaluators to have AWS IAM accounts, credentials, and credit cards makes evaluating the project difficult.

### The Local S3 Alternative: SeaweedFS
SentinelML solves this elegantly by running **SeaweedFS** locally inside Docker:
- SeaweedFS exposes a **100% compliant Amazon S3 REST API** on port `8333`.
- MLflow and Scikit-Learn interact with SeaweedFS using the official `boto3` AWS SDK, believing they are talking to real AWS S3.
- All model binaries and datasets are stored locally in Docker volumes at zero cost.

### How to Deploy to Real AWS (Cloud Bridge)
The entire SentinelML architecture is cloud-native. To deploy to real AWS:
1. Create an Amazon S3 Bucket (e.g. `s3://my-company-sentinelml-bucket`).
2. Create an AWS RDS PostgreSQL instance and an AWS ElastiCache Redis instance.
3. Update `.env`:
   ```ini
   DATABASE_URL=postgresql+psycopg://user:password@rds-host.amazonaws.com:5432/sentinelml
   MLFLOW_S3_ENDPOINT_URL=https://s3.us-east-1.amazonaws.com
   AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
   AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
   MLFLOW_BUCKET=my-company-sentinelml-bucket
   CELERY_BROKER_URL=redis://elasticache-host.amazonaws.com:6379/0
   ```
4. **No code changes are needed!** The exact same Docker containers will now run on AWS EC2 or AWS ECS/EKS.

---

## 6. Complete Setup Guide from Scratch (Zero to Hero)

### Step 1: Prerequisites
Ensure the following are installed on your machine (Windows, Mac, or Linux):
1. **Docker Desktop** (with Linux Containers / WSL2 enabled on Windows):
   - Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
   - Start Docker Desktop and verify the status says **"Engine running"** (green icon).
2. **Git**:
   - Verify in terminal: `git --version`.
3. **Python 3.10 or 3.11** (Optional, only needed if running tests locally outside Docker):
   - Verify: `python --version`.

---

### Step 2: Clone the Repository & Checkout Branch
Open PowerShell (or Bash) and run:
```bash
git clone https://github.com/saurya992004/sentieMl.git
cd sentieMl
git checkout arjun-agrawal
```

---

### Step 3: Configure Environment Variables
A `.env` file is already created in the repository root. Ensure it contains the following keys:
```ini
DATABASE_URL=sqlite:///./sentinelml.db
MLFLOW_TRACKING_URI=sqlite:///./mlflow.db
MLFLOW_ARTIFACT_ROOT=./mlruns
UPLOAD_DIR=./uploads
MAX_UPLOAD_MB=25
CORS_ORIGINS=http://localhost:5173
POSTGRES_DB=sentinelml
POSTGRES_USER=sentinelml
POSTGRES_PASSWORD=change-this-postgres-password
POSTGRES_MLFLOW_DB=mlflow
MINIO_ROOT_USER=sentinelml
MINIO_ROOT_PASSWORD=change-this-minio-password
MLFLOW_BUCKET=mlflow
AWS_DEFAULT_REGION=us-east-1
VITE_API_URL=http://localhost:8000
GEMINI_API_KEY=your-gemini-api-key-here
```
*(Note: If you do not have a Gemini API key, leave it blank or grab a free key from [Google AI Studio](https://aistudio.google.com/). The system automatically falls back to statistical heuristics if no key is supplied).*

---

### Step 4: Launch the Full Docker Stack
Ensure Docker Desktop is open and running, then execute:
```bash
docker compose up --build
```
*(To run in the background as detached daemon, add `-d`: `docker compose up -d --build`).*

Docker will automatically pull, build, and launch **7 microservices**:
1. `sentieml-postgres-1`: PostgreSQL 16 database.
2. `sentieml-redis-1`: Redis 7 in-memory cache and Celery broker.
3. `sentieml-object-store-1`: SeaweedFS S3-compatible storage.
4. `sentieml-object-store-init-1`: Initializes the S3 `mlflow` bucket.
5. `sentieml-mlflow-1`: MLflow Tracking Server & Model Registry.
6. `sentieml-api-1`: FastAPI Web Application.
7. `sentieml-worker-1`: Celery background training worker.
8. `sentieml-frontend-1`: React Dashboard served via Nginx.

---

### Step 5: Verify Running Services
Run:
```bash
docker compose ps
```
You should see all containers listed with status `Up` or `Up (healthy)`.

Open your browser and navigate to:
- **SentinelML React Cockpit:** [http://localhost:5173](http://localhost:5173)
- **FastAPI Interactive Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **MLflow Tracking UI:** [http://localhost:5000](http://localhost:5000)
- **SeaweedFS S3 Storage UI:** [http://localhost:8333](http://localhost:8333)

---

## 7. Running Tests & Verifying Integrity

To execute the automated unit and end-to-end test suite:
```bash
python -m pytest backend/tests
```
All **13 test suites** will execute:
- `test_api.py`: FastAPI health, CORS, and endpoint routing.
- `test_complete_phase3.py`: Full Phase 3 schemas and governance integrity.
- `test_datasets.py`: File upload, validation, and storage.
- `test_phase3_e2e.py`: End-to-end retraining, promotion, and rollback workflows.
- `test_profiler.py`: Missing values, IQR outlier detection, Pearson correlation matrix.
- `test_recommender.py`: Algorithm suitability scoring across classification and regression.

---

## 8. Live Exam Demo Script (Step-by-Step Walkthrough)

To deliver a compelling 5-minute presentation to your professors or evaluators, open **SentinelML Dashboard** ([http://localhost:5173](http://localhost:5173)) and **MLflow UI** ([http://localhost:5000](http://localhost:5000)) in two side-by-side browser windows:

### Phase 1: Data Ingestion & Automated Profiling
1. Click **"Dataset Intelligence"** in the left navigation.
2. Select `Churn_Modelling.csv` (or upload any new CSV).
3. Point out the **Quality Score (e.g. 96/100)**, the IQR outlier counts, and the Pearson correlation matrix.
4. Highlight the **"✨ Gemini AI Executive Data Intelligence Brief"** card at the top, explaining that Gemini 2.5 Flash evaluated the empirical statistics dynamically.

### Phase 2: Algorithm Recommendation
1. Click **"Model Recommendations"**.
2. Select `Churn_Modelling.csv`, set target column to `Exited`, and click **Generate Model Recommendations**.
3. Point out how SentinelML detected a binary classification problem and ranked `Random Forest`, `Gradient Boosting`, and `Extra Trees` as top fits with explicit reasoning.
4. Show the **"✨ Gemini AI Model Architecture Guidance"** card detailing the variance and tuning trade-offs.

### Phase 3: Custom Hyperparameter Tuning & MLflow Training
1. Click **"Train & Experiments"**.
2. Check the algorithms you want to train (e.g. `Random Forest` and `Gradient Boosting`).
3. Adjust the **Estimators** slider to `250` and **Max Depth** to `8`.
4. Click **"View Generated Pipeline Code"**: Show the evaluators the exact Scikit-Learn Python code being executed by the background worker.
5. Click **"Start Asynchronous Training Run"**.
6. Show the live Celery task progress updating in real time from $5\%$ to $100\%$.
7. Switch to the **MLflow UI** window: Refresh and show the registered experiment runs, logged hyperparameters (`hp_n_estimators: 250`, `hp_max_depth: 8`), and CloudPickle model artifacts stored in S3!

### Phase 4: Multi-Objective Model Selection & SHAP Explainability
1. Click **"Model Comparison"**: Show the Pareto ranking comparing model accuracy against latency (ms) and artifact size (KB).
2. Click **"Model Explainability"**: Select the trained model.
3. Show the **Global Feature Importance Chart** (e.g. `Age`, `NumOfProducts`, `Balance`).
4. Show the **"✨ Gemini AI Strategic Business Interpretation"** card, showing how the AI translates SHAP values into business retention strategies.

### Phase 5: Production Drift, Governance & 1-Click Rollback
1. Click **"Drift & Performance"**: Click **"Simulate Production Inference & Check Drift"**.
2. Show the **Kolmogorov-Smirnov (KS-test)** and **Population Stability Index (PSI)** charts detecting feature distribution shift.
3. Click **"Governance & Approvals"**:
   - Show the current model in `PRODUCTION`.
   - Promote Version 2 to `PRODUCTION`.
   - Click **"Rollback to Previous Version"**: Show how the platform safely rolls back to Version 1 with zero downtime, and point to the immutable **Audit Log** at the bottom documenting the rollback timestamp and operator identity!
