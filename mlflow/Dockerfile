FROM ghcr.io/mlflow/mlflow:v3.18.0

# Required by MLflow's PostgreSQL tracking backend and S3-compatible artifacts.
RUN pip install --no-cache-dir "psycopg[binary]>=3.2,<4.0" boto3
