#!/bin/sh
set -eu

# SentinelML application metadata and MLflow metadata both use PostgreSQL,
# but must be isolated because both systems own a table named `datasets`.
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE ${POSTGRES_MLFLOW_DB};
EOSQL
