import os

# Must be set before the application modules create their SQLAlchemy engine.
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_sentinelml.db")
os.environ.setdefault("UPLOAD_DIR", "./test_uploads")
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///./test_mlflow.db")

