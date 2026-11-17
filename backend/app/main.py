import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .database import Base, apply_phase2_schema, engine
from .routers import datasets, mlflow, training

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.upload_path
    Base.metadata.create_all(bind=engine)
    apply_phase2_schema()
    yield


app = FastAPI(title="SentinelML API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[x.strip() for x in settings.cors_origins.split(",")], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(datasets.router)
app.include_router(training.router)
app.include_router(mlflow.router)


@app.get("/health")
def health():
    return {"status": "ok"}

