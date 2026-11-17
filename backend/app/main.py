import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .database import Base, apply_phase2_schema, apply_phase3_schema, engine
from .routers import datasets, governance, mlflow, models, monitoring, retraining, training

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.upload_path
    Base.metadata.create_all(bind=engine)
    apply_phase2_schema()
    apply_phase3_schema()
    yield


app = FastAPI(title="SentinelML API — Phase 3", version="0.3.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router)
app.include_router(training.router)
app.include_router(mlflow.router)
app.include_router(models.router)
app.include_router(monitoring.router)
app.include_router(retraining.router)
app.include_router(governance.router)


@app.get("/health")
def health():
    return {"status": "ok"}
