"""FastAPI application for the evolution service."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from evolution_svc.routers.evolution import router as evolution_router
from evolution_svc.storage import create_db_and_tables

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifecycle of the FastAPI application."""
    logger.info("Starting up evolution service...")
    create_db_and_tables()
    logger.info("Database initialized successfully")
    yield
    logger.info("Shutdown complete")


app = FastAPI(lifespan=lifespan)
app.include_router(evolution_router)


@app.get("/health")
async def health_check():
    """Liveness probe - is the service running?"""
    return {"status": "healthy", "service": "evolution-svc"}
