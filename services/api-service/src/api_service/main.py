import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Set

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlmodel import Session

from api_service.dependencies import get_db
from api_service.storage import create_db_and_tables

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Track running background tasks
_running_tasks: Set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifecycle of the FastAPI application."""
    # Startup
    logger.info("Starting up API service...")
    logger.info("Initializing database...")
    create_db_and_tables()
    logger.info("Database initialized successfully")
    yield
    # Shutdown - wait for background tasks
    if _running_tasks:
        logger.info(f"Waiting for {len(_running_tasks)} tasks to complete...")
        await asyncio.gather(*_running_tasks, return_exceptions=True)
    logger.info("Shutdown complete")


app = FastAPI(lifespan=lifespan)

# Configure CORS
# In production, restrict origins to specific domains
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Liveness probe - is the service running?"""
    return {"status": "healthy"}


@app.get("/ready")
async def readiness_check(db: Session = Depends(get_db)):
    """Readiness probe - can the service handle requests?"""
    try:
        # Test database connection
        db.execute(text("SELECT 1"))
        return {"status": "ready", "database": "connected"}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "error": "Database unavailable"},
        )


@app.get("/")
async def root():
    """Returns a welcome message for the API."""
    return {"message": "Welcome to the API Service"}
