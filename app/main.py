"""
AFL Multi Builder - FastAPI Application Entry Point.
"""
import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.routes import router
from app.db.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle handler."""
    logger.info("Starting AFL Multi Builder...")

    # Ensure required directories exist
    os.makedirs("data/models", exist_ok=True)
    os.makedirs("data/logs", exist_ok=True)

    # Initialise database
    try:
        init_db()
        logger.info("Database initialised.")
    except Exception as e:
        logger.warning(f"Database init warning: {e}")

    yield

    logger.info("Shutting down AFL Multi Builder...")


app = FastAPI(
    title="AFL Multi Builder",
    description=(
        "AFL prediction, pricing, and multi-building application. "
        "Generates calibrated probability estimates, detects expected value, "
        "and builds correlation-aware multi recommendations."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(router, prefix="/api/v1")

# Static files for UI
ui_dir = Path(__file__).parent / "ui"
if ui_dir.exists():
    app.mount("/static", StaticFiles(directory=str(ui_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_ui():
        return FileResponse(str(ui_dir / "index.html"))
else:
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "message": "AFL Multi Builder API",
            "docs": "/docs",
            "health": "/api/v1/health",
        }
