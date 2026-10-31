"""
ThreatLens FastAPI Application Main Entrypoint

Initializes the FastAPI application, sets up lifespan event handling for one-time
model loading, attaches CORS middleware, registers standardized exception handlers,
and mounts API route modules.
"""

from contextlib import asynccontextmanager
import os
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from threatlens.api.dependencies import container
from threatlens.api.routes import (
    explanation_router,
    health_router,
    model_info_router,
    prediction_router,
)
from threatlens.config import API_CONFIG
from threatlens.utils.logger import get_logger

logger = get_logger("api_main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan event handler: loads model and explainer artifacts once during startup."""
    logger.info("Initializing ThreatLens API services on startup...")
    try:
        container.load_all()
        logger.info("ThreatLens API services initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize ThreatLens model services: {e}", exc_info=True)
        # Note: In production or testing, allow the app to initialize so /health can report status
    yield
    logger.info("Shutting down ThreatLens API services...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title=API_CONFIG.TITLE,
        version=API_CONFIG.VERSION,
        description=API_CONFIG.DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS configuration (supports override via environment variable ALLOWED_ORIGINS)
    cors_env = os.getenv("ALLOWED_ORIGINS", "")
    cors_origins = [o.strip() for o in cors_env.split(",") if o.strip()] if cors_env else list(API_CONFIG.CORS_ORIGINS)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Global Exception Handlers (Security: Do not leak stack traces to client)
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning(f"Request validation error on {request.url.path}: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Unprocessable Entity",
                "detail": str(exc),
                "status_code": 422,
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "detail": "An unexpected error occurred during processing. Stack traces are withheld for security.",
                "status_code": 500,
            },
        )

    # Mount Route Modules
    app.include_router(health_router)
    app.include_router(model_info_router)
    app.include_router(prediction_router)
    app.include_router(explanation_router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("threatlens.api.main:app", host=API_CONFIG.HOST, port=API_CONFIG.PORT, reload=False)
