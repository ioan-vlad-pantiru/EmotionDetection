"""
FastAPI application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.model_manager import ModelManager


def create_application() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Emotion Detection API",
        description="API for bilingual emotion detection using lexicon and TF-IDF features",
        version="1.0.0",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)
    
    # Initialize model manager on startup
    @app.on_event("startup")
    async def startup_event():
        """Initialize models on application startup."""
        model_manager = ModelManager()
        await model_manager.initialize()
        app.state.model_manager = model_manager
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {"message": "Emotion Detection API", "version": "1.0.0"}
    
    return app


app = create_application()
