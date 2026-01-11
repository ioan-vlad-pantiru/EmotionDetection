"""
API v1 router.
"""
import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Annotated

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from app.models.schemas import (
    PredictionRequest,
    PredictionResponse,
    ModelsListResponse,
    HealthResponse,
    ErrorResponse,
)
from app.services.prediction_service import PredictionService
from app.core.model_manager import ModelManager
from app.core.exceptions import (
    ModelNotFoundError,
    ExtractorNotFoundError,
    InvalidModelError,
    InvalidLanguageError,
)
from src.config import LANGUAGES, MODEL_TYPES


api_router = APIRouter()


def get_model_manager(request: Request) -> ModelManager:
    """Dependency to get model manager from app state."""
    return request.app.state.model_manager


def get_prediction_service(
    model_manager: Annotated[ModelManager, Depends(get_model_manager)]
) -> PredictionService:
    """Dependency to get prediction service."""
    return PredictionService(model_manager)


@api_router.get("/health", response_model=HealthResponse)
async def health_check(
    model_manager: Annotated[ModelManager, Depends(get_model_manager)]
):
    """Health check endpoint."""
    return {
        "status": "ok",
        "models_loaded": len(model_manager.models),
        "initialized": model_manager.is_initialized(),
    }


@api_router.get("/models", response_model=ModelsListResponse)
async def list_models(
    model_manager: Annotated[ModelManager, Depends(get_model_manager)]
):
    """List all available models."""
    models = model_manager.list_available_models()
    return {"models": models}


@api_router.post("/predict", response_model=PredictionResponse)
async def predict_emotion(
    request: PredictionRequest,
    prediction_service: Annotated[PredictionService, Depends(get_prediction_service)]
):
    """
    Predict emotion for given text.
    
    - **text**: Text to analyze (required)
    - **lang**: Language code - "en" or "ro" (required)
    - **model**: Model type - "lexicon", "ml", or "hybrid" (required)
    """
    # Validate language
    if request.lang not in LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid language. Must be one of {LANGUAGES}"
        )
    
    # Validate model type
    if request.model not in MODEL_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model type. Must be one of {MODEL_TYPES}"
        )
    
    try:
        result = await prediction_service.predict(
            text=request.text,
            lang=request.lang,
            model_type=request.model
        )
        return result
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ExtractorNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except InvalidModelError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
