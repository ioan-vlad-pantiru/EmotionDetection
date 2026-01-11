"""
Pydantic schemas for request/response models.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class PredictionRequest(BaseModel):
    """Request schema for emotion prediction."""
    text: str = Field(..., min_length=1, description="Text to analyze for emotion")
    lang: str = Field(..., description="Language code (en or ro)")
    model: str = Field(..., description="Model type (lexicon, ml, or hybrid)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "I am so happy today!",
                "lang": "en",
                "model": "hybrid"
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for emotion prediction."""
    emotion: str = Field(..., description="Detected emotion")
    text: str = Field(..., description="Input text")
    lang: str = Field(..., description="Language used")
    model: str = Field(..., description="Model used")
    confidence: float = Field(..., description="Confidence score for predicted emotion")
    probabilities: Dict[str, float] = Field(..., description="Probability scores for all emotions")
    
    class Config:
        json_schema_extra = {
            "example": {
                "emotion": "joy",
                "text": "I am so happy today!",
                "lang": "en",
                "model": "hybrid",
                "confidence": 0.85,
                "probabilities": {
                    "joy": 0.85,
                    "neutral": 0.10,
                    "anger": 0.02,
                    "sadness": 0.01,
                    "fear": 0.01,
                    "surprise": 0.01
                }
            }
        }


class ModelInfo(BaseModel):
    """Model information schema."""
    lang: str
    model: str
    key: str


class ModelsListResponse(BaseModel):
    """Response schema for listing available models."""
    models: List[ModelInfo]


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    models_loaded: int
    initialized: bool


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: Optional[str] = None
