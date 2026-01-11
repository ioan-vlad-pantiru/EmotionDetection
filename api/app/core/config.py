"""
Application configuration.
"""
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings."""
    
    # API settings
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Emotion Detection API"
    
    # CORS settings
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    
    # API server settings
    API_PORT: int = 5001
    
    # Model settings
    MODELS_DIR: str = "models"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
