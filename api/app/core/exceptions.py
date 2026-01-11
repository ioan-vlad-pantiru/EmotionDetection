"""
Custom exceptions for the application.
"""


class EmotionDetectionError(Exception):
    """Base exception for emotion detection errors."""
    pass


class ModelNotFoundError(EmotionDetectionError):
    """Raised when a model is not found."""
    pass


class ExtractorNotFoundError(EmotionDetectionError):
    """Raised when extractors are not available."""
    pass


class InvalidModelError(EmotionDetectionError):
    """Raised when an invalid model type is specified."""
    pass


class InvalidLanguageError(EmotionDetectionError):
    """Raised when an invalid language is specified."""
    pass
