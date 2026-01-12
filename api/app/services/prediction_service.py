"""
Service layer for emotion prediction.
"""
import sys
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.models.infer import predict_lexicon_only, predict_ml_only, predict_hybrid
from app.core.exceptions import ModelNotFoundError, ExtractorNotFoundError, InvalidModelError


class PredictionService:
    """Service for making emotion predictions."""
    
    def __init__(self, model_manager):
        """Initialize prediction service with model manager."""
        self.model_manager = model_manager
    
    async def predict(
        self,
        text: str,
        lang: str,
        model_type: str
    ) -> Dict:
        """
        Predict emotion for given text.
        
        Args:
            text: Input text to analyze
            lang: Language code (en or ro)
            model_type: Model type (lexicon, ml, or hybrid)
        
        Returns:
            Dictionary with prediction results including probabilities
        
        Raises:
            ModelNotFoundError: If model is not available
            ExtractorNotFoundError: If extractors are not available
            InvalidModelError: If model type is invalid
        """
        if not self.model_manager.is_initialized():
            raise RuntimeError("Models are not initialized yet")
        
        # Get model info
        model_info = self.model_manager.get_model_info(lang, model_type)
        if model_info is None:
            raise ModelNotFoundError(f"Model {lang}_{model_type} not available")
        
        # Get extractors
        extractors = self.model_manager.get_extractors(lang)
        if extractors is None:
            raise ExtractorNotFoundError(f"Extractors not initialized for {lang}")
        
        # Run prediction in thread pool to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._predict_sync,
            text,
            model_type,
            model_info["path"],
            extractors,
            lang  # Pass language for negation post-processing
        )
        
        pred_labels, probabilities, label_mapping = result
        emotion = pred_labels[0] if pred_labels else "unknown"
        
        # Build probabilities dictionary
        proba_dict = {}
        if probabilities is not None and len(probabilities) > 0:
            int_to_label = {v: k for k, v in label_mapping.items()}
            proba_array = probabilities[0]
            for i, prob in enumerate(proba_array):
                label = int_to_label.get(i, f"label_{i}")
                proba_dict[label] = float(prob)
        
        return {
            "emotion": emotion,
            "text": text,
            "lang": lang,
            "model": model_type,
            "probabilities": proba_dict,
            "confidence": float(proba_dict.get(emotion, 0.0)) if proba_dict else 0.0
        }
    
    def _predict_sync(
        self,
        text: str,
        model_type: str,
        model_path: Path,
        extractors: Dict,
        lang: str = "en"
    ) -> tuple:
        """Synchronous prediction with probabilities."""
        from src.utils.io import load_model_bundle
        from src.features.fusion import FeatureFusion
        
        bundle = load_model_bundle(model_path)
        label_mapping = bundle["label_mapping"]
        
        # Get saved extractors from model bundle (these match what was used during training)
        saved_tfidf = bundle.get("vectorizer")
        saved_scaler = bundle.get("scaler")
        
        if model_type == "lexicon":
            if extractors["lexicon_extractor"] is None:
                raise ExtractorNotFoundError("Lexicon extractor not available")
            _, pred_labels, probabilities = predict_lexicon_only(
                [text],
                model_path,
                extractors["lexicon_extractor"],
                return_proba=True
            )
        elif model_type == "ml":
            # Use the extractor saved with the model, not the one from model_manager
            if saved_tfidf is not None:
                # Use saved extractor (matches training)
                _, pred_labels, probabilities = predict_ml_only(
                    [text],
                    model_path,
                    saved_tfidf,
                    return_proba=True,
                    lang=lang
                )
            else:
                # Fallback to passed extractor
                _, pred_labels, probabilities = predict_ml_only(
                    [text],
                    model_path,
                    extractors["tfidf_extractor"],
                    return_proba=True,
                    lang=lang
                )
        elif model_type == "hybrid":
            # Reconstruct fusion with saved components
            if saved_tfidf is not None and saved_scaler is not None:
                # Use saved extractors (matches training)
                if extractors["lexicon_extractor"] is None:
                    raise ExtractorNotFoundError("Lexicon extractor not available")
                saved_fusion = FeatureFusion(extractors["lexicon_extractor"], saved_tfidf)
                saved_fusion.scaler = saved_scaler
                saved_fusion._fitted = True
                _, pred_labels, probabilities = predict_hybrid(
                    [text],
                    model_path,
                    saved_fusion,
                    return_proba=True
                )
            elif extractors["fusion"] is not None:
                # Fallback to passed fusion
                _, pred_labels, probabilities = predict_hybrid(
                    [text],
                    model_path,
                    extractors["fusion"],
                    return_proba=True
                )
            else:
                raise ExtractorNotFoundError("Fusion extractor not available")
        else:
            raise InvalidModelError(f"Invalid model type: {model_type}")
        
        return pred_labels, probabilities, label_mapping
