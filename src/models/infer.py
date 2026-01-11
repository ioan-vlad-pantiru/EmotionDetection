"""
Model inference for emotion classification.
"""
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path

from src.utils.io import load_model_bundle
from src.features.lexicon_features import LexiconFeatureExtractor
from src.features.tfidf_features import TFIDFFeatureExtractor
from src.features.fusion import FeatureFusion


def predict_lexicon_only(
    texts: List[str],
    model_path: Path,
    lexicon_extractor: LexiconFeatureExtractor,
    return_proba: bool = False,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Predict using lexicon-only model.
    
    Args:
        texts: Input texts
        model_path: Path to saved model
        lexicon_extractor: Lexicon feature extractor
        return_proba: If True, also return prediction probabilities
        
    Returns:
        Tuple of (predictions, predicted_labels, probabilities)
        probabilities: None if return_proba=False, else array of probabilities
    """
    bundle = load_model_bundle(model_path)
    classifier = bundle["classifier"]
    scaler = bundle["scaler"]
    label_mapping = bundle["label_mapping"]
    
    # Extract features
    X = lexicon_extractor.extract_batch(texts)
    
    # Scale
    X_scaled = scaler.transform(X)
    
    # Predict
    predictions = classifier.predict(X_scaled)
    
    # Get probabilities if requested
    probabilities = None
    if return_proba:
        probabilities = classifier.predict_proba(X_scaled)
    
    # Convert to labels
    int_to_label = {v: k for k, v in label_mapping.items()}
    predicted_labels = [int_to_label[pred] for pred in predictions]
    
    return predictions, predicted_labels, probabilities


def predict_ml_only(
    texts: List[str],
    model_path: Path,
    tfidf_extractor: TFIDFFeatureExtractor,
    return_proba: bool = False,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Predict using ML-only model.
    
    Args:
        texts: Input texts
        model_path: Path to saved model
        tfidf_extractor: TF-IDF feature extractor (must be fitted)
        return_proba: If True, also return prediction probabilities
        
    Returns:
        Tuple of (predictions, predicted_labels, probabilities)
        probabilities: None if return_proba=False, else array of probabilities
    """
    bundle = load_model_bundle(model_path)
    classifier = bundle["classifier"]
    label_mapping = bundle["label_mapping"]
    
    # Use the provided extractor (should be fitted)
    # Extract features
    X = tfidf_extractor.transform(texts)
    
    # Predict
    predictions = classifier.predict(X)
    
    # Get probabilities if requested
    probabilities = None
    if return_proba:
        probabilities = classifier.predict_proba(X)
    
    # Convert to labels
    int_to_label = {v: k for k, v in label_mapping.items()}
    predicted_labels = [int_to_label[pred] for pred in predictions]
    
    return predictions, predicted_labels, probabilities


def predict_hybrid(
    texts: List[str],
    model_path: Path,
    fusion: FeatureFusion,
    return_proba: bool = False,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Predict using hybrid model.
    
    Args:
        texts: Input texts
        model_path: Path to saved model
        fusion: Feature fusion object
        return_proba: If True, also return prediction probabilities
        
    Returns:
        Tuple of (predictions, predicted_labels, probabilities)
        probabilities: None if return_proba=False, else array of probabilities
    """
    bundle = load_model_bundle(model_path)
    classifier = bundle["classifier"]
    label_mapping = bundle["label_mapping"]
    
    # Extract features
    X = fusion.transform(texts)
    
    # Predict
    predictions = classifier.predict(X)
    
    # Get probabilities if requested
    probabilities = None
    if return_proba:
        probabilities = classifier.predict_proba(X)
    
    # Convert to labels
    int_to_label = {v: k for k, v in label_mapping.items()}
    predicted_labels = [int_to_label[pred] for pred in predictions]
    
    return predictions, predicted_labels, probabilities

