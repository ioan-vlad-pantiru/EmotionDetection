"""
Model inference for emotion classification.
"""
import os
# Disable TensorFlow to avoid Keras 3 compatibility issues
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["SKIP_TF_INSTALL"] = "1"

import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import torch

from src.utils.io import load_model_bundle
from src.features.lexicon_features import LexiconFeatureExtractor
from src.features.tfidf_features import TFIDFFeatureExtractor
from src.features.fusion import FeatureFusion

# Try to import transformers (optional)
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def softmax(x, axis=1):
    """Compute softmax values for each set of scores."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def get_probabilities(classifier, X):
    """
    Get prediction probabilities from a classifier.
    Handles both models with predict_proba and models without (e.g., LinearSVC).
    
    Args:
        classifier: Trained classifier
        X: Feature matrix
        
    Returns:
        Probability array
    """
    if hasattr(classifier, 'predict_proba'):
        try:
            return classifier.predict_proba(X)
        except (AttributeError, TypeError):
            pass
    
    # For models without predict_proba (e.g., LinearSVC), use decision_function
    if hasattr(classifier, 'decision_function'):
        try:
            decision_scores = classifier.decision_function(X)
            # Handle both binary and multiclass
            if decision_scores.ndim == 1:
                # Binary classification - convert to probabilities
                # Using sigmoid: 1 / (1 + exp(-score))
                prob_positive = 1.0 / (1.0 + np.exp(-decision_scores))
                probabilities = np.column_stack([1 - prob_positive, prob_positive])
            else:
                # Multiclass - use softmax to convert scores to probabilities
                probabilities = softmax(decision_scores, axis=1)
            return probabilities
        except Exception:
            pass
    
    # Fallback: return uniform probabilities
    n_classes = len(classifier.classes_)
    return np.ones((X.shape[0], n_classes)) / n_classes


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
    
    # Handle feature count mismatch (old models may have been trained with fewer features)
    expected_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else scaler.mean_.shape[0] if hasattr(scaler, 'mean_') else X.shape[1]
    
    if X.shape[1] != expected_features:
        if X.shape[1] > expected_features:
            # New extractor has more features (e.g., negation features added)
            # Take only the first N features to match old model
            print(f"Warning: Feature mismatch - extracting {X.shape[1]} features but model expects {expected_features}. Using first {expected_features} features.")
            X = X[:, :expected_features]
        else:
            # New extractor has fewer features - pad with zeros
            print(f"Warning: Feature mismatch - extracting {X.shape[1]} features but model expects {expected_features}. Padding with zeros.")
            padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
            X = np.hstack([X, padding])
    
    # Scale
    X_scaled = scaler.transform(X)
    
    # Predict
    predictions = classifier.predict(X_scaled)
    
    # Get probabilities if requested
    probabilities = None
    if return_proba:
        probabilities = get_probabilities(classifier, X_scaled)
    
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
        probabilities = get_probabilities(classifier, X)
    
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
        probabilities = get_probabilities(classifier, X)
    
    # Convert to labels
    int_to_label = {v: k for k, v in label_mapping.items()}
    predicted_labels = [int_to_label[pred] for pred in predictions]
    
    return predictions, predicted_labels, probabilities


def predict_transformer(
    texts: List[str],
    model_path: Path,
    return_proba: bool = False,
    batch_size: int = 32,
    max_length: int = 128,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Predict using transformer model (BERT/RoBERTa).
    
    Args:
        texts: Input texts
        model_path: Path to saved model bundle
        return_proba: If True, also return prediction probabilities
        batch_size: Batch size for inference
        max_length: Maximum sequence length
        device: Device to use ('cuda' or 'cpu'). If None, auto-detect.
        
    Returns:
        Tuple of (predictions, predicted_labels, probabilities)
        probabilities: None if return_proba=False, else array of probabilities
    """
    if not HAS_TRANSFORMERS:
        raise ImportError(
            "transformers library not installed. "
            "Install it with: pip install transformers torch"
        )
    
    # Load model bundle
    bundle = load_model_bundle(model_path)
    
    # Get model directory or load from bundle
    if "model_dir" in bundle:
        model_dir = Path(bundle["model_dir"])
    else:
        # Try to infer from model_path
        model_dir = model_path.parent / f"{model_path.stem}_model"
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    label_mapping = bundle.get("label_mapping", bundle.get("label_mapping"))
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # Process texts in batches
    all_predictions = []
    all_probabilities = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move to device
        encodings = {k: v.to(device) for k, v in encodings.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**encodings)
            logits = outputs.logits
        
        # Get predictions
        batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
        all_predictions.extend(batch_predictions)
        
        # Get probabilities if requested
        if return_proba:
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            all_probabilities.extend(probabilities)
    
    predictions = np.array(all_predictions)
    probabilities = np.array(all_probabilities) if return_proba else None
    
    # Convert to labels
    int_to_label = {v: k for k, v in label_mapping.items()}
    predicted_labels = [int_to_label[pred] for pred in predictions]
    
    return predictions, predicted_labels, probabilities


def predict_transformer_hybrid(
    texts: List[str],
    model_path: Path,
    lexicon_extractor: LexiconFeatureExtractor,
    tfidf_extractor: TFIDFFeatureExtractor,
    return_proba: bool = False,
    batch_size: int = 32,
    max_length: int = 128,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Predict using transformer hybrid model (transformer + lexicon + TF-IDF).
    
    Args:
        texts: Input texts
        model_path: Path to saved model bundle
        lexicon_extractor: Lexicon feature extractor
        tfidf_extractor: TF-IDF feature extractor
        return_proba: If True, also return prediction probabilities
        batch_size: Batch size for transformer embeddings
        max_length: Maximum sequence length
        device: Device to use ('cuda' or 'cpu'). If None, auto-detect.
        
    Returns:
        Tuple of (predictions, predicted_labels, probabilities)
        probabilities: None if return_proba=False, else array of probabilities
    """
    if not HAS_TRANSFORMERS:
        raise ImportError(
            "transformers library not installed. "
            "Install it with: pip install transformers torch"
        )
    
    # Load model bundle
    bundle = load_model_bundle(model_path)
    classifier = bundle["classifier"]
    label_mapping = bundle["label_mapping"]
    hybrid_fusion = bundle["vectorizer"]
    
    # Get transformer model name from metadata or fusion object
    transformer_model = bundle.get("metadata", {}).get("transformer_model", "roberta-base")
    if hasattr(hybrid_fusion, "transformer_model"):
        transformer_model = hybrid_fusion.transformer_model
    
    # Extract transformer embeddings
    from src.models.train_transformer_hybrid import extract_transformer_embeddings
    
    transformer_emb = extract_transformer_embeddings(
        texts, transformer_model, batch_size, max_length, device
    )
    
    # Extract TF-IDF features
    tfidf_features = tfidf_extractor.transform(texts)
    
    # Extract lexicon features
    lexicon_features = lexicon_extractor.extract_batch(texts)
    
    # Scale lexicon features
    if hasattr(hybrid_fusion, "scaler") and hybrid_fusion.scaler is not None:
        lexicon_features = hybrid_fusion.scaler.transform(lexicon_features)
    
    # Combine features
    from scipy.sparse import hstack, csr_matrix
    
    lexicon_sparse = csr_matrix(lexicon_features)
    transformer_sparse = csr_matrix(transformer_emb)
    
    X_combined = hstack([tfidf_features, transformer_sparse, lexicon_sparse])
    
    # Convert to dense if needed
    if X_combined.shape[1] < 100000:
        X_combined = X_combined.toarray()
    
    # Predict
    predictions = classifier.predict(X_combined)
    
    # Get probabilities if requested
    probabilities = None
    if return_proba:
        probabilities = get_probabilities(classifier, X_combined)
    
    # Convert to labels
    int_to_label = {v: k for k, v in label_mapping.items()}
    predicted_labels = [int_to_label[pred] for pred in predictions]
    
    return predictions, predicted_labels, probabilities

