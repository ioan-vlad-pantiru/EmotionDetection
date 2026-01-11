"""
Model training for emotion classification.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Optional
from pathlib import Path

from src.config import (
    MODELS_DIR,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_STATE,
    CLASSIFIER_MAX_ITER,
    CLASSIFIER_SOLVER,
)
from src.utils.io import save_model_bundle
from src.features.lexicon_features import LexiconFeatureExtractor
from src.features.tfidf_features import TFIDFFeatureExtractor
from src.features.fusion import FeatureFusion


def train_lexicon_only(
    texts: List[str],
    labels: List[str],
    lexicon_extractor: LexiconFeatureExtractor,
    label_to_int: Dict[str, int],
    lang: str,
    model_name: str = "lexicon",
) -> Tuple[Path, Dict]:
    """
    Train lexicon-only model.
    
    Args:
        texts: Training texts
        labels: Training labels
        lexicon_extractor: Lexicon feature extractor
        label_to_int: Label to integer mapping
        lang: Language code
        model_name: Model name for saving
        
    Returns:
        Tuple of (model_path, metadata)
    """
    print(f"\nTraining {model_name} model for {lang}...")
    
    # Convert labels to integers
    y = np.array([label_to_int[label] for label in labels])
    
    # Extract lexicon features
    print("Extracting lexicon features...")
    X = lexicon_extractor.extract_batch(texts)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_STATE, stratify=y_train
    )
    
    # Use shape[0] for sparse matrices
    train_size = X_train.shape[0] if hasattr(X_train, 'shape') else len(X_train)
    val_size = X_val.shape[0] if hasattr(X_val, 'shape') else len(X_val)
    test_size = X_test.shape[0] if hasattr(X_test, 'shape') else len(X_test)
    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train classifier
    print("Training classifier...")
    classifier = LogisticRegression(
        solver=CLASSIFIER_SOLVER,
        max_iter=CLASSIFIER_MAX_ITER,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    classifier.fit(X_train_scaled, y_train)
    
    # Evaluate on validation set
    val_score = classifier.score(X_val_scaled, y_val)
    print(f"Validation accuracy: {val_score:.4f}")
    
    # Save model
    model_dir = MODELS_DIR / lang
    model_path = model_dir / f"{model_name}.joblib"
    
    # Get sizes (handle sparse matrices)
    train_size = X_train.shape[0] if hasattr(X_train, 'shape') else len(X_train)
    val_size = X_val.shape[0] if hasattr(X_val, 'shape') else len(X_val)
    test_size = X_test.shape[0] if hasattr(X_test, 'shape') else len(X_test)
    
    metadata = {
        "model_type": model_name,
        "lang": lang,
        "n_features": X.shape[1],
        "n_classes": len(label_to_int),
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "val_accuracy": float(val_score),
    }
    
    save_model_bundle(
        model_path=model_path,
        vectorizer=None,
        scaler=scaler,
        classifier=classifier,
        label_mapping=label_to_int,
        metadata=metadata,
    )
    
    print(f"Model saved to {model_path}")
    
    return model_path, metadata


def train_ml_only(
    texts: List[str],
    labels: List[str],
    tfidf_extractor: TFIDFFeatureExtractor,
    label_to_int: Dict[str, int],
    lang: str,
    model_name: str = "ml",
) -> Tuple[Path, Dict]:
    """
    Train ML-only (TF-IDF) model.
    
    Args:
        texts: Training texts
        labels: Training labels
        tfidf_extractor: TF-IDF feature extractor
        label_to_int: Label to integer mapping
        lang: Language code
        model_name: Model name for saving
        
    Returns:
        Tuple of (model_path, metadata)
    """
    print(f"\nTraining {model_name} model for {lang}...")
    
    # Convert labels to integers
    y = np.array([label_to_int[label] for label in labels])
    
    # Extract TF-IDF features
    print("Extracting TF-IDF features...")
    X = tfidf_extractor.fit_transform(texts)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_STATE, stratify=y_train
    )
    
    # Use shape[0] for sparse matrices
    train_size = X_train.shape[0] if hasattr(X_train, 'shape') else len(X_train)
    val_size = X_val.shape[0] if hasattr(X_val, 'shape') else len(X_val)
    test_size = X_test.shape[0] if hasattr(X_test, 'shape') else len(X_test)
    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Train classifier (TF-IDF is already normalized, no scaling needed)
    print("Training classifier...")
    classifier = LogisticRegression(
        solver=CLASSIFIER_SOLVER,
        max_iter=CLASSIFIER_MAX_ITER,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    classifier.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_score = classifier.score(X_val, y_val)
    print(f"Validation accuracy: {val_score:.4f}")
    
    # Save model
    model_dir = MODELS_DIR / lang
    model_path = model_dir / f"{model_name}.joblib"
    
    # Get sizes (handle sparse matrices)
    train_size = X_train.shape[0] if hasattr(X_train, 'shape') else len(X_train)
    val_size = X_val.shape[0] if hasattr(X_val, 'shape') else len(X_val)
    test_size = X_test.shape[0] if hasattr(X_test, 'shape') else len(X_test)
    
    metadata = {
        "model_type": model_name,
        "lang": lang,
        "n_features": X.shape[1],
        "n_classes": len(label_to_int),
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "val_accuracy": float(val_score),
    }
    
    save_model_bundle(
        model_path=model_path,
        vectorizer=tfidf_extractor,  # Save entire extractor
        scaler=None,
        classifier=classifier,
        label_mapping=label_to_int,
        metadata=metadata,
    )
    
    print(f"Model saved to {model_path}")
    
    return model_path, metadata


def train_hybrid(
    texts: List[str],
    labels: List[str],
    fusion: FeatureFusion,
    label_to_int: Dict[str, int],
    lang: str,
    model_name: str = "hybrid",
) -> Tuple[Path, Dict]:
    """
    Train hybrid (fusion) model.
    
    Args:
        texts: Training texts
        labels: Training labels
        fusion: Feature fusion object
        label_to_int: Label to integer mapping
        lang: Language code
        model_name: Model name for saving
        
    Returns:
        Tuple of (model_path, metadata)
    """
    print(f"\nTraining {model_name} model for {lang}...")
    
    # Convert labels to integers
    y = np.array([label_to_int[label] for label in labels])
    
    # Extract fused features
    print("Extracting fused features...")
    X = fusion.fit_transform(texts)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_STATE, stratify=y_train
    )
    
    # Use shape[0] for sparse matrices
    train_size = X_train.shape[0] if hasattr(X_train, 'shape') else len(X_train)
    val_size = X_val.shape[0] if hasattr(X_val, 'shape') else len(X_val)
    test_size = X_test.shape[0] if hasattr(X_test, 'shape') else len(X_test)
    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Train classifier (fused features already scaled)
    print("Training classifier...")
    classifier = LogisticRegression(
        solver=CLASSIFIER_SOLVER,
        max_iter=CLASSIFIER_MAX_ITER,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    classifier.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_score = classifier.score(X_val, y_val)
    print(f"Validation accuracy: {val_score:.4f}")
    
    # Save model
    model_dir = MODELS_DIR / lang
    model_path = model_dir / f"{model_name}.joblib"
    
    # Get sizes (handle sparse matrices)
    train_size = X_train.shape[0] if hasattr(X_train, 'shape') else len(X_train)
    val_size = X_val.shape[0] if hasattr(X_val, 'shape') else len(X_val)
    test_size = X_test.shape[0] if hasattr(X_test, 'shape') else len(X_test)
    
    metadata = {
        "model_type": model_name,
        "lang": lang,
        "n_features": X.shape[1],
        "n_classes": len(label_to_int),
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "val_accuracy": float(val_score),
    }
    
    # Save fusion components separately
    save_model_bundle(
        model_path=model_path,
        vectorizer=fusion.tfidf_extractor,  # Save TF-IDF extractor
        scaler=fusion.scaler,  # Save fusion scaler
        classifier=classifier,
        label_mapping=label_to_int,
        metadata=metadata,
    )
    
    print(f"Model saved to {model_path}")
    
    return model_path, metadata

