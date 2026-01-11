"""
Transformer-based hybrid model training.
Combines transformer embeddings + lexicon features + TF-IDF features for maximum accuracy.
"""
import os
# Disable TensorFlow to avoid Keras 3 compatibility issues
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["SKIP_TF_INSTALL"] = "1"

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, accuracy_score
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
import joblib
from scipy.sparse import hstack, csr_matrix

from src.config import (
    MODELS_DIR,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_STATE,
)
from src.utils.io import save_model_bundle
from src.features.lexicon_features import LexiconFeatureExtractor
from src.features.tfidf_features import TFIDFFeatureExtractor
from src.features.fusion import FeatureFusion


def extract_transformer_embeddings(
    texts: List[str],
    model_name: str = "roberta-base",
    batch_size: int = 8,  # Small batch size for memory efficiency (24GB RAM)
    max_length: int = 128,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Extract transformer embeddings from texts (memory-efficient version).
    
    Args:
        texts: List of input texts
        model_name: HuggingFace model identifier
        batch_size: Batch size for processing (smaller = less memory)
        max_length: Maximum sequence length
        device: Device to use ('cuda' or 'cpu'). If None, auto-detect.
        
    Returns:
        Embedding matrix (n_samples, embedding_dim)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    # Process in smaller batches and clear memory
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        if batch_num % 100 == 0 or batch_num == total_batches:
            print(f"      Processing batch {batch_num}/{total_batches}...", end='\r')
        
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
        
        # Extract embeddings
        with torch.no_grad():
            outputs = model(**encodings)
            # Use mean pooling of all token embeddings
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_embeddings.append(embeddings)
            
            # Clear GPU cache if using CUDA/MPS
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()
        
        # Clear intermediate variables immediately
        del encodings, outputs, embeddings
    
    print()  # New line after progress
    
    # Clean up model from memory
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
    
    return np.vstack(all_embeddings)


def train_transformer_hybrid(
    texts: List[str],
    labels: List[str],
    lexicon_extractor: LexiconFeatureExtractor,
    tfidf_extractor: TFIDFFeatureExtractor,
    label_to_int: Dict[str, int],
    lang: str,
    model_name: str = "transformer_hybrid",
    transformer_model: str = "roberta-base",
    use_tuning: bool = True,
    max_length: int = 128,
    batch_size: int = 8,  # Small batch size for memory efficiency (24GB RAM)
) -> Tuple[Path, Dict]:
    """
    Train hybrid model combining transformer embeddings + lexicon + TF-IDF.
    
    Args:
        texts: Training texts
        labels: Training labels
        lexicon_extractor: Lexicon feature extractor
        tfidf_extractor: TF-IDF feature extractor
        label_to_int: Label to integer mapping
        lang: Language code
        model_name: Model name for saving
        transformer_model: HuggingFace model identifier
        use_tuning: If True, use hyperparameter tuning
        max_length: Maximum sequence length for transformer
        batch_size: Batch size for transformer embeddings
        
    Returns:
        Tuple of (model_path, metadata)
    """
    print(f"\nTraining transformer hybrid model ({transformer_model}) for {lang}...")
    
    # Convert labels to integers
    y = np.array([label_to_int[label] for label in labels])
    num_labels = len(label_to_int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_STATE, stratify=y_train
    )
    
    train_size = len(X_train)
    val_size = len(X_val)
    test_size = len(X_test)
    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Extract all feature types (memory-efficient: process and combine incrementally)
    print("\nExtracting features...")
    
    # 1. Transformer embeddings (process one split at a time to save memory)
    print("  Extracting transformer embeddings (this may take a while)...")
    print("    Training set...")
    train_transformer_emb = extract_transformer_embeddings(
        X_train, transformer_model, batch_size, max_length
    )
    print(f"    Transformer embeddings shape: {train_transformer_emb.shape}")
    
    print("    Validation set...")
    val_transformer_emb = extract_transformer_embeddings(
        X_val, transformer_model, batch_size, max_length
    )
    
    print("    Test set...")
    test_transformer_emb = extract_transformer_embeddings(
        X_test, transformer_model, batch_size, max_length
    )
    
    # 2. TF-IDF features
    print("  Extracting TF-IDF features...")
    tfidf_extractor.fit(X_train)
    train_tfidf = tfidf_extractor.transform(X_train)
    val_tfidf = tfidf_extractor.transform(X_val)
    test_tfidf = tfidf_extractor.transform(X_test)
    print(f"    TF-IDF features shape: {train_tfidf.shape}")
    
    # 3. Lexicon features
    print("  Extracting lexicon features...")
    train_lexicon = lexicon_extractor.extract_batch(X_train)
    val_lexicon = lexicon_extractor.extract_batch(X_val)
    test_lexicon = lexicon_extractor.extract_batch(X_test)
    print(f"    Lexicon features shape: {train_lexicon.shape}")
    
    # Scale lexicon features
    from sklearn.preprocessing import StandardScaler
    lexicon_scaler = StandardScaler()
    train_lexicon_scaled = lexicon_scaler.fit_transform(train_lexicon)
    val_lexicon_scaled = lexicon_scaler.transform(val_lexicon)
    test_lexicon_scaled = lexicon_scaler.transform(test_lexicon)
    
    # Combine all features (memory-efficient: convert to sparse immediately)
    print("\nCombining features...")
    
    # Convert to sparse immediately to save memory
    train_lexicon_sparse = csr_matrix(train_lexicon_scaled)
    val_lexicon_sparse = csr_matrix(val_lexicon_scaled)
    test_lexicon_sparse = csr_matrix(test_lexicon_scaled)
    
    # Delete original arrays to free memory
    del train_lexicon_scaled, val_lexicon_scaled, test_lexicon_scaled
    del train_lexicon, val_lexicon, test_lexicon
    
    train_transformer_sparse = csr_matrix(train_transformer_emb)
    val_transformer_sparse = csr_matrix(val_transformer_emb)
    test_transformer_sparse = csr_matrix(test_transformer_emb)
    
    # Delete transformer embeddings to free memory
    del train_transformer_emb, val_transformer_emb, test_transformer_emb
    
    # Concatenate: TF-IDF + Transformer + Lexicon
    print("  Concatenating features...")
    X_train_combined = hstack([train_tfidf, train_transformer_sparse, train_lexicon_sparse])
    X_val_combined = hstack([val_tfidf, val_transformer_sparse, val_lexicon_sparse])
    X_test_combined = hstack([test_tfidf, test_transformer_sparse, test_lexicon_sparse])
    
    # Free intermediate sparse matrices
    del train_transformer_sparse, val_transformer_sparse, test_transformer_sparse
    del train_lexicon_sparse, val_lexicon_sparse, test_lexicon_sparse
    del train_tfidf, val_tfidf, test_tfidf
    
    print(f"  Combined features shape: {X_train_combined.shape}")
    
    # Use sparse matrices directly - avoid feature selection to save memory
    # Most sklearn models can handle sparse matrices efficiently
    print("  Using sparse matrices for training (memory-efficient)...")
    
    selector = None
    
    # Keep as sparse - sklearn LogisticRegression and LinearSVC handle sparse well
    X_train_dense = X_train_combined
    X_val_dense = X_val_combined
    X_test_dense = X_test_combined
    
    # Only convert to dense if absolutely necessary and small enough
    # For 75k features, we'll use sparse matrices
    if hasattr(X_train_combined, 'toarray'):
        # Check if we can afford dense conversion
        estimated_memory_gb = (X_train_combined.shape[0] * X_train_combined.shape[1] * 8) / (1024**3)
        if estimated_memory_gb < 8:  # Only if less than 8GB
            print(f"  Estimated memory: {estimated_memory_gb:.2f}GB - converting to dense...")
            X_train_dense = X_train_combined.toarray()
            X_val_dense = X_val_combined.toarray()
            X_test_dense = X_test_combined.toarray()
            del X_train_combined, X_val_combined, X_test_combined
        else:
            print(f"  Estimated memory: {estimated_memory_gb:.2f}GB - keeping sparse (memory-efficient)")
            # Keep sparse - will work with LogisticRegression and LinearSVC
    else:
        # Already dense
        pass
    
    
    # Try multiple models (both support sparse matrices)
    print("\nTraining and comparing models...")
    models_to_try = {}
    
    # Note: Both LogisticRegression and LinearSVC support sparse matrices
    # Use 'liblinear' or 'saga' solver for LogisticRegression with sparse matrices
    # LinearSVC natively supports sparse matrices
    
    # 1. Logistic Regression (use saga for sparse matrices + multiclass)
    if use_tuning:
        lr_param_grid = {
            'C': [0.5, 1.0, 2.0, 5.0],
            'solver': ['saga'],  # saga supports sparse matrices and multiclass
            'max_iter': [2000, 5000],
        }
        lr_base = LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE)
        lr_search = RandomizedSearchCV(
            lr_base, lr_param_grid, n_iter=8, cv=3,
            scoring='f1_macro', n_jobs=-1, random_state=RANDOM_STATE, verbose=1
        )
        lr_search.fit(X_train_dense, y_train)
        models_to_try['LogisticRegression'] = lr_search.best_estimator_
        print(f"  LR best params: {lr_search.best_params_}, score: {lr_search.best_score_:.4f}")
    else:
        models_to_try['LogisticRegression'] = LogisticRegression(
            C=2.0, solver='saga', max_iter=5000,  # saga for sparse + multiclass
            class_weight='balanced', random_state=RANDOM_STATE
        )
        models_to_try['LogisticRegression'].fit(X_train_dense, y_train)
    
    # 2. Linear SVM (natively supports sparse matrices)
    if use_tuning:
        svm_param_grid = {
            'C': [0.5, 1.0, 2.0],
            'loss': ['squared_hinge'],
        }
        svm_base = LinearSVC(class_weight='balanced', random_state=RANDOM_STATE, max_iter=5000)
        svm_search = RandomizedSearchCV(
            svm_base, svm_param_grid, n_iter=3, cv=3,
            scoring='f1_macro', n_jobs=-1, random_state=RANDOM_STATE, verbose=1
        )
        svm_search.fit(X_train_dense, y_train)
        models_to_try['LinearSVM'] = svm_search.best_estimator_
        print(f"  SVM best params: {svm_search.best_params_}, score: {svm_search.best_score_:.4f}")
    else:
        models_to_try['LinearSVM'] = LinearSVC(
            C=1.0, loss='squared_hinge', class_weight='balanced',
            random_state=RANDOM_STATE, max_iter=5000
        )
        models_to_try['LinearSVM'].fit(X_train_dense, y_train)
    
    # Evaluate models on validation set
    print("\nEvaluating models on validation set...")
    val_scores = {}
    for name, model in models_to_try.items():
        y_pred = model.predict(X_val_dense)
        score = f1_score(y_val, y_pred, average='macro')
        val_scores[name] = score
        print(f"  {name}: F1-macro = {score:.4f}")
    
    # Select best model
    best_model_name = max(val_scores, key=val_scores.get)
    best_model = models_to_try[best_model_name]
    print(f"\nBest model: {best_model_name} (F1-macro: {val_scores[best_model_name]:.4f})")
    
    # Final evaluation on test set
    test_score = best_model.score(X_test_dense, y_test)
    test_f1 = f1_score(y_test, best_model.predict(X_test_dense), average='macro')
    print(f"Test accuracy: {test_score:.4f}, Test F1-macro: {test_f1:.4f}")
    
    # Save model
    model_dir = MODELS_DIR / lang
    model_path = model_dir / f"{model_name}.joblib"
    
    metadata = {
        "model_type": model_name,
        "model_algorithm": best_model_name,
        "transformer_model": transformer_model,
        "lang": lang,
        "n_features": X_train_combined.shape[1],
        "n_classes": len(label_to_int),
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "val_accuracy": float(best_model.score(X_val_dense, y_val)),
        "val_f1_macro": float(val_scores[best_model_name]),
        "test_accuracy": float(test_score),
        "test_f1_macro": float(test_f1),
        "feature_types": ["tfidf", "transformer", "lexicon"],
    }
    
    # Save model bundle
    # Create a fusion-like object for compatibility
    class HybridFusion:
        def __init__(self, tfidf_extractor, lexicon_extractor, lexicon_scaler, transformer_model):
            self.tfidf_extractor = tfidf_extractor
            self.lexicon_extractor = lexicon_extractor
            self.scaler = lexicon_scaler
            self.transformer_model = transformer_model
    
    hybrid_fusion = HybridFusion(tfidf_extractor, lexicon_extractor, lexicon_scaler, transformer_model)
    
    save_model_bundle(
        model_path=model_path,
        vectorizer=hybrid_fusion,
        scaler=None,  # Already scaled
        classifier=best_model,
        label_mapping=label_to_int,
        metadata=metadata,
    )
    
    print(f"Model saved to {model_path}")
    
    return model_path, metadata
