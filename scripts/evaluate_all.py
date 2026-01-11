"""
Evaluate all models and generate comparison reports.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import List
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import LANGUAGES, MODEL_TYPES, MODELS_DIR, METRICS_DIR, ERRORS_DIR
from src.datasets.goemotions import load_goemotions_simple
from src.datasets.red_ro import load_red
from src.lexicons.roemolex import RoEmoLex
from src.lexicons.emolex_en import EmoLexEN
from src.features.lexicon_features import LexiconFeatureExtractor
from src.features.tfidf_features import TFIDFFeatureExtractor
from src.features.fusion import FeatureFusion
from src.models.infer import predict_lexicon_only, predict_ml_only, predict_hybrid
from src.utils.metrics import compute_metrics, print_metrics_summary
from src.utils.io import save_metrics, save_error_analysis


def categorize_error(text: str, gold: str, pred: str) -> dict:
    """
    Categorize error type.
    
    Args:
        text: Input text
        gold: Gold label
        pred: Predicted label
        
    Returns:
        Dictionary with error categories
    """
    categories = {
        "has_negation": False,
        "has_emoji": False,
        "is_short": len(text.split()) < 5,
        "has_sarcasm": False,
    }
    
    # Check for negation
    negation_words = ["not", "never", "no", "nu", "niciodată", "fără"]
    categories["has_negation"] = any(word in text.lower() for word in negation_words)
    
    # Check for emoji (simple check)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    categories["has_emoji"] = bool(emoji_pattern.search(text))
    
    # Check for sarcasm markers
    sarcasm_markers = ["yeah right", "sure", "lol", "haha", "right", "sigur", "bine"]
    text_lower = text.lower()
    categories["has_sarcasm"] = any(marker in text_lower for marker in sarcasm_markers)
    
    return categories


def get_top_tfidf_features(text: str, vectorizer, top_k: int = 10) -> list:
    """
    Get top TF-IDF features for a text.
    
    Args:
        text: Input text
        vectorizer: Fitted TF-IDF vectorizer
        top_k: Number of top features to return
        
    Returns:
        List of top feature names
    """
    try:
        X = vectorizer.transform([text])
        feature_names = vectorizer.get_feature_names_out()
        
        # Get non-zero features
        nonzero_indices = X.nonzero()[1]
        scores = X.data
        
        # Sort by score
        if len(scores) > 0:
            top_indices = nonzero_indices[np.argsort(scores)[-top_k:]]
            top_features = [feature_names[i] for i in top_indices]
            return top_features[:top_k]
    except:
        pass
    
    return []


def evaluate_model(
    texts: List[str],
    labels: List[str],
    label_to_int: dict,
    model_path: Path,
    model_type: str,
    lexicon_extractor=None,
    tfidf_extractor=None,
    fusion=None,
    lang: str = "en",
) -> tuple:
    """
    Evaluate a model and return metrics and errors.
    
    Returns:
        Tuple of (metrics_dict, error_samples)
    """
    # Convert labels to integers
    y_true = np.array([label_to_int[label] for label in labels])
    
    # Predict
    if model_type == "lexicon":
        y_pred, pred_labels = predict_lexicon_only(texts, model_path, lexicon_extractor)
    elif model_type == "ml":
        y_pred, pred_labels = predict_ml_only(texts, model_path, tfidf_extractor)
    elif model_type == "hybrid":
        y_pred, pred_labels = predict_hybrid(texts, model_path, fusion)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Compute metrics
    int_to_label = {v: k for k, v in label_to_int.items()}
    labels_list = [int_to_label[i] for i in sorted(label_to_int.values())]
    
    metrics = compute_metrics(y_true, y_pred, labels_list, label_to_int)
    
    # Collect error samples
    errors = []
    misclassified_indices = np.where(y_true != y_pred)[0]
    
    # Get top 50 errors
    top_errors = misclassified_indices[:50]
    
    for idx in top_errors:
        text = texts[idx]
        gold_label = int_to_label[y_true[idx]]
        pred_label = int_to_label[y_pred[idx]]
        
        # Get feature information
        error_info = {
            "text": text,
            "gold_label": gold_label,
            "predicted_label": pred_label,
            "categories": categorize_error(text, gold_label, pred_label),
        }
        
        # Add TF-IDF features if available
        if model_type in ["ml", "hybrid"]:
            if model_type == "ml" and tfidf_extractor:
                error_info["top_tfidf_features"] = get_top_tfidf_features(
                    text, tfidf_extractor.vectorizer
                )
            elif model_type == "hybrid" and fusion:
                error_info["top_tfidf_features"] = get_top_tfidf_features(
                    text, fusion.tfidf_extractor.vectorizer
                )
        
        # Add lexicon features if available
        if model_type in ["lexicon", "hybrid"] and lexicon_extractor:
            tokens = re.findall(r'\b\w+\b', text.lower())
            emotion_scores = lexicon_extractor.lexicon.get_emotion_scores(tokens)
            error_info["lexicon_emotions"] = emotion_scores
        
        errors.append(error_info)
    
    return metrics, errors


def main():
    print("=" * 60)
    print("Evaluating all models")
    print("=" * 60)
    
    all_results = []
    
    for lang in LANGUAGES:
        print(f"\n{'='*60}")
        print(f"Language: {lang.upper()}")
        print(f"{'='*60}")
        
        # Load dataset
        print(f"\nLoading {lang} dataset...")
        try:
            if lang == "en":
                texts, labels, label_to_int = load_goemotions_simple()
            else:
                texts, labels, label_to_int = load_red()
            
            print(f"Loaded {len(texts)} examples")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            continue
        
        # Load lexicon (optional - needed only for lexicon and hybrid models)
        lexicon = None
        lexicon_extractor = None
        fusion = None
        
        print(f"\nLoading {lang} lexicon...")
        try:
            if lang == "en":
                lexicon = EmoLexEN()
            else:
                lexicon = RoEmoLex()
            
            # Create lexicon-based extractors only if lexicon loaded successfully
            lexicon_extractor = LexiconFeatureExtractor(lexicon, lang=lang)
            fusion = FeatureFusion(lexicon_extractor, TFIDFFeatureExtractor())
        except Exception as e:
            print(f"⚠ Warning: Lexicon not available: {e}")
            print("  Will evaluate ML-only model. Lexicon and hybrid models will be skipped.")
        
        # Create TF-IDF extractor (always available)
        tfidf_extractor = TFIDFFeatureExtractor()
        
        # Fit extractors on full dataset
        print("Fitting feature extractors...")
        tfidf_extractor.fit(texts)
        if fusion is not None:
            fusion.fit(texts)
        
        # Evaluate each model
        for model_type in MODEL_TYPES:
            model_path = MODELS_DIR / lang / f"{model_type}.joblib"
            
            if not model_path.exists():
                print(f"\n⚠ Model not found: {model_path}")
                print("   Skipping this model. Train it first using: python scripts/train_all.py")
                continue
            
            # Check if model type requires lexicon
            if model_type in ["lexicon", "hybrid"]:
                if lexicon_extractor is None:
                    print(f"\n⚠ Skipping {model_type} model: lexicon not available")
                    continue
            
            print(f"\n{'─'*60}")
            print(f"Evaluating {model_type} model...")
            print(f"{'─'*60}")
            
            try:
                metrics, errors = evaluate_model(
                    texts,
                    labels,
                    label_to_int,
                    model_path,
                    model_type,
                    lexicon_extractor=lexicon_extractor,
                    tfidf_extractor=tfidf_extractor,
                    fusion=fusion,
                    lang=lang,
                )
                
                # Print metrics
                print_metrics_summary(metrics)
                
                # Save metrics
                metrics_path = METRICS_DIR / f"{lang}_{model_type}_metrics.json"
                save_metrics(metrics, metrics_path)
                print(f"Metrics saved to {metrics_path}")
                
                # Save errors
                errors_path = ERRORS_DIR / f"{lang}_{model_type}_errors.json"
                save_error_analysis(errors, errors_path)
                print(f"Error analysis saved to {errors_path}")
                
                # Store for comparison table
                all_results.append({
                    "lang": lang,
                    "model": model_type,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                })
                
            except Exception as e:
                print(f"Error evaluating {model_type} model: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Create comparison table
    if all_results:
        print("\n" + "=" * 60)
        print("COMPARISON TABLE")
        print("=" * 60)
        
        df = pd.DataFrame(all_results)
        df = df.pivot(index="lang", columns="model", values=["accuracy", "macro_f1", "weighted_f1"])
        print(df)
        
        # Save comparison
        comparison_path = METRICS_DIR / "comparison_table.csv"
        df.to_csv(comparison_path)
        print(f"\nComparison table saved to {comparison_path}")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

