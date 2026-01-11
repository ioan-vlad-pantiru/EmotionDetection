"""
I/O utilities for saving and loading models and data.
"""
import joblib
import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict


def save_model_bundle(
    model_path: Path,
    vectorizer: Any = None,
    scaler: Any = None,
    classifier: Any = None,
    label_mapping: Dict[str, int] = None,
    metadata: Dict[str, Any] = None,
):
    """
    Save a complete model bundle (vectorizer, scaler, classifier, mappings).
    
    Args:
        model_path: Path to save the bundle
        vectorizer: TF-IDF vectorizer (can be None for lexicon-only)
        scaler: Feature scaler (can be None)
        classifier: Trained classifier
        label_mapping: Mapping from label strings to integers
        metadata: Additional metadata dictionary
    """
    bundle = {
        "vectorizer": vectorizer,
        "scaler": scaler,
        "classifier": classifier,
        "label_mapping": label_mapping,
        "metadata": metadata or {},
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)


def load_model_bundle(model_path: Path) -> Dict[str, Any]:
    """
    Load a complete model bundle.
    
    Args:
        model_path: Path to the bundle file
        
    Returns:
        Dictionary containing vectorizer, scaler, classifier, label_mapping, metadata
    """
    return joblib.load(model_path)


def save_metrics(metrics: Dict[str, Any], output_path: Path, format: str = "json"):
    """
    Save evaluation metrics to file.
    
    Args:
        metrics: Dictionary of metrics
        output_path: Output file path
        format: 'json' or 'csv'
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
    elif format == "csv":
        # Flatten nested dictionaries for CSV
        flat_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_metrics[f"{key}_{subkey}"] = subvalue
            else:
                flat_metrics[key] = value
        df = pd.DataFrame([flat_metrics])
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unknown format: {format}")


def save_error_analysis(errors: list, output_path: Path):
    """
    Save error analysis results to JSON.
    
    Args:
        errors: List of error dictionaries
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)


