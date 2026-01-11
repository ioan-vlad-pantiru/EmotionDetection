"""
Evaluation metrics for emotion classification.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from typing import Dict, List, Tuple
import json


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    label_to_int: Dict[str, int],
) -> Dict:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels (integer encoded)
        y_pred: Predicted labels (integer encoded)
        labels: List of label strings
        label_to_int: Mapping from label string to integer
        
    Returns:
        Dictionary of metrics
    """
    # Convert integer labels back to strings for reporting
    int_to_label = {v: k for k, v in label_to_int.items()}
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Macro averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    
    # Weighted averages
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-class metrics dictionary
    per_class = {}
    for i, label in enumerate(labels):
        if i < len(precision):
            per_class[label] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
    
    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }
    
    return metrics


def print_metrics_summary(metrics: Dict):
    """
    Print a human-readable summary of metrics.
    
    Args:
        metrics: Metrics dictionary from compute_metrics
    """
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print("\nPer-class metrics:")
    print("-" * 60)
    for label, scores in metrics["per_class"].items():
        print(f"{label:15s} | P: {scores['precision']:.3f} | R: {scores['recall']:.3f} | F1: {scores['f1']:.3f} | Support: {scores['support']}")
    print("="*60 + "\n")


