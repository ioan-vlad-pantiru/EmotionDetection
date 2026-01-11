"""
Label mapping utilities for converting dataset labels to Plutchik 8 emotions.
"""
from typing import Dict, List, Optional
from src.config import PLUTCHIK_8, NEUTRAL_LABEL


# GoEmotions to Plutchik 8 mapping
GOEMOTIONS_TO_PLUTCHIK = {
    # Direct mappings
    "anger": "anger",
    "fear": "fear",
    "joy": "joy",
    "sadness": "sadness",
    "disgust": "disgust",
    "surprise": "surprise",
    
    # Anticipation mappings
    "curiosity": "anticipation",
    "desire": "anticipation",
    "optimism": "anticipation",
    
    # Trust mappings
    "approval": "trust",
    "admiration": "trust",
    "gratitude": "trust",
    "caring": "trust",
    
    # Neutral
    "neutral": NEUTRAL_LABEL,
    
    # Other mappings (map to nearest emotion)
    "amusement": "joy",
    "excitement": "joy",
    "love": "joy",
    "relief": "joy",
    "pride": "joy",
    
    "nervousness": "fear",
    "embarrassment": "fear",
    
    "disappointment": "sadness",
    "grief": "sadness",
    "remorse": "sadness",
    
    "annoyance": "anger",
    "disapproval": "anger",
    
    # Confusion -> surprise
    "confusion": "surprise",
    
    # Realization -> surprise
    "realization": "surprise",
}


def map_goemotions_label(label: str) -> Optional[str]:
    """
    Map GoEmotions label to Plutchik 8 emotion.
    
    Args:
        label: GoEmotions label string
        
    Returns:
        Mapped label (Plutchik 8 emotion or neutral), or None if should be dropped
    """
    return GOEMOTIONS_TO_PLUTCHIK.get(label.lower(), None)


def map_red_label(label: str) -> Optional[str]:
    """
    Map RED/REDv2 label to Plutchik 8 emotion.
    
    Args:
        label: RED label string
        
    Returns:
        Mapped label (Plutchik 8 emotion or neutral), or None if should be dropped
    """
    label_lower = label.lower().strip()
    
    # Direct mappings
    mapping = {
        "anger": "anger",
        "fear": "fear",
        "joy": "joy",
        "sadness": "sadness",
        "surprise": "surprise",
        "trust": "trust",
        "neutral": NEUTRAL_LABEL,
        "disgust": "disgust",
        "anticipation": "anticipation",
    }
    
    # Romanian variants
    ro_mapping = {
        "furie": "anger",
        "frică": "fear",
        "bucurie": "joy",
        "tristețe": "sadness",
        "surpriză": "surprise",
        "încredere": "trust",
        "dezgust": "disgust",
        "anticipare": "anticipation",
        "neutru": NEUTRAL_LABEL,
    }
    
    result = mapping.get(label_lower) or ro_mapping.get(label_lower)
    return result


def create_label_mapping(labels: List[str], include_neutral: bool = True) -> Dict[str, int]:
    """
    Create integer mapping for labels.
    
    Args:
        labels: List of unique label strings
        include_neutral: Whether to include neutral label
        
    Returns:
        Dictionary mapping label string -> integer
    """
    # Sort labels for consistency
    sorted_labels = sorted(set(labels))
    
    # Create mapping
    label_to_int = {}
    for idx, label in enumerate(sorted_labels):
        label_to_int[label] = idx
    
    return label_to_int


def get_available_labels(labels: List[str], target_labels: List[str]) -> List[str]:
    """
    Get intersection of available labels and target labels.
    
    Args:
        labels: Available labels from dataset
        target_labels: Target labels (Plutchik 8 + neutral)
        
    Returns:
        List of labels that are in both sets
    """
    return sorted(set(labels) & set(target_labels))


