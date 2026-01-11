"""
Improved negation handling for emotion detection.
"""
from typing import List, Dict, Set
from src.config import PLUTCHIK_8

# Emotion inversion mapping (opposite emotions)
EMOTION_INVERSIONS = {
    "joy": "sadness",
    "sadness": "joy",
    "anger": "trust",
    "trust": "anger",
    "fear": "anticipation",
    "anticipation": "fear",
    "surprise": "anticipation",  # Less clear, but surprise often negates anticipation
    "disgust": "trust",  # Disgust often opposes trust/acceptance
    "neutral": "neutral",  # Neutral stays neutral
}


def invert_emotion(emotion: str) -> str:
    """Get the inverse/opposite emotion."""
    return EMOTION_INVERSIONS.get(emotion, emotion)


def has_negation_with_emotion(tokens: List[str], negation_positions: Set[int], 
                               emotion_words: Set[str]) -> Dict[str, bool]:
    """
    Check if negation co-occurs with emotion words.
    
    Args:
        tokens: List of tokens
        negation_positions: Set of token indices in negation window
        emotion_words: Set of emotion-related words
        
    Returns:
        Dictionary mapping emotion -> whether it appears with negation
    """
    result = {}
    for i, token in enumerate(tokens):
        if i in negation_positions and token in emotion_words:
            # This emotion word is negated
            result[token] = True
    return result


def get_negation_features(tokens: List[str], negation_positions: List[int], 
                          lang: str) -> Dict[str, float]:
    """
    Extract explicit negation features.
    
    Args:
        tokens: List of tokens
        negation_positions: List of token indices in negation window
        lang: Language code
        
    Returns:
        Dictionary of negation features
    """
    negation_positions_set = set(negation_positions)
    
    features = {
        "has_negation": 1.0 if len(negation_positions) > 0 else 0.0,
        "negation_window_size": float(len(negation_positions)),
        "negation_ratio": float(len(negation_positions)) / max(len(tokens), 1),
    }
    
    # Check if negation affects emotion words
    if len(negation_positions) > 0:
        # Count how many tokens in negation window
        features["negated_tokens"] = float(len(negation_positions))
    else:
        features["negated_tokens"] = 0.0
    
    return features
