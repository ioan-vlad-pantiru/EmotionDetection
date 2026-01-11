"""
Text preprocessing utilities for English and Romanian.
"""
import re
import emoji
from typing import List, Tuple


# Negation cues
NEGATION_CUES_EN = {"not", "never", "no", "n't", "cannot", "can't", "won't", "don't", "didn't"}
NEGATION_CUES_RO = {"nu", "niciodată", "fără", "nici", "niciodată", "niciun", "niciuna"}


def preprocess_text(text: str, lang: str = "en") -> Tuple[str, List[int]]:
    """
    Preprocess text: lowercase, remove URLs/mentions, normalize, tokenize.
    Also returns negation positions for lexicon feature adjustment.
    
    Args:
        text: Input text
        lang: Language code ('en' or 'ro')
        
    Returns:
        Tuple of (preprocessed_text, negation_positions)
        negation_positions: List of token indices that are in negation window
    """
    if not text:
        return "", []
    
    # Convert emojis to text
    text = emoji.demojize(text, delimiters=(" ", " "))
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove mentions but keep hashtag text
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)  # Keep hashtag text
    
    # Normalize elongated characters (soooo -> soo)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Lowercase
    text = text.lower()
    
    # Tokenize (simple regex)
    tokens = re.findall(r'\b\w+\b', text)
    
    # Find negation positions
    negation_positions = find_negation_positions(tokens, lang)
    
    # Join tokens back for TF-IDF
    preprocessed_text = " ".join(tokens)
    
    return preprocessed_text, negation_positions


def find_negation_positions(tokens: List[str], lang: str) -> List[int]:
    """
    Find token positions that are within negation window.
    Improved to handle more negation patterns.
    
    Args:
        tokens: List of tokens
        lang: Language code
        
    Returns:
        List of token indices within negation window
    """
    negation_cues = NEGATION_CUES_EN if lang == "en" else NEGATION_CUES_RO
    negation_positions = set()
    
    for i, token in enumerate(tokens):
        if token in negation_cues:
            # Extended window: mark tokens in window after negation (up to 5 tokens)
            # This catches cases like "nu sunt fericit" where "fericit" is 2 tokens after "nu"
            for j in range(i + 1, min(i + 1 + 5, len(tokens))):
                negation_positions.add(j)
    
    return sorted(negation_positions)


def normalize_diacritics(text: str) -> str:
    """
    Normalize Romanian diacritics (ș/ş, ț/ţ variants).
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Map common variants to standard forms
    replacements = {
        'ş': 'ș',
        'Ş': 'Ș',
        'ţ': 'ț',
        'Ţ': 'Ț',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def extract_stylistic_features(text: str) -> dict:
    """
    Extract stylistic features: exclamation count, uppercase ratio, repeated punctuation.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of stylistic features
    """
    features = {}
    
    # Exclamation count
    features["exclamation_count"] = text.count("!")
    
    # Question mark count
    features["question_count"] = text.count("?")
    
    # Repeated punctuation patterns
    features["repeated_exclamation"] = len(re.findall(r'!{2,}', text))
    features["repeated_question"] = len(re.findall(r'\?{2,}', text))
    
    # Uppercase token ratio
    tokens = re.findall(r'\b\w+\b', text)
    if tokens:
        uppercase_count = sum(1 for t in tokens if t.isupper() and len(t) > 1)
        features["uppercase_ratio"] = uppercase_count / len(tokens)
    else:
        features["uppercase_ratio"] = 0.0
    
    return features


