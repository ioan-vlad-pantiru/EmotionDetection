"""
Lexicon-based feature extraction.
"""
import numpy as np
from typing import List, Dict
import re

from src.config import PLUTCHIK_8
from src.utils.text import preprocess_text, extract_stylistic_features


class LexiconFeatureExtractor:
    """
    Extract lexicon-based features from text.
    """
    
    def __init__(self, lexicon, lang: str = "en"):
        """
        Initialize feature extractor.
        
        Args:
            lexicon: Lexicon object (RoEmoLex or EmoLexEN)
            lang: Language code
        """
        self.lexicon = lexicon
        self.lang = lang
        self.emotion_names = PLUTCHIK_8
    
    def extract(self, text: str) -> np.ndarray:
        """
        Extract lexicon features from text.
        
        Args:
            text: Input text
            
        Returns:
            Feature vector as numpy array
        """
        # Preprocess and get negation positions
        preprocessed_text, negation_positions = preprocess_text(text, self.lang)
        
        # Tokenize
        tokens = re.findall(r'\b\w+\b', preprocessed_text.lower())
        
        # Get emotion scores and counts
        emotion_scores = self.lexicon.get_emotion_scores(tokens, negation_positions)
        emotion_counts = self.lexicon.get_emotion_counts(tokens, negation_positions)
        
        # Extract stylistic features
        stylistic = extract_stylistic_features(text)
        
        # Build feature vector
        features = []
        feature_names = []
        
        # Emotion counts (8 features)
        for emotion in self.emotion_names:
            count = emotion_counts.get(emotion, 0)
            features.append(count)
            feature_names.append(f"count_{emotion}")
        
        # Emotion intensity sums (8 features)
        for emotion in self.emotion_names:
            score = emotion_scores.get(emotion, 0.0)
            features.append(score)
            feature_names.append(f"intensity_{emotion}")
        
        # Stylistic features
        features.append(stylistic.get("exclamation_count", 0))
        feature_names.append("exclamation_count")
        
        features.append(stylistic.get("question_count", 0))
        feature_names.append("question_count")
        
        features.append(stylistic.get("repeated_exclamation", 0))
        feature_names.append("repeated_exclamation")
        
        features.append(stylistic.get("repeated_question", 0))
        feature_names.append("repeated_question")
        
        features.append(stylistic.get("uppercase_ratio", 0.0))
        feature_names.append("uppercase_ratio")
        
        # Store feature names for later use
        self.feature_names = feature_names
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        if not hasattr(self, "feature_names"):
            # Extract from dummy text to initialize
            self.extract("dummy")
        return self.feature_names
    
    def extract_batch(self, texts: List[str]) -> np.ndarray:
        """
        Extract features for a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        features_list = [self.extract(text) for text in texts]
        return np.vstack(features_list)


