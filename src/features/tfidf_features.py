"""
TF-IDF feature extraction.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
import numpy as np

from src.config import TFIDF_MAX_FEATURES, TFIDF_WORD_NGRAMS, TFIDF_CHAR_NGRAMS


class TFIDFFeatureExtractor:
    """
    Extract TF-IDF features from text.
    """
    
    def __init__(
        self,
        max_features: int = TFIDF_MAX_FEATURES,
        word_ngrams: tuple = TFIDF_WORD_NGRAMS,
        char_ngrams: tuple = TFIDF_CHAR_NGRAMS,
    ):
        """
        Initialize TF-IDF extractor.
        
        Args:
            max_features: Maximum number of features
            word_ngrams: Tuple of (min, max) word n-gram range
            char_ngrams: Tuple of (min, max) character n-gram range
        """
        self.max_features = max_features
        self.word_ngrams = word_ngrams
        self.char_ngrams = char_ngrams
        
        # Create vectorizer with both word and char n-grams
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=word_ngrams,
            analyzer="word",
            lowercase=True,
            token_pattern=r'\b\w+\b',
        )
        
        # Character n-gram vectorizer (if needed)
        self.char_vectorizer = TfidfVectorizer(
            max_features=max_features // 2,  # Split features between word and char
            ngram_range=char_ngrams,
            analyzer="char",
            lowercase=True,
        )
        
        self._fitted = False
    
    def fit(self, texts: List[str]):
        """
        Fit vectorizers on training texts.
        
        Args:
            texts: List of training texts
        """
        # Fit word vectorizer
        self.vectorizer.fit(texts)
        
        # Fit char vectorizer
        self.char_vectorizer.fit(texts)
        
        self._fitted = True
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to TF-IDF features.
        
        Args:
            texts: List of texts
            
        Returns:
            Sparse matrix of TF-IDF features
        """
        if not self._fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        # Get word features
        word_features = self.vectorizer.transform(texts)
        
        # Get char features
        char_features = self.char_vectorizer.transform(texts)
        
        # Combine (hstack)
        from scipy.sparse import hstack
        combined = hstack([word_features, char_features])
        
        return combined
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Sparse matrix of TF-IDF features
        """
        self.fit(texts)
        return self.transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names (word + char n-grams).
        
        Returns:
            List of feature names
        """
        if not self._fitted:
            return []
        word_names = [f"word_{name}" for name in self.vectorizer.get_feature_names_out()]
        char_names = [f"char_{name}" for name in self.char_vectorizer.get_feature_names_out()]
        return word_names + char_names

