"""
Improved TF-IDF feature extraction with better parameters.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
import numpy as np

from src.config import TFIDF_MAX_FEATURES, TFIDF_WORD_NGRAMS, TFIDF_CHAR_NGRAMS


class TFIDFFeatureExtractorImproved:
    """
    Improved TF-IDF extractor with better parameters for emotion detection.
    """
    
    def __init__(
        self,
        max_features: int = TFIDF_MAX_FEATURES,
        word_ngrams: tuple = (1, 3),  # Extended to trigrams
        char_ngrams: tuple = (3, 6),  # Extended character n-grams
        min_df: int = 2,  # Minimum document frequency
        max_df: float = 0.95,  # Maximum document frequency (remove very common words)
        sublinear_tf: bool = True,  # Apply sublinear TF scaling
    ):
        """
        Initialize improved TF-IDF extractor.
        
        Args:
            max_features: Maximum number of features
            word_ngrams: Tuple of (min, max) word n-gram range
            char_ngrams: Tuple of (min, max) character n-gram range
            min_df: Minimum document frequency (ignore terms with lower frequency)
            max_df: Maximum document frequency (ignore terms with higher frequency)
            sublinear_tf: Apply sublinear TF scaling (1 + log(tf))
        """
        self.max_features = max_features
        self.word_ngrams = word_ngrams
        self.char_ngrams = char_ngrams
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        
        # Create word vectorizer with improved parameters
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=word_ngrams,
            analyzer="word",
            lowercase=True,
            token_pattern=r'\b\w+\b',
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            norm='l2',  # L2 normalization
            smooth_idf=True,  # Smooth IDF weights
        )
        
        # Character n-gram vectorizer with improved parameters
        self.char_vectorizer = TfidfVectorizer(
            max_features=max_features // 2,  # Split features between word and char
            ngram_range=char_ngrams,
            analyzer="char",
            lowercase=True,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            norm='l2',
            smooth_idf=True,
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
