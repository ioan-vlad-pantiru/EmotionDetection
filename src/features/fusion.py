"""
Feature fusion: combine lexicon and TF-IDF features.
"""
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import StandardScaler
from typing import List

from src.features.lexicon_features import LexiconFeatureExtractor
from src.features.tfidf_features import TFIDFFeatureExtractor


class FeatureFusion:
    """
    Combine lexicon and TF-IDF features.
    """
    
    def __init__(
        self,
        lexicon_extractor: LexiconFeatureExtractor,
        tfidf_extractor: TFIDFFeatureExtractor,
        scale_lexicon: bool = True,
    ):
        """
        Initialize feature fusion.
        
        Args:
            lexicon_extractor: Lexicon feature extractor
            tfidf_extractor: TF-IDF feature extractor
            scale_lexicon: Whether to scale lexicon features
        """
        self.lexicon_extractor = lexicon_extractor
        self.tfidf_extractor = tfidf_extractor
        self.scale_lexicon = scale_lexicon
        
        if scale_lexicon:
            self.scaler = StandardScaler(with_mean=False)  # Don't center sparse-friendly
        else:
            self.scaler = None
        
        self._fitted = False
    
    def fit(self, texts: List[str]):
        """
        Fit fusion pipeline on training texts.
        
        Args:
            texts: List of training texts
        """
        # Fit TF-IDF extractor
        self.tfidf_extractor.fit(texts)
        
        # Extract lexicon features for scaling
        lexicon_features = self.lexicon_extractor.extract_batch(texts)
        
        # Fit scaler if needed
        if self.scaler is not None:
            self.scaler.fit(lexicon_features)
        
        self._fitted = True
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to fused features.
        
        Args:
            texts: List of texts
            
        Returns:
            Combined feature matrix (sparse or dense)
        """
        if not self._fitted:
            raise ValueError("Fusion must be fitted before transform")
        
        # Get TF-IDF features (sparse)
        tfidf_features = self.tfidf_extractor.transform(texts)
        
        # Get lexicon features (dense)
        # Determine expected feature count from scaler
        expected_features = None
        if self.scaler is not None:
            expected_features = self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else self.scaler.mean_.shape[0] if hasattr(self.scaler, 'mean_') else None
        
        # Extract with or without negation based on scaler expectations
        if expected_features == 21:
            # Old model expects 21 features (without negation)
            lexicon_features = self.lexicon_extractor.extract_batch(texts, include_negation=False)
        else:
            # New model or unknown - extract with negation features
            lexicon_features = self.lexicon_extractor.extract_batch(texts, include_negation=True)
        
        # Scale lexicon features
        if self.scaler is not None:
            lexicon_features = self.scaler.transform(lexicon_features)
        
        # Convert lexicon to sparse for hstack
        lexicon_sparse = csr_matrix(lexicon_features)
        
        # Concatenate
        fused_features = hstack([tfidf_features, lexicon_sparse])
        
        return fused_features
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Combined feature matrix
        """
        self.fit(texts)
        return self.transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """
        Get combined feature names.
        
        Returns:
            List of feature names
        """
        tfidf_names = self.tfidf_extractor.get_feature_names()
        lexicon_names = self.lexicon_extractor.get_feature_names()
        return tfidf_names + lexicon_names


