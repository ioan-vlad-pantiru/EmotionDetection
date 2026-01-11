"""
Model manager for loading and managing emotion detection models.
"""
import sys
from pathlib import Path
from typing import Dict, Optional
import asyncio

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.config import MODELS_DIR, LANGUAGES, MODEL_TYPES
from src.datasets.goemotions import load_goemotions_simple
from src.datasets.red_ro import load_red
from src.lexicons.roemolex import RoEmoLex
from src.lexicons.emolex_en import EmoLexEN
from src.features.lexicon_features import LexiconFeatureExtractor
from src.features.tfidf_features import TFIDFFeatureExtractor
from src.features.fusion import FeatureFusion


class ModelManager:
    """Manages loading and access to emotion detection models."""
    
    def __init__(self):
        """Initialize model manager."""
        self.models: Dict[str, Dict] = {}
        self.extractors: Dict[str, Dict] = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize all models and extractors asynchronously."""
        if self._initialized:
            return
        
        # Run initialization in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._initialize_sync)
        self._initialized = True
    
    def _initialize_sync(self):
        """Synchronous initialization of models."""
        print("Initializing models and extractors...")
        
        for lang in LANGUAGES:
            print(f"\nLoading {lang} models...")
            
            # Load dataset to fit extractors
            try:
                if lang == "en":
                    texts, labels, label_to_int = load_goemotions_simple()
                else:
                    texts, labels, label_to_int = load_red()
                
                print(f"Loaded {len(texts)} examples for {lang}")
            except Exception as e:
                print(f"Error loading {lang} dataset: {e}")
                continue
            
            # Load lexicon
            lexicon = None
            lexicon_extractor = None
            fusion = None
            
            try:
                if lang == "en":
                    lexicon = EmoLexEN()
                else:
                    lexicon = RoEmoLex()
                
                lexicon_extractor = LexiconFeatureExtractor(lexicon, lang=lang)
                fusion = FeatureFusion(lexicon_extractor, TFIDFFeatureExtractor())
                print(f"Lexicon loaded for {lang}")
            except Exception as e:
                print(f"Warning: Lexicon not available for {lang}: {e}")
            
            # Create TF-IDF extractor and fit it
            tfidf_extractor = TFIDFFeatureExtractor()
            tfidf_extractor.fit(texts)
            print(f"TF-IDF extractor fitted for {lang}")
            
            # Fit fusion if available
            if fusion is not None:
                fusion.fit(texts)
                print(f"Fusion extractor fitted for {lang}")
            
            # Store extractors
            self.extractors[lang] = {
                "lexicon_extractor": lexicon_extractor,
                "tfidf_extractor": tfidf_extractor,
                "fusion": fusion,
            }
            
            # Load models
            for model_type in MODEL_TYPES:
                model_path = MODELS_DIR / lang / f"{model_type}.joblib"
                
                if not model_path.exists():
                    print(f"  Model not found: {model_path}")
                    continue
                
                try:
                    model_key = f"{lang}_{model_type}"
                    self.models[model_key] = {
                        "path": model_path,
                        "lang": lang,
                        "type": model_type,
                    }
                    print(f"  Loaded {lang} {model_type} model")
                except Exception as e:
                    print(f"  Error loading {lang} {model_type} model: {e}")
        
        print("\nModel initialization complete!")
    
    def get_model_info(self, lang: str, model_type: str) -> Optional[Dict]:
        """Get model information."""
        model_key = f"{lang}_{model_type}"
        return self.models.get(model_key)
    
    def get_extractors(self, lang: str) -> Optional[Dict]:
        """Get extractors for a language."""
        return self.extractors.get(lang)
    
    def is_initialized(self) -> bool:
        """Check if models are initialized."""
        return self._initialized
    
    def list_available_models(self) -> list:
        """List all available models."""
        return [
            {
                "lang": info["lang"],
                "model": info["type"],
                "key": key
            }
            for key, info in self.models.items()
        ]
