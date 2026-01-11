"""
Run individual experiment: train and evaluate a specific model.
"""
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import LANGUAGES, MODEL_TYPES
from src.datasets.goemotions import load_goemotions_simple
from src.datasets.red_ro import load_red
from src.lexicons.roemolex import RoEmoLex
from src.lexicons.emolex_en import EmoLexEN
from src.features.lexicon_features import LexiconFeatureExtractor
from src.features.tfidf_features import TFIDFFeatureExtractor
from src.features.fusion import FeatureFusion
from src.models.train import train_lexicon_only, train_ml_only, train_hybrid


def main():
    parser = argparse.ArgumentParser(description="Run emotion detection experiment")
    parser.add_argument("--lang", choices=LANGUAGES, required=True, help="Language code")
    parser.add_argument("--model", choices=MODEL_TYPES, required=True, help="Model type")
    
    args = parser.parse_args()
    
    lang = args.lang
    model_type = args.model
    
    print(f"Running experiment: {model_type} model for {lang}")
    print("=" * 60)
    
    # Load dataset
    print(f"\nLoading {lang} dataset...")
    if lang == "en":
        texts, labels, label_to_int = load_goemotions_simple()
    else:
        texts, labels, label_to_int = load_red()
    
    print(f"Loaded {len(texts)} examples")
    
    # Load lexicon
    print(f"\nLoading {lang} lexicon...")
    if lang == "en":
        try:
            lexicon = EmoLexEN()
        except Exception as e:
            print(f"Warning: {e}")
            lexicon = EmoLexEN()  # Will use fallback
    else:
        lexicon = RoEmoLex()
    
    print(f"Lexicon vocabulary size: {lexicon.get_vocabulary_size()}")
    
    # Create feature extractors
    lexicon_extractor = LexiconFeatureExtractor(lexicon, lang=lang)
    tfidf_extractor = TFIDFFeatureExtractor()
    fusion = FeatureFusion(lexicon_extractor, tfidf_extractor)
    
    # Train model
    if model_type == "lexicon":
        train_lexicon_only(texts, labels, lexicon_extractor, label_to_int, lang)
    elif model_type == "ml":
        train_ml_only(texts, labels, tfidf_extractor, label_to_int, lang)
    elif model_type == "hybrid":
        train_hybrid(texts, labels, fusion, label_to_int, lang)
    
    print("\nExperiment completed!")


if __name__ == "__main__":
    main()


