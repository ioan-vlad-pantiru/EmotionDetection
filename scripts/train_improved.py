"""
Train improved models for all languages.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import LANGUAGES, MODEL_TYPES
from src.datasets.goemotions import load_goemotions_simple
from src.datasets.red_ro import load_red
from src.lexicons.roemolex import RoEmoLex
from src.lexicons.emolex_en import EmoLexEN
from src.features.lexicon_features import LexiconFeatureExtractor
from src.features.tfidf_features_improved import TFIDFFeatureExtractorImproved
from src.features.fusion import FeatureFusion
from src.models.train_improved import train_ml_only_improved, train_hybrid_improved
from src.models.train import train_lexicon_only


def main():
    print("=" * 60)
    print("Training IMPROVED models")
    print("=" * 60)
    
    # Ask user if they want hyperparameter tuning (slower but better)
    use_tuning = input("\nUse hyperparameter tuning? (y/n, default=n): ").lower().strip() == 'y'
    if use_tuning:
        print("Using hyperparameter tuning (this will take longer)...")
    else:
        print("Using default improved parameters (faster)...")
    
    for lang in LANGUAGES:
        print(f"\n{'='*60}")
        print(f"Language: {lang.upper()}")
        print(f"{'='*60}")
        
        # Load dataset
        print(f"\nLoading {lang} dataset...")
        try:
            if lang == "en":
                texts, labels, label_to_int = load_goemotions_simple()
            else:
                texts, labels, label_to_int = load_red()
            
            print(f"Loaded {len(texts)} examples with {len(label_to_int)} labels")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            continue
        
        # Load lexicon (optional - needed only for lexicon and hybrid models)
        lexicon = None
        lexicon_extractor = None
        fusion = None
        
        print(f"\nLoading {lang} lexicon...")
        try:
            if lang == "en":
                lexicon = EmoLexEN()
            else:
                lexicon = RoEmoLex()
            
            print(f"Lexicon vocabulary size: {lexicon.get_vocabulary_size()}")
            
            # Create lexicon-based extractors only if lexicon loaded successfully
            lexicon_extractor = LexiconFeatureExtractor(lexicon, lang=lang)
            # Use improved TF-IDF extractor for fusion
            fusion = FeatureFusion(
                lexicon_extractor,
                TFIDFFeatureExtractorImproved()
            )
        except Exception as e:
            print(f"⚠ Warning: Lexicon not available: {e}")
            print("  Will train ML-only model (TF-IDF). Lexicon and hybrid models will be skipped.")
        
        # Create improved TF-IDF extractor (always available)
        tfidf_extractor = TFIDFFeatureExtractorImproved()
        
        # Train models
        for model_type in MODEL_TYPES:
            try:
                print(f"\n{'─'*60}")
                print(f"Training {model_type} model...")
                print(f"{'─'*60}")
                
                if model_type == "lexicon":
                    if lexicon_extractor is None:
                        print(f"  ⚠ Skipping {model_type} model: lexicon not available")
                        continue
                    # Use original lexicon training (it's already good)
                    train_lexicon_only(texts, labels, lexicon_extractor, label_to_int, lang)
                elif model_type == "ml":
                    train_ml_only_improved(
                        texts, labels, tfidf_extractor, label_to_int, lang,
                        use_tuning=use_tuning
                    )
                elif model_type == "hybrid":
                    if fusion is None:
                        print(f"  ⚠ Skipping {model_type} model: lexicon not available")
                        continue
                    train_hybrid_improved(
                        texts, labels, fusion, label_to_int, lang,
                        use_tuning=use_tuning
                    )
                
            except Exception as e:
                print(f"Error training {model_type} model: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run: python scripts/evaluate_all.py")
    print("2. Compare results with previous models")


if __name__ == "__main__":
    main()
