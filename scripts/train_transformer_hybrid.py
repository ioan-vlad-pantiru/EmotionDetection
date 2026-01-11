"""
Train transformer hybrid models (transformer + lexicon + TF-IDF) for all languages.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import LANGUAGES
from src.datasets.goemotions import load_goemotions_simple
from src.datasets.red_ro import load_red
from src.lexicons.roemolex import RoEmoLex
from src.lexicons.emolex_en import EmoLexEN
from src.features.lexicon_features import LexiconFeatureExtractor
from src.features.tfidf_features_improved import TFIDFFeatureExtractorImproved
from src.models.train_transformer_hybrid import train_transformer_hybrid


def main():
    print("=" * 60)
    print("Training TRANSFORMER HYBRID models")
    print("=" * 60)
    print("\nThis combines:")
    print("  - Transformer embeddings (BERT/RoBERTa)")
    print("  - Lexicon features")
    print("  - TF-IDF features")
    print("\nExpected accuracy: 80-85%+")
    
    # Model options
    print("\nAvailable transformer models:")
    print("1. bert-base-uncased (English, fast, good accuracy)")
    print("2. roberta-base (English, better accuracy) - RECOMMENDED")
    print("3. distilbert-base-uncased (English, faster, smaller)")
    print("4. xlm-roberta-base (Multilingual, works for both EN and RO)")
    
    model_choice = input("\nSelect model (1-4, default=2): ").strip() or "2"
    
    model_map = {
        "1": "bert-base-uncased",
        "2": "roberta-base",
        "3": "distilbert-base-uncased",
        "4": "xlm-roberta-base",
    }
    
    model_type = model_map.get(model_choice, "roberta-base")
    
    # For Romanian, use multilingual model
    multilingual_models = ["xlm-roberta-base"]
    if model_type not in multilingual_models:
        print(f"\n⚠ Warning: {model_type} is English-only.")
        print("For Romanian, consider using 'xlm-roberta-base' (option 4)")
        use_for_ro = input("Use this model for Romanian anyway? (y/n, default=n): ").lower().strip() == 'y'
        if not use_for_ro and "ro" in LANGUAGES:
            print("Skipping Romanian with this model.")
    else:
        use_for_ro = True
    
    for lang in LANGUAGES:
        # Skip Romanian if using English-only model
        if lang == "ro" and model_type not in multilingual_models and not use_for_ro:
            continue
            
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
        
        # Load lexicon
        lexicon = None
        lexicon_extractor = None
        
        print(f"\nLoading {lang} lexicon...")
        try:
            if lang == "en":
                lexicon = EmoLexEN()
            else:
                lexicon = RoEmoLex()
            
            print(f"Lexicon vocabulary size: {lexicon.get_vocabulary_size()}")
            lexicon_extractor = LexiconFeatureExtractor(lexicon, lang=lang)
        except Exception as e:
            print(f"⚠ Warning: Lexicon not available: {e}")
            print("  Transformer hybrid requires lexicon. Skipping...")
            continue
        
        # Create TF-IDF extractor
        tfidf_extractor = TFIDFFeatureExtractorImproved()
        
        # Train transformer hybrid model
        try:
            print(f"\n{'─'*60}")
            print(f"Training transformer hybrid model...")
            print(f"{'─'*60}")
            
            train_transformer_hybrid(
                texts=texts,
                labels=labels,
                lexicon_extractor=lexicon_extractor,
                tfidf_extractor=tfidf_extractor,
                label_to_int=label_to_int,
                lang=lang,
                model_name="transformer_hybrid",
                transformer_model=model_type if lang == "en" or model_type in multilingual_models else "xlm-roberta-base",
                use_tuning=True,
                max_length=128,
                batch_size=32,
            )
            
        except Exception as e:
            print(f"Error training transformer hybrid model: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run: python scripts/evaluate_all.py")
    print("2. Compare transformer hybrid results with other models")


if __name__ == "__main__":
    main()
