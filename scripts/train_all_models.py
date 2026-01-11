"""
Train all models: lexicon, ML, hybrid, transformer, and transformer hybrid.
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
from src.models.train_transformer import train_transformer
from src.models.train_transformer_hybrid import train_transformer_hybrid


def main():
    print("=" * 60)
    print("Training ALL models")
    print("=" * 60)
    
    # Ask user which models to train
    print("\nWhich models would you like to train?")
    print("1. Best models only (transformer, transformer hybrid) - RECOMMENDED")
    print("   Expected accuracy: 80-87%")
    print("2. Traditional ML models (lexicon, ml, hybrid)")
    print("   Expected accuracy: 55-60%")
    print("3. Transformer models only (pure transformer, transformer hybrid)")
    print("   Expected accuracy: 80-87%")
    print("4. All models")
    print("   Expected accuracy: 55-87% (varies by model)")
    
    choice = input("\nSelect option (1-4, default=1): ").strip() or "1"
    
    train_traditional = choice in ["2", "4"]
    train_transformer_models = choice in ["1", "3", "4"]
    
    # Hyperparameter tuning for traditional models
    use_tuning = True
    if train_traditional:
        use_tuning_input = input("\nUse hyperparameter tuning for traditional models? (y/n, default=y): ").lower().strip()
        use_tuning = use_tuning_input != 'n'
        if use_tuning:
            print("Using hyperparameter tuning (this will take longer)...")
        else:
            print("Using default improved parameters (faster)...")
    else:
        print("\nSkipping traditional ML models (focusing on best models only)")
    
    # Transformer model selection
    transformer_model = None
    if train_transformer_models:
        print("\nAvailable transformer models:")
        print("1. bert-base-uncased (English, fast, good accuracy)")
        print("2. roberta-base (English, better accuracy) - RECOMMENDED")
        print("3. distilbert-base-uncased (English, faster, smaller)")
        print("4. xlm-roberta-base (Multilingual, works for both EN and RO)")
        
        model_choice = input("\nSelect transformer model (1-4, default=2): ").strip() or "2"
        
        model_map = {
            "1": "bert-base-uncased",
            "2": "roberta-base",
            "3": "distilbert-base-uncased",
            "4": "xlm-roberta-base",
        }
        
        transformer_model = model_map.get(model_choice, "roberta-base")
        
        # For Romanian, use multilingual model
        multilingual_models = ["xlm-roberta-base"]
        if transformer_model not in multilingual_models:
            print(f"\n⚠ Warning: {transformer_model} is English-only.")
            print("For Romanian, consider using 'xlm-roberta-base' (option 4)")
            use_for_ro = input("Use this model for Romanian anyway? (y/n, default=n): ").lower().strip() == 'y'
        else:
            use_for_ro = True
    else:
        use_for_ro = True
    
    for lang in LANGUAGES:
        # Skip Romanian if using English-only transformer model
        if train_transformer_models and lang == "ro" and transformer_model not in multilingual_models and not use_for_ro:
            print(f"\n{'='*60}")
            print(f"Language: {lang.upper()} - Skipping (English-only transformer model)")
            print(f"{'='*60}")
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
        
        # Load lexicon (needed for lexicon, hybrid, and transformer hybrid)
        lexicon = None
        lexicon_extractor = None
        fusion = None
        
        if train_traditional or train_transformer_models:
            print(f"\nLoading {lang} lexicon...")
            try:
                if lang == "en":
                    lexicon = EmoLexEN()
                else:
                    lexicon = RoEmoLex()
                
                print(f"Lexicon vocabulary size: {lexicon.get_vocabulary_size()}")
                
                lexicon_extractor = LexiconFeatureExtractor(lexicon, lang=lang)
                
                # Create fusion for traditional hybrid
                if train_traditional:
                    fusion = FeatureFusion(
                        lexicon_extractor,
                        TFIDFFeatureExtractorImproved()
                    )
            except Exception as e:
                print(f"⚠ Warning: Lexicon not available: {e}")
                if train_traditional:
                    print("  Will train ML-only model (TF-IDF). Lexicon and hybrid models will be skipped.")
                if train_transformer_models:
                    print("  Transformer hybrid requires lexicon. Will train pure transformer only.")
        
        # Create TF-IDF extractor
        tfidf_extractor = TFIDFFeatureExtractorImproved()
        
        # Train traditional ML models
        if train_traditional:
            for model_type in MODEL_TYPES:
                try:
                    print(f"\n{'─'*60}")
                    print(f"Training {model_type} model...")
                    print(f"{'─'*60}")
                    
                    if model_type == "lexicon":
                        if lexicon_extractor is None:
                            print(f"  ⚠ Skipping {model_type} model: lexicon not available")
                            continue
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
        
        # Train transformer models
        if train_transformer_models:
            # 1. Pure transformer model
            try:
                print(f"\n{'─'*60}")
                print(f"Training transformer model...")
                print(f"{'─'*60}")
                
                train_transformer(
                    texts=texts,
                    labels=labels,
                    label_to_int=label_to_int,
                    lang=lang,
                    model_name="transformer",
                    model_type=transformer_model if lang == "en" or transformer_model in multilingual_models else "xlm-roberta-base",
                    use_tuning=True,
                    max_length=128,
                    batch_size=16,
                    learning_rate=2e-5,
                    num_epochs=3,
                )
            except Exception as e:
                print(f"Error training transformer model: {e}")
                import traceback
                traceback.print_exc()
            
            # 2. Transformer hybrid model
            if lexicon_extractor is not None:
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
                        transformer_model=transformer_model if lang == "en" or transformer_model in multilingual_models else "xlm-roberta-base",
                        use_tuning=True,
                        max_length=128,
                        batch_size=8,  # Optimized for 24GB RAM
                    )
                except Exception as e:
                    print(f"Error training transformer hybrid model: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"\n{'─'*60}")
                print(f"Skipping transformer hybrid: lexicon not available")
                print(f"{'─'*60}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run: python scripts/evaluate_all.py")
    print("2. Compare results across all models")


if __name__ == "__main__":
    main()
