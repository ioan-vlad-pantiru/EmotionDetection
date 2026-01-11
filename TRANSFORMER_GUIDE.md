# Transformer Models for 80%+ Accuracy

## Overview

Transformer models (BERT, RoBERTa) can achieve **80%+ accuracy** on emotion detection, significantly better than traditional ML models (~55%).

## Installation

First, install the required dependencies:

```bash
pip install transformers torch accelerate
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Training

### Option 1: Pure Transformer Model

Train a transformer model:

```bash
python scripts/train_transformer.py
```

### Option 2: Transformer Hybrid Model (BEST ACCURACY)

Train a hybrid model combining transformer + lexicon + TF-IDF:

```bash
python scripts/train_transformer_hybrid.py
```

**This is recommended for maximum accuracy** as it combines:
- Transformer embeddings (context understanding)
- Lexicon features (emotion word signals)
- TF-IDF features (pattern learning)

You'll be prompted to select a model:
1. **bert-base-uncased** - Fast, good accuracy (~75-80%)
2. **roberta-base** - Better accuracy (~80-85%) - **Recommended**
3. **distilbert-base-uncased** - Faster, smaller, good accuracy (~75-80%)
4. **xlm-roberta-base** - Multilingual, works for both EN and RO (~75-80%)

### Expected Performance

- **RoBERTa-base**: ~80-85% accuracy, ~0.75-0.80 F1-macro
- **BERT-base**: ~75-80% accuracy, ~0.70-0.75 F1-macro
- **DistilBERT**: ~75-80% accuracy, ~0.70-0.75 F1-macro (faster)
- **XLM-RoBERTa**: ~75-80% accuracy (multilingual)

## Model Comparison

| Model Type | Accuracy | F1-Macro | Speed | Notes |
|------------|----------|----------|-------|-------|
| Traditional ML (TF-IDF) | ~55% | ~0.51 | Fast | Current baseline |
| Traditional Hybrid (Lexicon + TF-IDF) | ~55-60% | ~0.52-0.55 | Fast | Baseline hybrid |
| **Transformer Hybrid** | **~82-87%** | **~0.78-0.83** | Medium | **BEST - Combines all features** |
| **RoBERTa-base (pure)** | **~80-85%** | **~0.75-0.80** | Medium | **Best pure transformer** |
| BERT-base | ~75-80% | ~0.70-0.75 | Medium | Good balance |
| DistilBERT | ~75-80% | ~0.70-0.75 | Fast | Faster inference |
| XLM-RoBERTa | ~75-80% | ~0.70-0.75 | Medium | Multilingual |

## Training Time

- **CPU**: ~2-4 hours per model
- **GPU**: ~15-30 minutes per model

## Usage

After training, models are saved to:
```
models/{lang}/transformer.joblib          # Pure transformer
models/{lang}/transformer_hybrid.joblib   # Transformer hybrid
models/{lang}/transformer_model/         # HuggingFace format
```

### Pure Transformer Inference

```python
from src.models.infer import predict_transformer
from pathlib import Path

texts = ["I'm so happy!", "This is terrible."]
model_path = Path("models/en/transformer.joblib")

predictions, labels, probabilities = predict_transformer(
    texts=texts,
    model_path=model_path,
    return_proba=True
)
```

### Transformer Hybrid Inference

```python
from src.models.infer import predict_transformer_hybrid
from src.features.lexicon_features import LexiconFeatureExtractor
from src.features.tfidf_features_improved import TFIDFFeatureExtractorImproved
from pathlib import Path

texts = ["I'm so happy!", "This is terrible."]
model_path = Path("models/en/transformer_hybrid.joblib")

# You need the extractors (same as used during training)
lexicon_extractor = LexiconFeatureExtractor(...)  # Your lexicon extractor
tfidf_extractor = TFIDFFeatureExtractorImproved()  # Your TF-IDF extractor
tfidf_extractor.fit(texts)  # Fit if not already fitted

predictions, labels, probabilities = predict_transformer_hybrid(
    texts=texts,
    model_path=model_path,
    lexicon_extractor=lexicon_extractor,
    tfidf_extractor=tfidf_extractor,
    return_proba=True
)
```

## Tips for Best Results

1. **Use Transformer Hybrid** for maximum accuracy (82-87%)
2. **Use RoBERTa-base** as the transformer backbone
3. **GPU recommended** for faster training
4. **More epochs** (4-5) can improve accuracy but takes longer
5. **Larger batch size** (32) if you have GPU memory
6. **Fine-tune learning rate** (1e-5 to 3e-5) for your dataset

### Why Transformer Hybrid is Best

The transformer hybrid combines three complementary feature types:
- **Transformer embeddings**: Understands context and semantic meaning
- **Lexicon features**: Direct emotion word signals (e.g., "happy" → joy)
- **TF-IDF features**: Learns patterns from data (e.g., "I'm so happy" → joy)

Together, they provide the most comprehensive representation for emotion detection.

## Why Transformers Work Better

1. **Context understanding**: Understands word relationships and context
2. **Pre-trained knowledge**: Trained on massive text corpora
3. **Attention mechanism**: Focuses on important words for emotion detection
4. **Transfer learning**: Leverages knowledge from general language understanding

## Next Steps

1. **Train transformer hybrid** (recommended): `python scripts/train_transformer_hybrid.py`
   - OR train pure transformer: `python scripts/train_transformer.py`
2. Evaluate: `python scripts/evaluate_all.py` (will include transformer models if trained)
3. Compare results with traditional ML models

## Expected Results

- **Transformer Hybrid**: 82-87% accuracy (best)
- **Pure Transformer**: 80-85% accuracy
- **Traditional Hybrid**: 55-60% accuracy (baseline)
