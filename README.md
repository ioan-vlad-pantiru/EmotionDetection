# Bilingual Emotion Detection System

A practical emotion detection system for English and Romanian using lexicon features and TF-IDF with feature fusion. This implementation follows a simple, reproducible approach using scikit-learn without deep learning.

## Overview

This system implements a **feature fusion pipeline** that combines:
1. **Lexicon features**: Emotion scores and counts from emotion lexicons (RoEmoLex for Romanian, EmoLex for English)
2. **TF-IDF features**: Word and character n-grams
3. **Hybrid model**: Concatenated and scaled features fed to a linear classifier

### Architecture

```
Text → Preprocessing → Feature Extraction → Feature Fusion → Linear Classifier → Emotion Label
                        ├─ Lexicon Features
                        └─ TF-IDF Features
```

**📖 For a detailed intuitive explanation of how everything works end-to-end, see [EXPLANATION.md](EXPLANATION.md)**

## Datasets

### English: GoEmotions
- **Source**: [GoEmotions Dataset](https://github.com/google-research/google-research/tree/master/goemotions)
- **Size**: ~58,000 Reddit comments
- **Original Labels**: 27 emotions + neutral
- **Mapping**: Mapped to Plutchik's 8 emotions + neutral (see Label Mapping section)

### Romanian: RED / REDv2
- **Source**: [REDv2 on HuggingFace](https://huggingface.co/datasets/Alegzandra/REDv2) | [GitHub Repository](https://github.com/Alegzandra/RED-Romanian-Emotion-Datasets)
- **REDv1**: 4,047 tweets, 5 emotions (single-label)
- **REDv2**: 5,449 tweets, 7 emotions (multi-label): anger, fear, joy, sadness, surprise, trust, neutral
- **Format**: REDv2 is multi-label (we use single-label by taking first emotion from boolean fields or agreed_labels vector)
- **Loading**: Automatically loads from HuggingFace (`Alegzandra/REDv2`)

## Lexicons

### RoEmoLex (Romanian) - **Mandatory**
- **Reference**: [RoEmoLex - A Romanian Emotion Lexicon](https://www.cs.ubbcluj.ro/~studia-i/journal/journal/article/view/13)
- **Format**: CSV/TSV with columns: word, emotion, score
- **Placement**: `data/raw/roemolex.csv` or `data/raw/roemolex.tsv`
- **Emotions**: Maps to Plutchik's 8 emotions

### EmoLex (English) - Optional
- **Reference**: [NRC Emotion Lexicon](http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)
- **Format**: Tab-separated file
- **Placement**: `data/raw/emolex_en.txt`
- **Fallback**: Minimal built-in lexicon if not found

## Label Mapping

### English (GoEmotions → Plutchik 8)

| GoEmotions Label | Plutchik Emotion |
|-----------------|------------------|
| anger | anger |
| fear | fear |
| joy | joy |
| sadness | sadness |
| disgust | disgust |
| surprise | surprise |
| curiosity, desire, optimism | anticipation |
| approval, admiration, gratitude, caring | trust |
| neutral | neutral |
| amusement, excitement, love, relief, pride | joy |
| nervousness, embarrassment | fear |
| disappointment, grief, remorse | sadness |
| annoyance, disapproval | anger |
| confusion, realization | surprise |

### Romanian (RED → Plutchik 8)

| RED Label | Plutchik Emotion |
|-----------|------------------|
| anger / furie | anger |
| fear / frică | fear |
| joy / bucurie | joy |
| sadness / tristețe | sadness |
| surprise / surpriză | surprise |
| trust / încredere | trust |
| disgust / dezgust | disgust |
| anticipation / anticipare | anticipation |
| neutral / neutru | neutral |

**Note**: Romanian dataset may not contain all 8 emotions. The system trains on available labels and computes lexicon features for all 8 emotions.

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Or install via pip:

```bash
pip install scikit-learn pandas numpy datasets emoji joblib scipy
```

**Note**: `tensorflow-datasets` is optional. The project uses HuggingFace `datasets` library by default, which is more reliable across Python versions. If you need TensorFlow Datasets support, install it separately (may require Python < 3.14 due to `dm-tree` build issues).

## Usage

### 1. Download Data

Download all required datasets and lexicons:

```bash
python scripts/download_data.py
```

This will:
- Download GoEmotions dataset (English)
- Download RED dataset (Romanian)
- Check for RoEmoLex lexicon (you may need to download manually)
- Check for EmoLex lexicon (optional, fallback available)

**Manual Downloads**:
- **RoEmoLex**: Download from [the paper page](https://www.cs.ubbcluj.ro/~studia-i/journal/journal/article/view/13) and place in `data/raw/roemolex.csv` (required for Romanian lexicon features)
- **EmoLex**: Already included if you downloaded NRC EmoLex (optional, fallback available)

**Note**: REDv2 dataset is automatically downloaded from HuggingFace, so no manual download needed!

### 2. Train Models

Train all models for all languages:

```bash
python scripts/train_all.py
```

This trains three models per language:
- **lexicon**: Lexicon-only features
- **ml**: TF-IDF-only features
- **hybrid**: Fused features

### 3. Evaluate Models

Evaluate all models and generate reports:

```bash
python scripts/evaluate_all.py
```

This generates:
- Metrics JSON files in `reports/metrics/`
- Error analysis JSON files in `reports/errors/`
- Comparison table CSV in `reports/metrics/comparison_table.csv`

### 4. Generate Visualizations and Report

Generate comprehensive charts and summary report:

```bash
python scripts/generate_report.py
```

This creates:
- **Charts** in `reports/charts/`:
  - `comparison_overall.png`: Overall metrics comparison
  - `comparison_en.png` / `comparison_ro.png`: Per-language model comparisons
  - `per_class_en.png` / `per_class_ro.png`: Per-class performance charts
  - `confusion_matrix_en.png` / `confusion_matrix_ro.png`: Confusion matrix heatmaps
- **Summary Report**: `reports/evaluation_report.md` with detailed metrics and tables

### 5. Run Individual Experiments

Train and evaluate a specific model:

```bash
python src/experiments/run_experiment.py --lang en --model hybrid
python src/experiments/run_experiment.py --lang ro --model lexicon
```

Options:
- `--lang`: `en` or `ro`
- `--model`: `lexicon`, `ml`, or `hybrid`

## Repository Structure

```
emotion-hybrid/
├── README.md
├── pyproject.toml
├── requirements.txt
├── data/
│   ├── raw/           # Downloaded datasets + lexicons
│   └── processed/     # Cached processed splits
├── models/            # Saved trained models
│   ├── en/
│   └── ro/
├── reports/
│   ├── metrics/       # JSON/CSV metrics
│   └── errors/        # Error analysis samples
├── src/
│   ├── config.py
│   ├── utils/
│   │   ├── io.py
│   │   ├── text.py
│   │   └── metrics.py
│   ├── lexicons/
│   │   ├── roemolex.py
│   │   └── emolex_en.py
│   ├── datasets/
│   │   ├── goemotions.py
│   │   ├── red_ro.py
│   │   └── mapping.py
│   ├── features/
│   │   ├── lexicon_features.py
│   │   ├── tfidf_features.py
│   │   └── fusion.py
│   ├── models/
│   │   ├── train.py
│   │   └── infer.py
│   └── experiments/
│       └── run_experiment.py
└── scripts/
    ├── download_data.py
    ├── train_all.py
    └── evaluate_all.py
```

## Feature Engineering

### Lexicon Features (21 features)
- 8 emotion counts (number of words with each emotion)
- 8 emotion intensity sums (total emotion scores)
- 5 stylistic features:
  - Exclamation count
  - Question mark count
  - Repeated exclamation patterns
  - Repeated question patterns
  - Uppercase token ratio

### TF-IDF Features (up to 50,000 features)
- Word n-grams: (1, 2)
- Character n-grams: (3, 5)
- Maximum features: 50,000

### Feature Fusion
- Lexicon features: Scaled with `StandardScaler(with_mean=False)`
- TF-IDF features: Used as-is (already normalized)
- Concatenation: `hstack([tfidf_sparse, lexicon_dense_scaled])`

## Models

All models use **LogisticRegression** with:
- Solver: `saga`
- Max iterations: 2000
- Class weights: `balanced`
- Train/Val/Test split: 70%/10%/20% (stratified)

### Model Types

1. **Lexicon-only**: Trained on lexicon features only
2. **ML-only**: Trained on TF-IDF features only
3. **Hybrid**: Trained on fused features

## Preprocessing

Both languages undergo:
1. **Lowercasing**
2. **URL removal**
3. **Mention removal** (keep hashtag text: `#happy` → `happy`)
4. **Elongated character normalization** (`soooo` → `soo`)
5. **Emoji-to-text conversion** (using `emoji` package)
6. **Tokenization** (regex-based)
7. **Negation handling**: Downweight lexicon emotion scores in negation window (3-4 tokens after negation cues)

### Negation Cues
- **English**: `not`, `never`, `no`, `n't`, `cannot`, `can't`, `won't`, `don't`, `didn't`
- **Romanian**: `nu`, `niciodată`, `fără`, `nici`, `niciun`, `niciuna`

## Evaluation Metrics

- **Accuracy**
- **Macro F1**
- **Weighted F1**
- **Per-class**: Precision, Recall, F1, Support
- **Confusion Matrix**

## Error Analysis

Error analysis categorizes misclassifications by:
- **Negation present**: Text contains negation cues
- **Emoji present**: Text contains emojis
- **Very short text**: Less than 5 words
- **Sarcasm markers**: Contains phrases like "yeah right", "sure", "lol", etc.

Top 50 errors per model are saved with:
- Original text
- Gold label
- Predicted label
- Top TF-IDF features
- Lexicon emotion scores

## Expected Outputs

After running `evaluate_all.py` and `generate_report.py`, you should see:

```
reports/
├── metrics/
│   ├── en_lexicon_metrics.json
│   ├── en_ml_metrics.json
│   ├── en_hybrid_metrics.json
│   ├── ro_lexicon_metrics.json
│   ├── ro_ml_metrics.json
│   ├── ro_hybrid_metrics.json
│   └── comparison_table.csv
├── errors/
│   ├── en_lexicon_errors.json
│   ├── en_ml_errors.json
│   ├── en_hybrid_errors.json
│   ├── ro_lexicon_errors.json
│   ├── ro_ml_errors.json
│   └── ro_hybrid_errors.json
├── charts/
│   ├── comparison_overall.png
│   ├── comparison_en.png
│   ├── comparison_ro.png
│   ├── per_class_en.png
│   ├── per_class_ro.png
│   ├── confusion_matrix_en.png
│   └── confusion_matrix_ro.png
└── evaluation_report.md
```

## Reproduction Steps

1. **Setup**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download data**:
   ```bash
   python scripts/download_data.py
   ```
   (Manually download RoEmoLex if needed)

3. **Train models**:
   ```bash
   python scripts/train_all.py
   ```

4. **Evaluate**:
   ```bash
   python scripts/evaluate_all.py
   ```

5. **Check results**:
   - View metrics in `reports/metrics/`
   - Analyze errors in `reports/errors/`

## Design Decisions

1. **Single-label classification**: Multi-label datasets are converted to single-label by taking the first/major label after mapping
2. **Plutchik 8 emotions**: Standard emotion taxonomy for consistency
3. **Simple models**: LogisticRegression instead of deep learning for reproducibility and speed
4. **Feature scaling**: Lexicon features scaled, TF-IDF left as-is
5. **Negation handling**: Simple downweighting (0.3x) in negation window

## Citations

If you use this code, please cite:

- **GoEmotions**: Demszky, D., et al. (2020). "GoEmotions: A Dataset of Fine-Grained Emotions." ACL.
- **RED**: Dumitrescu, S. D., et al. (2020). "RED: Romanian Emotion Dataset." LREC.
- **RoEmoLex**: Banea, C., et al. "RoEmoLex - A Romanian Emotion Lexicon." Studia Universitatis Babeș-Bolyai Informatica.

## License

This project is provided as-is for research and educational purposes.


