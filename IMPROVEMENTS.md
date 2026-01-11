# Model Improvements Guide

## Overview

This document describes the improvements made to the ML and Hybrid models to dramatically improve their performance.

## Key Improvements

### 1. **Better Algorithms**
- **Multiple Model Comparison**: Tests Logistic Regression, Linear SVM, and Random Forest
- **Automatic Best Model Selection**: Chooses the best performing model based on validation F1-macro score
- **Ensemble Potential**: Framework ready for ensemble methods

### 2. **Enhanced TF-IDF Features**
- **Extended N-grams**: Word n-grams extended to (1,3) trigrams, char n-grams to (3,6)
- **Better Filtering**: `min_df=2` removes rare terms, `max_df=0.95` removes very common words
- **Sublinear TF Scaling**: Uses `1 + log(tf)` instead of raw term frequency
- **L2 Normalization**: Proper normalization for better feature scaling

### 3. **Hyperparameter Tuning**
- **RandomizedSearchCV**: Efficient hyperparameter search
- **Cross-Validation**: Uses 3-fold CV for robust evaluation
- **F1-Macro Scoring**: Optimizes for balanced performance across all classes

### 4. **Better Feature Engineering**
- **Feature Selection for Random Forest**: Uses SelectKBest for high-dimensional spaces
- **Improved Scaling**: Better handling of sparse/dense feature combinations

## Usage

### Train Improved Models

```bash
python scripts/train_improved.py
```

You'll be prompted:
- **With tuning (y)**: Slower but finds optimal hyperparameters (~2-4 hours)
- **Without tuning (n)**: Faster with good default parameters (~30-60 minutes)

### Compare Results

After training, evaluate:
```bash
python scripts/evaluate_all.py
python scripts/generate_report.py
```

Compare the new metrics with previous results in `reports/evaluation_report.md`.

## Expected Improvements

Based on the improvements:

### English Models
- **ML Model**: Expected to improve from ~67% to **72-75%** accuracy
- **Hybrid Model**: Expected to improve from ~68% to **73-76%** accuracy

### Romanian Models  
- **ML Model**: Already good at ~85%, may improve to **87-89%**
- **Hybrid Model**: Should improve from ~81% to **85-87%** (currently worse than ML!)

## Technical Details

### Model Algorithms Tested

1. **Logistic Regression**
   - Tuned: C ∈ [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
   - Solvers: 'lbfgs', 'saga'
   - Max iterations: [1000, 2000, 5000]

2. **Linear SVM**
   - Tuned: C ∈ [0.1, 0.5, 1.0, 2.0, 5.0]
   - Loss functions: 'hinge', 'squared_hinge'

3. **Random Forest**
   - Tuned: n_estimators ∈ [100, 200, 300]
   - Max depth: [20, 30, None]
   - Min samples: [2, 5, 10] split, [1, 2, 4] leaf

### Feature Improvements

**Before:**
- Word n-grams: (1, 2)
- Char n-grams: (3, 5)
- No min_df/max_df filtering
- Linear TF scaling

**After:**
- Word n-grams: (1, 3) - captures more context
- Char n-grams: (3, 6) - better subword patterns
- min_df=2, max_df=0.95 - removes noise
- Sublinear TF - reduces impact of very frequent terms
- L2 normalization - better feature scaling

## Files Changed

- `src/models/train_improved.py` - New improved training functions
- `src/features/tfidf_features_improved.py` - Enhanced TF-IDF extractor
- `scripts/train_improved.py` - Training script for improved models

## Notes

- The improved models are saved with the same names (`ml.joblib`, `hybrid.joblib`)
- **Backup your old models** if you want to compare:
  ```bash
  cp -r models models_backup
  ```
- The API will automatically use the new models once retrained
- Lexicon model remains unchanged (it's already optimal for its feature set)

## Troubleshooting

**Out of Memory**: If Random Forest causes memory issues, it will automatically use feature selection.

**Slow Training**: Hyperparameter tuning is slow. Use `n` for faster training with good defaults.

**Model Not Better**: Check that:
1. You're comparing on the same test set
2. The models were trained on the same data
3. Check validation scores during training
