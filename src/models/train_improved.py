"""
Improved model training with better algorithms and hyperparameter tuning.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import joblib

from src.config import (
    MODELS_DIR,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_STATE,
)
from src.utils.io import save_model_bundle
from src.features.lexicon_features import LexiconFeatureExtractor
from src.features.tfidf_features import TFIDFFeatureExtractor
from src.features.tfidf_features_improved import TFIDFFeatureExtractorImproved
from src.features.fusion import FeatureFusion


def train_ml_only_improved(
    texts: List[str],
    labels: List[str],
    tfidf_extractor: TFIDFFeatureExtractor,
    label_to_int: Dict[str, int],
    lang: str,
    model_name: str = "ml",
    use_tuning: bool = True,
) -> Tuple[Path, Dict]:
    """
    Train improved ML-only (TF-IDF) model with better algorithms.
    
    Args:
        texts: Training texts
        labels: Training labels
        tfidf_extractor: TF-IDF feature extractor
        label_to_int: Label to integer mapping
        lang: Language code
        model_name: Model name for saving
        use_tuning: If True, use hyperparameter tuning
        
    Returns:
        Tuple of (model_path, metadata)
    """
    print(f"\nTraining improved {model_name} model for {lang}...")
    
    # Convert labels to integers
    y = np.array([label_to_int[label] for label in labels])
    
    # Extract TF-IDF features
    print("Extracting TF-IDF features...")
    X = tfidf_extractor.fit_transform(texts)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_STATE, stratify=y_train
    )
    
    train_size = X_train.shape[0]
    val_size = X_val.shape[0]
    test_size = X_test.shape[0]
    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Try multiple models and select best
    print("Training and comparing models...")
    
    models_to_try = {}
    
    # 1. Improved Logistic Regression
    if use_tuning:
        lr_param_grid = {
            'C': [0.5, 1.0, 2.0, 5.0],
            'solver': ['lbfgs'],
            'max_iter': [2000, 5000],
        }
        lr_base = LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE)
        lr_search = RandomizedSearchCV(
            lr_base, lr_param_grid, n_iter=8, cv=3, 
            scoring='f1_macro', n_jobs=-1, random_state=RANDOM_STATE, verbose=1
        )
        lr_search.fit(X_train, y_train)
        models_to_try['LogisticRegression'] = lr_search.best_estimator_
        print(f"  LR best params: {lr_search.best_params_}, score: {lr_search.best_score_:.4f}")
    else:
        models_to_try['LogisticRegression'] = LogisticRegression(
            C=2.0, solver='lbfgs', max_iter=5000,
            class_weight='balanced', random_state=RANDOM_STATE
        )
        models_to_try['LogisticRegression'].fit(X_train, y_train)
    
    # 2. Linear SVM
    if use_tuning:
        svm_param_grid = {
            'C': [0.5, 1.0, 2.0],
            'loss': ['squared_hinge'],
        }
        svm_base = LinearSVC(class_weight='balanced', random_state=RANDOM_STATE, max_iter=5000)
        svm_search = RandomizedSearchCV(
            svm_base, svm_param_grid, n_iter=3, cv=3,
            scoring='f1_macro', n_jobs=-1, random_state=RANDOM_STATE, verbose=1
        )
        svm_search.fit(X_train, y_train)
        models_to_try['LinearSVM'] = svm_search.best_estimator_
        print(f"  SVM best params: {svm_search.best_params_}, score: {svm_search.best_score_:.4f}")
    else:
        models_to_try['LinearSVM'] = LinearSVC(
            C=1.0, loss='squared_hinge', class_weight='balanced',
            random_state=RANDOM_STATE, max_iter=5000
        )
        models_to_try['LinearSVM'].fit(X_train, y_train)
    
    # 3. Random Forest (with limited features for speed)
    # Use feature selection or limit features for RF
    if X_train.shape[1] > 10000:
        # For large feature spaces, use a subset or feature selection
        from sklearn.feature_selection import SelectKBest, chi2
        k_best = min(10000, X_train.shape[1])
        selector = SelectKBest(chi2, k=k_best)
        X_train_rf = selector.fit_transform(X_train, y_train)
        X_val_rf = selector.transform(X_val)
        print(f"  Using {k_best} best features for Random Forest")
    else:
        X_train_rf = X_train
        X_val_rf = X_val
        selector = None
    
    if use_tuning:
        rf_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [20, 30],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
        }
        rf_base = RandomForestClassifier(
            class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
        )
        rf_search = RandomizedSearchCV(
            rf_base, rf_param_grid, n_iter=8, cv=3,
            scoring='f1_macro', n_jobs=-1, random_state=RANDOM_STATE, verbose=1
        )
        rf_search.fit(X_train_rf, y_train)
        models_to_try['RandomForest'] = rf_search.best_estimator_
        models_to_try['_rf_selector'] = selector
        print(f"  RF best params: {rf_search.best_params_}, score: {rf_search.best_score_:.4f}")
    else:
        models_to_try['RandomForest'] = RandomForestClassifier(
            n_estimators=200, max_depth=30, min_samples_split=5,
            class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
        )
        models_to_try['RandomForest'].fit(X_train_rf, y_train)
        models_to_try['_rf_selector'] = selector
    
    # Evaluate all models on validation set
    print("\nEvaluating models on validation set...")
    val_scores = {}
    for name, model in models_to_try.items():
        if name.startswith('_'):
            continue
        if name == 'RandomForest':
            y_pred = model.predict(X_val_rf)
        else:
            y_pred = model.predict(X_val)
        score = f1_score(y_val, y_pred, average='macro')
        val_scores[name] = score
        print(f"  {name}: F1-macro = {score:.4f}")
    
    # Select best model
    best_model_name = max(val_scores, key=val_scores.get)
    best_model = models_to_try[best_model_name]
    print(f"\nBest model: {best_model_name} (F1-macro: {val_scores[best_model_name]:.4f})")
    
    # If Random Forest was best, we need to handle feature selection
    rf_selector = models_to_try.get('_rf_selector')
    if best_model_name == 'RandomForest' and rf_selector is not None:
        # Wrap model with selector
        from sklearn.pipeline import Pipeline
        best_model = Pipeline([
            ('selector', rf_selector),
            ('classifier', best_model)
        ])
    
    # Final evaluation on test set
    if best_model_name == 'RandomForest' and rf_selector is not None:
        X_test_final = rf_selector.transform(X_test) if hasattr(rf_selector, 'transform') else X_test
    else:
        X_test_final = X_test
    
    test_score = best_model.score(X_test_final, y_test)
    test_f1 = f1_score(y_test, best_model.predict(X_test_final), average='macro')
    print(f"Test accuracy: {test_score:.4f}, Test F1-macro: {test_f1:.4f}")
    
    # Save model
    model_dir = MODELS_DIR / lang
    model_path = model_dir / f"{model_name}.joblib"
    
    metadata = {
        "model_type": model_name,
        "model_algorithm": best_model_name,
        "lang": lang,
        "n_features": X.shape[1],
        "n_classes": len(label_to_int),
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "val_accuracy": float(best_model.score(X_val_rf if best_model_name == 'RandomForest' and rf_selector is not None else X_val, y_val)),
        "val_f1_macro": float(val_scores[best_model_name]),
        "test_accuracy": float(test_score),
        "test_f1_macro": float(test_f1),
    }
    
    # Save model bundle
    save_model_bundle(
        model_path=model_path,
        vectorizer=tfidf_extractor,
        scaler=None,
        classifier=best_model,
        label_mapping=label_to_int,
        metadata=metadata,
    )
    
    print(f"Model saved to {model_path}")
    
    return model_path, metadata


def train_hybrid_improved(
    texts: List[str],
    labels: List[str],
    fusion: FeatureFusion,
    label_to_int: Dict[str, int],
    lang: str,
    model_name: str = "hybrid",
    use_tuning: bool = True,
) -> Tuple[Path, Dict]:
    """
    Train improved hybrid (fusion) model with better algorithms.
    
    Args:
        texts: Training texts
        labels: Training labels
        fusion: Feature fusion object
        label_to_int: Label to integer mapping
        lang: Language code
        model_name: Model name for saving
        use_tuning: If True, use hyperparameter tuning
        
    Returns:
        Tuple of (model_path, metadata)
    """
    print(f"\nTraining improved {model_name} model for {lang}...")
    
    # Convert labels to integers
    y = np.array([label_to_int[label] for label in labels])
    
    # Extract fused features
    print("Extracting fused features...")
    X = fusion.fit_transform(texts)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_STATE, stratify=y_train
    )
    
    train_size = X_train.shape[0]
    val_size = X_val.shape[0]
    test_size = X_test.shape[0]
    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Convert sparse to dense if needed for some models
    if hasattr(X_train, 'toarray'):
        # For hybrid, features are already scaled, but we might need dense for some models
        X_train_dense = X_train.toarray() if X_train.shape[1] < 50000 else X_train
        X_val_dense = X_val.toarray() if X_val.shape[1] < 50000 else X_val
        X_test_dense = X_test.toarray() if X_test.shape[1] < 50000 else X_test
    else:
        X_train_dense = X_train
        X_val_dense = X_val
        X_test_dense = X_test
    
    # Try multiple models
    print("Training and comparing models...")
    
    models_to_try = {}
    
    # 1. Improved Logistic Regression
    if use_tuning:
        lr_param_grid = {
            'C': [0.5, 1.0, 2.0, 5.0],
            'solver': ['lbfgs'],
            'max_iter': [2000, 5000],
        }
        lr_base = LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE)
        lr_search = RandomizedSearchCV(
            lr_base, lr_param_grid, n_iter=8, cv=3,
            scoring='f1_macro', n_jobs=-1, random_state=RANDOM_STATE, verbose=1
        )
        lr_search.fit(X_train_dense, y_train)
        models_to_try['LogisticRegression'] = lr_search.best_estimator_
        print(f"  LR best params: {lr_search.best_params_}, score: {lr_search.best_score_:.4f}")
    else:
        models_to_try['LogisticRegression'] = LogisticRegression(
            C=2.0, solver='lbfgs', max_iter=5000,
            class_weight='balanced', random_state=RANDOM_STATE
        )
        models_to_try['LogisticRegression'].fit(X_train_dense, y_train)
    
    # 2. Linear SVM
    if use_tuning:
        svm_param_grid = {
            'C': [0.5, 1.0, 2.0],
            'loss': ['squared_hinge'],
        }
        svm_base = LinearSVC(class_weight='balanced', random_state=RANDOM_STATE, max_iter=5000)
        svm_search = RandomizedSearchCV(
            svm_base, svm_param_grid, n_iter=3, cv=3,
            scoring='f1_macro', n_jobs=-1, random_state=RANDOM_STATE, verbose=1
        )
        svm_search.fit(X_train_dense, y_train)
        models_to_try['LinearSVM'] = svm_search.best_estimator_
        print(f"  SVM best params: {svm_search.best_params_}, score: {svm_search.best_score_:.4f}")
    else:
        models_to_try['LinearSVM'] = LinearSVC(
            C=1.0, loss='squared_hinge', class_weight='balanced',
            random_state=RANDOM_STATE, max_iter=5000
        )
        models_to_try['LinearSVM'].fit(X_train_dense, y_train)
    
    # Evaluate models on validation set
    print("\nEvaluating models on validation set...")
    val_scores = {}
    for name, model in models_to_try.items():
        y_pred = model.predict(X_val_dense)
        score = f1_score(y_val, y_pred, average='macro')
        val_scores[name] = score
        print(f"  {name}: F1-macro = {score:.4f}")
    
    # Select best model
    best_model_name = max(val_scores, key=val_scores.get)
    best_model = models_to_try[best_model_name]
    print(f"\nBest model: {best_model_name} (F1-macro: {val_scores[best_model_name]:.4f})")
    
    # Final evaluation on test set
    test_score = best_model.score(X_test_dense, y_test)
    test_f1 = f1_score(y_test, best_model.predict(X_test_dense), average='macro')
    print(f"Test accuracy: {test_score:.4f}, Test F1-macro: {test_f1:.4f}")
    
    # Save model
    model_dir = MODELS_DIR / lang
    model_path = model_dir / f"{model_name}.joblib"
    
    metadata = {
        "model_type": model_name,
        "model_algorithm": best_model_name,
        "lang": lang,
        "n_features": X.shape[1],
        "n_classes": len(label_to_int),
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "val_accuracy": float(best_model.score(X_val_dense, y_val)),
        "val_f1_macro": float(val_scores[best_model_name]),
        "test_accuracy": float(test_score),
        "test_f1_macro": float(test_f1),
    }
    
    # Save fusion components separately
    save_model_bundle(
        model_path=model_path,
        vectorizer=fusion.tfidf_extractor,
        scaler=fusion.scaler,
        classifier=best_model,
        label_mapping=label_to_int,
        metadata=metadata,
    )
    
    print(f"Model saved to {model_path}")
    
    return model_path, metadata
