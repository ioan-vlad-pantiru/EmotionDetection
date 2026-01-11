"""
Transformer-based model training for emotion detection.
Uses BERT/RoBERTa models to achieve 80%+ accuracy.
"""
import os
# Disable TensorFlow to avoid Keras 3 compatibility issues
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["SKIP_TF_INSTALL"] = "1"

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
import joblib

from src.config import (
    MODELS_DIR,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_STATE,
)


class EmotionDataset(Dataset):
    """Dataset class for emotion detection."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def train_transformer(
    texts: List[str],
    labels: List[str],
    label_to_int: Dict[str, int],
    lang: str,
    model_name: str = "transformer",
    model_type: str = "bert-base-uncased",
    use_tuning: bool = True,
    max_length: int = 128,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    warmup_steps: int = 500,
) -> Tuple[Path, Dict]:
    """
    Train transformer-based model (BERT/RoBERTa) for emotion detection.
    
    Args:
        texts: Training texts
        labels: Training labels
        label_to_int: Label to integer mapping
        lang: Language code
        model_name: Model name for saving
        model_type: HuggingFace model identifier (e.g., 'bert-base-uncased', 'roberta-base')
        use_tuning: If True, use hyperparameter tuning
        max_length: Maximum sequence length
        batch_size: Training batch size
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        warmup_steps: Number of warmup steps
        
    Returns:
        Tuple of (model_path, metadata)
    """
    print(f"\nTraining transformer model ({model_type}) for {lang}...")
    
    # Convert labels to integers
    y = np.array([label_to_int[label] for label in labels])
    num_labels = len(label_to_int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_STATE, stratify=y_train
    )
    
    train_size = len(X_train)
    val_size = len(X_val)
    test_size = len(X_test)
    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Load tokenizer and model
    print(f"Loading {model_type}...")
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_type,
        num_labels=num_labels,
        problem_type="single_label_classification"
    )
    
    # Create datasets
    train_dataset = EmotionDataset(X_train, y_train.tolist(), tokenizer, max_length)
    val_dataset = EmotionDataset(X_val, y_val.tolist(), tokenizer, max_length)
    test_dataset = EmotionDataset(X_test, y_test.tolist(), tokenizer, max_length)
    
    # Training arguments
    output_dir = MODELS_DIR / lang / f"{model_name}_checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_dir=str(output_dir / "logs"),
        logging_steps=100,
        eval_strategy="epoch",  # Changed from evaluation_strategy to eval_strategy
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",  # Changed from "f1" to match compute_metrics output
        greater_is_better=True,
        seed=RANDOM_STATE,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    )
    
    # Compute metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro')
        return {
            'accuracy': accuracy,
            'f1_macro': f1
        }
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    
    # Train
    print("Training transformer model...")
    trainer.train()
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_results = trainer.evaluate()
    val_accuracy = val_results['eval_accuracy']
    val_f1 = val_results['eval_f1_macro']
    print(f"Validation accuracy: {val_accuracy:.4f}, F1-macro: {val_f1:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    test_accuracy = test_results['eval_accuracy']
    test_f1 = test_results['eval_f1_macro']
    print(f"Test accuracy: {test_accuracy:.4f}, F1-macro: {test_f1:.4f}")
    
    # Save model
    model_dir = MODELS_DIR / lang
    model_path = model_dir / f"{model_name}.joblib"
    
    # Save transformer model and tokenizer separately
    final_model_dir = model_dir / f"{model_name}_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    
    # Save metadata and label mapping
    metadata = {
        "model_type": model_name,
        "model_architecture": model_type,
        "lang": lang,
        "num_labels": num_labels,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "val_accuracy": float(val_accuracy),
        "val_f1_macro": float(val_f1),
        "test_accuracy": float(test_accuracy),
        "test_f1_macro": float(test_f1),
        "max_length": max_length,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
    }
    
    # Save model bundle (compatible with existing loading code)
    model_bundle = {
        "model": trainer.model,
        "tokenizer": tokenizer,
        "label_mapping": label_to_int,
        "metadata": metadata,
        "model_dir": str(final_model_dir),
    }
    
    joblib.dump(model_bundle, model_path)
    
    print(f"Model saved to {model_path}")
    print(f"Model directory: {final_model_dir}")
    
    return model_path, metadata
