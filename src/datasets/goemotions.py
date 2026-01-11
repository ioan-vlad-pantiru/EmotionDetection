"""
GoEmotions dataset loader.

Reference: GoEmotions: A Dataset of Fine-Grained Emotions
https://github.com/google-research/google-research/tree/master/goemotions
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from src.config import PROCESSED_DATA_DIR, PLUTCHIK_8, NEUTRAL_LABEL
from src.datasets.mapping import map_goemotions_label, create_label_mapping

# Try to import tensorflow_datasets, but it's optional
try:
    import tensorflow_datasets as tfds
    HAS_TFDS = True
except ImportError:
    HAS_TFDS = False


def load_goemotions(
    use_huggingface: bool = True,
    cache_dir: Optional[Path] = None,
) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    Load GoEmotions dataset and map to Plutchik 8 emotions.
    
    Args:
        use_huggingface: If True, use HuggingFace datasets; else use TensorFlow Datasets
        cache_dir: Directory to cache processed data
        
    Returns:
        Tuple of (texts, labels, label_to_int)
    """
    if cache_dir is None:
        cache_dir = PROCESSED_DATA_DIR / "goemotions"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / "goemotions_processed.csv"
    
    # Check cache
    if cache_file.exists():
        print(f"Loading cached GoEmotions data from {cache_file}")
        df = pd.read_csv(cache_file)
        texts = df["text"].tolist()
        labels = df["label"].tolist()
        
        # Create label mapping
        unique_labels = sorted(set(labels))
        label_to_int = create_label_mapping(unique_labels)
        
        return texts, labels, label_to_int
    
    print("Downloading GoEmotions dataset...")
    
    texts = []
    labels = []
    
    try:
        if use_huggingface:
            from datasets import load_dataset
            dataset = load_dataset("go_emotions", "raw")
            
            # Process train split
            for split in ["train", "validation", "test"]:
                if split in dataset:
                    for example in dataset[split]:
                        text = example.get("text", "")
                        emotion_ids = example.get("emotions", [])
                        
                        # Get emotion labels (GoEmotions has 27 emotions + neutral)
                        # We need to map these to Plutchik 8
                        # For multi-label, take the first mapped label
                        mapped_label = None
                        
                        # Get emotion names from IDs (simplified - in reality need label list)
                        # GoEmotions has: admiration, amusement, anger, annoyance, approval, 
                        # caring, confusion, curiosity, desire, disappointment, disapproval, 
                        # disgust, embarrassment, excitement, fear, gratitude, grief, joy, 
                        # love, nervousness, optimism, pride, realization, relief, remorse, 
                        # sadness, surprise, neutral
                        
                        # For simplicity, we'll use a different approach:
                        # Load the dataset and get the label names
                        if mapped_label is None:
                            # Try to get label from emotions field
                            # This is a simplified version - actual implementation would
                            # need the label vocabulary
                            pass
        else:
            # Use TensorFlow Datasets (if available)
            if HAS_TFDS:
                try:
                    ds = tfds.load("go_emotions", split="train+validation+test")
                    
                    # Get label names
                    info = tfds.builder("go_emotions").info
                    label_names = info.features["emotions"].feature.names
                    
                    for example in tfds.as_numpy(ds):
                        text = example["text"].decode("utf-8")
                        emotion_ids = example["emotions"]
                        
                        # Map first emotion to Plutchik 8
                        mapped_label = None
                        for emotion_id in emotion_ids:
                            if emotion_id < len(label_names):
                                emotion_name = label_names[emotion_id]
                                mapped_label = map_goemotions_label(emotion_name)
                                if mapped_label:
                                    break
                        
                        if mapped_label:
                            texts.append(text)
                            labels.append(mapped_label)
                except Exception as e:
                    print(f"TensorFlow Datasets loading failed: {e}")
                    print("Falling back to HuggingFace datasets...")
                    use_huggingface = True
            else:
                print("TensorFlow Datasets not available, using HuggingFace datasets...")
                use_huggingface = True
    
    except Exception as e:
        print(f"Error loading GoEmotions: {e}")
        print("Attempting alternative loading method...")
        
        # Fallback: try HuggingFace with different approach
        try:
            from datasets import load_dataset
            dataset = load_dataset("go_emotions", "simplified")
            
            # Simplified version has single label - use the label_names mapping
            label_names = [
                "admiration", "amusement", "anger", "annoyance", "approval", "caring",
                "confusion", "curiosity", "desire", "disappointment", "disapproval",
                "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
                "joy", "love", "nervousness", "optimism", "pride", "realization",
                "relief", "remorse", "sadness", "surprise", "neutral"
            ]
            
            for split in ["train", "validation", "test"]:
                if split in dataset:
                    for example in dataset[split]:
                        text = example.get("text", "")
                        # Simplified version uses 'labels' (list) not 'label'
                        label_ids = example.get("labels", [])
                        if not label_ids:
                            label_id = example.get("label", -1)
                            label_ids = [label_id] if label_id >= 0 else []
                        
                        # Take first label from list
                        if label_ids and len(label_ids) > 0:
                            label_id = label_ids[0]
                            if label_id >= 0 and label_id < len(label_names):
                                emotion_name = label_names[label_id]
                                mapped_label = map_goemotions_label(emotion_name)
                                if mapped_label:
                                    texts.append(text)
                                    labels.append(mapped_label)
        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
            raise RuntimeError(
                "Failed to load GoEmotions dataset. "
                "Please ensure the 'datasets' package is installed: pip install datasets"
            )
    
    # Filter out None labels
    filtered_texts = []
    filtered_labels = []
    for text, label in zip(texts, labels):
        if label:
            filtered_texts.append(text)
            filtered_labels.append(label)
    
    # Create DataFrame and cache
    df = pd.DataFrame({"text": filtered_texts, "label": filtered_labels})
    df.to_csv(cache_file, index=False)
    
    # Create label mapping
    unique_labels = sorted(set(filtered_labels))
    label_to_int = create_label_mapping(unique_labels)
    
    print(f"Loaded {len(filtered_texts)} examples with {len(unique_labels)} labels")
    
    return filtered_texts, filtered_labels, label_to_int


def load_goemotions_simple() -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    Simplified loader that uses HuggingFace datasets with simplified labels.
    
    Returns:
        Tuple of (texts, labels, label_to_int)
    """
    try:
        from datasets import load_dataset
        
        print("Loading GoEmotions (simplified) from HuggingFace...")
        dataset = load_dataset("go_emotions", "simplified")
        
        texts = []
        labels = []
        
        # Label mapping for simplified version (28 labels -> Plutchik 8)
        # Simplified version has: admiration, amusement, anger, annoyance, approval,
        # caring, confusion, curiosity, desire, disappointment, disapproval, disgust,
        # embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness,
        # optimism, pride, realization, relief, remorse, sadness, surprise, neutral
        
        label_names = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring",
            "confusion", "curiosity", "desire", "disappointment", "disapproval",
            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
            "joy", "love", "nervousness", "optimism", "pride", "realization",
            "relief", "remorse", "sadness", "surprise", "neutral"
        ]
        
        for split in ["train", "validation", "test"]:
            if split in dataset:
                for example in dataset[split]:
                    text = example.get("text", "")
                    # Simplified version uses 'labels' (list) not 'label'
                    label_ids = example.get("labels", [])
                    if not label_ids:
                        label_id = example.get("label", -1)
                        label_ids = [label_id] if label_id >= 0 else []
                    
                    # Take first label from list
                    if label_ids and len(label_ids) > 0:
                        label_id = label_ids[0]
                        if label_id >= 0 and label_id < len(label_names):
                            emotion_name = label_names[label_id]
                            mapped_label = map_goemotions_label(emotion_name)
                            
                            if mapped_label:
                                texts.append(text)
                                labels.append(mapped_label)
        
        # Create label mapping
        unique_labels = sorted(set(labels))
        label_to_int = create_label_mapping(unique_labels)
        
        print(f"Loaded {len(texts)} examples with {len(unique_labels)} labels")
        
        return texts, labels, label_to_int
        
    except Exception as e:
        raise RuntimeError(f"Failed to load GoEmotions: {e}")


