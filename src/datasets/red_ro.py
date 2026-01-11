"""
RED / REDv2 Romanian Emotion Dataset loader.

Reference: RED - Romanian Emotion Dataset
https://github.com/dumitrescustefan/RED
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import requests
import zipfile

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, PLUTCHIK_8, NEUTRAL_LABEL
from src.datasets.mapping import map_red_label, create_label_mapping


def download_red_dataset(output_dir: Path = None) -> Path:
    """
    Download RED dataset from GitHub.
    
    Args:
        output_dir: Directory to save dataset
        
    Returns:
        Path to downloaded dataset directory
    """
    if output_dir is None:
        output_dir = RAW_DATA_DIR / "red"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to download from HuggingFace first (easier)
    try:
        from datasets import load_dataset
        print("Loading RED dataset from HuggingFace...")
        # Try different possible dataset names
        dataset = None
        for name in ["Alegzandra/REDv2", "Alegzandra/RED-Romanian-Emotion-Datasets", "dumitrescustefan/red", "red"]:
            try:
                dataset = load_dataset(name)
                break
            except Exception as e:
                print(f"  Trying {name}... failed: {e}")
                continue
        
        if dataset:
            # Save to CSV
            for split in ["train", "validation", "test"]:
                if split in dataset:
                    df = pd.DataFrame(dataset[split])
                    df.to_csv(output_dir / f"red_{split}.csv", index=False)
            return output_dir
    except Exception as e:
        print(f"HuggingFace download failed: {e}")
    
    print("Please manually download RED dataset from:")
    print("https://github.com/Alegzandra/RED-Romanian-Emotion-Datasets")
    print("Download REDv2 (multi-label) or REDv1 (single-label) CSV files")
    print(f"and place CSV files in {output_dir}")
    return output_dir


def load_red(
    data_dir: Optional[Path] = None,
    use_single_label: bool = True,
    cache_dir: Optional[Path] = None,
) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    Load RED/REDv2 dataset.
    
    Args:
        data_dir: Directory containing RED dataset files
        use_single_label: If True, use single-label version; else multi-label
        cache_dir: Directory to cache processed data
        
    Returns:
        Tuple of (texts, labels, label_to_int)
    """
    if data_dir is None:
        data_dir = RAW_DATA_DIR / "red"
    
    if cache_dir is None:
        cache_dir = PROCESSED_DATA_DIR / "red"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / "red_processed.csv"
    
    # Check cache
    if cache_file.exists():
        print(f"Loading cached RED data from {cache_file}")
        df = pd.read_csv(cache_file)
        texts = df["text"].tolist()
        labels = df["label"].tolist()
        
        unique_labels = sorted(set(labels))
        label_to_int = create_label_mapping(unique_labels)
        
        return texts, labels, label_to_int
    
    # Try to load from HuggingFace
    try:
        from datasets import load_dataset
        print("Loading RED dataset from HuggingFace...")
        
        # Try different dataset names
        dataset = None
        for name in ["Alegzandra/REDv2", "Alegzandra/RED-Romanian-Emotion-Datasets", "dumitrescustefan/red", "red"]:
            try:
                dataset = load_dataset(name)
                break
            except:
                continue
        
        if dataset is None:
            raise ValueError("RED dataset not found on HuggingFace")
        
        texts = []
        labels = []
        
        # REDv2 emotion order in agreed_labels: ['Sadness', 'Surprise', 'Fear', 'Anger', 'Neutral', 'Trust', 'Joy']
        emotion_order = ['Sadness', 'Surprise', 'Fear', 'Anger', 'Neutral', 'Trust', 'Joy']
        
        for split in ["train", "validation", "test"]:
            if split in dataset:
                for example in dataset[split]:
                    text = example.get("text", "") or example.get("sentence", "")
                    if not text or not text.strip():
                        continue
                    
                    if use_single_label:
                        # Check boolean emotion fields first (REDv2 format)
                        emotion_found = None
                        for emotion in emotion_order:
                            if example.get(emotion, False):
                                emotion_found = emotion
                                break
                        
                        # If no boolean fields, try agreed_labels vector
                        if emotion_found is None:
                            agreed_labels = example.get("agreed_labels", [])
                            if agreed_labels and isinstance(agreed_labels, list):
                                # Find first emotion with value 1
                                for idx, val in enumerate(agreed_labels):
                                    if val == 1 and idx < len(emotion_order):
                                        emotion_found = emotion_order[idx]
                                        break
                        
                        # Fallback to label field
                        if emotion_found is None:
                            emotion_found = example.get("label", "") or example.get("emotion", "")
                        
                        if emotion_found:
                            mapped_label = map_red_label(str(emotion_found))
                            if mapped_label:
                                texts.append(text)
                                labels.append(mapped_label)
                    else:
                        # Multi-label: collect all emotions
                        emotion_list = []
                        for emotion in emotion_order:
                            if example.get(emotion, False):
                                emotion_list.append(emotion)
                        
                        # If no boolean fields, use agreed_labels
                        if not emotion_list:
                            agreed_labels = example.get("agreed_labels", [])
                            if agreed_labels and isinstance(agreed_labels, list):
                                for idx, val in enumerate(agreed_labels):
                                    if val == 1 and idx < len(emotion_order):
                                        emotion_list.append(emotion_order[idx])
                        
                        # Take first label for single-label classification
                        if emotion_list:
                            mapped_label = map_red_label(str(emotion_list[0]))
                            if mapped_label:
                                texts.append(text)
                                labels.append(mapped_label)
    
    except Exception as e:
        print(f"HuggingFace loading failed: {e}")
        print("Trying to load from local files...")
        
        # Try loading from local CSV files
        texts = []
        labels = []
        
        # Look for CSV files
        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            # Download dataset
            print("Dataset not found. Attempting download...")
            download_red_dataset(data_dir)
            csv_files = list(data_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(
                f"RED dataset not found. Please download from "
                "https://github.com/Alegzandra/RED-Romanian-Emotion-Datasets "
                f"and place CSV files in {data_dir}"
            )
        
        # Load CSV files
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, encoding="utf-8")
                
                # Find text and label columns
                text_col = None
                label_col = None
                
                for col in df.columns:
                    col_lower = col.lower()
                    if text_col is None and ("text" in col_lower or "sentence" in col_lower or "tweet" in col_lower):
                        text_col = col
                    if label_col is None and ("label" in col_lower or "emotion" in col_lower):
                        label_col = col
                
                if text_col and label_col:
                    for _, row in df.iterrows():
                        text = str(row[text_col])
                        label = str(row[label_col])
                        
                        mapped_label = map_red_label(label)
                        if mapped_label:
                            texts.append(text)
                            labels.append(mapped_label)
            except Exception as e2:
                print(f"Error loading {csv_file}: {e2}")
                continue
    
    if not texts:
        raise RuntimeError("Failed to load any data from RED dataset")
    
    # Filter out empty texts
    filtered_texts = []
    filtered_labels = []
    for text, label in zip(texts, labels):
        if text and text.strip():
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


