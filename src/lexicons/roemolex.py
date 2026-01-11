"""
RoEmoLex Romanian Emotion Lexicon loader.

Reference: RoEmoLex - A Romanian Emotion Lexicon
https://www.cs.ubbcluj.ro/~studia-i/journal/journal/article/view/13

The lexicon should be placed in data/raw/roemolex.csv or data/raw/roemolex.tsv
Expected format: word, emotion, score (or word, emotion, flag)
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import re

from src.config import RAW_DATA_DIR, PLUTCHIK_8
from src.utils.text import normalize_diacritics


class RoEmoLex:
    """
    RoEmoLex Romanian Emotion Lexicon.
    
    Supports loading from CSV/TSV files with format:
    - word, emotion, score (or intensity)
    - word, emotion, flag (binary)
    """
    
    def __init__(self, lexicon_path: Path = None):
        """
        Initialize RoEmoLex loader.
        
        Args:
            lexicon_path: Path to lexicon file. If None, searches in data/raw/
        """
        if lexicon_path is None:
            # Try to find lexicon file
            possible_paths = [
                RAW_DATA_DIR / "roemolex.csv",
                RAW_DATA_DIR / "roemolex.tsv",
                RAW_DATA_DIR / "RoEmoLex.csv",
                RAW_DATA_DIR / "RoEmoLex.tsv",
            ]
            lexicon_path = None
            for path in possible_paths:
                if path.exists():
                    lexicon_path = path
                    break
            
            if lexicon_path is None:
                raise FileNotFoundError(
                    f"RoEmoLex file not found. Please place it in {RAW_DATA_DIR}/ "
                    "as roemolex.csv or roemolex.tsv"
                )
        
        self.lexicon_path = lexicon_path
        self.word_emotions = defaultdict(lambda: defaultdict(float))
        self._load_lexicon()
    
    def _load_lexicon(self):
        """Load lexicon from file."""
        # Try CSV first, then TSV
        try:
            if self.lexicon_path.suffix == ".csv":
                df = pd.read_csv(self.lexicon_path, encoding="utf-8")
            else:
                df = pd.read_csv(self.lexicon_path, sep="\t", encoding="utf-8")
        except Exception as e:
            raise ValueError(f"Failed to load lexicon from {self.lexicon_path}: {e}")
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Check if this is the wide format (one column per emotion) or long format (word, emotion, score)
        emotion_cols = [col for col in df.columns if col in PLUTCHIK_8]
        
        if emotion_cols:
            # Wide format: word, anger, anticipation, disgust, fear, joy, sadness, surprise, trust, ...
            if "word" not in df.columns:
                raise ValueError(f"Lexicon must have 'word' column. Found: {df.columns.tolist()}")
            
            # Load from wide format
            for _, row in df.iterrows():
                word = str(row["word"]).lower().strip()
                if not word:
                    continue
                
                # Normalize word (handle diacritics)
                word_normalized = normalize_diacritics(word)
                
                # For each emotion column, if value is 1 (or > 0), add to lexicon
                for emotion in PLUTCHIK_8:
                    if emotion in df.columns:
                        val = row[emotion]
                        # Handle both string "1"/"0" and numeric 1/0
                        if isinstance(val, str):
                            val = 1 if val.strip() in ["1", "1.0", "True", "true"] else 0
                        else:
                            val = 1 if float(val) > 0 else 0
                        
                        if val > 0:
                            self.word_emotions[word_normalized][emotion] = float(val)
        else:
            # Long format: word, emotion, score/intensity/flag
            required_cols = ["word", "emotion"]
            if not all(col in df.columns for col in required_cols):
                raise ValueError(
                    f"Lexicon must have columns: {required_cols} or emotion columns {PLUTCHIK_8}. Found: {df.columns.tolist()}"
                )
            
            # Find score column
            score_col = None
            for col in ["score", "intensity", "flag", "value"]:
                if col in df.columns:
                    score_col = col
                    break
            
            # If no score column, assume binary (1 if present)
            if score_col is None:
                score_col = "score"
                df[score_col] = 1.0
            
            # Map emotions to Plutchik 8 (normalize emotion names)
            emotion_mapping = {
                "anger": "anger",
                "furie": "anger",
                "fear": "fear",
                "frică": "fear",
                "anticipation": "anticipation",
                "anticipare": "anticipation",
                "trust": "trust",
                "încredere": "trust",
                "surprise": "surprise",
                "surpriză": "surprise",
                "sadness": "sadness",
                "tristețe": "sadness",
                "joy": "joy",
                "bucurie": "joy",
                "disgust": "disgust",
                "dezgust": "disgust",
            }
            
            # Load lexicon entries
            for _, row in df.iterrows():
                word = str(row["word"]).lower().strip()
                word = normalize_diacritics(word)
                
                emotion_raw = str(row["emotion"]).lower().strip()
                emotion = emotion_mapping.get(emotion_raw, emotion_raw)
                
                # Only keep Plutchik 8 emotions
                if emotion not in PLUTCHIK_8:
                    continue
                
                score = float(row[score_col])
                
                # Store emotion score for word
                self.word_emotions[word][emotion] = max(
                    self.word_emotions[word][emotion], score
                )
    
    def get_emotion_scores(self, tokens: List[str], negation_positions: List[int] = None) -> Dict[str, float]:
        """
        Get emotion scores for a list of tokens.
        
        Args:
            tokens: List of token strings
            negation_positions: List of token indices in negation window
            
        Returns:
            Dictionary mapping emotion -> total score
        """
        emotion_scores = defaultdict(float)
        negation_positions = set(negation_positions or [])
        
        for i, token in enumerate(tokens):
            token = normalize_diacritics(token.lower())
            
            # Check if token has emotion associations
            if token in self.word_emotions:
                # Check if in negation window
                if i in negation_positions:
                    # Downweight or invert (simple: downweight by 0.3)
                    weight = 0.3
                else:
                    weight = 1.0
                
                for emotion, score in self.word_emotions[token].items():
                    emotion_scores[emotion] += score * weight
        
        return dict(emotion_scores)
    
    def get_emotion_counts(self, tokens: List[str], negation_positions: List[int] = None) -> Dict[str, int]:
        """
        Get emotion counts (number of words with each emotion) for tokens.
        
        Args:
            tokens: List of token strings
            negation_positions: List of token indices in negation window
            
        Returns:
            Dictionary mapping emotion -> count
        """
        emotion_counts = defaultdict(int)
        negation_positions = set(negation_positions or [])
        
        for i, token in enumerate(tokens):
            token = normalize_diacritics(token.lower())
            
            if token in self.word_emotions:
                # In negation window, don't count (or count with lower weight)
                if i not in negation_positions:
                    for emotion in self.word_emotions[token]:
                        emotion_counts[emotion] += 1
        
        return dict(emotion_counts)
    
    def get_vocabulary_size(self) -> int:
        """Return number of unique words in lexicon."""
        return len(self.word_emotions)


