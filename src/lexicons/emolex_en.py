"""
NRC EmoLex (English) loader.

Reference: NRC Emotion Lexicon
http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm

This is a simplified loader. The lexicon should be placed in data/raw/emolex_en.txt
Format: word<TAB>emotion<TAB>association (0 or 1)
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

from src.config import RAW_DATA_DIR, PLUTCHIK_8
from src.utils.text import normalize_diacritics


class EmoLexEN:
    """
    NRC EmoLex English Emotion Lexicon loader.
    
    Supports loading from tab-separated file with format:
    word<TAB>emotion<TAB>association
    """
    
    def __init__(self, lexicon_path: Path = None):
        """
        Initialize EmoLex loader.
        
        Args:
            lexicon_path: Path to lexicon file. If None, searches in data/raw/
        """
        if lexicon_path is None:
            # Try to find lexicon file
            possible_paths = [
                RAW_DATA_DIR / "emolex_en.txt",
                RAW_DATA_DIR / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
                RAW_DATA_DIR / "NRC-Emotion-Lexicon.txt",
                RAW_DATA_DIR / "emolex_en.tsv",
            ]
            lexicon_path = None
            for path in possible_paths:
                if path.exists():
                    lexicon_path = path
                    break
            
            if lexicon_path is None:
                # Create a minimal fallback lexicon
                print(f"Warning: EmoLex file not found at {RAW_DATA_DIR}. Using minimal fallback.")
                self.word_emotions = self._create_fallback_lexicon()
                return
        
        self.lexicon_path = lexicon_path
        self.word_emotions = defaultdict(lambda: defaultdict(float))
        self._load_lexicon()
    
    def _create_fallback_lexicon(self) -> defaultdict:
        """Create a minimal fallback lexicon with common emotion words."""
        fallback = defaultdict(lambda: defaultdict(float))
        
        # Basic emotion words (very minimal)
        emotion_words = {
            "anger": ["angry", "mad", "furious", "rage", "hate"],
            "fear": ["afraid", "scared", "terrified", "worried", "anxious"],
            "joy": ["happy", "glad", "joyful", "excited", "pleased"],
            "sadness": ["sad", "depressed", "unhappy", "sorrow", "grief"],
            "surprise": ["surprised", "shocked", "amazed", "astonished"],
            "disgust": ["disgusted", "revolted", "sickened"],
            "trust": ["trust", "confident", "sure", "reliable"],
            "anticipation": ["anticipate", "expect", "hope", "wait"],
        }
        
        for emotion, words in emotion_words.items():
            for word in words:
                fallback[word.lower()][emotion] = 1.0
        
        return fallback
    
    def _load_lexicon(self):
        """Load lexicon from file."""
        try:
            # NRC EmoLex format: word<TAB>emotion<TAB>association
            df = pd.read_csv(
                self.lexicon_path,
                sep="\t",
                header=None,
                names=["word", "emotion", "association"],
                encoding="utf-8",
            )
        except Exception as e:
            print(f"Warning: Failed to load EmoLex from {self.lexicon_path}: {e}")
            print("Using fallback lexicon.")
            self.word_emotions = self._create_fallback_lexicon()
            return
        
        # Map NRC emotions to Plutchik 8
        emotion_mapping = {
            "anger": "anger",
            "fear": "fear",
            "joy": "joy",
            "sadness": "sadness",
            "surprise": "surprise",
            "disgust": "disgust",
            "trust": "trust",
            "anticipation": "anticipation",
        }
        
        # Load lexicon entries
        for _, row in df.iterrows():
            word = str(row["word"]).lower().strip()
            emotion_raw = str(row["emotion"]).lower().strip()
            association = int(row["association"])
            
            # Only process if association is 1
            if association != 1:
                continue
            
            emotion = emotion_mapping.get(emotion_raw)
            if emotion and emotion in PLUTCHIK_8:
                self.word_emotions[word][emotion] = 1.0
    
    def get_emotion_scores(self, tokens: List[str], negation_positions: List[int] = None) -> Dict[str, float]:
        """
        Get emotion scores for a list of tokens with improved negation handling.
        
        Args:
            tokens: List of token strings
            negation_positions: List of token indices in negation window
            
        Returns:
            Dictionary mapping emotion -> total score
        """
        from src.utils.negation import invert_emotion
        
        emotion_scores = defaultdict(float)
        negation_positions = set(negation_positions or [])
        
        for i, token in enumerate(tokens):
            token = token.lower()
            
            if token in self.word_emotions:
                if i in negation_positions:
                    # Improved negation: invert emotions instead of just downweighting
                    for emotion, score in self.word_emotions[token].items():
                        # Invert the emotion (joy -> sadness, etc.)
                        inverted_emotion = invert_emotion(emotion)
                        # Apply inverted emotion with reduced weight (0.5x)
                        emotion_scores[inverted_emotion] += score * 0.5
                        # Also reduce original emotion (0.1x to cancel it out)
                        emotion_scores[emotion] += score * 0.1
                else:
                    # Normal: add emotion scores
                    for emotion, score in self.word_emotions[token].items():
                        emotion_scores[emotion] += score
        
        return dict(emotion_scores)
    
    def get_emotion_counts(self, tokens: List[str], negation_positions: List[int] = None) -> Dict[str, int]:
        """
        Get emotion counts for tokens.
        
        Args:
            tokens: List of token strings
            negation_positions: List of token indices in negation window
            
        Returns:
            Dictionary mapping emotion -> count
        """
        emotion_counts = defaultdict(int)
        negation_positions = set(negation_positions or [])
        
        for i, token in enumerate(tokens):
            token = token.lower()
            
            if token in self.word_emotions:
                if i not in negation_positions:
                    for emotion in self.word_emotions[token]:
                        emotion_counts[emotion] += 1
        
        return dict(emotion_counts)
    
    def get_vocabulary_size(self) -> int:
        """Return number of unique words in lexicon."""
        return len(self.word_emotions)


