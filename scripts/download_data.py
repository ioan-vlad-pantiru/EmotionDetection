"""
Download all required datasets and lexicons.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RAW_DATA_DIR
from src.datasets.goemotions import load_goemotions_simple
from src.datasets.red_ro import download_red_dataset, load_red
from src.lexicons.roemolex import RoEmoLex
from src.lexicons.emolex_en import EmoLexEN


def main():
    print("=" * 60)
    print("Downloading datasets and lexicons")
    print("=" * 60)
    
    # Download GoEmotions (English)
    print("\n1. Downloading GoEmotions dataset (English)...")
    try:
        texts, labels, _ = load_goemotions_simple()
        print(f"   ✓ GoEmotions: {len(texts)} examples loaded")
    except Exception as e:
        print(f"   ✗ GoEmotions download failed: {e}")
        print("   Please ensure tensorflow-datasets or datasets package is installed")
    
    # Download RED (Romanian)
    print("\n2. Downloading RED dataset (Romanian)...")
    try:
        red_dir = download_red_dataset()
        texts, labels, _ = load_red(data_dir=red_dir)
        print(f"   ✓ RED: {len(texts)} examples loaded")
    except Exception as e:
        print(f"   ✗ RED download failed: {e}")
        print("   Please download manually from:")
        print("   https://github.com/Alegzandra/RED-Romanian-Emotion-Datasets")
        print("   Download REDv2 (multi-label) or REDv1 (single-label) CSV files")
        print(f"   and place CSV files in {RAW_DATA_DIR / 'red'}")
    
    # Check RoEmoLex
    print("\n3. Checking RoEmoLex lexicon (Romanian)...")
    try:
        lexicon = RoEmoLex()
        print(f"   ✓ RoEmoLex: {lexicon.get_vocabulary_size()} words loaded")
    except Exception as e:
        print(f"   ✗ RoEmoLex not found: {e}")
        print("   Please download RoEmoLex from:")
        print("   https://www.cs.ubbcluj.ro/~studia-i/journal/journal/article/view/13")
        print(f"   and place it in {RAW_DATA_DIR}/ as roemolex.csv or roemolex.tsv")
    
    # Check EmoLex (English)
    print("\n4. Checking EmoLex lexicon (English)...")
    try:
        lexicon = EmoLexEN()
        print(f"   ✓ EmoLex: {lexicon.get_vocabulary_size()} words loaded")
    except Exception as e:
        print(f"   ⚠ EmoLex not found: {e}")
        print("   Using fallback lexicon. For better results, download NRC EmoLex from:")
        print("   http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm")
        print(f"   and place it in {RAW_DATA_DIR}/ as emolex_en.txt")
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


