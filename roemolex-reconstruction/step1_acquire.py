"""
Step 1: Acquire base lexicon (Romanian NRC EmoLex)
"""
import sys
from pathlib import Path
import hashlib
import logging

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    RAW_DIR, CHECKPOINT_STEP1, NRC_EMOLEX_RO_RAW,
    LOG_FILE
)
from utils import (
    setup_logging, compute_file_hash, save_checkpoint_meta,
    checkpoint_exists
)


def download_nrc_emolex_ro(output_path: Path) -> bool:
    """
    Download or locate Romanian NRC EmoLex.
    
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    # Check if file already exists
    if output_path.exists():
        logger.info(f"Romanian NRC EmoLex already exists at {output_path}")
        return True
    
    # Try to find in parent project's data directory
    parent_data = Path(__file__).parent.parent / "data" / "raw"
    possible_locations = [
        parent_data / "nrc_emolex_ro.txt",  # Romanian-specific file
        parent_data / "NRC-Emotion-Lexicon-Romanian.txt",
        parent_data / "roemolex.csv",  # Original RoEmoLex format
        parent_data / "roemolex.tsv",
        Path.home() / "Downloads" / "NRC-Emotion-Lexicon" / "OneFilePerLanguage" / "Romanian.txt",
        Path.home() / "Downloads" / "NRC-Emotion-Lexicon" / "Romanian.txt",
    ]
    
    for loc in possible_locations:
        if loc.exists():
            # Verify it's actually Romanian (check for Romanian characters)
            try:
                with open(loc, 'r', encoding='utf-8') as f:
                    sample = f.read(1000)
                    # Check for Romanian-specific characters: ă, â, î, ș, ț
                    if any(char in sample for char in ['ă', 'â', 'î', 'ș', 'ț', 'Ă', 'Â', 'Î', 'Ș', 'Ț']):
                        logger.info(f"Found Romanian EmoLex at {loc}, copying...")
                        import shutil
                        shutil.copy(loc, output_path)
                        return True
                    else:
                        logger.warning(f"Found file at {loc} but it doesn't appear to contain Romanian text. Skipping...")
            except Exception as e:
                logger.warning(f"Error checking file {loc}: {e}")
                continue
    
    # Try to download from HuggingFace datasets
    try:
        from datasets import load_dataset
        logger.info("Attempting to download Romanian EmoLex from HuggingFace...")
        
        # Try different dataset names
        dataset = None
        for name in ["nrc_emotion_lexicon", "emotion_lexicon"]:
            try:
                dataset = load_dataset(name)
                break
            except:
                continue
        
        if dataset:
            # Extract Romanian entries
            logger.info("Extracting Romanian entries...")
            ro_entries = []
            for split in dataset.keys():
                for example in dataset[split]:
                    lang = example.get("language", "").lower()
                    if "romanian" in lang or "ro" == lang:
                        ro_entries.append(example)
            
            if ro_entries:
                # Write to file
                with open(output_path, 'w', encoding='utf-8') as f:
                    # Write header
                    f.write("word\temotion\tassociation\n")
                    for entry in ro_entries:
                        word = entry.get("word", "")
                        emotion = entry.get("emotion", "")
                        association = entry.get("association", "0")
                        f.write(f"{word}\t{emotion}\t{association}\n")
                logger.info(f"Downloaded {len(ro_entries)} Romanian entries")
                return True
    except Exception as e:
        logger.warning(f"Failed to download from HuggingFace: {e}")
    
    # Fallback: create minimal placeholder
    logger.warning("Could not find Romanian EmoLex. Creating placeholder file.")
    logger.warning("Please manually download Romanian NRC EmoLex and place it at:")
    logger.warning(f"  {output_path}")
    
    # Create empty placeholder
    output_path.write_text("word\temotion\tassociation\n", encoding='utf-8')
    return False


def run_step1(resume: bool = True) -> bool:
    """Run Step 1: Acquire base lexicon."""
    logger = setup_logging(LOG_FILE)
    logger.info("=" * 60)
    logger.info("Step 1: Acquire base lexicon (Romanian NRC EmoLex)")
    logger.info("=" * 60)
    
    # Check checkpoint
    if resume and checkpoint_exists(CHECKPOINT_STEP1):
        logger.info("Step 1 already completed. Skipping.")
        return True
    
    # Download/locate Romanian EmoLex
    success = download_nrc_emolex_ro(NRC_EMOLEX_RO_RAW)
    
    if not NRC_EMOLEX_RO_RAW.exists():
        logger.error("Failed to acquire Romanian EmoLex file")
        return False
    
    # Compute hash
    file_hash = compute_file_hash(NRC_EMOLEX_RO_RAW)
    file_size = NRC_EMOLEX_RO_RAW.stat().st_size
    
    # Count lines
    line_count = sum(1 for _ in open(NRC_EMOLEX_RO_RAW, 'r', encoding='utf-8'))
    
    # Save checkpoint
    save_checkpoint_meta(
        CHECKPOINT_STEP1,
        input_files={},
        output_files={str(NRC_EMOLEX_RO_RAW): file_hash},
        record_counts={str(NRC_EMOLEX_RO_RAW): line_count},
        extra_meta={
            "file_size_bytes": file_size,
            "source": "NRC EmoLex Romanian translation"
        }
    )
    
    logger.info(f"Step 1 complete. File: {NRC_EMOLEX_RO_RAW}")
    logger.info(f"  Hash: {file_hash}")
    logger.info(f"  Size: {file_size} bytes")
    logger.info(f"  Lines: {line_count}")
    
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")
    args = parser.parse_args()
    
    run_step1(resume=not args.no_resume)

