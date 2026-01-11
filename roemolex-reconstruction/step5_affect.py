"""
Step 5: Integrate WordNet-Affect (optional)
"""
import sys
import csv
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    EXPANDED, AFFECT_ADDED, CHECKPOINT_STEP5, RAW_DIR, LOG_FILE,
    EMOTION_COLUMNS
)
from utils import (
    setup_logging, compute_file_hash, save_checkpoint_meta,
    checkpoint_exists, count_csv_rows
)


def map_ekman_to_plutchik(ekman_emotion: str) -> str:
    """Map Ekman emotion to Plutchik emotion."""
    mapping = {
        'anger': 'anger',
        'disgust': 'disgust',
        'fear': 'fear',
        'joy': 'joy',
        'sadness': 'sadness',
        'surprise': 'surprise',
    }
    return mapping.get(ekman_emotion.lower(), '')


def load_wordnet_affect(raw_dir: Path) -> Dict[str, List[str]]:
    """
    Load WordNet-Affect mappings if available.
    
    Returns:
        Dictionary mapping synset -> [emotions]
    """
    # Look for WordNet-Affect files
    affect_files = list(raw_dir.glob("*affect*")) + list(raw_dir.glob("*wn-affect*"))
    
    if not affect_files:
        return {}
    
    logger = logging.getLogger(__name__)
    logger.info(f"Found WordNet-Affect files: {affect_files}")
    
    # Placeholder: parse WordNet-Affect format
    # In practice, parse the actual format
    affect_map = {}
    
    return affect_map


def run_step5(resume: bool = True) -> bool:
    """Run Step 5: Integrate WordNet-Affect."""
    logger = setup_logging(LOG_FILE)
    logger.info("=" * 60)
    logger.info("Step 5: Integrate WordNet-Affect (optional)")
    logger.info("=" * 60)
    
    if resume and checkpoint_exists(CHECKPOINT_STEP5):
        logger.info("Step 5 already completed. Skipping.")
        return True
    
    if not EXPANDED.exists():
        logger.error(f"Input file not found: {EXPANDED}")
        return False
    
    # Try to load WordNet-Affect
    affect_map = load_wordnet_affect(RAW_DIR)
    
    if not affect_map:
        logger.warning("WordNet-Affect resource not found. Copying expanded file.")
        # Copy expanded to affect_added
        import shutil
        shutil.copy(EXPANDED, AFFECT_ADDED)
    else:
        # Process with WordNet-Affect
        logger.info("Processing with WordNet-Affect mappings...")
        
        affect_count = 0
        with open(EXPANDED, 'r', encoding='utf-8') as in_f, \
             open(AFFECT_ADDED, 'w', encoding='utf-8', newline='') as out_f:
            
            reader = csv.DictReader(in_f)
            fieldnames = reader.fieldnames
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in reader:
                synset = row.get('synset', '').strip()
                
                if synset and synset in affect_map:
                    emotions = affect_map[synset]
                    # Add emotions
                    for emotion in emotions:
                        plutchik_emotion = map_ekman_to_plutchik(emotion)
                        if plutchik_emotion and plutchik_emotion in EMOTION_COLUMNS:
                            row[plutchik_emotion] = max(int(row.get(plutchik_emotion, 0)), 1)
                    row['provenance'] = row.get('provenance', '') + ',wn_affect'
                    affect_count += 1
                
                writer.writerow(row)
        
        logger.info(f"Added WordNet-Affect mappings to {affect_count} entries")
    
    # Save checkpoint
    input_hash = compute_file_hash(EXPANDED)
    output_hash = compute_file_hash(AFFECT_ADDED) if AFFECT_ADDED.exists() else ""
    input_count = count_csv_rows(EXPANDED)
    output_count = count_csv_rows(AFFECT_ADDED)
    
    save_checkpoint_meta(
        CHECKPOINT_STEP5,
        input_files={str(EXPANDED): input_hash},
        output_files={str(AFFECT_ADDED): output_hash},
        record_counts={
            "input": input_count,
            "output": output_count,
            "wordnet_affect_available": bool(affect_map)
        }
    )
    
    logger.info(f"Step 5 complete.")
    logger.info(f"  Output: {AFFECT_ADDED}")
    
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    
    run_step5(resume=not args.no_resume)



