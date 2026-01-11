"""
Step 6: Derived emotions (dyads)
"""
import sys
import csv
import json
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    AFFECT_ADDED, FINAL_CSV, FINAL_JSONL, CHECKPOINT_STEP6, LOG_FILE,
    EMOTION_COLUMNS, POLARITY_COLUMNS, DERIVED_EMOTIONS
)
from utils import (
    setup_logging, compute_file_hash, save_checkpoint_meta,
    checkpoint_exists, count_csv_rows
)


def compute_derived_emotions(row: Dict) -> Dict:
    """Compute derived emotion flags based on base emotions."""
    derived = {}
    
    for derived_name, base_emotions in DERIVED_EMOTIONS.items():
        # Check if both base emotions are present
        if len(base_emotions) == 2:
            e1, e2 = base_emotions
            if row.get(e1, 0) == 1 and row.get(e2, 0) == 1:
                derived[derived_name] = 1
            else:
                derived[derived_name] = 0
        else:
            derived[derived_name] = 0
    
    return derived


def run_step6(resume: bool = True) -> bool:
    """Run Step 6: Derived emotions."""
    logger = setup_logging(LOG_FILE)
    logger.info("=" * 60)
    logger.info("Step 6: Derived emotions (dyads)")
    logger.info("=" * 60)
    
    if resume and checkpoint_exists(CHECKPOINT_STEP6):
        logger.info("Step 6 already completed. Skipping.")
        return True
    
    if not AFFECT_ADDED.exists():
        logger.error(f"Input file not found: {AFFECT_ADDED}")
        return False
    
    # Fieldnames for output
    base_fields = ['word'] + EMOTION_COLUMNS + POLARITY_COLUMNS
    extra_fields = ['synset', 'pos', 'sumo', 'provenance']
    derived_fields = list(DERIVED_EMOTIONS.keys())
    fieldnames = base_fields + extra_fields + derived_fields
    
    logger.info("Computing derived emotions...")
    
    csv_rows = []
    jsonl_rows = []
    
    with open(AFFECT_ADDED, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Compute derived emotions
            derived = compute_derived_emotions(row)
            
            # Add derived columns
            new_row = row.copy()
            for derived_name, value in derived.items():
                new_row[derived_name] = value
            
            csv_rows.append(new_row)
            jsonl_rows.append(new_row)
    
    # Write CSV
    logger.info(f"Writing {len(csv_rows)} rows to {FINAL_CSV}...")
    with open(FINAL_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    
    # Write JSONL
    logger.info(f"Writing {len(jsonl_rows)} rows to {FINAL_JSONL}...")
    with open(FINAL_JSONL, 'w', encoding='utf-8') as f:
        for row in jsonl_rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    
    # Save checkpoint
    input_hash = compute_file_hash(AFFECT_ADDED)
    output_csv_hash = compute_file_hash(FINAL_CSV)
    output_jsonl_hash = compute_file_hash(FINAL_JSONL)
    
    save_checkpoint_meta(
        CHECKPOINT_STEP6,
        input_files={str(AFFECT_ADDED): input_hash},
        output_files={
            str(FINAL_CSV): output_csv_hash,
            str(FINAL_JSONL): output_jsonl_hash
        },
        record_counts={
            "input": len(csv_rows),
            "output_csv": len(csv_rows),
            "output_jsonl": len(jsonl_rows)
        }
    )
    
    logger.info(f"Step 6 complete.")
    logger.info(f"  CSV: {FINAL_CSV}")
    logger.info(f"  JSONL: {FINAL_JSONL}")
    
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    
    run_step6(resume=not args.no_resume)



