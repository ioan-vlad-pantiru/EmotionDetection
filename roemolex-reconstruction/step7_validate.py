"""
Step 7: Final consistency checks
"""
import sys
import csv
import json
from pathlib import Path
from collections import Counter, defaultdict
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    FINAL_CSV, STATS_JSON, CHECKPOINT_STEP7, LOG_FILE,
    EMOTION_COLUMNS, POLARITY_COLUMNS
)
from utils import (
    setup_logging, compute_file_hash, save_checkpoint_meta,
    checkpoint_exists, count_csv_rows
)


def validate_utf8(file_path: Path) -> bool:
    """Validate file is UTF-8 encoded."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read()
        return True
    except UnicodeDecodeError:
        return False


def check_duplicates(file_path: Path) -> Dict:
    """Check for duplicate rows."""
    seen = set()
    duplicates = 0
    total = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            # Create signature
            sig = tuple(sorted(row.items()))
            if sig in seen:
                duplicates += 1
            seen.add(sig)
    
    return {
        "total_rows": total,
        "unique_rows": len(seen),
        "duplicates": duplicates
    }


def compute_label_distribution(file_path: Path) -> Dict:
    """Compute distribution of emotion labels."""
    emotion_counts = defaultdict(int)
    total_rows = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            for emotion in EMOTION_COLUMNS + POLARITY_COLUMNS:
                if int(row.get(emotion, 0)) == 1:
                    emotion_counts[emotion] += 1
    
    return {
        "total_rows": total_rows,
        "emotion_distribution": dict(emotion_counts),
        "rows_with_emotions": sum(1 for v in emotion_counts.values() if v > 0)
    }


def validate_schema(file_path: Path, expected_fields: list) -> Dict:
    """Validate CSV schema."""
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        actual_fields = reader.fieldnames
    
    missing = set(expected_fields) - set(actual_fields or [])
    extra = set(actual_fields or []) - set(expected_fields)
    
    return {
        "expected_fields": expected_fields,
        "actual_fields": list(actual_fields or []),
        "missing_fields": list(missing),
        "extra_fields": list(extra),
        "schema_valid": len(missing) == 0
    }


def run_step7(resume: bool = True) -> bool:
    """Run Step 7: Final validation."""
    logger = setup_logging(LOG_FILE)
    logger.info("=" * 60)
    logger.info("Step 7: Final consistency checks")
    logger.info("=" * 60)
    
    if resume and checkpoint_exists(CHECKPOINT_STEP7):
        logger.info("Step 7 already completed. Skipping.")
        return True
    
    if not FINAL_CSV.exists():
        logger.error(f"Final CSV not found: {FINAL_CSV}")
        return False
    
    stats = {}
    
    # 1. UTF-8 validation
    logger.info("Validating UTF-8 encoding...")
    stats["utf8_valid"] = validate_utf8(FINAL_CSV)
    if not stats["utf8_valid"]:
        logger.error("File is not valid UTF-8!")
        return False
    
    # 2. Schema validation
    logger.info("Validating schema...")
    expected_fields = ['word'] + EMOTION_COLUMNS + POLARITY_COLUMNS
    schema_check = validate_schema(FINAL_CSV, expected_fields)
    stats["schema"] = schema_check
    if not schema_check["schema_valid"]:
        logger.warning(f"Schema issues: missing={schema_check['missing_fields']}, extra={schema_check['extra_fields']}")
    
    # 3. Duplicate check
    logger.info("Checking for duplicates...")
    dup_check = check_duplicates(FINAL_CSV)
    stats["duplicates"] = dup_check
    if dup_check["duplicates"] > 0:
        logger.warning(f"Found {dup_check['duplicates']} duplicate rows")
    
    # 4. Label distribution
    logger.info("Computing label distribution...")
    label_dist = compute_label_distribution(FINAL_CSV)
    stats["label_distribution"] = label_dist
    
    # 5. File stats
    file_hash = compute_file_hash(FINAL_CSV)
    file_size = FINAL_CSV.stat().st_size
    row_count = count_csv_rows(FINAL_CSV)
    
    stats["file"] = {
        "path": str(FINAL_CSV),
        "sha256": file_hash,
        "size_bytes": file_size,
        "row_count": row_count
    }
    
    # Save stats
    with open(STATS_JSON, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # Save checkpoint
    save_checkpoint_meta(
        CHECKPOINT_STEP7,
        input_files={},
        output_files={str(FINAL_CSV): file_hash},
        record_counts={"rows": row_count},
        extra_meta=stats
    )
    
    logger.info("Step 7 complete.")
    logger.info(f"  Total rows: {row_count}")
    logger.info(f"  Unique rows: {dup_check['unique_rows']}")
    logger.info(f"  Duplicates: {dup_check['duplicates']}")
    logger.info(f"  Stats saved to: {STATS_JSON}")
    
    # Print emotion distribution
    logger.info("\nEmotion distribution:")
    for emotion, count in sorted(label_dist["emotion_distribution"].items(), key=lambda x: -x[1]):
        logger.info(f"  {emotion}: {count}")
    
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    
    run_step7(resume=not args.no_resume)



