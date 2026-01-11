"""
Step 2: Normalize + clean (streaming)
"""
import sys
import csv
import json
import sqlite3
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    NRC_EMOLEX_RO_RAW, BASE_CLEANED, CHECKPOINT_STEP2,
    EMOTION_COLUMNS, POLARITY_COLUMNS, LOG_FILE, INTERIM_DIR
)
from utils import (
    setup_logging, compute_file_hash, save_checkpoint_meta,
    checkpoint_exists, create_dedup_db, count_csv_rows
)


def normalize_word(word: str) -> str:
    """Normalize word: lowercase, preserve diacritics, normalize whitespace."""
    return word.lower().strip()


def normalize_row(row: Dict) -> Dict:
    """Normalize a row from raw EmoLex format."""
    # Check if this is the new format with Romanian Word column
    romanian_word = row.get('Romanian Word', '').strip()
    if romanian_word:
        # New format: has Romanian Word column
        word = normalize_word(romanian_word)
        
        # Create normalized row with all emotion columns
        normalized = {
            'word': word,
            'anger': int(row.get('anger', 0)),
            'anticipation': int(row.get('anticipation', 0)),
            'disgust': int(row.get('disgust', 0)),
            'fear': int(row.get('fear', 0)),
            'joy': int(row.get('joy', 0)),
            'sadness': int(row.get('sadness', 0)),
            'surprise': int(row.get('surprise', 0)),
            'trust': int(row.get('trust', 0)),
            'positive': int(row.get('positive', 0)),
            'negative': int(row.get('negative', 0)),
        }
        return normalized
    
    # Old format: word<TAB>emotion<TAB>association
    word = normalize_word(row.get('word', ''))
    emotion = row.get('emotion', '').lower().strip()
    association = int(row.get('association', 0))
    
    # Create normalized row with all emotion columns
    normalized = {
        'word': word,
        'anger': 0,
        'anticipation': 0,
        'disgust': 0,
        'fear': 0,
        'joy': 0,
        'sadness': 0,
        'surprise': 0,
        'trust': 0,
        'positive': 0,
        'negative': 0,
    }
    
    # Map emotion to column
    emotion_mapping = {
        'anger': 'anger',
        'anticipation': 'anticipation',
        'disgust': 'disgust',
        'fear': 'fear',
        'joy': 'joy',
        'sadness': 'sadness',
        'surprise': 'surprise',
        'trust': 'trust',
    }
    
    if association == 1:
        if emotion in emotion_mapping:
            normalized[emotion_mapping[emotion]] = 1
        elif emotion in ['positive']:
            normalized['positive'] = 1
        elif emotion in ['negative']:
            normalized['negative'] = 1
    
    return normalized


def aggregate_emotions(rows: List[Dict]) -> Dict:
    """Aggregate multiple rows for the same word."""
    if not rows:
        return None
    
    # Start with first row
    aggregated = rows[0].copy()
    
    # OR all emotion flags
    for row in rows[1:]:
        for col in EMOTION_COLUMNS + POLARITY_COLUMNS:
            aggregated[col] = max(aggregated.get(col, 0), row.get(col, 0))
    
    return aggregated


def has_any_emotion(row: Dict) -> bool:
    """Check if row has any emotion or polarity tags."""
    for col in EMOTION_COLUMNS + POLARITY_COLUMNS:
        if row.get(col, 0) == 1:
            return True
    return False


def run_step2(resume: bool = True) -> bool:
    """Run Step 2: Normalize + clean."""
    logger = setup_logging(LOG_FILE)
    logger.info("=" * 60)
    logger.info("Step 2: Normalize + clean")
    logger.info("=" * 60)
    
    # Check checkpoint
    if resume and checkpoint_exists(CHECKPOINT_STEP2):
        logger.info("Step 2 already completed. Skipping.")
        return True
    
    if not NRC_EMOLEX_RO_RAW.exists():
        logger.error(f"Input file not found: {NRC_EMOLEX_RO_RAW}")
        return False
    
    # Use SQLite for deduplication (memory-efficient)
    dedup_db_path = INTERIM_DIR / "dedup_step2.db"
    conn = create_dedup_db(dedup_db_path)
    
    # Process file line by line
    word_groups = defaultdict(list)
    total_rows = 0
    skipped_empty = 0
    
    logger.info(f"Reading from {NRC_EMOLEX_RO_RAW}...")
    
    # Check file format by reading first line
    with open(NRC_EMOLEX_RO_RAW, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        has_header = 'Romanian Word' in first_line or 'English Word' in first_line
        f.seek(0)  # Reset to beginning
    
    if has_header:
        # New format: tab-separated with header, Romanian Word column
        logger.info("Detected format: Tab-separated with Romanian Word column")
        with open(NRC_EMOLEX_RO_RAW, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                total_rows += 1
                normalized = normalize_row(row)
                
                if not normalized['word']:
                    skipped_empty += 1
                    continue
                
                if not has_any_emotion(normalized):
                    skipped_empty += 1
                    continue
            
                word_groups[normalized['word']].append(normalized)
                
                # Process in batches to avoid memory buildup
                if len(word_groups) >= 10000:
                    # Write batch
                    _write_word_groups(word_groups, BASE_CLEANED, conn)
                    word_groups.clear()
    else:
        # Old format: word<TAB>emotion<TAB>association (no header)
        logger.info("Detected format: word<TAB>emotion<TAB>association (no header)")
        with open(NRC_EMOLEX_RO_RAW, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) < 3:
                    continue
                
                row = {
                    'word': parts[0],
                    'emotion': parts[1],
                    'association': parts[2]
                }
                
                total_rows += 1
                normalized = normalize_row(row)
                
                if not normalized['word']:
                    skipped_empty += 1
                    continue
                
                if not has_any_emotion(normalized):
                    skipped_empty += 1
                    continue
                
                word_groups[normalized['word']].append(normalized)
                
                # Process in batches to avoid memory buildup
                if len(word_groups) >= 10000:
                    # Write batch
                    _write_word_groups(word_groups, BASE_CLEANED, conn)
                    word_groups.clear()
    
    # Write remaining
    if word_groups:
        _write_word_groups(word_groups, BASE_CLEANED, conn)
    
    conn.close()
    dedup_db_path.unlink()  # Clean up temp DB
    
    # Ensure output file exists (even if empty)
    if not BASE_CLEANED.exists():
        # Create empty file with header
        from config import EMOTION_COLUMNS, POLARITY_COLUMNS
        fieldnames = ['word'] + EMOTION_COLUMNS + POLARITY_COLUMNS
        with open(BASE_CLEANED, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    # Count output
    output_count = count_csv_rows(BASE_CLEANED)
    
    # Compute hashes
    input_hash = compute_file_hash(NRC_EMOLEX_RO_RAW)
    output_hash = compute_file_hash(BASE_CLEANED) if BASE_CLEANED.exists() else ""
    
    # Save checkpoint
    save_checkpoint_meta(
        CHECKPOINT_STEP2,
        input_files={str(NRC_EMOLEX_RO_RAW): input_hash},
        output_files={str(BASE_CLEANED): output_hash},
        record_counts={
            "input": total_rows,
            "output": output_count,
            "skipped": skipped_empty
        }
    )
    
    logger.info(f"Step 2 complete.")
    logger.info(f"  Input rows: {total_rows}")
    logger.info(f"  Output rows: {output_count}")
    logger.info(f"  Skipped: {skipped_empty}")
    logger.info(f"  Output: {BASE_CLEANED}")
    
    return True


def _write_word_groups(word_groups: Dict, output_path: Path, conn):
    """Write aggregated word groups to CSV and dedup DB."""
    from config import EMOTION_COLUMNS, POLARITY_COLUMNS
    
    fieldnames = ['word'] + EMOTION_COLUMNS + POLARITY_COLUMNS
    file_exists = output_path.exists()
    
    with open(output_path, 'a' if file_exists else 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        
        for word, rows in word_groups.items():
            aggregated = aggregate_emotions(rows)
            if not aggregated:
                continue
            
            # Check dedup
            unique_key = _get_unique_key(aggregated)
            try:
                conn.execute(
                    "INSERT INTO entries (unique_key, word, row_data) VALUES (?, ?, ?)",
                    (unique_key, word, json.dumps(aggregated, ensure_ascii=False))
                )
                writer.writerow(aggregated)
            except sqlite3.IntegrityError:
                continue  # Duplicate


def _get_unique_key(row: Dict) -> str:
    """Generate unique key for deduplication."""
    from config import EMOTION_COLUMNS, POLARITY_COLUMNS
    word = row.get('word', '').lower().strip()
    tags = "|".join([f"{col}:{row.get(col, 0)}" for col in EMOTION_COLUMNS + POLARITY_COLUMNS])
    return f"{word}|{tags}"




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    
    run_step2(resume=not args.no_resume)

