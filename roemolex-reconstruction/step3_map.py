"""
Step 3: RoWordNet mapping (chunked + parallel)
"""
import sys
import csv
import json
from pathlib import Path
from typing import Dict, List
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BASE_CLEANED, MAPPED, CHECKPOINT_STEP3, SHARDS_DIR, CHECKPOINTS_DIR,
    CHUNK_SIZE, MAX_WORKERS, LOG_FILE, EMOTION_COLUMNS, POLARITY_COLUMNS
)
from utils import (
    setup_logging, compute_file_hash, save_checkpoint_meta,
    checkpoint_exists, stream_csv_chunks, write_csv_chunk,
    external_merge_shards, count_csv_rows
)
from config import MAX_WORKERS as DEFAULT_MAX_WORKERS


def map_word_to_rownet(word: str) -> Dict:
    """
    Map a word to RoWordNet synset, POS, and SUMO.
    
    This is a placeholder - in practice, you would use RoWordNet API/library.
    """
    # Placeholder implementation
    # In real implementation, query RoWordNet database/API
    
    result = {
        'synset': '',  # e.g., "ROWN-12345-n"
        'pos': '',  # n, v, a, r
        'sumo': '',  # SUMO category if available
    }
    
    # Simple heuristic: try to infer POS from word endings (very basic)
    word_lower = word.lower()
    if word_lower.endswith(('are', 'ere', 'ire', 'ă', 'ează')):
        result['pos'] = 'v'  # verb
    elif word_lower.endswith(('ie', 'ție', 'ție', 'are')):
        result['pos'] = 'n'  # noun
    elif word_lower.endswith(('ic', 'esc', 'os')):
        result['pos'] = 'a'  # adjective
    
    return result


def process_shard(shard_data: tuple) -> tuple:
    """
    Process a shard of rows: map to RoWordNet.
    
    Args:
        shard_data: (shard_id, rows_list)
    
    Returns:
        (shard_id, processed_rows, error_count)
    """
    shard_id, rows = shard_data
    processed = []
    errors = 0
    
    for row in rows:
        try:
            word = row.get('word', '')
            mapped = map_word_to_rownet(word)
            
            # Add mapped fields to row
            new_row = row.copy()
            new_row['synset'] = mapped['synset']
            new_row['pos'] = mapped['pos']
            new_row['sumo'] = mapped['sumo']
            new_row['provenance'] = 'nrc_emolex'
            
            processed.append(new_row)
        except Exception as e:
            errors += 1
            continue
    
    return (shard_id, processed, errors)


def run_step3(resume: bool = True) -> bool:
    """Run Step 3: RoWordNet mapping."""
    logger = setup_logging(LOG_FILE)
    logger.info("=" * 60)
    logger.info("Step 3: RoWordNet mapping (chunked + parallel)")
    logger.info("=" * 60)
    
    # Check checkpoint
    if resume and checkpoint_exists(CHECKPOINT_STEP3):
        logger.info("Step 3 already completed. Skipping.")
        return True
    
    if not BASE_CLEANED.exists():
        logger.error(f"Input file not found: {BASE_CLEANED}")
        return False
    
    # Determine shards to process
    shard_files = sorted(SHARDS_DIR.glob("step3_shard_*.csv"))
    shard_checkpoints = {f.stem.replace('step3_', '') for f in CHECKPOINTS_DIR.glob("step3_shard_*.done")}
    
    # Create shards if needed
    if not shard_files:
        logger.info("Creating shards from input file...")
        shard_id = 0
        for chunk in stream_csv_chunks(BASE_CLEANED, chunk_size=CHUNK_SIZE):
            shard_path = SHARDS_DIR / f"step3_shard_{shard_id:04d}.csv"
            write_csv_chunk(chunk, shard_path)
            shard_id += 1
        shard_files = sorted(SHARDS_DIR.glob("step3_shard_*.csv"))
    
    # Process shards in parallel
    workers = MAX_WORKERS if 'MAX_WORKERS' in globals() else DEFAULT_MAX_WORKERS
    logger.info(f"Processing {len(shard_files)} shards with {workers} workers...")
    
    # Prepare shard data
    shard_data_list = []
    for shard_file in shard_files:
        shard_id = shard_file.stem.replace('step3_shard_', '')
        if f"step3_shard_{shard_id}" in shard_checkpoints:
            logger.info(f"  Skipping shard {shard_id} (already processed)")
            continue
        
        # Read shard
        rows = []
        with open(shard_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        shard_data_list.append((shard_id, rows))
    
    if not shard_data_list:
        logger.info("All shards already processed. Merging...")
    else:
        # Process in parallel
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_shard, data): data[0] for data in shard_data_list}
            
            for future in as_completed(futures):
                shard_id, processed_rows, errors = future.result()
                
                # Write processed shard
                output_shard = SHARDS_DIR / f"step3_mapped_shard_{shard_id}.csv"
                fieldnames = list(processed_rows[0].keys()) if processed_rows else []
                write_csv_chunk(processed_rows, output_shard, fieldnames)
                
                # Mark shard as done
                checkpoint_path = CHECKPOINTS_DIR / f"step3_shard_{shard_id}.done"
                checkpoint_path.touch()
                
                logger.info(f"  Shard {shard_id}: {len(processed_rows)} rows, {errors} errors")
    
    # Merge shards
    logger.info("Merging shards...")
    mapped_shards = sorted(SHARDS_DIR.glob("step3_mapped_shard_*.csv"))
    
    if mapped_shards:
        fieldnames = ['word'] + EMOTION_COLUMNS + POLARITY_COLUMNS + ['synset', 'pos', 'sumo', 'provenance']
        # Remove old dedup DB if exists
        dedup_db_path = MAPPED.parent / "dedup_step3.db"
        if dedup_db_path.exists():
            dedup_db_path.unlink()
        external_merge_shards(
            str(SHARDS_DIR / "step3_mapped_shard_*.csv"),
            MAPPED,
            fieldnames,
            dedup=True,
            dedup_db=dedup_db_path
        )
    
    # Compute hashes and counts
    input_hash = compute_file_hash(BASE_CLEANED)
    output_hash = compute_file_hash(MAPPED) if MAPPED.exists() else ""
    input_count = count_csv_rows(BASE_CLEANED)
    output_count = count_csv_rows(MAPPED)
    
    # Save checkpoint
    save_checkpoint_meta(
        CHECKPOINT_STEP3,
        input_files={str(BASE_CLEANED): input_hash},
        output_files={str(MAPPED): output_hash},
        record_counts={
            "input": input_count,
            "output": output_count,
            "shards_processed": len(mapped_shards)
        }
    )
    
    logger.info(f"Step 3 complete.")
    logger.info(f"  Input rows: {input_count}")
    logger.info(f"  Output rows: {output_count}")
    logger.info(f"  Output: {MAPPED}")
    
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    
    global MAX_WORKERS
    if args.workers:
        MAX_WORKERS = args.workers
    else:
        MAX_WORKERS = DEFAULT_MAX_WORKERS
    
    run_step3(resume=not args.no_resume)

