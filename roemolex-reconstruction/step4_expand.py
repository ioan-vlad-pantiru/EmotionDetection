"""
Step 4: Expand with RoWordNet synonyms (bounded growth + disk dedup)
"""
import sys
import csv
import json
from pathlib import Path
from typing import Dict, List
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MAPPED, EXPANDED, CHECKPOINT_STEP4, SHARDS_DIR, CHECKPOINTS_DIR,
    CHUNK_SIZE, MAX_WORKERS, LOG_FILE, EMOTION_COLUMNS, POLARITY_COLUMNS
)
from utils import (
    setup_logging, compute_file_hash, save_checkpoint_meta,
    checkpoint_exists, stream_csv_chunks, write_csv_chunk,
    external_merge_shards, count_csv_rows, get_unique_key
)
from config import MAX_WORKERS as DEFAULT_MAX_WORKERS


def get_synonyms_from_synset(synset: str) -> List[str]:
    """
    Get synonyms from a RoWordNet synset.
    
    This is a placeholder - in practice, query RoWordNet database.
    """
    # Placeholder: return empty list
    # In real implementation, query RoWordNet for all lemmas in synset
    return []


def process_expansion_shard(shard_data: tuple) -> tuple:
    """Process a shard: expand with synonyms."""
    shard_id, rows = shard_data
    expanded_rows = []
    errors = 0
    
    for row in rows:
        try:
            # Add original row
            expanded_rows.append(row.copy())
            
            # If synset exists, get synonyms
            synset = row.get('synset', '').strip()
            if synset:
                synonyms = get_synonyms_from_synset(synset)
                
                for synonym in synonyms:
                    # Create new row with synonym
                    new_row = row.copy()
                    new_row['word'] = synonym.lower().strip()
                    new_row['provenance'] = 'rownet_synonym'
                    expanded_rows.append(new_row)
        except Exception as e:
            errors += 1
            continue
    
    return (shard_id, expanded_rows, errors)


def run_step4(resume: bool = True) -> bool:
    """Run Step 4: Expand with synonyms."""
    logger = setup_logging(LOG_FILE)
    logger.info("=" * 60)
    logger.info("Step 4: Expand with RoWordNet synonyms")
    logger.info("=" * 60)
    
    if resume and checkpoint_exists(CHECKPOINT_STEP4):
        logger.info("Step 4 already completed. Skipping.")
        return True
    
    if not MAPPED.exists():
        logger.error(f"Input file not found: {MAPPED}")
        return False
    
    # Create shards if needed
    shard_files = sorted(SHARDS_DIR.glob("step4_shard_*.csv"))
    shard_checkpoints = {f.stem.replace('step4_', '') for f in CHECKPOINTS_DIR.glob("step4_shard_*.done")}
    
    if not shard_files:
        logger.info("Creating shards from input file...")
        shard_id = 0
        for chunk in stream_csv_chunks(MAPPED, chunk_size=CHUNK_SIZE):
            shard_path = SHARDS_DIR / f"step4_shard_{shard_id:04d}.csv"
            write_csv_chunk(chunk, shard_path)
            shard_id += 1
        shard_files = sorted(SHARDS_DIR.glob("step4_shard_*.csv"))
    
    # Process shards
    workers = MAX_WORKERS if 'MAX_WORKERS' in globals() else DEFAULT_MAX_WORKERS
    logger.info(f"Processing {len(shard_files)} shards with {workers} workers...")
    
    shard_data_list = []
    for shard_file in shard_files:
        shard_id = shard_file.stem.replace('step4_shard_', '')
        if f"step4_shard_{shard_id}" in shard_checkpoints:
            logger.info(f"  Skipping shard {shard_id} (already processed)")
            continue
        
        rows = []
        with open(shard_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        shard_data_list.append((shard_id, rows))
    
    if shard_data_list:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_expansion_shard, data): data[0] for data in shard_data_list}
            
            for future in as_completed(futures):
                shard_id, expanded_rows, errors = future.result()
                
                output_shard = SHARDS_DIR / f"step4_expanded_shard_{shard_id}.csv"
                fieldnames = list(expanded_rows[0].keys()) if expanded_rows else []
                write_csv_chunk(expanded_rows, output_shard, fieldnames)
                
                checkpoint_path = CHECKPOINTS_DIR / f"step4_shard_{shard_id}.done"
                checkpoint_path.touch()
                
                logger.info(f"  Shard {shard_id}: {len(expanded_rows)} rows ({errors} errors)")
    
    # Merge with deduplication
    logger.info("Merging shards with deduplication...")
    expanded_shards = sorted(SHARDS_DIR.glob("step4_expanded_shard_*.csv"))
    
    if expanded_shards:
        from config import EMOTION_COLUMNS, POLARITY_COLUMNS
        fieldnames = ['word'] + EMOTION_COLUMNS + POLARITY_COLUMNS + ['synset', 'pos', 'sumo', 'provenance']
        # Remove old dedup DB if exists
        dedup_db_path = EXPANDED.parent / "dedup_step4.db"
        if dedup_db_path.exists():
            dedup_db_path.unlink()
        external_merge_shards(
            str(SHARDS_DIR / "step4_expanded_shard_*.csv"),
            EXPANDED,
            fieldnames,
            dedup=True,
            dedup_db=dedup_db_path
        )
    
    # Save checkpoint
    input_hash = compute_file_hash(MAPPED)
    output_hash = compute_file_hash(EXPANDED) if EXPANDED.exists() else ""
    input_count = count_csv_rows(MAPPED)
    output_count = count_csv_rows(EXPANDED)
    
    save_checkpoint_meta(
        CHECKPOINT_STEP4,
        input_files={str(MAPPED): input_hash},
        output_files={str(EXPANDED): output_hash},
        record_counts={
            "input": input_count,
            "output": output_count,
            "expansion_ratio": output_count / input_count if input_count > 0 else 0
        }
    )
    
    logger.info(f"Step 4 complete.")
    logger.info(f"  Input rows: {input_count}")
    logger.info(f"  Output rows: {output_count}")
    logger.info(f"  Expansion ratio: {output_count / input_count if input_count > 0 else 0:.2f}x")
    
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    
    run_step4(resume=not args.no_resume)

