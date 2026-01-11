"""
Utility functions for checkpointing, hashing, and streaming operations.
"""
import hashlib
import json
import csv
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple
from datetime import datetime
import logging


def setup_logging(log_file: Path):
    """Setup logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    if not file_path.exists():
        return ""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_checkpoint_meta(checkpoint_path: Path) -> Optional[Dict]:
    """Load checkpoint metadata JSON."""
    meta_path = checkpoint_path.with_suffix('.meta.json')
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def save_checkpoint_meta(
    checkpoint_path: Path,
    input_files: Dict[str, str],
    output_files: Dict[str, str],
    record_counts: Dict[str, int],
    extra_meta: Optional[Dict] = None
):
    """Save checkpoint metadata."""
    meta = {
        "timestamp": datetime.now().isoformat(),
        "input_files": input_files,  # {filename: hash}
        "output_files": output_files,  # {filename: hash}
        "record_counts": record_counts,  # {filename: count}
        **(extra_meta or {})
    }
    
    meta_path = checkpoint_path.with_suffix('.meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    # Create done marker
    checkpoint_path.touch()


def checkpoint_exists(checkpoint_path: Path, verify_hashes: bool = True) -> bool:
    """Check if checkpoint exists and optionally verify file hashes."""
    if not checkpoint_path.exists():
        return False
    
    if not verify_hashes:
        return True
    
    meta = load_checkpoint_meta(checkpoint_path)
    if meta is None:
        return True  # Done file exists but no meta - assume OK
    
    # Verify output file hashes
    for filename, expected_hash in meta.get("output_files", {}).items():
        file_path = Path(filename)
        if not file_path.exists():
            return False
        actual_hash = compute_file_hash(file_path)
        if actual_hash != expected_hash:
            return False
    
    return True


def stream_csv_chunks(
    csv_path: Path,
    chunk_size: int = 20000,
    skip_header: bool = True
) -> Iterator[List[Dict]]:
    """
    Stream CSV file in chunks without loading entire file into memory.
    
    Yields:
        List of dictionaries (rows) for each chunk
    """
    chunk = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            chunk.append(row)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        
        # Yield remaining chunk
        if chunk:
            yield chunk


def write_csv_chunk(chunk: List[Dict], output_path: Path, fieldnames: Optional[List[str]] = None):
    """Write a chunk of rows to CSV."""
    if not chunk:
        return
    
    file_exists = output_path.exists()
    mode = 'a' if file_exists else 'w'
    
    if fieldnames is None:
        fieldnames = list(chunk[0].keys())
    
    with open(output_path, mode, encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(chunk)


def create_dedup_db(db_path: Path) -> sqlite3.Connection:
    """Create SQLite database for deduplication."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entries (
            unique_key TEXT PRIMARY KEY,
            word TEXT,
            pos TEXT,
            tags_signature TEXT,
            synset TEXT,
            row_data TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_word ON entries(word)")
    return conn


def get_unique_key(row: Dict, include_synset: bool = True) -> str:
    """Generate unique key for a row."""
    word = row.get('word', '').lower().strip()
    pos = row.get('pos', '').strip()
    
    # Create tags signature from emotion columns
    tags = []
    for col in ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust', 'positive', 'negative']:
        val = row.get(col, '0')
        tags.append(f"{col}:{val}")
    tags_signature = "|".join(tags)
    
    synset = row.get('synset', '').strip() if include_synset else ''
    
    return f"{word}|{pos}|{tags_signature}|{synset}"


def external_merge_shards(
    shard_pattern: str,
    output_path: Path,
    fieldnames: List[str],
    dedup: bool = True,
    dedup_db: Optional[Path] = None
):
    """
    Merge multiple shard files into one output file using external merge.
    
    Args:
        shard_pattern: Pattern to match shard files (e.g., "work/shards/shard_*.csv")
        output_path: Output file path
        fieldnames: Column names
        dedup: Whether to deduplicate
        dedup_db: Optional SQLite database path for deduplication
    """
    import glob
    
    shard_files = sorted(glob.glob(shard_pattern))
    
    if not shard_files:
        logging.warning(f"No shard files found matching {shard_pattern}")
        return
    
    conn = None
    if dedup and dedup_db:
        conn = create_dedup_db(dedup_db)
    
    seen_keys = set()
    total_written = 0
    
    with open(output_path, 'w', encoding='utf-8', newline='') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        
        for shard_file in shard_files:
            logging.info(f"Processing shard: {shard_file}")
            with open(shard_file, 'r', encoding='utf-8') as in_f:
                reader = csv.DictReader(in_f)
                for row in reader:
                    should_write = True
                    
                    if dedup:
                        unique_key = get_unique_key(row)
                        if unique_key in seen_keys:
                            should_write = False  # Skip duplicate
                        else:
                            seen_keys.add(unique_key)
                            
                            if conn:
                                # Also store in DB for persistence
                                row_data = json.dumps(row, ensure_ascii=False)
                                try:
                                    conn.execute(
                                        "INSERT INTO entries (unique_key, word, pos, tags_signature, synset, row_data) VALUES (?, ?, ?, ?, ?, ?)",
                                        (unique_key, row.get('word', ''), row.get('pos', ''), 
                                         get_unique_key(row, include_synset=False), row.get('synset', ''), row_data)
                                    )
                                except sqlite3.IntegrityError:
                                    # Already in DB - skip
                                    should_write = False
                    
                    # Write row to output file only if not duplicate
                    if should_write:
                        writer.writerow(row)
                        total_written += 1
    
    if conn:
        conn.commit()
        conn.close()
    
    logging.info(f"Merged {len(shard_files)} shards into {output_path} ({total_written} rows)")


def count_csv_rows(csv_path: Path) -> int:
    """Count rows in CSV file (excluding header)."""
    if not csv_path.exists():
        return 0
    
    count = 0
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for _ in reader:
            count += 1
    return count

