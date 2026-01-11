# RoEmoLex Reconstruction Pipeline - Complete Explanation

## 🎯 Purpose

Reconstruct a Romanian emotion lexicon following the RoEmoLex methodology, with strict memory constraints (8GB RAM, target <2GB usage) and full restartability.

## 📁 Project Structure

```
roemolex-reconstruction/
├── config.py              # Configuration (paths, constants)
├── utils.py               # Utility functions (checkpointing, hashing, streaming)
├── pipeline.py            # Main pipeline runner
├── step1_acquire.py       # Step 1: Download/locate Romanian EmoLex
├── step2_normalize.py     # Step 2: Normalize + clean (streaming)
├── step3_map.py          # Step 3: RoWordNet mapping (parallel)
├── step4_expand.py       # Step 4: Synonym expansion (parallel)
├── step5_affect.py       # Step 5: WordNet-Affect integration
├── step6_derived.py      # Step 6: Derived emotions (dyads)
├── step7_validate.py     # Step 7: Validation + statistics
├── step8_readme.py       # Step 8: Generate documentation
├── data/
│   ├── raw/              # Input files
│   ├── interim/          # Intermediate processed files
│   └── out/              # Final outputs
├── work/
│   ├── checkpoints/      # Checkpoint markers + metadata
│   └── shards/           # Chunk outputs from parallel workers
└── logs/
    └── run.log           # Pipeline execution log
```

## 🔄 Pipeline Flow

```
Step 1: Acquire
  ↓
  data/raw/nrc_emolex_ro.txt
  ↓
Step 2: Normalize (streaming)
  ↓
  data/interim/base_cleaned.csv
  ↓
Step 3: Map to RoWordNet (parallel chunks)
  ↓
  data/interim/mapped.csv
  ↓
Step 4: Expand synonyms (parallel chunks)
  ↓
  data/interim/expanded.csv
  ↓
Step 5: Add WordNet-Affect (optional)
  ↓
  data/interim/affect_added.csv
  ↓
Step 6: Compute derived emotions
  ↓
  data/out/roemolex_recon.csv
  data/out/roemolex_recon.jsonl
  ↓
Step 7: Validate
  ↓
  data/out/stats.json
  ↓
Step 8: Generate README
  ↓
  data/out/README.md
```

## 🛡️ Memory Management Strategies

### 1. **Streaming Processing**
- Never load entire files into memory
- Process CSV files line-by-line or in chunks
- Use Python's `csv` module (not pandas for large files)

### 2. **Chunked Processing**
- Split large files into shards (20k rows each)
- Process shards independently
- Merge shards using external merge (streaming)

### 3. **Disk-Backed Deduplication**
- Use SQLite for deduplication (not in-memory sets)
- Unique keys stored in database
- Database can be cleaned up after merge

### 4. **Bounded Parallelism**
- Limit workers to 2-4 (prevents memory explosion)
- Each worker processes one shard at a time
- Workers write directly to disk (no shared memory)

### 5. **Checkpointing**
- Write intermediate results frequently
- Each step writes its output before checkpointing
- Can resume from any step

## 🔍 Step-by-Step Details

### Step 1: Acquire Base Lexicon

**Input**: None (downloads/locates)
**Output**: `data/raw/nrc_emolex_ro.txt`

**Process**:
1. Check if file exists locally
2. Try to find in parent project's data directory
3. Try to download from HuggingFace datasets
4. Create placeholder if not found (with warning)

**Memory**: Minimal (just file I/O)

---

### Step 2: Normalize + Clean

**Input**: `data/raw/nrc_emolex_ro.txt`
**Output**: `data/interim/base_cleaned.csv`

**Process** (streaming):
1. Read file line-by-line (not all at once)
2. Normalize each row:
   - Lowercase words
   - Preserve diacritics
   - Map emotions to Plutchik 8
3. Aggregate rows for same word (OR emotion flags)
4. Filter: remove rows with no emotions
5. Deduplicate using SQLite (disk-backed)
6. Write output incrementally (batches of 10k words)

**Memory**: ~10k words in buffer at a time (~1-2MB)

**Key Technique**: 
- Use `defaultdict(list)` to group by word
- Flush to disk when buffer reaches 10k entries
- SQLite ensures no duplicates without loading all into memory

---

### Step 3: RoWordNet Mapping

**Input**: `data/interim/base_cleaned.csv`
**Output**: `data/interim/mapped.csv`

**Process** (chunked + parallel):
1. Split input into shards (20k rows each)
2. For each shard (in parallel):
   - Map each word to RoWordNet synset
   - Assign POS (part of speech)
   - Add SUMO category
   - Write to `work/shards/step3_mapped_shard_XXX.csv`
3. Mark shard as done: `work/checkpoints/step3_shard_XXX.done`
4. Merge all shards (external merge, streaming)
5. Deduplicate during merge

**Memory**: 
- Per worker: 20k rows × ~200 bytes = ~4MB
- Total: 4MB × MAX_WORKERS = ~16MB max

**Resume Logic**:
- Check which shards have `.done` markers
- Skip completed shards
- Only process remaining shards

---

### Step 4: Expand with Synonyms

**Input**: `data/interim/mapped.csv`
**Output**: `data/interim/expanded.csv`

**Process** (similar to Step 3):
1. Split into shards
2. For each shard (parallel):
   - For each entry with synset:
     - Fetch synonyms from RoWordNet
     - Create new row for each synonym (copy emotion tags)
   - Write expanded shard
3. Merge with deduplication

**Memory**: Same as Step 3 (~16MB max)

**Growth Control**:
- Each word expands to ~2-5 synonyms on average
- Bounded by synset size
- Deduplication prevents explosion

---

### Step 5: WordNet-Affect Integration

**Input**: `data/interim/expanded.csv`
**Output**: `data/interim/affect_added.csv`

**Process**:
1. Try to load WordNet-Affect resource
2. If found: map Ekman → Plutchik emotions
3. If not found: copy expanded.csv (pipeline continues)
4. Stream processing (line-by-line)

**Memory**: Minimal (streaming)

**Fault Tolerance**: Pipeline doesn't fail if resource missing

---

### Step 6: Derived Emotions

**Input**: `data/interim/affect_added.csv`
**Output**: `data/out/roemolex_recon.csv`, `data/out/roemolex_recon.jsonl`

**Process** (streaming):
1. Read each row
2. Compute derived emotions (dyads):
   - If `joy=1` AND `trust=1` → `love=1`
   - If `anticipation=1` AND `joy=1` → `optimism=1`
   - etc.
3. Write to CSV and JSONL

**Memory**: One row at a time (~200 bytes)

---

### Step 7: Validation

**Input**: `data/out/roemolex_recon.csv`
**Output**: `data/out/stats.json`

**Checks**:
1. UTF-8 encoding validation
2. Schema validation (all columns present)
3. Duplicate detection
4. Label distribution statistics
5. File hash computation

**Memory**: Streaming (counts, not full file)

---

### Step 8: Generate README

**Input**: Stats, final files
**Output**: `data/out/README.md`

**Process**: Generate markdown documentation

---

## 🔄 Checkpoint System

### Checkpoint Files

Each step creates:
- `work/checkpoints/stepN_NAME.done` - Marker file (empty)
- `work/checkpoints/stepN_NAME.meta.json` - Metadata

### Metadata Structure

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "input_files": {
    "data/interim/input.csv": "sha256_hash"
  },
  "output_files": {
    "data/interim/output.csv": "sha256_hash"
  },
  "record_counts": {
    "input": 10000,
    "output": 9500
  }
}
```

### Resume Logic

```python
if checkpoint_exists(step_checkpoint):
    # Verify file hashes match
    if verify_hashes:
        skip_step()
    else:
        recompute()
else:
    compute()
```

### Shard Checkpoints

For parallel steps (3, 4):
- Each shard has its own checkpoint: `step3_shard_0001.done`
- Resume skips completed shards
- Only processes remaining shards

---

## 💾 Memory Usage Breakdown

**Per Step** (approximate):

- Step 1: <10MB (file I/O)
- Step 2: ~50MB (10k word buffer + SQLite)
- Step 3: ~20MB per worker (20k rows × ~1KB/row)
- Step 4: ~20MB per worker (same)
- Step 5: <10MB (streaming)
- Step 6: <10MB (streaming)
- Step 7: <10MB (streaming)
- Step 8: <5MB (text generation)

**Peak Usage**: ~100-150MB (well under 2GB target)

---

## 🚀 Usage Examples

### Run Full Pipeline

```bash
python pipeline.py
```

### Run Specific Step

```bash
python pipeline.py --step 2
```

### Force Recompute

```bash
python pipeline.py --no-resume
```

### Custom Workers

```bash
python pipeline.py --workers 3
```

### Run Individual Step

```bash
python step2_normalize.py
python step2_normalize.py --no-resume  # Force recompute
```

---

## 🐛 Troubleshooting

### Out of Memory

**Symptoms**: Process killed, "MemoryError"

**Solutions**:
1. Reduce `CHUNK_SIZE` in `config.py` (default: 20000 → try 10000)
2. Reduce `MAX_WORKERS` (default: 2-4 → try 2)
3. Check disk space (intermediate files can be large)

### Pipeline Interrupted

**Solution**: Just re-run! Pipeline automatically resumes from last checkpoint.

```bash
python pipeline.py  # Resumes automatically
```

### Checkpoint Verification Failed

**Symptoms**: Step recomputes even though `.done` exists

**Cause**: Output file hash changed (file modified externally)

**Solution**: 
- Check `work/checkpoints/stepN.meta.json` for expected hashes
- Verify output files haven't been modified
- Use `--no-resume` to force recompute

### Missing Romanian EmoLex

**Symptoms**: Step 1 creates placeholder file

**Solution**:
1. Download Romanian NRC EmoLex
2. Place in `data/raw/nrc_emolex_ro.txt`
3. Re-run Step 1: `python step1_acquire.py --no-resume`

---

## 📊 Output Schema

### CSV Columns

**Base columns**:
- `word`: Romanian word (lowercase, diacritics preserved)

**Emotion columns** (0 or 1):
- `anger`, `anticipation`, `disgust`, `fear`, `joy`, `sadness`, `surprise`, `trust`

**Polarity columns** (0 or 1):
- `positive`, `negative`

**Metadata columns**:
- `synset`: RoWordNet synset ID (e.g., "ROWN-12345-n")
- `pos`: Part of speech (n/v/a/r)
- `sumo`: SUMO category (if available)
- `provenance`: Source (e.g., "nrc_emolex,rownet_synonym")

**Derived emotion columns** (0 or 1):
- `love`, `submission`, `awe`, `disapproval`, `remorse`, `contempt`, `aggressiveness`, `optimism`

---

## 🔐 Deterministic Output

**Ensures**:
- Same input → same output (hash-verified)
- Stable row ordering
- No random elements
- Reproducible results

**How**:
- Sort shards before merging
- Deterministic hashing (SHA256)
- Stable key generation for deduplication

---

## 📝 Key Design Decisions

1. **SQLite for Dedup**: Better than in-memory sets for large datasets
2. **External Merge**: Doesn't require loading all shards
3. **Checkpointing**: Enables resume without losing work
4. **Streaming**: Never loads full files
5. **Bounded Parallelism**: Prevents memory explosion
6. **Idempotent**: Safe to re-run

---

## 🎓 Understanding the Code

### Checkpoint Functions (`utils.py`)

- `checkpoint_exists()`: Check if step completed
- `save_checkpoint_meta()`: Save metadata + create `.done` marker
- `load_checkpoint_meta()`: Load metadata JSON

### Streaming Functions (`utils.py`)

- `stream_csv_chunks()`: Yield chunks of CSV rows
- `write_csv_chunk()`: Append chunk to CSV
- `external_merge_shards()`: Merge shards without loading all

### Deduplication (`utils.py`)

- `create_dedup_db()`: Create SQLite dedup database
- `get_unique_key()`: Generate unique key for row

---

## ✅ Verification Checklist

After running pipeline, verify:

- [ ] `data/out/roemolex_recon.csv` exists
- [ ] `data/out/roemolex_recon.jsonl` exists
- [ ] `data/out/stats.json` exists
- [ ] `data/out/README.md` exists
- [ ] All checkpoints in `work/checkpoints/` have `.done` files
- [ ] Log file `logs/run.log` shows no errors
- [ ] Stats show reasonable row counts and distributions

---

## 🔗 Integration with Emotion Detection Project

The reconstructed lexicon can be used in the emotion detection system:

1. Place `roemolex_recon.csv` in `emotion-detection/data/raw/roemolex.csv`
2. The existing `RoEmoLex` loader will automatically use it
3. Train Romanian models with lexicon features

---

This pipeline is designed to be **robust, memory-efficient, and restartable** - perfect for running on resource-constrained systems while ensuring data quality and reproducibility.



