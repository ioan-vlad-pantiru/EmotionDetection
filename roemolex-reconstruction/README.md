# RoEmoLex Reconstruction Pipeline

A memory-efficient, restartable pipeline for reconstructing a Romanian emotion lexicon inspired by RoEmoLex (Briciu & Lupea, 2017).

## ⚠️ Important Notice

**This is a reconstructed lexicon, not the official RoEmoLex dataset.** This reconstruction follows the methodology described in the RoEmoLex paper but is independently generated and may differ from the original.

## Features

- ✅ **Memory-efficient**: Streaming processing, stays under 1-2GB RAM
- ✅ **Restartable**: Resume from checkpoints after interruption
- ✅ **Parallel processing**: Bounded parallelism (2-4 workers)
- ✅ **Fault-tolerant**: Checkpointing at each step
- ✅ **Idempotent**: Re-running doesn't duplicate output

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python pipeline.py

# Run specific step
python pipeline.py --step 1

# Force recompute (ignore checkpoints)
python pipeline.py --no-resume

# Use custom worker count
python pipeline.py --workers 3
```

## Pipeline Steps

1. **Step 1**: Acquire base lexicon (Romanian NRC EmoLex)
2. **Step 2**: Normalize + clean (streaming deduplication)
3. **Step 3**: RoWordNet mapping (chunked + parallel)
4. **Step 4**: Expand with synonyms (bounded growth + disk dedup)
5. **Step 5**: Integrate WordNet-Affect (optional)
6. **Step 6**: Derived emotions (dyads)
7. **Step 7**: Final validation
8. **Step 8**: Generate README

## Output Files

All outputs are in `data/out/`:

- `roemolex_recon.csv`: Main lexicon in CSV format
- `roemolex_recon.jsonl`: Same data in JSONL format
- `stats.json`: Statistics and validation results
- `README.md`: Detailed documentation

## Checkpoints

Checkpoints are stored in `work/checkpoints/`:
- `.done` files mark completed steps
- `.meta.json` files contain hashes and metadata
- Pipeline automatically resumes from last checkpoint

## Memory Management

- Chunk size: 20,000 rows per shard
- Max workers: 2-4 (configurable)
- SQLite used for deduplication (disk-backed)
- Streaming CSV processing (no full-file loading)

## Requirements

- Python 3.8+
- 8GB RAM (target: <2GB usage)
- Disk space: ~500MB for intermediate files
- RoWordNet access (optional, for synonym expansion)

## Troubleshooting

**Out of memory?**
- Reduce `CHUNK_SIZE` in `config.py`
- Reduce `MAX_WORKERS` in `config.py`

**Pipeline interrupted?**
- Just re-run: `python pipeline.py` (resumes automatically)
- Check `logs/run.log` for details

**Missing Romanian EmoLex?**
- Place Romanian NRC EmoLex file in `data/raw/nrc_emolex_ro.txt`
- Or download from HuggingFace datasets

## License

Follows the same license as source materials (NRC EmoLex).



