# Quick Start Guide - Running the Pipeline

## Step 1: Activate Virtual Environment

```bash
cd /Users/Andreea/Documents/NLP/emotion-detection
source .venv/bin/activate
```

## Step 2: Install Dependencies

```bash
cd roemolex-reconstruction
pip install -r requirements.txt
```

## Step 3: Run the Pipeline

### Option A: Run Everything (Recommended)

```bash
python pipeline.py
```

This will:
- Run all 8 steps sequentially
- Automatically resume from checkpoints if interrupted
- Show progress in console and `logs/run.log`

### Option B: Run Specific Step

```bash
# Run only step 1 (acquire lexicon)
python pipeline.py --step 1

# Run only step 2 (normalize)
python pipeline.py --step 2

# etc.
```

### Option C: Run Individual Step Scripts

```bash
# Run step 1 directly
python step1_acquire.py

# Run step 2 directly
python step2_normalize.py

# Force recompute (ignore checkpoint)
python step2_normalize.py --no-resume
```

## Step 4: Check Results

After running, check the outputs:

```bash
# View final lexicon
head -20 data/out/roemolex_recon.csv

# View statistics
cat data/out/stats.json

# View logs
tail -50 logs/run.log
```

## Common Commands

```bash
# Run with custom worker count (for parallel steps)
python pipeline.py --workers 3

# Force recompute everything (ignore all checkpoints)
python pipeline.py --no-resume

# Run from step 3 onwards
python pipeline.py --step 3
# Then run remaining steps
python pipeline.py --step 4
# etc.
```

## What to Expect

1. **Step 1**: Will try to find/download Romanian EmoLex
   - If not found, creates placeholder file with instructions
   - Check `data/raw/nrc_emolex_ro.txt`

2. **Step 2**: Normalizes and cleans the lexicon
   - Output: `data/interim/base_cleaned.csv`

3. **Steps 3-4**: RoWordNet mapping and expansion
   - These use placeholder functions (you'll need to integrate RoWordNet API)
   - Will still produce output files

4. **Steps 5-8**: Continue processing and generate final outputs

## Troubleshooting

**If Step 1 fails to find Romanian EmoLex:**
```bash
# Manually place the file here:
# data/raw/nrc_emolex_ro.txt

# Then re-run step 1
python step1_acquire.py --no-resume
```

**If pipeline gets interrupted:**
```bash
# Just re-run - it will resume automatically
python pipeline.py
```

**Check progress:**
```bash
# See which steps completed
ls -la work/checkpoints/*.done

# View detailed log
tail -f logs/run.log
```

## Output Files

After successful completion, you'll have:

- `data/out/roemolex_recon.csv` - Main lexicon (CSV)
- `data/out/roemolex_recon.jsonl` - Same lexicon (JSONL)
- `data/out/stats.json` - Statistics and validation
- `data/out/README.md` - Generated documentation



