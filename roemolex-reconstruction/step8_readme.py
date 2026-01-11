"""
Step 8: Generate README
"""
import sys
from pathlib import Path
import json
import logging
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    README_OUT, STATS_JSON, FINAL_CSV, FINAL_JSONL, CHECKPOINT_STEP8, LOG_FILE
)
from utils import (
    setup_logging, save_checkpoint_meta, checkpoint_exists
)


def run_step8(resume: bool = True) -> bool:
    """Run Step 8: Generate README."""
    logger = setup_logging(LOG_FILE)
    logger.info("=" * 60)
    logger.info("Step 8: Generate README")
    logger.info("=" * 60)
    
    if resume and checkpoint_exists(CHECKPOINT_STEP8):
        logger.info("Step 8 already completed. Skipping.")
        return True
    
    # Load stats if available
    stats = {}
    if STATS_JSON.exists():
        with open(STATS_JSON, 'r', encoding='utf-8') as f:
            stats = json.load(f)
    
    # Generate README
    readme_content = f"""# RoEmoLex-Reconstructed

**⚠️ IMPORTANT**: This is a reconstructed lexicon, not the official RoEmoLex dataset. 
This reconstruction follows the methodology described in the RoEmoLex paper (Briciu & Lupea, 2017) 
but is independently generated and may differ from the original.

## Overview

This Romanian emotion lexicon contains words annotated with Plutchik's 8 basic emotions 
(anger, anticipation, disgust, fear, joy, sadness, surprise, trust) plus polarity (positive/negative) 
and derived emotions (dyads).

## Methodology

### Data Sources

1. **Base Lexicon**: Romanian translation of NRC EmoLex
   - Source: NRC Emotion Lexicon (Romanian translation)
   - Format: Word-level emotion associations

2. **RoWordNet Integration** (if available):
   - Part-of-speech tagging
   - Synset mapping
   - SUMO category assignment

3. **Synonym Expansion**:
   - Expanded using RoWordNet synsets
   - Each synonym inherits emotion tags from its synset

4. **WordNet-Affect Integration** (optional):
   - Ekman emotions mapped to Plutchik emotions
   - Additional emotion annotations

5. **Derived Emotions**:
   - Computed dyads from base emotions
   - Examples: love (joy+trust), optimism (anticipation+joy)

### Processing Pipeline

The reconstruction follows these steps:

1. **Acquisition**: Download/locate Romanian NRC EmoLex
2. **Normalization**: Clean, normalize, deduplicate
3. **RoWordNet Mapping**: Add POS, synset, SUMO
4. **Synonym Expansion**: Expand with RoWordNet synonyms
5. **WordNet-Affect Integration**: Add affect annotations (if available)
6. **Derived Emotions**: Compute emotion dyads
7. **Validation**: Consistency checks and statistics
8. **Documentation**: Generate this README

### Provenance Fields

Each entry includes a `provenance` field indicating the source:
- `nrc_emolex`: Original NRC EmoLex entry
- `rownet_synonym`: Expanded via RoWordNet synonym
- `wn_affect`: Added from WordNet-Affect
- `manual_rules`: Manually assigned (for anticipation/trust)

## File Formats

### CSV Format (`roemolex_recon.csv`)

Columns:
- `word`: Romanian word (lowercase, diacritics preserved)
- `anger`, `anticipation`, `disgust`, `fear`, `joy`, `sadness`, `surprise`, `trust`: Emotion flags (0 or 1)
- `positive`, `negative`: Polarity flags (0 or 1)
- `synset`: RoWordNet synset ID (if available)
- `pos`: Part of speech (n/v/a/r)
- `sumo`: SUMO category (if available)
- `provenance`: Source information
- Derived emotions: `love`, `submission`, `awe`, `disapproval`, `remorse`, `contempt`, `aggressiveness`, `optimism`

### JSONL Format (`roemolex_recon.jsonl`)

One JSON object per line, same fields as CSV.

## Statistics

"""
    
    if stats:
        file_stats = stats.get("file", {})
        readme_content += f"""
- **Total entries**: {file_stats.get('row_count', 'N/A')}
- **File size**: {file_stats.get('size_bytes', 0) / 1024:.2f} KB
- **SHA256**: `{file_stats.get('sha256', 'N/A')}`

### Emotion Distribution

"""
        dist = stats.get("label_distribution", {}).get("emotion_distribution", {})
        for emotion, count in sorted(dist.items(), key=lambda x: -x[1]):
            readme_content += f"- **{emotion}**: {count} entries\n"
    
    readme_content += f"""

## Reproducibility

### Requirements

- Python 3.8+
- Required packages: See `requirements.txt`
- RoWordNet access (optional, for synonym expansion)
- WordNet-Affect resource (optional)

### Running the Pipeline

```bash
# Run all steps
python pipeline.py

# Run specific step
python step1_acquire.py
python step2_normalize.py
# ... etc

# Resume from checkpoint (default)
python pipeline.py --resume

# Force recompute
python pipeline.py --no-resume
```

### Checkpoints

The pipeline uses checkpoint files in `work/checkpoints/` to enable resume:
- Each step writes a `.done` marker when complete
- Metadata (hashes, counts) stored in `.meta.json`
- Re-running skips completed steps if inputs haven't changed

## Limitations

1. **RoWordNet Dependency**: Synonym expansion requires RoWordNet access
2. **WordNet-Affect**: Optional resource, may not be available
3. **Reconstruction Quality**: This is an automated reconstruction and may contain errors
4. **Coverage**: May not match the original RoEmoLex coverage

## Citation

If you use this reconstructed lexicon, please cite:

**Original RoEmoLex**:
```
Briciu, A., & Lupea, M. (2017). RoEmoLex - A Romanian Emotion Lexicon. 
Studia Universitatis Babeș-Bolyai Informatica.
```

**NRC EmoLex**:
```
Mohammad, S. M., & Turney, P. D. (2013). Crowdsourcing a Word-Emotion Association Lexicon. 
Computational Intelligence, 29(3), 436-465.
```

## License

This reconstruction follows the same license as the source materials (NRC EmoLex).
Please refer to the original NRC EmoLex license terms.

## Generated

Generated on: {datetime.now().isoformat()}
Pipeline version: 1.0
"""
    
    # Write README
    with open(README_OUT, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    # Save checkpoint
    save_checkpoint_meta(
        CHECKPOINT_STEP8,
        input_files={},
        output_files={str(README_OUT): ""},
        record_counts={}
    )
    
    logger.info(f"Step 8 complete. README saved to: {README_OUT}")
    
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    
    run_step8(resume=not args.no_resume)



