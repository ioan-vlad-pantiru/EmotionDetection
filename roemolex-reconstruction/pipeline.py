"""
Main pipeline runner for RoEmoLex reconstruction.
"""
import sys
import argparse
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LOG_FILE
from utils import setup_logging

# Import all steps
from step1_acquire import run_step1
from step2_normalize import run_step2
from step3_map import run_step3
from step4_expand import run_step4
from step5_affect import run_step5
from step6_derived import run_step6
from step7_validate import run_step7
from step8_readme import run_step8


def main():
    parser = argparse.ArgumentParser(
        description="RoEmoLex Reconstruction Pipeline"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from checkpoints (force recompute)"
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=range(1, 9),
        help="Run only a specific step (1-8)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override MAX_WORKERS for parallel steps"
    )
    
    args = parser.parse_args()
    
    logger = setup_logging(LOG_FILE)
    logger.info("=" * 60)
    logger.info("RoEmoLex Reconstruction Pipeline")
    logger.info("=" * 60)
    logger.info(f"Resume mode: {not args.no_resume}")
    if args.step:
        logger.info(f"Running only step {args.step}")
    if args.workers:
        logger.info(f"Using {args.workers} workers")
        import config
        config.MAX_WORKERS = args.workers
    
    resume = not args.no_resume
    
    steps = [
        ("Step 1: Acquire base lexicon", run_step1),
        ("Step 2: Normalize + clean", run_step2),
        ("Step 3: RoWordNet mapping", run_step3),
        ("Step 4: Expand with synonyms", run_step4),
        ("Step 5: Integrate WordNet-Affect", run_step5),
        ("Step 6: Derived emotions", run_step6),
        ("Step 7: Final validation", run_step7),
        ("Step 8: Generate README", run_step8),
    ]
    
    if args.step:
        # Run only specified step
        step_idx = args.step - 1
        if 0 <= step_idx < len(steps):
            name, func = steps[step_idx]
            logger.info(f"\nRunning {name}...")
            success = func(resume)
            if not success:
                logger.error(f"{name} failed!")
                return 1
        else:
            logger.error(f"Invalid step number: {args.step}")
            return 1
    else:
        # Run all steps
        for name, func in steps:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running {name}...")
            logger.info(f"{'='*60}")
            
            try:
                success = func(resume)
                if not success:
                    logger.error(f"{name} failed! Stopping pipeline.")
                    return 1
            except Exception as e:
                logger.error(f"{name} failed with error: {e}", exc_info=True)
                return 1
    
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete!")
    logger.info("=" * 60)
    logger.info(f"\nOutput files:")
    logger.info(f"  CSV: data/out/roemolex_recon.csv")
    logger.info(f"  JSONL: data/out/roemolex_recon.jsonl")
    logger.info(f"  Stats: data/out/stats.json")
    logger.info(f"  README: data/out/README.md")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

