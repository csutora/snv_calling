"""One-time data preparation: parse all VCFs and save feature matrices to parquet.

Run this once. Everything downstream (training, sweeps, ablations) reads from parquet.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from src.parse_vcf import (
    SAMPLES_STANDARD,
    add_derived_features,
    add_sample_relative_features,
    build_all_features,
    build_sample_features,
    encode_filter_flags,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # --- Training data: syn1-5, real1, real2_part1 ---
    train_path = OUTPUT_DIR / "features_train.parquet"
    logger.info("=== Building training features ===")
    train_samples = SAMPLES_STANDARD + ["real2_part1"]
    df_train = build_all_features(DATA_DIR, train_samples, max_parallel_samples=7)
    df_train.to_parquet(train_path)
    logger.info(f"Saved training features to {train_path}: {df_train.shape}")

    # --- real2_part2 (test) ---
    r2p2_path = OUTPUT_DIR / "features_real2_part2.parquet"
    logger.info("\n=== Building real2_part2 features ===")
    df_r2p2 = build_sample_features(DATA_DIR, "real2_part2")
    df_r2p2 = add_derived_features(df_r2p2)
    df_r2p2 = encode_filter_flags(df_r2p2)
    df_r2p2 = add_sample_relative_features(df_r2p2)
    df_r2p2.to_parquet(r2p2_path)
    logger.info(f"Saved real2_part2 features to {r2p2_path}: {df_r2p2.shape}")

    # --- test folder ---
    test_path = OUTPUT_DIR / "features_test.parquet"
    test_dir = DATA_DIR / "test"
    if test_dir.exists():
        logger.info("\n=== Building test features ===")
        df_test = build_sample_features(DATA_DIR, "test")
        df_test = add_derived_features(df_test)
        df_test = encode_filter_flags(df_test)
        df_test = add_sample_relative_features(df_test)
        df_test.to_parquet(test_path)
        logger.info(f"Saved test features to {test_path}: {df_test.shape}")

    elapsed = time.time() - t0
    logger.info(f"\nDone in {elapsed:.0f}s. All parquet files in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
