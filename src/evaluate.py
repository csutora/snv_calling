"""Evaluate a trained model on a new test dataset.

Usage:
    uv run python -m src.evaluate --model output/model.json --data-dir data/test --output predictions.bed

Takes a model path and a directory containing VCF files from the 4 callers,
produces a BED file of predicted true mutations.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.parse_vcf import (
    add_derived_features,
    add_sample_relative_features,
    build_sample_features,
    encode_filter_flags,
    get_feature_columns,
)
from src.train import save_predictions_bed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on test VCF data"
    )
    parser.add_argument(
        "--model", type=Path, required=True,
        help="Path to trained XGBoost model (.json)"
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Directory containing {mutect2,varscan,freebayes,vardict}.vcf.gz (or sample-named VCFs)"
    )
    parser.add_argument(
        "--sample-name", type=str, default="test",
        help="Sample name for VCF path resolution (default: 'test')"
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output BED file path for predicted true mutations"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Prediction threshold (default: use stored threshold from sweep)"
    )
    args = parser.parse_args()

    # Load model
    logger.info(f"Loading model from {args.model}")
    model = xgb.Booster()
    model.load_model(str(args.model))

    # Parse VCFs and build features
    logger.info(f"Parsing VCFs from {args.data_dir}")
    # We need to point parse_vcf at the right directory
    # The data_dir parent should contain a folder named sample_name
    data_parent = args.data_dir.parent
    sample_name = args.data_dir.name

    df = build_sample_features(data_parent, sample_name)
    df = add_derived_features(df)
    df = encode_filter_flags(df)
    df = add_sample_relative_features(df)

    # Align features to match model's expected order
    model_features = model.feature_names
    for col in model_features:
        if col not in df.columns:
            df[col] = np.nan
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    dmat = xgb.DMatrix(df[model_features], feature_names=model_features)

    # Predict
    logger.info("Generating predictions")
    probs = model.predict(dmat)

    # Threshold
    threshold = args.threshold
    if threshold is None:
        # Default from our best sweep
        threshold = 0.256
        logger.info(f"Using default threshold: {threshold}")
    else:
        logger.info(f"Using provided threshold: {threshold}")

    predictions = df[["chrom", "pos"]].copy()
    predictions["prob"] = probs
    predictions["prediction"] = (probs >= threshold).astype(int)

    # Save
    save_predictions_bed(predictions, args.output)

    n_pred = predictions["prediction"].sum()
    n_total = len(predictions)
    logger.info(f"Predicted {n_pred} mutations out of {n_total} candidates")
    logger.info(f"Output saved to {args.output}")


if __name__ == "__main__":
    main()
