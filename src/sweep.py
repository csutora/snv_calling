"""Optuna hyperparameter sweep with wandb logging.

Supports two training configurations:
  --mode real-only    Train only on real1 + real2_part1
  --mode mask-syn     Train on all data, masking domain-dependent features for synthetic samples
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import wandb
import xgboost as xgb

from src.domain_analysis import get_domain_dependent_features
from src.parse_vcf import get_feature_columns
from src.train import loso_cv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")


def prepare_data(mode: str) -> tuple[pd.DataFrame, list[str]]:
    """Load and prepare data for the given mode."""
    cache_path = OUTPUT_DIR / "features_train.parquet"
    df = pd.read_parquet(cache_path)

    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    if mode == "real-only":
        df = df[df["sample"].isin(["real1", "real2_part1"])].reset_index(drop=True)
        logger.info(f"Real-only: {len(df)} rows, {df['label'].sum()} positive")

    elif mode == "mask-syn":
        dep_features = get_domain_dependent_features(df)
        syn_mask = df["sample"].str.startswith("syn")
        for col in dep_features:
            if col in df.columns:
                df.loc[syn_mask, col] = np.nan
        logger.info(f"Mask-syn: masked {len(dep_features)} features on {syn_mask.sum()} synthetic rows")

    feature_cols = get_feature_columns(df)
    # Replace any remaining inf
    for col in feature_cols:
        if df[col].dtype in (np.float64, np.float32):
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df, feature_cols


def objective(
    trial: optuna.Trial,
    df: pd.DataFrame,
    feature_cols: list[str],
    mode: str,
) -> float:
    """Optuna objective: maximize LOSO CV F1 on real folds."""
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cuda",
        "nthread": -1,
        "seed": 42,
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0, 5.0),
    }

    # Scale pos weight with tunable factor
    n_neg = (df["label"] == 0).sum()
    n_pos = (df["label"] == 1).sum()
    spw_factor = trial.suggest_float("scale_pos_weight_factor", 0.3, 3.0)
    params["scale_pos_weight"] = (n_neg / n_pos) * spw_factor

    cv_results = loso_cv(
        df, feature_cols, params,
        n_estimators=2000,
        early_stopping_rounds=50,
    )

    # Primary metric: F1 on real folds only (what we actually care about)
    real_fold_f1s = [
        f["f1_optimal"] for f in cv_results["folds"]
        if f["sample"].startswith("real")
    ]
    real_mean_f1 = float(np.mean(real_fold_f1s)) if real_fold_f1s else cv_results["mean_f1"]

    # Log everything
    log_data = {
        "trial": trial.number,
        "real_mean_f1": real_mean_f1,
        "cv_mean_f1": cv_results["mean_f1"],
        "cv_global_f1": cv_results["global_f1"],
        "cv_global_threshold": cv_results["global_threshold"],
    }
    for fold in cv_results["folds"]:
        log_data[f"fold_{fold['sample']}_f1"] = fold["f1_optimal"]
    log_data.update(params)
    wandb.log(log_data)

    logger.info(f"  Trial {trial.number}: real_mean_f1={real_mean_f1:.4f}, "
                f"cv_mean_f1={cv_results['mean_f1']:.4f}")

    return real_mean_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["real-only", "mask-syn"], required=True)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    df, feature_cols = prepare_data(args.mode)

    if not args.no_wandb:
        wandb.init(
            project="variant-calling",
            name=f"sweep-{args.mode}",
            config={"mode": args.mode, "n_trials": args.n_trials, "n_features": len(feature_cols)},
        )
    else:
        wandb.init(mode="disabled")

    study = optuna.create_study(
        direction="maximize",
        study_name=f"sweep-{args.mode}",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        lambda trial: objective(trial, df, feature_cols, args.mode),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    logger.info(f"\nBest trial: {study.best_trial.number}")
    logger.info(f"Best real F1: {study.best_value:.4f}")
    logger.info(f"Best params:")
    for k, v in study.best_params.items():
        logger.info(f"  {k}: {v}")

    # Save results
    import json
    results = {
        "mode": args.mode,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial": study.best_trial.number,
    }
    out_path = OUTPUT_DIR / f"sweep_{args.mode.replace('-', '_')}_best.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved best params to {out_path}")

    wandb.log({"best_real_f1": study.best_value, "best_params": study.best_params})
    wandb.finish()


if __name__ == "__main__":
    main()
