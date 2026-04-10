"""Training pipeline: load precomputed features, train, evaluate, and predict.

Assumes `python -m src.prepare_data` has already been run.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
import xgboost as xgb

from src.parse_vcf import get_feature_columns
from src.train import (
    evaluate_baselines,
    find_optimal_threshold,
    finetune_model,
    loso_cv,
    per_sample_f1,
    predict,
    save_predictions_bed,
    train_final_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")


def load_features(path: Path) -> pd.DataFrame:
    """Load parquet and coerce booleans to int for XGBoost."""
    df = pd.read_parquet(path)
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
    return df


def main():
    parser = argparse.ArgumentParser(description="Variant calling meta-learner pipeline")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--model-path", type=Path, default=OUTPUT_DIR / "model.json")
    parser.add_argument("--real-weight", type=float, default=1.0,
                        help="Weight multiplier for real samples during training")
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--no-rank-features", action="store_true",
                        help="Ablation: drop sample-relative rank features")
    parser.add_argument("--no-vaf-ratio", action="store_true",
                        help="Ablation: drop VAF ratio/concordance features")
    parser.add_argument("--real-only", action="store_true",
                        help="Ablation: train only on real1 + real2_part1")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name for this run (used in wandb)")
    parser.add_argument("--finetune", action="store_true",
                        help="Finetune: train on all data, then continue boosting on real-only")
    parser.add_argument("--stacked", action="store_true",
                        help="Stacked: all-data OOF preds as feature for real-only model")
    parser.add_argument("--mask-syn-raw", action="store_true",
                        help="Mask domain-dependent raw features to NaN for synthetic samples")
    parser.add_argument("--params-file", type=Path, default=None,
                        help="JSON file with sweep best params (overrides defaults)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load precomputed features ---
    train_path = OUTPUT_DIR / "features_train.parquet"
    if not train_path.exists():
        logger.error(f"{train_path} not found. Run `python -m src.prepare_data` first.")
        return

    logger.info("=== Loading features ===")
    df = load_features(train_path)

    # Mask domain-dependent features for synthetic samples
    if args.mask_syn_raw:
        from src.domain_analysis import get_domain_dependent_features
        dep_features = get_domain_dependent_features(df)
        syn_mask = df["sample"].str.startswith("syn")
        n_masked = syn_mask.sum()
        for col in dep_features:
            if col in df.columns:
                df.loc[syn_mask, col] = np.nan
        logger.info(f"  Masked {len(dep_features)} domain-dependent features "
                     f"to NaN on {n_masked} synthetic rows")

    # Stacked mode: train all-data model first, add its predictions as a feature,
    # then train real-only with that extra feature
    if args.stacked:
        logger.info("\n=== Stage 1: All-data model for stacking ===")
        s1_feature_cols = get_feature_columns(df)
        # Drop feature groups if requested (apply same ablations to stage 1)
        if args.no_rank_features:
            s1_feature_cols = [c for c in s1_feature_cols if not c.endswith("_rank")]
        if args.no_vaf_ratio:
            vaf_ratio_cols = {"af_max_min_ratio", "af_pairwise_mad"}
            vaf_ratio_cols |= {f"{c}_af_dev" for c in ["mutect2", "varscan", "freebayes", "vardict"]}
            s1_feature_cols = [c for c in s1_feature_cols if c not in vaf_ratio_cols]

        for col in s1_feature_cols:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)

        n_neg = (df["label"] == 0).sum()
        n_pos = (df["label"] == 1).sum()
        s1_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 5, "lambda": 1.0, "alpha": 0.1,
            "scale_pos_weight": n_neg / n_pos,
            "tree_method": "hist", "device": "cuda", "nthread": -1, "seed": 42,
        }

        # LOSO CV to get honest OOF predictions for all samples
        s1_cv = loso_cv(df, s1_feature_cols, s1_params, n_estimators=2000, early_stopping_rounds=50)
        df["stage1_pred"] = s1_cv["oof_preds"]
        logger.info(f"  Stage 1 LOSO global F1: {s1_cv['global_f1']:.4f}")

        # Train stage-1 final model on all data (for test-time predictions)
        s1_best_iters = [f["best_iteration"] for f in s1_cv["folds"]]
        s1_final = train_final_model(df, s1_feature_cols, s1_params,
                                     n_estimators=int(np.median(s1_best_iters)) + 50)

        # Generate stage-1 predictions for test sets
        for test_name in ["real2_part2", "test"]:
            test_path = OUTPUT_DIR / f"features_{test_name}.parquet"
            if test_path.exists():
                df_t = load_features(test_path)
                for col in s1_feature_cols:
                    if col not in df_t.columns:
                        df_t[col] = np.nan
                dmat = xgb.DMatrix(df_t[s1_feature_cols], feature_names=s1_feature_cols)
                df_t["stage1_pred"] = s1_final.predict(dmat)
                # Re-save with stage1_pred column
                df_t.to_parquet(OUTPUT_DIR / f"features_{test_name}_stacked.parquet")
                logger.info(f"  Stage 1 predictions for {test_name}: mean={df_t['stage1_pred'].mean():.3f}")

        # Now filter to real-only for stage 2
        df = df[df["sample"].isin(["real1", "real2_part1"])].reset_index(drop=True)
        logger.info(f"  Stage 2: real-only training with stage1_pred, {len(df)} rows")

    # Ablation: real-only training (without stacking)
    elif args.real_only:
        df = df[df["sample"].isin(["real1", "real2_part1"])].reset_index(drop=True)
        logger.info(f"  ABLATION: real-only training, kept {len(df)} rows")

    feature_cols = get_feature_columns(df)

    # Ablation: drop feature groups
    if args.no_rank_features:
        feature_cols = [c for c in feature_cols if not c.endswith("_rank")]
        logger.info(f"  ABLATION: dropped rank features, {len(feature_cols)} features remain")
    if args.no_vaf_ratio:
        vaf_ratio_cols = {"af_max_min_ratio", "af_pairwise_mad"}
        vaf_ratio_cols |= {f"{c}_af_dev" for c in ["mutect2", "varscan", "freebayes", "vardict"]}
        feature_cols = [c for c in feature_cols if c not in vaf_ratio_cols]
        logger.info(f"  ABLATION: dropped VAF ratio features, {len(feature_cols)} features remain")

    logger.info(f"Loaded {len(df)} rows, {df['label'].sum()} positive, {len(feature_cols)} features")

    # --- Baselines ---
    logger.info("\n=== Baselines ===")
    baselines = evaluate_baselines(df)
    for k, v in baselines.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    for sample in df["sample"].unique():
        mask = df["sample"] == sample
        sb = evaluate_baselines(df[mask])
        logger.info(f"  {sample}: 2+callers F1={sb['baseline_2plus_f1']:.4f}, "
                     f"best2 F1={sb['baseline_best2_f1']:.4f}")

    # --- XGBoost LOSO CV ---
    logger.info("\n=== LOSO Cross-Validation ===")

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "lambda": 1.0,
        "alpha": 0.1,
        "tree_method": "hist",
        "device": "cuda",
        "nthread": -1,
        "seed": 42,
    }

    n_neg = (df["label"] == 0).sum()
    n_pos = (df["label"] == 1).sum()
    base_spw = n_neg / n_pos

    # Override with sweep best params if provided
    if args.params_file and args.params_file.exists():
        import json
        with open(args.params_file) as f:
            sweep_data = json.load(f)
        best = sweep_data["best_params"]
        for k in ("max_depth", "learning_rate", "subsample", "colsample_bytree",
                   "min_child_weight", "lambda", "alpha", "gamma"):
            if k in best:
                xgb_params[k] = best[k]
        spw_factor = best.get("scale_pos_weight_factor", 1.0)
        xgb_params["scale_pos_weight"] = base_spw * spw_factor
        logger.info(f"  Loaded sweep params from {args.params_file} (spw_factor={spw_factor:.3f})")
    else:
        xgb_params["scale_pos_weight"] = base_spw
    logger.info(f"  Class balance: {n_pos} pos / {n_neg} neg, "
                f"scale_pos_weight={xgb_params['scale_pos_weight']:.2f}")

    sample_weights = None
    if args.real_weight != 1.0:
        sample_weights = {
            s: args.real_weight if s.startswith("real") else 1.0
            for s in df["sample"].unique()
        }
        logger.info(f"  Sample weights: {sample_weights}")

    if not args.no_wandb:
        wandb.init(
            project="variant-calling",
            name=args.run_name,
            config={
                "xgb_params": xgb_params,
                "real_weight": args.real_weight,
                "n_features": len(feature_cols),
                "real_only": args.real_only,
                "no_rank_features": args.no_rank_features,
                "no_vaf_ratio": args.no_vaf_ratio,
                "finetune": args.finetune,
                "stacked": args.stacked,
                "mask_syn_raw": args.mask_syn_raw,
            },
        )

    cv_results = loso_cv(
        df, feature_cols, xgb_params,
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping,
        sample_weights=sample_weights,
    )

    if not args.no_wandb:
        wandb.log({
            "cv_mean_f1": cv_results["mean_f1"],
            "cv_std_f1": cv_results["std_f1"],
            "cv_global_f1": cv_results["global_f1"],
            "cv_global_threshold": cv_results["global_threshold"],
        })
        for fold in cv_results["folds"]:
            wandb.log({
                f"fold_{fold['sample']}_f1": fold["f1_optimal"],
                f"fold_{fold['sample']}_threshold": fold["best_threshold"],
            })

    # --- Train final model ---
    logger.info("\n=== Training final model ===")
    best_iters = [f["best_iteration"] for f in cv_results["folds"]]
    final_n = int(np.median(best_iters)) + 50
    logger.info(f"  Using {final_n} estimators (median CV best + 50)")

    final_model = train_final_model(
        df, feature_cols, xgb_params,
        n_estimators=final_n,
        sample_weights=sample_weights,
    )
    final_model.save_model(str(args.model_path))
    logger.info(f"  Saved model to {args.model_path}")

    # --- Finetune on real data ---
    if args.finetune:
        logger.info("\n=== Finetuning on real data ===")
        # Load full training data for finetuning (even if --real-only was set for base)
        df_full = load_features(train_path)
        for col in feature_cols:
            if col not in df_full.columns:
                df_full[col] = np.nan
        final_model = finetune_model(
            final_model, df_full, feature_cols, xgb_params,
        )
        ft_path = args.model_path.with_stem(args.model_path.stem + "_finetuned")
        final_model.save_model(str(ft_path))
        logger.info(f"  Saved finetuned model to {ft_path}")

    # --- Training predictions ---
    logger.info("\n=== Training predictions ===")
    if args.finetune:
        # Re-tune threshold on real data using the finetuned model
        df_real = df[df["sample"].isin(["real1", "real2_part1"])].reset_index(drop=True) if not args.real_only else df
        dmat = xgb.DMatrix(df_real[feature_cols], feature_names=feature_cols)
        real_probs = final_model.predict(dmat)
        threshold, _ = find_optimal_threshold(df_real["label"].values, real_probs)
        logger.info(f"  Threshold (re-tuned on real data): {threshold:.4f}")
    else:
        threshold = cv_results["global_threshold"]
        logger.info(f"  Threshold: {threshold:.4f}")

    train_preds = predict(final_model, df, feature_cols, threshold)
    train_metrics = per_sample_f1(df, train_preds["prediction"].values)
    for sample, m in train_metrics.items():
        logger.info(f"  {sample}: F1={m['f1']:.4f}, P={m['precision']:.4f}, "
                     f"R={m['recall']:.4f} ({m['n_pred']} pred / {m['n_true']} true)")

    save_predictions_bed(train_preds, OUTPUT_DIR / "train_predictions.bed")
    for sample in df["sample"].unique():
        mask = df["sample"] == sample
        save_predictions_bed(train_preds[mask.values], OUTPUT_DIR / f"{sample}_predictions.bed")

    # --- Test predictions ---
    logger.info("\n=== Test predictions ===")
    for test_name in ["real2_part2", "test"]:
        # Use stacked features if available (from --stacked mode)
        stacked_path = OUTPUT_DIR / f"features_{test_name}_stacked.parquet"
        plain_path = OUTPUT_DIR / f"features_{test_name}.parquet"
        test_path = stacked_path if args.stacked and stacked_path.exists() else plain_path

        if not test_path.exists():
            logger.warning(f"  {test_path} not found, skipping")
            continue

        df_test = load_features(test_path)
        for col in feature_cols:
            if col not in df_test.columns:
                df_test[col] = np.nan

        preds = predict(final_model, df_test, feature_cols, threshold)
        save_predictions_bed(preds, OUTPUT_DIR / f"{test_name}_predictions.bed")
        logger.info(f"  {test_name}: {preds['prediction'].sum()} predicted mutations")

    # --- Feature importance ---
    logger.info("\n=== Top 20 features (gain) ===")
    importance = final_model.get_score(importance_type="gain")
    for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]:
        logger.info(f"  {feat}: {imp:.1f}")

    if not args.no_wandb:
        table = wandb.Table(columns=["feature", "importance"])
        for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            table.add_data(feat, imp)
        wandb.log({"feature_importance": table})
        for sample, m in train_metrics.items():
            wandb.log({f"train_{sample}_f1": m["f1"]})
        wandb.finish()

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
