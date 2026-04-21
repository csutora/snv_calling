"""Training pipeline: XGBoost with LOSO CV, threshold tuning, and wandb."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Find the threshold that maximizes F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns n+1 precision/recall values, n thresholds
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


# ---------------------------------------------------------------------------
# Leave-One-Sample-Out Cross-Validation
# ---------------------------------------------------------------------------

def loso_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    xgb_params: dict,
    n_estimators: int = 1000,
    early_stopping_rounds: int = 50,
    sample_weights: dict[str, float] | None = None,
) -> dict:
    """Run Leave-One-Sample-Out cross-validation.

    Returns per-fold and aggregate metrics, plus out-of-fold predictions.
    """
    samples = df["sample"].unique()
    results = []
    oof_preds = np.full(len(df), np.nan)

    for fold_sample in samples:
        train_mask = df["sample"] != fold_sample
        val_mask = df["sample"] == fold_sample

        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, "label"]
        X_val = df.loc[val_mask, feature_cols]
        y_val = df.loc[val_mask, "label"]

        # Sample weights
        weights = None
        if sample_weights:
            weights = df.loc[train_mask, "sample"].map(sample_weights).values

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights, feature_names=feature_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dval, "val")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )

        y_prob = model.predict(dval)
        oof_preds[val_mask.values] = y_prob

        # Optimal threshold on this fold
        threshold, best_f1 = find_optimal_threshold(y_val.values, y_prob)

        # Also compute F1 at 0.5 for comparison
        f1_05 = f1_score(y_val, (y_prob >= 0.5).astype(int))

        prec = precision_score(y_val, (y_prob >= threshold).astype(int), zero_division=0)
        rec = recall_score(y_val, (y_prob >= threshold).astype(int), zero_division=0)

        fold_result = {
            "sample": fold_sample,
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "n_pos": int(y_val.sum()),
            "best_threshold": threshold,
            "f1_optimal": best_f1,
            "f1_0.5": f1_05,
            "precision": prec,
            "recall": rec,
            "best_iteration": model.best_iteration,
        }
        results.append(fold_result)
        logger.info(f"  Fold {fold_sample}: F1={best_f1:.4f} (threshold={threshold:.3f}), "
                     f"F1@0.5={f1_05:.4f}, P={prec:.4f}, R={rec:.4f}")

    # Aggregate
    fold_f1s = [r["f1_optimal"] for r in results]
    fold_thresholds = [r["best_threshold"] for r in results]

    # Global threshold from OOF predictions
    valid_mask = ~np.isnan(oof_preds)
    global_threshold, global_f1 = find_optimal_threshold(
        df.loc[valid_mask, "label"].values,
        oof_preds[valid_mask],
    )

    # Real threshold: mean of per-fold optimal thresholds on real samples
    real_folds = [r for r in results if r["sample"].startswith("real")]
    if real_folds:
        real_threshold = float(np.mean([r["best_threshold"] for r in real_folds]))
        # Compute F1 at this threshold on real OOF predictions
        real_fold_mask = df["sample"].str.startswith("real") & valid_mask
        real_preds_at_thresh = (oof_preds[real_fold_mask.values] >= real_threshold).astype(int)
        real_f1 = f1_score(df.loc[real_fold_mask, "label"].values, real_preds_at_thresh, zero_division=0)
    else:
        real_threshold, real_f1 = global_threshold, global_f1

    summary = {
        "folds": results,
        "mean_f1": float(np.mean(fold_f1s)),
        "std_f1": float(np.std(fold_f1s)),
        "mean_threshold": float(np.mean(fold_thresholds)),
        "global_threshold": global_threshold,
        "global_f1": global_f1,
        "real_threshold": real_threshold,
        "real_f1": real_f1,
        "oof_preds": oof_preds,
    }

    logger.info(f"  LOSO CV: mean F1={summary['mean_f1']:.4f} +/- {summary['std_f1']:.4f}, "
                f"global F1={global_f1:.4f} (threshold={global_threshold:.3f}), "
                f"real F1={real_f1:.4f} (threshold={real_threshold:.3f})")

    return summary


# ---------------------------------------------------------------------------
# Train final model on all data
# ---------------------------------------------------------------------------

def train_final_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    xgb_params: dict,
    n_estimators: int = 1000,
    sample_weights: dict[str, float] | None = None,
) -> xgb.Booster:
    """Train the final model on all labeled data."""
    X = df[feature_cols]
    y = df["label"]

    weights = None
    if sample_weights:
        weights = df["sample"].map(sample_weights).values

    dtrain = xgb.DMatrix(X, label=y, weight=weights, feature_names=feature_cols)

    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=n_estimators,
        verbose_eval=False,
    )

    return model


# ---------------------------------------------------------------------------
# Finetune: continue boosting on real-only data
# ---------------------------------------------------------------------------

def finetune_model(
    base_model: xgb.Booster,
    df: pd.DataFrame,
    feature_cols: list[str],
    xgb_params: dict,
    n_estimators: int = 200,
    early_stopping_rounds: int = 30,
) -> xgb.Booster:
    """Continue boosting an existing model on real-only data.

    Uses LOSO on the real samples for early stopping to avoid overfitting.
    """
    real_samples = [s for s in df["sample"].unique() if s.startswith("real")]
    df_real = df[df["sample"].isin(real_samples)].reset_index(drop=True)

    logger.info(f"  Finetuning on {len(df_real)} real-data rows "
                f"({df_real['label'].sum()} positive) from {real_samples}")

    # Use one real sample as validation for early stopping
    # Train on the larger one, validate on the smaller
    sample_sizes = df_real.groupby("sample").size()
    val_sample = sample_sizes.idxmin()
    train_mask = df_real["sample"] != val_sample
    val_mask = df_real["sample"] == val_sample

    X_train = df_real.loc[train_mask, feature_cols]
    y_train = df_real.loc[train_mask, "label"]
    X_val = df_real.loc[val_mask, feature_cols]
    y_val = df_real.loc[val_mask, "label"]

    # Recompute scale_pos_weight for real data
    ft_params = xgb_params.copy()
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    ft_params["scale_pos_weight"] = n_neg / n_pos
    # Lower learning rate for finetuning
    ft_params["learning_rate"] = ft_params.get("learning_rate", 0.05) * 0.5

    logger.info(f"  Finetune LR: {ft_params['learning_rate']:.4f}, "
                f"scale_pos_weight: {ft_params['scale_pos_weight']:.2f}, "
                f"val_sample: {val_sample}")

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

    finetuned = xgb.train(
        ft_params,
        dtrain,
        num_boost_round=n_estimators,
        evals=[(dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        xgb_model=base_model,
        verbose_eval=False,
    )

    logger.info(f"  Finetuned: added {finetuned.best_iteration} trees "
                f"(total: {finetuned.num_boosted_rounds()})")

    return finetuned


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def predict(
    model: xgb.Booster,
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float,
) -> pd.DataFrame:
    """Generate predictions and return a DataFrame with chrom, pos, prediction."""
    X = df[feature_cols]
    dmat = xgb.DMatrix(X, feature_names=feature_cols)
    y_prob = model.predict(dmat)

    result = df[["chrom", "pos"]].copy()
    result["prob"] = y_prob
    result["prediction"] = (y_prob >= threshold).astype(int)
    return result


# ---------------------------------------------------------------------------
# Evaluation against baselines
# ---------------------------------------------------------------------------

def evaluate_baselines(df: pd.DataFrame) -> dict:
    """Compute F1 for the two required baselines plus individual callers."""
    results = {}
    y_true = df["label"].values

    # Individual caller performance
    callers = ["mutect2", "varscan", "freebayes", "vardict"]
    for caller in callers:
        y_pred = df[f"{caller}_pass"].astype(int).values
        results[f"{caller}_f1"] = f1_score(y_true, y_pred, zero_division=0)
        results[f"{caller}_precision"] = precision_score(y_true, y_pred, zero_division=0)
        results[f"{caller}_recall"] = recall_score(y_true, y_pred, zero_division=0)

    # Baseline 1: >= 2 callers agree
    n_pass = df[[f"{c}_pass" for c in callers]].sum(axis=1)
    y_pred_2plus = (n_pass >= 2).astype(int).values
    results["baseline_2plus_f1"] = f1_score(y_true, y_pred_2plus, zero_division=0)
    results["baseline_2plus_precision"] = precision_score(y_true, y_pred_2plus, zero_division=0)
    results["baseline_2plus_recall"] = recall_score(y_true, y_pred_2plus, zero_division=0)

    # Baseline 2: Intersection of 2 best callers
    caller_f1s = {c: results[f"{c}_f1"] for c in callers}
    sorted_callers = sorted(caller_f1s, key=caller_f1s.get, reverse=True)
    best_two = sorted_callers[:2]
    y_pred_best2 = (df[f"{best_two[0]}_pass"].astype(int) & df[f"{best_two[1]}_pass"].astype(int)).values
    results["baseline_best2_callers"] = f"{best_two[0]}+{best_two[1]}"
    results["baseline_best2_f1"] = f1_score(y_true, y_pred_best2, zero_division=0)
    results["baseline_best2_precision"] = precision_score(y_true, y_pred_best2, zero_division=0)
    results["baseline_best2_recall"] = recall_score(y_true, y_pred_best2, zero_division=0)

    return results


# ---------------------------------------------------------------------------
# Save predictions as BED file
# ---------------------------------------------------------------------------

def save_predictions_bed(predictions: pd.DataFrame, output_path: Path):
    """Save predicted true mutations as a BED file."""
    positives = predictions[predictions["prediction"] == 1][["chrom", "pos"]].copy()
    positives["end"] = positives["pos"]
    positives = positives[["chrom", "pos", "end"]]
    positives.to_csv(output_path, sep="\t", header=False, index=False)
    logger.info(f"Saved {len(positives)} predictions to {output_path}")


# ---------------------------------------------------------------------------
# Per-sample F1 computation
# ---------------------------------------------------------------------------

def per_sample_f1(
    df: pd.DataFrame,
    y_pred: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute F1, precision, recall per sample."""
    results = {}
    for sample in df["sample"].unique():
        mask = df["sample"] == sample
        y_true_s = df.loc[mask, "label"].values
        y_pred_s = y_pred[mask.values]
        results[sample] = {
            "f1": f1_score(y_true_s, y_pred_s, zero_division=0),
            "precision": precision_score(y_true_s, y_pred_s, zero_division=0),
            "recall": recall_score(y_true_s, y_pred_s, zero_division=0),
            "n_true": int(y_true_s.sum()),
            "n_pred": int(y_pred_s.sum()),
        }
    return results
