"""SHAP analysis of the final model.

Generates interpretability plots:
- Global feature importance (bar plot of mean |SHAP|)
- Beeswarm plot showing per-feature SHAP value distributions
- Top feature dependence plots
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src.domain_analysis import get_domain_dependent_features
from src.parse_vcf import get_feature_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
PLOTS_DIR = OUTPUT_DIR / "plots"


def load_model_and_data(
    model_path: Path,
    mask_syn: bool = True,
) -> tuple[xgb.Booster, pd.DataFrame, list[str]]:
    """Load the trained model and prepare data for SHAP analysis."""
    model = xgb.Booster()
    model.load_model(str(model_path))

    df = pd.read_parquet(OUTPUT_DIR / "features_train.parquet")
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    if mask_syn:
        dep_features = get_domain_dependent_features(df)
        syn_mask = df["sample"].str.startswith("syn")
        for col in dep_features:
            if col in df.columns:
                df.loc[syn_mask, col] = np.nan
        logger.info(f"Masked {len(dep_features)} domain-dependent features on synthetic rows")

    feature_cols = get_feature_columns(df)
    return model, df, feature_cols


def compute_shap_values(
    model: xgb.Booster,
    df: pd.DataFrame,
    feature_cols: list[str],
    sample_n: int = 5000,
) -> tuple[shap.Explanation, pd.DataFrame]:
    """Compute SHAP values on a subsample of the data."""
    # Subsample for efficiency — SHAP on 300K rows is slow
    rng = np.random.RandomState(42)
    if len(df) > sample_n:
        idx = rng.choice(len(df), sample_n, replace=False)
        df_sample = df.iloc[idx].reset_index(drop=True)
    else:
        df_sample = df

    X = df_sample[feature_cols]
    dmat = xgb.DMatrix(X, feature_names=feature_cols)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(dmat)

    explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X.values,
        feature_names=feature_cols,
    )

    return explanation, df_sample


def plot_global_importance(explanation: shap.Explanation, max_display: int = 25):
    """Bar plot of mean absolute SHAP values."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(explanation, max_display=max_display, show=False, ax=ax)
    ax.set_title("Global Feature Importance (mean |SHAP|)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "shap_global_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved shap_global_importance.png")


def plot_beeswarm(explanation: shap.Explanation, max_display: int = 25):
    """Beeswarm plot showing SHAP value distributions."""
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(explanation, max_display=max_display, show=False, plot_size=None)
    plt.title("SHAP Beeswarm Plot")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved shap_beeswarm.png")


def plot_dependence(
    explanation: shap.Explanation,
    feature_cols: list[str],
    top_n: int = 6,
):
    """Dependence plots for the top features."""
    mean_abs_shap = np.abs(explanation.values).mean(axis=0)
    top_indices = np.argsort(-mean_abs_shap)[:top_n]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, idx in enumerate(top_indices):
        ax = axes[i // 3, i % 3]
        feat_name = feature_cols[idx]
        shap.plots.scatter(
            explanation[:, idx],
            color=explanation,
            show=False,
            ax=ax,
        )
        ax.set_title(feat_name)

    fig.suptitle("SHAP Dependence Plots (Top 6 Features)", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "shap_dependence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved shap_dependence.png")


def plot_real_vs_syn_shap(
    model: xgb.Booster,
    df: pd.DataFrame,
    feature_cols: list[str],
    sample_n: int = 2000,
):
    """Compare SHAP patterns between synthetic and real true mutations."""
    pos = df[df["label"] == 1]
    pos_syn = pos[pos["sample"].str.startswith("syn")]
    pos_real = pos[~pos["sample"].str.startswith("syn")]

    rng = np.random.RandomState(42)

    explainer = shap.TreeExplainer(model)

    for subset, label in [(pos_syn, "synthetic"), (pos_real, "real")]:
        n = min(sample_n, len(subset))
        sub = subset.sample(n, random_state=rng)
        X = sub[feature_cols]
        dmat = xgb.DMatrix(X, feature_names=feature_cols)
        sv = explainer.shap_values(dmat)

        explanation = shap.Explanation(
            values=sv,
            base_values=explainer.expected_value,
            data=X.values,
            feature_names=feature_cols,
        )

        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(explanation, max_display=15, show=False, plot_size=None)
        plt.title(f"SHAP: {label.capitalize()} True Mutations")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"shap_beeswarm_{label}.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved shap_beeswarm_{label}.png")


def plot_caller_interaction(explanation: shap.Explanation, feature_cols: list[str]):
    """SHAP interaction analysis for caller agreement features."""
    # Focus on caller-related features
    caller_features = ["n_callers_pass", "caller_pattern",
                       "mutect2_pass", "varscan_pass", "freebayes_pass", "vardict_pass"]
    caller_indices = [feature_cols.index(f) for f in caller_features if f in feature_cols]

    mean_abs = np.abs(explanation.values[:, caller_indices]).mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    caller_names = [feature_cols[i] for i in caller_indices]
    bars = ax.barh(range(len(caller_names)), mean_abs, color="#1f77b4")
    ax.set_yticks(range(len(caller_names)))
    ax.set_yticklabels(caller_names)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Caller Agreement Feature Importance")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "shap_caller_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved shap_caller_importance.png")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = OUTPUT_DIR / "model.json"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Run pipeline first.")
        return

    logger.info("=== Loading model and data ===")
    model, df, feature_cols = load_model_and_data(model_path, mask_syn=True)

    logger.info("=== Computing SHAP values ===")
    explanation, df_sample = compute_shap_values(model, df, feature_cols, sample_n=5000)

    logger.info("=== Generating plots ===")
    plot_global_importance(explanation)
    plot_beeswarm(explanation)
    plot_dependence(explanation, feature_cols)
    plot_caller_interaction(explanation, feature_cols)

    logger.info("=== Comparing synthetic vs real ===")
    plot_real_vs_syn_shap(model, df, feature_cols)

    # Print top features for the writeup
    mean_abs_shap = np.abs(explanation.values).mean(axis=0)
    sorted_idx = np.argsort(-mean_abs_shap)
    logger.info("\n=== Top 20 features by mean |SHAP| ===")
    for i in sorted_idx[:20]:
        logger.info(f"  {feature_cols[i]}: {mean_abs_shap[i]:.4f}")

    logger.info(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
