"""Analyze feature domain shift between synthetic and real samples.

Identifies which raw features have significantly different distributions
between synthetic and real true-mutation samples, indicating they are
domain-dependent and should be masked for synthetic training data.

The output is used to define the masking strategy in the training pipeline:
synthetic samples get domain-dependent features set to NaN, forcing XGBoost
to learn from domain-invariant features on synthetic data while still
using all features on real data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.parse_vcf import CALLER_NAMES, get_feature_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")

# KS statistic threshold: features above this are considered domain-dependent.
# Chosen based on visual inspection of the distribution gap — features below
# this threshold have nearly identical distributions between syn and real.
KS_THRESHOLD = 0.1

# Features that are inherently non-numeric or already domain-invariant by design
SKIP_SUFFIXES = ("_pass", "_has_data", "_rank")
SKIP_PREFIXES = ("mutect2_filt_", "varscan_filt_", "freebayes_filt_", "vardict_filt_")
SKIP_EXACT = {"n_callers_pass", "n_callers_data", "caller_pattern", "af_n_reported"}


def analyze_domain_shift(df: pd.DataFrame) -> pd.DataFrame:
    """Compare feature distributions between synthetic and real true mutations.

    Uses the two-sample Kolmogorov-Smirnov test to quantify distributional
    differences. Only considers true-positive rows (label=1) to compare
    what real mutations look like across domains.

    Returns a DataFrame with per-feature statistics, sorted by KS statistic.
    """
    feature_cols = get_feature_columns(df)

    # Filter to raw numeric features only
    raw_cols = []
    for c in feature_cols:
        if any(c.endswith(s) for s in SKIP_SUFFIXES):
            continue
        if any(c.startswith(p) for p in SKIP_PREFIXES):
            continue
        if c in SKIP_EXACT:
            continue
        raw_cols.append(c)

    # Compare distributions on true mutations only
    pos = df[df["label"] == 1]
    pos_syn = pos[pos["sample"].str.startswith("syn")]
    pos_real = pos[~pos["sample"].str.startswith("syn")]

    logger.info(f"Comparing {len(raw_cols)} raw features between "
                f"{len(pos_syn)} synthetic and {len(pos_real)} real true mutations")

    results = []
    for col in raw_cols:
        s = pos_syn[col].dropna().astype(float)
        r = pos_real[col].dropna().astype(float)

        if len(s) < 50 or len(r) < 50:
            continue

        # Subsample for KS test efficiency (statistic is sample-size independent)
        rng = np.random.RandomState(42)
        s_sample = s.sample(min(5000, len(s)), random_state=rng)
        r_sample = r.sample(min(5000, len(r)), random_state=rng)
        ks_stat, ks_pval = ks_2samp(s_sample, r_sample)

        s_med = s.median()
        r_med = r.median()
        ratio = min(s_med, r_med) / max(abs(s_med), abs(r_med)) if max(abs(s_med), abs(r_med)) > 0 else 1.0

        results.append({
            "feature": col,
            "syn_median": s_med,
            "real_median": r_med,
            "median_ratio": ratio,
            "syn_std": s.std(),
            "real_std": r.std(),
            "ks_stat": ks_stat,
            "ks_pval": ks_pval,
            "domain_dependent": ks_stat > KS_THRESHOLD,
            "syn_n": len(s),
            "real_n": len(r),
        })

    results_df = pd.DataFrame(results).sort_values("ks_stat", ascending=False)
    return results_df


def get_domain_dependent_features(df: pd.DataFrame) -> list[str]:
    """Return the list of features that are domain-dependent (KS > threshold).

    These features should be masked to NaN for synthetic samples during training.
    """
    analysis = analyze_domain_shift(df)
    dependent = analysis[analysis["domain_dependent"]]["feature"].tolist()
    invariant = analysis[~analysis["domain_dependent"]]["feature"].tolist()

    logger.info(f"Domain-dependent features (KS > {KS_THRESHOLD}): {len(dependent)}")
    logger.info(f"Domain-invariant features (KS <= {KS_THRESHOLD}): {len(invariant)}")

    return dependent


def main():
    train_path = OUTPUT_DIR / "features_train.parquet"
    if not train_path.exists():
        logger.error(f"{train_path} not found. Run `python -m src.prepare_data` first.")
        return

    df = pd.read_parquet(train_path)
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    analysis = analyze_domain_shift(df)

    # Print results
    print(f"\n{'Feature':<35} {'Syn med':>10} {'Real med':>10} {'Ratio':>7} "
          f"{'Syn std':>10} {'Real std':>10} {'KS':>7} {'Domain dep?':>12}")
    print("-" * 107)

    for _, row in analysis.iterrows():
        marker = "YES ***" if row["ks_stat"] > 0.3 else "YES **" if row["ks_stat"] > 0.2 else \
                 "YES *" if row["ks_stat"] > KS_THRESHOLD else "no"
        print(f"{row['feature']:<35} {row['syn_median']:>10.3f} {row['real_median']:>10.3f} "
              f"{row['median_ratio']:>7.3f} {row['syn_std']:>10.3f} {row['real_std']:>10.3f} "
              f"{row['ks_stat']:>7.3f} {marker:>12}")

    dependent = analysis[analysis["domain_dependent"]]["feature"].tolist()
    invariant = analysis[~analysis["domain_dependent"]]["feature"].tolist()

    print(f"\n--- Summary ---")
    print(f"KS threshold: {KS_THRESHOLD}")
    print(f"Domain-dependent (to mask on synthetic): {len(dependent)}")
    print(f"Domain-invariant (keep everywhere):      {len(invariant)}")

    print(f"\nDomain-dependent features:")
    for f in dependent:
        print(f"  {f}")

    print(f"\nDomain-invariant features:")
    for f in invariant:
        print(f"  {f}")

    # Save analysis
    analysis.to_csv(OUTPUT_DIR / "domain_shift_analysis.csv", index=False)
    logger.info(f"Saved analysis to {OUTPUT_DIR / 'domain_shift_analysis.csv'}")


if __name__ == "__main__":
    main()
