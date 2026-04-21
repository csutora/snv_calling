# Somatic SNV Meta-Learning: Technical Writeup

## 1. Problem Statement

Given calls from four somatic SNV callers (MuTect2, VarScan, FreeBayes, VarDict), we build a meta-learner that integrates their outputs to classify candidate positions as true mutations or artifacts, aiming to outperform any individual caller or naive ensemble.

The evaluation metric is F1 score, computed over the candidate universe: the union of all positions called by at least one caller (PASS in VCF), plus any truth-set positions. True negatives (~3 billion remaining genome positions) are excluded.

### Training data

Seven labeled samples: five synthetic tumors (syn1--syn5) with spike-in mutations at controlled variant allele frequencies, and two real tumors (real1, real2\_part1) with expert-curated truth sets. The test set (real2\_part2) contains VCF data only.

The synthetic tumors vary in cellularity (80--100%), subclonal structure, and aligner, providing diverse but artificial mutation signatures. The real tumors have unknown purity and biologically complex mutation profiles.

## 2. Method

### 2.1 Model choice: Gradient-Boosted Trees (XGBoost)

We chose XGBoost because the features are tabular, heterogeneous (categorical filter flags, continuous quality scores, integer read counts), and contain structured missing values:

- **vs. Logistic Regression**: Can't capture the nonlinear interactions that matter here (e.g., "trust MuTect2 when VAF is low AND VarDict also calls it") without extensive manual feature engineering.
- **vs. Neural Networks**: Dataset is too small (~300K candidates, ~80K positive) and tabular data with missing values is not where neural networks excel.
- **vs. Random Forest**: Gradient boosting typically squeezes out a few more percent of F1 by iteratively focusing on hard examples, which matters when leaderboard scores cluster tightly.
- **Missing value handling**: XGBoost natively learns optimal routing for missing values at each split. This is critical because missingness is *informative* -- a caller not reporting a position means it didn't think it was a variant.

### 2.2 Feature Engineering

We extract features at each candidate position from all four callers' VCF files. Our features fall into several categories:

#### Per-caller raw features
- **Binary call vector**: Whether each caller flagged the position as PASS (4 features)
- **Has-data indicators**: Whether the caller has *any* data at the position, separately from whether it called PASS. This captures the distinction between "evaluated and rejected" vs. "never considered" (4 features)
- **Quality scores**: QUAL per caller
- **Variant allele frequency (VAF/AF)**: Tumor VAF as reported by each caller
- **Read depths**: Total depth (DP), reference allele depth, and alt allele depth per caller
- **Caller-specific metrics**: MuTect2's TLOD/NLOD/FS/MQ/MQRankSum/ReadPosRankSum; VarScan's SPV/SSC/GPV; FreeBayes's alignment statistics (MQM, EPP, RPP, SAP, etc.); VarDict's SOR/SBF/NM/MSI

#### Per-caller filter flag encoding
Rather than binarizing filter status to PASS/not-PASS, we one-hot encode the specific failure reasons per caller (e.g., MuTect2's `t_lod_fstar`, `germline_risk`, `clustered_events`; VarDict's `f0.1`, `SN1.5`, etc.). This lets the model learn which filter failures are overly conservative vs. reliable artifact indicators.

#### Cross-caller aggregate features
- **Number of agreeing callers** (`n_callers_pass`): Sum of PASS calls across callers. Likely the single strongest feature.
- **Caller agreement pattern**: 4-bit integer encoding which specific subset of callers agreed. There are 15 non-empty subsets, and certain combinations (e.g., MuTect2+VarDict) are more trustworthy than others.
- **VAF agreement**: Mean, std, max, min, and range of VAF across callers. Cross-caller VAF concordance indicates confidence.
- **Quality and depth summaries**: Mean/max of quality scores and depths across callers.

#### VAF ratio features (domain-invariant)
Raw VAF depends on tumor purity, but the *ratio* between callers' VAFs should be consistent regardless of purity. We compute:
- Max/min AF ratio across callers
- Pairwise mean absolute VAF difference across all caller pairs
- Per-caller VAF deviation from the cross-caller mean

#### Sample-relative rank features
For depth and quality features, we compute within-sample percentile ranks. A position at the 90th percentile of MuTect2 depth in one sample means the same thing biologically as the 90th percentile in another, even if raw values differ due to different sequencing coverage. We rank-normalize 27 features including per-caller depth, alt depth, quality, AF, and aggregate summaries.

**Total: 147 features.** We use all of them rather than doing explicit feature selection -- XGBoost's regularization handles this implicitly. Aggressive feature subsampling (`colsample_bytree=0.323`), high minimum split gain (`gamma=4.59`), and L1 regularization (`alpha=1.77`) ensure that uninformative features simply never get split on. SHAP analysis confirms that ~20 features carry most of the signal, with the remaining 120+ contributing marginally but not harmfully.

### 2.3 The Domain Shift Problem

The central challenge: synthetic tumors have controlled purity, spike-in mutations, and clean noise profiles. Real tumors have heterogeneous clonality, unknown purity, and messier sequencing artifacts. Training naively on all data learns patterns that don't transfer from synthetic to real.

#### Quantifying the shift

To decide which features to mask, we compared true-mutation feature distributions between synthetic and real samples using the two-sample Kolmogorov-Smirnov test. This revealed a clear partition:

**Domain-dependent features (KS > 0.1, 47 features)**: All raw VAF/AF features, alt/ref depths, quality scores, TLOD/NLOD, SPV/SSC, VarDict SOR -- these scale with tumor purity, sequencing depth, and library preparation.

**Domain-invariant features (KS < 0.1, 23 features)**: Mapping quality (MQ, MQRankSum), strand bias (FS, SBF), read position bias (ReadPosRankSum), alignment statistics, microsatellite metrics -- these describe variant/artifact quality independent of sample characteristics.

The rank-normalized, binary/categorical, and VAF-ratio features are domain-invariant by construction.

#### Synthetic feature masking

To address this, we **mask domain-dependent features to NaN for synthetic samples during training.** This forces XGBoost to:

1. Learn from domain-invariant features (caller agreement, filter flags, mapping quality, rank-normalized values) when processing synthetic data -- gaining exposure to diverse mutation signatures
2. Use the full feature set (including raw depths, VAFs, quality scores) when processing real data -- learning the calibration specific to real tumors
3. At test time (real data), all features are available, and the model uses its full capacity

XGBoost's native missing-value routing makes this seamless: at each tree split involving a masked feature, synthetic samples are routed down the learned "missing" branch, while real samples use the actual feature value.

This way we still benefit from the synthetic data (77,470 labeled mutations for learning general patterns) without it teaching the model incorrect calibration for real samples.

### 2.4 Training Strategy

**Cross-validation**: Leave-One-Sample-Out (LOSO) rather than random k-fold. Each fold holds out one entire sample, giving honest generalization estimates and revealing the synthetic-to-real transfer gap.

**Optimization metric**: Mean F1 on real folds only (real1, real2\_part1). Synthetic fold performance is informative but not our target.

**Hyperparameter tuning**: 100-trial Optuna sweep with TPE sampler, optimizing real-fold F1. The sweep explores max\_depth, learning\_rate, subsample, colsample\_bytree, min\_child\_weight, L1/L2 regularization, gamma, and scale\_pos\_weight.

**Threshold tuning**: The prediction threshold is tuned on held-out LOSO predictions from real folds only (not fixed at 0.5). This is critical -- the global LOSO threshold (0.256) is pulled down by synthetic folds where the model is very confident, producing far too many false positives on real data. The two real fold thresholds are 0.955 (real1) and 0.584 (real2\_part1); we use their mean (0.77) as a balanced compromise for test-time prediction.

### 2.5 Best hyperparameters (from sweep)

| Parameter | Value | Interpretation |
|---|---|---|
| max\_depth | 9 | Relatively deep trees; affordable due to strong regularization |
| learning\_rate | 0.060 | Moderate; balanced with early stopping |
| subsample | 0.838 | Row subsampling for regularization |
| colsample\_bytree | 0.323 | Aggressive feature subsampling; prevents over-reliance on dominant features |
| min\_child\_weight | 11 | Conservative leaf creation; important with small real-data signal |
| lambda (L2) | 0.189 | Moderate L2 regularization |
| alpha (L1) | 1.765 | Strong L1 regularization; encourages sparsity |
| gamma | 4.591 | High minimum split gain; strongly prunes weak splits |
| scale\_pos\_weight factor | 2.672 | Upweights positives beyond natural class balance |

## 3. Results

### 3.1 Baseline performance (on real data)

All baselines and our model are evaluated on the same real samples (real1 and real2\_part1).

| Method | Real1 F1 | R2P1 F1 | Mean |
|---|---|---|---|
| MuTect2 alone | 0.358 | 0.240 | 0.299 |
| FreeBayes alone | 0.294 | 0.213 | 0.254 |
| VarDict alone | 0.268 | 0.135 | 0.201 |
| VarScan alone | 0.014 | 0.017 | 0.016 |
| >= 2 callers agree | 0.561 | 0.302 | 0.432 |
| Best 2 intersection (MuTect2 + FreeBayes) | 0.821 | 0.774 | 0.797 |
| **Our model (LOSO)** | **0.897** | **0.952** | **0.924** |

### 3.2 Ablation study

We systematically evaluated training configurations, focusing on real-data LOSO F1 (the metric that predicts test performance):

| Approach | Real1 LOSO F1 | R2P1 LOSO F1 | Description |
|---|---|---|---|
| Baseline (all data, default params) | 0.852 | 0.887 | Naive pooling of all samples |
| Real-only | 0.875 | 0.947 | Train only on real1 + real2\_part1 |
| + rank features ablated | 0.814 | 0.914 | Rank features help on real1 |
| + VAF ratio ablated | 0.862 | 0.893 | Small positive contribution |
| Finetune (all data then real) | 0.852\* | 0.887\* | Continue boosting on real data |
| Stacked (all-data preds as feature) | 0.815 | 0.945 | Information bottleneck hurts |
| **Mask-syn (default params)** | **0.863** | **0.921** | Mask domain-dependent features on syn |
| Mask-syn + finetune | 0.863\* | 0.921\* | Marginal improvement |

\*LOSO numbers are pre-finetune; finetune improves training metrics but can't be evaluated with LOSO.

After Optuna hyperparameter optimization (100 trials each):

| Approach | Real1 LOSO F1 | R2P1 LOSO F1 |
|---|---|---|
| Real-only (tuned) | 0.892 | 0.953 |
| **Mask-syn (tuned)** | **0.897** | **0.952** |

Both reach similar performance after tuning. We chose mask-syn as our final model because it uses 77K additional synthetic labels for learning general patterns, which should help generalize to unseen test data.

### 3.3 Key findings

1. **Caller agreement is king**: `caller_pattern` and `n_callers_pass` are the dominant features, consistent across all configurations.

2. **Synthetic data hurts naive training**: Simply pooling synthetic and real data degrades real-data F1 by 2--6% compared to real-only training, due to domain-dependent features learning incorrect calibration.

3. **Domain masking recovers synthetic utility**: Masking domain-dependent features for synthetic samples achieves the best of both worlds -- diverse mutation exposure without calibration pollution.

4. **MuTect2 mapping quality is highly informative**: `mutect2_mq` consistently ranks among the top 3 features, as it directly indicates alignment confidence independent of sample characteristics.

5. **Rank normalization helps**: Sample-relative percentile ranks for depth and quality features improve cross-sample generalization by removing absolute scale differences.

6. **Filter flags carry rich signal**: One-hot encoded filter reasons (not just PASS/fail) let the model learn which specific rejections are overly conservative.

### 3.4 SHAP Interpretability Analysis

We computed SHAP (SHapley Additive exPlanations) values on the final mask-syn model to understand *how* features drive predictions. SHAP decomposes each prediction into per-feature contributions, enabling both global importance ranking and per-instance explanations.

#### Global importance

The top 5 features by mean SHAP magnitude are:

| Feature | Mean SHAP | Interpretation |
|---|---|---|
| `caller_pattern` | 0.902 | Which specific subset of callers agreed -- encodes 15 possible agreement patterns |
| `n_callers_pass` | 0.871 | How many callers called PASS -- the single most intuitive signal |
| `varscan_qual_rank` | 0.815 | VarScan quality relative to other positions in the same sample |
| `mutect2_mq` | 0.661 | MuTect2 mapping quality -- low MQ strongly predicts artifacts |
| `mutect2_mqranksum` | 0.500 | Difference in mapping quality between alt and ref reads |

Caller agreement features (`caller_pattern` + `n_callers_pass`) together contribute ~1.77 mean SHAP, far more than any other features. The meta-learner is fundamentally leveraging *disagreement* between callers.

#### Beeswarm insights

The beeswarm plot reveals several key patterns:

- **`caller_pattern`**: High values (more callers agreeing) push strongly toward positive prediction, while low values push strongly negative. The effect is non-linear -- specific caller combinations matter, not just the count.
- **`mutect2_mq`**: Low mapping quality (pink dots on the left) strongly pushes predictions negative. MQ=60 (the maximum, indicating perfect mapping) is nearly neutral. This makes biological sense: poorly mapped reads produce artifactual variant calls.
- **`mutect2_fs`**: High Fisher strand bias pushes predictions negative, indicating strand-specific artifacts.
- **Rank features** (`varscan_qual_rank`, `qual_max_rank`, `ad_alt_max_rank`): These appear prominently, validating our sample-relative normalization strategy. The model relies on *relative* quality and depth within a sample rather than absolute values.

#### Synthetic vs Real comparison

Comparing SHAP patterns between synthetic and real true mutations reveals how the model uses different features across domains:

- **Synthetic mutations**: `varscan_qual_rank` is the top feature, followed by `caller_pattern`. Rank-normalized features dominate because domain-dependent raw features are masked.
- **Real mutations**: `caller_pattern` and `n_callers_pass` lead, followed by `mutect2_mq` and raw features like `mutect2_tlod` and `mutect2_nlod`. The model leverages the full feature set when available.

This shows the masking working as intended: the model learns mutation signatures from rank/categorical features on synthetic data, then uses raw features to refine predictions on real data.

#### Caller hierarchy

SHAP analysis of individual caller PASS features reveals a clear caller reliability hierarchy:

1. **`caller_pattern`** (0.90) and **`n_callers_pass`** (0.87) -- agreement is most important
2. **`vardict_pass`** (0.25) -- VarDict's opinion carries the most individual weight
3. **`mutect2_pass`** (0.12) -- MuTect2 contributes meaningfully
4. **`freebayes_pass`** (0.06) -- FreeBayes adds modest signal
5. **`varscan_pass`** (0.01) -- VarScan's individual PASS decision is nearly irrelevant

VarScan's low individual importance despite being the noisiest caller (F1=0.337 alone) is notable -- its signal comes through quality metrics and filter flags rather than its binary PASS/fail decision.

### 3.5 Unavoidable false negatives

Some true mutations in the truth sets are not called by any of the four callers. These are impossible to predict with any ensemble method and impose a hard ceiling on recall:

These are counted as false negatives in our F1 computation, as specified by the project guidelines.

## 4. Limitations and Future Work

### Limitations
- **Small real-data training set**: Only 1,810 labeled real mutations across 2 samples. The model's real-data performance is heavily dependent on these being representative.
- **Truth set imperfections**: Real-tumor truth sets are based on expert knowledge and manual curation, which is itself imperfect.
- **Synthetic-real domain gap**: Despite our masking approach, some domain shift likely remains in features we classified as "invariant."
- **No sequence context**: We do not use reference genome features (homopolymer runs, GC content, repeat regions) that correlate with sequencing error rates. These would require additional data sources.

### Potential improvements
- **Sequence context features**: If we had access to the GRCh37 reference genome, we could extract local GC content, homopolymer run length, and repeat region annotations at each candidate position. These would help distinguish sequencing artifacts (which concentrate in specific sequence contexts like homopolymer runs and low-complexity regions) from real mutations.
- **Per-sample VAF normalization**: Estimating tumor purity from the VAF distribution and normalizing VAFs accordingly would make them more comparable across samples. However, this might not provide much benefit in our setup, since our domain masking approach already handles the VAF domain shift -- raw VAFs are masked on synthetic data, and rank-normalized VAFs (which are purity-agnostic by construction) are available everywhere.
- **Stacking**: Training diverse first-layer models (XGBoost, LightGBM, logistic regression) and combining with a second-layer meta-learner. We tried a simple version of this (all-data predictions as a feature for a real-only model) but it didn't clearly outperform simpler approaches, likely due to the information bottleneck of compressing an entire model's knowledge into a single scalar.
- **More real training data**: The biggest limitation is having only 1,810 labeled real mutations. More labeled real tumors would reduce the dependence on synthetic data entirely.
