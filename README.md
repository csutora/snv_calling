# Variant Calling Meta-Learning Project

Somatic SNV classification by integrating calls from MuTect2, VarScan, FreeBayes, and VarDict using gradient-boosted trees (XGBoost) with domain-adaptive synthetic feature masking.

## Setup

Simplest way via [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

If you prefer pip:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing results

The full pipeline has three stages: data preparation, training, and evaluation. \
If you're not using uv, copy the commands without the `uv run` part.

### 1. Prepare features

Parses all VCF files, extracts per-caller features, computes derived/rank features, and saves to parquet:

```bash
uv run python -m src.prepare_data
```

Produces `output/features_{train,real2_part2,test}.parquet`.

### 2. Train and evaluate

Run the final model (mask-syn with optimized hyperparameters):

```bash
uv run python -m src.pipeline --no-wandb --mask-syn-raw \
  --params-file output/sweep_mask_syn_best.json
```

This will:
- Load precomputed features
- Print baseline comparisons (individual callers, 2+ agreement, best-2 intersection)
- Run LOSO cross-validation and report per-fold F1
- Train the final model on all data
- Generate predictions for training samples, real2_part2, and test
- Save BED files to `output/`

To enable [wandb](https://wandb.ai) logging, remove `--no-wandb`.

### 3. Run hyperparameter sweep (optional)

```bash
uv run python -m src.sweep --mode mask-syn --n-trials 100 --no-wandb
uv run python -m src.sweep --mode real-only --n-trials 100 --no-wandb
```

Best parameters are saved to `output/sweep_{mask_syn,real_only}_best.json`.

### 4. SHAP analysis (optional)

Generates interpretability plots in `output/plots/`:

```bash
uv run python -m src.shap_analysis
```

### 5. Domain shift analysis (optional)

Prints per-feature KS statistics between synthetic and real samples:

```bash
uv run python -m src.domain_analysis
```

## Evaluating on a held-out test set

To generate predictions for a new test dataset using a trained model:

```bash
uv run python -m src.evaluate \
  --model output/model.json \
  --data-dir path/to/test_vcf_directory \
  --output predictions.bed
```

The `--data-dir` should be a directory containing `{mutect2,varscan,freebayes,vardict}.vcf.gz` (with `.tbi` indices). The naming convention follows the `test/` folder format.

Optional arguments:
- `--threshold 0.256` -- prediction threshold (defaults to 0.256, the value from our best sweep)
- `--sample-name test` -- sample name for VCF path resolution

## Training configurations (ablation flags)

The pipeline supports several training configurations via command-line flags:

| Flag | Description |
|---|---|
| `--mask-syn-raw` | Mask domain-dependent features for synthetic samples (recommended) |
| `--real-only` | Train only on real1 + real2_part1 |
| `--finetune` | Train on all data, then continue boosting on real data only |
| `--stacked` | Use all-data model predictions as a feature for real-only training |
| `--no-rank-features` | Drop sample-relative rank features |
| `--no-vaf-ratio` | Drop cross-caller VAF ratio features |
| `--params-file FILE` | Load hyperparameters from a sweep results JSON |
| `--real-weight N` | Upweight real samples during training |
