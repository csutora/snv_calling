# variant calling meta-learning project

somatic snv classification by integrating calls from mutect2, varscan, freebayes, and vardict using gradient-boosted trees (xgboost) with domain-adaptive synthetic feature masking

## setup

simplest way via [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

if you prefer pip:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## reproducing results

the full pipeline has three stages: data preparation, training, and evaluation \
if you're not using uv, copy the commands without the `uv run` part

### 1. prepare features

parses all vcf files, extracts per-caller features, computes derived/rank features, and saves to parquet:

```bash
uv run python -m src.prepare_data
```

produces `output/features_{train,real2_part2,test}.parquet`

### 2. train and evaluate

run the final model (mask-syn with optimized hyperparameters):

```bash
uv run python -m src.pipeline --no-wandb --mask-syn-raw \
  --params-file output/sweep_mask_syn_best.json
```

this will:
- load precomputed features
- print baseline comparisons (individual callers, 2+ agreement, best-2 intersection)
- run leave-one-sample-out cross-validation and report per-fold f1
- train the final model on all data
- generate predictions for training samples, real2_part2, and test
- save bed files to `output/`

to enable [wandb](https://wandb.ai) logging, remove `--no-wandb`

### 3. run hyperparameter sweep (optional)

```bash
uv run python -m src.sweep --mode mask-syn --n-trials 100 --no-wandb
uv run python -m src.sweep --mode real-only --n-trials 100 --no-wandb
```

best parameters are saved to `output/sweep_{mask_syn,real_only}_best.json`

### 4. shap analysis (optional)

generates interpretability plots in `output/plots/`:

```bash
uv run python -m src.shap_analysis
```

### 5. domain shift analysis (optional)

prints per-feature kolmogorov-smirnov statistics between synthetic and real samples:

```bash
uv run python -m src.domain_analysis
```

## evaluating on a held-out test set

to generate predictions for a new test dataset using a trained model:

```bash
uv run python -m src.evaluate \
  --model output/model.json \
  --data-dir path/to/test_vcf_directory \
  --output predictions.bed
```

the `--data-dir` should be a directory containing `{mutect2,varscan,freebayes,vardict}.vcf.gz` (with `.tbi` indices)

optional arguments:
- `--threshold 0.77` -- prediction threshold (defaults to 0.77, tuned on real loso folds)
- `--sample-name test` -- sample name for vcf path resolution

## training configurations (ablation flags)

the pipeline supports several training configurations via command-line flags:

| flag | description |
|---|---|
| `--mask-syn-raw` | mask domain-dependent features for synthetic samples (recommended) |
| `--real-only` | train only on real1 + real2_part1 |
| `--finetune` | train on all data, then continue boosting on real data only |
| `--stacked` | use all-data model predictions as a feature for real-only training |
| `--no-rank-features` | drop sample-relative rank features |
| `--no-vaf-ratio` | drop cross-caller vaf ratio features |
| `--params-file file` | load hyperparameters from a sweep results json |
| `--real-weight n` | upweight real samples during training |
