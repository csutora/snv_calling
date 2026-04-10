"""Parse VCF files from the 4 somatic SNV callers and extract features."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pysam

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-caller feature extraction
# ---------------------------------------------------------------------------

CALLER_NAMES = ["mutect2", "varscan", "freebayes", "vardict"]

# Sample directories and their VCF naming patterns
SAMPLES_STANDARD = ["syn1", "syn2", "syn3", "syn4", "syn5", "real1"]
# real2_part1 and real2_part2 have different naming conventions


def _get_vcf_path(data_dir: Path, sample: str, caller: str) -> Path:
    """Resolve VCF path accounting for different naming conventions."""
    sample_dir = data_dir / sample

    if sample in SAMPLES_STANDARD:
        return sample_dir / f"{sample}-{caller}.vcf.gz"
    elif sample == "real2_part1":
        # real2_part1 uses: real2_{caller}_chr1to5.vcf.gz
        caller_name = caller if caller != "mutect2" else "mutect"
        return sample_dir / f"real2_{caller_name}_chr1to5.vcf.gz"
    elif sample == "real2_part2":
        caller_name = caller if caller != "mutect2" else "mutect"
        return sample_dir / f"real2_{caller_name}_rest.vcf.gz"
    elif sample == "test":
        caller_name = caller if caller != "mutect2" else "mutect2"
        return sample_dir / f"{caller_name}.vcf.gz"
    else:
        raise ValueError(f"Unknown sample: {sample}")


def _get_truth_path(data_dir: Path, sample: str) -> Path | None:
    """Resolve truth BED path. Returns None for test samples."""
    if sample in SAMPLES_STANDARD:
        return data_dir / sample / f"{sample}_truth.bed"
    elif sample == "real2_part1":
        return data_dir / sample / "real2_truth_chr1to5.bed"
    elif sample in ("real2_part2", "test"):
        return None
    else:
        raise ValueError(f"Unknown sample: {sample}")


def load_truth(data_dir: Path, sample: str) -> set[tuple[str, int]]:
    """Load truth BED file as a set of (chrom, pos) tuples."""
    truth_path = _get_truth_path(data_dir, sample)
    if truth_path is None:
        return set()

    truth = set()
    with open(truth_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            chrom = str(parts[0])
            pos = int(parts[1])  # BED start (0-based or 1-based varies; these are 1-based per inspection)
            truth.add((chrom, pos))
    return truth


def _is_snv(ref: str, alt: str) -> bool:
    """Check if a variant is a single nucleotide variant."""
    return len(ref) == 1 and len(alt) == 1 and ref != alt


# ---------------------------------------------------------------------------
# Feature extraction per caller
# ---------------------------------------------------------------------------

def _safe_float(val) -> float | None:
    """Convert a value to float, returning None if not possible.

    Handles tuples/lists by taking the first element (multi-allelic sites).
    Converts inf to None (XGBoost handles NaN natively, but not inf).
    """
    if val is None:
        return None
    if isinstance(val, (tuple, list)):
        if len(val) == 0:
            return None
        val = val[0]
    try:
        result = float(val)
        if np.isinf(result):
            return None
        return result
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> int | None:
    """Convert a value to int, handling tuples/lists."""
    if val is None:
        return None
    if isinstance(val, (tuple, list)):
        if len(val) == 0:
            return None
        val = val[0]
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _info_get(info, key):
    """Safely get an INFO field — pysam throws ValueError if key isn't in header."""
    try:
        return info.get(key)
    except (ValueError, KeyError):
        return None


def _format_get(sample, key):
    """Safely get a FORMAT field — pysam throws ValueError if key isn't in header."""
    try:
        return sample.get(key)
    except (ValueError, KeyError):
        return None


def _extract_mutect2_features(record: pysam.VariantRecord) -> dict:
    """Extract features from a MuTect2 VCF record."""
    info = record.info
    tumor = record.samples[0]  # First sample is tumor

    # FORMAT fields
    ad = _format_get(tumor, "AD")
    af = _safe_float(_format_get(tumor, "AF"))
    dp = _safe_int(_format_get(tumor, "DP"))

    ref_depth = _safe_int(ad[0] if ad and len(ad) > 0 else None)
    alt_depth = _safe_int(ad[1] if ad and len(ad) > 1 else None)

    return {
        "mutect2_qual": _safe_float(record.qual),
        "mutect2_af": af,
        "mutect2_dp": dp,
        "mutect2_ad_ref": ref_depth,
        "mutect2_ad_alt": alt_depth,
        "mutect2_fs": _safe_float(_info_get(info, "FS")),
        "mutect2_mq": _safe_float(_info_get(info, "MQ")),
        "mutect2_mqranksum": _safe_float(_info_get(info, "MQRankSum")),
        "mutect2_readposranksum": _safe_float(_info_get(info, "ReadPosRankSum")),
        "mutect2_tlod": _safe_float(_info_get(info, "TLOD")),
        "mutect2_nlod": _safe_float(_info_get(info, "NLOD")),
        "mutect2_qd": _safe_float(_info_get(info, "QD")),
        "mutect2_clippingranksum": _safe_float(_info_get(info, "ClippingRankSum")),
        "mutect2_hrun": _safe_int(_info_get(info, "HRun")),
        "mutect2_gc": _safe_float(_info_get(info, "GC")),
    }


def _extract_varscan_features(record: pysam.VariantRecord) -> dict:
    """Extract features from a VarScan VCF record."""
    info = record.info
    # VarScan puts normal first, tumor second
    tumor = record.samples[1] if len(record.samples) > 1 else record.samples[0]

    dp = _safe_int(_format_get(tumor, "DP"))
    rd = _safe_int(_format_get(tumor, "RD"))
    ad = _safe_int(_format_get(tumor, "AD"))
    freq_str = _format_get(tumor, "FREQ")

    freq = None
    if freq_str:
        try:
            freq = float(str(freq_str).replace("%", "")) / 100.0
        except (ValueError, TypeError):
            pass

    return {
        "varscan_qual": _safe_float(record.qual),
        "varscan_af": freq,
        "varscan_dp": dp,
        "varscan_ad_ref": rd,
        "varscan_ad_alt": ad,
        "varscan_spv": _safe_float(_info_get(info, "SPV")),
        "varscan_ssc": _safe_float(_info_get(info, "SSC")),
        "varscan_gpv": _safe_float(_info_get(info, "GPV")),
    }


def _extract_freebayes_features(record: pysam.VariantRecord) -> dict:
    """Extract features from a FreeBayes VCF record."""
    info = record.info
    tumor = record.samples[0]  # First sample is tumor

    ad = _format_get(tumor, "AD")
    ao = _format_get(tumor, "AO")
    ro = _format_get(tumor, "RO")
    dp = _safe_int(_format_get(tumor, "DP"))

    ref_depth = _safe_int(ad[0] if ad and len(ad) > 0 else ro)
    alt_depth = _safe_int(ao if ao is not None else (ad[1] if ad and len(ad) > 1 else None))

    af = None
    if alt_depth and dp and dp > 0:
        af = alt_depth / dp

    return {
        "freebayes_qual": _safe_float(record.qual),
        "freebayes_af": af,
        "freebayes_dp": dp,
        "freebayes_ad_ref": ref_depth,
        "freebayes_ad_alt": alt_depth,
        "freebayes_mqm": _safe_float(_info_get(info, "MQM")),
        "freebayes_mqmr": _safe_float(_info_get(info, "MQMR")),
        "freebayes_sap": _safe_float(_info_get(info, "SAP")),
        "freebayes_srp": _safe_float(_info_get(info, "SRP")),
        "freebayes_epp": _safe_float(_info_get(info, "EPP")),
        "freebayes_eppr": _safe_float(_info_get(info, "EPPR")),
        "freebayes_rpp": _safe_float(_info_get(info, "RPP")),
        "freebayes_rppr": _safe_float(_info_get(info, "RPPR")),
        "freebayes_ab": _safe_float(_info_get(info, "AB")),
        "freebayes_odds": _safe_float(_info_get(info, "ODDS")),
    }


def _extract_vardict_features(record: pysam.VariantRecord) -> dict:
    """Extract features from a VarDict VCF record."""
    info = record.info
    tumor = record.samples[0]

    af = _safe_float(_format_get(tumor, "AF"))
    dp = _safe_int(_format_get(tumor, "DP"))
    vd = _safe_int(_format_get(tumor, "VD"))
    rd = _safe_int(_format_get(tumor, "RD"))
    mq = _safe_float(_format_get(tumor, "MQ"))
    nm = _safe_float(_format_get(tumor, "NM"))
    qual = _safe_float(_format_get(tumor, "QUAL"))
    sbf = _safe_float(_format_get(tumor, "SBF"))
    oddratio = _safe_float(_format_get(tumor, "ODDRATIO"))
    hiaf = _safe_float(_format_get(tumor, "HIAF"))
    pmean = _safe_float(_format_get(tumor, "PMEAN"))
    sn = _safe_float(_format_get(tumor, "SN"))
    adjaf = _safe_float(_format_get(tumor, "ADJAF"))

    return {
        "vardict_qual": _safe_float(record.qual),
        "vardict_af": af,
        "vardict_dp": dp,
        "vardict_ad_ref": rd,
        "vardict_ad_alt": vd,
        "vardict_mq": mq,
        "vardict_nm": nm,
        "vardict_format_qual": qual,
        "vardict_sbf": sbf,
        "vardict_oddratio": oddratio,
        "vardict_hiaf": hiaf,
        "vardict_pmean": pmean,
        "vardict_sn": sn,
        "vardict_adjaf": adjaf,
        "vardict_sor": _safe_float(_info_get(info, "SOR")),
        "vardict_ssf": _safe_float(_info_get(info, "SSF")),
        "vardict_msi": _safe_float(_info_get(info, "MSI")),
        "vardict_msilen": _safe_float(_info_get(info, "MSILEN")),
    }


_EXTRACTORS = {
    "mutect2": _extract_mutect2_features,
    "varscan": _extract_varscan_features,
    "freebayes": _extract_freebayes_features,
    "vardict": _extract_vardict_features,
}


# ---------------------------------------------------------------------------
# Two-pass VCF parsing: first collect PASS positions, then extract features
# ---------------------------------------------------------------------------

def _collect_pass_positions(vcf_path: Path) -> set[tuple[str, int]]:
    """Fast first pass: collect only positions with PASS filter (SNVs only)."""
    positions: set[tuple[str, int]] = set()
    vcf = pysam.VariantFile(str(vcf_path))
    for record in vcf:
        if "PASS" not in record.filter:
            continue
        ref = record.ref
        for alt in record.alts or []:
            if _is_snv(ref, alt):
                positions.add((str(record.chrom), record.pos))
    vcf.close()
    return positions


def parse_caller_vcf_at_positions(
    vcf_path: Path,
    caller: str,
    candidate_positions: set[tuple[str, int]],
) -> dict[tuple[str, int], dict]:
    """Parse a caller's VCF, extracting features only at candidate positions.

    Returns a dict mapping (chrom, pos) -> feature dict.
    """
    extractor = _EXTRACTORS[caller]
    records: dict[tuple[str, int], dict] = {}

    vcf = pysam.VariantFile(str(vcf_path))

    for record in vcf:
        chrom = str(record.chrom)
        pos = record.pos
        key = (chrom, pos)

        if key not in candidate_positions:
            continue

        ref = record.ref
        for alt in record.alts or []:
            if not _is_snv(ref, alt):
                continue

            filters = list(record.filter)
            is_pass = "PASS" in filters

            features = extractor(record)
            features[f"{caller}_has_data"] = True
            features[f"{caller}_pass"] = is_pass
            features[f"{caller}_filter"] = ";".join(filters) if filters else "."

            if key not in records or (is_pass and not records[key].get(f"{caller}_pass")):
                records[key] = features

    vcf.close()
    logger.info(f"  {caller}: extracted features at {len(records)} / {len(candidate_positions)} candidate positions "
                f"({sum(1 for v in records.values() if v.get(f'{caller}_pass'))} PASS)")
    return records


# ---------------------------------------------------------------------------
# Build feature matrix for a sample
# ---------------------------------------------------------------------------

def _collect_pass_for_caller(data_dir: Path, sample: str, caller: str) -> set[tuple[str, int]]:
    """Worker function for parallel pass-1 collection."""
    vcf_path = _get_vcf_path(data_dir, sample, caller)
    positions = _collect_pass_positions(vcf_path)
    return positions


def _extract_features_for_caller(
    data_dir: Path, sample: str, caller: str, candidates: set[tuple[str, int]]
) -> tuple[str, dict[tuple[str, int], dict]]:
    """Worker function for parallel pass-2 extraction."""
    vcf_path = _get_vcf_path(data_dir, sample, caller)
    records = parse_caller_vcf_at_positions(vcf_path, caller, candidates)
    return caller, records


def build_sample_features(
    data_dir: Path,
    sample: str,
) -> pd.DataFrame:
    """Build the feature matrix for a single sample.

    Two-pass approach with parallel VCF processing across callers:
    1. Fast scan of all VCFs to collect PASS positions (parallel)
    2. Build candidate universe (PASS positions + truth)
    3. Extract features only at candidate positions (parallel)
    """
    logger.info(f"Processing sample: {sample}")

    # Pass 1: collect PASS positions from each caller (parallel)
    pass_positions: set[tuple[str, int]] = set()
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_collect_pass_for_caller, data_dir, sample, caller): caller
            for caller in CALLER_NAMES
        }
        for future in as_completed(futures):
            caller = futures[future]
            caller_pass = future.result()
            logger.info(f"  {caller}: {len(caller_pass)} PASS SNV positions")
            pass_positions |= caller_pass

    # Load truth
    truth = load_truth(data_dir, sample)

    # Candidate universe
    candidates = pass_positions | truth
    logger.info(f"  Candidate universe: {len(candidates)} positions "
                f"({len(pass_positions)} from callers, {len(truth)} truth, "
                f"{len(truth - pass_positions)} truth-only)")

    # Pass 2: extract features at candidate positions (parallel)
    caller_data: dict[str, dict[tuple[str, int], dict]] = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_extract_features_for_caller, data_dir, sample, caller, candidates): caller
            for caller in CALLER_NAMES
        }
        for future in as_completed(futures):
            caller_name, records = future.result()
            n_pass = sum(1 for v in records.values() if v.get(f"{caller_name}_pass"))
            logger.info(f"  {caller_name}: extracted features at {len(records)} / {len(candidates)} candidate positions ({n_pass} PASS)")
            caller_data[caller_name] = records

    # Build rows
    rows = []
    for chrom, pos in sorted(candidates, key=lambda x: (x[0].zfill(3) if x[0].isdigit() else x[0], x[1])):
        row: dict = {"chrom": chrom, "pos": pos}

        for caller in CALLER_NAMES:
            key = (chrom, pos)
            if key in caller_data[caller]:
                row.update(caller_data[caller][key])
            else:
                row[f"{caller}_has_data"] = False
                row[f"{caller}_pass"] = False

        row["label"] = 1 if (chrom, pos) in truth else 0
        row["sample"] = sample
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"  Built {len(df)} rows, {df['label'].sum()} positive")
    return df


# ---------------------------------------------------------------------------
# Derived / aggregate features
# ---------------------------------------------------------------------------

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features computed across callers."""
    df = df.copy()

    # Number of callers with PASS
    pass_cols = [f"{c}_pass" for c in CALLER_NAMES]
    df["n_callers_pass"] = df[pass_cols].sum(axis=1)

    # Number of callers with any data (PASS or not)
    data_cols = [f"{c}_has_data" for c in CALLER_NAMES]
    df["n_callers_data"] = df[data_cols].sum(axis=1)

    # VAF agreement across callers
    af_cols = [f"{c}_af" for c in CALLER_NAMES]
    af_vals = df[af_cols].astype(float)
    df["af_mean"] = af_vals.mean(axis=1, skipna=True)
    df["af_std"] = af_vals.std(axis=1, skipna=True)
    df["af_max"] = af_vals.max(axis=1, skipna=True)
    df["af_min"] = af_vals.min(axis=1, skipna=True)
    df["af_range"] = df["af_max"] - df["af_min"]
    df["af_n_reported"] = af_vals.notna().sum(axis=1)

    # Quality score summaries
    qual_cols = [f"{c}_qual" for c in CALLER_NAMES]
    qual_vals = df[qual_cols].astype(float)
    df["qual_mean"] = qual_vals.mean(axis=1, skipna=True)
    df["qual_max"] = qual_vals.max(axis=1, skipna=True)
    df["qual_min"] = qual_vals.min(axis=1, skipna=True)

    # Depth summaries
    dp_cols = [f"{c}_dp" for c in CALLER_NAMES]
    dp_vals = df[dp_cols].astype(float)
    df["dp_mean"] = dp_vals.mean(axis=1, skipna=True)
    df["dp_max"] = dp_vals.max(axis=1, skipna=True)

    # Alt depth summaries
    ad_alt_cols = [f"{c}_ad_alt" for c in CALLER_NAMES]
    ad_alt_vals = df[ad_alt_cols].astype(float)
    df["ad_alt_mean"] = ad_alt_vals.mean(axis=1, skipna=True)
    df["ad_alt_max"] = ad_alt_vals.max(axis=1, skipna=True)

    # Agreement pattern (which specific callers agree) — encode as integer
    # 4-bit pattern: mutect2=1, varscan=2, freebayes=4, vardict=8
    df["caller_pattern"] = (
        df["mutect2_pass"].astype(int) * 1 +
        df["varscan_pass"].astype(int) * 2 +
        df["freebayes_pass"].astype(int) * 4 +
        df["vardict_pass"].astype(int) * 8
    )

    # --- VAF ratio features (#2: domain-invariant cross-caller concordance) ---
    # Max/min AF ratio: if callers agree on VAF, ratio ≈ 1 regardless of purity
    af_min_nonzero = af_vals.where(af_vals > 0).min(axis=1, skipna=True)
    af_max_val = af_vals.max(axis=1, skipna=True)
    df["af_max_min_ratio"] = af_max_val / af_min_nonzero.replace(0, np.nan)

    # Pairwise VAF concordance: mean absolute difference between all caller pairs
    af_pairs = []
    for i, c1 in enumerate(CALLER_NAMES):
        for c2 in CALLER_NAMES[i+1:]:
            pair_diff = (df[f"{c1}_af"].astype(float) - df[f"{c2}_af"].astype(float)).abs()
            af_pairs.append(pair_diff)
    df["af_pairwise_mad"] = pd.concat(af_pairs, axis=1).mean(axis=1, skipna=True)

    # Per-caller VAF deviation from cross-caller mean
    for c in CALLER_NAMES:
        df[f"{c}_af_dev"] = (df[f"{c}_af"].astype(float) - df["af_mean"]).abs()

    return df


def add_sample_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sample-relative normalized features (#1: domain-invariant).

    For depth and quality features, compute within-sample percentile ranks
    so the model learns relative signal strength rather than absolute values.
    """
    df = df.copy()

    # Features to rank-normalize within each sample
    rank_features = []
    for c in CALLER_NAMES:
        rank_features.extend([
            f"{c}_dp", f"{c}_ad_alt", f"{c}_ad_ref", f"{c}_qual", f"{c}_af",
        ])
    # Also aggregate features
    rank_features.extend([
        "dp_mean", "dp_max", "ad_alt_mean", "ad_alt_max",
        "qual_mean", "qual_max", "af_mean",
    ])

    # Only rank features that exist in the dataframe
    rank_features = [f for f in rank_features if f in df.columns]

    for feat in rank_features:
        col = df[feat].astype(float)
        # Percentile rank within each sample (0-1), NaN stays NaN
        df[f"{feat}_rank"] = df.groupby("sample")[feat].rank(pct=True, na_option="keep")

    return df


# ---------------------------------------------------------------------------
# Per-caller filter flag encoding
# ---------------------------------------------------------------------------

def encode_filter_flags(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the most common per-caller filter flags."""
    df = df.copy()

    # Collect common filter reasons per caller
    common_filters = {
        "mutect2": [
            "t_lod_fstar", "alt_allele_in_normal", "clustered_events",
            "germline_risk", "panel_of_normals", "strand_artifact",
            "homologous_mapping_event", "str_contraction", "MinAF",
        ],
        "varscan": [
            "REJECT", "SpvFreq", "indelError", "str10",
        ],
        "freebayes": [
            "REJECT", "FBQualDepth",
        ],
        "vardict": [
            "REJECT", "f0.1", "p8", "SN1.5", "Bias", "Q0",
            "d5", "NM4.25", "LowAlleleDepth", "q22.5", "v3",
            "pSTD", "MAF0.05", "LowFreqQuality", "P0.01Likely",
            "DIFF0.2", "MSI12", "InGap", "Cluster0bp",
        ],
    }

    for caller, filters in common_filters.items():
        filter_col = f"{caller}_filter"
        if filter_col not in df.columns:
            continue
        for filt in filters:
            df[f"{caller}_filt_{filt}"] = df[filter_col].fillna("").str.contains(filt, regex=False).astype(int)

    return df


# ---------------------------------------------------------------------------
# Full pipeline: build features for all samples
# ---------------------------------------------------------------------------

def _build_sample_worker(data_dir: Path, sample: str) -> pd.DataFrame:
    """Worker for parallel sample processing."""
    return build_sample_features(data_dir, sample)


def build_all_features(
    data_dir: Path,
    samples: list[str] | None = None,
    max_parallel_samples: int = 4,
) -> pd.DataFrame:
    """Build feature matrix across all samples (parallel across samples)."""
    if samples is None:
        samples = SAMPLES_STANDARD + ["real2_part1"]

    # Process samples in parallel — each sample internally parallelizes across callers
    # Use fewer parallel samples to avoid oversubscription (each sample uses 4 workers)
    dfs = []
    with ProcessPoolExecutor(max_workers=max_parallel_samples) as pool:
        futures = {
            pool.submit(_build_sample_worker, data_dir, sample): sample
            for sample in samples
        }
        for future in as_completed(futures):
            sample = futures[future]
            df = future.result()
            logger.info(f"Completed {sample}: {len(df)} rows")
            dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = add_derived_features(combined)
    combined = encode_filter_flags(combined)
    combined = add_sample_relative_features(combined)

    logger.info(f"Total: {len(combined)} rows, {combined['label'].sum()} positive, "
                f"{combined.shape[1]} features")
    return combined


# ---------------------------------------------------------------------------
# Feature columns (exclude identifiers and labels)
# ---------------------------------------------------------------------------

NON_FEATURE_COLS = {"chrom", "pos", "label", "sample"}
FILTER_TEXT_COLS = {f"{c}_filter" for c in CALLER_NAMES}

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names."""
    exclude = NON_FEATURE_COLS | FILTER_TEXT_COLS
    return [c for c in df.columns if c not in exclude]
