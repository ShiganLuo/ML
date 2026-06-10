"""Mutation-level feature extraction for SV frequency correction.

This module turns breakpoint-level BAM evidence and probe sequence information
into one feature row per mutation.

Core idea
---------
* Reads are first collected around both breakpoints of a mutation.
* The same read is counted only once per mutation, even if it is seen in both
  breakpoint windows.
* Probe/sequence features are aggregated from all probes that belong to the
  mutation.

This is the right level for model training because the label is mutation-level
ddPCR frequency, not read-level evidence.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.LogUtil import setup_logger
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import pysam

logger = setup_logger(__name__)



def parse_csv_list(value: str) -> List[str]:
    """Parse a comma-separated string into a cleaned list."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def select_feature_columns(
    frame: pd.DataFrame,
    label_columns: set[str],
    exclude_columns: List[str],
) -> List[str]:
    """Select numeric training columns after removing label/excluded columns."""
    feature_frame = frame.select_dtypes(include=[np.number]).copy()
    for column in label_columns.union(exclude_columns):
        if column in feature_frame.columns:
            feature_frame = feature_frame.drop(columns=[column])
    return list(feature_frame.columns)


def preprocess_features(
    frame: pd.DataFrame,
    feature_columns: List[str],
    var_threshold: float = 1e-8,
    corr_threshold: float = 0.95,
    outdir: Optional[str] = None,
    keep_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Preprocess full feature matrix without train/test distinction.

    Steps
    -----
    1. Standard scaling on all rows.
    2. Near-constant feature filtering using variance threshold.
    3. Correlation-based feature dropping with variance-based representative.

    Parameters
    ----------
    frame : pandas.DataFrame
        Input feature table.
    feature_columns : list[str]
        Columns used as model features.
    var_threshold : float, optional
        Variance threshold for near-constant feature filtering.
    corr_threshold : float, optional
        Correlation threshold for defining highly correlated feature clusters.
    outdir : str or None, optional
        Output directory for preprocessed artifacts.
    keep_columns : list[str] or None, optional
        Non-feature columns to keep in output, preserved as-is.

    Returns
    -------
    tuple
        ``(processed_frame, diagnostics)``. Correlated features are grouped into
        connected components where abs(corr) > corr_threshold; only the
        highest-variance feature in each component is kept.
    """
    X = frame[feature_columns].fillna(0.0).values.astype(float)

    try:
        s = np.linalg.svd(X, compute_uv=False)
        cond_before = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
        min_sv_before = float(s[-1])
    except Exception:
        cond_before = float("inf")
        min_sv_before = 0.0

    logger.info("Preprocessing: samples=%d, features=%d", X.shape[0], X.shape[1])
    logger.info(
        "Preprocessing: min singular value(before)=%.6e, condition number(before)=%.6e",
        min_sv_before,
        cond_before,
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    vt = VarianceThreshold(threshold=var_threshold)
    vt.fit(X_scaled)
    keep_mask = vt.get_support()

    kept_columns = [c for c, keep in zip(feature_columns, keep_mask) if keep]
    removed_by_variance = [c for c, keep in zip(feature_columns, keep_mask) if not keep]
    logger.info("Preprocessing: removed %d features by variance threshold", len(removed_by_variance))

    if not kept_columns:
        raise RuntimeError("No features remain after VarianceThreshold")

    X_var = X_scaled[:, keep_mask]

    full_frame = pd.DataFrame(X_var, columns=kept_columns)
    corr = full_frame.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    feature_variances = frame[kept_columns].fillna(0.0).var(axis=0, ddof=0)

    correlated_graph: Dict[str, set[str]] = {col: set() for col in kept_columns}
    for col in upper.columns:
        for row_name, corr_val in upper[col].items():
            if not np.isfinite(corr_val) or corr_val <= corr_threshold:
                continue
            correlated_graph[col].add(row_name)
            correlated_graph[row_name].add(col)

    visited: set[str] = set()
    dropped_correlated: List[str] = []
    correlation_clusters: List[Dict[str, object]] = []
    for col in kept_columns:
        if col in visited:
            continue
        stack = [col]
        component: List[str] = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            stack.extend(correlated_graph[node])

        if len(component) <= 1:
            continue

        best_col = max(
            component,
            key=lambda name: float(feature_variances.get(name, 0.0)),
        )
        for name in component:
            if name != best_col:
                dropped_correlated.append(name)
        correlation_clusters.append({
            "representative": best_col,
            "members": sorted(component),
            "dropped": sorted([name for name in component if name != best_col]),
        })

    logger.info(
        "Preprocessing: dropping %d highly correlated features (>%.2f)",
        len(dropped_correlated),
        corr_threshold,
    )

    processed_feature_columns = [c for c in kept_columns if c not in dropped_correlated]
    if not processed_feature_columns:
        raise RuntimeError("No features remain after correlation-based dropping")

    selected_indices = [kept_columns.index(c) for c in processed_feature_columns]
    X_processed = X_var[:, selected_indices]

    try:
        s2 = np.linalg.svd(X_processed, compute_uv=False)
        cond_after = float(s2[0] / s2[-1]) if s2[-1] > 0 else float("inf")
        min_sv_after = float(s2[-1])
    except Exception:
        cond_after = float("inf")
        min_sv_after = 0.0

    logger.info(
        "Preprocessing: final features=%d, min sv(after)=%.6e, cond(after)=%.6e",
        len(processed_feature_columns),
        min_sv_after,
        cond_after,
    )
    diagnostics: Dict[str, object] = {
        "cond_before": cond_before,
        "min_sv_before": min_sv_before,
        "removed_by_variance": removed_by_variance,
        "dropped_correlated": dropped_correlated,
        "correlation_clusters": correlation_clusters,
        "cond_after": cond_after,
        "min_sv_after": min_sv_after,
    }
    X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_columns)
    if keep_columns:
        kept_frame = frame[keep_columns].copy()
        X_processed_df = pd.concat([kept_frame.reset_index(drop=True), X_processed_df], axis=1)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        diag_path = f"{outdir}/preprocessing_diagnostics.json"
        metadata_path = f"{outdir}/preprocessing_metadata.json"
        scaler_path = f"{outdir}/preprocessing_scaler.joblib"
        features_path = f"{outdir}/preprocessed_features.tsv"
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2)
        metadata = {
            "feature_columns": feature_columns,
            "keep_mask": keep_mask.tolist(),
            "kept_columns": kept_columns,
            "processed_feature_columns": processed_feature_columns,
            "var_threshold": var_threshold,
            "corr_threshold": corr_threshold,
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        joblib.dump(scaler, scaler_path)
        logger.info("Saved preprocessing diagnostics to %s", diag_path)
        logger.info("Saved preprocessing metadata to %s", metadata_path)
        logger.info("Saved preprocessing scaler to %s", scaler_path)
        X_processed_df.to_csv(features_path, sep="\t", index=False)

    return X_processed_df, diagnostics

def load_preprocessing(preprocess_dir: str) -> Dict[str, object]:
    """Load preprocessing metadata and scaler.

    Parameters
    ----------
    preprocess_dir : str
        Directory containing preprocessing artifacts.

    Returns
    -------
    dict
        Dictionary with metadata and fitted scaler.
    """
    metadata_path = os.path.join(preprocess_dir, "preprocessing_metadata.json")
    scaler_path = os.path.join(preprocess_dir, "preprocessing_scaler.joblib")
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    metadata["scaler"] = joblib.load(scaler_path)
    return metadata



def parse_breakpoint(value: object) -> Tuple[str, int]:
    """Parse a breakpoint string into chromosome and 1-based position."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError("Breakpoint value is missing")

    text = str(value).strip()
    if not text:
        raise ValueError("Breakpoint value is empty")
    if ":" not in text:
        raise ValueError("Invalid breakpoint format: {0}".format(text))

    chrom, pos_text = text.split(":", 1)
    return chrom.strip(), int(pos_text)


def _normalize_contig(bam: pysam.AlignmentFile, chrom: str) -> str:
    """Map a chromosome label to a contig present in the BAM."""
    if chrom in bam.references:
        return chrom
    if chrom.startswith("chr") and chrom[3:] in bam.references:
        return chrom[3:]
    prefixed = "chr{0}".format(chrom)
    if prefixed in bam.references:
        return prefixed
    return chrom


def _shannon_entropy(seq: str) -> float:
    if not seq:
        return 0.0
    seq = seq.upper()
    counts = np.array([seq.count(base) for base in "ACGT"], dtype=float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def _longest_homopolymer(seq: str) -> int:
    if not seq:
        return 0
    best = 1
    current = 1
    seq = seq.upper()
    for idx in range(1, len(seq)):
        if seq[idx] == seq[idx - 1]:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def _sequence_features(seq: object) -> Dict[str, float]:
    """Compute probe sequence complexity features."""
    if seq is None or (isinstance(seq, float) and math.isnan(seq)):
        seq_text = ""
    else:
        seq_text = str(seq).strip().upper()

    if not seq_text:
        return {
            "probe_len": 0.0,
            "probe_gc": 0.0,
            "probe_entropy": 0.0,
            "probe_longest_homopolymer": 0.0,
            "probe_n_fraction": 0.0,
        }

    probe_len = float(len(seq_text))
    gc = float(seq_text.count("G") + seq_text.count("C")) / probe_len
    n_fraction = float(seq_text.count("N")) / probe_len
    return {
        "probe_len": probe_len,
        "probe_gc": gc,
        "probe_entropy": _shannon_entropy(seq_text),
        "probe_longest_homopolymer": float(_longest_homopolymer(seq_text)),
        "probe_n_fraction": n_fraction,
    }


def _mean_depth(bam: pysam.AlignmentFile, chrom: str, start: int, end: int) -> float:
    """Compute mean depth on a 0-based half-open interval."""
    if end <= start:
        return 0.0
    try:
        arrays = bam.count_coverage(chrom, start, end, quality_threshold=0)
    except (ValueError, OSError):
        return 0.0
    total = np.sum(arrays)
    return float(total) / float(end - start)


def _count_support_reads(read: pysam.AlignedSegment, min_softclip: int) -> Tuple[bool, bool, bool]:
    """Classify one read as soft-clipped, split and/or discordant support."""
    soft_clip = False
    split_read = False
    discordant = False

    if read.is_supplementary or read.has_tag("SA"):
        split_read = True

    cigartuples = read.cigartuples or []
    if cigartuples:
        left_clip = cigartuples[0][1] if cigartuples[0][0] == 4 else 0
        right_clip = cigartuples[-1][1] if cigartuples[-1][0] == 4 else 0
        if left_clip >= min_softclip or right_clip >= min_softclip:
            soft_clip = True
            split_read = True

    if read.is_paired:
        if not read.is_proper_pair:
            discordant = True
        elif read.next_reference_id != read.reference_id:
            discordant = True
        elif abs(read.template_length) > 10000:
            discordant = True

    return soft_clip, split_read, discordant


def _collect_reads_around_breakpoint(
    bam: pysam.AlignmentFile,
    chrom: str,
    pos: int,
    window: int,
    min_softclip: int,
    breakpoint_name: str,
) -> List[Dict[str, object]]:
    """Collect read-level evidence around one breakpoint."""
    contig = _normalize_contig(bam, chrom)
    pos0 = max(0, pos - 1)
    fetch_start = max(0, pos0 - window)
    fetch_end = pos0 + window

    records: List[Dict[str, object]] = []
    try:
        for read in bam.fetch(contig, fetch_start, fetch_end):
            soft_clip, split_read, discordant = _count_support_reads(read, min_softclip)
            records.append(
                {
                    "read_id": read.query_name,
                    "breakpoint": breakpoint_name,
                    "mapq": float(read.mapping_quality),
                    "softclip": int(soft_clip),
                    "split_read": int(split_read),
                    "discordant": int(discordant),
                    "template_length": abs(float(read.template_length)) if read.template_length not in (None, 0) else np.nan,
                }
            )
    except (ValueError, OSError):
        pass
    return records


def _aggregate_read_records(records: List[Dict[str, object]]) -> Dict[str, float]:
    """Aggregate read-level records to mutation-level read features."""
    if not records:
        return {
            "read_total_unique": 0.0,
            "read_support_unique": 0.0,
            "read_support_fraction": 0.0,
            "read_mapq_mean": 0.0,
            "read_mapq_max": 0.0,
            "read_softclip_fraction": 0.0,
            "read_split_fraction": 0.0,
            "read_discordant_fraction": 0.0,
            "read_template_mean": 0.0,
            "read_template_std": 0.0,
            "read_breakpoint_balance": 0.0,
        }

    df = pd.DataFrame(records)
    df = df.sort_values(["read_id", "mapq"], ascending=[True, False])
    df = df.drop_duplicates(subset=["read_id"], keep="first")
    support_flag = ((df["softclip"] > 0) | (df["split_read"] > 0) | (df["discordant"] > 0)).astype(int)

    breakpoint_counts = df.groupby("breakpoint")["read_id"].nunique().tolist()
    if len(breakpoint_counts) < 2:
        breakpoint_balance = 0.0
    else:
        breakpoint_balance = abs(breakpoint_counts[0] - breakpoint_counts[1]) / (sum(breakpoint_counts) + 1e-6)

    template_series = df["template_length"].dropna()

    return {
        "read_total_unique": float(df["read_id"].nunique()),
        "read_support_unique": float(support_flag.sum()),
        "read_support_fraction": float(support_flag.mean()),
        "read_mapq_mean": float(df["mapq"].mean()) if not df.empty else 0.0,
        "read_mapq_max": float(df["mapq"].max()) if not df.empty else 0.0,
        "read_softclip_fraction": float(df["softclip"].mean()) if not df.empty else 0.0,
        "read_split_fraction": float(df["split_read"].mean()) if not df.empty else 0.0,
        "read_discordant_fraction": float(df["discordant"].mean()) if not df.empty else 0.0,
        "read_template_mean": float(template_series.mean()) if not template_series.empty else 0.0,
        "read_template_std": float(template_series.std()) if len(template_series) > 1 else 0.0,
        "read_breakpoint_balance": float(breakpoint_balance),
    }


def _aggregate_probe_features(
    df_probe: pd.DataFrame,
    chrom1: str,
    pos1: int,
    chrom2: str,
    pos2: int,
    flank: int,
) -> Optional[Dict[str, float]]:
    """Aggregate probe/sequence features to mutation level.

    The probe table is expected to have columns ``chr``, ``start``, ``end`` and
    ``sequence``. Probes are assigned to a mutation when they overlap either
    breakpoint neighborhood. If no probe overlaps, all probes are used as a
    fallback so the output remains defined.
    """
    if df_probe is None or df_probe.empty:
        return None

    probe_df = df_probe.copy()

    chr_column = None
    for column in ["chr", "chrom", "Chrom", "Chr"]:
        if column in probe_df.columns:
            chr_column = column
            break

    start_column = None
    for column in ["start", "Start"]:
        if column in probe_df.columns:
            start_column = column
            break

    end_column = None
    for column in ["end", "End"]:
        if column in probe_df.columns:
            end_column = column
            break

    seq_column = None
    for column in ["sequence", "seq", "probe_seq", "probe_sequence", "ProbeSeq"]:
        if column in probe_df.columns:
            seq_column = column
            break

    if chr_column is not None and start_column is not None and end_column is not None:
        probe_df = probe_df.copy()
        probe_df[start_column] = pd.to_numeric(probe_df[start_column], errors="coerce")
        probe_df[end_column] = pd.to_numeric(probe_df[end_column], errors="coerce")

        left1 = max(0, pos1 - flank)
        right1 = pos1 + flank
        left2 = max(0, pos2 - flank)
        right2 = pos2 + flank

        overlaps_bp1 = (
            (probe_df[chr_column].astype(str) == str(chrom1))
            & probe_df[start_column].notna()
            & probe_df[end_column].notna()
            & (probe_df[end_column] >= left1)
            & (probe_df[start_column] <= right1)
        )
        overlaps_bp2 = (
            (probe_df[chr_column].astype(str) == str(chrom2))
            & probe_df[start_column].notna()
            & probe_df[end_column].notna()
            & (probe_df[end_column] >= left2)
            & (probe_df[start_column] <= right2)
        )
        probe_df = probe_df.loc[overlaps_bp1 | overlaps_bp2].copy()

    if probe_df.empty:
        probe_df = df_probe.copy()

    if seq_column is None:
        probe_features = [_sequence_features(None) for _ in range(len(probe_df))]
    else:
        probe_features = [_sequence_features(value) for value in probe_df[seq_column].tolist()]

    if not probe_features:
        probe_features = [_sequence_features(None)]

    feature_df = pd.DataFrame(probe_features)
    return {
        "probe_count": float(len(probe_df)),
        "probe_len_mean": float(feature_df["probe_len"].mean()),
        "probe_gc_mean": float(feature_df["probe_gc"].mean()),
        "probe_entropy_mean": float(feature_df["probe_entropy"].mean()),
        "probe_longest_homopolymer_mean": float(feature_df["probe_longest_homopolymer"].mean()),
        "probe_n_fraction_mean": float(feature_df["probe_n_fraction"].mean()),
        "probe_gc_std": float(feature_df["probe_gc"].std()) if len(feature_df) > 1 else 0.0,
    }


def _breakpoint_depth_features(
    bam: pysam.AlignmentFile,
    chrom: str,
    pos: int,
    flank: int,
    window: int,
) -> Dict[str, float]:
    """Compute depth features around one breakpoint."""
    contig = _normalize_contig(bam, chrom)
    pos0 = max(0, pos - 1)
    left_start = max(0, pos0 - flank)
    left_end = pos0
    right_start = pos0
    right_end = pos0 + flank
    center_start = max(0, pos0 - window)
    center_end = pos0 + window

    depth_left = _mean_depth(bam, contig, left_start, left_end)
    depth_right = _mean_depth(bam, contig, right_start, right_end)
    depth_center = _mean_depth(bam, contig, center_start, center_end)
    depth_diff = abs(depth_right - depth_left)
    depth_sum = depth_left + depth_right + 1e-6

    return {
        "depth_left": depth_left,
        "depth_right": depth_right,
        "depth_center": depth_center,
        "depth_sum": depth_sum,
        "depth_diff": depth_diff,
        "depth_asymmetry": depth_diff / depth_sum,
    }


def extract_bam_feature(
    bam_file: str,
    pos1: object,
    pos2: object,
    df_probe: Optional[pd.DataFrame],
    flank: int = 100,
    window: int = 20,
    min_softclip: int = 5,
) -> Dict[str, float]:
    """Extract mutation-level features for one SV record.

    Parameters
    ----------
    bam_file : str
        BAM path.
    mutation : str
        Mutation identifier.
    pos1, pos2 : object
        Breakpoints in ``chr:pos`` format.
    df_probe : pandas.DataFrame or None
        Probe/sequence table. Multiple probes for one mutation are aggregated.
    freq : float
        Observed SV frequency.
    ddpcr_af : float
        ddPCR truth frequency.
    flank : int, optional
        Window for depth feature calculation.
    window : int, optional
        Window for collecting read evidence.
    min_softclip : int, optional
        Minimum soft-clipping length to count as breakpoint support.

    Returns
    -------
    dict
        Mutation-level feature dictionary.
    """

    chrom1, pos1_int = parse_breakpoint(pos1)
    chrom2, pos2_int = parse_breakpoint(pos2)

    bam = pysam.AlignmentFile(bam_file, "rb")
    try:
        bp1_depth = _breakpoint_depth_features(bam, chrom1, pos1_int, flank=flank, window=window)
        bp2_depth = _breakpoint_depth_features(bam, chrom2, pos2_int, flank=flank, window=window)

        read_records = []
        read_records.extend(_collect_reads_around_breakpoint(bam, chrom1, pos1_int, window, min_softclip, "bp1"))
        read_records.extend(_collect_reads_around_breakpoint(bam, chrom2, pos2_int, window, min_softclip, "bp2"))
        combined = {
            "bp_support_max": max(bp1_depth["depth_sum"], bp2_depth["depth_sum"]),
            "bp_depth_min": min(bp1_depth["depth_sum"], bp2_depth["depth_sum"]),
            "bp_depth_asymmetry_mean": (bp1_depth["depth_asymmetry"] + bp2_depth["depth_asymmetry"]) / 2.0,
        }

        for prefix, features in (("bp1", bp1_depth), ("bp2", bp2_depth)):
            for key, value in features.items():
                combined["{0}_{1}".format(prefix, key)] = value
        read_features = _aggregate_read_records(read_records)
        combined.update(read_features)

        if df_probe is not None:
            probe_features = _aggregate_probe_features(df_probe, chrom1, pos1_int, chrom2, pos2_int, flank)
            combined.update(probe_features) if probe_features is not None else None
        return combined
    finally:
        bam.close()


SV_TYPE_CATEGORIES = ["CTX-1", "CTX-2", "DEL/ITX", "DUP/ITX", "INV"]


def encode_sv_type(frame: pd.DataFrame, column: str = "FusionType") -> pd.DataFrame:
    """Add one-hot encoded SV type columns to the feature matrix.

    Parameters
    ----------
    frame : pandas.DataFrame
        Input dataframe containing the SV type column.
    column : str, optional
        Name of the column containing SV type labels.

    Returns
    -------
    pandas.DataFrame
        DataFrame with new binary columns ``sv_type_CTX-1``, ``sv_type_CTX-2``, etc.
        The original column is preserved.
    """
    frame = frame.copy()
    for sv_type in SV_TYPE_CATEGORIES:
        col_name = f"sv_type_{sv_type}"
        frame[col_name] = (frame[column] == sv_type).astype(float)
    return frame


def parser_table(
        infile: str, 
        probe_infile: Optional[str] = None,
        outdir: Optional[str] = None,
        extra_keep_cols: List[str] = ["ddPCR_AF", "Freq", "FusionType"],
        probe_columns:List[str] = ["chr", "start", "end", "sequence"],
        bam_columns: List[str] = ["BamPath", "Pos1", "Pos2"]
    ) -> pd.DataFrame:
    """
    Function: Parse the SV table and emit mutation-level feature rows.
    Parameters:
        - infile: Path to the input SV table (TSV format).
        - probe_infile: Path to the probe sequence table (TSV format). chr, start, end and sequence columns are expected but not strictly required.
        - outdir: Optional directory to save intermediate feature files.
        - extra_keep_cols: extra columns thar you want to keep in final features matrix
        - probe_columns: probe_infile must contain these columns
        - bam_columns: script need these columns to produce feature matrix.
    Returns:
        - DataFrame containing extracted features for each mutation.
    """
    df = pd.read_csv(infile, sep="\t")
    df_probe = pd.read_csv(probe_infile, sep="\t") if probe_infile else None
    if df_probe is not None:
        if not all(col in df_probe.columns for col in probe_columns):
            raise ValueError(f"Probe infile must contain columns: {probe_columns} if it is provided")
    if not all(col in df.columns for col in bam_columns):
        raise ValueError(f"infile must contain columns: {bam_columns}")

    rows: List[Dict[str, float]] = []
    df = df[bam_columns + extra_keep_cols].copy()
    df = df.drop_duplicates()
    for _, row in df.iterrows():
        features = extract_bam_feature(
            bam_file=row["BamPath"],
            pos1=row["Pos1"],
            pos2=row["Pos2"],
            df_probe=df_probe,
        )
        for col in extra_keep_cols:
            features[col] = row[col]
        rows.append(features)
    df_features = pd.DataFrame(rows)
    if "FusionType" in df_features.columns:
        df_features = encode_sv_type(df_features, "FusionType")
    df_features = df_features.loc[:, extra_keep_cols + [c for c in df_features.columns if c not in extra_keep_cols]]
    if outdir:
        features_path = f"{outdir}/raw_extracted_features.tsv"
        df_features.to_csv(features_path, sep="\t", index=False)
        logger.info("Saved raw extracted features to %s", features_path)
    return df_features


if __name__ == "__main__":
    outdir = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/output/ML_nochip/feature"
    # df = parser_table(
    #     infile="/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/output/ML_nochip/ddPCR_SV_comparison.tsv",
    #     outdir=outdir,
    #     bam_columns=["BamPath", "Pos1", "Pos2",],
    #     probe_columns=["chr", "start", "end", "sequence"],
    #     extra_keep_cols = ["原始编号", "FusionGene","FusionExon", "sampleID","ddPCR_AF","Freq", "FusionType"]
    # )
    df = pd.read_csv(f"{outdir}/raw_extracted_features.tsv", sep="\t")
    feature_cols = [c for c in df.columns if c not in {"原始编号", "FusionGene","FusionExon", "sampleID","ddPCR_AF","Freq", "FusionType"} and not c.startswith("sv_type_")]
    keep_cols = [c for c in df.columns if c not in feature_cols]
    logger.info(f"Feature columns: {feature_cols}")
    df_feature, diagnostics = preprocess_features(
        df,
        feature_columns = feature_cols,
        outdir=outdir,
        keep_columns=keep_cols,
    )
    logger.info(f"Preprocessing diagnostics: {diagnostics}")