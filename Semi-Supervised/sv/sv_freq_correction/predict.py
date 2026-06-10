"""Predict corrected SV frequencies using trained models."""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List, Dict, Optional
import joblib
import numpy as np
import pandas as pd

try:
    from .features import apply_preprocessing, extract_bam_feature
except ImportError:
    from features import apply_preprocessing, extract_bam_feature


def predict_AF(
    infile: str,
    probe_infile: str,
    model_path: str,
    outdir: str,
    feature_cols_path: Optional[str] = None,
    preprocess_dir: Optional[str] = None,
    keep_columns: Optional[List[str]] = None,
    probe_columns:List[str] = ["chr", "start", "end", "sequence"],
    infile_columns: List[str] = ["BamPath", "Pos1", "Pos2", "mutation", "Freq"]
) -> None:
    """
    Function: Predict corrected SV frequencies using a trained model.
    Parameters:
    - infile: TSV file containing SV data with columns(BamPath, Pos1, Pos2, mutation, Freq)
    - probe_infile: TSV file containing probe information.
    - model_path: Path to the trained model file (joblib format).
    - feature_cols_path: Optional JSON file with feature column names.
    - preprocess_dir: Optional directory with preprocessing artifacts.
    - keep_columns: Optional list of non-feature columns to keep in output.
    - outdir: Directory to save the output TSV file with predictions.
    - bam_col: Column name in infile for BAM file paths.
    - pos1_col: Column name for the first position of the SV.
    - pos2_col: Column name for the second position of the SV.
    - seq_col: Column name for the sequence context around the SV.
    - raw_freq_col: Column name for the raw frequency of the SV.
    - ddpcr_col: Column name for the ddPCR measured allele frequency of the SV.
    """
    df = pd.read_csv(infile, sep="\t")
    df_probe = pd.read_csv(probe_infile, sep="\t")
    if not all(col in df_probe.columns for col in probe_columns):
        raise ValueError(f"Probe infile must contain columns: {probe_columns}")
    if not all(col in df.columns for col in infile_columns):
        raise ValueError(f"infile must contain columns: {infile_columns}")
    if keep_columns is None:
        keep_columns = list(df.columns)

    rows: List[Dict[str, float]] = []
    for _, row in df.iterrows():
        features = extract_bam_feature(
            bam_file=row["BamPath"],
            pos1=row["Pos1"],
            pos2=row["Pos2"],
            df_probe=df_probe,
        )
        for col in keep_columns:
            if col in row.index:
                features[col] = row[col]
        rows.append(features)

    df_feature = pd.DataFrame(rows)
    if preprocess_dir:
        processed_frame, processed_feature_columns = apply_preprocessing(
            frame=df_feature,
            preprocess_dir=preprocess_dir,
            keep_columns=keep_columns,
        )
        X = processed_frame[processed_feature_columns].to_numpy(dtype=float)
        out = processed_frame.copy()
    else:
        if not feature_cols_path:
            raise ValueError("feature_cols_path is required when preprocess_dir is not provided")
        with open(feature_cols_path, "r", encoding="utf-8") as handle:
            feature_cols = json.load(handle)
        X = df_feature[feature_cols].to_numpy(dtype=float)
        out = df_feature.copy()

    model = joblib.load(model_path)
    preds = model.predict(X)

    training_config_path = os.path.join(os.path.dirname(model_path), "training_config.json")
    if os.path.exists(training_config_path):
        with open(training_config_path, "r", encoding="utf-8") as handle:
            training_config = json.load(handle)
        if training_config.get("target_transform") == "logit":
            preds = 1.0 / (1.0 + np.exp(-preds))

    out["predicted_AF"] = preds
    outdir = os.path.join(outdir, os.path.basename(model_path).replace(".joblib", ""))
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "predictions.tsv")
    out.to_csv(out_path, sep="\t", index=False)

def main():
    parser = argparse.ArgumentParser(description="Predict corrected SV frequency")
    parser.add_argument("--table", required=True, help="TSV with Pos1, Pos2 and BamPath")
    parser.add_argument("--probe_infile", required=True, help="TSV with probe information")
    parser.add_argument("--model", required=True, help="trained model path")
    parser.add_argument("--feature-cols", default=None, help="JSON file with feature column names")
    parser.add_argument("--preprocess-dir", default=None, help="Directory with preprocessing artifacts")
    parser.add_argument("--keep-cols", nargs="+", default=None, help="Non-feature columns to keep in output")
    parser.add_argument("--out", required=True, help="prediction output TSV")
    args = parser.parse_args()
    predict_AF(
        infile=args.table,
        probe_infile=args.probe_infile,
        model_path=args.model,
        feature_cols_path=args.feature_cols,
        outdir=args.out,
        preprocess_dir=args.preprocess_dir,
        keep_columns=args.keep_cols,
    )
def run():
    table = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/data/SV_jiaozheng_yanzheng.tsv"
    probe_infile = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/data/cd6_nomerge.bed"
    model_path = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/output/ML/train/model_ridge_0.1.joblib"
    feature_cols_path = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/output/ML/train/feature_cols.json"
    outdir = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/output/ML/predict/validation"
    model_paths = glob.glob("/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/output/ML/train/model_*.joblib")
    for model_path in model_paths:
        predict_AF(
            infile=table,
            probe_infile=probe_infile,
            model_path=model_path,
            feature_cols_path=feature_cols_path,
            outdir=outdir
        )

if __name__ == "__main__":
    run()
    pass
