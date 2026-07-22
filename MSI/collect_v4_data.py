#!/usr/bin/env python3
# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Collect v4 new sequencing data from Excel.

Pipeline:
    Excel → filter samples in RIA2026Menglu → resolve site_path → v4_newsequencing.tsv

Usage:
    python collect_v4_data.py

Input:
    - data/MSI/v4_newsequencing/MSI补充测试样本待选集.xlsx (IHC as MSI_real)

Output:
    - output/MSI/data/v4_newsequencing.tsv (sample_id, MSI_real, site_path)
"""

import os
import sys
import logging
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from common.LogUtil import setup_logger


# ── Paths ──
EXCEL_PATH = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/data/MSI/v4_newsequencing/MSI补充测试样本待选集.xlsx"
SEARCH_DIR = "/GeneCloud003/prod/project/Clinical/cnc_process/OncoTop/RIA2026Menglu"
OUT_TSV = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/data/v4_newsequencing.tsv"


def read_excel_samples(excel_path: str) -> pd.DataFrame:
    """Read all sheets, extract sample_id and IHC as MSI_real."""
    logger = logging.getLogger("read_excel")
    xls = pd.ExcelFile(excel_path)
    logger.info(f"Excel sheets: {xls.sheet_names}")

    dfs = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(excel_path, sheet_name=sheet)
        dfs.append(df)
        logger.info(f"  {sheet}: {len(df)} rows")

    all_df = pd.concat(dfs, ignore_index=True)

    # Extract key columns
    result = pd.DataFrame()
    result['sample_id'] = all_df['样本编号'].astype(str).str.strip()
    result['MSI_real'] = all_df['IHC'].astype(str).str.strip()

    # Drop invalid rows
    result = result[result['sample_id'].notna() & (result['sample_id'] != 'nan')]
    result = result[result['MSI_real'].notna() & (result['MSI_real'] != 'nan')]

    # Extra columns if available
    if '肿瘤细胞含量' in all_df.columns:
        result['tumor_content'] = pd.to_numeric(
            all_df['肿瘤细胞含量'].astype(str).str.replace('%', ''), errors='coerce')
        if result['tumor_content'].max() > 1:
            result['tumor_content'] = result['tumor_content'] / 100

    if 'TMB状态' in all_df.columns:
        result['TMB_status'] = all_df.loc[result.index, 'TMB状态'].values

    logger.info(f"Total samples from Excel: {len(result)}")
    logger.info(f"MSI_real distribution: {result['MSI_real'].value_counts().to_dict()}")
    return result


def build_sample_index(search_dir: str) -> dict:
    """Build {sample_id: sample_dir} index from search_dir."""
    logger = logging.getLogger("build_index")
    sample_map = {}

    task_dirs = [d for d in os.listdir(search_dir)
                 if d.startswith('OncoTOP') and os.path.isdir(os.path.join(search_dir, d))]
    logger.info(f"Task directories: {task_dirs}")

    for task in task_dirs:
        task_path = os.path.join(search_dir, task)
        for item in os.listdir(task_path):
            item_path = os.path.join(task_path, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                sample_map[item] = item_path

    logger.info(f"Indexed {len(sample_map)} sample directories")
    return sample_map


def resolve_site_path(sample_dir: str, sample_id: str) -> str:
    """Find site.txt in sample directory."""
    candidates = [
        os.path.join(sample_dir, f"indicator/msi/TopMsi/{sample_id}_cancer_dedup_realign.site.txt"),
        os.path.join(sample_dir, f"indicator/msi/top_msi/{sample_id}_cancer_dedup_realign.site.txt"),
        os.path.join(sample_dir, f"indicator/msi/TopMsi/{sample_id}_cancer_sort_markdup_realign_recal.site.txt"),
        os.path.join(sample_dir, f"indicator/msi/top_msi/{sample_id}_cancer_sort_markdup_realign_recal.site.txt"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def main():
    logger = setup_logger("collect_v4_data")

    # Step 1: Read Excel
    logger.info(f"Reading Excel: {EXCEL_PATH}")
    df = read_excel_samples(EXCEL_PATH)

    # Step 2: Build sample index
    logger.info(f"Building sample index from: {SEARCH_DIR}")
    sample_map = build_sample_index(SEARCH_DIR)

    # Step 3: Match and resolve paths
    logger.info("Matching samples and resolving site paths...")
    matched_rows = []
    unmatched = []

    for _, row in df.iterrows():
        sid = row['sample_id']
        if sid in sample_map:
            sample_dir = sample_map[sid]
            site_path = resolve_site_path(sample_dir, sid)
            if site_path:
                matched_rows.append({
                    '样本编号': sid,
                    'MSI_real': row['MSI_real'],
                    'site_path': site_path,
                    'sample_dir': sample_dir,
                })
            else:
                unmatched.append((sid, 'site_path not found'))
        else:
            unmatched.append((sid, 'not in RIA2026Menglu'))

    result = pd.DataFrame(matched_rows)

    # Merge extra columns
    if 'tumor_content' in df.columns or 'TMB_status' in df.columns:
        extra = df.set_index('sample_id')[['tumor_content', 'TMB_status']].to_dict('index')
        for col in ['tumor_content', 'TMB_status']:
            if col in df.columns:
                result[col] = result['样本编号'].map(lambda x: extra.get(x, {}).get(col))

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Results:")
    logger.info(f"  Excel total: {len(df)}")
    logger.info(f"  Matched: {len(result)}")
    logger.info(f"  Unmatched: {len(unmatched)}")
    logger.info(f"  MSI_real: {result['MSI_real'].value_counts().to_dict()}")
    if 'tumor_content' in result.columns:
        tc = result['tumor_content'].dropna()
        logger.info(f"  tumor_content: n={len(tc)}, mean={tc.mean():.3f}, median={tc.median():.3f}")
    if 'TMB_status' in result.columns:
        logger.info(f"  TMB_status: {result['TMB_status'].value_counts(dropna=False).to_dict()}")

    if unmatched:
        logger.info(f"\n  Unmatched samples (first 10):")
        for sid, reason in unmatched[:10]:
            logger.info(f"    {sid}: {reason}")

    # Save
    os.makedirs(os.path.dirname(OUT_TSV), exist_ok=True)
    result.to_csv(OUT_TSV, sep='\t', index=False)
    logger.info(f"\nSaved to {OUT_TSV}")


if __name__ == '__main__':
    main()
