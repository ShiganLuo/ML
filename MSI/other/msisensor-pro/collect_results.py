# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Collect msisensor-pro results and merge with sample metadata."""

import os
import sys
import argparse
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MSI_RESULT_DIR = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/MSIsensor-pro/msisensor_pro_results"
ALL_INFO_FILE = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/data/all_info.tsv"
OUTPUT_FILE = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/data/msisensor_pro_merged.tsv"


def parse_msi_file(msi_path):
    """Parse a single .msi result file.

    Parameters
    ----------
    msi_path : str
        Path to the .msi file.

    Returns
    -------
    dict or None
        Dict with keys ``total_sites``, ``unstable_sites``, ``msi_pct``,
        or ``None`` if parsing fails.
    """
    try:
        with open(msi_path, encoding="utf-8") as f:
            lines = f.read().strip().split('\n')
        if len(lines) < 2:
            return None
        parts = lines[1].split('\t')
        return {
            "total_sites": int(parts[0]),
            "unstable_sites": int(parts[1]),
            "msi_pct": float(parts[2]),
        }
    except Exception as e:
        logger.warning(f"Failed to parse {msi_path}: {e}")
        return None


def collect_results(result_dir):
    """Collect all .msi results from the result directory.

    Parameters
    ----------
    result_dir : str
        Root directory containing per-sample subdirectories.

    Returns
    -------
    pd.DataFrame
        Columns: ``sample_id``, ``total_sites``, ``unstable_sites``,
        ``msi_pct``.
    """
    rows = []
    count = 0
    for entry in os.scandir(result_dir):
        if entry.is_dir():
            sample_id = entry.name
            msi_file = os.path.join(entry.path, f"{sample_id}.msi")
            if os.path.isfile(msi_file):
                parsed = parse_msi_file(msi_file)
                if parsed is not None:
                    rows.append({"sample_id": sample_id, **parsed})
                    count += 1
            if count % 100 == 0 and count > 0:
                logger.info(f"Processed {count} samples...")
    df = pd.DataFrame(rows)
    logger.info(f"Collected {len(df)} results from {result_dir}")
    return df


def extract_sample_id_from_bam(bam_path):
    """Extract sample_id from bam_path by splitting on ``_cancer``.

    Parameters
    ----------
    bam_path : str
        Path containing ``_cancer`` delimiter, e.g.
        ``/path/to/189001654D_189001655D_cancer.bam``.

    Returns
    -------
    str or None
        The portion before ``_cancer``, or ``None`` if not found.
    """
    if not isinstance(bam_path, str):
        return None
    basename = os.path.basename(bam_path)
    if "_cancer" in basename:
        return basename.split("_cancer")[0]
    return None


def merge_with_metadata(msi_df, all_info_file):
    """Merge MSI results with sample metadata.

    Parameters
    ----------
    msi_df : pd.DataFrame
        MSI results with ``sample_id`` column.
    all_info_file : str
        Path to ``all_info.tsv``.

    Returns
    -------
    pd.DataFrame
        Merged dataframe.
    """
    meta = pd.read_csv(all_info_file, sep='\t')
    meta["sample_id"] = meta["site_feature"].apply(extract_sample_id_from_bam)
    logger.info(f"Metadata: {len(meta)} rows, {meta['sample_id'].notna().sum()} with valid sample_id")

    merged = pd.merge(msi_df, meta, on="sample_id", how="inner")
    logger.info(f"Merged: {len(merged)} rows")
    return merged


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Collect msisensor-pro results")
    parser.add_argument(
        "--result-dir", default=MSI_RESULT_DIR,
        help="Directory containing per-sample result subdirectories",
    )
    parser.add_argument(
        "--all-info", default=ALL_INFO_FILE,
        help="Path to all_info.tsv",
    )
    parser.add_argument(
        "--output", "-o", default=OUTPUT_FILE,
        help="Output TSV path",
    )
    args = parser.parse_args()

    # Collect
    msi_df = collect_results(args.result_dir)
    if msi_df.empty:
        logger.error("No results collected")
        sys.exit(1)

    # Merge
    merged = merge_with_metadata(msi_df, args.all_info)

    # Write
    merged.to_csv(args.output, sep='\t', index=False)
    logger.info(f"Output written to {args.output}")

    # Summary
    print(f"\nSummary:")
    print(f"  Total samples collected: {len(msi_df)}")
    print(f"  Successfully merged:     {len(merged)}")
    if len(merged) > 0:
        print(f"  MSI-H (true):            {len(merged[merged['MSI_real'] == 'MSI-H'])}")
        print(f"  MSS (true):              {len(merged[merged['MSI_real'] == 'MSS'])}")
        print(f"  Mean MSI% (MSI-H):       {merged[merged['MSI_real'] == 'MSI-H']['msi_pct'].mean():.2f}%")
        print(f"  Mean MSI% (MSS):         {merged[merged['MSI_real'] == 'MSS']['msi_pct'].mean():.2f}%")


if __name__ == "__main__":
    main()
