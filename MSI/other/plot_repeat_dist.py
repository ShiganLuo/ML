# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Visualize repeat length distributions for MSS vs MSI-H samples.

Usage:
    python plot_repeat_dist.py --all-info all_info.tsv -o output/
"""

import os
import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# site.txt column indices (headerless after skiprows=1)
COL_CHROM = 0
COL_POS = 1
COL_REPEAT_TIMES = 4
COL_REPEAT_DICT = 10
COL_DEPTH = 11


def parse_repeat_dict(dist_str: str) -> dict:
    """Parse repeat_dict string to {repeat_times: count}."""
    counts = {}
    for item in str(dist_str).split(','):
        parts = item.split(':')
        if len(parts) == 2:
            try:
                counts[int(parts[0])] = int(parts[1])
            except ValueError:
                continue
    return counts


def load_sample_loci(site_file: str, min_depth: int = 10) -> list:
    """Load all locus distributions from one site file."""
    if not os.path.isfile(site_file):
        return []
    try:
        df = pd.read_csv(site_file, sep='\t', header=None, skiprows=1)
    except Exception:
        return []

    loci = []
    for _, row in df.iterrows():
        try:
            depth = int(row[COL_DEPTH])
            if depth < min_depth:
                continue
            counts = parse_repeat_dict(row[COL_REPEAT_DICT])
            if not counts:
                continue
            loci.append({
                'chrom': str(row[COL_CHROM]),
                'pos': int(row[COL_POS]),
                'repeat_times': int(row[COL_REPEAT_TIMES]),
                'depth': depth,
                'counts': counts,
            })
        except Exception:
            continue
    return loci


def plot_all_loci_for_sample(ax, loci: list, sample_id: str = "", max_loci: int = None):
    """Plot all locus distributions for one sample as stacked bars.

    Each locus is a horizontal row. x-axis = shift (obs - ref),
    color intensity = frequency of that shift.
    """
    if max_loci:
        loci = loci[:max_loci]

    n_loci = len(loci)
    if n_loci == 0:
        ax.set_title(f'{sample_id} (no loci)')
        return

    # Determine shift range
    all_shifts = set()
    for locus in loci:
        ref = locus['repeat_times']
        for rt in locus['counts']:
            all_shifts.add(rt - ref)

    if not all_shifts:
        return

    min_shift = min(all_shifts)
    max_shift = max(all_shifts)
    shift_range = max_shift - min_shift + 1

    # Build heatmap matrix: rows=loci, cols=shifts
    matrix = np.zeros((n_loci, shift_range))
    for i, locus in enumerate(loci):
        ref = locus['repeat_times']
        total = sum(locus['counts'].values())
        for rt, count in locus['counts'].items():
            j = rt - ref - min_shift
            matrix[i, j] = count / total

    extent = [min_shift - 0.5, max_shift + 0.5, n_loci - 0.5, -0.5]
    ax.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest',
              extent=extent, vmin=0, vmax=1)
    ax.axvline(0, color='green', linestyle='--', alpha=0.5, lw=0.8)
    ax.set_xlabel('Repeat count shift (obs - ref)')
    ax.set_ylabel('Locus index')
    ax.set_title(f'{sample_id} (n={n_loci} loci)', fontsize=8)


def plot_aggregated_distribution(ax, loci: list, label: str = "",
                                 bin_range: tuple = (-5, 5)):
    """Plot aggregated (repeat_times - ref) distribution across all loci."""
    # Collect shifts and weights instead of expanding individual reads
    shifts = []
    weights = []
    for locus in loci:
        ref = locus['repeat_times']
        total = sum(locus['counts'].values())
        for rt, count in locus['counts'].items():
            shift = rt - ref
            shifts.append(shift)
            weights.append(count / total)

    shifts = np.array(shifts)
    weights = np.array(weights)
    bins = np.arange(bin_range[0] - 0.5, bin_range[1] + 1.5, 1)
    ax.hist(shifts, bins=bins, weights=weights, density=True, alpha=0.7,
            edgecolor='black', linewidth=0.5,
            label=f'{label} loci={len(loci)}')
    ax.axvline(0, color='green', linestyle='--', alpha=0.5, label='ref (shift=0)')
    ax.set_xlabel('Repeat count shift (obs - ref)')
    ax.set_ylabel('Density')
    ax.set_title(f'{label} (n={len(loci)} loci)', fontsize=9)
    ax.legend(fontsize=7)


def main():
    parser = argparse.ArgumentParser(description="Plot repeat length distributions")
    parser.add_argument("--all-info", required=True, help="Path to all_info.tsv")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory")
    parser.add_argument("--msi-col", default="MSI_real",
                        help="Column name for MSI status (default: MSI_real)")
    parser.add_argument("--site-col", default="site_feature",
                        help="Column name for site file paths (default: site_feature)")
    parser.add_argument("--min-depth", type=int, default=10,
                        help="Min depth to include a locus (default: 10)")
    parser.add_argument("--n-mss", type=int, default=3,
                        help="Number of MSS samples to plot (default: 3)")
    parser.add_argument("--n-msi", type=int, default=3,
                        help="Number of MSI-H samples to plot (default: 3)")
    parser.add_argument("--max-loci", type=int, default=None,
                        help="Max loci per sample in heatmap (None=all)")
    parser.add_argument("--bin-range", type=int, nargs=2, default=[-5, 5],
                        help="Shift range for aggregated plot (default: -5 5)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load metadata
    meta = pd.read_csv(args.all_info, sep='\t')
    meta = meta.loc[meta["origin"] == "BL", :].dropna()
    logger.info(f"Loaded {len(meta)} rows")

    mss_files = meta[meta[args.msi_col] == 'MSS'][args.site_col].dropna().head(args.n_mss).tolist()
    msi_files = meta[meta[args.msi_col] == 'MSI-H'][args.site_col].dropna().head(args.n_msi).tolist()
    logger.info(f"MSS samples: {len(mss_files)}, MSI-H samples: {len(msi_files)}")

    # --- Plot 1: All loci per sample (heatmap) ---
    n_rows = len(mss_files) + len(msi_files)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows))
    if n_rows == 1:
        axes = [axes]

    row = 0
    for fpath in mss_files:
        sid = os.path.basename(fpath).split('.')[0]
        loci = load_sample_loci(fpath, min_depth=args.min_depth)
        plot_all_loci_for_sample(axes[row], loci, sample_id=f'MSS {sid}',
                                 max_loci=args.max_loci)
        row += 1

    for fpath in msi_files:
        sid = os.path.basename(fpath).split('.')[0]
        loci = load_sample_loci(fpath, min_depth=args.min_depth)
        plot_all_loci_for_sample(axes[row], loci, sample_id=f'MSI-H {sid}',
                                 max_loci=args.max_loci)
        row += 1

    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'all_loci_heatmap.png'), dpi=150)
    plt.close()
    logger.info("Saved all_loci_heatmap.png")

    # --- Plot 2: Aggregated distributions ---
    logger.info("Loading all loci for aggregated plot...")
    all_mss_loci = []
    for fpath in meta[meta[args.msi_col] == 'MSS'][args.site_col].dropna().tolist():
        all_mss_loci.extend(load_sample_loci(fpath, min_depth=args.min_depth))

    all_msi_loci = []
    for fpath in meta[meta[args.msi_col] == 'MSI-H'][args.site_col].dropna().tolist():
        all_msi_loci.extend(load_sample_loci(fpath, min_depth=args.min_depth))

    logger.info(f"Total MSS loci: {len(all_mss_loci)}, MSI-H loci: {len(all_msi_loci)}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bin_range = tuple(args.bin_range)
    plot_aggregated_distribution(axes[0], all_mss_loci, label='MSS', bin_range=bin_range)
    plot_aggregated_distribution(axes[1], all_msi_loci, label='MSI-H', bin_range=bin_range)
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'aggregated_dist.png'), dpi=150)
    plt.close()
    logger.info("Saved aggregated_dist.png")

    print(f"\nDone. Output: {args.output_dir}/")
    print(f"  all_loci_heatmap.png  - Heatmap of all loci per sample")
    print(f"  aggregated_dist.png   - Aggregated shift distributions")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    main()
