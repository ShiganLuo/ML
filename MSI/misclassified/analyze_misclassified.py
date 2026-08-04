#!/usr/bin/env python3
# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Analyze misclassified vs correctly classified samples:
tumor purity and TMB differences."""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'

PRED_PATH = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/results/v4/predictions/xgboost_weighted_v3test.tsv"
V2_PATH = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/data/all_info_v2.tsv"
OUT_DIR = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/results/v4"
ROUTE_NAME = "xgboost_weighted"


def extract_fd(sid: str) -> str:
    parts = sid.split('_')
    for p in parts:
        if p.endswith('FD') or p.endswith('F1D') or p.endswith('F2D'):
            return p
    return sid


def p_to_stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'ns'


def draw_sig_line(ax, x1, x2, y, h, p):
    stars = p_to_stars(p)
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, color='black')
    ax.text((x1 + x2) / 2, y + h, stars, ha='center', va='bottom', fontsize=11, fontweight='bold')


def draw_stacked_bar_with_labels(ax, x_positions, ct_data, tmb_order, tmb_colors, width=0.5):
    """Draw stacked bar chart and label each segment correctly.
    Thin segments get labels placed to the right of the bar."""
    # First pass: draw all bars
    bottom = np.zeros(len(x_positions))
    segments = []
    for ts in tmb_order:
        vals = ct_data[ts].values.astype(float)
        ax.bar(x_positions, vals, width, bottom=bottom, label=ts,
               color=tmb_colors.get(ts, '#999'), edgecolor='white', lw=0.5)
        for i, v in enumerate(vals):
            if v > 0:
                segments.append({
                    'x_idx': i, 'x': x_positions[i], 'b': bottom[i], 'h': v,
                    'text': str(int(v)), 'dark': ts == 'TMB-H'
                })
        bottom += vals

    # Second pass: measure pixels and place labels
    fig = ax.get_figure()
    fig.canvas.draw()
    ax_h_px = ax.bbox.height
    ylim = ax.get_ylim()
    data_per_px = (ylim[1] - ylim[0]) / ax_h_px
    min_h_px = 14

    # Group segments by x_idx
    from collections import defaultdict
    by_group = defaultdict(list)
    for seg in segments:
        by_group[seg['x_idx']].append(seg)

    for x_idx, group_segs in by_group.items():
        inside = []
        outside = []
        for seg in group_segs:
            if seg['h'] / data_per_px >= min_h_px:
                inside.append(seg)
            else:
                outside.append(seg)

        # Place inside labels
        for seg in inside:
            color = 'white' if seg['dark'] else 'black'
            ax.text(seg['x'], seg['b'] + seg['h'] / 2, seg['text'],
                    ha='center', va='center', fontsize=10, fontweight='bold', color=color)

        # Place outside labels at each segment's midpoint, to the right
        for seg in outside:
            ax.text(seg['x'] + width / 2 + 0.06, seg['b'] + seg['h'] / 2, seg['text'],
                    ha='left', va='center', fontsize=9, fontweight='bold', color='black')


def main():
    pred = pd.read_csv(PRED_PATH, sep='\t')
    v2 = pd.read_csv(V2_PATH, sep='\t')

    pred['fd_id'] = pred['sample_id'].apply(extract_fd)
    v2_dedup = v2.drop_duplicates('sample_id')[['sample_id', 'tumor_content', 'TMB_status']]
    v2_dedup = v2_dedup.rename(columns={'sample_id': 'fd_id'})

    merged = pred.merge(v2_dedup, on='fd_id', how='left')
    merged['tumor_content'] = pd.to_numeric(merged['tumor_content'], errors='coerce')

    def subgroup(row):
        if row['MSI_status'] == 'MSI-H' and row['predicted'] == 'MSI-H': return 'TP'
        elif row['MSI_status'] == 'MSI-H' and row['predicted'] == 'MSS': return 'FN'
        elif row['MSI_status'] == 'MSS' and row['predicted'] == 'MSS': return 'TN'
        else: return 'FP'
    merged['subgroup'] = merged.apply(subgroup, axis=1)

    # ── Tumor Purity ──
    purity = merged.dropna(subset=['tumor_content'])
    fn_p = purity[purity['subgroup'] == 'FN']['tumor_content']
    tp_p = purity[purity['subgroup'] == 'TP']['tumor_content']
    fp_p = purity[purity['subgroup'] == 'FP']['tumor_content']
    tn_p = purity[purity['subgroup'] == 'TN']['tumor_content']

    p_fn_tp = stats.mannwhitneyu(fn_p, tp_p, alternative='two-sided').pvalue if len(fn_p) > 0 and len(tp_p) > 0 else 1.0
    p_fp_tn = stats.mannwhitneyu(fp_p, tn_p, alternative='two-sided').pvalue if len(fp_p) > 0 and len(tn_p) > 0 else 1.0

    # ── TMB ──
    tmb = merged.dropna(subset=['TMB_status'])
    tmb_order = ['TMB-H', 'TMB-L', 'TMB-U']
    tmb_colors = {'TMB-H': '#F44336', 'TMB-L': '#4CAF50', 'TMB-U': '#9E9E9E'}

    msih_tmb = tmb[tmb['MSI_status'] == 'MSI-H']
    mss_tmb = tmb[tmb['MSI_status'] == 'MSS']
    ct_msih = pd.crosstab(msih_tmb['subgroup'], msih_tmb['TMB_status'])
    ct_mss = pd.crosstab(mss_tmb['subgroup'], mss_tmb['TMB_status'])

    p_msih_chi = stats.chi2_contingency(ct_msih)[1] if ct_msih.shape[0] == 2 and ct_msih.shape[1] >= 2 else 1.0
    p_mss_chi = stats.chi2_contingency(ct_mss)[1] if ct_mss.shape[0] == 2 and ct_mss.shape[1] >= 2 else 1.0

    # ── Plots ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Misclassified vs Correct: Tumor Purity & TMB ({ROUTE_NAME})', fontsize=14, fontweight='bold')

    # ── 3a. Purity: MSI-H (TP vs FN) ──
    ax = axes[0][0]
    msih_purity = purity[purity['MSI_status'] == 'MSI-H']
    order_msih = ['TP', 'FN']
    palette_msih = {'TP': '#2196F3', 'FN': '#F44336'}
    sns.violinplot(data=msih_purity, x='subgroup', y='tumor_content', order=order_msih,
                   palette=palette_msih, cut=0, inner='box', ax=ax)
    for i, label in enumerate(order_msih):
        n = (msih_purity['subgroup'] == label).sum()
        ymax = msih_purity[msih_purity['subgroup'] == label]['tumor_content'].max()
        ax.text(i, ymax + 3, f'n={n}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ymax_all = msih_purity['tumor_content'].max()
    sig_y = ymax_all * 1.25
    draw_sig_line(ax, 0, 1, sig_y, ymax_all * 0.05, p_fn_tp)
    ax.set_ylim(-5, sig_y + ymax_all * 0.3)
    ax.set_xlabel('')
    ax.set_ylabel('Tumor Purity (%)')
    ax.set_title('MSI-H: TP vs FN (purity)')

    # ── 3b. Purity: MSS (TN vs FP) ──
    ax = axes[0][1]
    mss_purity = purity[purity['MSI_status'] == 'MSS']
    order_mss = ['TN', 'FP']
    palette_mss = {'TN': '#4CAF50', 'FP': '#FF9800'}
    sns.violinplot(data=mss_purity, x='subgroup', y='tumor_content', order=order_mss,
                   palette=palette_mss, cut=0, inner='box', ax=ax)
    for i, label in enumerate(order_mss):
        n = (mss_purity['subgroup'] == label).sum()
        ymax = mss_purity[mss_purity['subgroup'] == label]['tumor_content'].max()
        ax.text(i, ymax + 3, f'n={n}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ymax_all = mss_purity['tumor_content'].max()
    sig_y = ymax_all * 1.25
    draw_sig_line(ax, 0, 1, sig_y, ymax_all * 0.05, p_fp_tn)
    ax.set_ylim(-5, sig_y + ymax_all * 0.3)
    ax.set_xlabel('')
    ax.set_ylabel('Tumor Purity (%)')
    ax.set_title('MSS: TN vs FP (purity)')

    # ── 3c. TMB: MSI-H ──
    ax = axes[1][0]
    order_msih_grp = ['TP', 'FN']
    ct_msih_plot = ct_msih.reindex(index=order_msih_grp, columns=tmb_order, fill_value=0)
    x_msih = np.arange(len(order_msih_grp))
    draw_stacked_bar_with_labels(ax, x_msih, ct_msih_plot, tmb_order, tmb_colors)
    y_top_msih = ct_msih_plot.values.sum(axis=1).max()
    sig_y_msih = y_top_msih * 1.15
    draw_sig_line(ax, 0, 1, sig_y_msih, y_top_msih * 0.04, p_msih_chi)
    ax.set_xticks(x_msih)
    ax.set_xticklabels([f'{l}\n(n={(msih_tmb["subgroup"]==l).sum()})' for l in order_msih_grp], fontsize=10)
    ax.set_ylim(0, sig_y_msih + y_top_msih * 0.2)
    ax.set_ylabel('Count')
    ax.set_title('MSI-H: TP vs FN (TMB)')
    ax.legend(fontsize=8)

    # ── 3d. TMB: MSS ──
    ax = axes[1][1]
    order_mss_grp = ['TN', 'FP']
    ct_mss_plot = ct_mss.reindex(index=order_mss_grp, columns=tmb_order, fill_value=0)
    x_mss = np.arange(len(order_mss_grp))
    draw_stacked_bar_with_labels(ax, x_mss, ct_mss_plot, tmb_order, tmb_colors)
    y_top_mss = ct_mss_plot.values.sum(axis=1).max()
    sig_y_mss = y_top_mss * 1.15
    draw_sig_line(ax, 0, 1, sig_y_mss, y_top_mss * 0.04, p_mss_chi)
    ax.set_xticks(x_mss)
    ax.set_xticklabels([f'{l}\n(n={(mss_tmb["subgroup"]==l).sum()})' for l in order_mss_grp], fontsize=10)
    ax.set_ylim(0, sig_y_mss + y_top_mss * 0.2)
    ax.set_ylabel('Count')
    ax.set_title('MSS: TN vs FP (TMB)')
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT_DIR, 'misclassified_analysis.png'), dpi=300)
    plt.close(fig)
    print(f"Plot saved: {os.path.join(OUT_DIR, 'misclassified_analysis.png')}")

    merged.to_csv(os.path.join(OUT_DIR, 'misclassified_analysis.tsv'), sep='\t', index=False)
    print(f"Data saved: {os.path.join(OUT_DIR, 'misclassified_analysis.tsv')}")


if __name__ == '__main__':
    main()
