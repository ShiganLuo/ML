# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""MSI detection using msisensor-pro msi_pct with n_sigma threshold.

Usage:
    python msi_pct_train.py --input msisensor_pro_merged.tsv -o output/
"""

import os
import sys
import argparse
import logging
from scipy.integrate import trapezoid
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_roc(y_true, scores, pos_label='MSI-H'):
    """Compute ROC curve and AUC."""
    
    y_bin = (np.array(y_true) == pos_label).astype(int)
    idx = np.argsort(scores)[::-1]
    y_s = y_bin[idx]
    tps = np.cumsum(y_s)
    fps = np.cumsum(1 - y_s)
    tpr = np.concatenate([[0], tps / tps[-1]])
    fpr = np.concatenate([[0], fps / fps[-1]])
    return fpr, tpr, trapezoid(tpr, fpr)


def find_best_threshold(y_true, scores, pos_label='MSI-H'):
    """Find threshold maximizing Youden's J (TPR - FPR)."""
    y_bin = (np.array(y_true) == pos_label).astype(int)
    idx = np.argsort(scores)[::-1]
    s = scores[idx]
    y_s = y_bin[idx]
    tps = np.cumsum(y_s)
    fps = np.cumsum(1 - y_s)
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]
    j = tpr - fpr
    best = np.argmax(j)
    return s[best], tpr[best], fpr[best]


def evaluate(y_true, scores, threshold, pos='MSI-H'):
    """Evaluate predictions."""
    pred = np.where(scores >= threshold, pos, 'MSS')
    y = np.array(y_true)
    tp = int(np.sum((pred == pos) & (y == pos)))
    tn = int(np.sum((pred != pos) & (y != pos)))
    fp = int(np.sum((pred == pos) & (y != pos)))
    fn = int(np.sum((pred != pos) & (y == pos)))
    return {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'sens': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'spec': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'acc': (tp + tn) / len(y),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="MSI detection using msi_pct")
    parser.add_argument("--input", required=True, help="Path to msisensor_pro_merged.tsv")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory")
    parser.add_argument("--msi-col", default="MSI_real", help="Column for MSI status (default: MSI_real)")
    parser.add_argument("--n-sigma", type=float, default=3.0, help="Threshold = mean + n*std (default: 3.0)")
    parser.add_argument("--threshold-method", choices=['nsigma', 'youden', 'fixed'],
                         default='nsigma', help="Threshold method (default: nsigma)")
    parser.add_argument("--threshold", type=float, default=None,
                         help="Fixed threshold (only for --threshold-method fixed)")
    parser.add_argument("--test-size", type=float, default=0.2, help="BL test fraction (default: 0.2)")
    parser.add_argument("--use-renqun-mss", action="store_true", default=False,
                         help="Include renqun MSS samples in training")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from sklearn.model_selection import train_test_split

    # Load data
    df = pd.read_csv(args.input, sep='\t')
    logger.info(f"Loaded {len(df)} rows")
    if args.msi_col != 'MSI_status':
        df.rename(columns={args.msi_col: 'MSI_status'}, inplace=True)
    df = df.set_index('sample_id')

    # Filter valid samples (renqun doesn't need MSI_status)
    df = df.dropna(subset=['msi_pct', 'origin'])
    logger.info(f"After filtering: {len(df)} rows")

    # Split BL into train/test
    bl_df = df[df['origin'] == 'BL']
    bl_train_df, bl_test_df = train_test_split(
        bl_df, test_size=args.test_size, random_state=42, stratify=bl_df['MSI_status']
    )
    logger.info(f"BL split: train={len(bl_train_df)}, test={len(bl_test_df)}")

    # Train set: BL train MSS (+ renqun MSS if requested)
    train_df = bl_train_df[bl_train_df['MSI_status'] == 'MSS']
    if args.use_renqun_mss:
        renqun_mss = df[(df['origin'] == 'renqun') & (df['MSI_status'] == 'MSS')]
        train_df = pd.concat([train_df, renqun_mss])
        logger.info(f"Added {len(renqun_mss)} renqun MSS to training")

    val_df = df[df['origin'] == 'PCR']
    predict_df = df[df['origin'] == 'renqun']

    logger.info(f"Train (MSS): {len(train_df)}, BL test: {len(bl_test_df)}, "
                f"PCR val: {len(val_df)}, Renqun: {len(predict_df)}")

    # Set threshold
    mss_scores = train_df['msi_pct'].values

    if args.threshold_method == 'nsigma':
        threshold = float(np.mean(mss_scores) + args.n_sigma * np.std(mss_scores))
        logger.info(f"Threshold (n_sigma={args.n_sigma}): mean={np.mean(mss_scores):.3f}, "
                    f"std={np.std(mss_scores):.3f}, thr={threshold:.3f}")
    elif args.threshold_method == 'youden':
        # Use all BL train data (MSS + MSI-H) with labels
        bl_train_all_scores = bl_train_df['msi_pct'].values
        threshold, _, _ = find_best_threshold(bl_train_df['MSI_status'].values, bl_train_all_scores)
        logger.info(f"Threshold (Youden's J): thr={threshold:.3f}")
    elif args.threshold_method == 'fixed':
        if args.threshold is None:
            logger.error("--threshold must be set when using --threshold-method fixed")
            sys.exit(1)
        threshold = float(args.threshold)
        logger.info(f"Threshold (fixed): thr={threshold:.3f}")

    # Evaluate on BL test
    bl_test_scores = bl_test_df['msi_pct'].values
    _, _, bl_test_auc = compute_roc(bl_test_df['MSI_status'].values, bl_test_scores)
    bl_test_eval = evaluate(bl_test_df['MSI_status'].values, bl_test_scores, threshold)

    # Evaluate on PCR val
    val_auc, val_eval = None, None
    if len(val_df) > 0:
        val_scores = val_df['msi_pct'].values
        _, _, val_auc = compute_roc(val_df['MSI_status'].values, val_scores)
        val_eval = evaluate(val_df['MSI_status'].values, val_scores, threshold)

    # Predict renqun
    predict_scores = predict_df['msi_pct'].values
    predict_labels = np.where(predict_scores >= threshold, 'MSI-H', 'MSS')

    # --- Report ---
    lines = []
    lines.append(f"{'='*70}")
    lines.append(f"  Method: msi_pct + {args.threshold_method}")
    lines.append(f"  Threshold: {threshold:.3f} (n_sigma={args.n_sigma})")
    lines.append(f"  MSS mean={np.mean(mss_scores):.3f}, std={np.std(mss_scores):.3f}")
    lines.append(f"{'='*70}")

    lines.append(f"\n  BL Test Set (held-out)")
    lines.append(f"  AUC   = {bl_test_auc:.4f}")
    lines.append(f"  Sens  = {bl_test_eval['sens']:.4f}")
    lines.append(f"  Spec  = {bl_test_eval['spec']:.4f}")
    lines.append(f"  Acc   = {bl_test_eval['acc']:.4f}")
    lines.append(f"  TP={bl_test_eval['TP']} TN={bl_test_eval['TN']} FP={bl_test_eval['FP']} FN={bl_test_eval['FN']}")
    if 'cancertype' in bl_test_df.columns:
        lines.append(f"  --- By cancer type ---")
        bl_test_out = bl_test_df.copy()
        bl_test_out['predicted'] = np.where(bl_test_scores >= threshold, 'MSI-H', 'MSS')
        for ct in sorted(bl_test_out['cancertype'].dropna().unique()):
            mask = bl_test_out['cancertype'] == ct
            n = mask.sum()
            n_tp = ((bl_test_out.loc[mask, 'predicted'] == 'MSI-H') & (bl_test_out.loc[mask, 'MSI_status'] == 'MSI-H')).sum()
            n_fp = ((bl_test_out.loc[mask, 'predicted'] == 'MSI-H') & (bl_test_out.loc[mask, 'MSI_status'] == 'MSS')).sum()
            n_fn = ((bl_test_out.loc[mask, 'predicted'] == 'MSS') & (bl_test_out.loc[mask, 'MSI_status'] == 'MSI-H')).sum()
            n_tn = ((bl_test_out.loc[mask, 'predicted'] == 'MSS') & (bl_test_out.loc[mask, 'MSI_status'] == 'MSS')).sum()
            lines.append(f"  {str(ct):15s}: n={n:>4d} TP={n_tp} FP={n_fp} FN={n_fn} TN={n_tn}")

    if val_eval:
        lines.append(f"\n  PCR Validation (independent)")
        lines.append(f"  AUC   = {val_auc:.4f}")
        lines.append(f"  Sens  = {val_eval['sens']:.4f}")
        lines.append(f"  Spec  = {val_eval['spec']:.4f}")
        lines.append(f"  Acc   = {val_eval['acc']:.4f}")
        lines.append(f"  TP={val_eval['TP']} TN={val_eval['TN']} FP={val_eval['FP']} FN={val_eval['FN']}")
        if 'cancertype' in val_df.columns:
            lines.append(f"  --- By cancer type ---")
            val_out = val_df.copy()
            val_out['predicted'] = np.where(val_df['msi_pct'].values >= threshold, 'MSI-H', 'MSS')
            for ct in sorted(val_out['cancertype'].dropna().unique()):
                mask = val_out['cancertype'] == ct
                n = mask.sum()
                n_tp = ((val_out.loc[mask, 'predicted'] == 'MSI-H') & (val_out.loc[mask, 'MSI_status'] == 'MSI-H')).sum()
                n_fp = ((val_out.loc[mask, 'predicted'] == 'MSI-H') & (val_out.loc[mask, 'MSI_status'] == 'MSS')).sum()
                n_fn = ((val_out.loc[mask, 'predicted'] == 'MSS') & (val_out.loc[mask, 'MSI_status'] == 'MSI-H')).sum()
                n_tn = ((val_out.loc[mask, 'predicted'] == 'MSS') & (val_out.loc[mask, 'MSI_status'] == 'MSS')).sum()
                lines.append(f"  {str(ct):15s}: n={n:>4d} TP={n_tp} FP={n_fp} FN={n_fn} TN={n_tn}")

    lines.append(f"\n  Renqun Prediction")
    lines.append(f"  Total: {len(predict_df)}")
    lines.append(f"  MSI-H: {(predict_labels == 'MSI-H').sum()} ({(predict_labels == 'MSI-H').mean()*100:.1f}%)")
    for ct in sorted(predict_df['cancertype'].dropna().unique()):
        mask = predict_df['cancertype'] == ct
        n = mask.sum()
        n_h = (predict_labels[mask.values] == 'MSI-H').sum()
        lines.append(f"  {str(ct):15s}: {n:>5d}, MSI-H={n_h} ({n_h/n*100:.1f}%)")
    lines.append(f"{'='*70}")

    report = '\n'.join(lines)
    print(report)
    with open(os.path.join(args.output_dir, 'report.txt'), 'w') as f:
        f.write(report + '\n')

    # Save scores
    bl_train_out = bl_train_df[['MSI_status']].copy()
    if 'cancertype' in bl_train_df.columns:
        bl_train_out['cancertype'] = bl_train_df['cancertype']
    bl_train_out['score'] = bl_train_df['msi_pct']
    bl_train_out['predicted'] = np.where(bl_train_df['msi_pct'] >= threshold, 'MSI-H', 'MSS')
    bl_train_out.to_csv(os.path.join(args.output_dir, 'bl_train_scores.tsv'), sep='\t')

    bl_test_out_save = bl_test_df[['MSI_status']].copy()
    if 'cancertype' in bl_test_df.columns:
        bl_test_out_save['cancertype'] = bl_test_df['cancertype']
    bl_test_out_save['score'] = bl_test_df['msi_pct']
    bl_test_out_save['predicted'] = np.where(bl_test_scores >= threshold, 'MSI-H', 'MSS')
    bl_test_out_save.to_csv(os.path.join(args.output_dir, 'bl_test_scores.tsv'), sep='\t')

    if len(val_df) > 0:
        val_out_save = val_df[['MSI_status']].copy()
        if 'cancertype' in val_df.columns:
            val_out_save['cancertype'] = val_df['cancertype']
        val_out_save['score'] = val_df['msi_pct']
        val_out_save['predicted'] = np.where(val_df['msi_pct'].values >= threshold, 'MSI-H', 'MSS')
        val_out_save.to_csv(os.path.join(args.output_dir, 'pcr_val_scores.tsv'), sep='\t')

    predict_out = predict_df[['MSI_status', 'cancertype']].copy()
    predict_out['score'] = predict_scores
    predict_out['predicted'] = predict_labels
    predict_out.to_csv(os.path.join(args.output_dir, 'predictions.tsv'), sep='\t')

    # Misclassified
    misclassified = []
    for name, out_df in [('bl_test', bl_test_out_save),
                         ('pcr_val', val_out_save if len(val_df) > 0 else None)]:
        if out_df is None:
            continue
        wrong = out_df[out_df['MSI_status'] != out_df['predicted']]
        if len(wrong) > 0:
            wrong = wrong.copy()
            wrong['dataset'] = name
            misclassified.append(wrong)
    if misclassified:
        mc_df = pd.concat(misclassified)
        mc_df.to_csv(os.path.join(args.output_dir, 'misclassified.tsv'), sep='\t')
        logger.info(f"Misclassified: {len(mc_df)}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # ROC (BL test + PCR val)
    ax = axes[0]
    fpr, tpr, _ = compute_roc(bl_test_df['MSI_status'].values, bl_test_scores)
    ax.plot(fpr, tpr, 'r-', lw=2, label=f"BL test AUC={bl_test_auc:.3f}")
    if val_auc:
        fpr_v, tpr_v, _ = compute_roc(val_df['MSI_status'].values, val_df['msi_pct'].values)
        ax.plot(fpr_v, tpr_v, 'g-.', lw=1.5, label=f"PCR val AUC={val_auc:.3f}")
    ax.plot([0,1],[0,1],'k--', lw=0.5)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC')
    ax.legend()

    # BL test scores
    ax = axes[1]
    mss = bl_test_df[bl_test_df['MSI_status'] == 'MSS']['msi_pct'].values
    msih = bl_test_df[bl_test_df['MSI_status'] == 'MSI-H']['msi_pct'].values
    ax.hist(mss, bins=40, alpha=0.5, color='blue', label='MSS')
    ax.hist(msih, bins=40, alpha=0.5, color='red', label='MSI-H')
    ax.axvline(threshold, color='green', linestyle='--', label=f'Thr={threshold:.2f}')
    ax.set_xlabel('msi_pct')
    ax.set_title('BL Test Scores')
    ax.legend()

    # PCR val scores
    ax = axes[2]
    if len(val_df) > 0:
        val_mss = val_df[val_df['MSI_status'] == 'MSS']['msi_pct'].values
        val_msih = val_df[val_df['MSI_status'] == 'MSI-H']['msi_pct'].values
        ax.hist(val_mss, bins=40, alpha=0.5, color='blue', label='MSS')
        ax.hist(val_msih, bins=40, alpha=0.5, color='red', label='MSI-H')
        ax.axvline(threshold, color='green', linestyle='--', label=f'Thr={threshold:.2f}')
        ax.set_xlabel('msi_pct')
        ax.set_title(f"PCR Val (AUC={val_auc:.3f})")
        ax.legend()
    else:
        ax.set_title('PCR Val (N/A)')

    # Renqun scores
    ax = axes[3]
    ax.hist(predict_scores[predict_labels == 'MSS'], bins=40, alpha=0.5, color='blue', label='Pred MSS')
    ax.hist(predict_scores[predict_labels == 'MSI-H'], bins=40, alpha=0.5, color='red', label='Pred MSI-H')
    ax.axvline(threshold, color='green', linestyle='--', label=f'Thr={threshold:.2f}')
    ax.set_xlabel('msi_pct')
    ax.set_title('Renqun Scores')
    ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'results.png'), dpi=150)
    plt.close()

    print(f"\nOutput: {args.output_dir}/")


if __name__ == "__main__":
    main()
