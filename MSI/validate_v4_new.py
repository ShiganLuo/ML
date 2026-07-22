#!/usr/bin/env python3
"""Evaluate xgboost_weighted on v4_new (v4 without v3 duplicates).

Three experiments:
  1. v3 test set: how well the model performs on v3 held-out data
  2. v3→v4_new (v3 thr): apply v3 model + v3 threshold to new v4 samples
  3. v3→v4_new (Youden): apply v3 model, re-optimize threshold on v4_new
  4. v4_new self-eval: train on v4_new (80/20 split), test on held-out

Usage:
    python validate_v4_new.py
"""

import os
import sys
import json
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from MSI import MSIDetectionPipeline, compute_roc, evaluate, find_best_threshold
from compare_features import load_config, _build_pipeline_components, _make_detector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

CONFIG = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/config/compare_features.json"
V3_TSV = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/data/all_info_dedup.tsv"
V4_TSV = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/data/v4_newsequencing.tsv"
OUTPUT_DIR = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/results/v4_new"


def get_v3_sample_ids(v3_path: str) -> set:
    """Extract sample IDs from v3 data."""
    df = pd.read_csv(v3_path, sep='\t')
    ids = set()
    for val in df['site_feature'].dropna():
        sid = os.path.basename(val).split('_cancer')[0]
        ids.add(sid)
    return ids


def get_v4_sample_ids(v4_path: str) -> set:
    """Extract sample IDs from v4 data."""
    df = pd.read_csv(v4_path, sep='\t')
    col = '样本编号' if '样本编号' in df.columns else 'sample_id'
    return set(df[col].astype(str).str.strip())


def load_v4_new(v4_path: str, v3_ids: set) -> pd.DataFrame:
    """Load v4 and remove samples that overlap with v3."""
    df = pd.read_csv(v4_path, sep='\t')
    col = '样本编号' if '样本编号' in df.columns else 'sample_id'
    df[col] = df[col].astype(str).str.strip()
    df = df[~df[col].isin(v3_ids)].copy()
    df.rename(columns={col: 'sample_id'}, inplace=True)
    df['sample_id'] = df['sample_id'].astype(str)
    df = df.set_index('sample_id')
    if 'origin' not in df.columns:
        df['origin'] = 'v4_new'
    if 'cancertype' not in df.columns:
        df['cancertype'] = 'unknown'
    return df


def load_v3_meta(v3_path: str, all_info_v2: str = None) -> pd.DataFrame:
    """Load v3 training metadata."""
    meta = pd.read_csv(v3_path, sep='\t')
    meta['sample_id'] = meta['site_feature'].apply(
        lambda x: os.path.basename(x).split('_cancer')[0] if isinstance(x, str) else None)
    meta = meta.set_index('sample_id')

    if all_info_v2 and os.path.isfile(all_info_v2):
        v2 = pd.read_csv(all_info_v2, sep='\t')
        v2 = v2.set_index('sample_id')
        if 'tumor_content' in v2.columns:
            meta['tumor_content'] = pd.to_numeric(v2['tumor_content'], errors='coerce')
        if 'TMB_status' in v2.columns:
            tmb = v2['TMB_status'].reindex(meta.index)
            for cat in ['TMB-H', 'TMB-L', 'TMB-U']:
                col = cat.replace("-", "_")
                meta[col] = (tmb == cat).astype(float)
                meta.loc[tmb.isna(), col] = np.nan
    return meta


def train_on_v3(cfg: dict, meta: pd.DataFrame) -> dict:
    """Train pipeline on v3 data and return results."""
    strat_cfg = cfg['strategies'][0]
    det_cfg = cfg['detectors'][0]

    fe, locus_sel, feat_sel, sf, train_filter, required_features = _build_pipeline_components(strat_cfg)
    det = _make_detector(det_cfg)

    pipeline = MSIDetectionPipeline(
        feature_extractor=fe, locus_selector=locus_sel,
        feature_selector=feat_sel, sample_filter=sf,
        detector=det, train_filter=train_filter,
        required_features=required_features,
    )

    run_cfg = cfg.get('pipeline', {})
    results = pipeline.run(
        meta,
        n_sigma=run_cfg.get('n_sigma', 3.0),
        site_file_col=run_cfg.get('site_file_col', 'site_feature'),
        test_size=run_cfg.get('test_size', 0.2),
        cache_dir=cfg.get('cache_dir', '/tmp/msi_cache'),
        msi_col=run_cfg.get('msi_col', 'MSI_real'),
        threshold_method=run_cfg.get('threshold_method', 'cv'),
        cv_folds=run_cfg.get('cv_folds', 5),
    )

    return {
        'pipeline': pipeline,
        'fe': fe, 'det': det, 'sf': sf,
        'selected_cols': results['selected_cols'],
        'threshold': results['threshold'],
        'test_results': results['test'],
        'cancer_thresholds': results.get('cancer_thresholds', {}),
    }


def predict_on_new(model: dict, new_df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from new data and predict using trained model."""
    fe = model['fe']
    det = model['det']
    sf = model['sf']
    selected_cols = model['selected_cols']

    site_col = 'site_path' if 'site_path' in new_df.columns else 'site_feature'
    new_features, _ = fe.extract_batch(new_df[site_col].values, new_df.index.values)

    _join_cols = [c for c in ['MSI_real', 'MSI_status', 'origin', 'cancertype', 'MSI_CNC',
                               'tumor_content', 'TMB_H', 'TMB_L', 'TMB_U']
                  if c in new_df.columns]
    new_features = new_features.join(new_df[_join_cols], how='inner')
    if 'MSI_real' in new_features.columns and 'MSI_status' not in new_features.columns:
        new_features.rename(columns={'MSI_real': 'MSI_status'}, inplace=True)

    new_features = sf.filter(new_features)

    available = [c for c in selected_cols if c in new_features.columns]
    missing = [c for c in selected_cols if c not in new_features.columns]
    if missing:
        logger.warning(f"Missing {len(missing)} features: {missing}")

    X = np.nan_to_num(new_features[available].values, nan=0.0)
    if hasattr(det, 'set_feature_names'):
        det.set_feature_names(available)

    scores = det.score(X)
    new_features['score'] = scores
    return new_features


def print_eval(y_true, scores, predictions, label):
    """Print evaluation metrics."""
    fpr, tpr, auc_val = compute_roc(y_true, scores)
    tp = int(((predictions == 'MSI-H') & (y_true == 'MSI-H')).sum())
    fp = int(((predictions == 'MSI-H') & (y_true == 'MSS')).sum())
    fn = int(((predictions == 'MSS') & (y_true == 'MSI-H')).sum())
    tn = int(((predictions == 'MSS') & (y_true == 'MSS')).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = (tp + tn) / len(y_true)
    print(f"  {label:<40} AUC={auc_val:.4f} Sens={sens*100:5.1f}% Spec={spec*100:5.1f}% "
          f"Acc={acc*100:5.1f}% TP={tp} FP={fp} FN={fn} TN={tn}")
    return {'auc': auc_val, 'sens': sens, 'spec': spec, 'acc': acc, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def plot_results(all_experiments: dict, output_dir: str):
    """Plot ROC curves and score distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = plt.cm.tab10

    # ROC
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    for i, (name, exp) in enumerate(all_experiments.items()):
        fpr, tpr, auc_val = compute_roc(exp['y_true'], exp['scores'])
        ax.plot(fpr, tpr, color=colors(i), linewidth=2, label=f"{name} (AUC={auc_val:.3f})")
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('ROC: xgboost_weighted — v3 test / v4_new')
    ax.legend(loc='lower right', fontsize=9, frameon=False)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2)

    # Score distributions
    ax = axes[1]
    for i, (name, exp) in enumerate(all_experiments.items()):
        mss = exp['scores'][exp['y_true'] == 'MSS']
        msih = exp['scores'][exp['y_true'] == 'MSI-H']
        if len(msih) > 0:
            ax.hist(msih, bins=20, alpha=0.4, color=colors(i), histtype='step', linewidth=1.5,
                    label=f"{name} MSI-H")
        if len(mss) > 0:
            ax.hist(mss, bins=20, alpha=0.2, color=colors(i), histtype='stepfilled')
    ax.set_xlabel('Score'); ax.set_ylabel('Count')
    ax.set_title('Score Distributions')
    ax.legend(fontsize=8, ncol=2, frameon=False)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {output_dir}/roc_curves.png")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    route_name = "xgboost_weighted"

    # ── Load data ──
    logger.info("Loading data...")
    v3_ids = get_v3_sample_ids(V3_TSV)
    v4_ids = get_v4_sample_ids(V4_TSV)
    v4_new_ids = v4_ids - v3_ids
    logger.info(f"v3: {len(v3_ids)} samples, v4: {len(v4_ids)} samples, v4_new: {len(v4_new_ids)} (removed {len(v4_ids & v3_ids)} duplicates)")

    v4_new_df = load_v4_new(V4_TSV, v3_ids)
    n_msih = int((v4_new_df['MSI_real'] == 'MSI-H').sum())
    n_mss = int((v4_new_df['MSI_real'] == 'MSS').sum())
    logger.info(f"v4_new: {len(v4_new_df)} samples, MSI-H={n_msih}, MSS={n_mss}")

    # ── Load config ──
    cfg = load_config(CONFIG, route_names=[route_name])[route_name]
    all_info_v2 = cfg.get('all_info_v2')
    meta = load_v3_meta(V3_TSV, all_info_v2)

    # ── Experiment 1: Train on v3, evaluate on v3 test set ──
    print(f"\n{'='*80}")
    print(f"  xgboost_weighted — v4_new validation (no v3 duplicates)")
    print(f"  v4_new: {len(v4_new_df)} samples (MSI-H={n_msih}, MSS={n_mss})")
    print(f"{'='*80}\n")

    logger.info("=== Exp 1: Training on v3 ===")
    model = train_on_v3(cfg, meta)

    v3_test = model['test_results']
    y_true_v3 = v3_test['df']['MSI_status'].values
    scores_v3 = v3_test['scores']
    v3_thr = model['threshold']
    preds_v3 = np.where(scores_v3 >= v3_thr, 'MSI-H', 'MSS')

    print("  --- Experiment 1: v3 test set (held-out 20%) ---")
    r1 = print_eval(y_true_v3, scores_v3, preds_v3,
                    f"v3 test (thr={v3_thr:.4f})")

    # ── Experiment 2: v3 model → v4_new with v3 threshold ──
    logger.info("=== Exp 2: v3 model → v4_new (v3 threshold) ===")
    v4_features = predict_on_new(model, v4_new_df)
    y_true_v4 = v4_features['MSI_status'].values
    scores_v4 = v4_features['score'].values
    preds_v4_v3thr = np.where(scores_v4 >= v3_thr, 'MSI-H', 'MSS')

    print(f"\n  --- Experiment 2: v3→v4_new (v3 thr={v3_thr:.4f}) ---")
    r2 = print_eval(y_true_v4, scores_v4, preds_v4_v3thr,
                    f"v3→v4_new (v3 thr)")

    # ── Experiment 3: v3 model → v4_new with Youden threshold ──
    youden_thr, _, _ = find_best_threshold(y_true_v4, scores_v4)
    preds_v4_youden = np.where(scores_v4 >= youden_thr, 'MSI-H', 'MSS')

    print(f"\n  --- Experiment 3: v3→v4_new (Youden thr={youden_thr:.4f}) ---")
    r3 = print_eval(y_true_v4, scores_v4, preds_v4_youden,
                    f"v3→v4_new (Youden)")

    # ── Experiment 4: v4_new self-eval (80/20) ──
    logger.info("=== Exp 4: v4_new self-eval (80/20) ===")

    # Build fresh pipeline
    strat_cfg = cfg['strategies'][0]
    det_cfg = cfg['detectors'][0]
    fe, locus_sel, feat_sel, sf, train_filter, required_features = _build_pipeline_components(strat_cfg)
    det = _make_detector(det_cfg)
    pipeline = MSIDetectionPipeline(
        feature_extractor=fe, locus_selector=locus_sel,
        feature_selector=feat_sel, sample_filter=sf,
        detector=det, train_filter=train_filter,
        required_features=required_features,
    )

    # Prepare v4_new data for pipeline
    v4_new_for_pipeline = v4_new_df.copy()
    if 'site_path' in v4_new_for_pipeline.columns and 'site_feature' not in v4_new_for_pipeline.columns:
        v4_new_for_pipeline['site_feature'] = v4_new_for_pipeline['site_path']
    if 'MSI_status' not in v4_new_for_pipeline.columns:
        v4_new_for_pipeline['MSI_status'] = v4_new_for_pipeline['MSI_real']
    if 'MSI_CNC' not in v4_new_for_pipeline.columns:
        v4_new_for_pipeline['MSI_CNC'] = v4_new_for_pipeline.get('MSI_real', 'unknown')
    if 'origin' not in v4_new_for_pipeline.columns:
        v4_new_for_pipeline['origin'] = 'v4_new'
    if 'cancertype' not in v4_new_for_pipeline.columns:
        v4_new_for_pipeline['cancertype'] = 'unknown'

    run_cfg = cfg.get('pipeline', {})
    v4_results = pipeline.run(
        v4_new_for_pipeline,
        n_sigma=run_cfg.get('n_sigma', 3.0),
        site_file_col='site_path' if 'site_path' in v4_new_for_pipeline.columns else 'site_feature',
        test_size=0.2,
        cache_dir=None,
        msi_col='MSI_real',
        threshold_method=run_cfg.get('threshold_method', 'cv'),
        cv_folds=run_cfg.get('cv_folds', 5),
    )

    # Use the pipeline's test set results
    v4_test = v4_results['test']
    y_true_v4test = v4_test['df']['MSI_status'].values
    scores_v4test = v4_test['scores']
    v4_thr = v4_results['threshold']
    preds_v4test = np.where(scores_v4test >= v4_thr, 'MSI-H', 'MSS')

    print(f"\n  --- Experiment 4: v4_new self-eval (80/20, thr={v4_thr:.4f}) ---")
    r4 = print_eval(y_true_v4test, scores_v4test, preds_v4test,
                    f"v4_new self-eval (80/20)")

    # ── Summary ──
    print(f"\n{'='*80}")
    print(f"  Summary")
    print(f"{'='*80}")
    print(f"  {'Experiment':<45} {'AUC':>6} {'Sens':>6} {'Spec':>6} {'Acc':>6}")
    print(f"  {'-'*45} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for name, r in [("v3 test (held-out 20%)", r1),
                    ("v3→v4_new (v3 threshold)", r2),
                    ("v3→v4_new (Youden re-threshold)", r3),
                    (f"v4_new self-eval (80/20, n={len(y_true_v4test)})", r4)]:
        print(f"  {name:<45} {r['auc']:.4f} {r['sens']*100:5.1f}% {r['spec']*100:5.1f}% {r['acc']*100:5.1f}%")
    print(f"{'='*80}")

    # ── Plot ──
    all_exp = {
        'v3 test': {'y_true': y_true_v3, 'scores': scores_v3},
        'v3→v4_new (v3 thr)': {'y_true': y_true_v4, 'scores': scores_v4},
        'v4_new self-eval': {'y_true': y_true_v4test, 'scores': scores_v4test},
    }
    plot_results(all_exp, OUTPUT_DIR)

    # ── Save ──
    summary_rows = [
        {'experiment': 'v3 test', 'threshold': v3_thr, **r1},
        {'experiment': 'v3→v4_new (v3 thr)', 'threshold': v3_thr, **r2},
        {'experiment': 'v3→v4_new (Youden)', 'threshold': youden_thr, **r3},
        {'experiment': 'v4_new self-eval', 'threshold': v4_thr, **r4},
    ]
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_DIR, 'summary.tsv'), sep='\t', index=False)
    logger.info(f"Saved {OUTPUT_DIR}/summary.tsv")


if __name__ == '__main__':
    main()
