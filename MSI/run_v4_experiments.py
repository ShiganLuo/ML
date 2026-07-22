#!/usr/bin/env python3
# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Run multiple MSI routes on v4 data with detailed output.

Experiments:
  1. v3 train → v4 test (cross-reagent evaluation)
  2. v4 self-evaluation (80/20 split)

For each route:
  - Features used
  - Configuration details
  - Per-sample predictions
  - Performance metrics (AUC, Sens, Spec, Acc)
  - ROC curve

Output:
  - experiment_summary.tsv (all routes, all experiments)
  - predictions/<route>_v3to4.tsv (per-sample predictions, v3→v4)
  - predictions/<route>_v4self.tsv (per-sample predictions, v4 self-eval)
  - features/<route>.txt (selected features)
  - config/<route>.json (route configuration)
  - roc_all_routes.png (combined ROC)
  - logs/<route>.log (detailed log per route)
"""

import os
import sys
import json
import copy
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from MSI import MSIDetectionPipeline, compute_roc, evaluate
from compare_features import load_config, _build_pipeline_components, _make_detector

# ── Config ──
CONFIG = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/config/compare_features.json"
ALL_INFO = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/data/all_info_dedup.tsv"
V4_DATA = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/data/v4_newsequencing.tsv"
OUT_DIR = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/results/v4"

# Routes to evaluate (representative set)
ROUTES = [
    "advanced_xgb",        # current best
    "xgboost_weighted",    # weighted strategy
    "weighted_tmb",        # with TMB features
    "unstable_xgb",        # unstable locus detection
    "sensitive_xgb",       # high sensitivity
    "mahal_vs_cosine",     # mahalanobis detector
    "cosine_advanced",     # cosine detector
    "full_upgrade_v2",     # full upgrade
    "no_locus_filter",     # no locus filtering (baseline)
]

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
})


def setup_route_logger(route_name, out_dir):
    """Create per-route logger."""
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{route_name}.log")

    logger = logging.getLogger(f"route.{route_name}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)-7s %(message)s'))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(f'[%(name)s] %(message)s'))
    logger.addHandler(ch)

    return logger


def load_v3_data():
    df = pd.read_csv(ALL_INFO, sep='\t')
    bl = df[df['origin'] == 'BL'].copy()
    bl['sample_id'] = bl['site_feature'].apply(
        lambda x: os.path.basename(x).split('_cancer')[0] if isinstance(x, str) else None)
    bl = bl.set_index('sample_id')
    bl = bl[bl['MSI_real'].notna()]
    return bl


def load_v4_data():
    df = pd.read_csv(V4_DATA, sep='\t')
    df['sample_id'] = df['样本编号'].astype(str).str.strip()
    df = df.set_index('sample_id')
    return df


def extract_features(fe, sf, meta, site_col='site_feature', logger=None):
    if logger:
        logger.info(f"Extracting features from {len(meta)} samples (col={site_col})")
    features, _ = fe.extract_batch(meta[site_col].values, meta.index.values)
    join_cols = [c for c in ['MSI_real', 'cancertype', 'tumor_content', 'TMB_status'] if c in meta.columns]
    features = features.join(meta[join_cols], how='inner')
    if 'MSI_real' in features.columns:
        features.rename(columns={'MSI_real': 'MSI_status'}, inplace=True)
    features = sf.filter(features)
    if logger:
        logger.info(f"After filter: {len(features)} samples")
    return features


def calc_metrics(y_true, scores, threshold):
    predictions = np.where(scores >= threshold, 'MSI-H', 'MSS')
    tp = int(((predictions == 'MSI-H') & (y_true == 'MSI-H')).sum())
    fp = int(((predictions == 'MSI-H') & (y_true == 'MSS')).sum())
    fn = int(((predictions == 'MSS') & (y_true == 'MSI-H')).sum())
    tn = int(((predictions == 'MSS') & (y_true == 'MSS')).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = (tp + tn) / len(y_true)
    fpr_arr, tpr_arr, _ = roc_curve(y_true, scores, pos_label='MSI-H')
    auc_val = auc(fpr_arr, tpr_arr)
    return {
        'auc': auc_val, 'sens': sens, 'spec': spec, 'acc': acc,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'threshold': threshold, 'fpr': fpr_arr, 'tpr': tpr_arr,
        'n_msih': tp + fn, 'n_mss': tn + fp,
    }


def run_one_route(route_name, cfg_full, v3_meta, v4_meta, out_dir, global_logger):
    """Run one route: v3→v4 and v4 self-eval."""
    logger = setup_route_logger(route_name, out_dir)

    route_configs = load_config(CONFIG, route_names=[route_name])
    cfg = route_configs[route_name]
    strat_cfg = cfg['strategies'][0]
    det_cfg = cfg['detectors'][0]
    run_cfg = cfg.get('pipeline', {})

    # Log configuration
    logger.info(f"{'='*60}")
    logger.info(f"Route: {route_name}")
    logger.info(f"Strategy: {strat_cfg['name']}")
    logger.info(f"Detector: {det_cfg['name']}")
    logger.info(f"Locus selector: {strat_cfg.get('locus_selector', {}).get('type', 'none')}")
    logger.info(f"Feature selector: {strat_cfg.get('feature_selector', {}).get('type', 'none')}")
    logger.info(f"Pipeline config: {json.dumps(run_cfg, indent=2)}")
    logger.info(f"{'='*60}")

    # Save config
    config_dir = os.path.join(out_dir, "config")
    os.makedirs(config_dir, exist_ok=True)
    with open(os.path.join(config_dir, f"{route_name}.json"), 'w') as f:
        json.dump(cfg, f, indent=2)

    results = {}

    # ── Experiment 1: v3 train → v4 test ──
    logger.info("\n--- Experiment 1: v3 train → v4 test ---")
    try:
        fe, locus_sel, feat_sel, sf, train_filter, req_feats = _build_pipeline_components(strat_cfg)
        det = _make_detector(det_cfg)
        pipeline = MSIDetectionPipeline(
            feature_extractor=fe, locus_selector=locus_sel,
            feature_selector=feat_sel, sample_filter=sf,
            detector=det, train_filter=train_filter,
            required_features=req_feats,
        )

        logger.info("Training on v3 data...")
        res_v3 = pipeline.run(
            v3_meta,
            n_sigma=run_cfg.get('n_sigma', 3.0),
            site_file_col='site_feature',
            test_size=run_cfg.get('test_size', 0.2),
            cache_dir=cfg.get('cache_dir'),
            msi_col='MSI_real',
            threshold_method=run_cfg.get('threshold_method', 'cv'),
            cv_folds=run_cfg.get('cv_folds', 5),
        )

        threshold = res_v3['threshold']
        selected_cols = res_v3['selected_cols']
        logger.info(f"Training done. Threshold={threshold:.4f}, features={len(selected_cols)}")
        logger.info(f"Features: {selected_cols}")

        # Save features
        feat_dir = os.path.join(out_dir, "features")
        os.makedirs(feat_dir, exist_ok=True)
        with open(os.path.join(feat_dir, f"{route_name}.txt"), 'w') as f:
            f.write(f"Route: {route_name}\n")
            f.write(f"Strategy: {strat_cfg['name']}\n")
            f.write(f"Detector: {det_cfg['name']}\n")
            f.write(f"Locus selector: {strat_cfg.get('locus_selector', {}).get('type', 'none')}\n")
            f.write(f"Feature selector: {strat_cfg.get('feature_selector', {}).get('type', 'none')}\n")
            f.write(f"Threshold: {threshold:.4f}\n")
            f.write(f"Features ({len(selected_cols)}):\n")
            for feat in selected_cols:
                f.write(f"  {feat}\n")

        # v3 test metrics
        m_v3test = calc_metrics(
            res_v3['test']['df']['MSI_status'].values,
            res_v3['test']['scores'], threshold)
        logger.info(f"v3 test: AUC={m_v3test['auc']:.4f} Sens={m_v3test['sens']:.4f} Spec={m_v3test['spec']:.4f}")

        # v4 evaluation
        logger.info("Extracting v4 features...")
        v4_features = extract_features(fe, sf, v4_meta, site_col='site_path', logger=logger)
        available = [c for c in selected_cols if c in v4_features.columns]
        missing = [c for c in selected_cols if c not in v4_features.columns]
        if missing:
            logger.warning(f"Missing {len(missing)} features: {missing}")

        X_v4 = np.nan_to_num(v4_features[available].values, nan=0.0)
        if hasattr(det, 'set_feature_names'):
            det.set_feature_names(available)
        v4_scores = det.score(X_v4)
        y_true_v4 = v4_features['MSI_status'].values

        # v3 threshold
        m_v3to4 = calc_metrics(y_true_v4, v4_scores, threshold)
        logger.info(f"v3→v4 (v3 thr): AUC={m_v3to4['auc']:.4f} Sens={m_v3to4['sens']:.4f} Spec={m_v3to4['spec']:.4f}")

        # Youden threshold
        fpr_v4, tpr_v4, thr_v4 = roc_curve(y_true_v4, v4_scores, pos_label='MSI-H')
        j_scores = tpr_v4 - fpr_v4
        youden_thr = thr_v4[int(np.argmax(j_scores))]
        m_v3to4_youden = calc_metrics(y_true_v4, v4_scores, youden_thr)
        logger.info(f"v3→v4 (Youden): AUC={m_v3to4_youden['auc']:.4f} Sens={m_v3to4_youden['sens']:.4f} Spec={m_v3to4_youden['spec']:.4f} Thr={youden_thr:.4f}")

        # Save predictions
        pred_dir = os.path.join(out_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        pred_df = v4_features[['MSI_status']].copy()
        pred_df['score'] = v4_scores
        pred_df['predicted_v3thr'] = np.where(v4_scores >= threshold, 'MSI-H', 'MSS')
        pred_df['predicted_youden'] = np.where(v4_scores >= youden_thr, 'MSI-H', 'MSS')
        pred_df['correct_v3thr'] = (pred_df['MSI_status'] == pred_df['predicted_v3thr'])
        pred_df['correct_youden'] = (pred_df['MSI_status'] == pred_df['predicted_youden'])
        pred_df.to_csv(os.path.join(pred_dir, f"{route_name}_v3to4.tsv"), sep='\t')

        results['v3to4'] = {
            'v3test': m_v3test, 'v3to4_v3thr': m_v3to4, 'v3to4_youden': m_v3to4_youden,
            'features': selected_cols, 'threshold': threshold, 'youden_threshold': youden_thr,
            'n_features_used': len(available), 'n_features_missing': len(missing),
        }

    except Exception as e:
        logger.error(f"Experiment 1 failed: {e}", exc_info=True)
        results['v3to4'] = {'error': str(e)}

    # ── Experiment 2: v4 self-eval ──
    logger.info("\n--- Experiment 2: v4 self-eval (80/20) ---")
    try:
        fe2, locus_sel2, feat_sel2, sf2, train_filter2, req_feats2 = _build_pipeline_components(strat_cfg)
        det2 = _make_detector(det_cfg)
        pipeline2 = MSIDetectionPipeline(
            feature_extractor=fe2, locus_selector=locus_sel2,
            feature_selector=feat_sel2, sample_filter=sf2,
            detector=det2, train_filter=train_filter2,
            required_features=req_feats2,
        )

        v4_meta2 = v4_meta.copy()
        v4_meta2['site_feature'] = v4_meta2['site_path']
        if 'cancertype' not in v4_meta2.columns:
            v4_meta2['cancertype'] = 'unknown'
        if 'MSI_CNC' not in v4_meta2.columns:
            v4_meta2['MSI_CNC'] = v4_meta2['MSI_real']

        logger.info("Training on v4 data...")
        res_v4 = pipeline2.run(
            v4_meta2,
            n_sigma=run_cfg.get('n_sigma', 3.0),
            site_file_col='site_feature',
            test_size=0.2,
            cache_dir=None,
            msi_col='MSI_real',
            threshold_method='cv',
            cv_folds=5,
        )

        threshold_v4 = res_v4['threshold']
        selected_cols_v4 = res_v4['selected_cols']
        logger.info(f"Training done. Threshold={threshold_v4:.4f}, features={len(selected_cols_v4)}")
        logger.info(f"Features: {selected_cols_v4}")

        # Save features
        with open(os.path.join(feat_dir, f"{route_name}_v4self.txt"), 'w') as f:
            f.write(f"Route: {route_name} (v4 self-eval)\n")
            f.write(f"Threshold: {threshold_v4:.4f}\n")
            f.write(f"Features ({len(selected_cols_v4)}):\n")
            for feat in selected_cols_v4:
                f.write(f"  {feat}\n")

        # Test metrics
        test_df = res_v4['test']['df']
        test_scores = res_v4['test']['scores']
        m_v4test = calc_metrics(test_df['MSI_status'].values, test_scores, threshold_v4)
        logger.info(f"v4 test: AUC={m_v4test['auc']:.4f} Sens={m_v4test['sens']:.4f} Spec={m_v4test['spec']:.4f}")

        # Save predictions
        pred_df2 = test_df[['MSI_status']].copy()
        pred_df2['score'] = test_scores
        pred_df2['predicted'] = np.where(test_scores >= threshold_v4, 'MSI-H', 'MSS')
        pred_df2['correct'] = (pred_df2['MSI_status'] == pred_df2['predicted'])
        pred_df2.to_csv(os.path.join(pred_dir, f"{route_name}_v4self.tsv"), sep='\t')

        results['v4self'] = {
            'v4test': m_v4test, 'features': selected_cols_v4,
            'threshold': threshold_v4,
        }

    except Exception as e:
        logger.error(f"Experiment 2 failed: {e}", exc_info=True)
        results['v4self'] = {'error': str(e)}

    return results


def plot_combined_roc(all_results, out_path):
    """Plot ROC curves for all routes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for exp_idx, exp_key in enumerate(['v3to4', 'v4self']):
        ax = axes[exp_idx]
        ci = 0
        for route_name, res in all_results.items():
            if exp_key not in res or 'error' in res[exp_key]:
                continue
            r = res[exp_key]
            if exp_key == 'v3to4':
                m = r['v3to4_youden']
                label = f"{route_name} (AUC={m['auc']:.3f})"
            else:
                m = r['v4test']
                label = f"{route_name} (AUC={m['auc']:.3f})"
            ax.plot(m['fpr'], m['tpr'], color=colors[ci % 10], lw=1.5, label=label)
            ci += 1

        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('FPR (1 - Specificity)')
        ax.set_ylabel('TPR (Sensitivity)')
        title = 'v3→v4 (Youden threshold)' if exp_key == 'v3to4' else 'v4 Self-Eval (Test Set)'
        ax.set_title(title)
        ax.legend(loc='lower right', fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    global_logger = logging.getLogger("main")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    global_logger.info(f"Loading data...")
    v3_meta = load_v3_data()
    v4_meta = load_v4_data()
    global_logger.info(f"v3: {len(v3_meta)} samples, v4: {len(v4_meta)} samples")
    global_logger.info(f"Routes to evaluate: {len(ROUTES)}")

    all_results = {}
    summary_rows = []

    for i, route_name in enumerate(ROUTES):
        global_logger.info(f"\n{'='*60}")
        global_logger.info(f"[{i+1}/{len(ROUTES)}] Running route: {route_name}")
        global_logger.info(f"{'='*60}")

        res = run_one_route(route_name, None, v3_meta, v4_meta, OUT_DIR, global_logger)
        all_results[route_name] = res

        # Collect summary
        if 'v3to4' in res and 'error' not in res['v3to4']:
            r = res['v3to4']
            summary_rows.append({
                'route': route_name, 'experiment': 'v3→v4 (v3 thr)',
                'auc': r['v3to4_v3thr']['auc'], 'sens': r['v3to4_v3thr']['sens'],
                'spec': r['v3to4_v3thr']['spec'], 'acc': r['v3to4_v3thr']['acc'],
                'threshold': r['threshold'], 'n_features': r['n_features_used'],
            })
            summary_rows.append({
                'route': route_name, 'experiment': 'v3→v4 (Youden)',
                'auc': r['v3to4_youden']['auc'], 'sens': r['v3to4_youden']['sens'],
                'spec': r['v3to4_youden']['spec'], 'acc': r['v3to4_youden']['acc'],
                'threshold': r['youden_threshold'], 'n_features': r['n_features_used'],
            })
        if 'v4self' in res and 'error' not in res['v4self']:
            r = res['v4self']
            summary_rows.append({
                'route': route_name, 'experiment': 'v4 self-eval',
                'auc': r['v4test']['auc'], 'sens': r['v4test']['sens'],
                'spec': r['v4test']['spec'], 'acc': r['v4test']['acc'],
                'threshold': r['threshold'], 'n_features': len(r['features']),
            })

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUT_DIR, "experiment_summary.tsv")
    summary_df.to_csv(summary_path, sep='\t', index=False)

    # Plot ROC
    plot_combined_roc(all_results, os.path.join(OUT_DIR, "roc_all_routes.png"))

    # Print summary table
    print(f"\n\n{'#'*80}")
    print(f"  SUMMARY: {len(ROUTES)} routes evaluated")
    print(f"{'#'*80}")
    for exp in ['v3→v4 (v3 thr)', 'v3→v4 (Youden)', 'v4 self-eval']:
        sub = summary_df[summary_df['experiment'] == exp]
        if sub.empty:
            continue
        print(f"\n  {exp}:")
        print(f"  {'Route':<25} {'AUC':>8} {'Sens':>8} {'Spec':>8} {'Acc':>8} {'Thr':>8} {'Feat':>6}")
        print(f"  {'-'*72}")
        for _, row in sub.sort_values('auc', ascending=False).iterrows():
            print(f"  {row['route']:<25} {row['auc']:>8.4f} {row['sens']:>8.4f} {row['spec']:>8.4f} {row['acc']:>8.4f} {row['threshold']:>8.4f} {row['n_features']:>6.0f}")

    print(f"\n  Output: {OUT_DIR}")
    print(f"    experiment_summary.tsv")
    print(f"    roc_all_routes.png")
    print(f"    predictions/<route>_*.tsv")
    print(f"    features/<route>*.txt")
    print(f"    config/<route>.json")
    print(f"    logs/<route>.log")
    print()


if __name__ == '__main__':
    main()
