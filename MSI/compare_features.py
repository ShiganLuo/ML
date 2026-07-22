# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Compare classification ability of different aggregation strategies and detectors.

Runs each (strategy, detector) combination through the MSI pipeline and produces:
  - feature_comparison.tsv        : per-strategy per-detector metrics
  - feature_comparison_scores.png : scatter plot of Sensitivity vs Specificity
  - feature_comparison_roc.png    : ROC curves with legend

All pipeline parameters are controlled via JSON config. See config/compare_features.json.
"""

import os
import sys
import logging
import argparse
import json
import warnings
warnings.filterwarnings("ignore")
from typing import List, Dict, Any, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.text import Text
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Project imports ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from MSI import (
    AGG_STRATEGIES, AggregationStrategy,
    BaselineAggregation, TailAggregation, DistAggregation,
    BioPriorAggregation, PercentileAggregation, CountAggregation,
    ShiftRichAggregation, NoiseFilteredAggregation, MonoFocusAggregation,
    AllAggregation, InteractionAggregation, DistributionAggregation,
    MultiThresholdAggregation, WeightedAggregation, OptimizedAggregation,
    LocusLevelAggregation, AdvancedAggregation, UnstableLocusAggregation,
    SensitiveAggregation, LocusScoreAggregation,
    FeatureExtractor, AUCBasedLocusSelector,
    TwoStageSelector, SingleVariableAUCSelector,
    CombinedFilter, QualityFilter, DepthFilter,
    BinaryClassifierDetector, MahalanobisDetector,
    MSIPercentageDetector, OneClassSVMDetector,
    IsolationForestDetector, OCLRDetector,
    UnstableProportionDetector,
    CosineDetector,
    MSIDetectionPipeline, compute_roc, evaluate,
    Detector, NullLocusSelector,
)

_LOG_FMT = "%(asctime)s %(levelname)-7s %(message)s"
_LOG_DATE = "%Y-%m-%d %H:%M:%S"

# Console handler (always active)
logging.basicConfig(level=logging.INFO, format=_LOG_FMT, datefmt=_LOG_DATE)
logger = logging.getLogger(__name__)


def _add_file_handler(output_dir: str) -> logging.FileHandler:
    """Attach a file handler to root logger so all modules' logs are captured."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'compare_features.log')
    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_LOG_DATE))
    # Attach to root logger so pipeline, features, selectors, etc. all log to file
    root = logging.getLogger()
    root.addHandler(fh)
    logger.debug(f"Log file: {log_path}")
    return fh


# ── Registries ──

# Aggregation strategy registry: name -> constructor
STRATEGY_REGISTRY: Dict[str, Callable[..., AggregationStrategy]] = {
    'baseline':        lambda **p: BaselineAggregation(**p),
    'tail':            lambda **p: TailAggregation(**p),
    'dist':            lambda **p: DistAggregation(**p),
    'bio_prior':       lambda **p: BioPriorAggregation(**p),
    'percentile':      lambda **p: PercentileAggregation(**p),
    'count':           lambda **p: CountAggregation(**p),
    'shift_rich':      lambda **p: ShiftRichAggregation(**p),
    'noise_filtered':  lambda **p: NoiseFilteredAggregation(**p),
    'mono_focus':      lambda **p: MonoFocusAggregation(**p),
    'all':             lambda **p: AllAggregation(**p),
    'interaction':     lambda **p: InteractionAggregation(**p),
    'distribution':    lambda **p: DistributionAggregation(**p),
    'multi_threshold': lambda **p: MultiThresholdAggregation(**p),
    'weighted':        lambda **p: WeightedAggregation(**p),
    'optimized':       lambda **p: OptimizedAggregation(**p),
    'locus_level':     lambda **p: LocusLevelAggregation(**p),
    'advanced':        lambda **p: AdvancedAggregation(**p),
    'unstable_locus':  lambda **p: UnstableLocusAggregation(**p),
    'sensitive':       lambda **p: SensitiveAggregation(**p),
    'locus_score':     lambda **p: LocusScoreAggregation(**p),
}

# Detector registry: name -> constructor
DETECTOR_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Detector]] = {
    'xgboost':     lambda p: BinaryClassifierDetector(method='xgboost', **p),
    'logistic':    lambda p: BinaryClassifierDetector(method='logistic', **p),
    'mahalanobis': lambda p: MahalanobisDetector(**p),
    'msi_pct':     lambda p: MSIPercentageDetector(**p),
    'ocsvm':       lambda p: OneClassSVMDetector(**p),
    'iforest':     lambda p: IsolationForestDetector(**p),
    'oclr':        lambda p: OCLRDetector(**p),
    'unstable_prop': lambda p: UnstableProportionDetector(**p),
    'cosine':      lambda p: CosineDetector(**p),
}

DETECTOR_GROUPS: Dict[str, List[str]] = {
    'all':      list(DETECTOR_REGISTRY.keys()),
    'oneclass': ['mahalanobis', 'ocsvm', 'iforest', 'oclr'],
    'binary':   ['xgboost', 'logistic'],
}


# ── Defaults (complete pipeline config, every node exposed) ──

DEFAULTS: Dict[str, Any] = {
    "all_info": "output/MSI/data/all_info.tsv",
    "cache_dir": "output/MSI/data/feature",
    "output_dir": "output/MSI/results/compare/compare_features",
    "max_workers": 4,

    "pipeline": {
        "n_sigma": 3.0,
        "site_file_col": "site_feature",
        "test_size": 0.2,
        "msi_col": "MSI_real",
        "threshold_method": "youden",
    },

    "strategies": [
        {
            "name": "baseline", "params": {},
            "feature_extractor": {"min_depth": 10},
            "locus_selector":    {"type": "auc", "params": {"auc_threshold": 0.8, "min_depth": 30}},
            "feature_selector":  {"type": "two_stage", "params": {"auc_threshold": 0.8, "top_k": 50}},
            "quality_filter":    {"min_loci": 50}
        }
    ],

    "detectors": [
        {"name": "xgboost",     "params": {}},
        {"name": "mahalanobis", "params": {}},
    ],
}


# ── Config loading ──

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge *override* into *base*, returning a new dict.

    Nested dicts are merged recursively; lists and scalars are replaced.
    """
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_config(
    config_path: str,
    route_name: Optional[str] = None,
    route_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load a JSON config and return {route_name: resolved_cfg, ...}.

    Parameters
    ----------
    config_path : str
        Path to JSON config file.
    route_name : str, optional
        If given, only return this single route.  Otherwise return all routes.
    route_names : list of str, optional
        If given, return these specific routes. Overrides route_name.

    Returns
    -------
    dict
        Mapping of route name -> fully resolved config dict (defaults merged in).
    """
    with open(config_path) as f:
        raw = json.load(f)

    defaults = _deep_merge(DEFAULTS, raw.get("defaults", {}))
    routes = raw.get("routes", {})
    base_output_dir = defaults.get("output_dir", "")

    def _resolve(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Merge defaults into route and resolve relative output_dir."""
        merged = _deep_merge(defaults, cfg)
        out = merged.get("output_dir", "")
        if out and not os.path.isabs(out) and base_output_dir:
            merged["output_dir"] = os.path.join(base_output_dir, out)
        return merged

    if not routes:
        return {"default": defaults}

    # Determine which routes to load
    names = route_names or ([route_name] if route_name else None)

    if names:
        result = {}
        for name in names:
            if name not in routes:
                available = ", ".join(routes.keys())
                raise ValueError(f"Route '{name}' not found. Available: {available}")
            result[name] = _resolve(routes[name])
        return result

    return {name: _resolve(route) for name, route in routes.items()}


# ── Object builders ──

def _make_strategy(strat_cfg: Dict[str, Any]) -> AggregationStrategy:
    """Instantiate an AggregationStrategy from its config dict.

    Parameters
    ----------
    strat_cfg : dict
        Must have 'name' key.  'params' is passed to the constructor.
    """
    name: str = strat_cfg['name']
    params: Dict[str, Any] = strat_cfg.get('params', {})
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name](**params)


def _make_detector(det_cfg: Dict[str, Any]) -> Detector:
    """Instantiate a Detector from its config dict.

    Parameters
    ----------
    det_cfg : dict
        Must have 'name' key.  'params' is passed to the constructor.
    """
    name: str = det_cfg['name']
    params: Dict[str, Any] = det_cfg.get('params', {})
    if name not in DETECTOR_REGISTRY:
        raise ValueError(f"Unknown detector: {name}. Available: {list(DETECTOR_REGISTRY.keys())}")
    return DETECTOR_REGISTRY[name](params)


def _resolve_items(
    items: Any,
    registry: Dict[str, Any],
    groups: Optional[Dict[str, List[str]]] = None,
    default: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Normalize a config list into [{name, params}, ...] dicts.

    Accepts:
      - list of dicts: [{"name": "...", "params": {...}}, ...]
      - list of strings: ["name1", "name2", ...]  (params={})
      - null/None -> use default
    """
    if items is None:
        return default or []

    result: List[Dict[str, Any]] = []
    seen: set = set()

    raw_items = items if isinstance(items, list) else [items]
    for item in raw_items:
        if isinstance(item, dict):
            name = item['name']
            params_key = json.dumps(item.get('params', {}), sort_keys=True)
            composite = f"{name}:{params_key}"
            if composite not in seen:
                result.append(item)
                seen.add(composite)
        elif isinstance(item, str):
            key = item.lower().strip()
            if groups and key in groups:
                for g in groups[key]:
                    if g not in seen:
                        result.append({"name": g, "params": {}})
                        seen.add(g)
            elif key in registry:
                if key not in seen:
                    result.append({"name": key, "params": {}})
                    seen.add(key)
            else:
                logger.warning(f"Unknown item '{key}', skipping. "
                               f"Available: {list(registry.keys())}")

    return result or (default or [])


def _resolve_strategies(strategies: Any) -> List[Dict[str, Any]]:
    """Resolve strategies config into list of {name, params} dicts."""
    return _resolve_items(
        strategies, STRATEGY_REGISTRY, default=list(DEFAULTS['strategies'])
    )


def _resolve_detectors(detectors: Any) -> List[Dict[str, Any]]:
    """Resolve detectors config into list of {name, params} dicts."""
    return _resolve_items(
        detectors, DETECTOR_REGISTRY, groups=DETECTOR_GROUPS,
        default=list(DEFAULTS['detectors'])
    )


# ── Label repulsion (force-directed) ──

def _bbox_overlap_area(bi: Any, bj: Any) -> float:
    """Compute pixel overlap area between two bboxes."""
    dx = max(0.0, min(bi.xmax, bj.xmax) - max(bi.xmin, bj.xmin))
    dy = max(0.0, min(bi.ymax, bj.ymax) - max(bi.ymin, bj.ymin))
    return dx * dy


def repel_labels(
    ax: Axes,
    texts: List[Text],
    max_iter: int = 200,
    pad_px: float = 6,
) -> int:
    """Push overlapping labels apart; handles co-located data points.
    Phase 1: force-directed nudge (fast, handles most cases).
    Phase 2: for stuck labels, try 8 candidate positions and pick the one
    with minimum total overlap.  This handles co-located data points where
    forces cancel out.
    """
    if len(texts) < 2:
        return 0

    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    origins = [text.get_position() for text in texts]
    data_pts = [ax.transData.transform(p) for p in origins]
    inv = ax.transData.inverted()

    def _get_bboxes() -> List[Any]:
        fig.canvas.draw()
        return [t.get_window_extent(renderer=renderer) for t in texts]

    def _total_overlap(bboxes: List[Any]) -> float:
        total = 0.0
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                total += _bbox_overlap_area(bboxes[i], bboxes[j])
            # Penalise label covering another data point
            bi = bboxes[i]
            for j, pt in enumerate(data_pts):
                if j != i and bi.width > 0:
                    if bi.xmin <= pt[0] <= bi.xmax and bi.ymin <= pt[1] <= bi.ymax:
                        total += bi.width * bi.height * 0.3
        return total

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # ── Phase 1: force-directed ──
    for iteration in range(max_iter):
        bboxes = _get_bboxes()
        forces: Dict[int, Tuple[float, float]] = {k: (0.0, 0.0) for k in range(len(texts))}
        any_overlap = False

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                bi, bj = bboxes[i], bboxes[j]
                if bi.width == 0 or bj.width == 0:
                    continue
                bi_p = bi.expanded(1 + pad_px / bi.width, 1 + pad_px / bi.height)
                bj_p = bj.expanded(1 + pad_px / bj.width, 1 + pad_px / bj.height)
                if not bi_p.overlaps(bj_p):
                    continue
                any_overlap = True
                area = _bbox_overlap_area(bi_p, bj_p)
                cix, ciy = (bi.xmin + bi.xmax) / 2, (bi.ymin + bi.ymax) / 2
                cjx, cjy = (bj.xmin + bj.xmax) / 2, (bj.ymin + bj.ymax) / 2
                vx, vy = cix - cjx, ciy - cjy
                dist = (vx ** 2 + vy ** 2) ** 0.5
                if dist < 1e-6:
                    vx, vy, dist = 1.0, 0.0, 1.0
                mag = max(3.0, area ** 0.5) * 1.5
                ux, uy = vx / dist, vy / dist
                forces[i] = (forces[i][0] + ux * mag, forces[i][1] + uy * mag)
                forces[j] = (forces[j][0] - ux * mag, forces[j][1] - uy * mag)

        if not any_overlap:
            return iteration

        # Restoring spring + bounds clamp
        for i, text in enumerate(texts):
            cur = text.get_position()
            dx_data = origins[i][0] - cur[0]
            dy_data = origins[i][1] - cur[1]
            fx = forces[i][0] + dx_data * 0.05
            fy = forces[i][1] + dy_data * 0.05
            mag = (fx ** 2 + fy ** 2) ** 0.5
            if mag < 1e-6:
                continue
            max_disp = 20.0
            if mag > max_disp:
                fx, fy = fx / mag * max_disp, fy / mag * max_disp
            origin_disp = inv.transform([0, 0])
            dxd, dyd = inv.transform([fx, fy]) - origin_disp
            new_x = max(xmin, min(xmax, cur[0] + dxd))
            new_y = max(ymin, min(ymax, cur[1] + dyd))
            text.set_position((new_x, new_y))

    # ── Phase 2: greedy candidate placement for stuck overlaps ──
    import math
    angles = [k * math.pi / 4 for k in range(8)]  # 0, 45, 90, ..., 315
    radii = [20, 30, 45]

    for _ in range(5):
        bboxes = _get_bboxes()
        # Sort by overlap severity (worst first)
        overlap_scores = []
        for i in range(len(texts)):
            score = 0.0
            for j in range(len(texts)):
                if i != j:
                    score += _bbox_overlap_area(bboxes[i], bboxes[j])
            overlap_scores.append((score, i))
        overlap_scores.sort(reverse=True)

        improved = False
        for score, i in overlap_scores:
            if score < 1.0:
                continue
            text = texts[i]
            cur = text.get_position()
            best_pos = cur
            best_overlap = _total_overlap(bboxes)

            # Try candidate positions: 8 angles × 3 radii + snapped positions
            ox, oy = data_pts[i]
            candidates = []
            for r in radii:
                for a in angles:
                    nx = ox + r * math.cos(a)
                    ny = oy + r * math.sin(a)
                    candidates.append((nx, ny))
            # Snap to quadrant corners (label fully clear of data cluster)
            for dx_sign, dy_sign in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                candidates.append((ox + 35 * dx_sign, oy + 15 * dy_sign))

            for nx, ny in candidates:
                text.set_position((nx, ny))
                fig.canvas.draw()
                new_bbox = text.get_window_extent(renderer=renderer)
                new_bboxes = list(bboxes)
                new_bboxes[i] = new_bbox
                new_overlap = _total_overlap(new_bboxes)
                if new_overlap < best_overlap - 0.1:
                    best_overlap = new_overlap
                    best_pos = (nx, ny)
                    improved = True

            text.set_position(best_pos)
            fig.canvas.draw()
            bboxes = _get_bboxes()

        if not improved:
            break

    return max_iter


# ── Pipeline runner ──

from MSI.feature_selectors import (
    XgbImportanceSelector, TwoStageXgbSelector,
    RelaxedAUCSelector, MultiMetricLocusSelector,
    LassoSelector, VarianceSelector, TwoStageVarianceSelector,
)
from MSI.filters import AnomalyFilter, MultivariateOutlierFilter


def _build_pipeline_components(
    strat_cfg: Dict[str, Any],
) -> Tuple[FeatureExtractor, Any, Any, CombinedFilter, Any, list]:
    """Build pipeline components from a strategy config dict.

    Parameters
    ----------
    strat_cfg : dict
        Strategy config with keys: name, params, feature_extractor,
        locus_selector, feature_selector, quality_filter.

    Returns
    -------
    tuple
        (FeatureExtractor, locus_selector, feature_selector, sample_filter, train_filter, required_features)
    """
    fe_cfg = strat_cfg.get('feature_extractor', {})
    agg_strategy = _make_strategy(strat_cfg)
    fe = FeatureExtractor(
        min_depth=fe_cfg.get('min_depth', 10),
        agg_strategy=agg_strategy,
    )

    # Locus selector: auto-skip for strategies that need all loci
    _ALL_LOCI_STRATEGIES = {'unstable_locus', 'advanced'}
    strat_name = strat_cfg.get('name', '')
    ls_cfg = strat_cfg.get('locus_selector', {"type": "auc", "params": {}})
    ls_type = ls_cfg.get('type', 'auc')
    ls_params = ls_cfg.get('params', {})
    if strat_name in _ALL_LOCI_STRATEGIES and ls_type != 'none':
        logger.info(f"Strategy '{strat_name}' needs all loci, overriding locus_selector to none")
        locus_sel = NullLocusSelector()
    elif ls_type == 'none':
        locus_sel = NullLocusSelector()
    elif ls_type == 'auc':
        locus_sel = AUCBasedLocusSelector(**ls_params)
    elif ls_type == 'relaxed_auc':
        locus_sel = RelaxedAUCSelector(**ls_params)
    elif ls_type == 'multi_metric':
        locus_sel = MultiMetricLocusSelector(**ls_params)
    else:
        raise ValueError(f"Unknown locus_selector type: {ls_type}")

    # Feature selector
    fs_cfg = strat_cfg.get('feature_selector', {"type": "two_stage", "params": {}})
    fs_type = fs_cfg.get('type', 'two_stage')
    fs_params = fs_cfg.get('params', {})
    if fs_type == 'two_stage':
        feat_sel = TwoStageSelector(**fs_params)
    elif fs_type == 'single_auc':
        feat_sel = SingleVariableAUCSelector(**fs_params)
    elif fs_type == 'xgb_importance':
        feat_sel = XgbImportanceSelector(**fs_params)
    elif fs_type == 'two_stage_xgb':
        feat_sel = TwoStageXgbSelector(**fs_params)
    elif fs_type == 'lasso':
        feat_sel = LassoSelector(**fs_params)
    elif fs_type == 'variance':
        feat_sel = VarianceSelector(**fs_params)
    elif fs_type == 'two_stage_variance':
        feat_sel = TwoStageVarianceSelector(**fs_params)
    else:
        raise ValueError(f"Unknown feature_selector type: {fs_type}")

    # Sample filters: quality filter applied to all, anomaly filter only to MSS training
    qf_cfg = strat_cfg.get('quality_filter', {"min_loci": 50})
    quality_filters = [QualityFilter(min_loci=qf_cfg.get('min_loci', 50))]
    sf = CombinedFilter(quality_filters)

    # Train-only filters (anomaly detection) — applied after split, only to MSS
    train_filters = []
    af_cfg = strat_cfg.get('anomaly_filter')
    if af_cfg:
        train_filters.append(AnomalyFilter(
            contamination=af_cfg.get('contamination', 0.05),
            n_estimators=af_cfg.get('n_estimators', 200),
        ))

    mv_cfg = strat_cfg.get('multivariate_outlier_filter')
    if mv_cfg:
        train_filters.append(MultivariateOutlierFilter(
            n_sigma=mv_cfg.get('n_sigma', 4.0),
        ))

    train_filter = CombinedFilter(train_filters) if train_filters else None
    required_features = strat_cfg.get('required_features', [])
    return fe, locus_sel, feat_sel, sf, train_filter, required_features


def _run_one_combination(
    strat_cfg: Dict[str, Any],
    det_cfg: Dict[str, Any],
    meta: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Run pipeline for a single (strategy, detector) combination.

    Parameters
    ----------
    strat_cfg : dict
        Strategy config: {name, params, feature_extractor, locus_selector, ...}.
    det_cfg : dict
        Detector config: {name, params}.
    meta : pd.DataFrame
        Sample metadata.
    cfg : dict
        Full resolved config dict (provides pipeline.run() params and cache_dir).
    """
    fe, locus_sel, feat_sel, sf, train_filter, required_features = _build_pipeline_components(strat_cfg)
    det = _make_detector(det_cfg)

    pipeline = MSIDetectionPipeline(
        feature_extractor=fe,
        locus_selector=locus_sel,
        feature_selector=feat_sel,
        sample_filter=sf,
        detector=det,
        train_filter=train_filter,
        required_features=required_features,
    )

    # Pipeline run parameters
    run_cfg = cfg.get('pipeline', {})
    results = pipeline.run(
        meta,
        n_sigma=run_cfg.get('n_sigma', 3.0),
        site_file_col=run_cfg.get('site_file_col', 'site_feature'),
        test_size=run_cfg.get('test_size', 0.2),
        cache_dir=cfg['cache_dir'],
        msi_col=run_cfg.get('msi_col', 'MSI_real'),
        threshold_method=run_cfg.get('threshold_method', 'cv'),
        cv_folds=run_cfg.get('cv_folds', 5),
    )
    return results


def _collect_result_row(
    detector_name: str,
    strat_name: str,
    res: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a result row from a pipeline result."""
    test = res['test']
    row: Dict[str, Any] = {
        'detector': detector_name,
        'strategy': strat_name,
        'auc': test['auc'],
        'sens': test['eval']['sens'],
        'spec': test['eval']['spec'],
        'acc': test['eval']['acc'],
        'tp': test['eval']['TP'],
        'fp': test['eval']['FP'],
        'fn': test['eval']['FN'],
        'tn': test['eval']['TN'],
        'threshold': res['threshold'],
        'threshold_method': res.get('threshold_method', '?'),
        'n_features': len(res.get('selected_cols', [])),
        'fpr': test.get('fpr'),
        'tpr': test.get('tpr'),
    }
    # CV-specific fields
    cv_folds = res.get('cv_fold_thresholds')
    if cv_folds:
        row['cv_threshold_std'] = float(np.std(cv_folds))
        row['cv_fold_thresholds'] = ','.join(f'{t:.4f}' for t in cv_folds)
    return row


def _save_combination_outputs(
    det_name: str,
    strat_name: str,
    res: Dict[str, Any],
    output_dir: str,
) -> None:
    """Save per-dataset TSVs and feature JSON for one (detector, strategy) combo."""
    combo_dir = os.path.join(output_dir, det_name, strat_name)
    os.makedirs(combo_dir, exist_ok=True)

    # BL test scores
    test = res['test']
    test_df = test['df'].copy()
    test_df['score'] = test['scores']
    test_df['predicted'] = np.where(test['scores'] >= res['threshold'], 'MSI-H', 'MSS')
    test_df.to_csv(os.path.join(combo_dir, 'bl_test_scores.tsv'), sep='\t')

    # Per-cancer evaluation on BL test set
    if 'cancertype' in test_df.columns:
        cancer_rows = []
        threshold = res['threshold']
        for ct in sorted(test_df['cancertype'].dropna().unique()):
            mask = test_df['cancertype'] == ct
            sub = test_df[mask]
            y_true = sub['MSI_status'].values
            scores = sub['score'].values
            pred = sub['predicted'].values
            n = len(sub)
            n_msih = int((y_true == 'MSI-H').sum())
            n_mss = int((y_true == 'MSS').sum())
            if n_msih == 0 and n_mss == 0:
                continue
            tp = int(((pred == 'MSI-H') & (y_true == 'MSI-H')).sum())
            fp = int(((pred == 'MSI-H') & (y_true == 'MSS')).sum())
            fn = int(((pred == 'MSS') & (y_true == 'MSI-H')).sum())
            tn = int(((pred == 'MSS') & (y_true == 'MSS')).sum())
            cancer_rows.append({
                'cancertype': ct,
                'n': n,
                'n_msih': n_msih,
                'n_mss': n_mss,
                'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                'sens': tp / (tp + fn) if (tp + fn) > 0 else None,
                'spec': tn / (tn + fp) if (tn + fp) > 0 else None,
                'acc': (tp + tn) / n if n > 0 else None,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else None,
            })
        if cancer_rows:
            cancer_df = pd.DataFrame(cancer_rows)
            cancer_df.to_csv(os.path.join(combo_dir, 'per_cancer_metrics.tsv'), sep='\t', index=False)

    # Feature JSON
    feature_info = {
        'detector': det_name,
        'strategy': strat_name,
        'selected_cols': res.get('selected_cols', []),
        'n_features': len(res.get('selected_cols', [])),
        'threshold': float(res['threshold']),
        'threshold_method': res.get('threshold_method', '?'),
        'n_sigma': float(res.get('n_sigma', 0)),
    }
    cv_folds = res.get('cv_fold_thresholds')
    if cv_folds:
        feature_info['cv_fold_thresholds'] = [float(t) for t in cv_folds]
    cancer_thresholds = res.get('cancer_thresholds')
    if cancer_thresholds:
        feature_info['cancer_thresholds'] = {k: float(v) for k, v in cancer_thresholds.items()}
    with open(os.path.join(combo_dir, 'features.json'), 'w') as f:
        json.dump(feature_info, f, indent=2, ensure_ascii=False)

    # Save trained model for reuse
    _save_model(res, combo_dir)


def _save_model(res: Dict[str, Any], combo_dir: str) -> None:
    """Save trained model artifacts for later prediction."""
    import pickle

    model_info = {
        'selected_cols': res.get('selected_cols', []),
        'threshold': float(res['threshold']),
        'threshold_method': res.get('threshold_method', '?'),
        'n_sigma': float(res.get('n_sigma', 0)),
    }

    # Save detector
    detector = res.get('detector')
    if detector is not None:
        det_path = os.path.join(combo_dir, 'detector.pkl')
        with open(det_path, 'wb') as f:
            pickle.dump(detector, f, protocol=4)
        model_info['detector_pkl'] = 'detector.pkl'

    # Save cancer thresholds
    cancer_thresholds = res.get('cancer_thresholds')
    if cancer_thresholds:
        model_info['cancer_thresholds'] = {k: float(v) for k, v in cancer_thresholds.items()}

    # Save feature selector if available
    feature_selector = res.get('feature_selector')
    if feature_selector is not None:
        fs_path = os.path.join(combo_dir, 'feature_selector.pkl')
        with open(fs_path, 'wb') as f:
            pickle.dump(feature_selector, f, protocol=4)
        model_info['feature_selector_pkl'] = 'feature_selector.pkl'

    # Save feature extractor config
    feature_extractor = res.get('feature_extractor')
    if feature_extractor is not None:
        fe_path = os.path.join(combo_dir, 'feature_extractor.pkl')
        with open(fe_path, 'wb') as f:
            pickle.dump(feature_extractor, f, protocol=4)
        model_info['feature_extractor_pkl'] = 'feature_extractor.pkl'

    # Save sample filter
    sample_filter = res.get('sample_filter')
    if sample_filter is not None:
        sf_path = os.path.join(combo_dir, 'sample_filter.pkl')
        with open(sf_path, 'wb') as f:
            pickle.dump(sample_filter, f, protocol=4)
        model_info['sample_filter_pkl'] = 'sample_filter.pkl'

    model_path = os.path.join(combo_dir, 'model.json')
    with open(model_path, 'w') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)


def _run_strategies_for_detector(
    det_cfg: Dict[str, Any],
    strat_configs: List[Dict[str, Any]],
    meta: pd.DataFrame,
    cfg: Dict[str, Any],
    max_workers: int,
    output_dir: str,
) -> List[Dict[str, Any]]:
    """Run all strategies for one detector in parallel."""
    results: List[Dict[str, Any]] = []
    detector_name = det_cfg['name']

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_strat = {
            executor.submit(_run_one_combination, s, det_cfg, meta, cfg): s
            for s in strat_configs
        }
        for future in as_completed(future_to_strat):
            strat_cfg = future_to_strat[future]
            _params = strat_cfg.get('params', {})
            strat_name = strat_cfg['name'] + (f"({','.join(f'{v}' for v in _params.values())})" if _params else "")
            try:
                res = future.result()
                row = _collect_result_row(detector_name, strat_name, res)
                results.append(row)
                # Save per-dataset results and feature JSON
                _save_combination_outputs(detector_name, strat_cfg['name'], res, output_dir)
                test = res['test']
                logger.info(
                    f"[{detector_name}] {strat_name}: "
                    f"AUC={test['auc']:.4f} Sens={test['eval']['sens']:.4f} "
                    f"Spec={test['eval']['spec']:.4f}"
                )
            except Exception as e:
                logger.error(f"[{detector_name}] {strat_name} failed: {e}")

    return results


def run_comparison(cfg: Dict[str, Any]) -> None:
    """Run pipeline for each (strategy, detector) combination and produce outputs.

    Parameters
    ----------
    cfg : dict
        Fully resolved config dict (defaults + route merged).
    """
    all_info: str = cfg['all_info']
    output_dir: str = cfg['output_dir']
    cache_dir: str = cfg['cache_dir']
    max_workers: int = cfg.get('max_workers', 4)

    os.makedirs(output_dir, exist_ok=True)

    # ── Setup file logging ──
    fh = _add_file_handler(output_dir)

    strat_configs = _resolve_strategies(cfg.get('strategies'))
    det_configs = _resolve_detectors(cfg.get('detectors'))

    # Load metadata
    meta = pd.read_csv(all_info, sep='\t')
    meta['sample_id'] = meta['site_feature'].apply(
        lambda x: os.path.basename(x).split('_cancer')[0] if isinstance(x, str) else None)
    meta = meta.set_index('sample_id')

    # Merge extra features from all_info_v2 (tumor_content, TMB_status)
    all_info_v2 = cfg.get('all_info_v2')
    if all_info_v2 and os.path.isfile(all_info_v2):
        v2 = pd.read_csv(all_info_v2, sep='\t')
        v2 = v2.set_index('sample_id')
        # tumor_content: numeric
        if 'tumor_content' in v2.columns:
            meta['tumor_content'] = pd.to_numeric(
                v2['tumor_content'], errors='coerce')
            logger.info(f"Merged tumor_content: "
                        f"{meta['tumor_content'].notna().sum()}/{len(meta)} non-null")
        # TMB_status: one-hot encode
        if 'TMB_status' in v2.columns:
            tmb = v2['TMB_status'].reindex(meta.index)
            for cat in ['TMB-H', 'TMB-L', 'TMB-U']:
                col = cat.replace("-", "_")  # TMB_H, TMB_L, TMB_U
                meta[col] = (tmb == cat).astype(float)
                meta.loc[tmb.isna(), col] = np.nan
            logger.info(f"Merged TMB_status one-hot: "
                        f"{tmb.notna().sum()}/{len(meta)} non-null")

    strat_names = [s['name'] for s in strat_configs]
    det_names = [d['name'] for d in det_configs]
    logger.info(f"Loaded {len(meta)} samples, {len(strat_configs)} strategies "
                f"({', '.join(strat_names)}), "
                f"{len(det_configs)} detectors ({', '.join(det_names)}), "
                f"max_workers={max_workers}")

    # ── Run all detectors in parallel ──
    all_results: List[Dict[str, Any]] = []
    n_det = min(len(det_configs), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=n_det) as outer:
        future_to_det = {
            outer.submit(
                _run_strategies_for_detector, det_cfg, strat_configs, meta, cfg, max_workers, output_dir,
            ): det_cfg['name']
            for det_cfg in det_configs
        }
        for future in as_completed(future_to_det):
            det_name = future_to_det[future]
            try:
                det_results = future.result()
                all_results.extend(det_results)
                logger.info(f"[{det_name}] completed: {len(det_results)}/{len(strat_configs)} "
                            f"strategies succeeded")
            except Exception as e:
                logger.error(f"[{det_name}] all strategies failed: {e}")

    if not all_results:
        logger.error("No results produced. Check detector/strategy configuration.")
        return

    # ── Write combined TSV ──
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ('fpr', 'tpr')}
                        for r in all_results])
    tsv_path = os.path.join(output_dir, 'feature_comparison.tsv')
    df.to_csv(tsv_path, sep='\t', index=False)
    logger.info(f"Saved {tsv_path}")

    # ── Collect per-cancer metrics from all combos ──
    cancer_summary_rows = []
    for det_name in det_names:
        for strat_cfg in strat_configs:
            strat_name = strat_cfg['name']
            cancer_path = os.path.join(output_dir, det_name, strat_name, 'per_cancer_metrics.tsv')
            if os.path.isfile(cancer_path):
                cdf = pd.read_csv(cancer_path, sep='\t')
                cdf['detector'] = det_name
                cdf['strategy'] = strat_name
                cancer_summary_rows.append(cdf)
    if cancer_summary_rows:
        cancer_summary = pd.concat(cancer_summary_rows, ignore_index=True)
        cancer_tsv = os.path.join(output_dir, 'per_cancer_summary.tsv')
        cancer_summary.to_csv(cancer_tsv, sep='\t', index=False)
        logger.info(f"Saved {cancer_tsv} ({len(cancer_summary)} rows)")

    # ── Plot 1: Scatter plot ──
    plot_scatter(df, output_dir)

    # ── Plot 2: ROC curves ──
    plot_roc(all_results, output_dir)

    # ── Dump full resolved config ──
    cfg_path = os.path.join(output_dir, 'config.json')
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)
    logger.info(f"Saved {cfg_path}")

    # ── Cleanup file handler ──
    logger.removeHandler(fh)
    fh.close()


# ── Plotting ──

def plot_scatter(df: pd.DataFrame, output_dir: str) -> None:
    """Scatter plot: Sensitivity vs Specificity per strategy, one subplot per detector."""
    detector_names = df['detector'].unique().tolist()
    n_det = len(detector_names)
    fig, axes = plt.subplots(1, max(n_det, 1), figsize=(7 * max(n_det, 1), 7),
                             squeeze=False, sharex=True, sharey=True)
    cmap = plt.cm.tab10

    # Compute global limits across all detectors so axes stay aligned
    global_sens_min = df['sens'].min() - 0.03 if len(df) > 0 else 0
    global_sens_max = df['sens'].max() + 0.05 if len(df) > 0 else 1
    global_spec_min = df['spec'].min() - 0.03 if len(df) > 0 else 0
    global_spec_max = df['spec'].max() + 0.05 if len(df) > 0 else 1

    for col_idx, det_name in enumerate(detector_names):
        ax = axes[0][col_idx]
        sub = df[df['detector'] == det_name].sort_values('strategy')
        sens = sub['sens'].values
        spec = sub['spec'].values
        strategies = sub['strategy'].tolist()
        texts: List[Text] = []

        for i, (s, sp, strat) in enumerate(zip(sens, spec, strategies)):
            color = cmap(i % 10)
            ax.scatter(s, sp, s=80, color=color, zorder=5,
                       edgecolors='white', linewidth=0.5)
            t = ax.text(s + 0.003, sp + 0.003, strat, fontsize=7, color=color,
                        fontweight='bold', ha='left', va='bottom')
            texts.append(t)

        if texts:
            repel_labels(ax, texts)

        ax.set_xlabel('Sensitivity (True Positive Rate)')
        ax.set_ylabel('Specificity (True Negative Rate)')
        ax.set_title(det_name)
        ax.set_xlim(global_sens_min, global_sens_max)
        ax.set_ylim(global_spec_min, global_spec_max)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2)

    fig.suptitle('Feature Strategy: Sensitivity vs Specificity', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'feature_comparison_scores.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved feature_comparison_scores.png")


def plot_roc(all_results: List[Dict[str, Any]], output_dir: str) -> None:
    """ROC curves with legend, one subplot per detector."""
    det_order: List[str] = []
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_results:
        d = r['detector']
        if d not in grouped:
            grouped[d] = []
            det_order.append(d)
        grouped[d].append(r)

    n_det = len(det_order)
    fig, axes = plt.subplots(1, max(n_det, 1), figsize=(7 * max(n_det, 1), 7),
                             squeeze=False)
    cmap = plt.cm.tab10

    for col_idx, det_name in enumerate(det_order):
        ax = axes[0][col_idx]
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')

        results = sorted(grouped[det_name], key=lambda r: r['strategy'])
        for i, r in enumerate(results):
            fpr = r.get('fpr')
            tpr = r.get('tpr')
            if fpr is None or tpr is None:
                continue
            color = cmap(i % 10)
            strat = r['strategy']
            auc_val = r['auc']
            ax.plot(fpr, tpr, color=color, linewidth=1.5,
                    label=f'{strat} (AUC={auc_val:.3f})')

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(det_name)
        ax.legend(loc='lower right', frameon=False, fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('ROC Curves: Feature Strategy Comparison', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'feature_comparison_roc.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved feature_comparison_roc.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compare MSI feature strategies across detectors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Config file mode (recommended):\n"
            "  compare_features.py --config config/compare_features.json\n"
            "  compare_features.py --config config/compare_features.json --route xgboost_weighted\n"
            "  compare_features.py --config config/compare_features.json --route unstable_basic unstable_combined\n"
            "\n"
            "CLI mode (backward compatible):\n"
            "  compare_features.py --all-info ... --output-dir ... --cache-dir ...\n"
        ),
    )

    parser.add_argument('--config', '-c', type=str, default=None,
                        help='JSON config file path')
    parser.add_argument('--route', '-r', type=str, nargs='+', default=None,
                        help='Route name(s) to run, space-separated (default: run all routes)')

    parser.add_argument('--all-info', type=str, default=None,
                        help='Path to all_info.tsv (overrides config)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Output directory (overrides config)')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Feature cache directory (overrides config)')
    parser.add_argument('--strategies', nargs='+', default=None,
                        help='Aggregation strategy names (overrides config)')
    parser.add_argument('--detectors', nargs='+', default=None,
                        help='Detector methods (overrides config). Groups: all, oneclass, binary.')
    parser.add_argument('--max-workers', type=int, default=None,
                        help='Number of parallel threads per detector')

    args = parser.parse_args()

    # CLI overrides
    cli_overrides: Dict[str, Any] = {}
    for key in ('all_info', 'output_dir', 'cache_dir', 'max_workers'):
        val = getattr(args, key, None)
        if val is not None:
            cli_overrides[key] = val
    # --strategies and --detectors from CLI become simple name lists
    if args.strategies is not None:
        cli_overrides['strategies'] = args.strategies
    if args.detectors is not None:
        cli_overrides['detectors'] = args.detectors

    if args.config:
        route_configs = load_config(args.config, route_names=args.route)
        for route_name, cfg in route_configs.items():
            cfg = _deep_merge(cfg, cli_overrides)
            logger.info(f"{'=' * 60}")
            logger.info(f"Route: {route_name}")
            logger.info(f"{'=' * 60}")
            run_comparison(cfg)
    elif args.all_info and args.output_dir and args.cache_dir:
        cfg: Dict[str, Any] = _deep_merge(DEFAULTS, cli_overrides)
        run_comparison(cfg)
    else:
        parser.error("Either --config or (--all-info + --output-dir + --cache-dir) is required")
