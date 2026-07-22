# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Shared utility functions for MSI detection."""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
def _common_features(lf: pd.DataFrame) -> Dict:
    """Compute common features shared by most aggregation strategies.

    Parameters
    ----------
    lf : pd.DataFrame
        Locus-level dataframe with columns: alt_ratio, entropy, del_ratio,
        ins_ratio, ref_ratio, mean_shift, max_shift.

    Returns
    -------
    Dict
        Dictionary of common feature name-value pairs.
    """
    alt = lf['alt_ratio']
    ent = lf['entropy']
    return {
        'n_loci': len(lf),
        'mean_alt': alt.mean(),
        'mean_entropy': ent.mean(),
        'mean_del_ratio': lf['del_ratio'].mean(),
        'mean_ins_ratio': lf['ins_ratio'].mean(),
        'mean_ref_ratio': lf['ref_ratio'].mean(),
        'mean_shift': lf['mean_shift'].mean(),
        'max_shift': lf['max_shift'].max(),
    }

def _unit_len_features(lf: pd.DataFrame) -> Dict:
    """Compute per unit_len (mono/di/tri) breakdown features.

    Parameters
    ----------
    lf : pd.DataFrame
        Locus-level dataframe with columns: unit_len, alt_ratio, entropy,
        ins_ratio, del_ratio.

    Returns
    -------
    Dict
        Dictionary with keys like ``alt_unit1``, ``entropy_unit2``, etc.
    """
    features = {}
    for ul in [1, 2, 3]:
        mask = lf['unit_len'] == ul
        if mask.sum() > 0:
            features[f'alt_unit{ul}'] = lf.loc[mask, 'alt_ratio'].mean()
            features[f'entropy_unit{ul}'] = lf.loc[mask, 'entropy'].mean()
            features[f'n_unit{ul}'] = int(mask.sum())
            features[f'ins_ratio_unit{ul}'] = lf.loc[mask, 'ins_ratio'].mean()
            features[f'del_ratio_unit{ul}'] = lf.loc[mask, 'del_ratio'].mean()
        else:
            features[f'alt_unit{ul}'] = 0
            features[f'entropy_unit{ul}'] = 0
            features[f'n_unit{ul}'] = 0
            features[f'ins_ratio_unit{ul}'] = 0
            features[f'del_ratio_unit{ul}'] = 0
    return features

def parse_repeat_counts(dist_str: str) -> Dict[int, int]:
    """Parse a repeat distribution string into a count dictionary.

    Parameters
    ----------
    dist_str : str
        Comma-separated distribution string, e.g. "5:120,6:30,7:5".

    Returns
    -------
    Dict[int, int]
        Mapping from repeat length to observed read count.
    """
    counts: Dict[int, int] = {}
    for item in dist_str.split(','):
        parts = item.split(':')
        if len(parts) == 2:
            try:
                counts[int(parts[0])] = int(parts[1])
            except ValueError:
                continue
    return counts


# ── Aggregation strategies ──

def compute_roc(y_true: np.ndarray, scores: np.ndarray, pos_label: str = 'MSI-H') -> Tuple:
    """Compute ROC curve and AUC.

    Sorts samples by anomaly score descending, sweeps all possible
    thresholds, and computes TPR/FPR at each. The algorithm:
    1. Rank samples by score (highest = most anomalous = predicted positive)
    2. Walk down the ranked list, accumulating TP and FP counts
    3. Normalize to TPR and FPR
    4. Compute AUC via trapezoidal integration

    No explicit threshold is needed — the curve covers all thresholds.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (e.g. ['MSI-H', 'MSS', 'MSS', ...]).
    scores : np.ndarray
        Anomaly scores for each sample (higher = more anomalous).
    pos_label : str
        Label considered as positive class.

    Returns
    -------
    tuple
        (fpr array, tpr array, auc scalar).
        fpr and tpr include the origin point (0, 0).
    """
    y_bin = (np.array(y_true) == pos_label).astype(int)
    idx = np.argsort(scores)[::-1]
    y_s = y_bin[idx]
    tps = np.cumsum(y_s)
    fps = np.cumsum(1 - y_s)
    tpr = np.concatenate([[0], tps / tps[-1]])
    fpr = np.concatenate([[0], fps / fps[-1]])
    return fpr, tpr, trapezoid(tpr, fpr)

def find_best_threshold(y_true: np.ndarray, scores: np.ndarray, pos_label: str = 'MSI-H') -> Tuple:
    """Find threshold maximizing Youden's J statistic.

    Youden's J = TPR - FPR, which balances sensitivity and specificity.
    The algorithm sorts samples by anomaly score descending, sweeps all
    candidate thresholds, and picks the one that maximizes J.

    Requires labeled data (both positive and negative samples), so it is
    suitable for evaluation on labeled test/validation sets but not for
    setting the decision boundary in a pure anomaly-detection training
    phase where only normal-class samples are available. For the latter,
    use n-sigma method instead (threshold = mean + n * std of normal scores).

    Parameters
    ----------
    y_true : np.ndarray
        True labels (e.g. ['MSI-H', 'MSS', 'MSS', ...]).
    scores : np.ndarray
        Anomaly scores for each sample.
    pos_label : str
        Label considered as positive class.

    Returns
    -------
    tuple
        (best_threshold, tpr_at_threshold, fpr_at_threshold)
    """
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

def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float, pos: str = 'MSI-H') -> Dict:
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
