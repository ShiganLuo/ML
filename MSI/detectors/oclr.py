# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""OCLR (One-Class Logistic Regression) detector."""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from .base import Detector

class OCLRDetector(Detector):
    """One-Class Logistic Regression (OCLR) anomaly detection.

    Learns a weight vector from normal (MSS) samples only by maximising
    the logistic likelihood with L2 regularisation:

        max_w  Σ_i log σ(w^T x_i)  -  λ/2 ||w||²

    Scores new samples by Spearman correlation with the learned weights.

    Reference: Sweeney et al., 2018, Cell Systems.

    Parameters
    ----------
    l2 : float
        L2 regularisation strength (lambda).
    scoring : str
        Scoring method: 'spearman', 'pearson', 'dot', or 'logistic'.
    max_iter : int
        Maximum L-BFGS-B iterations.
    """

    def __init__(self, l2: float = 1.0, scoring: str = 'spearman', max_iter: int = 1000):
        self.l2 = l2
        self.scoring = scoring
        self.max_iter = max_iter
        self.w_ = None
        self.scaler_ = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        out = np.empty_like(z)
        pos = z >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        exp_z = np.exp(z[neg])
        out[neg] = exp_z / (1.0 + exp_z)
        return out

    def _objective(self, w: np.ndarray, X: np.ndarray) -> Tuple[float, np.ndarray]:
        """Negative log-likelihood + L2 penalty with gradient."""
        z = X @ w
        sigma = self._sigmoid(z)
        nll = -np.sum(np.log(sigma + 1e-15))
        reg = 0.5 * self.l2 * np.dot(w, w)
        loss = nll + reg
        grad = -X.T @ (1.0 - sigma) + self.l2 * w
        return loss, grad

    def fit(self, X_train: np.ndarray) -> 'OCLRDetector':
        from scipy.optimize import minimize
        from sklearn.preprocessing import StandardScaler

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_train)

        n_features = X_scaled.shape[1]

        # Multiple random initialisations to escape saddle points.
        # After centering, the gradient at w=0 is zero, so different
        # random seeds converge to different local minima.
        best_loss = np.inf
        best_w = None
        for seed in range(20):
            rng = np.random.RandomState(seed)
            w0 = rng.randn(n_features) * 0.01
            res = minimize(
                self._objective, w0, args=(X_scaled,),
                method='L-BFGS-B', jac=True,
                options={'maxiter': self.max_iter, 'gtol': 1e-10},
            )
            if res.fun < best_loss:
                best_loss = res.fun
                best_w = res.x

        self.w_ = best_w
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        from scipy.stats import spearmanr
        X_scaled = self.scaler_.transform(X)
        w = self.w_

        if self.scoring == 'spearman':
            scores = np.array([spearmanr(w, X_scaled[i])[0] for i in range(X_scaled.shape[0])])
        elif self.scoring == 'pearson':
            w_norm = (w - w.mean()) / (np.std(w) + 1e-15)
            x_mean = X_scaled - X_scaled.mean(axis=1, keepdims=True)
            x_std = np.std(X_scaled, axis=1, keepdims=True) + 1e-15
            scores = (x_mean / x_std) @ w_norm / len(w)
        elif self.scoring == 'dot':
            scores = X_scaled @ w
        elif self.scoring == 'logistic':
            scores = self._sigmoid(X_scaled @ w)
        else:
            raise ValueError(f"Unknown scoring: {self.scoring}")

        # Negate: OCLR learns "normal" → normal gets HIGH scores.
        # For anomaly detection, anomalies should get HIGH scores.
        return -scores

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        scores = self.score(X)
        return np.where(scores >= threshold, 'MSI-H', 'MSS')
