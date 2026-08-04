# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Model interpretability module for MSI detection.

Provides detector-specific explanation methods:
- Logistic Regression: coefficient-based feature importance
- XGBoost: SHAP values
- One-class detectors: distance/contribution analysis
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class InterpretabilityBase(ABC):
    """Abstract base class for model interpretability."""
    
    @abstractmethod
    def fit(self, detector: Any, feature_names: List[str], X_train: np.ndarray, 
            y_train: Optional[np.ndarray] = None) -> 'InterpretabilityBase':
        """Fit the interpreter on trained detector."""
        pass
    
    @abstractmethod
    def explain(self, X: np.ndarray, sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate explanations for samples."""
        pass
    
    @abstractmethod
    def plot_feature_importance(self, output_path: str, top_n: int = 20) -> str:
        """Plot global feature importance."""
        pass
    
    @abstractmethod
    def plot_sample_explanation(self, X: np.ndarray, sample_idx: int, 
                                output_path: str) -> str:
        """Plot explanation for a single sample."""
        pass


class LogisticInterpretable(InterpretabilityBase):
    """Interpretability for Logistic Regression using coefficients.
    
    For clinical registration: coefficients directly show feature contributions
    to MSI probability. Can be expressed as log-odds ratio.
    """
    
    def __init__(self):
        self.coefficients_ = None
        self.feature_names_ = None
        self.intercept_ = None
        self.scaler_ = None
    
    def fit(self, detector: Any, feature_names: List[str], X_train: np.ndarray,
            y_train: Optional[np.ndarray] = None) -> 'LogisticInterpretable':
        """Extract coefficients from fitted logistic regression."""
        if not hasattr(detector, 'model_') or detector.model_ is None:
            raise ValueError("Detector must be fitted first")
        
        self.feature_names_ = list(feature_names)
        self.scaler_ = detector.scaler_
        
        # Get coefficients (scaled space)
        self.coefficients_ = detector.model_.coef_[0]
        self.intercept_ = detector.model_.intercept_[0]
        
        # Also compute unscaled coefficients for clinical interpretation
        if self.scaler_ is not None and hasattr(self.scaler_, 'scale_'):
            # coef_unscaled = coef_scaled / scale
            self.coef_unscaled_ = self.coefficients_ / self.scaler_.scale_
            self.intercept_unscaled_ = self.intercept_ - np.sum(
                self.coefficients_ * self.scaler_.mean_ / self.scaler_.scale_
            )
        else:
            self.coef_unscaled_ = self.coefficients_
            self.intercept_unscaled_ = self.intercept_
        
        logger.info(f"LogisticInterpretable: extracted {len(self.coefficients_)} coefficients")
        return self
    
    def explain(self, X: np.ndarray, sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compute feature contributions for samples.
        
        Returns contribution in log-odds space: contribution_i = coef_i * x_i
        """
        if self.coefficients_ is None:
            raise ValueError("Not fitted. Call fit() first.")
        
        n_samples = X.shape[0]
        
        # Scale features
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        # Compute per-feature contributions (coef * feature_value)
        contributions = X_scaled * self.coefficients_
        
        results = {
            'feature_names': self.feature_names_,
            'coefficients': self.coefficients_,
            'coef_unscaled': self.coef_unscaled_,
            'intercept': self.intercept_,
            'intercept_unscaled': self.intercept_unscaled_,
            'contributions': contributions,  # (n_samples, n_features)
            'sample_ids': sample_ids or [f'sample_{i}' for i in range(n_samples)],
        }
        
        return results
    
    def get_odds_ratios(self) -> pd.DataFrame:
        """Compute odds ratios for clinical interpretation.
        
        Odds ratio = exp(coef_unscaled): how much odds of MSI-H multiply
        per unit increase in feature.
        """
        if self.coef_unscaled_ is None:
            raise ValueError("Not fitted. Call fit() first.")
        
        df = pd.DataFrame({
            'feature': self.feature_names_,
            'coefficient': self.coef_unscaled_,
            'odds_ratio': np.exp(self.coef_unscaled_),
            'abs_coefficient': np.abs(self.coef_unscaled_),
        })
        df = df.sort_values('abs_coefficient', ascending=False)
        return df
    
    def plot_feature_importance(self, output_path: str, top_n: int = 20) -> str:
        """Plot coefficient-based feature importance."""
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        odds_df = self.get_odds_ratios().head(top_n)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: coefficients (directional)
        colors = ['#e74c3c' if c > 0 else '#3498db' for c in odds_df['coefficient']]
        ax1.barh(range(len(odds_df)), odds_df['coefficient'], color=colors, alpha=0.8)
        ax1.set_yticks(range(len(odds_df)))
        ax1.set_yticklabels(odds_df['feature'], fontsize=9)
        ax1.set_xlabel('Coefficient (log-odds per unit)', fontsize=11)
        ax1.set_title('Feature Coefficients', fontsize=12)
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax1.grid(axis='x', alpha=0.3)
        # Add color legend outside
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#e74c3c', alpha=0.8, label='Increases MSI-H'),
                          Patch(facecolor='#3498db', alpha=0.8, label='Decreases MSI-H')]
        ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
        
        # Right: odds ratios (multiplicative)
        ax2.barh(range(len(odds_df)), odds_df['odds_ratio'], color='#2ecc71', alpha=0.8)
        ax2.set_yticks(range(len(odds_df)))
        ax2.set_yticklabels(odds_df['feature'], fontsize=9)
        ax2.set_xlabel('Odds Ratio (exp(coef))', fontsize=11)
        ax2.set_title('Odds Ratios (>1 increases MSI-H odds)', fontsize=12)
        ax2.axvline(x=1, color='black', linestyle='--', linewidth=1)
        ax2.grid(axis='x', alpha=0.3)
        
        plt.suptitle('Logistic Regression Interpretability', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved feature importance plot: {output_path}")
        return output_path
    
    def plot_sample_explanation(self, X: np.ndarray, sample_idx: int,
                                output_path: str, 
                                true_label: str = None,
                                predicted_label: str = None,
                                score: float = None,
                                threshold: float = None,
                                X_mss: np.ndarray = None,
                                X_msih: np.ndarray = None) -> str:
        """Plot waterfall-style explanation for a single sample with context.
        
        Parameters
        ----------
        X : np.ndarray
            Test features.
        sample_idx : int
            Index of sample to explain.
        output_path : str
            Path to save plot.
        true_label : str, optional
            True MSI status ('MSI-H' or 'MSS').
        predicted_label : str, optional
            Predicted MSI status.
        score : float, optional
            Model score for this sample.
        threshold : float, optional
            Decision threshold.
        X_mss : np.ndarray, optional
            MSS training samples for reference distribution.
        X_msih : np.ndarray, optional
            MSI-H training samples for reference distribution.
        """
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        from scipy.special import expit, logit
        
        results = self.explain(X)
        contrib = results['contributions'][sample_idx]
        intercept = results['intercept']
        
        # Sort by absolute contribution, show all features
        sorted_idx = np.argsort(np.abs(contrib))[::-1]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [2, 1]})
        
        # Left: Waterfall plot
        ax1 = axes[0]
        cumulative = [intercept]
        labels = ['Intercept']
        colors_list = ['#95a5a6']
        
        for idx in sorted_idx:
            cumulative.append(cumulative[-1] + contrib[idx])
            labels.append(results['feature_names'][idx])
            colors_list.append('#e74c3c' if contrib[idx] > 0 else '#3498db')
        
        final_logit = cumulative[-1]
        final_prob = expit(final_logit)
        
        x_pos = range(len(cumulative))
        bars = ax1.bar(x_pos, cumulative, color=colors_list, alpha=0.8, edgecolor='white')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels, rotation=60, ha='right', fontsize=7)
        ax1.set_ylabel('Log-Odds', fontsize=11)
        
        # Add secondary y-axis for probability
        ax1_prob = ax1.twinx()
        ax1_prob.set_ylim(ax1.get_ylim())
        # Convert log-odds ticks to probability
        yticks_logit = ax1.get_yticks()
        yticks_prob = [expit(y) for y in yticks_logit]
        ax1_prob.set_yticks(yticks_logit)
        ax1_prob.set_yticklabels([f'{p:.2f}' for p in yticks_prob], fontsize=8)
        ax1_prob.set_ylabel('P(MSI-H)', fontsize=10)
        
        # Add threshold line if provided
        if threshold is not None:
            threshold_logit = logit(threshold)
            ax1.axhline(y=threshold_logit, color='red', linestyle='--', linewidth=1.5)
            ax1.text(len(cumulative)-0.5, threshold_logit, f'Threshold={threshold:.3f}',
                    color='red', fontsize=8, ha='right', va='bottom')
        
        # Title with sample info
        title_parts = []
        if true_label:
            title_parts.append(f'True: {true_label}')
        if predicted_label:
            title_parts.append(f'Pred: {predicted_label}')
        if score is not None:
            title_parts.append(f'Score: {score:.3f}')
        title = ' | '.join(title_parts) if title_parts else f'Sample {sample_idx}'
        ax1.set_title(title, fontsize=12, fontweight='bold')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.grid(axis='y', alpha=0.3)
        
        # Right: Feature value comparison (if reference distributions provided)
        ax2 = axes[1]
        if X_mss is not None and X_msih is not None:
            # Show top 8 features: sample value vs MSS/MSI-H distribution
            top_n = min(8, len(sorted_idx))
            feature_names = [results['feature_names'][i] for i in sorted_idx[:top_n]]
            sample_values = [X[sample_idx, i] for i in sorted_idx[:top_n]]
            mss_means = [X_mss[:, i].mean() for i in sorted_idx[:top_n]]
            msih_means = [X_msih[:, i].mean() for i in sorted_idx[:top_n]]
            
            y_pos = range(top_n)
            
            # Plot reference means
            ax2.scatter(mss_means, y_pos, color='#3498db', marker='|', s=200, 
                       linewidths=3, label='MSS mean', zorder=3)
            ax2.scatter(msih_means, y_pos, color='#e74c3c', marker='|', s=200, 
                       linewidths=3, label='MSI-H mean', zorder=3)
            
            # Plot sample value
            ax2.scatter(sample_values, y_pos, color='black', marker='o', s=80, 
                       label='This sample', zorder=4)
            
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(feature_names, fontsize=9)
            ax2.set_xlabel('Feature Value (standardized)', fontsize=10)
            ax2.set_title('Feature Comparison', fontsize=11)
            ax2.legend(loc='lower right', fontsize=8)
            ax2.grid(axis='x', alpha=0.3)
            ax2.invert_yaxis()
        else:
            # Just show contribution values as text
            ax2.axis('off')
            text_lines = ['Top Features:', '']
            for i, idx in enumerate(sorted_idx[:10]):
                fname = results['feature_names'][idx]
                c = contrib[idx]
                direction = '↑ MSI-H' if c > 0 else '↓ MSS'
                text_lines.append(f'{fname}: {c:+.3f} ({direction})')
            ax2.text(0.1, 0.9, '\n'.join(text_lines), transform=ax2.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved sample explanation: {output_path}")
        return output_path
    
    def generate_clinical_report(self, output_path: str) -> str:
        """Generate clinical-ready interpretation report."""
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        odds_df = self.get_odds_ratios()
        
        lines = [
            "# MSI Detection Model - Clinical Interpretability Report",
            "",
            "## Model Type: Logistic Regression with L2 Regularization",
            "",
            "## Decision Formula",
            "",
            "P(MSI-H) = sigmoid(intercept + sum(coef_i * feature_i))",
            "",
            f"Intercept (raw): {self.intercept_unscaled_:.4f}",
            "",
            "## Feature Coefficients and Odds Ratios",
            "",
            "| Rank | Feature | Coefficient | Odds Ratio | Interpretation |",
            "|------|---------|-------------|------------|----------------|",
        ]
        
        for rank, (_, row) in enumerate(odds_df.iterrows(), 1):
            coef = row['coefficient']
            or_val = row['odds_ratio']
            if coef > 0:
                interp = f"Each unit increase multiplies MSI-H odds by {or_val:.2f}x"
            else:
                interp = f"Each unit increase multiplies MSI-H odds by {or_val:.2f}x (protective)"
            lines.append(f"| {rank} | {row['feature']} | {coef:.4f} | {or_val:.4f} | {interp} |")
        
        lines.extend([
            "",
            "## Clinical Interpretation",
            "",
            "- Features with positive coefficients increase MSI-H probability",
            "- Features with negative coefficients decrease MSI-H probability",
            "- Odds Ratio > 1: feature increases MSI-H odds",
            "- Odds Ratio < 1: feature decreases MSI-H odds",
            "",
            "## Regulatory Notes",
            "",
            "- Model uses L2 regularization (Ridge) for stability",
            "- All features are standardized (z-score) before fitting",
            "- Coefficients represent log-odds change per standard deviation",
            "",
        ])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Saved clinical report: {output_path}")
        return output_path


class XGBoostSHAPInterpretable(InterpretabilityBase):
    """Interpretability for XGBoost using SHAP values.
    
    SHAP (SHapley Additive exPlanations) provides consistent,
    locally accurate feature attributions.
    """
    
    def __init__(self):
        self.explainer_ = None
        self.feature_names_ = None
        self.shap_values_ = None
    
    def fit(self, detector: Any, feature_names: List[str], X_train: np.ndarray,
            y_train: Optional[np.ndarray] = None) -> 'XGBoostSHAPInterpretable':
        """Initialize SHAP explainer for XGBoost."""
        try:
            import shap
        except ImportError:
            raise ImportError("shap package required for XGBoost interpretability. "
                            "Install with: pip install shap")
        
        if not hasattr(detector, 'model_') or detector.model_ is None:
            raise ValueError("Detector must be fitted first")
        
        self.feature_names_ = list(feature_names)
        
        # Create SHAP explainer
        # Use TreeExplainer for XGBoost (fast, exact)
        self.explainer_ = shap.TreeExplainer(detector.model_)
        
        # Compute SHAP values on training data for global importance
        X_scaled = detector.scaler_.transform(X_train) if detector.scaler_ else X_train
        self.shap_values_ = self.explainer_.shap_values(X_scaled)
        
        logger.info(f"XGBoostSHAPInterpretable: initialized with {len(feature_names)} features")
        return self
    
    def explain(self, X: np.ndarray, sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compute SHAP values for samples."""
        if self.explainer_ is None:
            raise ValueError("Not fitted. Call fit() first.")
        
        # Compute SHAP values
        shap_values = self.explainer_.shap_values(X)
        
        return {
            'feature_names': self.feature_names_,
            'shap_values': shap_values,  # (n_samples, n_features)
            'expected_value': self.explainer_.expected_value,
            'sample_ids': sample_ids or [f'sample_{i}' for i in range(X.shape[0])],
        }
    
    def plot_feature_importance(self, output_path: str, top_n: int = 20) -> str:
        """Plot SHAP-based feature importance (mean |SHAP|)."""
        import shap
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # Use SHAP's built-in plotting
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(self.shap_values_, feature_names=self.feature_names_,
                         show=False, max_display=top_n)
        plt.title('XGBoost Feature Importance (Mean |SHAP Value|)', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved SHAP feature importance: {output_path}")
        return output_path
    
    def plot_sample_explanation(self, X: np.ndarray, sample_idx: int,
                                output_path: str,
                                true_label: str = None,
                                predicted_label: str = None,
                                score: float = None,
                                threshold: float = None,
                                X_mss: np.ndarray = None,
                                X_msih: np.ndarray = None) -> str:
        """Plot SHAP waterfall with reference comparison."""
        import shap
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # Get SHAP values for this sample
        shap_values = self.explainer_.shap_values(X[sample_idx:sample_idx+1])
        
        # Create figure with two panels
        fig = plt.figure(figsize=(16, 8))
        
        # Left panel: SHAP waterfall (using shap's built-in)
        ax1 = fig.add_subplot(121)
        shap.waterfall_plot(shap.Explanation(
            values=shap_values[0],
            base_values=self.explainer_.expected_value,
            data=X[sample_idx],
            feature_names=self.feature_names_
        ), show=False, max_display=10)
        
        # Right panel: Feature comparison with reference
        ax2 = fig.add_subplot(122)
        if X_mss is not None and X_msih is not None:
            # Get top features by SHAP value magnitude
            sorted_idx = np.argsort(np.abs(shap_values[0]))[::-1][:8]
            
            feature_names = [self.feature_names_[i] for i in sorted_idx]
            sample_values = [X[sample_idx, i] for i in sorted_idx]
            mss_means = [X_mss[:, i].mean() for i in sorted_idx]
            msih_means = [X_msih[:, i].mean() for i in sorted_idx]
            
            y_pos = range(len(sorted_idx))
            
            # Plot reference means
            ax2.scatter(mss_means, y_pos, color='#3498db', marker='|', s=200, 
                       linewidths=3, label='MSS mean', zorder=3)
            ax2.scatter(msih_means, y_pos, color='#e74c3c', marker='|', s=200, 
                       linewidths=3, label='MSI-H mean', zorder=3)
            
            # Plot sample value
            ax2.scatter(sample_values, y_pos, color='black', marker='o', s=80, 
                       label='This sample', zorder=4)
            
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(feature_names, fontsize=9)
            ax2.set_xlabel('Feature Value (standardized)', fontsize=10)
            ax2.set_title('Feature Comparison', fontsize=11)
            ax2.legend(loc='lower right', fontsize=8)
            ax2.grid(axis='x', alpha=0.3)
            ax2.invert_yaxis()
        
        # Title with sample info
        title_parts = []
        if true_label:
            title_parts.append(f'True: {true_label}')
        if predicted_label:
            title_parts.append(f'Pred: {predicted_label}')
        if score is not None:
            title_parts.append(f'Score: {score:.3f}')
        title = ' | '.join(title_parts) if title_parts else f'Sample {sample_idx}'
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved SHAP sample explanation: {output_path}")
        return output_path


class DistanceBasedInterpretable(InterpretabilityBase):
    """Interpretability for distance-based detectors (Mahalanobis, Cosine, etc.)."""
    
    def __init__(self):
        self.reference_center_ = None
        self.feature_names_ = None
        self.cov_inv_ = None
    
    def fit(self, detector: Any, feature_names: List[str], X_train: np.ndarray,
            y_train: Optional[np.ndarray] = None) -> 'DistanceBasedInterpretable':
        """Extract reference distribution from fitted detector."""
        self.feature_names_ = list(feature_names)
        
        # Store training data statistics
        self.reference_center_ = np.mean(X_train, axis=0)
        self.train_std_ = np.std(X_train, axis=0)
        
        # Try to get covariance inverse from detector
        if hasattr(detector, 'cov_inv_'):
            self.cov_inv_ = detector.cov_inv_
        
        logger.info(f"DistanceBasedInterpretable: computed reference center")
        return self
    
    def explain(self, X: np.ndarray, sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compute per-feature distance contributions."""
        n_samples = X.shape[0]
        
        # Compute deviation from reference
        deviation = X - self.reference_center_
        
        # Normalize by std
        deviation_normalized = deviation / (self.train_std_ + 1e-10)
        
        # Per-feature contribution to distance
        if self.cov_inv_ is not None:
            # Mahalanobis decomposition
            contributions = deviation @ self.cov_inv_ * deviation
        else:
            # Simple squared deviation
            contributions = deviation_normalized ** 2
        
        return {
            'feature_names': self.feature_names_,
            'contributions': contributions,  # (n_samples, n_features)
            'deviation': deviation,
            'deviation_normalized': deviation_normalized,
            'reference_center': self.reference_center_,
            'sample_ids': sample_ids or [f'sample_{i}' for i in range(n_samples)],
        }
    
    def plot_feature_importance(self, output_path: str, top_n: int = 20) -> str:
        """Plot mean feature contribution across training data."""
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # This would need training data contributions
        # For now, plot reference center
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sorted_idx = np.argsort(np.abs(self.reference_center_))[::-1][:top_n]
        
        ax.barh(range(top_n), self.reference_center_[sorted_idx], 
               color='#9b59b6', alpha=0.8)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([self.feature_names_[i] for i in sorted_idx], fontsize=9)
        ax.set_xlabel('Reference Center Value (Mean)', fontsize=11)
        ax.set_title('Distance-Based Detector: Reference Center', fontsize=13)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved distance-based importance: {output_path}")
        return output_path
    
    def plot_sample_explanation(self, X: np.ndarray, sample_idx: int,
                                output_path: str) -> str:
        """Plot per-feature deviation for a sample."""
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        results = self.explain(X)
        contrib = results['contributions'][sample_idx]
        
        sorted_idx = np.argsort(contrib)[::-1][:15]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(sorted_idx)))
        ax.barh(range(len(sorted_idx)), contrib[sorted_idx], color=colors, alpha=0.8)
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([self.feature_names_[i] for i in sorted_idx], fontsize=9)
        ax.set_xlabel('Contribution to Distance', fontsize=11)
        ax.set_title(f'Sample {sample_idx}: Distance Contribution', fontsize=13)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved sample distance explanation: {output_path}")
        return output_path


def get_interpreter(detector: Any) -> InterpretabilityBase:
    """Factory function to get appropriate interpreter for detector type.
    
    Parameters
    ----------
    detector : Detector
        Fitted detector instance.
    
    Returns
    -------
    InterpretabilityBase
        Appropriate interpreter for the detector type.
    """
    # Check detector type by class name to avoid import issues
    detector_class_name = detector.__class__.__name__
    
    if detector_class_name == 'BinaryClassifierDetector':
        if hasattr(detector, 'method'):
            if detector.method == 'logistic':
                return LogisticInterpretable()
            elif detector.method == 'xgboost':
                try:
                    return XGBoostSHAPInterpretable()
                except ImportError:
                    logger.warning("SHAP not available, falling back to distance-based")
                    return DistanceBasedInterpretable()
    
    elif detector_class_name in ('MahalanobisDetector', 'CosineDetector'):
        return DistanceBasedInterpretable()
    
    # Default: try SHAP if available, otherwise distance-based
    try:
        return XGBoostSHAPInterpretable()
    except ImportError:
        return DistanceBasedInterpretable()


def add_interpretability_to_pipeline(pipeline_results: Dict[str, Any],
                                     output_dir: str,
                                     n_misclassified: int = 5) -> Dict[str, str]:
    """Add interpretability analysis to pipeline results.
    
    Parameters
    ----------
    pipeline_results : dict
        Results from MSIDetectionPipeline.run().
    output_dir : str
        Directory to save interpretability outputs.
    n_misclassified : int
        Number of FP and FN samples to explain each (default: 5).
    
    Returns
    -------
    dict
        Paths to generated interpretability files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get detector and features
    detector = pipeline_results.get('detector')
    selected_cols = pipeline_results.get('selected_cols', [])
    X_train = pipeline_results.get('X_train')
    y_train = pipeline_results.get('y_train')
    X_test = pipeline_results.get('X_test')
    test_results = pipeline_results.get('test', {})
    
    if detector is None or X_train is None:
        logger.warning("Missing detector or training data for interpretability")
        return {}
    
    # Get appropriate interpreter
    interpreter = get_interpreter(detector)
    
    # Fit interpreter
    interpreter.fit(detector, selected_cols, X_train, y_train)
    
    # Prepare reference distributions
    X_mss = X_train[y_train == 0] if y_train is not None else None
    X_msih = X_train[y_train == 1] if y_train is not None else None
    
    outputs = {}
    
    # Global feature importance
    importance_path = os.path.join(output_dir, 'feature_importance.png')
    outputs['feature_importance'] = interpreter.plot_feature_importance(importance_path)
    
    # Misclassified sample explanations (FP and FN separately)
    if X_test is not None and 'df' in test_results:
        test_df = test_results['df']
        test_scores = test_results.get('scores')
        threshold = pipeline_results.get('threshold', 0.5)
        
        if test_scores is not None and 'MSI_status' in test_df.columns:
            # Get predictions
            predictions = np.where(test_scores >= threshold, 'MSI-H', 'MSS')
            true_labels = test_df['MSI_status'].values
            
            # Find FP: predicted MSI-H but actually MSS
            fp_mask = (predictions == 'MSI-H') & (true_labels == 'MSS')
            fp_indices = np.where(fp_mask)[0]
            # Sort by score (highest confidence FP first)
            fp_sorted = fp_indices[np.argsort(-test_scores[fp_indices])]
            
            # Find FN: predicted MSS but actually MSI-H
            fn_mask = (predictions == 'MSS') & (true_labels == 'MSI-H')
            fn_indices = np.where(fn_mask)[0]
            # Sort by score (lowest confidence FN first, most surprising)
            fn_sorted = fn_indices[np.argsort(test_scores[fn_indices])]
            
            # Get sample IDs from test DataFrame index
            sample_ids = test_df.index.tolist()
            
            # Plot FP samples
            n_fp = min(n_misclassified, len(fp_sorted))
            for rank, idx in enumerate(fp_sorted[:n_fp]):
                score = test_scores[idx]
                actual_sample_id = sample_ids[idx] if idx < len(sample_ids) else f'idx{idx}'
                # Sanitize sample ID for filename
                safe_id = str(actual_sample_id).replace('/', '_').replace('\\', '_').replace(' ', '_')
                sample_path = os.path.join(
                    output_dir, f'FP_{rank+1}_{safe_id}_score{score:.3f}.png'
                )
                outputs[f'FP_{rank+1}'] = interpreter.plot_sample_explanation(
                    X_test, idx, sample_path,
                    true_label='MSS',
                    predicted_label='MSI-H',
                    score=score,
                    threshold=threshold,
                    X_mss=X_mss,
                    X_msih=X_msih
                )
            
            # Plot FN samples
            n_fn = min(n_misclassified, len(fn_sorted))
            for rank, idx in enumerate(fn_sorted[:n_fn]):
                score = test_scores[idx]
                actual_sample_id = sample_ids[idx] if idx < len(sample_ids) else f'idx{idx}'
                safe_id = str(actual_sample_id).replace('/', '_').replace('\\', '_').replace(' ', '_')
                sample_path = os.path.join(
                    output_dir, f'FN_{rank+1}_{safe_id}_score{score:.3f}.png'
                )
                outputs[f'FN_{rank+1}'] = interpreter.plot_sample_explanation(
                    X_test, idx, sample_path,
                    true_label='MSI-H',
                    predicted_label='MSS',
                    score=score,
                    threshold=threshold,
                    X_mss=X_mss,
                    X_msih=X_msih
                )
            
            logger.info(f"  Plotted FP: {n_fp}/{len(fp_indices)}, FN: {n_fn}/{len(fn_indices)}")
    
    # Clinical report for logistic regression
    if isinstance(interpreter, LogisticInterpretable):
        report_path = os.path.join(output_dir, 'clinical_interpretation.md')
        outputs['clinical_report'] = interpreter.generate_clinical_report(report_path)
        
        # Save odds ratios as TSV
        odds_df = interpreter.get_odds_ratios()
        odds_path = os.path.join(output_dir, 'odds_ratios.tsv')
        odds_df.to_csv(odds_path, sep='\t', index=False)
        outputs['odds_ratios'] = odds_path
    
    logger.info(f"Generated {len(outputs)} interpretability files in {output_dir}")
    return outputs
