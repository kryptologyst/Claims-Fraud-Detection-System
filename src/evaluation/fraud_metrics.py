"""Comprehensive evaluation metrics for fraud detection."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    balanced_accuracy_score, roc_curve
)
from sklearn.calibration import calibration_curve
import logging

logger = logging.getLogger(__name__)


class FraudDetectionEvaluator:
    """Comprehensive evaluator for fraud detection models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize fraud detection evaluator.
        
        Args:
            config: Evaluation configuration dictionary
        """
        self.config = config
        self.metrics = config.get('metrics', ['auc', 'precision', 'recall', 'f1'])
        self.k_values = config.get('k_values', [10, 50, 100])
        
    def evaluate(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model performance with comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Starting comprehensive model evaluation...")
        
        results = {}
        
        # Basic classification metrics
        if 'auc' in self.metrics:
            results['auc'] = roc_auc_score(y_true, y_proba)
            
        if 'precision' in self.metrics:
            results['precision'] = precision_score(y_true, y_pred, average='macro')
            
        if 'recall' in self.metrics:
            results['recall'] = recall_score(y_true, y_pred, average='macro')
            
        if 'f1' in self.metrics:
            results['f1'] = f1_score(y_true, y_pred, average='macro')
            
        if 'accuracy' in self.metrics:
            results['accuracy'] = (y_pred == y_true).mean()
            
        if 'balanced_accuracy' in self.metrics:
            results['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
            
        # Precision@K and Recall@K metrics
        if 'precision_at_k' in self.metrics or 'recall_at_k' in self.metrics:
            precision_k, recall_k = self._calculate_precision_recall_at_k(y_true, y_proba)
            results.update(precision_k)
            results.update(recall_k)
            
        # Specificity
        if 'specificity' in self.metrics:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
        # Cost-benefit analysis
        if self.config.get('cost_benefit', {}).get('enabled', False):
            cost_benefit = self._calculate_cost_benefit(y_true, y_pred, y_proba)
            results.update(cost_benefit)
            
        # Calibration metrics
        if self.config.get('calibration', {}).get('enabled', False):
            calibration_metrics = self._calculate_calibration_metrics(y_true, y_proba)
            results.update(calibration_metrics)
            
        logger.info(f"Evaluation completed. AUC: {results.get('auc', 'N/A'):.4f}")
        
        return results
        
    def _calculate_precision_recall_at_k(
        self, 
        y_true: np.ndarray, 
        y_proba: np.ndarray
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate precision@K and recall@K metrics.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Tuple of (precision@K dict, recall@K dict)
        """
        precision_k = {}
        recall_k = {}
        
        # Sort by probability (descending)
        sorted_indices = np.argsort(y_proba)[::-1]
        sorted_y_true = y_true[sorted_indices]
        
        for k in self.k_values:
            if k > len(y_true):
                continue
                
            # Top K predictions
            top_k_true = sorted_y_true[:k]
            
            # Precision@K: fraction of top K that are actually fraud
            precision_k[f'precision_at_{k}'] = top_k_true.mean()
            
            # Recall@K: fraction of all fraud cases captured in top K
            total_fraud = y_true.sum()
            if total_fraud > 0:
                recall_k[f'recall_at_{k}'] = top_k_true.sum() / total_fraud
            else:
                recall_k[f'recall_at_{k}'] = 0.0
                
        return precision_k, recall_k
        
    def _calculate_cost_benefit(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate cost-benefit analysis metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary of cost-benefit metrics
        """
        cost_config = self.config['cost_benefit']
        investigation_cost = cost_config['investigation_cost']
        fraud_loss = cost_config['fraud_loss']
        false_positive_cost = cost_config['false_positive_cost']
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate costs
        investigation_costs = (tp + fp) * investigation_cost
        fraud_losses = fn * fraud_loss
        false_positive_costs = fp * false_positive_cost
        
        total_cost = investigation_costs + fraud_losses + false_positive_costs
        
        # Calculate benefits
        fraud_prevented = tp * fraud_loss
        net_benefit = fraud_prevented - total_cost
        
        # ROI
        roi = net_benefit / total_cost if total_cost > 0 else 0.0
        
        return {
            'investigation_costs': investigation_costs,
            'fraud_losses': fraud_losses,
            'false_positive_costs': false_positive_costs,
            'total_cost': total_cost,
            'fraud_prevented': fraud_prevented,
            'net_benefit': net_benefit,
            'roi': roi
        }
        
    def _calculate_calibration_metrics(
        self, 
        y_true: np.ndarray, 
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate calibration metrics.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary of calibration metrics
        """
        try:
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_proba, n_bins=10
            )
            
            # Calculate calibration error
            calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            
            # Calculate Brier score
            brier_score = np.mean((y_proba - y_true) ** 2)
            
            return {
                'calibration_error': calibration_error,
                'brier_score': brier_score
            }
            
        except Exception as e:
            logger.warning(f"Calibration metrics calculation failed: {e}")
            return {}
            
    def get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Get confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)
        
    def get_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Get detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report string
        """
        return classification_report(y_true, y_pred)
        
    def find_optimal_threshold(
        self, 
        y_true: np.ndarray, 
        y_proba: np.ndarray,
        method: str = 'youden'
    ) -> float:
        """Find optimal threshold for binary classification.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            method: Method to use ('youden', 'f1', 'precision_recall')
            
        Returns:
            Optimal threshold
        """
        if method == 'youden':
            # Youden's J statistic
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            return thresholds[optimal_idx]
            
        elif method == 'f1':
            # F1 score optimization
            precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            return thresholds[optimal_idx]
            
        elif method == 'precision_recall':
            # Precision-recall curve optimization
            precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
            # Find threshold that maximizes precision while maintaining reasonable recall
            optimal_idx = np.argmax(precision + recall)
            return thresholds[optimal_idx]
            
        else:
            raise ValueError(f"Unknown threshold optimization method: {method}")
            
    def evaluate_at_threshold(
        self, 
        y_true: np.ndarray, 
        y_proba: np.ndarray, 
        threshold: float
    ) -> Dict[str, float]:
        """Evaluate model at a specific threshold.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            threshold: Classification threshold
            
        Returns:
            Dictionary of evaluation metrics at threshold
        """
        y_pred = (y_proba >= threshold).astype(int)
        return self.evaluate(y_true, y_pred, y_proba)
