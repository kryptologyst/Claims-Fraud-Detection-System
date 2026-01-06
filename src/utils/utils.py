"""Utility functions for fraud detection system."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import json
import pickle

logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # Set PyTorch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    # Set TensorFlow seed if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def save_results(results: Dict[str, Any], filepath: str):
    """Save results to file.
    
    Args:
        results: Results dictionary to save
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if filepath.suffix == '.json':
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    elif filepath.suffix == '.pkl':
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    logger.info(f"Results saved to {filepath}")


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from file.
    
    Args:
        filepath: Path to results file
        
    Returns:
        Loaded results dictionary
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            return json.load(f)
    elif filepath.suffix == '.pkl':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Calculate class weights for imbalanced datasets.
    
    Args:
        y: Target labels
        
    Returns:
        Dictionary mapping class labels to weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    
    return dict(zip(classes, weights))


def create_time_series_split(
    data: pd.DataFrame, 
    time_col: str, 
    n_splits: int = 5,
    test_size: float = 0.2
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create time series cross-validation splits.
    
    Args:
        data: Input dataframe
        time_col: Name of time column
        n_splits: Number of splits
        test_size: Fraction of data for testing
        
    Returns:
        List of (train_indices, test_indices) tuples
    """
    from sklearn.model_selection import TimeSeriesSplit
    
    # Sort by time
    data_sorted = data.sort_values(time_col)
    
    # Create time series split
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=int(len(data) * test_size))
    
    splits = []
    for train_idx, test_idx in tscv.split(data_sorted):
        splits.append((train_idx, test_idx))
    
    return splits


def detect_data_leakage(
    X: pd.DataFrame, 
    y: pd.Series, 
    time_col: Optional[str] = None
) -> Dict[str, Any]:
    """Detect potential data leakage in features.
    
    Args:
        X: Feature matrix
        y: Target labels
        time_col: Optional time column for temporal analysis
        
    Returns:
        Dictionary with leakage detection results
    """
    results = {
        'high_correlation_features': [],
        'perfect_correlation_features': [],
        'temporal_leakage': False,
        'warnings': []
    }
    
    # Check for perfect correlation with target
    for col in X.columns:
        if X[col].dtype in ['object', 'category']:
            continue
            
        correlation = abs(X[col].corr(y))
        
        if correlation > 0.99:
            results['perfect_correlation_features'].append(col)
            results['warnings'].append(f"Perfect correlation detected: {col}")
        elif correlation > 0.9:
            results['high_correlation_features'].append(col)
            results['warnings'].append(f"High correlation detected: {col}")
    
    # Check for temporal leakage
    if time_col and time_col in X.columns:
        # Check if future information is present
        time_data = X[time_col]
        if time_data.max() > time_data.min() + pd.Timedelta(days=365):
            results['temporal_leakage'] = True
            results['warnings'].append("Potential temporal leakage detected")
    
    return results


def validate_features(X: pd.DataFrame) -> Dict[str, Any]:
    """Validate feature quality and detect issues.
    
    Args:
        X: Feature matrix
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'missing_values': {},
        'infinite_values': {},
        'constant_features': [],
        'duplicate_features': [],
        'warnings': []
    }
    
    # Check for missing values
    missing_counts = X.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            results['missing_values'][col] = count
            results['warnings'].append(f"Missing values in {col}: {count}")
    
    # Check for infinite values
    for col in X.select_dtypes(include=[np.number]).columns:
        inf_count = np.isinf(X[col]).sum()
        if inf_count > 0:
            results['infinite_values'][col] = inf_count
            results['warnings'].append(f"Infinite values in {col}: {inf_count}")
    
    # Check for constant features
    for col in X.columns:
        if X[col].nunique() <= 1:
            results['constant_features'].append(col)
            results['warnings'].append(f"Constant feature: {col}")
    
    # Check for duplicate features
    duplicate_cols = X.columns[X.columns.duplicated()].tolist()
    results['duplicate_features'] = duplicate_cols
    if duplicate_cols:
        results['warnings'].append(f"Duplicate features: {duplicate_cols}")
    
    return results


def create_feature_summary(X: pd.DataFrame) -> pd.DataFrame:
    """Create a summary of feature statistics.
    
    Args:
        X: Feature matrix
        
    Returns:
        DataFrame with feature summary statistics
    """
    summary = X.describe(include='all').T
    
    # Add additional statistics
    summary['missing_count'] = X.isnull().sum()
    summary['missing_pct'] = (X.isnull().sum() / len(X)) * 100
    summary['unique_count'] = X.nunique()
    summary['unique_pct'] = (X.nunique() / len(X)) * 100
    
    return summary


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> Dict[str, str]:
    """Format metrics for display.
    
    Args:
        metrics: Dictionary of metrics
        precision: Number of decimal places
        
    Returns:
        Dictionary with formatted metric strings
    """
    formatted = {}
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if abs(value) < 0.001:
                formatted[key] = f"{value:.2e}"
            else:
                formatted[key] = f"{value:.{precision}f}"
        else:
            formatted[key] = str(value)
    
    return formatted
