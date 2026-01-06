"""Claims Fraud Detection System - Research & Educational Use Only."""

__version__ = "1.0.0"
__author__ = "AI Research Team"
__description__ = "Advanced fraud detection system for insurance claims"

# Import main components
from .data.synthetic_data import SyntheticClaimsData, ClaimsDataConfig
from .features.feature_engineering import FeatureEngineer
from .models.xgboost_model import XGBoostFraudDetector
from .models.lightgbm_model import LightGBMFraudDetector
from .models.isolation_forest_model import IsolationForestFraudDetector
from .evaluation.fraud_metrics import FraudDetectionEvaluator
from .utils.config import ConfigManager
from .utils.logging import setup_logging, get_logger
from .utils.utils import set_random_seeds, calculate_class_weights

__all__ = [
    "SyntheticClaimsData",
    "ClaimsDataConfig", 
    "FeatureEngineer",
    "XGBoostFraudDetector",
    "LightGBMFraudDetector",
    "IsolationForestFraudDetector",
    "FraudDetectionEvaluator",
    "ConfigManager",
    "setup_logging",
    "get_logger",
    "set_random_seeds",
    "calculate_class_weights"
]
