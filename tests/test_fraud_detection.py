"""Test suite for fraud detection system."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.synthetic_data import SyntheticClaimsData, ClaimsDataConfig
from src.features.feature_engineering import FeatureEngineer
from src.models.xgboost_model import XGBoostFraudDetector
from src.models.lightgbm_model import LightGBMFraudDetector
from src.models.isolation_forest_model import IsolationForestFraudDetector
from src.evaluation.fraud_metrics import FraudDetectionEvaluator
from src.utils.utils import set_random_seeds, calculate_class_weights


class TestSyntheticData:
    """Test synthetic data generation."""
    
    def test_data_generation(self):
        """Test basic data generation."""
        config = ClaimsDataConfig(n_samples=100, fraud_rate=0.2, random_seed=42)
        generator = SyntheticClaimsData(config)
        df = generator.generate_data()
        
        assert len(df) == 100
        assert 'fraudulent_claim' in df.columns
        assert df['fraudulent_claim'].dtype == int
        assert df['fraudulent_claim'].isin([0, 1]).all()
        
    def test_fraud_rate(self):
        """Test fraud rate is approximately correct."""
        config = ClaimsDataConfig(n_samples=1000, fraud_rate=0.15, random_seed=42)
        generator = SyntheticClaimsData(config)
        df = generator.generate_data()
        
        fraud_rate = df['fraudulent_claim'].mean()
        assert 0.1 <= fraud_rate <= 0.2  # Allow some variance
        
    def test_feature_distributions(self):
        """Test that features have expected distributions."""
        config = ClaimsDataConfig(n_samples=1000, fraud_rate=0.15, random_seed=42)
        generator = SyntheticClaimsData(config)
        df = generator.generate_data()
        
        # Check numerical features
        assert df['claim_amount'].min() > 0
        assert df['customer_age'].min() >= 18
        assert df['customer_age'].max() <= 80
        assert df['num_previous_claims'].min() >= 0
        
        # Check categorical features
        assert df['claim_type'].isin(['Accident', 'Fire', 'Theft', 'Health', 'Property']).all()
        assert df['location'].isin(['Urban', 'Suburban', 'Rural']).all()


class TestFeatureEngineering:
    """Test feature engineering."""
    
    def test_feature_preparation(self):
        """Test feature preparation pipeline."""
        # Create sample data
        data = {
            'claim_amount': [1000, 2000, 3000],
            'customer_age': [25, 35, 45],
            'claim_type': ['Accident', 'Fire', 'Theft'],
            'fraudulent_claim': [0, 1, 0]
        }
        df = pd.DataFrame(data)
        
        config = {
            'categorical_encoding': 'label',
            'scaling': 'standard',
            'feature_selection': False
        }
        
        engineer = FeatureEngineer(config)
        X_processed, feature_names = engineer.prepare_features(df, fit=True)
        
        assert X_processed.shape[0] == 3
        assert len(feature_names) > 0
        assert 'fraudulent_claim' not in feature_names
        
    def test_categorical_encoding(self):
        """Test categorical feature encoding."""
        data = {
            'claim_type': ['Accident', 'Fire', 'Theft', 'Accident'],
            'fraudulent_claim': [0, 1, 0, 1]
        }
        df = pd.DataFrame(data)
        
        config = {'categorical_encoding': 'label'}
        engineer = FeatureEngineer(config)
        X_processed, _ = engineer.prepare_features(df, fit=True)
        
        # Check that categorical features are encoded
        assert X_processed['claim_type'].dtype in [np.int64, np.int32]
        
    def test_feature_scaling(self):
        """Test feature scaling."""
        data = {
            'claim_amount': [1000, 2000, 3000],
            'customer_age': [25, 35, 45],
            'fraudulent_claim': [0, 1, 0]
        }
        df = pd.DataFrame(data)
        
        config = {'scaling': 'standard'}
        engineer = FeatureEngineer(config)
        X_processed, _ = engineer.prepare_features(df, fit=True)
        
        # Check that numerical features are scaled
        assert abs(X_processed['claim_amount'].mean()) < 0.1
        assert abs(X_processed['customer_age'].mean()) < 0.1


class TestModels:
    """Test machine learning models."""
    
    def test_xgboost_model(self):
        """Test XGBoost model training and prediction."""
        # Create sample data
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        y = pd.Series(np.random.randint(0, 2, 100))
        
        config = {
            'params': {
                'n_estimators': 10,
                'max_depth': 3,
                'learning_rate': 0.1,
                'random_state': 42
            }
        }
        
        model = XGBoostFraudDetector(config)
        model.fit(X, y)
        
        # Test prediction
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        assert len(predictions) == 100
        assert len(probabilities) == 100
        assert predictions.dtype == int
        assert probabilities.dtype == float
        assert (probabilities >= 0).all()
        assert (probabilities <= 1).all()
        
    def test_lightgbm_model(self):
        """Test LightGBM model training and prediction."""
        # Create sample data
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        y = pd.Series(np.random.randint(0, 2, 100))
        
        config = {
            'params': {
                'n_estimators': 10,
                'max_depth': 3,
                'learning_rate': 0.1,
                'objective': 'binary',
                'metric': 'auc',
                'verbose': -1,
                'random_state': 42
            }
        }
        
        model = LightGBMFraudDetector(config)
        model.fit(X, y)
        
        # Test prediction
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        assert len(predictions) == 100
        assert len(probabilities) == 100
        assert predictions.dtype == int
        assert probabilities.dtype == float
        
    def test_isolation_forest_model(self):
        """Test Isolation Forest model training and prediction."""
        # Create sample data
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        
        config = {
            'contamination': 0.1,
            'random_state': 42,
            'n_estimators': 10
        }
        
        model = IsolationForestFraudDetector(config)
        model.fit(X)
        
        # Test prediction
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        assert len(predictions) == 100
        assert len(probabilities) == 100
        assert predictions.dtype == int
        assert probabilities.dtype == float


class TestEvaluation:
    """Test evaluation metrics."""
    
    def test_basic_metrics(self):
        """Test basic evaluation metrics."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])
        y_proba = np.array([0.1, 0.9, 0.7, 0.8, 0.2])
        
        config = {
            'metrics': ['auc', 'precision', 'recall', 'f1'],
            'k_values': [2, 3]
        }
        
        evaluator = FraudDetectionEvaluator(config)
        metrics = evaluator.evaluate(y_true, y_pred, y_proba)
        
        assert 'auc' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 0 <= metrics['auc'] <= 1
        
    def test_precision_recall_at_k(self):
        """Test precision@K and recall@K metrics."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        
        config = {
            'metrics': ['precision_at_k', 'recall_at_k'],
            'k_values': [3, 5]
        }
        
        evaluator = FraudDetectionEvaluator(config)
        metrics = evaluator.evaluate(y_true, y_true, y_proba)
        
        assert 'precision_at_3' in metrics
        assert 'recall_at_3' in metrics
        assert 'precision_at_5' in metrics
        assert 'recall_at_5' in metrics


class TestUtils:
    """Test utility functions."""
    
    def test_set_random_seeds(self):
        """Test random seed setting."""
        set_random_seeds(42)
        val1 = np.random.rand()
        
        set_random_seeds(42)
        val2 = np.random.rand()
        
        assert val1 == val2
        
    def test_calculate_class_weights(self):
        """Test class weight calculation."""
        y = np.array([0, 0, 0, 1, 1])
        weights = calculate_class_weights(y)
        
        assert 0 in weights
        assert 1 in weights
        assert weights[1] > weights[0]  # Minority class should have higher weight


if __name__ == "__main__":
    pytest.main([__file__])
