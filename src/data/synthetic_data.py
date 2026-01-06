"""Synthetic claims data generator for fraud detection."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClaimsDataConfig:
    """Configuration for synthetic claims data generation."""
    n_samples: int = 10000
    fraud_rate: float = 0.15
    random_seed: int = 42
    features: Dict[str, Any] = None
    categorical_features: Dict[str, Any] = None


class SyntheticClaimsData:
    """Generates synthetic insurance claims data for fraud detection."""
    
    def __init__(self, config: ClaimsDataConfig):
        """Initialize synthetic data generator.
        
        Args:
            config: Configuration for data generation
        """
        self.config = config
        np.random.seed(config.random_seed)
        
        # Default feature configurations
        self.default_features = {
            'claim_amount': {
                'distribution': 'lognormal',
                'params': {'mean': 7.0, 'sigma': 1.0},
                'fraud_multiplier': 1.5
            },
            'customer_age': {
                'distribution': 'normal',
                'params': {'mean': 45, 'std': 15},
                'min_value': 18,
                'max_value': 80
            },
            'num_previous_claims': {
                'distribution': 'poisson',
                'params': {'lambda': 1.5},
                'max_value': 10
            },
            'days_since_last_claim': {
                'distribution': 'exponential',
                'params': {'scale': 365},
                'max_value': 2000
            },
            'policy_duration': {
                'distribution': 'normal',
                'params': {'mean': 5, 'std': 3},
                'min_value': 0.1,
                'max_value': 20
            }
        }
        
        # Default categorical features
        self.default_categorical = {
            'claim_type': {
                'categories': ['Accident', 'Fire', 'Theft', 'Health', 'Property'],
                'fraud_weights': [1.0, 1.2, 1.5, 0.8, 1.1]
            },
            'location': {
                'categories': ['Urban', 'Suburban', 'Rural'],
                'fraud_weights': [1.3, 1.0, 0.7]
            },
            'customer_segment': {
                'categories': ['Premium', 'Standard', 'Basic'],
                'fraud_weights': [0.5, 1.0, 1.8]
            },
            'claim_time': {
                'categories': ['Business_Hours', 'Evening', 'Night', 'Weekend'],
                'fraud_weights': [1.0, 1.1, 1.3, 1.2]
            }
        }
        
    def generate_data(self) -> pd.DataFrame:
        """Generate synthetic claims data.
        
        Returns:
            DataFrame with synthetic claims data
        """
        logger.info(f"Generating {self.config.n_samples} synthetic claims...")
        
        # Determine fraud labels first
        fraud_labels = np.random.binomial(1, self.config.fraud_rate, self.config.n_samples)
        
        # Generate numerical features
        numerical_data = self._generate_numerical_features(fraud_labels)
        
        # Generate categorical features
        categorical_data = self._generate_categorical_features(fraud_labels)
        
        # Combine all features
        data = {**numerical_data, **categorical_data}
        data['fraudulent_claim'] = fraud_labels
        
        df = pd.DataFrame(data)
        
        # Add derived features
        df = self._add_derived_features(df)
        
        logger.info(f"Generated data shape: {df.shape}")
        logger.info(f"Fraud rate: {df['fraudulent_claim'].mean():.3f}")
        
        return df
        
    def _generate_numerical_features(self, fraud_labels: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate numerical features."""
        features = self.config.features or self.default_features
        data = {}
        
        for feature_name, config in features.items():
            if feature_name in ['claim_amount', 'customer_age', 'num_previous_claims', 
                               'days_since_last_claim', 'policy_duration']:
                data[feature_name] = self._generate_numerical_feature(
                    feature_name, config, fraud_labels
                )
                
        return data
        
    def _generate_numerical_feature(
        self, 
        feature_name: str, 
        config: Dict[str, Any], 
        fraud_labels: np.ndarray
    ) -> np.ndarray:
        """Generate a single numerical feature."""
        distribution = config['distribution']
        params = config['params']
        
        if distribution == 'lognormal':
            values = np.random.lognormal(params['mean'], params['sigma'], self.config.n_samples)
        elif distribution == 'normal':
            values = np.random.normal(params['mean'], params['std'], self.config.n_samples)
        elif distribution == 'poisson':
            values = np.random.poisson(params['lambda'], self.config.n_samples)
        elif distribution == 'exponential':
            values = np.random.exponential(params['scale'], self.config.n_samples)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
            
        # Apply bounds if specified
        if 'min_value' in config:
            values = np.maximum(values, config['min_value'])
        if 'max_value' in config:
            values = np.minimum(values, config['max_value'])
            
        # Apply fraud multiplier
        if 'fraud_multiplier' in config:
            fraud_mask = fraud_labels == 1
            values[fraud_mask] *= config['fraud_multiplier']
            
        return values
        
    def _generate_categorical_features(self, fraud_labels: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate categorical features."""
        categorical_config = self.config.categorical_features or self.default_categorical
        data = {}
        
        for feature_name, config in categorical_config.items():
            data[feature_name] = self._generate_categorical_feature(
                feature_name, config, fraud_labels
            )
            
        return data
        
    def _generate_categorical_feature(
        self, 
        feature_name: str, 
        config: Dict[str, Any], 
        fraud_labels: np.ndarray
    ) -> np.ndarray:
        """Generate a single categorical feature."""
        categories = config['categories']
        fraud_weights = config['fraud_weights']
        
        # Create probability weights based on fraud status
        n_categories = len(categories)
        base_probs = np.ones(n_categories) / n_categories
        
        # Adjust probabilities for fraud cases
        fraud_probs = base_probs * np.array(fraud_weights)
        fraud_probs = fraud_probs / fraud_probs.sum()
        
        # Generate categorical values
        values = np.zeros(self.config.n_samples, dtype=object)
        
        # Non-fraud cases
        non_fraud_mask = fraud_labels == 0
        values[non_fraud_mask] = np.random.choice(
            categories, 
            size=non_fraud_mask.sum(), 
            p=base_probs
        )
        
        # Fraud cases
        fraud_mask = fraud_labels == 1
        values[fraud_mask] = np.random.choice(
            categories, 
            size=fraud_mask.sum(), 
            p=fraud_probs
        )
        
        return values
        
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to the dataset."""
        # Claim amount per previous claim
        df['claim_amount_per_previous'] = df['claim_amount'] / (df['num_previous_claims'] + 1)
        
        # Age group
        df['age_group'] = pd.cut(
            df['customer_age'], 
            bins=[0, 25, 35, 50, 65, 100], 
            labels=['Young', 'Adult', 'Middle', 'Senior', 'Elderly']
        )
        
        # High value claim flag
        df['high_value_claim'] = (df['claim_amount'] > df['claim_amount'].quantile(0.9)).astype(int)
        
        # Frequent claimant flag
        df['frequent_claimant'] = (df['num_previous_claims'] >= 3).astype(int)
        
        # Recent claim flag
        df['recent_claim'] = (df['days_since_last_claim'] < 30).astype(int)
        
        return df
