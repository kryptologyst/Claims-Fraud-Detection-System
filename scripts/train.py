"""Main training script for fraud detection system."""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any
import joblib

from src.data.synthetic_data import SyntheticClaimsData, ClaimsDataConfig
from src.features.feature_engineering import FeatureEngineer
from src.models.xgboost_model import XGBoostFraudDetector
from src.models.lightgbm_model import LightGBMFraudDetector
from src.models.isolation_forest_model import IsolationForestFraudDetector
from src.evaluation.fraud_metrics import FraudDetectionEvaluator
from src.utils.config import ConfigManager
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


class FraudDetectionPipeline:
    """Main pipeline for fraud detection system."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the fraud detection pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        # Initialize components
        self.data_generator = None
        self.feature_engineer = None
        self.models = {}
        self.evaluator = None
        
        # Data
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        log_file = f"logs/fraud_detection_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        setup_logging(level=log_level, log_file=log_file)
        
    def generate_data(self):
        """Generate synthetic claims data."""
        logger.info("Generating synthetic claims data...")
        
        data_config = ClaimsDataConfig(
            n_samples=self.config['data']['n_samples'],
            fraud_rate=self.config['data']['fraud_rate'],
            random_seed=self.config['data']['random_seed']
        )
        
        self.data_generator = SyntheticClaimsData(data_config)
        df = self.data_generator.generate_data()
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        X = df.drop('fraudulent_claim', axis=1)
        y = df['fraudulent_claim']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_seed'],
            stratify=y
        )
        
        logger.info(f"Data split completed. Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        
    def prepare_features(self):
        """Prepare features for modeling."""
        logger.info("Preparing features...")
        
        self.feature_engineer = FeatureEngineer(self.config['features'])
        
        # Prepare training features
        self.X_train_processed, self.feature_names = self.feature_engineer.prepare_features(
            self.X_train, target_col='fraudulent_claim', fit=True
        )
        
        # Prepare test features
        self.X_test_processed, _ = self.feature_engineer.prepare_features(
            self.X_test, target_col='fraudulent_claim', fit=False
        )
        
        logger.info(f"Feature preparation completed. Features: {len(self.feature_names)}")
        
    def train_models(self):
        """Train multiple fraud detection models."""
        logger.info("Training fraud detection models...")
        
        # XGBoost model
        xgb_config = {
            'params': self.config['model']['params'],
            'training': self.config['model'].get('training', {})
        }
        self.models['xgboost'] = XGBoostFraudDetector(xgb_config)
        self.models['xgboost'].fit(self.X_train_processed, self.y_train)
        
        # LightGBM model
        lgb_config = {
            'params': {
                **self.config['model']['params'],
                'objective': 'binary',
                'metric': 'auc',
                'verbose': -1
            },
            'training': self.config['model'].get('training', {})
        }
        self.models['lightgbm'] = LightGBMFraudDetector(lgb_config)
        self.models['lightgbm'].fit(self.X_train_processed, self.y_train)
        
        # Isolation Forest model
        if_config = {
            'contamination': self.config['data']['fraud_rate'],
            'random_state': self.config['data']['random_seed'],
            'n_estimators': 100
        }
        self.models['isolation_forest'] = IsolationForestFraudDetector(if_config)
        self.models['isolation_forest'].fit(self.X_train_processed, self.y_train)
        
        logger.info("Model training completed")
        
    def evaluate_models(self):
        """Evaluate all trained models."""
        logger.info("Evaluating models...")
        
        self.evaluator = FraudDetectionEvaluator(self.config['evaluation'])
        
        results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name}...")
            
            y_pred = model.predict(self.X_test_processed)
            y_proba = model.predict_proba(self.X_test_processed)
            
            metrics = self.evaluator.evaluate(self.y_test, y_pred, y_proba)
            results[model_name] = metrics
            
            logger.info(f"{model_name} - AUC: {metrics.get('auc', 'N/A'):.4f}")
            
        return results
        
    def save_models(self):
        """Save trained models."""
        logger.info("Saving models...")
        
        models_dir = Path(self.config['paths']['models_dir'])
        models_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = models_dir / f"{model_name}_model.pkl"
            model.save_model(str(model_path))
            
        # Save feature engineer
        feature_engineer_path = models_dir / "feature_engineer.pkl"
        joblib.dump(self.feature_engineer, feature_engineer_path)
        
        logger.info("Models saved successfully")
        
    def run_pipeline(self):
        """Run the complete fraud detection pipeline."""
        logger.info("Starting fraud detection pipeline...")
        
        # Setup
        self.setup_logging()
        
        # Pipeline steps
        self.generate_data()
        self.prepare_features()
        self.train_models()
        results = self.evaluate_models()
        self.save_models()
        
        # Print results
        logger.info("Pipeline completed successfully!")
        logger.info("Model Performance Summary:")
        for model_name, metrics in results.items():
            logger.info(f"{model_name}: AUC={metrics.get('auc', 'N/A'):.4f}, "
                       f"Precision={metrics.get('precision', 'N/A'):.4f}, "
                       f"Recall={metrics.get('recall', 'N/A'):.4f}")
        
        return results


def main():
    """Main function to run the fraud detection pipeline."""
    pipeline = FraudDetectionPipeline()
    results = pipeline.run_pipeline()
    
    # Print summary
    print("\n" + "="*50)
    print("FRAUD DETECTION SYSTEM - TRAINING COMPLETE")
    print("="*50)
    print("\nModel Performance Summary:")
    print("-" * 30)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  AUC: {metrics.get('auc', 'N/A'):.4f}")
        print(f"  Precision: {metrics.get('precision', 'N/A'):.4f}")
        print(f"  Recall: {metrics.get('recall', 'N/A'):.4f}")
        print(f"  F1-Score: {metrics.get('f1', 'N/A'):.4f}")
        
        # Print precision@K metrics if available
        precision_k_metrics = {k: v for k, v in metrics.items() if k.startswith('precision_at_')}
        if precision_k_metrics:
            print("  Precision@K:")
            for metric_name, value in precision_k_metrics.items():
                k = metric_name.split('_')[-1]
                print(f"    @{k}: {value:.4f}")
    
    print("\n" + "="*50)
    print("Models saved to 'models/' directory")
    print("Run 'streamlit run demo/app.py' to start the interactive demo")
    print("="*50)


if __name__ == "__main__":
    main()
