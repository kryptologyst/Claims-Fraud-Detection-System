"""Feature engineering utilities for fraud detection."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder, TargetEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature engineering for fraud detection."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize feature engineer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.feature_selector = None
        self.selected_features = None
        
    def prepare_features(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'fraudulent_claim',
        fit: bool = True
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for modeling.
        
        Args:
            df: Input dataframe
            target_col: Target column name
            fit: Whether to fit transformers
            
        Returns:
            Tuple of (processed dataframe, feature names)
        """
        logger.info("Starting feature preparation...")
        
        # Separate target from features
        if target_col in df.columns:
            y = df[target_col]
            X = df.drop(columns=[target_col])
        else:
            X = df
            y = None
            
        # Handle categorical features
        X_processed = self._handle_categorical_features(X, y, fit)
        
        # Handle numerical features
        X_processed = self._handle_numerical_features(X_processed, fit)
        
        # Feature selection
        if fit and y is not None:
            X_processed = self._select_features(X_processed, y)
        elif not fit and self.selected_features is not None:
            X_processed = X_processed[self.selected_features]
            
        logger.info(f"Feature preparation complete. Shape: {X_processed.shape}")
        
        return X_processed, list(X_processed.columns)
        
    def _handle_categorical_features(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series], 
        fit: bool
    ) -> pd.DataFrame:
        """Handle categorical feature encoding."""
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not categorical_cols:
            return X
            
        X_processed = X.copy()
        
        for col in categorical_cols:
            if fit:
                if self.config.get('categorical_encoding') == 'target' and y is not None:
                    encoder = TargetEncoder()
                    X_processed[col] = encoder.fit_transform(X[col], y)
                    self.encoders[col] = encoder
                else:
                    encoder = LabelEncoder()
                    X_processed[col] = encoder.fit_transform(X[col].astype(str))
                    self.encoders[col] = encoder
            else:
                if col in self.encoders:
                    encoder = self.encoders[col]
                    if hasattr(encoder, 'transform'):
                        X_processed[col] = encoder.transform(X[col].astype(str))
                    else:
                        # Handle unseen categories
                        X_processed[col] = X[col].astype(str).map(
                            lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                        )
                        
        return X_processed
        
    def _handle_numerical_features(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Handle numerical feature scaling."""
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numerical_cols:
            return X
            
        X_processed = X.copy()
        
        if self.config.get('scaling') == 'standard':
            if fit:
                scaler = StandardScaler()
                X_processed[numerical_cols] = scaler.fit_transform(X[numerical_cols])
                self.scalers['numerical'] = scaler
            else:
                if 'numerical' in self.scalers:
                    scaler = self.scalers['numerical']
                    X_processed[numerical_cols] = scaler.transform(X[numerical_cols])
                    
        return X_processed
        
    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select best features using statistical tests."""
        if not self.config.get('feature_selection', False):
            return X
            
        n_features = self.config.get('n_features_select', 10)
        n_features = min(n_features, X.shape[1])
        
        selector = SelectKBest(score_func=f_classif, k=n_features)
        X_selected = selector.fit_transform(X, y)
        
        self.feature_selector = selector
        self.selected_features = X.columns[selector.get_support()].tolist()
        
        logger.info(f"Selected {len(self.selected_features)} features")
        
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
        
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores."""
        if self.feature_selector is None:
            return None
            
        scores = self.feature_selector.scores_
        feature_names = self.selected_features
        
        return dict(zip(feature_names, scores))
