"""
Model training module for query-tafsir ranking.

This module provides utilities to train ranking models using:
- Logistic Regression
- Support Vector Machine (SVM)
- XGBoost
- LightGBM
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib


class RankingModel:
    """
    A class to train and use ranking models for query-tafsir ranking.
    
    Supports multiple model types:
    - logistic_regression: Logistic Regression classifier
    - svm: Support Vector Machine classifier
    - xgboost: XGBoost classifier
    - lightgbm: LightGBM classifier
    """
    
    SUPPORTED_MODELS = ['logistic_regression', 'svm', 'xgboost', 'lightgbm']
    
    def __init__(
        self,
        model_type: str = 'logistic_regression',
        model_params: Optional[Dict[str, Any]] = None,
        use_scaler: bool = True
    ):
        """
        Initialize the ranking model.
        
        Args:
            model_type: Type of model to use
            model_params: Parameters to pass to the model
            use_scaler: Whether to scale features
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported types: {self.SUPPORTED_MODELS}")
        
        self.model_type = model_type
        self.model_params = model_params or {}
        self.use_scaler = use_scaler
        
        self.model = None
        self.scaler = StandardScaler() if use_scaler else None
        self._is_fitted = False
        
    def _create_model(self):
        """Create the underlying model based on model_type."""
        if self.model_type == 'logistic_regression':
            default_params = {'max_iter': 1000, 'random_state': 42}
            default_params.update(self.model_params)
            return LogisticRegression(**default_params)
        
        elif self.model_type == 'svm':
            default_params = {'kernel': 'rbf', 'probability': True, 'random_state': 42}
            default_params.update(self.model_params)
            return SVC(**default_params)
        
        elif self.model_type == 'xgboost':
            try:
                from xgboost import XGBClassifier
                default_params = {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'eval_metric': 'logloss'
                }
                default_params.update(self.model_params)
                return XGBClassifier(**default_params)
            except ImportError:
                raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        
        elif self.model_type == 'lightgbm':
            try:
                from lightgbm import LGBMClassifier
                default_params = {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'verbosity': -1
                }
                default_params.update(self.model_params)
                return LGBMClassifier(**default_params)
            except ImportError:
                raise ImportError("LightGBM not installed. Install with: pip install lightgbm")
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> 'RankingModel':
        """
        Train the ranking model.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
            sample_weight: Optional sample weights
            
        Returns:
            Self
        """
        self.model = self._create_model()
        
        X_train = X.copy()
        if self.use_scaler:
            X_train = self.scaler.fit_transform(X_train)
        
        if sample_weight is not None:
            self.model.fit(X_train, y, sample_weight=sample_weight)
        else:
            self.model.fit(X_train, y)
        
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict relevance labels.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predicted labels
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_pred = X.copy()
        if self.use_scaler:
            X_pred = self.scaler.transform(X_pred)
        
        return self.model.predict(X_pred)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict relevance probabilities.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predicted probabilities
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_pred = X.copy()
        if self.use_scaler:
            X_pred = self.scaler.transform(X_pred)
        
        return self.model.predict_proba(X_pred)
    
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Predict ranking scores for documents.
        
        Higher scores indicate more relevant documents.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Ranking scores
        """
        proba = self.predict_proba(X)
        # Return probability of positive class (relevance=1)
        return proba[:, 1] if proba.ndim > 1 else proba
    
    def rank_documents(
        self, 
        X: np.ndarray, 
        doc_ids: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Rank documents by predicted relevance.
        
        Args:
            X: Feature matrix for query-document pairs
            doc_ids: Optional document IDs
            
        Returns:
            List of (doc_id, score) tuples sorted by score descending
        """
        scores = self.predict_scores(X)
        
        if doc_ids is None:
            doc_ids = list(range(len(scores)))
        
        ranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked
    
    def save(self, filepath: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        state = {
            'model_type': self.model_type,
            'model_params': self.model_params,
            'use_scaler': self.use_scaler,
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self._is_fitted
        }
        joblib.dump(state, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'RankingModel':
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded RankingModel instance
        """
        state = joblib.load(filepath)
        instance = cls(
            model_type=state['model_type'],
            model_params=state['model_params'],
            use_scaler=state['use_scaler']
        )
        instance.model = state['model']
        instance.scaler = state['scaler']
        instance._is_fitted = state['is_fitted']
        return instance


def train_multiple_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_types: Optional[List[str]] = None
) -> Dict[str, RankingModel]:
    """
    Train multiple ranking models for comparison.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_types: List of model types to train
        
    Returns:
        Dictionary mapping model type to trained model
    """
    if model_types is None:
        model_types = ['logistic_regression', 'svm', 'xgboost']
    
    models = {}
    for model_type in model_types:
        print(f"Training {model_type}...")
        model = RankingModel(model_type=model_type)
        model.fit(X_train, y_train)
        models[model_type] = model
    
    return models
