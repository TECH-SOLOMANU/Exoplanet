"""
Enhanced Ensemble Models for Exoplanet Detection
Based on research achieving 83%+ accuracy with ensemble methods.
Implements Stacking, Random Forest, Gradient Boosting, and XGBoost ensembles.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import asyncio
import joblib
import logging
from pathlib import Path
import pickle

# ML imports
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    AdaBoostClassifier, ExtraTreesClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, GridSearchCV, 
    RandomizedSearchCV, validation_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb

# Custom imports
from app.ml.advanced_feature_engineering import AdvancedFeatureEngineering

logger = logging.getLogger(__name__)

class EnsembleExoplanetDetector:
    """
    Advanced ensemble model for exoplanet detection based on research papers.
    Implements multiple ensemble strategies for maximum accuracy.
    """
    
    def __init__(self, model_path: str = "models/ensemble_models"):
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        self.feature_engineer = AdvancedFeatureEngineering()
        self.models = {}
        self.ensemble_weights = {}
        self.is_trained = False
        
        # Performance tracking
        self.training_scores = {}
        self.validation_scores = {}
        
    async def train_ensemble_models(self, X: pd.DataFrame, y: pd.Series, 
                                  cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train comprehensive ensemble models using advanced techniques.
        
        Args:
            X: Features DataFrame
            y: Target series
            cv_folds: Number of cross-validation folds
            
        Returns:
            Training results and metrics
        """
        logger.info("Starting ensemble model training with advanced techniques")
        
        # Feature engineering
        X_engineered = self.feature_engineer.extract_all_features(X)
        X_processed = self.feature_engineer.preprocess_features(
            X_engineered, target=y, fit_transformers=True
        )
        
        # Initialize cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Train individual models
        individual_results = await self._train_individual_models(X_processed, y, cv)
        
        # Train ensemble models
        ensemble_results = await self._train_ensemble_models(X_processed, y, cv)
        
        # Train stacking ensemble (best performer from research)
        stacking_results = await self._train_stacking_ensemble(X_processed, y, cv)
        
        # Calculate ensemble weights based on performance
        self._calculate_ensemble_weights()
        
        # Save models
        await self._save_models()
        
        self.is_trained = True
        
        # Compile results
        results = {
            'individual_models': individual_results,
            'ensemble_models': ensemble_results,
            'stacking_results': stacking_results,
            'best_model': self._get_best_model(),
            'ensemble_weights': self.ensemble_weights,
            'feature_count': X_processed.shape[1]
        }
        
        logger.info(f"Ensemble training completed. Best model: {results['best_model']['name']} "
                   f"with {results['best_model']['accuracy']:.4f} accuracy")
        
        return results
    
    async def _train_individual_models(self, X: pd.DataFrame, y: pd.Series, 
                                     cv: StratifiedKFold) -> Dict[str, Any]:
        """Train individual base models with hyperparameter optimization."""
        logger.info("Training individual base models")
        
        # Define base models with hyperparameter grids
        model_configs = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200, 500, 1000],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier(random_state=42, verbosity=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            },
            'extra_trees': {
                'model': ExtraTreesClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            }
        }
        
        results = {}
        
        for model_name, config in model_configs.items():
            logger.info(f"Training {model_name}")
            
            # Randomized search for efficiency with large parameter spaces
            search = RandomizedSearchCV(
                config['model'],
                config['params'],
                n_iter=20,  # Reasonable number for efficiency
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
            
            search.fit(X, y)
            
            # Get best model
            best_model = search.best_estimator_
            
            # Calculate comprehensive metrics
            metrics = self._calculate_cross_val_metrics(best_model, X, y, cv)
            
            # Store model and results
            self.models[model_name] = best_model
            results[model_name] = {
                'model': best_model,
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'metrics': metrics
            }
            
            logger.info(f"{model_name} - Best AUC: {search.best_score_:.4f}")
        
        return results
    
    async def _train_ensemble_models(self, X: pd.DataFrame, y: pd.Series, 
                                   cv: StratifiedKFold) -> Dict[str, Any]:
        """Train ensemble models using different strategies."""
        logger.info("Training ensemble models")
        
        results = {}
        
        # Voting ensemble with different combinations
        ensemble_configs = {
            'voting_soft': {
                'estimators': [
                    ('rf', self.models['random_forest']),
                    ('gb', self.models['gradient_boosting']),
                    ('xgb', self.models['xgboost'])
                ],
                'voting': 'soft'
            },
            'voting_hard': {
                'estimators': [
                    ('rf', self.models['random_forest']),
                    ('gb', self.models['gradient_boosting']),
                    ('xgb', self.models['xgboost'])
                ],
                'voting': 'hard'
            }
        }
        
        from sklearn.ensemble import VotingClassifier
        
        for ensemble_name, config in ensemble_configs.items():
            logger.info(f"Training {ensemble_name}")
            
            ensemble = VotingClassifier(**config)
            metrics = self._calculate_cross_val_metrics(ensemble, X, y, cv)
            
            # Fit the ensemble
            ensemble.fit(X, y)
            
            self.models[ensemble_name] = ensemble
            results[ensemble_name] = {
                'model': ensemble,
                'metrics': metrics
            }
        
        return results
    
    async def _train_stacking_ensemble(self, X: pd.DataFrame, y: pd.Series, 
                                     cv: StratifiedKFold) -> Dict[str, Any]:
        """Train stacking ensemble (best performer from research)."""
        logger.info("Training stacking ensemble models")
        
        results = {}
        
        # Stacking configurations based on research papers
        stacking_configs = {
            'stacking_lr': {
                'estimators': [
                    ('rf', self.models['random_forest']),
                    ('gb', self.models['gradient_boosting']),
                    ('xgb', self.models['xgboost']),
                    ('lgb', self.models['lightgbm'])
                ],
                'final_estimator': LogisticRegression(random_state=42, max_iter=1000),
                'cv': 3  # Inner CV for stacking
            },
            'stacking_xgb': {
                'estimators': [
                    ('rf', self.models['random_forest']),
                    ('gb', self.models['gradient_boosting']),
                    ('lgb', self.models['lightgbm']),
                    ('et', self.models['extra_trees'])
                ],
                'final_estimator': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'cv': 3
            },
            'stacking_neural': {
                'estimators': [
                    ('rf', self.models['random_forest']),
                    ('xgb', self.models['xgboost']),
                    ('lgb', self.models['lightgbm'])
                ],
                'final_estimator': MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    random_state=42,
                    max_iter=500
                ),
                'cv': 3
            }
        }
        
        for stack_name, config in stacking_configs.items():
            logger.info(f"Training {stack_name}")
            
            stacking_clf = StackingClassifier(**config)
            metrics = self._calculate_cross_val_metrics(stacking_clf, X, y, cv)
            
            # Fit the stacking ensemble
            stacking_clf.fit(X, y)
            
            self.models[stack_name] = stacking_clf
            results[stack_name] = {
                'model': stacking_clf,
                'metrics': metrics
            }
        
        return results
    
    def _calculate_cross_val_metrics(self, model, X: pd.DataFrame, y: pd.Series, 
                                   cv: StratifiedKFold) -> Dict[str, float]:
        """Calculate comprehensive cross-validation metrics."""
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        metrics = {}
        
        for metric in scoring_metrics:
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
            metrics[f'{metric}_mean'] = scores.mean()
            metrics[f'{metric}_std'] = scores.std()
        
        return metrics
    
    def _calculate_ensemble_weights(self):
        """Calculate ensemble weights based on performance."""
        weights = {}
        
        for model_name, model in self.models.items():
            if model_name in self.validation_scores:
                # Weight based on AUC score
                auc_score = self.validation_scores[model_name].get('roc_auc_mean', 0.5)
                weights[model_name] = max(0, auc_score - 0.5)  # Only positive contributions
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.ensemble_weights = {k: v/total_weight for k, v in weights.items()}
        else:
            # Equal weights if no valid scores
            self.ensemble_weights = {k: 1/len(weights) for k in weights.keys()}
    
    def _get_best_model(self) -> Dict[str, Any]:
        """Get the best performing model."""
        best_model = None
        best_score = 0
        best_name = ""
        
        for model_name, model in self.models.items():
            # Check if we have validation scores
            if hasattr(self, 'validation_scores') and model_name in self.validation_scores:
                score = self.validation_scores[model_name].get('roc_auc_mean', 0)
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_name = model_name
        
        return {
            'name': best_name,
            'model': best_model,
            'accuracy': best_score
        }
    
    async def predict(self, X: pd.DataFrame, use_ensemble: bool = True, 
                     return_probabilities: bool = False) -> np.ndarray:
        """
        Make predictions using the trained ensemble.
        
        Args:
            X: Features DataFrame
            use_ensemble: Whether to use ensemble prediction
            return_probabilities: Whether to return probabilities
            
        Returns:
            Predictions or probabilities
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Feature engineering
        X_engineered = self.feature_engineer.extract_all_features(X)
        X_processed = self.feature_engineer.preprocess_features(
            X_engineered, fit_transformers=False
        )
        
        if use_ensemble and len(self.models) > 1:
            # Ensemble prediction using weighted averaging
            predictions = {}
            probabilities = {}
            
            for model_name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_processed)[:, 1]
                    probabilities[model_name] = proba
                    predictions[model_name] = (proba > 0.5).astype(int)
                else:
                    pred = model.predict(X_processed)
                    predictions[model_name] = pred
                    probabilities[model_name] = pred.astype(float)
            
            # Weighted ensemble
            if return_probabilities:
                ensemble_proba = np.zeros(len(X_processed))
                for model_name, proba in probabilities.items():
                    weight = self.ensemble_weights.get(model_name, 1/len(probabilities))
                    ensemble_proba += weight * proba
                return ensemble_proba
            else:
                ensemble_pred = np.zeros(len(X_processed))
                for model_name, pred in predictions.items():
                    weight = self.ensemble_weights.get(model_name, 1/len(predictions))
                    ensemble_pred += weight * pred
                return (ensemble_pred > 0.5).astype(int)
        
        else:
            # Use best single model
            best_model_info = self._get_best_model()
            best_model = best_model_info['model']
            
            if return_probabilities and hasattr(best_model, 'predict_proba'):
                return best_model.predict_proba(X_processed)[:, 1]
            else:
                return best_model.predict(X_processed)
    
    async def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate the trained ensemble on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Comprehensive evaluation metrics
        """
        # Get predictions
        y_pred = await self.predict(X_test, use_ensemble=True, return_probabilities=False)
        y_proba = await self.predict(X_test, use_ensemble=True, return_probabilities=True)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Individual model performance
        individual_metrics = {}
        for model_name, model in self.models.items():
            X_engineered = self.feature_engineer.extract_all_features(X_test)
            X_processed = self.feature_engineer.preprocess_features(
                X_engineered, fit_transformers=False
            )
            
            pred = model.predict(X_processed)
            proba = model.predict_proba(X_processed)[:, 1] if hasattr(model, 'predict_proba') else pred
            
            individual_metrics[model_name] = {
                'accuracy': accuracy_score(y_test, pred),
                'precision': precision_score(y_test, pred, average='weighted'),
                'recall': recall_score(y_test, pred, average='weighted'),
                'f1_score': f1_score(y_test, pred, average='weighted'),
                'roc_auc': roc_auc_score(y_test, proba)
            }
        
        return {
            'ensemble_metrics': metrics,
            'individual_metrics': individual_metrics,
            'best_individual_model': max(individual_metrics.keys(), 
                                       key=lambda k: individual_metrics[k]['roc_auc'])
        }
    
    async def _save_models(self):
        """Save trained models to disk."""
        try:
            # Save individual models
            for model_name, model in self.models.items():
                model_file = self.model_path / f"{model_name}.pkl"
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
            
            # Save feature engineer
            fe_file = self.model_path / "feature_engineer.pkl"
            with open(fe_file, 'wb') as f:
                pickle.dump(self.feature_engineer, f)
            
            # Save ensemble weights
            weights_file = self.model_path / "ensemble_weights.pkl"
            with open(weights_file, 'wb') as f:
                pickle.dump(self.ensemble_weights, f)
            
            logger.info(f"Models saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    async def load_models(self):
        """Load trained models from disk."""
        try:
            # Load feature engineer
            fe_file = self.model_path / "feature_engineer.pkl"
            if fe_file.exists():
                with open(fe_file, 'rb') as f:
                    self.feature_engineer = pickle.load(f)
            
            # Load ensemble weights
            weights_file = self.model_path / "ensemble_weights.pkl"
            if weights_file.exists():
                with open(weights_file, 'rb') as f:
                    self.ensemble_weights = pickle.load(f)
            
            # Load individual models
            self.models = {}
            for model_file in self.model_path.glob("*.pkl"):
                if model_file.name not in ["feature_engineer.pkl", "ensemble_weights.pkl"]:
                    model_name = model_file.stem
                    with open(model_file, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
            
            self.is_trained = len(self.models) > 0
            logger.info(f"Loaded {len(self.models)} models from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.is_trained = False
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get feature importance from ensemble models.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature importances per model
        """
        importance_dict = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                feature_names = self.feature_engineer.feature_selector.get_feature_names_out() if \
                    hasattr(self.feature_engineer.feature_selector, 'get_feature_names_out') else \
                    [f'feature_{i}' for i in range(len(importances))]
                
                feature_imp = list(zip(feature_names, importances))
                feature_imp.sort(key=lambda x: x[1], reverse=True)
                importance_dict[model_name] = feature_imp[:top_n]
                
            elif hasattr(model, 'coef_'):
                # Linear models
                coefficients = np.abs(model.coef_[0])
                feature_names = self.feature_engineer.feature_selector.get_feature_names_out() if \
                    hasattr(self.feature_engineer.feature_selector, 'get_feature_names_out') else \
                    [f'feature_{i}' for i in range(len(coefficients))]
                
                feature_imp = list(zip(feature_names, coefficients))
                feature_imp.sort(key=lambda x: x[1], reverse=True)
                importance_dict[model_name] = feature_imp[:top_n]
        
        return importance_dict