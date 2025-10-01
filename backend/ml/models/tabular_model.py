import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import joblib
import shap
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class TabularExoplanetModel:
    """XGBoost model for exoplanet classification using tabular features"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.explainer = None
        self.model_path = model_path
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training/prediction"""
        feature_columns = [
            'pl_orbper',    # Orbital period
            'pl_rade',      # Planet radius
            'pl_bmasse',    # Planet mass
            'pl_eqt',       # Equilibrium temperature
            'pl_insol',     # Insolation flux
            'st_teff',      # Stellar effective temperature
            'st_rad',       # Stellar radius
            'st_mass',      # Stellar mass
            'st_logg',      # Stellar surface gravity
            'st_met',       # Stellar metallicity
            'sy_dist'       # Distance to system
        ]
        
        # Select available features
        available_features = [col for col in feature_columns if col in df.columns]
        features_df = df[available_features].copy()
        
        # Handle missing values
        features_df = features_df.fillna(features_df.median())
        
        # Add derived features
        if 'pl_orbper' in features_df.columns and 'st_mass' in features_df.columns:
            # Semi-major axis using Kepler's third law (approximate)
            features_df['pl_smax'] = np.power(
                features_df['pl_orbper']**2 * features_df['st_mass'], 1/3
            )
        
        if 'pl_rade' in features_df.columns and 'pl_bmasse' in features_df.columns:
            # Planet density
            features_df['pl_density'] = features_df['pl_bmasse'] / (features_df['pl_rade']**3)
        
        if 'pl_eqt' in features_df.columns and 'st_teff' in features_df.columns:
            # Temperature ratio
            features_df['temp_ratio'] = features_df['pl_eqt'] / features_df['st_teff']
        
        # Remove infinite and NaN values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(features_df.median())
        
        self.feature_names = list(features_df.columns)
        return features_df
    
    def train(self, df: pd.DataFrame, target_column: str = 'pl_status') -> Dict:
        """Train the XGBoost model"""
        try:
            logger.info("Starting tabular model training...")
            
            # Prepare features and target
            X = self.prepare_features(df)
            y = df[target_column].fillna('UNKNOWN')
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train XGBoost model
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Create SHAP explainer
            self.explainer = shap.TreeExplainer(self.model)
            
            # Save model
            if self.model_path:
                self.save_model()
            
            logger.info(f"Tabular model training completed. Accuracy: {accuracy:.4f}")
            
            return {
                "accuracy": accuracy,
                "feature_importance": dict(zip(
                    self.feature_names, 
                    self.model.feature_importances_
                )),
                "classes": list(self.label_encoder.classes_),
                "n_features": len(self.feature_names)
            }
            
        except Exception as e:
            logger.error(f"Failed to train tabular model: {e}")
            raise
    
    def predict(self, data: np.ndarray) -> Dict:
        """Make predictions with explanations"""
        try:
            if self.model is None:
                raise ValueError("Model not trained or loaded")
            
            # Scale features
            data_scaled = self.scaler.transform(data)
            
            # Predictions
            predictions = self.model.predict(data_scaled)
            probabilities = self.model.predict_proba(data_scaled)
            
            # SHAP explanations
            shap_values = self.explainer.shap_values(data_scaled)
            
            results = []
            for i in range(len(data)):
                result = {
                    "prediction": self.label_encoder.inverse_transform([predictions[i]])[0],
                    "confidence": float(np.max(probabilities[i])),
                    "probabilities": {
                        class_name: float(prob) 
                        for class_name, prob in zip(
                            self.label_encoder.classes_, 
                            probabilities[i]
                        )
                    },
                    "feature_importance": {
                        feature: float(importance) 
                        for feature, importance in zip(
                            self.feature_names, 
                            shap_values[predictions[i]][i]
                        )
                    }
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise
    
    def save_model(self):
        """Save trained model and components"""
        try:
            model_dir = Path(self.model_path).parent
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model components
            joblib.dump(self.model, f"{self.model_path}_xgb.pkl")
            joblib.dump(self.scaler, f"{self.model_path}_scaler.pkl")
            joblib.dump(self.label_encoder, f"{self.model_path}_encoder.pkl")
            joblib.dump(self.feature_names, f"{self.model_path}_features.pkl")
            
            logger.info(f"Model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self):
        """Load trained model and components"""
        try:
            self.model = joblib.load(f"{self.model_path}_xgb.pkl")
            self.scaler = joblib.load(f"{self.model_path}_scaler.pkl")
            self.label_encoder = joblib.load(f"{self.model_path}_encoder.pkl")
            self.feature_names = joblib.load(f"{self.model_path}_features.pkl")
            
            # Recreate SHAP explainer
            self.explainer = shap.TreeExplainer(self.model)
            
            logger.info(f"Model loaded from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise