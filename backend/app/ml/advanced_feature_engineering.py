"""
Advanced Feature Engineering for Exoplanet Detection
Based on research from:
- "Exoplanet detection using machine learning" (2022) - TSFRESH approach with 789 features
- "Assessment of Ensemble-Based Machine Learning Algorithms" (2024) - Ensemble methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.signal import find_peaks, periodogram
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import asyncio
import logging

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineering:
    """
    Advanced feature engineering for exoplanet detection based on latest research.
    Implements TSFRESH-inspired features and astronomical domain knowledge.
    """
    
    def __init__(self, use_tsfresh_features: bool = True, use_astronomical_features: bool = True):
        self.use_tsfresh_features = use_tsfresh_features
        self.use_astronomical_features = use_astronomical_features
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.pca = None
        
    def extract_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive features for exoplanet detection.
        
        Args:
            data: DataFrame with exoplanet parameters
            
        Returns:
            DataFrame with engineered features
        """
        features = data.copy()
        
        # Basic statistical features
        features = self._add_statistical_features(features)
        
        # Astronomical domain features
        if self.use_astronomical_features:
            features = self._add_astronomical_features(features)
            
        # Transit-specific features
        features = self._add_transit_features(features)
        
        # Ratio and interaction features
        features = self._add_ratio_features(features)
        
        # Polynomial features for key parameters
        features = self._add_polynomial_features(features)
        
        # Temporal features if period information available
        features = self._add_temporal_features(features)
        
        # Statistical distribution features
        features = self._add_distribution_features(features)
        
        logger.info(f"Generated {features.shape[1]} engineered features")
        return features
    
    def _add_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic statistical features inspired by TSFRESH efficiently."""
        # Prepare features dictionary for efficient creation
        feature_dict = {}
        
        # Copy original data
        for col in data.columns:
            feature_dict[col] = data[col].values
        
        # Get numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in data.columns:
                values = data[col].dropna()
                if len(values) > 0:
                    n_rows = len(data)
                    
                    # Basic statistics - broadcast to all rows
                    feature_dict[f'{col}_mean'] = np.full(n_rows, values.mean())
                    feature_dict[f'{col}_std'] = np.full(n_rows, values.std())
                    feature_dict[f'{col}_skew'] = np.full(n_rows, stats.skew(values))
                    feature_dict[f'{col}_kurtosis'] = np.full(n_rows, stats.kurtosis(values))
                    feature_dict[f'{col}_min'] = np.full(n_rows, values.min())
                    feature_dict[f'{col}_max'] = np.full(n_rows, values.max())
                    feature_dict[f'{col}_range'] = np.full(n_rows, values.max() - values.min())
                    feature_dict[f'{col}_q25'] = np.full(n_rows, values.quantile(0.25))
                    feature_dict[f'{col}_q75'] = np.full(n_rows, values.quantile(0.75))
                    feature_dict[f'{col}_iqr'] = np.full(n_rows, values.quantile(0.75) - values.quantile(0.25))
                    
                    # Robust statistics
                    feature_dict[f'{col}_median'] = np.full(n_rows, values.median())
                    feature_dict[f'{col}_mad'] = np.full(n_rows, np.median(np.abs(values - values.median())))
                    
                    # Count-based features as ratios
                    feature_dict[f'{col}_zero_ratio'] = np.full(n_rows, (values == 0).sum() / len(values))
                    feature_dict[f'{col}_above_mean_ratio'] = np.full(n_rows, (values > values.mean()).sum() / len(values))
                    feature_dict[f'{col}_below_mean_ratio'] = np.full(n_rows, (values < values.mean()).sum() / len(values))
        
        # Create DataFrame efficiently from dictionary
        features = pd.DataFrame(feature_dict)
        return features
    
    def _add_astronomical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add astronomical domain-specific features."""
        features = data.copy()
        
        # Planetary characteristics
        if 'koi_prad' in features.columns:
            # Planet radius categories
            features['planet_size_category'] = pd.cut(
                features['koi_prad'], 
                bins=[0, 1.25, 2.0, 4.0, float('inf')], 
                labels=[0, 1, 2, 3]  # Earth-size, Super-Earth, Neptune-size, Jupiter-size
            )
            
            # Earth-like planet indicator
            features['earth_like'] = ((features['koi_prad'] >= 0.8) & (features['koi_prad'] <= 1.25)).astype(int)
        
        # Stellar characteristics
        if 'koi_srad' in features.columns:
            features['stellar_size_category'] = pd.cut(
                features['koi_srad'],
                bins=[0, 0.8, 1.2, 2.0, float('inf')],
                labels=[0, 1, 2, 3]  # Small, Sun-like, Large, Giant
            )
        
        # Temperature-based features
        if 'koi_teq' in features.columns:
            # Habitable zone indicator (simplified)
            features['potentially_habitable'] = ((features['koi_teq'] >= 200) & (features['koi_teq'] <= 350)).astype(int)
            
            # Temperature categories
            features['temp_category'] = pd.cut(
                features['koi_teq'],
                bins=[0, 200, 350, 1000, float('inf')],
                labels=[0, 1, 2, 3]  # Frozen, Habitable, Hot, Very Hot
            )
        
        # Orbital characteristics
        if 'koi_period' in features.columns:
            # Period categories
            features['period_category'] = pd.cut(
                features['koi_period'],
                bins=[0, 10, 50, 200, float('inf')],
                labels=[0, 1, 2, 3]  # Hot, Warm, Cool, Cold
            )
            
            # Period logarithm (astronomical periods often log-distributed)
            features['log_period'] = np.log10(features['koi_period'] + 1)
        
        return features
    
    def _add_transit_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add transit-specific features from research papers."""
        features = data.copy()
        
        # Transit depth and duration features
        if 'koi_depth' in features.columns:
            features['log_depth'] = np.log10(features['koi_depth'] + 1)
            
            # Transit depth categories (ppm levels)
            features['depth_category'] = pd.cut(
                features['koi_depth'],
                bins=[0, 100, 1000, 10000, float('inf')],
                labels=[0, 1, 2, 3]  # Shallow, Medium, Deep, Very Deep
            )
        
        if 'koi_duration' in features.columns:
            features['log_duration'] = np.log10(features['koi_duration'] + 1)
        
        # Signal-to-noise related features
        if 'koi_dor' in features.columns:
            # Distance over radius ratio (orbital characteristics)
            features['log_dor'] = np.log10(features['koi_dor'] + 1)
        
        # Transit timing features
        if all(col in features.columns for col in ['koi_period', 'koi_duration']):
            # Transit duty cycle (fraction of orbit in transit)
            features['duty_cycle'] = features['koi_duration'] / (features['koi_period'] * 24)  # Convert period to hours
            
        return features
    
    def _add_ratio_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add ratio and interaction features."""
        features = data.copy()
        
        # Planet-to-star ratios
        if all(col in features.columns for col in ['koi_prad', 'koi_srad']):
            features['planet_star_radius_ratio'] = features['koi_prad'] / (features['koi_srad'] + 1e-8)
            features['log_radius_ratio'] = np.log10(features['planet_star_radius_ratio'] + 1e-8)
        
        # Density estimation (if mass available)
        if 'koi_prad' in features.columns:
            # Assuming Earth-like density for rough estimation
            features['estimated_density'] = 1 / (features['koi_prad'] ** 3)
        
        # Stellar flux features
        if all(col in features.columns for col in ['koi_srad', 'koi_dor']):
            # Stellar flux received by planet
            features['stellar_flux'] = (features['koi_srad'] ** 2) / (features['koi_dor'] ** 2)
            features['log_stellar_flux'] = np.log10(features['stellar_flux'] + 1e-8)
        
        # Transit impact parameter derived features
        if 'koi_impact' in features.columns:
            features['impact_squared'] = features['koi_impact'] ** 2
            features['grazing_transit'] = (features['koi_impact'] > 0.8).astype(int)
        
        return features
    
    def _add_polynomial_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add polynomial features for key parameters."""
        features = data.copy()
        
        key_features = ['koi_period', 'koi_prad', 'koi_srad', 'koi_teq']
        
        for feat in key_features:
            if feat in features.columns:
                # Polynomial terms
                features[f'{feat}_squared'] = features[feat] ** 2
                features[f'{feat}_sqrt'] = np.sqrt(np.abs(features[feat]))
                features[f'{feat}_log'] = np.log10(np.abs(features[feat]) + 1)
                
                # Reciprocal features
                features[f'{feat}_reciprocal'] = 1 / (features[feat] + 1e-8)
        
        return features
    
    def _add_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add temporal/frequency domain features."""
        features = data.copy()
        
        if 'koi_period' in features.columns:
            # Frequency domain features
            features['frequency'] = 1 / (features['koi_period'] + 1e-8)
            features['log_frequency'] = np.log10(features['frequency'])
            
            # Orbital velocity estimation (simplified)
            if 'koi_dor' in features.columns:
                features['orbital_velocity'] = 2 * np.pi * features['koi_dor'] / features['koi_period']
        
        return features
    
    def _add_distribution_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add statistical distribution features efficiently."""
        # Prepare features dictionary for efficient creation
        feature_dict = {}
        
        # Copy original data
        for col in data.columns:
            feature_dict[col] = data[col].values
        
        # Get numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in data.columns and not col.endswith('_rank'):
                col_data = data[col]
                
                # Percentile rank
                feature_dict[f'{col}_rank'] = col_data.rank(pct=True).values
                
                # Z-score (handle zero standard deviation)
                col_std = col_data.std()
                if col_std > 0:
                    feature_dict[f'{col}_zscore'] = ((col_data - col_data.mean()) / col_std).values
                else:
                    feature_dict[f'{col}_zscore'] = np.zeros(len(col_data))
        
        # Create DataFrame efficiently from dictionary
        features = pd.DataFrame(feature_dict)
        return features
    
    def preprocess_features(self, features: pd.DataFrame, target: Optional[pd.Series] = None, 
                          fit_transformers: bool = True) -> pd.DataFrame:
        """
        Preprocess features with scaling, selection, and dimensionality reduction.
        
        Args:
            features: Feature DataFrame
            target: Target series for supervised feature selection
            fit_transformers: Whether to fit transformers
            
        Returns:
            Preprocessed features
        """
        # Handle missing values efficiently
        features_clean = features.copy()
        
        # Fill numeric missing values with median
        numeric_cols = features_clean.select_dtypes(include=[np.number]).columns
        
        # Use SimpleImputer for better performance and consistency
        if not hasattr(self, 'imputer'):
            from sklearn.impute import SimpleImputer
            self.imputer = SimpleImputer(strategy='median')
        
        if fit_transformers:
            features_clean[numeric_cols] = self.imputer.fit_transform(features_clean[numeric_cols])
        else:
            features_clean[numeric_cols] = self.imputer.transform(features_clean[numeric_cols])
        
        # Remove constant and near-constant features
        constant_features = []
        for col in features_clean.columns:
            try:
                if features_clean[col].nunique() <= 1:
                    constant_features.append(col)
                elif features_clean[col].dtype in ['object', 'category']:
                    # Skip non-numeric columns for std calculation
                    continue
                elif features_clean[col].std() < 1e-8:  # Near-constant
                    constant_features.append(col)
            except (TypeError, ValueError):
                # Skip problematic columns
                continue
        
        if constant_features:
            features_clean = features_clean.drop(columns=constant_features)
        
        # Ensure we have numeric data only for scaling
        features_clean = features_clean.select_dtypes(include=[np.number])
        
        # Feature scaling
        if fit_transformers:
            features_scaled = pd.DataFrame(
                self.scaler.fit_transform(features_clean),
                columns=features_clean.columns,
                index=features_clean.index
            )
        else:
            features_scaled = pd.DataFrame(
                self.scaler.transform(features_clean),
                columns=features_clean.columns,
                index=features_clean.index
            )
        
        # Feature selection with better handling
        if target is not None and fit_transformers:
            # Ensure no NaN values before feature selection
            features_scaled = features_scaled.fillna(0)
            target_clean = target.fillna(0)
            
            # Select top features based on mutual information
            n_features = min(50, features_scaled.shape[1])  # Reduced to 50 for better performance
            
            # Use more robust feature selection
            try:
                self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
                features_selected = self.feature_selector.fit_transform(features_scaled, target_clean)
                
                # Get selected feature names
                selected_features = features_scaled.columns[self.feature_selector.get_support()]
                features_selected = pd.DataFrame(
                    features_selected,
                    columns=selected_features,
                    index=features_scaled.index
                )
            except Exception as e:
                # Fallback to simple variance-based selection if mutual info fails
                print(f"Feature selection failed, using variance threshold: {e}")
                from sklearn.feature_selection import VarianceThreshold
                selector = VarianceThreshold(threshold=0.01)
                features_selected = pd.DataFrame(
                    selector.fit_transform(features_scaled),
                    columns=features_scaled.columns[selector.get_support()],
                    index=features_scaled.index
                )
                self.feature_selector = selector
        elif self.feature_selector is not None:
            features_selected = pd.DataFrame(
                self.feature_selector.transform(features_scaled),
                columns=features_scaled.columns[self.feature_selector.get_support()],
                index=features_scaled.index
            )
        else:
            features_selected = features_scaled
        
        # Optional PCA for dimensionality reduction (if many features)
        if features_selected.shape[1] > 50 and fit_transformers:
            self.pca = PCA(n_components=0.95, random_state=42)  # Keep 95% variance
            features_pca = self.pca.fit_transform(features_selected)
            
            pca_columns = [f'pca_{i}' for i in range(features_pca.shape[1])]
            features_final = pd.DataFrame(
                features_pca,
                columns=pca_columns,
                index=features_selected.index
            )
        elif self.pca is not None:
            features_pca = self.pca.transform(features_selected)
            pca_columns = [f'pca_{i}' for i in range(features_pca.shape[1])]
            features_final = pd.DataFrame(
                features_pca,
                columns=pca_columns,
                index=features_selected.index
            )
        else:
            features_final = features_selected
        
        logger.info(f"Preprocessed features: {features_final.shape[1]} final features")
        return features_final
    
    def get_feature_importance_scores(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
        """
        Calculate feature importance scores using multiple methods.
        
        Args:
            features: Feature DataFrame
            target: Target series
            
        Returns:
            Dictionary of feature importance scores
        """
        importance_scores = {}
        
        # Mutual information scores
        mi_scores = mutual_info_classif(features, target, random_state=42)
        
        # F-statistic scores
        f_scores, _ = f_classif(features, target)
        
        # Correlation with target
        correlations = features.corrwith(target).abs()
        
        for i, feature in enumerate(features.columns):
            importance_scores[feature] = {
                'mutual_info': mi_scores[i],
                'f_statistic': f_scores[i],
                'correlation': correlations.iloc[i] if not pd.isna(correlations.iloc[i]) else 0,
                'combined_score': (mi_scores[i] + f_scores[i] / f_scores.max() + 
                                 (correlations.iloc[i] if not pd.isna(correlations.iloc[i]) else 0)) / 3
            }
        
        return importance_scores


class TSFreshInspiredFeatures:
    """
    TSFRESH-inspired feature extraction for light curve data.
    Based on the research paper achieving 94.8% AUC.
    """
    
    @staticmethod
    def extract_time_series_features(light_curve: np.array, time_points: Optional[np.array] = None) -> Dict[str, float]:
        """
        Extract comprehensive time series features from light curves.
        
        Args:
            light_curve: Array of brightness values
            time_points: Optional time points
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        if len(light_curve) == 0:
            return features
        
        # Basic statistical features
        features.update(TSFreshInspiredFeatures._basic_statistics(light_curve))
        
        # Energy and power features
        features.update(TSFreshInspiredFeatures._energy_features(light_curve))
        
        # Frequency domain features
        features.update(TSFreshInspiredFeatures._frequency_features(light_curve))
        
        # Peak and valley features
        features.update(TSFreshInspiredFeatures._peak_features(light_curve))
        
        # Complexity features
        features.update(TSFreshInspiredFeatures._complexity_features(light_curve))
        
        # Autocorrelation features
        features.update(TSFreshInspiredFeatures._autocorrelation_features(light_curve))
        
        return features
    
    @staticmethod
    def _basic_statistics(x: np.array) -> Dict[str, float]:
        """Basic statistical features."""
        return {
            'mean': np.mean(x),
            'std': np.std(x),
            'var': np.var(x),
            'skewness': stats.skew(x),
            'kurtosis': stats.kurtosis(x),
            'min': np.min(x),
            'max': np.max(x),
            'range': np.max(x) - np.min(x),
            'median': np.median(x),
            'q25': np.percentile(x, 25),
            'q75': np.percentile(x, 75),
            'iqr': np.percentile(x, 75) - np.percentile(x, 25),
            'mad': np.median(np.abs(x - np.median(x))),
            'cv': np.std(x) / (np.mean(x) + 1e-8)
        }
    
    @staticmethod
    def _energy_features(x: np.array) -> Dict[str, float]:
        """Energy and power-related features."""
        return {
            'energy': np.sum(x**2),
            'absolute_energy': np.sum(np.abs(x)**2),
            'mean_abs_change': np.mean(np.abs(np.diff(x))),
            'mean_change': np.mean(np.diff(x)),
            'mean_second_derivative': np.mean(np.diff(x, n=2)),
            'root_mean_square': np.sqrt(np.mean(x**2))
        }
    
    @staticmethod
    def _frequency_features(x: np.array) -> Dict[str, float]:
        """Frequency domain features."""
        if len(x) < 2:
            return {}
        
        # FFT features
        fft = np.fft.fft(x)
        fft_abs = np.abs(fft)
        
        features = {
            'fft_mean': np.mean(fft_abs),
            'fft_std': np.std(fft_abs),
            'spectral_centroid': np.sum(fft_abs * np.arange(len(fft_abs))) / (np.sum(fft_abs) + 1e-8),
            'spectral_variance': np.sum(((np.arange(len(fft_abs)) - 
                                       np.sum(fft_abs * np.arange(len(fft_abs))) / (np.sum(fft_abs) + 1e-8))**2 * fft_abs)) / (np.sum(fft_abs) + 1e-8)
        }
        
        # Power spectral density
        try:
            freqs, psd = periodogram(x)
            if len(psd) > 0:
                features.update({
                    'psd_mean': np.mean(psd),
                    'psd_std': np.std(psd),
                    'psd_max': np.max(psd),
                    'dominant_frequency': freqs[np.argmax(psd)] if len(freqs) > 0 else 0
                })
        except:
            pass
        
        return features
    
    @staticmethod
    def _peak_features(x: np.array) -> Dict[str, float]:
        """Peak and valley detection features."""
        if len(x) < 3:
            return {}
        
        peaks, _ = find_peaks(x)
        valleys, _ = find_peaks(-x)
        
        return {
            'n_peaks': len(peaks),
            'n_valleys': len(valleys),
            'peak_valley_ratio': len(peaks) / (len(valleys) + 1),
            'mean_peak_height': np.mean(x[peaks]) if len(peaks) > 0 else 0,
            'mean_valley_depth': np.mean(x[valleys]) if len(valleys) > 0 else 0,
            'peak_prominence': np.std(x[peaks]) if len(peaks) > 1 else 0
        }
    
    @staticmethod
    def _complexity_features(x: np.array) -> Dict[str, float]:
        """Complexity and entropy features."""
        if len(x) < 2:
            return {}
        
        # Approximate entropy
        def _maxdist(xi, xj, N):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _approximate_entropy(U, m, r):
            N = len(U)
            def _phi(m):
                patterns = np.array([U[i:i + m] for i in range(N - m + 1)])
                C = np.zeros(N - m + 1)
                for i in range(N - m + 1):
                    template_i = patterns[i]
                    for j in range(N - m + 1):
                        if _maxdist(template_i, patterns[j], m) <= r:
                            C[i] += 1.0
                phi = (N - m + 1.0) ** (-1) * np.sum(np.log(C / (N - m + 1.0)))
                return phi
            return _phi(m) - _phi(m + 1)
        
        try:
            approx_entropy = _approximate_entropy(x, 2, 0.2 * np.std(x))
        except:
            approx_entropy = 0
        
        return {
            'approximate_entropy': approx_entropy,
            'zero_crossing_rate': np.sum(np.diff(np.signbit(x - np.mean(x)))) / len(x),
            'longest_strike_above_mean': TSFreshInspiredFeatures._longest_strike_above_mean(x),
            'longest_strike_below_mean': TSFreshInspiredFeatures._longest_strike_below_mean(x),
            'count_above_mean': np.sum(x > np.mean(x)),
            'count_below_mean': np.sum(x < np.mean(x))
        }
    
    @staticmethod
    def _autocorrelation_features(x: np.array) -> Dict[str, float]:
        """Autocorrelation features."""
        if len(x) < 2:
            return {}
        
        # Autocorrelation at different lags
        autocorr_features = {}
        max_lag = min(10, len(x) // 2)
        
        for lag in range(1, max_lag + 1):
            if len(x) > lag:
                autocorr = np.corrcoef(x[:-lag], x[lag:])[0, 1]
                if not np.isnan(autocorr):
                    autocorr_features[f'autocorr_lag_{lag}'] = autocorr
        
        return autocorr_features
    
    @staticmethod
    def _longest_strike_above_mean(x: np.array) -> int:
        """Longest consecutive sequence above mean."""
        if len(x) == 0:
            return 0
        
        mean_val = np.mean(x)
        above_mean = x > mean_val
        
        max_length = 0
        current_length = 0
        
        for val in above_mean:
            if val:
                current_length += 1
                max_length = max(max_length, current_length)
            else:
                current_length = 0
        
        return max_length
    
    @staticmethod
    def _longest_strike_below_mean(x: np.array) -> int:
        """Longest consecutive sequence below mean."""
        if len(x) == 0:
            return 0
        
        mean_val = np.mean(x)
        below_mean = x < mean_val
        
        max_length = 0
        current_length = 0
        
        for val in below_mean:
            if val:
                current_length += 1
                max_length = max(max_length, current_length)
            else:
                current_length = 0
        
        return max_length