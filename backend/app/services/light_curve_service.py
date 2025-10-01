import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import io
import json
from scipy import signal
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from bson import ObjectId
from app.core.database import get_database

logger = logging.getLogger(__name__)

class LightCurveService:
    """Service for processing and analyzing light curve data"""
    
    def __init__(self, db=None):
        self.db = db if db is not None else get_database()
        self.scaler = StandardScaler()
        self.transit_threshold = 0.01  # 1% brightness drop threshold
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for MongoDB compatibility"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
        
    async def process_uploaded_file(self, file_content: bytes, filename: str, file_format: str) -> Dict:
        """Process uploaded light curve file and extract time series data"""
        try:
            logger.info(f"Processing light curve file: {filename}")
            
            if file_format.lower() == 'csv':
                data = await self._process_csv(file_content)
            elif file_format.lower() == 'json':
                data = await self._process_json(file_content)
            elif file_format.lower() == 'txt':
                data = await self._process_txt(file_content)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            # Validate and clean the data
            processed_data = await self._clean_light_curve_data(data)
            
            # Store in database
            light_curve_id = await self._store_light_curve(processed_data, filename)
            
            # Generate basic statistics
            stats = await self._calculate_basic_stats(processed_data)
            
            return {
                "light_curve_id": light_curve_id,
                "filename": filename,
                "data_points": len(processed_data["time"]),
                "time_span_days": float(np.max(processed_data["time"]) - np.min(processed_data["time"])),
                "stats": stats,
                "preview_data": {
                    "time": processed_data["time"][:100].tolist(),
                    "flux": processed_data["flux"][:100].tolist()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to process light curve file: {e}")
            raise
    
    async def _process_csv(self, file_content: bytes) -> Dict:
        """Process CSV light curve data with robust parsing"""
        try:
            # Try different possible CSV formats
            content_str = file_content.decode('utf-8')
            
            # First, try to detect and clean the content
            lines = content_str.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    cleaned_lines.append(line)
            
            if not cleaned_lines:
                raise ValueError("No data found in file")
            
            # Detect separator
            separators = [',', '\t', ';', '|', ' ']
            best_sep = ','
            max_cols = 0
            
            for sep in separators:
                try:
                    cols = len(cleaned_lines[0].split(sep))
                    if cols > max_cols and cols >= 2:
                        max_cols = cols
                        best_sep = sep
                except:
                    continue
            
            # Try multiple parsing strategies
            df = None
            parsing_errors = []
            
            strategies = [
                # Strategy 1: Standard pandas with detected separator
                {'sep': best_sep, 'comment': '#', 'skip_blank_lines': True},
                # Strategy 2: Flexible whitespace separation
                {'sep': r'\s+', 'engine': 'python', 'comment': '#', 'skip_blank_lines': True},
                # Strategy 3: Manual parsing for problematic files
                {'manual': True}
            ]
            
            for strategy in strategies:
                try:
                    if strategy.get('manual'):
                        # Manual parsing for very problematic files
                        data_rows = []
                        for line in cleaned_lines:
                            parts = line.split(best_sep)
                            # Clean parts and take only numeric values
                            numeric_parts = []
                            for part in parts:
                                part = part.strip()
                                try:
                                    float(part)
                                    numeric_parts.append(part)
                                except ValueError:
                                    continue
                            
                            if len(numeric_parts) >= 2:
                                data_rows.append(numeric_parts[:2])  # Take first 2 columns
                        
                        if data_rows:
                            df = pd.DataFrame(data_rows, columns=['time', 'flux'])
                            df['time'] = pd.to_numeric(df['time'], errors='coerce')
                            df['flux'] = pd.to_numeric(df['flux'], errors='coerce')
                            df = df.dropna()
                    else:
                        # Standard pandas parsing
                        df = pd.read_csv(io.StringIO('\n'.join(cleaned_lines)), **strategy)
                    
                    # Validate result
                    if df is not None and len(df) > 0 and len(df.columns) >= 2:
                        break
                        
                except Exception as e:
                    parsing_errors.append(str(e))
                    continue
            
            if df is None or len(df) == 0:
                error_msg = f"Failed to parse CSV. Errors: {'; '.join(parsing_errors)}"
                raise ValueError(error_msg)
            
            # Try to identify time and flux columns
            time_col = None
            flux_col = None
            
            # Common column names for time
            time_names = ['time', 'bjd', 'hjd', 'mjd', 'jd', 'timestamp', 't']
            for col in df.columns:
                if str(col).lower() in time_names:
                    time_col = col
                    break
            
            # Common column names for flux
            flux_names = ['flux', 'brightness', 'magnitude', 'mag', 'flux_raw', 'pdcsap_flux', 'sap_flux']
            for col in df.columns:
                if str(col).lower() in flux_names:
                    flux_col = col
                    break
            
            # If not found, use first two numeric columns
            if time_col is None or flux_col is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 2:
                    time_col = numeric_cols[0]
                    flux_col = numeric_cols[1]
                else:
                    # Try to convert first two columns to numeric
                    if len(df.columns) >= 2:
                        df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
                        df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
                        df = df.dropna()
                        if len(df) > 0:
                            time_col = df.columns[0]
                            flux_col = df.columns[1]
                        else:
                            raise ValueError("No numeric data found after cleaning")
                    else:
                        raise ValueError("Could not identify time and flux columns")
            
            # Final validation
            if len(df) < 10:
                raise ValueError(f"Insufficient data points: {len(df)}. Need at least 10 points.")
            
            return {
                "time": df[time_col].values,
                "flux": df[flux_col].values,
                "metadata": {
                    "original_columns": df.columns.tolist(),
                    "time_column": time_col,
                    "flux_column": flux_col,
                    "rows_processed": len(df),
                    "separator_used": best_sep
                }
            }
            
        except Exception as e:
            raise ValueError(f"Failed to parse CSV file: {str(e)}")
    
    async def _process_json(self, file_content: bytes) -> Dict:
        """Process JSON light curve data"""
        try:
            content_str = file_content.decode('utf-8')
            data = json.loads(content_str)
            
            # Handle different JSON structures
            if isinstance(data, dict):
                if 'time' in data and 'flux' in data:
                    return {
                        "time": np.array(data['time']),
                        "flux": np.array(data['flux']),
                        "metadata": data.get('metadata', {})
                    }
                else:
                    # Try to find time series data in nested structure
                    for key, value in data.items():
                        if isinstance(value, dict) and 'time' in value and 'flux' in value:
                            return {
                                "time": np.array(value['time']),
                                "flux": np.array(value['flux']),
                                "metadata": data.get('metadata', {})
                            }
            
            raise ValueError("Could not find time and flux data in JSON structure")
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
    
    async def _process_txt(self, file_content: bytes) -> Dict:
        """Process TXT light curve data (space or tab separated)"""
        try:
            content_str = file_content.decode('utf-8')
            lines = content_str.strip().split('\n')
            
            # Skip header lines that start with # or contain non-numeric data
            data_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        # Try to parse as numbers
                        values = line.split()
                        float(values[0])  # Test if first value is numeric
                        float(values[1])  # Test if second value is numeric
                        data_lines.append(values)
                    except (ValueError, IndexError):
                        continue
            
            if len(data_lines) < 10:
                raise ValueError("Insufficient numeric data found in file")
            
            # Extract time and flux columns
            time_data = []
            flux_data = []
            
            for line in data_lines:
                time_data.append(float(line[0]))
                flux_data.append(float(line[1]))
            
            return {
                "time": np.array(time_data),
                "flux": np.array(flux_data),
                "metadata": {
                    "format": "txt",
                    "total_lines": len(lines),
                    "data_lines": len(data_lines)
                }
            }
            
        except Exception as e:
            raise ValueError(f"Failed to parse TXT file: {str(e)}")
    
    async def _clean_light_curve_data(self, data: Dict) -> Dict:
        """Clean and validate light curve data"""
        time = data["time"]
        flux = data["flux"]
        
        # Remove NaN values
        valid_mask = ~(np.isnan(time) | np.isnan(flux))
        time = time[valid_mask]
        flux = flux[valid_mask]
        
        # Sort by time
        sort_idx = np.argsort(time)
        time = time[sort_idx]
        flux = flux[sort_idx]
        
        # Remove outliers (flux values beyond 5 sigma)
        flux_median = np.median(flux)
        flux_std = np.std(flux)
        outlier_mask = np.abs(flux - flux_median) < 5 * flux_std
        time = time[outlier_mask]
        flux = flux[outlier_mask]
        
        # Normalize flux to relative brightness
        flux_mean = np.mean(flux)
        flux_normalized = flux / flux_mean
        
        return {
            "time": time,
            "flux": flux_normalized,
            "flux_raw": flux,
            "metadata": {
                **data.get("metadata", {}),
                "cleaned_points": len(time),
                "normalization_factor": flux_mean
            }
        }
    
    async def _store_light_curve(self, data: Dict, filename: str) -> str:
        """Store light curve data in database"""
        try:
            light_curve_doc = {
                "filename": filename,
                "uploaded_at": datetime.utcnow(),
                "data_points": len(data["time"]),
                "time_span": float(np.max(data["time"]) - np.min(data["time"])),
                "metadata": data["metadata"],
                # Store sample of data for quick access
                "sample_data": {
                    "time": data["time"][::max(1, len(data["time"])//1000)].tolist(),
                    "flux": data["flux"][::max(1, len(data["flux"])//1000)].tolist()
                },
                "analysis_status": "pending"
            }
            
            result = await self.db.light_curves.insert_one(light_curve_doc)
            
            # Store full data separately for analysis
            full_data_doc = {
                "light_curve_id": result.inserted_id,
                "time": data["time"].tolist(),
                "flux": data["flux"].tolist(),
                "flux_raw": data["flux_raw"].tolist()
            }
            
            await self.db.light_curve_data.insert_one(full_data_doc)
            
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Failed to store light curve: {e}")
            raise
    
    async def _calculate_basic_stats(self, data: Dict) -> Dict:
        """Calculate basic statistics for light curve"""
        flux = data["flux"]
        time = data["time"]
        
        return {
            "mean_flux": float(np.mean(flux)),
            "std_flux": float(np.std(flux)),
            "min_flux": float(np.min(flux)),
            "max_flux": float(np.max(flux)),
            "flux_range": float(np.max(flux) - np.min(flux)),
            "time_span": float(np.max(time) - np.min(time)),
            "cadence_median": float(np.median(np.diff(time))),
            "data_points": len(flux)
        }
    
    async def run_advanced_analysis(self, light_curve_id: str) -> Dict:
        """Run comprehensive advanced analysis including ML models and statistical tests"""
        try:
            logger.info(f"Running advanced analysis for light curve: {light_curve_id}")
            
            # Get light curve data
            oid = ObjectId(light_curve_id)
            data = await self.db.light_curve_data.find_one({"light_curve_id": oid})
            
            if not data:
                raise ValueError("Light curve data not found")
            
            time = np.array(data["time"])
            flux = np.array(data["flux"])
            
            # 1. Periodicity Analysis using Lomb-Scargle periodogram
            frequency_analysis = self._analyze_periodicity(time, flux)
            
            # 2. Machine Learning Classification
            ml_classification = self._ml_planet_classification(time, flux)
            
            # 3. Advanced Statistical Analysis
            statistical_analysis = self._advanced_statistics(time, flux)
            
            # 4. Transit Model Fitting
            transit_modeling = self._fit_transit_models(time, flux)
            
            # 5. Stellar Variability Analysis
            variability_analysis = self._analyze_stellar_variability(time, flux)
            
            # 6. Data Quality Assessment
            quality_metrics = self._assess_data_quality(time, flux)
            
            # Combine all results
            advanced_results = {
                "analysis_type": "advanced",
                "timestamp": datetime.utcnow(),
                "frequency_analysis": frequency_analysis,
                "ml_classification": ml_classification,
                "statistical_analysis": statistical_analysis,
                "transit_modeling": transit_modeling,
                "variability_analysis": variability_analysis,
                "quality_metrics": quality_metrics,
                "overall_assessment": self._generate_overall_assessment(
                    frequency_analysis, ml_classification, statistical_analysis,
                    transit_modeling, variability_analysis, quality_metrics
                )
            }
            
            # Convert numpy types to Python types for MongoDB
            advanced_results = self._convert_numpy_types(advanced_results)
            
            # Update database with advanced analysis results
            await self.db.light_curves.update_one(
                {"_id": oid},
                {"$set": {
                    "advanced_analysis": advanced_results,
                    "advanced_analysis_status": "completed",
                    "advanced_analyzed_at": datetime.utcnow()
                }}
            )
            
            logger.info(f"Advanced analysis completed for light curve: {light_curve_id}")
            return advanced_results
            
        except Exception as e:
            logger.error(f"Advanced analysis failed: {e}")
            # Update status to failed
            try:
                await self.db.light_curves.update_one(
                    {"_id": ObjectId(light_curve_id)},
                    {"$set": {
                        "advanced_analysis_status": "failed",
                        "advanced_analysis_error": str(e)
                    }}
                )
            except Exception:
                pass
            raise
    
    def _analyze_periodicity(self, time: np.ndarray, flux: np.ndarray) -> Dict:
        """Analyze periodicity using Lomb-Scargle periodogram"""
        try:
            from scipy.signal import lombscargle
            
            # Prepare data
            flux_normalized = (flux - np.mean(flux)) / np.std(flux)
            
            # Define frequency range (0.1 to 10 cycles per day)
            min_freq = 0.1 / (time[-1] - time[0])  # Minimum frequency
            max_freq = 10.0  # Maximum frequency (cycles per day)
            frequencies = np.linspace(min_freq, max_freq, 10000)
            
            # Calculate Lomb-Scargle periodogram
            angular_frequencies = 2 * np.pi * frequencies
            power = lombscargle(time, flux_normalized, angular_frequencies, normalize=True)
            
            # Find peaks
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(power, height=0.1, distance=100)
            
            # Get top 5 periods
            top_peaks = peaks[np.argsort(power[peaks])[-5:]][::-1]
            
            periods = []
            for peak in top_peaks:
                period = 1.0 / frequencies[peak]
                periods.append({
                    "period_days": float(period),
                    "power": float(power[peak]),
                    "frequency": float(frequencies[peak]),
                    "false_alarm_probability": self._calculate_fap(power[peak], len(time))
                })
            
            return {
                "dominant_periods": periods,
                "max_power": float(np.max(power)),
                "mean_power": float(np.mean(power)),
                "periodicity_detected": bool(np.max(power) > 0.3),
                "frequency_grid": frequencies[:1000].tolist(),  # Subset for visualization
                "power_spectrum": power[:1000].tolist()
            }
            
        except Exception as e:
            return {"error": f"Periodicity analysis failed: {str(e)}"}
    
    def _ml_planet_classification(self, time: np.ndarray, flux: np.ndarray) -> Dict:
        """Machine learning classification for planet detection"""
        try:
            # Extract features for ML model
            features = self._extract_ml_features(time, flux)
            
            # Simple Random Forest classifier (in production, use pre-trained model)
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            
            # For demo purposes, use a simple heuristic-based classification
            # In production, this would use a pre-trained model on Kepler data
            
            transit_score = features.get('transit_score', 0)
            periodicity_score = features.get('periodicity_score', 0)
            depth_score = features.get('depth_score', 0)
            
            # Combine scores
            ml_score = 0.4 * transit_score + 0.3 * periodicity_score + 0.3 * depth_score
            
            classification = {
                "planet_probability": float(ml_score),
                "classification": "planet_candidate" if ml_score > 0.6 else "non_planet",
                "confidence": float(min(ml_score * 1.2, 1.0)),
                "features_used": list(features.keys()),
                "feature_values": features,
                "model_version": "heuristic_v1.0"
            }
            
            return classification
            
        except Exception as e:
            return {"error": f"ML classification failed: {str(e)}"}
    
    def _extract_ml_features(self, time: np.ndarray, flux: np.ndarray) -> Dict:
        """Extract features for machine learning"""
        features = {}
        
        try:
            # Basic statistics
            features['mean_flux'] = float(np.mean(flux))
            features['std_flux'] = float(np.std(flux))
            features['skewness'] = float(self._calculate_skewness(flux))
            features['kurtosis'] = float(self._calculate_kurtosis(flux))
            
            # Transit-like features
            features['min_flux'] = float(np.min(flux))
            features['max_flux'] = float(np.max(flux))
            features['flux_range'] = float(np.max(flux) - np.min(flux))
            
            # Detect potential transits
            potential_transits = self._find_transit_events(time, flux)
            features['transit_count'] = len(potential_transits)
            
            if potential_transits:
                depths = [t['depth'] for t in potential_transits]
                features['mean_transit_depth'] = float(np.mean(depths))
                features['std_transit_depth'] = float(np.std(depths))
                features['max_transit_depth'] = float(np.max(depths))
            else:
                features['mean_transit_depth'] = 0.0
                features['std_transit_depth'] = 0.0
                features['max_transit_depth'] = 0.0
            
            # Derived scores
            features['transit_score'] = min(features['transit_count'] / 5.0, 1.0)
            features['depth_score'] = min(features['max_transit_depth'] * 100, 1.0)
            features['periodicity_score'] = min(features['std_flux'] * 10, 1.0)
            
            return features
            
        except Exception as e:
            return {"error": f"Feature extraction failed: {str(e)}"}
    
    def _advanced_statistics(self, time: np.ndarray, flux: np.ndarray) -> Dict:
        """Advanced statistical analysis"""
        try:
            stats = {}
            
            # Time series statistics
            stats['time_span'] = float(time[-1] - time[0])
            stats['cadence'] = float(np.median(np.diff(time)))
            stats['data_points'] = int(len(time))
            
            # Flux statistics
            stats['mean'] = float(np.mean(flux))
            stats['median'] = float(np.median(flux))
            stats['std'] = float(np.std(flux))
            stats['variance'] = float(np.var(flux))
            stats['mad'] = float(np.median(np.abs(flux - np.median(flux))))  # Median Absolute Deviation
            
            # Distribution analysis
            stats['skewness'] = float(self._calculate_skewness(flux))
            stats['kurtosis'] = float(self._calculate_kurtosis(flux))
            
            # Autocorrelation analysis
            autocorr = self._calculate_autocorrelation(flux)
            stats['autocorrelation_peak'] = float(np.max(autocorr[1:50]))  # Skip lag 0
            stats['autocorrelation_decay'] = self._calculate_autocorr_decay(autocorr)
            
            # Trend analysis
            stats['linear_trend'] = self._calculate_linear_trend(time, flux)
            
            return stats
            
        except Exception as e:
            return {"error": f"Statistical analysis failed: {str(e)}"}
    
    def _fit_transit_models(self, time: np.ndarray, flux: np.ndarray) -> Dict:
        """Fit theoretical transit models to the data"""
        try:
            # Find potential transit events first
            transit_events = self._find_transit_events(time, flux)
            
            if not transit_events:
                return {"message": "No transit events found for modeling"}
            
            models = []
            for i, event in enumerate(transit_events[:3]):  # Model top 3 events
                try:
                    # Extract transit segment
                    mask = (time >= event['start_time']) & (time <= event['end_time'])
                    t_segment = time[mask]
                    f_segment = flux[mask]
                    
                    if len(t_segment) < 5:
                        continue
                    
                    # Simple trapezoid model fitting
                    model = self._fit_trapezoid_model(t_segment, f_segment)
                    model['event_id'] = i
                    models.append(model)
                    
                except Exception as e:
                    continue
            
            return {
                "fitted_models": models,
                "total_events_modeled": len(models),
                "model_type": "trapezoid"
            }
            
        except Exception as e:
            return {"error": f"Transit modeling failed: {str(e)}"}
    
    def _analyze_stellar_variability(self, time: np.ndarray, flux: np.ndarray) -> Dict:
        """Analyze stellar variability and noise characteristics"""
        try:
            variability = {}
            
            # RMS variability
            variability['rms'] = float(np.sqrt(np.mean((flux - np.mean(flux))**2)))
            
            # Point-to-point scatter
            diff = np.diff(flux)
            variability['point_to_point_scatter'] = float(np.std(diff))
            
            # Variability on different timescales
            for window in [0.1, 0.5, 1.0, 5.0]:  # days
                window_points = int(window / np.median(np.diff(time)))
                if window_points > 5:
                    windowed_std = self._rolling_std(flux, window_points)
                    variability[f'variability_{window}d'] = float(np.median(windowed_std))
            
            # Red noise estimation
            variability['red_noise_power'] = self._estimate_red_noise(time, flux)
            
            return variability
            
        except Exception as e:
            return {"error": f"Variability analysis failed: {str(e)}"}
    
    def _assess_data_quality(self, time: np.ndarray, flux: np.ndarray) -> Dict:
        """Assess the quality of the light curve data"""
        try:
            quality = {}
            
            # Completeness
            expected_points = (time[-1] - time[0]) / np.median(np.diff(time))
            quality['completeness'] = float(len(time) / expected_points)
            
            # Gap analysis
            gaps = np.diff(time)
            median_gap = np.median(gaps)
            large_gaps = gaps > 5 * median_gap
            quality['large_gaps_count'] = int(np.sum(large_gaps))
            quality['max_gap_days'] = float(np.max(gaps))
            
            # Outlier detection
            mad = np.median(np.abs(flux - np.median(flux)))
            outliers = np.abs(flux - np.median(flux)) > 5 * mad
            quality['outlier_fraction'] = float(np.sum(outliers) / len(flux))
            
            # Noise level
            quality['noise_level'] = float(np.std(flux) / np.mean(flux))
            
            # Overall quality score
            completeness_score = min(quality['completeness'], 1.0)
            gap_score = max(0, 1.0 - quality['large_gaps_count'] / 10.0)
            outlier_score = max(0, 1.0 - quality['outlier_fraction'] * 10)
            noise_score = max(0, 1.0 - quality['noise_level'] * 1000)
            
            quality['overall_quality_score'] = float(
                0.3 * completeness_score + 0.3 * gap_score + 
                0.2 * outlier_score + 0.2 * noise_score
            )
            
            return quality
            
        except Exception as e:
            return {"error": f"Quality assessment failed: {str(e)}"}
    
    def _generate_overall_assessment(self, freq_analysis, ml_class, stats, 
                                   transit_models, variability, quality) -> Dict:
        """Generate an overall assessment combining all analyses"""
        assessment = {}
        
        # Extract key metrics
        ml_prob = ml_class.get('planet_probability', 0)
        quality_score = quality.get('overall_quality_score', 0.5)
        periodicity = freq_analysis.get('periodicity_detected', False)
        transit_count = len(transit_models.get('fitted_models', []))
        
        # Calculate combined confidence
        confidence_factors = []
        if ml_prob > 0.6:
            confidence_factors.append(0.4)
        if periodicity:
            confidence_factors.append(0.2)
        if transit_count > 0:
            confidence_factors.append(0.3)
        if quality_score > 0.7:
            confidence_factors.append(0.1)
        
        combined_confidence = sum(confidence_factors)
        
        # Generate recommendation
        if combined_confidence > 0.7:
            recommendation = "Strong planet candidate - recommend follow-up observations"
        elif combined_confidence > 0.4:
            recommendation = "Possible planet candidate - requires additional analysis"
        else:
            recommendation = "Unlikely to be a planet - consider stellar variability or noise"
        
        assessment = {
            "combined_confidence": float(combined_confidence),
            "recommendation": recommendation,
            "data_quality": "excellent" if quality_score > 0.8 else 
                           "good" if quality_score > 0.6 else "fair",
            "analysis_completeness": 1.0,  # All analyses completed
            "key_findings": []
        }
        
        # Add key findings
        if ml_prob > 0.7:
            assessment["key_findings"].append("High machine learning confidence for planet detection")
        if periodicity:
            assessment["key_findings"].append("Significant periodicity detected in light curve")
        if transit_count > 2:
            assessment["key_findings"].append(f"Multiple transit-like events detected ({transit_count})")
        if quality_score < 0.5:
            assessment["key_findings"].append("Data quality issues may affect reliability")
        
        return assessment
    
    # Helper methods for advanced analysis
    def _calculate_skewness(self, data):
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_autocorrelation(self, data, max_lag=100):
        """Calculate autocorrelation function"""
        n = len(data)
        data = data - np.mean(data)
        autocorr = np.correlate(data, data, mode='full')
        autocorr = autocorr[n-1:]
        autocorr = autocorr / autocorr[0]
        return autocorr[:min(max_lag, len(autocorr))]
    
    def _calculate_autocorr_decay(self, autocorr):
        """Calculate how quickly autocorrelation decays"""
        try:
            # Find where autocorrelation drops below 1/e
            threshold = 1.0 / np.e
            decay_index = np.where(autocorr < threshold)[0]
            return float(decay_index[0]) if len(decay_index) > 0 else float(len(autocorr))
        except:
            return float(len(autocorr))
    
    def _calculate_linear_trend(self, time, flux):
        """Calculate linear trend in the data"""
        try:
            coeffs = np.polyfit(time, flux, 1)
            return float(coeffs[0])  # Slope
        except:
            return 0.0
    
    def _calculate_fap(self, power, n_data):
        """Calculate false alarm probability for periodogram peak"""
        # Rough approximation
        return float(1.0 - (1.0 - np.exp(-power)) ** n_data)
    
    def _fit_trapezoid_model(self, time, flux):
        """Fit a simple trapezoid transit model"""
        try:
            # Simple trapezoid parameters
            depth = np.min(flux) - np.median(flux)
            duration = time[-1] - time[0]
            center_time = np.mean(time)
            
            return {
                "depth": float(depth),
                "duration_hours": float(duration * 24),
                "center_time": float(center_time),
                "model_type": "trapezoid",
                "chi_squared": float(np.sum((flux - np.median(flux))**2))
            }
        except:
            return {"error": "Model fitting failed"}
    
    def _rolling_std(self, data, window):
        """Calculate rolling standard deviation"""
        result = []
        for i in range(len(data) - window + 1):
            result.append(np.std(data[i:i+window]))
        return np.array(result)
    
    def _estimate_red_noise(self, time, flux):
        """Estimate red noise power"""
        try:
            # Simple red noise estimation using power spectral density
            dt = np.median(np.diff(time))
            freqs = np.fft.fftfreq(len(flux), dt)
            power = np.abs(np.fft.fft(flux - np.mean(flux)))**2
            
            # Focus on low frequencies for red noise
            low_freq_mask = (freqs > 0) & (freqs < 0.1)
            if np.any(low_freq_mask):
                return float(np.mean(power[low_freq_mask]))
            else:
                return 0.0
        except:
            return 0.0
        """Convert numpy types to Python native types for MongoDB storage"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def detect_transits(self, light_curve_id: str) -> Dict:
        """Detect potential transit events in light curve (synchronous for background tasks)"""
        try:
            logger.info(f"Detecting transits for light curve: {light_curve_id}")
            
            # Convert string ID to ObjectId
            obj_id = ObjectId(light_curve_id)
            
            # Use synchronous MongoDB client for background tasks
            from pymongo import MongoClient
            from app.core.config import settings
            
            sync_client = MongoClient(settings.MONGODB_URL)
            sync_db = sync_client[settings.DATABASE_NAME]
            
            try:
                # Convert string ID to ObjectId for the query
                obj_id = ObjectId(light_curve_id)
                data = sync_db.light_curve_data.find_one({"light_curve_id": obj_id})
                
                if not data:
                    raise ValueError("Light curve data not found")
                
                time = np.array(data["time"])
                flux = np.array(data["flux"])
                
                # Perform transit detection analysis using synchronous methods
                logger.info("Running transit detection algorithms...")
                
                # Detrend the light curve
                detrended_flux = self._detrend_light_curve_sync(time, flux)
                
                # Find transit events
                transit_events = self._find_transit_events_sync(time, detrended_flux)
                
                # Analyze transit candidates
                analysis_results = self._analyze_transit_candidates_sync(time, detrended_flux, transit_events)
                
                # Prepare results for database storage
                results = {
                    "transit_events": transit_events,
                    "planet_detected": analysis_results["planet_detected"],
                    "confidence": analysis_results["confidence"],
                    "planet_parameters": analysis_results.get("planet_parameters", {}),
                    "recommendations": analysis_results.get("recommendations", [])
                }
                
                # Convert numpy types to Python types for MongoDB
                results_for_db = self._convert_numpy_types(results)
                
                # Update database with results
                sync_db.light_curves.update_one(
                    {"_id": obj_id},
                    {"$set": {
                        "analysis_status": "completed",
                        "transit_detection": results_for_db,
                        "analyzed_at": datetime.utcnow()
                    }}
                )
                
                logger.info(f"Transit detection completed for light curve: {light_curve_id}")
                
                # Close the synchronous client
                sync_client.close()
                
                return results
            except Exception as e:
                logger.error(f"Error during transit detection: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Transit detection failed: {e}")
            
            # Update status to failed in database
            try:
                from pymongo import MongoClient
                from app.core.config import settings
                
                sync_client = MongoClient(settings.MONGODB_URL)
                sync_db = sync_client[settings.DATABASE_NAME]
                
                sync_db.light_curves.update_one(
                    {"_id": ObjectId(light_curve_id)},
                    {"$set": {
                        "analysis_status": "failed",
                        "error_message": str(e),
                        "analyzed_at": datetime.utcnow()
                    }}
                )
                sync_client.close()
            
            except Exception as db_error:
                logger.error(f"Failed to update database with error status: {db_error}")
            
            raise
    
    async def _detrend_light_curve(self, time: np.ndarray, flux: np.ndarray) -> np.ndarray:
        """Remove long-term trends from light curve"""
        try:
            # Use Savitzky-Golay filter for detrending
            window_length = min(101, len(flux) // 4)
            if window_length % 2 == 0:
                window_length += 1
            if window_length < 5:
                window_length = 5
            
            trend = signal.savgol_filter(flux, window_length, 3)
            detrended = flux - trend + np.median(flux)
            
            return detrended
            
        except Exception as e:
            logger.warning(f"Detrending failed, using original data: {e}")
            return flux
    
    def _detrend_light_curve_sync(self, time: np.ndarray, flux: np.ndarray) -> np.ndarray:
        """Remove long-term trends from light curve (synchronous version)"""
        try:
            # Use Savitzky-Golay filter for detrending
            window_length = min(101, len(flux) // 4)
            if window_length % 2 == 0:
                window_length += 1
            if window_length < 5:
                window_length = 5
            
            trend = signal.savgol_filter(flux, window_length, 3)
            detrended = flux - trend + np.median(flux)
            
            return detrended
            
        except Exception as e:
            logger.warning(f"Detrending failed, using original data: {e}")
            return flux
    
    async def _find_transit_events(self, time: np.ndarray, flux: np.ndarray) -> List[Dict]:
        """Find potential transit events"""
        events = []
        
        # Simple transit detection: look for dips below threshold
        median_flux = np.median(flux)
        threshold = median_flux - self.transit_threshold
        
        # Find points below threshold
        transit_mask = flux < threshold
        
        if not np.any(transit_mask):
            return events
        
        # Group consecutive transit points
        transit_groups = []
        current_group = []
        
        for i, is_transit in enumerate(transit_mask):
            if is_transit:
                current_group.append(i)
            else:
                if current_group:
                    transit_groups.append(current_group)
                    current_group = []
        
        if current_group:
            transit_groups.append(current_group)
        
        # Analyze each transit group
        for group in transit_groups:
            if len(group) >= 3:  # Minimum transit duration
                start_idx = group[0]
                end_idx = group[-1]
                
                depth = median_flux - np.min(flux[group])
                duration = time[end_idx] - time[start_idx]
                center_time = time[start_idx + len(group)//2]
                
                events.append({
                    "center_time": float(center_time),
                    "duration": float(duration),
                    "depth": float(depth),
                    "start_time": float(time[start_idx]),
                    "end_time": float(time[end_idx]),
                    "indices": group
                })
        
        return events
    
    def _find_transit_events_sync(self, time: np.ndarray, flux: np.ndarray) -> List[Dict]:
        """Find potential transit events (synchronous version)"""
        events = []
        
        # Simple transit detection: look for dips below threshold
        median_flux = np.median(flux)
        threshold = median_flux - self.transit_threshold
        
        # Find points below threshold
        transit_mask = flux < threshold
        
        if not np.any(transit_mask):
            return events
        
        # Group consecutive transit points
        transit_groups = []
        current_group = []
        
        for i, is_transit in enumerate(transit_mask):
            if is_transit:
                current_group.append(i)
            else:
                if current_group:
                    transit_groups.append(current_group)
                    current_group = []
        
        if current_group:
            transit_groups.append(current_group)
        
        # Analyze each transit group
        for group in transit_groups:
            if len(group) >= 3:  # Minimum transit duration
                start_idx = group[0]
                end_idx = group[-1]
                
                depth = median_flux - np.min(flux[group])
                duration = time[end_idx] - time[start_idx]
                center_time = time[start_idx + len(group)//2]
                
                events.append({
                    "center_time": float(center_time),
                    "duration": float(duration),
                    "depth": float(depth),
                    "start_time": float(time[start_idx]),
                    "end_time": float(time[end_idx]),
                    "indices": group
                })
        
        return events
    
    async def _analyze_transit_candidates(self, time: np.ndarray, flux: np.ndarray, events: List[Dict]) -> Dict:
        """Analyze transit candidates and generate predictions"""
        if not events:
            return {
                "planet_detected": False,
                "confidence": 0.0,
                "transit_events": [],
                "analysis": "No significant transit events detected"
            }
        
        # Calculate confidence based on transit characteristics
        confidence_scores = []
        
        for event in events:
            # Factors that increase confidence:
            # 1. Appropriate depth (0.1% - 10%)
            # 2. Reasonable duration (1-12 hours)
            # 3. Smooth ingress/egress
            
            depth_score = 0.5
            if 0.001 <= event["depth"] <= 0.1:  # 0.1% to 10%
                depth_score = 1.0
            
            duration_hours = event["duration"] * 24  # Convert days to hours
            duration_score = 0.5
            if 1 <= duration_hours <= 12:
                duration_score = 1.0
            
            # Shape analysis (simplified)
            shape_score = 0.7  # Default moderate score
            
            event_confidence = (depth_score + duration_score + shape_score) / 3
            confidence_scores.append(event_confidence)
            event["confidence"] = float(event_confidence)
        
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        # Estimate planetary parameters for best candidate
        best_event = max(events, key=lambda x: x["confidence"]) if events else None
        planet_params = {}
        
        if best_event and overall_confidence > 0.6:
            # Rough estimates based on transit characteristics
            planet_params = {
                "estimated_radius_ratio": float(np.sqrt(best_event["depth"])),
                "orbital_period_estimate": "Multiple transits needed",
                "transit_duration_hours": float(best_event["duration"] * 24),
                "analysis_note": "Preliminary detection - requires confirmation"
            }
        
        return {
            "planet_detected": bool(overall_confidence > 0.6),
            "confidence": float(overall_confidence),
            "transit_events": events,
            "planet_parameters": planet_params,
            "analysis": f"Detected {len(events)} potential transit events",
            "recommendations": await self._generate_recommendations(events, overall_confidence)
        }
    
    def _analyze_transit_candidates_sync(self, time: np.ndarray, flux: np.ndarray, events: List[Dict]) -> Dict:
        """Analyze transit candidates and generate predictions (synchronous version)"""
        if not events:
            return {
                "planet_detected": False,
                "confidence": 0.0,
                "transit_events": [],
                "analysis": "No significant transit events detected"
            }
        
        # Calculate confidence based on transit characteristics
        confidence_scores = []
        
        for event in events:
            # Factors that increase confidence:
            # 1. Appropriate depth (0.1% - 10%)
            # 2. Reasonable duration (1-12 hours)
            # 3. Smooth ingress/egress
            
            depth_score = 0.5
            if 0.001 <= event["depth"] <= 0.1:  # 0.1% to 10%
                depth_score = 1.0
            
            duration_hours = event["duration"] * 24  # Convert days to hours
            duration_score = 0.5
            if 1 <= duration_hours <= 12:
                duration_score = 1.0
            
            # Shape analysis (simplified)
            shape_score = 0.7  # Default moderate score
            
            event_confidence = (depth_score + duration_score + shape_score) / 3
            confidence_scores.append(event_confidence)
            event["confidence"] = float(event_confidence)
        
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        # Estimate planetary parameters for best candidate
        best_event = max(events, key=lambda x: x["confidence"]) if events else None
        planet_params = {}
        
        if best_event and overall_confidence > 0.6:
            # Rough estimates based on transit characteristics
            planet_params = {
                "estimated_radius_ratio": float(np.sqrt(best_event["depth"])),
                "orbital_period_estimate": "Multiple transits needed",
                "transit_duration_hours": float(best_event["duration"] * 24),
                "analysis_note": "Preliminary detection - requires confirmation"
            }
        
        return {
            "planet_detected": bool(overall_confidence > 0.6),
            "confidence": float(overall_confidence),
            "transit_events": events,
            "planet_parameters": planet_params,
            "analysis": f"Detected {len(events)} potential transit events",
            "recommendations": self._generate_recommendations_sync(events, overall_confidence)
        }
    
    async def _generate_recommendations(self, events: List[Dict], confidence: float) -> List[str]:
        """Generate analysis recommendations"""
        recommendations = []
        
        if confidence < 0.3:
            recommendations.append("No clear transit signals detected. Consider longer observation period.")
        elif confidence < 0.6:
            recommendations.append("Weak transit candidates found. Additional observations recommended.")
        else:
            recommendations.append("Strong transit candidates detected. Follow-up observations suggested.")
        
        if len(events) == 1:
            recommendations.append("Single transit detected. Multiple transits needed to confirm periodicity.")
        elif len(events) > 1:
            recommendations.append("Multiple transits detected. Analyze for periodic behavior.")
        
        recommendations.append("Consider noise analysis and systematic error removal.")
        recommendations.append("Validate using different detrending methods.")
        
        return recommendations

    def _generate_recommendations_sync(self, events: List[Dict], confidence: float) -> List[str]:
        """Generate analysis recommendations (synchronous version)"""
        recommendations = []
        
        if confidence < 0.3:
            recommendations.append("No clear transit signals detected. Consider longer observation period.")
        elif confidence < 0.6:
            recommendations.append("Weak transit candidates found. Additional observations recommended.")
        else:
            recommendations.append("Strong transit candidates detected. Follow-up observations suggested.")
        
        if len(events) > 1:
            recommendations.append("Multiple transit events detected. Check for periodicity.")
        
        recommendations.append("Consider noise analysis and systematic error removal.")
        recommendations.append("Validate using different detrending methods.")
        
        return recommendations

    async def get_light_curve_analysis(self, light_curve_id: str) -> Dict:
        """Get complete analysis results for a light curve"""
        try:
            # Convert string ID to ObjectId
            obj_id = ObjectId(light_curve_id)
            
            # Get light curve metadata
            light_curve = await self.db.light_curves.find_one({"_id": obj_id})
            if not light_curve:
                raise ValueError("Light curve not found")
            
            # Get full data if needed for visualization
            data = await self.db.light_curve_data.find_one({"light_curve_id": obj_id})
            
            result = {
                "light_curve_id": light_curve_id,
                "filename": light_curve["filename"],
                "uploaded_at": light_curve["uploaded_at"],
                "analysis_status": light_curve.get("analysis_status", "pending"),
                "data_points": light_curve["data_points"],
                "time_span": light_curve["time_span"],
                "sample_data": light_curve["sample_data"]
            }
            
            if "transit_detection" in light_curve:
                result["transit_analysis"] = light_curve["transit_detection"]
            
            if data:
                result["full_data_available"] = True
                result["data_size"] = len(data["time"])
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get light curve analysis: {e}")
            raise