import asyncio
import sys
sys.path.append('.')
from app.ml.enhanced_ensemble_models import EnsembleExoplanetDetector
from app.core.database import get_database
import pandas as pd
import numpy as np

async def test_ensemble():
    print('Testing Enhanced Ensemble Models...')
    
    # Create sample data similar to NASA exoplanet data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic exoplanet features
    data = {
        'koi_period': np.random.lognormal(2, 1, n_samples),  # Orbital period
        'koi_prad': np.random.lognormal(0, 0.5, n_samples),  # Planet radius
        'koi_teq': np.random.normal(500, 200, n_samples),    # Equilibrium temperature
        'koi_sma': np.random.lognormal(-1, 0.5, n_samples),  # Semi-major axis
        'koi_dor': np.random.lognormal(2, 0.3, n_samples),   # Distance over radius ratio
        'koi_incl': np.random.normal(90, 10, n_samples),     # Inclination
        'koi_impact': np.random.uniform(0, 1, n_samples),    # Impact parameter
        'koi_duration': np.random.lognormal(2, 0.5, n_samples),  # Transit duration
        'koi_depth': np.random.lognormal(6, 1, n_samples),   # Transit depth
    }
    
    X = pd.DataFrame(data)
    
    # Create realistic target (more planets with shorter periods and certain sizes)
    y = (
        (X['koi_period'] < 50) & 
        (X['koi_prad'] > 0.5) & 
        (X['koi_prad'] < 4) & 
        (X['koi_teq'] > 200) & 
        (X['koi_teq'] < 1000)
    ).astype(int)
    
    print(f'Training data: {len(X)} samples')
    print(f'Target distribution: {y.sum()} confirmed, {len(y) - y.sum()} candidates')
    
    # Initialize ensemble detector
    ensemble_detector = EnsembleExoplanetDetector()
    
    # Train models (small CV for speed)
    results = await ensemble_detector.train_ensemble_models(X, y, cv_folds=3)
    
    print('\nTraining Results:')
    print(f'Best model: {results["best_model"]["name"]}')
    print(f'Best accuracy: {results["best_model"]["accuracy"]:.4f}')
    print(f'Features generated: {results["feature_count"]}')
    print(f'Models trained: {len(results["individual_models"])}')
    
    # Test prediction
    test_sample = X.iloc[[0]].copy()
    prediction = await ensemble_detector.predict(test_sample, use_ensemble=True)
    probabilities = await ensemble_detector.predict(test_sample, use_ensemble=True, return_probabilities=True)
    
    print(f'\nTest prediction: {prediction[0]} (probability: {probabilities[0]:.4f})')
    
    print('\nEnhanced ensemble models testing completed successfully!')

if __name__ == "__main__":
    asyncio.run(test_ensemble())