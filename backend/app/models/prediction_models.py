from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class PredictionRequest(BaseModel):
    """Request model for exoplanet prediction"""
    orbital_period: float
    planet_radius: float
    planet_mass: Optional[float] = None
    equilibrium_temp: Optional[float] = None
    stellar_temp: Optional[float] = None
    stellar_radius: Optional[float] = None
    stellar_mass: Optional[float] = None

class PredictionResponse(BaseModel):
    """Response model for exoplanet prediction"""
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    feature_importance: Dict[str, float]