from fastapi import APIRouter, HTTPException
from app.services.nasa_service import NASADataService
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/light-curve/{planet_name}")
async def get_light_curve(planet_name: str):
    """Get light curve data for exoplanet transit analysis"""
    try:
        nasa_service = NASADataService()
        light_curve_data = await nasa_service.fetch_light_curve_data(planet_name)
        
        if light_curve_data:
            return light_curve_data
        else:
            raise HTTPException(status_code=404, detail=f"Light curve data not found for {planet_name}")
            
    except Exception as e:
        logger.error(f"Failed to fetch light curve for {planet_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))