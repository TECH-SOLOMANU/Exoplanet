from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from app.services.nasa_service import NASADataService
from app.core.database import get_database
from typing import Optional
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

nasa_service = NASADataService()

@router.get("/fetch")
async def fetch_nasa_data(background_tasks: BackgroundTasks):
    """
    Trigger NASA data fetch from Exoplanet Archive API
    This runs as a background task to avoid request timeout
    """
    try:
        background_tasks.add_task(nasa_service.fetch_all_data)
        return {
            "message": "NASA data fetch initiated",
            "status": "processing",
            "note": "Data will be available shortly"
        }
    except Exception as e:
        logger.error(f"Failed to initiate NASA data fetch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_fetch_status():
    """Get status of latest NASA data fetch"""
    try:
        db = get_database()
        
        # Get latest fetch status from database
        status = await db.fetch_status.find_one(
            {}, 
            sort=[("timestamp", -1)]
        )
        
        if not status:
            return {
                "status": "no_data",
                "message": "No fetch attempts recorded"
            }
        
        return {
            "status": status.get("status", "unknown"),
            "timestamp": status.get("timestamp"),
            "records_fetched": status.get("records_fetched", 0),
            "message": status.get("message", "")
        }
        
    except Exception as e:
        logger.error(f"Failed to get fetch status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/latest")
async def get_latest_discoveries(limit: int = 10):
    """Get latest exoplanet discoveries from NASA"""
    try:
        db = get_database()
        
        # Get most recent discoveries (include all statuses, not just confirmed)
        cursor = db.exoplanets.find(
            {},  # Remove status filter to include all planets
            sort=[("pl_disc", -1), ("created_at", -1)],  # Sort by discovery year, then creation time
            limit=limit
        )
        
        discoveries = []
        async for planet in cursor:
            planet["_id"] = str(planet["_id"])  # Convert ObjectId to string
            discoveries.append(planet)
        
        return {
            "count": len(discoveries),
            "latest_discoveries": discoveries
        }
        
    except Exception as e:
        logger.error(f"Failed to get latest discoveries: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_nasa_data_stats():
    """Get statistics about NASA data in database"""
    try:
        db = get_database()
        
        # Count exoplanets by status
        pipeline = [
            {"$group": {
                "_id": "$pl_status",
                "count": {"$sum": 1}
            }}
        ]
        
        stats_cursor = db.exoplanets.aggregate(pipeline)
        status_counts = {}
        total_count = 0
        
        async for stat in stats_cursor:
            status = stat["_id"] or "UNKNOWN"
            count = stat["count"]
            status_counts[status] = count
            total_count += count
        
        # Get discovery year distribution
        year_pipeline = [
            {"$match": {"pl_disc": {"$exists": True, "$ne": None}}},
            {"$group": {
                "_id": "$pl_disc",
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id": -1}},
            {"$limit": 10}
        ]
        
        year_cursor = db.exoplanets.aggregate(year_pipeline)
        year_distribution = []
        
        async for year_stat in year_cursor:
            year_distribution.append({
                "year": year_stat["_id"],
                "discoveries": year_stat["count"]
            })
        
        return {
            "total_exoplanets": total_count,
            "status_distribution": status_counts,
            "recent_years_distribution": year_distribution,
            "database_collections": {
                "exoplanets": await db.exoplanets.count_documents({}),
                "predictions": await db.predictions.count_documents({}),
                "light_curves": await db.light_curves.count_documents({})
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get NASA data stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search")
async def search_exoplanets(
    name: Optional[str] = None,
    status: Optional[str] = None,
    discovery_year: Optional[int] = None,
    limit: int = 20,
    skip: int = 0
):
    """Search exoplanets with filters"""
    try:
        db = get_database()
        
        # Build query filter
        query = {}
        if name:
            query["pl_name"] = {"$regex": name, "$options": "i"}
        if status:
            query["pl_status"] = status.upper()
        if discovery_year:
            query["pl_disc"] = discovery_year
        
        # Execute search
        cursor = db.exoplanets.find(
            query,
            limit=limit,
            skip=skip,
            sort=[("pl_disc", -1)]
        )
        
        results = []
        async for planet in cursor:
            planet["_id"] = str(planet["_id"])
            results.append(planet)
        
        # Get total count for pagination
        total_count = await db.exoplanets.count_documents(query)
        
        return {
            "results": results,
            "total_count": total_count,
            "page_info": {
                "limit": limit,
                "skip": skip,
                "has_more": (skip + limit) < total_count
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to search exoplanets: {e}")
        raise HTTPException(status_code=500, detail=str(e))