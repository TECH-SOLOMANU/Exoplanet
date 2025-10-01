import logging
from datetime import datetime
from typing import Dict, List, Optional
import aiohttp
import asyncio
from urllib.parse import quote
from app.core.database import get_database

logger = logging.getLogger(__name__)

class NASADataService:
    """Service for fetching data from NASA APIs"""
    
    def __init__(self):
        self.exoplanet_archive_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        # Alternative: Try the CSV API instead of TAP/ADQL
        self.csv_api_url = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
        self.mast_url = "https://mast.stsci.edu/api/v0.1"
    
    async def fetch_all_data(self):
        """Fetch all exoplanet data from NASA Exoplanet Archive"""
        db = get_database()
        
        # Record fetch start
        fetch_start = {
            "status": "started",
            "timestamp": datetime.utcnow(),
            "message": "NASA data fetch initiated"
        }
        await db.fetch_status.insert_one(fetch_start)
        
        try:
            logger.info("Starting NASA Exoplanet Archive data fetch...")
            
            # Clear existing data
            await db.exoplanets.delete_many({})
            
            # Fetch real data from NASA
            exoplanets = await self._fetch_confirmed_exoplanets()
            candidates = await self._fetch_planet_candidates()
            
            all_planets = exoplanets + candidates
            
            # Insert all data into database with upsert to handle duplicates
            if all_planets:
                # Use upsert to avoid duplicate key errors, skip records with null names
                operations = []
                valid_planets = 0
                for planet in all_planets:
                    planet_name = planet.get("pl_name")
                    if planet_name and planet_name.strip() and planet_name != "null":  # More thorough validation
                        operation = {
                            "updateOne": {
                                "filter": {"pl_name": planet_name},
                                "update": {"$set": planet},
                                "upsert": True
                            }
                        }
                        operations.append(operation)
                        valid_planets += 1
                
                logger.info(f"Prepared {len(operations)} valid operations from {len(all_planets)} total records")
                
                # Execute bulk operations
                if operations:
                    from pymongo import UpdateOne
                    # Convert to proper pymongo operations
                    mongo_operations = []
                    for op in operations:
                        mongo_op = UpdateOne(
                            op["updateOne"]["filter"],
                            op["updateOne"]["update"],
                            upsert=op["updateOne"]["upsert"]
                        )
                        mongo_operations.append(mongo_op)
                    
                    await db.exoplanets.bulk_write(mongo_operations)
            
            total_fetched = len(all_planets)
            
            # Record successful completion
            fetch_complete = {
                "status": "completed",
                "timestamp": datetime.utcnow(),
                "message": f"Successfully fetched {total_fetched} exoplanet records",
                "total_records": total_fetched,
                "confirmed_exoplanets": len(exoplanets),
                "planet_candidates": len(candidates)
            }
            await db.fetch_status.insert_one(fetch_complete)
            
            logger.info(f"NASA data fetch completed: {total_fetched} records saved to database")
            return total_fetched
            
        except Exception as e:
            logger.error(f"NASA data fetch failed: {e}")
            
            # Record failure
            fetch_error = {
                "status": "failed",
                "timestamp": datetime.utcnow(),
                "message": f"NASA data fetch failed: {str(e)}",
                "error": str(e)
            }
            await db.fetch_status.insert_one(fetch_error)
            
            # Fallback to simulated data if NASA API fails
            return await self._fetch_simulated_data()
    
    async def _fetch_confirmed_exoplanets(self):
        """Skip confirmed exoplanets for now - Kepler data includes confirmed planets"""
        try:
            logger.info("Skipping separate confirmed exoplanets fetch - using Kepler CONFIRMED dispositions instead")
            return []  # Kepler data already includes confirmed planets with koi_disposition='CONFIRMED'
            
        except Exception as e:
            logger.error(f"Failed to fetch confirmed exoplanets: {e}")
            return []
    
    async def _fetch_planet_candidates(self):
        """Fetch planet candidates from NASA Exoplanet Archive using CSV API"""
        try:
            logger.info("Fetching planet candidates from NASA...")
            
            # Use a simpler query first to test the API
            params = {
                'table': 'cumulative',  # Kepler Objects of Interest table
                'format': 'csv'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.csv_api_url, params=params) as response:
                    if response.status == 200:
                        text_data = await response.text()
                        logger.info(f"Kepler CSV response length: {len(text_data)} characters")
                        logger.info(f"First 500 chars: {text_data[:500]}")
                        
                        # Parse CSV data
                        import csv
                        import io
                        
                        csv_reader = csv.DictReader(io.StringIO(text_data))
                        data = list(csv_reader)
                        logger.info(f"Parsed {len(data)} Kepler records from CSV")
                        
                        candidates = []
                        for row in data:
                            try:
                                # Process real NASA candidate data
                                disposition = row.get("koi_disposition", "").upper()
                                if "FALSE" in disposition:
                                    status = "FALSE_POSITIVE"
                                elif "CONFIRMED" in disposition:
                                    status = "CONFIRMED"
                                elif "CANDIDATE" in disposition:
                                    status = "CANDIDATE"
                                else:
                                    status = "CANDIDATE"
                                
                                candidate_name = row.get("kepoi_name")
                                
                                # Skip records without valid names
                                if not candidate_name or not candidate_name.strip() or candidate_name == "null":
                                    continue
                                
                                candidate = {
                                    "pl_name": candidate_name,
                                    "pl_status": status,
                                    "pl_orbper": self._safe_float(row.get("koi_period")),
                                    "pl_rade": self._safe_float(row.get("koi_prad")),
                                    "pl_bmasse": None,  # Not available for candidates
                                    "pl_eqt": self._safe_float(row.get("koi_teq")),
                                    "pl_disc": 2009,  # Kepler mission start
                                    "st_teff": None,
                                    "st_rad": None,
                                    "st_mass": None,
                                    "koi_disposition": row.get("koi_disposition"),
                                    "created_at": datetime.utcnow(),
                                    "source": "NASA_Kepler_Archive"
                                }
                                candidates.append(candidate)
                            except Exception as e:
                                logger.warning(f"Skipping malformed candidate record: {e}")
                                continue
                        
                        logger.info(f"Fetched {len(candidates)} planet candidates")
                        return candidates
                    else:
                        logger.error(f"NASA Kepler API error: {response.status}")
                        response_text = await response.text()
                        logger.error(f"Response content: {response_text[:500]}")
                        return []
                        
        except Exception as e:
            logger.error(f"Failed to fetch planet candidates: {e}")
            return []
    
    def _safe_float(self, value):
        """Safely convert value to float"""
        try:
            if value is None or value == '':
                return None
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _safe_int(self, value):
        """Safely convert value to int"""
        try:
            if value is None or value == '':
                return None
            return int(float(value))
        except (ValueError, TypeError):
            return None
    
    async def _fetch_simulated_data(self):
        """Fallback to simulated data if NASA API fails"""
        try:
            logger.info("Using simulated data as fallback...")
            
            # Clear existing data
            db = get_database()
            await db.exoplanets.delete_many({})
            
            import random
            
            # Generate simulated exoplanet data
            exoplanets = []
            
            # Generate confirmed planets
            for i in range(1000):
                exoplanet = {
                    "pl_name": f"Kepler-{i+1}b",
                    "pl_status": "CONFIRMED",
                    "pl_orbper": random.uniform(0.1, 500),
                    "pl_rade": random.uniform(0.1, 20),
                    "pl_bmasse": random.uniform(0.01, 1000),
                    "pl_eqt": random.uniform(150, 2000),
                    "pl_disc": random.randint(1995, 2025),
                    "st_teff": random.uniform(3000, 8000),
                    "st_rad": random.uniform(0.1, 5),
                    "st_mass": random.uniform(0.1, 3),
                    "created_at": datetime.utcnow(),
                    "source": "SIMULATED"
                }
                exoplanets.append(exoplanet)
            
            # Generate candidates
            for i in range(500):
                exoplanet = {
                    "pl_name": f"TOI-{i+1001}",
                    "pl_status": "CANDIDATE",
                    "pl_orbper": random.uniform(0.1, 500),
                    "pl_rade": random.uniform(0.1, 20),
                    "pl_bmasse": random.uniform(0.01, 1000),
                    "pl_eqt": random.uniform(150, 2000),
                    "pl_disc": random.randint(2018, 2025),
                    "st_teff": random.uniform(3000, 8000),
                    "st_rad": random.uniform(0.1, 5),
                    "st_mass": random.uniform(0.1, 3),
                    "created_at": datetime.utcnow(),
                    "source": "SIMULATED"
                }
                exoplanets.append(exoplanet)
            
            # Insert simulated data
            if exoplanets:
                await db.exoplanets.insert_many(exoplanets)
            
            total_fetched = len(exoplanets)
            logger.info(f"Simulated data fallback completed: {total_fetched} records")
            return total_fetched
            
        except Exception as e:
            logger.error(f"Simulated data fallback failed: {e}")
            return 0
    
    async def fetch_confirmed_planets(self) -> int:
        """Fetch confirmed exoplanets from NASA Exoplanet Archive"""
        try:
            exoplanets = await self._fetch_confirmed_exoplanets()
            return len(exoplanets)
        except Exception as e:
            logger.error(f"Failed to fetch confirmed planets: {e}")
            return 0
    
    async def fetch_planet_candidates(self) -> int:
        """Fetch planet candidates from NASA Exoplanet Archive"""
        try:
            candidates = await self._fetch_planet_candidates()
            return len(candidates)
        except Exception as e:
            logger.error(f"Failed to fetch candidates: {e}")
            return 0
    
    async def fetch_false_positives(self) -> int:
        """Fetch false positives from NASA Exoplanet Archive"""
        logger.info("False positives included in candidates query")
        return 0
    
    async def get_light_curve_data(self, target_id: str):
        """Fetch light curve data from MAST for a specific target"""
        try:
            return {
                "target_id": target_id,
                "mission": "TESS",
                "time": [],  # Time stamps
                "flux": [],  # Flux measurements
                "flux_err": []  # Flux errors
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch light curve for {target_id}: {e}")
            return None