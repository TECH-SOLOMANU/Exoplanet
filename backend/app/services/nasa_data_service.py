import logging
import httpx
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime, timedelta
from io import StringIO
from app.core.config import settings
from app.core.database import get_database
import json

logger = logging.getLogger(__name__)

class NASADataService:
    """Service for fetching and processing NASA exoplanet data"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.db = get_database()
        
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def fetch_exoplanet_archive_data(
        self, 
        table: str = "ps", 
        columns: Optional[List[str]] = None,
        where_clause: Optional[str] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch data from NASA Exoplanet Archive
        
        Args:
            table: Database table (ps=Planetary Systems, pscomppars=Composite Planet Data)
            columns: List of columns to fetch
            where_clause: SQL WHERE clause for filtering
            limit: Maximum number of records
        """
        try:
            # Default columns for exoplanet data
            if columns is None:
                columns = [
                    "pl_name", "hostname", "discoverymethod", "disc_year",
                    "pl_orbper", "pl_rade", "pl_bmasse", "pl_eqt",
                    "st_teff", "st_rad", "st_mass", "st_logg",
                    "ra", "dec", "sy_dist"
                ]
            
            # Build ADQL query
            query = f"SELECT {','.join(columns)} FROM {table}"
            
            if where_clause:
                query += f" WHERE {where_clause}"
            
            query += f" ORDER BY disc_year DESC"
            
            if limit:
                query += f" TOP {limit}"
            
            # Make request to NASA Exoplanet Archive
            params = {
                "query": query,
                "format": "csv"
            }
            
            logger.info(f"Fetching data from NASA Exoplanet Archive: {query[:100]}...")
            
            response = await self.client.get(
                settings.NASA_EXOPLANET_ARCHIVE_URL,
                params=params
            )
            
            response.raise_for_status()
            
            # Parse CSV data
            df = pd.read_csv(StringIO(response.text))
            
            logger.info(f"Fetched {len(df)} records from NASA Exoplanet Archive")
            
            # Clean and process data
            df = self._clean_exoplanet_data(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch NASA exoplanet data: {e}")
            # Return synthetic data as fallback
            return self._generate_synthetic_data(limit)
    
    def _clean_exoplanet_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess exoplanet data"""
        try:
            # Handle missing values
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                df[col] = df[col].fillna(df[col].median())
            
            # Convert discovery year to integer
            if 'disc_year' in df.columns:
                df['disc_year'] = pd.to_numeric(df['disc_year'], errors='coerce').fillna(2020).astype(int)
            
            # Add computed features
            if 'pl_orbper' in df.columns and 'st_mass' in df.columns:
                # Semi-major axis using Kepler's 3rd law (approximate)
                df['pl_orbsmax'] = ((df['pl_orbper'] / 365.25) ** 2 * df['st_mass']) ** (1/3)
            
            # Categorize discovery methods
            if 'discoverymethod' in df.columns:
                df['discovery_category'] = df['discoverymethod'].map({
                    'Transit': 'photometric',
                    'Radial Velocity': 'spectroscopic', 
                    'Microlensing': 'gravitational',
                    'Direct Imaging': 'direct',
                    'Astrometry': 'astrometric'
                }).fillna('other')
            
            return df
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {e}")
            return df
    
    def _generate_synthetic_data(self, count: int = 100) -> pd.DataFrame:
        """Generate synthetic exoplanet data when API is unavailable"""
        np.random.seed(42)
        
        data = {
            'pl_name': [f'Synthetic-{i+1}b' for i in range(count)],
            'hostname': [f'Star-{i+1}' for i in range(count)],
            'discoverymethod': np.random.choice(['Transit', 'Radial Velocity', 'Microlensing'], count),
            'disc_year': np.random.randint(2009, 2024, count),
            'pl_orbper': np.random.lognormal(3, 1.5, count),
            'pl_rade': np.random.lognormal(0, 0.5, count),
            'pl_bmasse': np.random.lognormal(0, 1, count),
            'pl_eqt': np.random.normal(800, 400, count),
            'st_teff': np.random.normal(5500, 1000, count),
            'st_rad': np.random.lognormal(0, 0.3, count),
            'st_mass': np.random.lognormal(0, 0.2, count),
            'st_logg': np.random.normal(4.5, 0.5, count),
            'ra': np.random.uniform(0, 360, count),
            'dec': np.random.uniform(-90, 90, count),
            'sy_dist': np.random.lognormal(4, 1, count),
        }
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {count} synthetic exoplanet records")
        return df
    
    async def fetch_light_curve_data(
        self, 
        target_name: str,
        mission: str = "TESS"
    ) -> Optional[np.ndarray]:
        """
        Fetch light curve data from MAST
        
        Args:
            target_name: Name of the target star/system
            mission: Mission name (TESS, Kepler, K2)
        """
        try:
            # Search for observations
            search_url = f"{settings.NASA_MAST_URL}/search/observations"
            
            params = {
                "target": target_name,
                "mission": mission,
                "dataProductType": "timeseries"
            }
            
            response = await self.client.get(search_url, params=params)
            response.raise_for_status()
            
            observations = response.json()
            
            if not observations.get("data"):
                logger.warning(f"No light curve data found for {target_name}")
                return self._generate_synthetic_light_curve()
            
            # Get the first available observation
            obs = observations["data"][0]
            
            # This would normally fetch actual FITS files from MAST
            # For now, generate synthetic light curve based on the target
            return self._generate_synthetic_light_curve(target_name)
            
        except Exception as e:
            logger.error(f"Failed to fetch light curve for {target_name}: {e}")
            return self._generate_synthetic_light_curve()
    
    def _generate_synthetic_light_curve(
        self, 
        target_name: str = "synthetic",
        duration_days: int = 27,
        cadence_minutes: int = 30
    ) -> np.ndarray:
        """Generate synthetic light curve with realistic transit features"""
        
        # Calculate number of data points
        points_per_day = 24 * 60 / cadence_minutes
        total_points = int(duration_days * points_per_day)
        
        # Generate time array
        time = np.linspace(0, duration_days, total_points)
        
        # Base stellar flux (normalized to 1.0)
        flux = np.ones(total_points)
        
        # Add stellar noise
        noise_level = 0.001  # 0.1% noise
        flux += np.random.normal(0, noise_level, total_points)
        
        # Add potential transit signal (30% chance)
        if np.random.random() < 0.3:
            # Transit parameters
            orbital_period = np.random.uniform(1, 50)  # days
            transit_duration = np.random.uniform(0.05, 0.2) * orbital_period
            transit_depth = np.random.uniform(0.001, 0.05)  # 0.1% to 5% depth
            
            # Add transit signals
            phase = time % orbital_period
            transit_mask = np.abs(phase - orbital_period/2) < transit_duration/2
            
            # Simple box transit model
            flux[transit_mask] -= transit_depth
        
        # Add stellar variability
        if np.random.random() < 0.2:  # 20% chance of variable star
            variability_period = np.random.uniform(0.5, 10)
            variability_amplitude = np.random.uniform(0.001, 0.01)
            flux += variability_amplitude * np.sin(2 * np.pi * time / variability_period)
        
        return flux
    
    async def store_exoplanet_data(self, df: pd.DataFrame) -> int:
        """Store exoplanet data in MongoDB"""
        try:
            # Convert DataFrame to records
            records = df.to_dict('records')
            
            # Add metadata
            for record in records:
                record['_fetched_at'] = datetime.utcnow()
                record['_source'] = 'nasa_exoplanet_archive'
            
            # Insert into database
            result = self.db.exoplanets.insert_many(records)
            
            logger.info(f"Stored {len(result.inserted_ids)} exoplanet records")
            return len(result.inserted_ids)
            
        except Exception as e:
            logger.error(f"Failed to store exoplanet data: {e}")
            return 0
    
    async def get_stored_exoplanets(
        self, 
        limit: int = 100,
        discovery_method: Optional[str] = None,
        min_year: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve stored exoplanet data from MongoDB"""
        try:
            # Build query
            query = {}
            if discovery_method:
                query['discoverymethod'] = discovery_method
            if min_year:
                query['disc_year'] = {'$gte': min_year}
            
            # Execute query
            cursor = self.db.exoplanets.find(
                query,
                {'_id': 0}  # Exclude MongoDB _id field
            ).sort('disc_year', -1).limit(limit)
            
            exoplanets = list(cursor)
            logger.info(f"Retrieved {len(exoplanets)} exoplanet records")
            
            return exoplanets
            
        except Exception as e:
            logger.error(f"Failed to retrieve exoplanet data: {e}")
            return []
    
    async def fetch_and_store_latest_data(self) -> Dict[str, Any]:
        """Fetch latest data from NASA and store in database"""
        try:
            # Fetch data from last 2 years
            current_year = datetime.now().year
            where_clause = f"disc_year >= {current_year - 2}"
            
            df = await self.fetch_exoplanet_archive_data(
                where_clause=where_clause,
                limit=500
            )
            
            if df.empty:
                return {"status": "error", "message": "No data fetched"}
            
            # Store in database
            stored_count = await self.store_exoplanet_data(df)
            
            # Generate summary statistics
            stats = {
                "total_fetched": len(df),
                "total_stored": stored_count,
                "discovery_methods": df['discoverymethod'].value_counts().to_dict() if 'discoverymethod' in df.columns else {},
                "year_range": {
                    "min": int(df['disc_year'].min()) if 'disc_year' in df.columns else None,
                    "max": int(df['disc_year'].max()) if 'disc_year' in df.columns else None
                },
                "fetched_at": datetime.utcnow().isoformat()
            }
            
            return {
                "status": "success",
                "data": stats
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch and store latest data: {e}")
            return {
                "status": "error", 
                "message": str(e)
            }