from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "NASA Exoplanet Detection Platform"
    VERSION: str = "1.0.0"
    
    # Database
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "exoplanet_db"
    
    # Security
    SECRET_KEY: str = "nasa-space-apps-challenge-2025-exoplanet-detection"
    JWT_SECRET_KEY: str = "nasa-space-apps-challenge-2025-jwt-secret"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    ENCRYPTION_KEY: str = ""
    SESSION_SECRET: str = ""
    
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # CORS
    ALLOWED_HOSTS: List[str] = ["*"]
    CORS_ORIGINS: str = "http://localhost:3000"
    
    # NASA APIs
    NASA_EXOPLANET_ARCHIVE_URL: str = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    NASA_MAST_URL: str = "https://mast.stsci.edu/api/v0.1"
    NASA_API_KEY: str = ""
    NASA_BASE_URL: str = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    
    # AI/ML APIs
    GEMINI_API_KEY: str = ""
    
    # ML Models
    MODEL_PATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
    TABULAR_MODEL_PATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "tabular_model.pkl")
    CNN_MODEL_PATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "cnn_model.h5")
    
    # Data Processing
    DATA_PATH: str = "./data"
    BATCH_SIZE: int = 32
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    
    # Scheduling
    FETCH_SCHEDULE_HOURS: int = 24  # Fetch NASA data every 24 hours
    
    # Redis for Celery
    REDIS_URL: str = "redis://localhost:6379"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()