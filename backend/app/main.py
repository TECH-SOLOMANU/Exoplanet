from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from app.api.v1.router import api_router
from app.core.config import settings
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NASA Exoplanet Detection API",
    description="AI/ML powered exoplanet detection system using NASA datasets",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    """Initialize application"""
    logger.info("Starting NASA Exoplanet Detection Platform...")
    
    # Try to connect to database if available
    try:
        if os.getenv("DEVELOPMENT") != "True":
            from app.core.database import connect_to_mongo
            await connect_to_mongo()
            logger.info("Database connected successfully")
        else:
            logger.info("Running in development mode - database mocked")
    except Exception as e:
        logger.warning(f"Database connection failed, running with mocked data: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources"""
    logger.info("Shutting down...")
    try:
        if os.getenv("DEVELOPMENT") != "True":
            from app.core.database import close_mongo_connection
            await close_mongo_connection()
    except Exception:
        pass

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "NASA Exoplanet Detection Platform API",
        "version": "1.0.0",
        "challenge": "NASA Space Apps Challenge 2025",
        "docs": "/docs",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "exoplanet-detection-api"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )