from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from typing import Optional
import pandas as pd
from io import StringIO
import logging
from app.services.light_curve_service import LightCurveService
from app.core.database import Database, get_database

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload CSV file for batch prediction"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        
        return {
            "message": "CSV uploaded successfully",
            "filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns)
        }
        
    except Exception as e:
        logger.error(f"CSV upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/light-curve")
async def upload_light_curve(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Database = Depends(get_database)
):
    """Upload and process light curve file (CSV, JSON, TXT)"""
    try:
        # Validate file format
        allowed_extensions = ['.csv', '.json', '.txt', '.dat']
        file_extension = None
        for ext in allowed_extensions:
            if file.filename.lower().endswith(ext):
                file_extension = ext[1:]  # Remove the dot
                break
        
        if not file_extension:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Read and validate file content
        file_content = await file.read()
        
        # Check file size (max 50MB)
        if len(file_content) > 50 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum size is 50MB."
            )
        
        if len(file_content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded."
            )
        
        # Initialize service with database connection
        light_curve_service = LightCurveService(db)
        
        try:
            # Process the light curve data
            result = await light_curve_service.process_uploaded_file(
                file_content, file.filename, file_extension
            )
            
            # Schedule background analysis
            background_tasks.add_task(
                light_curve_service.detect_transits,
                result["light_curve_id"]
            )
            
            return {
                "message": "Light curve uploaded and processing initiated",
                "light_curve_id": result["light_curve_id"],
                "filename": result["filename"],
                "data_points": result["data_points"],
                "time_span_days": result["time_span_days"],
                "stats": result["stats"],
                "preview_data": result["preview_data"],
                "analysis_status": "processing",
                "note": "Transit analysis running in background"
            }
            
        except ValueError as ve:
            # Handle specific parsing/data errors with user-friendly messages
            error_message = str(ve)
            if "Failed to parse CSV" in error_message:
                raise HTTPException(
                    status_code=400,
                    detail=f"CSV parsing error: Please ensure your file has properly formatted time and flux columns with numeric data. Details: {error_message}"
                )
            elif "No data found" in error_message:
                raise HTTPException(
                    status_code=400,
                    detail="No valid data found in file. Please check that your file contains time-series photometry data."
                )
            elif "Insufficient data points" in error_message:
                raise HTTPException(
                    status_code=400,
                    detail="File contains too few data points for analysis. Please provide at least 10 measurements."
                )
            elif "Could not identify" in error_message:
                raise HTTPException(
                    status_code=400,
                    detail="Unable to identify time and flux columns. Please ensure your file has at least two numeric columns representing time and brightness/flux measurements."
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Data processing error: {error_message}"
                )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Light curve upload failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error during file processing: {str(e)}"
        )

@router.get("/light-curve/{light_curve_id}/analysis")
async def get_light_curve_analysis(light_curve_id: str, db: Database = Depends(get_database)):
    """Get analysis results for uploaded light curve"""
    try:
        light_curve_service = LightCurveService(db)
        result = await light_curve_service.get_light_curve_analysis(light_curve_id)
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/light-curve/{light_curve_id}/reanalyze")
async def reanalyze_light_curve(
    light_curve_id: str,
    background_tasks: BackgroundTasks,
    db: Database = Depends(get_database)
):
    """Trigger reanalysis of light curve"""
    try:
        light_curve_service = LightCurveService(db)
        # Schedule background analysis
        background_tasks.add_task(
            light_curve_service.detect_transits,
            light_curve_id
        )
        
        return {
            "message": "Reanalysis initiated",
            "light_curve_id": light_curve_id,
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Reanalysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/light-curve/{light_curve_id}/advanced-analysis")
async def run_advanced_analysis(
    light_curve_id: str,
    background_tasks: BackgroundTasks,
    db: Database = Depends(get_database)
):
    """Run comprehensive advanced analysis including ML models and statistical tests"""
    try:
        light_curve_service = LightCurveService(db)
        
        # Verify light curve exists
        from bson import ObjectId
        light_curve = await db.light_curves.find_one({"_id": ObjectId(light_curve_id)})
        if not light_curve:
            raise HTTPException(status_code=404, detail="Light curve not found")
        
        # Schedule advanced analysis in background
        background_tasks.add_task(
            light_curve_service.run_advanced_analysis,
            light_curve_id
        )
        
        # Update status to indicate advanced analysis is running
        await db.light_curves.update_one(
            {"_id": ObjectId(light_curve_id)},
            {"$set": {"advanced_analysis_status": "processing"}}
        )
        
        return {
            "message": "Advanced analysis initiated",
            "light_curve_id": light_curve_id,
            "status": "processing",
            "estimated_completion": "2-3 minutes",
            "analysis_includes": [
                "Machine Learning Classification",
                "Periodicity Analysis (Lomb-Scargle)",
                "Advanced Statistical Tests",
                "Transit Model Fitting",
                "Stellar Variability Analysis",
                "Data Quality Assessment"
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Advanced analysis initiation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/light-curve/{light_curve_id}/advanced-analysis")
async def get_advanced_analysis_results(
    light_curve_id: str,
    db: Database = Depends(get_database)
):
    """Get advanced analysis results"""
    try:
        from bson import ObjectId
        light_curve = await db.light_curves.find_one({"_id": ObjectId(light_curve_id)})
        
        if not light_curve:
            raise HTTPException(status_code=404, detail="Light curve not found")
        
        # Check if advanced analysis has been run
        if "advanced_analysis" not in light_curve:
            return {
                "light_curve_id": light_curve_id,
                "status": "not_started",
                "message": "Advanced analysis has not been initiated. Use POST /advanced-analysis to start."
            }
        
        status = light_curve.get("advanced_analysis_status", "unknown")
        
        if status == "processing":
            return {
                "light_curve_id": light_curve_id,
                "status": "processing",
                "message": "Advanced analysis is currently running. Please check back in a few minutes."
            }
        elif status == "failed":
            return {
                "light_curve_id": light_curve_id,
                "status": "failed",
                "error": light_curve.get("advanced_analysis_error", "Unknown error"),
                "message": "Advanced analysis failed. You can try running it again."
            }
        elif status == "completed":
            # Return full advanced analysis results
            results = light_curve["advanced_analysis"]
            
            # Add metadata
            results["light_curve_id"] = light_curve_id
            results["status"] = "completed"
            results["basic_info"] = {
                "filename": light_curve.get("filename"),
                "uploaded_at": light_curve.get("uploaded_at"),
                "data_points": light_curve.get("data_points"),
                "time_span_days": light_curve.get("time_span_days")
            }
            
            return results
        else:
            return {
                "light_curve_id": light_curve_id,
                "status": status,
                "message": f"Unknown analysis status: {status}"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get advanced analysis results: {e}")
        raise HTTPException(status_code=500, detail=str(e))