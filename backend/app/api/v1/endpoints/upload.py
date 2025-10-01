from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Optional
import pandas as pd
from io import StringIO
import logging

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
async def upload_light_curve(file: UploadFile = File(...)):
    """Upload light curve file (FITS or CSV)"""
    try:
        if not (file.filename.endswith('.fits') or file.filename.endswith('.csv')):
            raise HTTPException(status_code=400, detail="Only FITS or CSV files are supported")
        
        return {
            "message": "Light curve file uploaded successfully",
            "filename": file.filename,
            "note": "Light curve processing will be implemented with real data"
        }
        
    except Exception as e:
        logger.error(f"Light curve upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))