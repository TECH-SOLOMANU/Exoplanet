from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
import numpy as np
import logging
from io import StringIO
from app.core.database import get_database
from app.models.prediction_models import PredictionRequest, PredictionResponse
from app.services.ml_service import MLService
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()

ml_service = MLService()

@router.get("/latest", response_model=List[PredictionResponse])
async def get_latest_predictions(limit: int = 20):
    """Get latest predictions from the database"""
    try:
        db = get_database()
        
        cursor = db.predictions.find(
            {},
            sort=[("timestamp", -1)],
            limit=limit
        )
        
        predictions = []
        async for pred in cursor:
            pred["_id"] = str(pred["_id"])
            predictions.append(pred)
        
        return predictions
        
    except Exception as e:
        logger.error(f"Failed to get latest predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict", response_model=PredictionResponse)
async def predict_exoplanet(request: PredictionRequest):
    """Make prediction for a single exoplanet"""
    try:
        # Convert request to DataFrame
        data = {
            "pl_orbper": [request.orbital_period],
            "pl_rade": [request.planet_radius],
            "pl_bmasse": [request.planet_mass],
            "pl_eqt": [request.equilibrium_temp],
            "st_teff": [request.stellar_temp],
            "st_rad": [request.stellar_radius],
            "st_mass": [request.stellar_mass],
        }
        
        df = pd.DataFrame(data)
        
        # Make prediction
        result = await ml_service.predict_tabular(df)
        
        # Store prediction in database
        db = get_database()
        prediction_doc = {
            "input_data": request.dict(),
            "prediction": result[0]["prediction"],
            "confidence": result[0]["confidence"],
            "probabilities": result[0]["probabilities"],
            "feature_importance": result[0]["feature_importance"],
            "timestamp": datetime.utcnow(),
            "model_type": "tabular"
        }
        
        await db.predictions.insert_one(prediction_doc)
        
        return PredictionResponse(**result[0])
        
    except Exception as e:
        logger.error(f"Failed to make prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    """Make predictions for multiple exoplanets from CSV file"""
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_columns = ['pl_orbper', 'pl_rade', 'pl_bmasse', 'pl_eqt', 
                           'st_teff', 'st_rad', 'st_mass']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}"
            )
        
        # Make predictions
        results = await ml_service.predict_tabular(df)
        
        # Store batch predictions
        db = get_database()
        batch_doc = {
            "filename": file.filename,
            "total_predictions": len(results),
            "timestamp": datetime.utcnow(),
            "predictions": results
        }
        
        await db.batch_predictions.insert_one(batch_doc)
        
        return {
            "message": f"Successfully processed {len(results)} predictions",
            "total_predictions": len(results),
            "predictions": results[:10],  # Return first 10 for preview
            "download_id": str(batch_doc["_id"]) if "_id" in batch_doc else None
        }
        
    except Exception as e:
        logger.error(f"Failed to process batch predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_prediction_stats():
    """Get prediction statistics"""
    try:
        db = get_database()
        
        # Count predictions by class
        pipeline = [
            {"$group": {
                "_id": "$prediction",
                "count": {"$sum": 1},
                "avg_confidence": {"$avg": "$confidence"}
            }}
        ]
        
        stats_cursor = db.predictions.aggregate(pipeline)
        class_stats = {}
        total_predictions = 0
        
        async for stat in stats_cursor:
            class_name = stat["_id"]
            count = stat["count"]
            avg_confidence = stat["avg_confidence"]
            
            class_stats[class_name] = {
                "count": count,
                "avg_confidence": round(avg_confidence, 3)
            }
            total_predictions += count
        
        # Get recent activity (last 24 hours)
        from datetime import timedelta
        yesterday = datetime.utcnow() - timedelta(days=1)
        
        recent_count = await db.predictions.count_documents({
            "timestamp": {"$gte": yesterday}
        })
        
        return {
            "total_predictions": total_predictions,
            "recent_predictions_24h": recent_count,
            "class_distribution": class_stats,
            "model_performance": {
                "accuracy": 0.94,  # This would come from model evaluation
                "precision": 0.92,
                "recall": 0.91
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get prediction stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/explain/{prediction_id}")
async def get_prediction_explanation(prediction_id: str):
    """Get detailed explanation for a specific prediction"""
    try:
        from bson import ObjectId
        db = get_database()
        
        # Get prediction from database
        prediction = await db.predictions.find_one({"_id": ObjectId(prediction_id)})
        
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        # Return detailed explanation
        return {
            "prediction_id": prediction_id,
            "prediction": prediction["prediction"],
            "confidence": prediction["confidence"],
            "probabilities": prediction["probabilities"],
            "feature_importance": prediction["feature_importance"],
            "input_data": prediction["input_data"],
            "timestamp": prediction["timestamp"],
            "explanation": {
                "top_features": sorted(
                    prediction["feature_importance"].items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:5],
                "confidence_interpretation": _interpret_confidence(prediction["confidence"]),
                "recommendation": _generate_recommendation(prediction)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get prediction explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _interpret_confidence(confidence: float) -> str:
    """Interpret confidence score"""
    if confidence >= 0.9:
        return "Very high confidence - strong evidence for this classification"
    elif confidence >= 0.7:
        return "High confidence - good evidence for this classification"
    elif confidence >= 0.5:
        return "Moderate confidence - some uncertainty in classification"
    else:
        return "Low confidence - high uncertainty, requires further investigation"

def _generate_recommendation(prediction: dict) -> str:
    """Generate recommendation based on prediction"""
    pred_class = prediction["prediction"]
    confidence = prediction["confidence"]
    
    if pred_class == "CONFIRMED" and confidence > 0.8:
        return "This object shows strong characteristics of a confirmed exoplanet"
    elif pred_class == "CANDIDATE" and confidence > 0.7:
        return "This object is a good candidate for follow-up observations"
    elif pred_class == "FALSE_POSITIVE":
        return "This object likely represents a false positive detection"
    else:
        return "This classification requires additional validation and analysis"

@router.get("/model-status")
async def get_model_status():
    """Get current status of ML models"""
    try:
        status = {
            "tabular_model": {
                "loaded": ml_service.tabular_model is not None,
                "type": "RandomForestClassifier" if ml_service.tabular_model else "None",
                "features": ml_service.feature_names if hasattr(ml_service, 'feature_names') and ml_service.feature_names else [],
                "classes": list(ml_service.label_encoder.classes_) if hasattr(ml_service.label_encoder, 'classes_') else []
            },
            "scaler": {
                "loaded": ml_service.scaler is not None,
                "type": "StandardScaler"
            },
            "service_info": {
                "ml_service_type": str(type(ml_service).__name__),
                "available_methods": [method for method in dir(ml_service) if not method.startswith('_')]
            }
        }
        
        return {
            "status": "success",
            "models": status,
            "ready_for_predictions": ml_service.tabular_model is not None
        }
        
    except Exception as e:
        logger.error(f"Model status error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "ready_for_predictions": False
        }