from fastapi import APIRouter
from app.api.v1.endpoints import nasa_data, predictions, upload, models, light_curves

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    nasa_data.router, 
    prefix="/nasa", 
    tags=["NASA Data"]
)

api_router.include_router(
    light_curves.router,
    prefix="/nasa",
    tags=["Light Curves"]
)

api_router.include_router(
    predictions.router, 
    prefix="/predictions", 
    tags=["Predictions"]
)

api_router.include_router(
    upload.router, 
    prefix="/upload", 
    tags=["Data Upload"]
)

api_router.include_router(
    models.router, 
    prefix="/models", 
    tags=["ML Models"]
)