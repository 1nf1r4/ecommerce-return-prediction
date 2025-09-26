from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.prediction import prediction_agent
import uvicorn
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the FastAPI app from prediction agent
app = prediction_agent.app

# Add additional routes for the main application
@app.get("/api/status")
async def api_status():
    """Get API status and statistics"""
    return {
        "api_name": "E-commerce Return Prediction API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "health": "/health",
            "logs": "/predictions/logs",
            "model_info": "/model/info"
        },
        "features": {
            "feature_engineering": True,
            "real_time_prediction": True,
            "batch_processing": True,
            "logging": True,
            "error_handling": True
        }
    }

@app.get("/api/docs")
async def api_documentation():
    """Get API documentation links"""
    return {
        "documentation": "/docs",
        "redoc": "/redoc",
        "openapi_schema": "/openapi.json",
        "description": "E-commerce Return Prediction API with Feature Engineering and Prediction capabilities"
    }

if __name__ == "__main__":
    import sys
    
    # Check if running in development or production mode
    reload_mode = "--dev" in sys.argv or "--development" in sys.argv
    
    logger.info(f"Starting E-commerce Return Prediction API ({'Development' if reload_mode else 'Production'} mode)")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=reload_mode,  # Enable reload only in development
        log_level="info"
    )
