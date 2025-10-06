"""
Minimal API fallback when models are not available
This provides basic endpoints that work without ML models
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any
import time
import logging

logger = logging.getLogger(__name__)

# Create a minimal router
router = APIRouter()

class PredictionResponse(BaseModel):
    predicted_return_probability: float
    confidence_score: float
    recommendation: str
    processing_time_ms: float
    model_version: str
    fallback_mode: bool = True

@router.get("/")
def minimal_root():
    """Minimal root endpoint"""
    return {
        "message": "E-commerce Return Prediction API - Minimal Mode",
        "status": "running",
        "mode": "fallback",
        "available_endpoints": ["/health", "/predict/single"]
    }

@router.get("/health")
def minimal_health():
    """Minimal health check"""
    return {
        "status": "healthy",
        "service": "return-prediction-api",
        "mode": "minimal",
        "timestamp": int(time.time())
    }

@router.post("/predict/single")
def minimal_predict(order_data: Dict[str, Any]) -> PredictionResponse:
    """
    Minimal prediction endpoint using simple heuristics
    Works without ML models as a fallback
    """
    start_time = time.time()
    
    try:
        # Simple heuristic-based prediction
        price = float(order_data.get('price', 0))
        category = order_data.get('product_category', '').lower()
        payment_method = order_data.get('payment_method', '').lower()
        
        # Basic heuristic rules
        risk_score = 0.3  # Base risk
        
        # Price-based adjustments
        if price > 500:
            risk_score += 0.2
        elif price < 50:
            risk_score += 0.1
        
        # Category-based adjustments
        high_risk_categories = ['electronics', 'clothing', 'beauty']
        if category in high_risk_categories:
            risk_score += 0.15
        
        # Payment method adjustments
        if payment_method == 'cash_on_delivery':
            risk_score += 0.1
        
        # Ensure score is between 0 and 1
        risk_score = min(max(risk_score, 0.0), 1.0)
        
        # Generate recommendation
        if risk_score < 0.3:
            recommendation = "Low return risk - proceed with standard processing"
        elif risk_score < 0.6:
            recommendation = "Medium return risk - consider additional verification"
        else:
            recommendation = "High return risk - recommend manual review"
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            predicted_return_probability=risk_score,
            confidence_score=0.7,  # Fixed confidence for heuristic
            recommendation=recommendation,
            processing_time_ms=processing_time,
            model_version="heuristic-v1.0",
            fallback_mode=True
        )
        
    except Exception as e:
        logger.error(f"Minimal prediction failed: {str(e)}")
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            predicted_return_probability=0.5,
            confidence_score=0.1,
            recommendation="Unable to assess - manual review recommended",
            processing_time_ms=processing_time,
            model_version="heuristic-v1.0",
            fallback_mode=True
        )
