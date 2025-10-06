"""
Order Processing API Endpoints
Purpose: Handle HTTP requests for order processing and validation
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

# Import the agents
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.order_processing import get_order_processing_agent, OrderProcessingAgent
from agents.model_inference import get_inference_agent, ModelInferenceAgent
from agents.feature_engineering import get_feature_engineering_agent, FeatureEngineeringAgent

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/orders", tags=["order-processing"])

# Request/Response models
class OrderProcessingRequest(BaseModel):
    """Request model for order processing"""
    order_id: Optional[str] = Field(None, description="Optional order ID")
    price: float = Field(..., gt=0, description="Product price in USD")
    quantity: int = Field(..., gt=0, description="Order quantity")
    product_category: str = Field(..., description="Product category")
    gender: str = Field(..., description="Customer gender")
    payment_method: str = Field(..., description="Payment method")
    age: int = Field(..., ge=18, le=100, description="Customer age")
    location: str = Field(..., description="Customer location")
    discount_applied: Optional[float] = Field(0.0, ge=0, le=100, description="Discount percentage")
    shipping_method: Optional[str] = Field("Standard", description="Shipping method")
    order_date: Optional[str] = Field(None, description="Order date (YYYY-MM-DD)")

class BatchOrderProcessingRequest(BaseModel):
    """Request model for batch order processing"""
    orders: List[OrderProcessingRequest] = Field(..., max_items=50, description="List of orders (max 50)")

class OrderProcessingResponse(BaseModel):
    """Response model for order processing"""
    success: bool
    order_id: str
    prediction: Optional[Dict[str, Any]] = None
    features: Optional[Dict[str, Any]] = None
    risk_level: Optional[str] = None
    confidence: Optional[float] = None
    recommendations: Optional[List[str]] = None
    error: Optional[str] = None
    processing_timestamp: str

class BatchOrderProcessingResponse(BaseModel):
    """Response model for batch order processing"""
    success: bool
    batch_size: int
    successful_count: int
    failed_count: int
    results: List[OrderProcessingResponse]
    processing_timestamp: str
    summary: Optional[Dict[str, Any]] = None

# Dependencies
def get_order_agent() -> OrderProcessingAgent:
    """Dependency to provide order processing agent"""
    return get_order_processing_agent()

def get_model_agent() -> ModelInferenceAgent:
    """Dependency to provide model inference agent"""
    return get_inference_agent()

def get_feature_agent() -> FeatureEngineeringAgent:
    """Dependency to provide feature engineering agent"""
    return get_feature_engineering_agent()

# Helper functions
def determine_risk_level(probability: float) -> str:
    """Determine risk level based on return probability"""
    if probability <= 0.3:
        return "LOW"
    elif probability <= 0.6:
        return "MEDIUM"
    else:
        return "HIGH"

def get_recommendations(risk_level: str, features: Dict[str, Any]) -> List[str]:
    """Get business recommendations based on risk level and features"""
    recommendations = []
    
    if risk_level == "HIGH":
        recommendations.append("Consider manual review before fulfillment")
        recommendations.append("Verify product description and customer expectations")
        if features.get('total_order_value', 0) > 200:
            recommendations.append("Consider requiring signature on delivery")
    elif risk_level == "MEDIUM":
        recommendations.append("Monitor order for potential issues")
        recommendations.append("Ensure quality packaging")
    else:
        recommendations.append("Process normally")
        recommendations.append("Standard fulfillment recommended")
    
    return recommendations

@router.post("/process", response_model=OrderProcessingResponse)
async def process_order(
    request: OrderProcessingRequest,
    order_agent: OrderProcessingAgent = Depends(get_order_agent),
    model_agent: ModelInferenceAgent = Depends(get_model_agent),
    feature_agent: FeatureEngineeringAgent = Depends(get_feature_agent)
):
    """
    Process a single order through validation, feature extraction, and prediction
    """
    try:
        # Convert request to dictionary
        order_data = request.model_dump()
        
        # Process order through order processing agent (basic features)
        processing_result = order_agent.process_single_order(order_data)
        
        if not processing_result['success']:
            return OrderProcessingResponse(
                success=False,
                order_id=processing_result['order_id'],
                error=processing_result['error'],
                processing_timestamp=datetime.now().isoformat()
            )
        
        # Apply advanced feature engineering
        basic_features_df = processing_result['prediction_ready_data']
        engineered_features_df = feature_agent.transform(basic_features_df)
        
        # Get prediction from model agent
        prediction_result = model_agent.predict_single(engineered_features_df)
        
        if not prediction_result['success']:
            return OrderProcessingResponse(
                success=False,
                order_id=processing_result['order_id'],
                error=f"Prediction failed: {prediction_result.get('error', 'Unknown error')}",
                processing_timestamp=datetime.now().isoformat()
            )
        
        # Extract prediction details
        prediction_data = prediction_result['prediction']
        return_probability = prediction_data.get('return_probability', 0.0)
        risk_level = determine_risk_level(return_probability)
        recommendations = get_recommendations(risk_level, processing_result['features'])
        
        # Store prediction in database
        from utils.supabase_service import get_supabase_service
        supabase_service = get_supabase_service()
        
        # Prepare prediction data for storage
        prediction_storage_data = {
            'order_id': processing_result['order_id'],
            'order_value': processing_result['features'].get('Total_Order_Value', 0),
            'return_probability': return_probability,
            'will_return': prediction_data.get('will_return', False),
            'confidence': return_probability,
            'risk_level': risk_level,
            'category': order_data.get('product_category', 'Unknown'),
            'price': order_data.get('price', 0),
            'quantity': order_data.get('quantity', 1),
            'age': order_data.get('age', 0),
            'gender': order_data.get('gender', 'Unknown'),
            'location': order_data.get('location', 'Unknown'),
            'payment_method': order_data.get('payment_method', 'Unknown'),
            'shipping_method': order_data.get('shipping_method', 'Standard'),
            'discount_applied': order_data.get('discount_applied', 0),
            'recommendations': recommendations,
            'timestamp': processing_result['processing_timestamp']
        }
        
        # Store in Supabase
        await supabase_service.store_prediction(prediction_storage_data)
        
        return OrderProcessingResponse(
            success=True,
            order_id=processing_result['order_id'],
            prediction=prediction_data,
            features=processing_result['features'],
            risk_level=risk_level,
            confidence=return_probability,
            recommendations=recommendations,
            processing_timestamp=processing_result['processing_timestamp']
        )
        
    except Exception as e:
        error_msg = f"Error processing order: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@router.post("/batch-process", response_model=BatchOrderProcessingResponse)
async def process_batch_orders(
    request: BatchOrderProcessingRequest,
    order_agent: OrderProcessingAgent = Depends(get_order_agent),
    model_agent: ModelInferenceAgent = Depends(get_model_agent),
    feature_agent: FeatureEngineeringAgent = Depends(get_feature_agent)
) -> BatchOrderProcessingResponse:
    """
    Process multiple orders in batch
    """
    try:
        # Convert requests to list of dictionaries
        orders_data = [order.model_dump() for order in request.orders]
        
        # Process batch through order processing agent
        batch_result = order_agent.process_batch_orders(orders_data)
        
        if not batch_result['success']:
            return BatchOrderProcessingResponse(
                success=False,
                batch_size=batch_result['batch_size'],
                successful_count=0,
                failed_count=batch_result['batch_size'],
                results=[],
                processing_timestamp=datetime.now().isoformat()
            )
        
        # Process predictions for successful orders
        batch_responses = []
        high_risk_count = 0
        medium_risk_count = 0
        low_risk_count = 0
        
        for result in batch_result['results']:
            if result['success']:
                try:
                    # Apply feature engineering and get prediction
                    basic_features_df = result['prediction_ready_data']
                    engineered_features_df = feature_agent.transform(basic_features_df)
                    prediction_result = model_agent.predict_single(engineered_features_df)
                    
                    if prediction_result['success']:
                        prediction_data = prediction_result['prediction']
                        return_probability = prediction_data.get('return_probability', 0.0)
                        risk_level = determine_risk_level(return_probability)
                        recommendations = get_recommendations(risk_level, result['features'])
                        
                        # Count risk levels
                        if risk_level == "HIGH":
                            high_risk_count += 1
                        elif risk_level == "MEDIUM":
                            medium_risk_count += 1
                        else:
                            low_risk_count += 1
                        
                        batch_responses.append(OrderProcessingResponse(
                            success=True,
                            order_id=result['order_id'],
                            prediction=prediction_data,
                            features=result['features'],
                            risk_level=risk_level,
                            confidence=return_probability,
                            recommendations=recommendations,
                            processing_timestamp=result['processing_timestamp']
                        ))
                    else:
                        batch_responses.append(OrderProcessingResponse(
                            success=False,
                            order_id=result['order_id'],
                            error=f"Prediction failed: {prediction_result.get('error', 'Unknown error')}",
                            processing_timestamp=result['processing_timestamp']
                        ))
                except Exception as e:
                    batch_responses.append(OrderProcessingResponse(
                        success=False,
                        order_id=result['order_id'],
                        error=f"Processing error: {str(e)}",
                        processing_timestamp=result['processing_timestamp']
                    ))
            else:
                batch_responses.append(OrderProcessingResponse(
                    success=False,
                    order_id=result['order_id'],
                    error=result['error'],
                    processing_timestamp=datetime.now().isoformat()
                ))
        
        # Create summary
        summary = {
            "risk_distribution": {
                "high_risk": high_risk_count,
                "medium_risk": medium_risk_count,
                "low_risk": low_risk_count
            },
            "total_orders_requiring_review": high_risk_count + medium_risk_count,
            "processing_success_rate": (batch_result['successful_count'] / batch_result['batch_size']) * 100
        }
        
        return BatchOrderProcessingResponse(
            success=True,
            batch_size=batch_result['batch_size'],
            successful_count=batch_result['successful_count'],
            failed_count=batch_result['failed_count'],
            results=batch_responses,
            processing_timestamp=batch_result['processing_timestamp'],
            summary=summary
        )
        
    except Exception as e:
        error_msg = f"Error processing batch orders: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@router.get("/validation-rules")
async def get_validation_rules():
    """
    Get current validation rules and constraints
    """
    return {
        "required_fields": [
            "price", "quantity", "product_category", "gender", 
            "payment_method", "age", "location"
        ],
        "optional_fields": [
            "order_id", "discount_applied", "shipping_method", "order_date"
        ],
        "constraints": {
            "price": {"type": "float", "min": 0.01, "description": "Must be greater than 0"},
            "quantity": {"type": "int", "min": 1, "description": "Must be at least 1"},
            "age": {"type": "int", "min": 18, "max": 100, "description": "Must be between 18 and 100"},
            "discount_applied": {"type": "float", "min": 0, "max": 100, "description": "Percentage between 0 and 100"}
        },
        "allowed_values": {
            "product_category": ["Electronics", "Clothing", "Books", "Home & Garden", "Sports", "Beauty", "Toys", "Automotive", "Health", "Home"],
            "gender": ["Male", "Female", "Other"],
            "payment_method": ["Credit Card", "Debit Card", "PayPal", "Bank Transfer", "Cash", "Digital Wallet", "Gift Card"],
            "shipping_method": ["Standard", "Express", "Next-Day"]
        },
        "risk_thresholds": {
            "low_risk": "≤ 30% return probability",
            "medium_risk": "31-60% return probability", 
            "high_risk": "> 60% return probability"
        }
    }

@router.get("/stats")
async def get_processing_stats(
    order_agent: OrderProcessingAgent = Depends(get_order_agent)
):
    """
    Get order processing statistics
    """
    return order_agent.get_processing_stats()

@router.get("/health")
async def health_check():
    """
    Health check for order processing service
    """
    try:
        order_agent = get_order_processing_agent()
        model_agent = get_inference_agent()
        
        return {
            "status": "healthy",
            "service": "order-processing-api",
            "agents": {
                "order_processing": "active",
                "model_inference": "active"
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "order-processing-api",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }