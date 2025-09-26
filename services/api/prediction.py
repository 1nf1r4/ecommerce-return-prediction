from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import asyncio
import json
import os
import uuid
from pathlib import Path

# Import the Feature Engineering Agent
from agents.feature_engineering import FeatureEngineeringAgent, validate_input_data, get_default_feature_values

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    """Request model for prediction API"""
    
    # Required fields
    price: float = Field(..., gt=0, description="Product price")
    quantity: int = Field(..., gt=0, description="Quantity ordered")
    
    # Optional fields with defaults
    discount_rate: Optional[float] = Field(0.0, ge=0, le=1, description="Discount rate (0-1)")
    discount_amount: Optional[float] = Field(0.0, ge=0, description="Discount amount")
    user_location: Optional[str] = Field("Unknown", description="User location")
    product_category: Optional[str] = Field("Unknown", description="Product category")
    customer_segment: Optional[str] = Field("Regular", description="Customer segment")
    payment_method: Optional[str] = Field("Unknown", description="Payment method")
    customer_age: Optional[int] = Field(30, ge=18, le=100, description="Customer age")
    days_since_last_order: Optional[int] = Field(30, ge=0, description="Days since last order")
    order_date: Optional[str] = Field(None, description="Order date (YYYY-MM-DD)")
    user_gender: Optional[str] = Field("Unknown", description="User gender")
    shipping_method: Optional[str] = Field("Standard", description="Shipping method")
    
    @validator('order_date')
    def validate_order_date(cls, v):
        if v is None:
            return datetime.now().strftime('%Y-%m-%d')
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Order date must be in YYYY-MM-DD format')
    
    class Config:
        json_schema_extra = {
            "example": {
                "price": 99.99,
                "quantity": 2,
                "discount_rate": 0.1,
                "discount_amount": 10.0,
                "user_location": "New York",
                "product_category": "Electronics",
                "customer_segment": "Premium",
                "payment_method": "credit_card",
                "customer_age": 35,
                "days_since_last_order": 15,
                "order_date": "2024-01-15",
                "user_gender": "Female",
                "shipping_method": "Express"
            }
        }

class PredictionResponse(BaseModel):
    """Response model for prediction API"""
    
    prediction_id: str = Field(..., description="Unique prediction ID")
    return_probability: float = Field(..., ge=0, le=1, description="Probability of return (0-1)")
    confidence_score: float = Field(..., ge=0, le=1, description="Model confidence score")
    risk_category: str = Field(..., description="Risk category (Low/Medium/High)")
    features_used: Dict[str, Any] = Field(..., description="Engineered features used for prediction")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    
    predictions: List[PredictionRequest] = Field(..., description="List of prediction requests")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "price": 99.99,
                        "quantity": 2,
                        "product_category": "Electronics"
                    },
                    {
                        "price": 49.99,
                        "quantity": 1,
                        "product_category": "Clothing"
                    }
                ]
            }
        }

class PredictionAPIAgent:
    """
    Prediction API Agent for E-commerce Return Prediction
    
    Purpose: Handle HTTP requests and responses
    Functions:
    - Validate input parameters
    - Orchestrate preprocessing and inference
    - Format prediction results with confidence scores
    - Log predictions to database
    - Handle error responses and timeouts
    """
    
    def __init__(self, model_path: str = None):
        self.feature_agent = FeatureEngineeringAgent()
        self.model = None
        self.model_version = "1.0"
        self.prediction_logs = []
        self.model_path = model_path or "models/random_forest_model.pkl"
        
        # Load the trained model
        self.load_model()
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="E-commerce Return Prediction API",
            description="API for predicting return probability of e-commerce orders",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self.setup_routes()
        
    def load_model(self):
        """Load the trained model from disk"""
        try:
            full_model_path = Path(__file__).parent.parent / self.model_path
            
            if full_model_path.exists():
                # Suppress sklearn version warnings when loading the model
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
                    with open(full_model_path, 'rb') as f:
                        self.model = pickle.load(f)
                logger.info(f"Model loaded successfully from {full_model_path}")
            else:
                logger.warning(f"Model file not found at {full_model_path}")
                # Create a dummy model for testing
                self.create_dummy_model()
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.create_dummy_model()
    
    def create_dummy_model(self):
        """Create a dummy model for testing when real model is not available"""
        from sklearn.ensemble import RandomForestClassifier
        
        logger.info("Creating dummy model for testing")
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Create dummy training data
        X_dummy = np.random.rand(100, 20)
        y_dummy = np.random.randint(0, 2, 100)
        self.model.fit(X_dummy, y_dummy)
        
        # Set feature columns for consistency
        self.feature_agent.feature_columns = [f'feature_{i}' for i in range(20)]
    
    def validate_input_parameters(self, request: PredictionRequest) -> bool:
        """
        Validate input parameters
        
        Args:
            request: PredictionRequest object
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Convert to dict for validation
            data = request.dict()
            
            # Use the validation function from feature engineering
            if not validate_input_data(data):
                return False
                
            # Additional API-specific validations
            if request.price <= 0:
                logger.error("Price must be positive")
                return False
                
            if request.quantity <= 0:
                logger.error("Quantity must be positive")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating input: {e}")
            return False
    
    def orchestrate_preprocessing_and_inference(self, request: PredictionRequest) -> Dict[str, Any]:
        """
        Orchestrate preprocessing and inference
        
        Args:
            request: PredictionRequest object
            
        Returns:
            Dict containing prediction results
        """
        start_time = datetime.now()
        
        try:
            # Convert request to dictionary
            raw_data = request.dict()
            
            # Engineer features
            logger.info("Starting feature engineering")
            engineered_features = self.feature_agent.process_features(raw_data, fit=False)
            
            # Convert to feature vector
            feature_vector = self.feature_agent.get_feature_vector(engineered_features)
            
            # Ensure we have the right number of features
            if len(feature_vector) != len(self.feature_agent.feature_columns):
                # Pad or truncate as needed
                target_length = len(self.feature_agent.feature_columns)
                if len(feature_vector) < target_length:
                    feature_vector.extend([0.0] * (target_length - len(feature_vector)))
                else:
                    feature_vector = feature_vector[:target_length]
            
            # Make prediction
            logger.info("Making prediction")
            X = np.array(feature_vector).reshape(1, -1)
            
            # Create DataFrame with proper feature names to avoid sklearn warning
            import pandas as pd
            X_df = pd.DataFrame(X, columns=self.feature_agent.feature_columns)
            
            # Get prediction probability
            if hasattr(self.model, 'predict_proba'):
                prob_scores = self.model.predict_proba(X_df)[0]
                return_probability = prob_scores[1] if len(prob_scores) > 1 else prob_scores[0]
            else:
                # Fallback for models without predict_proba
                prediction = self.model.predict(X_df)[0]
                return_probability = float(prediction)
            
            # Calculate confidence score (using prediction probability)
            confidence_score = max(return_probability, 1 - return_probability)
            
            # Determine risk category
            if return_probability >= 0.7:
                risk_category = "High"
            elif return_probability >= 0.4:
                risk_category = "Medium"
            else:
                risk_category = "Low"
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = {
                'prediction_id': str(uuid.uuid4()),
                'return_probability': float(return_probability),
                'confidence_score': float(confidence_score),
                'risk_category': risk_category,
                'features_used': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                for k, v in engineered_features.items()},
                'model_version': self.model_version,
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': processing_time
            }
            
            logger.info(f"Prediction completed in {processing_time:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error in preprocessing and inference: {e}")
            raise
    
    def format_prediction_results(self, results: Dict[str, Any]) -> PredictionResponse:
        """
        Format prediction results with confidence scores
        
        Args:
            results: Raw prediction results
            
        Returns:
            PredictionResponse object
        """
        try:
            return PredictionResponse(**results)
            
        except Exception as e:
            logger.error(f"Error formatting results: {e}")
            raise HTTPException(status_code=500, detail="Error formatting prediction results")
    
    def log_prediction_to_database(self, request: PredictionRequest, response: PredictionResponse):
        """
        Log predictions to database (in-memory for now)
        
        Args:
            request: Original request
            response: Prediction response
        """
        try:
            log_entry = {
                'prediction_id': response.prediction_id,
                'timestamp': response.timestamp,
                'request_data': request.dict(),
                'prediction_result': response.dict(),
                'model_version': response.model_version
            }
            
            self.prediction_logs.append(log_entry)
            
            # Keep only last 1000 logs in memory
            if len(self.prediction_logs) > 1000:
                self.prediction_logs = self.prediction_logs[-1000:]
                
            logger.info(f"Logged prediction: {response.prediction_id}")
            
        except Exception as e:
            logger.error(f"Error logging prediction: {e}")
    
    def handle_error_responses(self, error: Exception) -> HTTPException:
        """
        Handle error responses and timeouts
        
        Args:
            error: Exception that occurred
            
        Returns:
            HTTPException with appropriate status code and message
        """
        if isinstance(error, ValueError):
            return HTTPException(status_code=400, detail=f"Invalid input: {str(error)}")
        elif isinstance(error, TimeoutError):
            return HTTPException(status_code=408, detail="Request timeout")
        elif isinstance(error, FileNotFoundError):
            return HTTPException(status_code=500, detail="Model not found")
        else:
            logger.error(f"Unexpected error: {error}")
            return HTTPException(status_code=500, detail="Internal server error")
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            """Health check endpoint"""
            return {
                "message": "E-commerce Return Prediction API",
                "status": "running",
                "version": "1.0.0",
                "model_version": self.model_version,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/health")
        async def health_check():
            """Detailed health check"""
            return {
                "status": "healthy",
                "model_loaded": self.model is not None,
                "feature_agent_ready": True,
                "total_predictions": len(self.prediction_logs),
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
            """
            Single prediction endpoint
            
            Args:
                request: PredictionRequest object
                background_tasks: FastAPI background tasks
                
            Returns:
                PredictionResponse object
            """
            try:
                # Validate input parameters
                if not self.validate_input_parameters(request):
                    raise ValueError("Invalid input parameters")
                
                # Orchestrate preprocessing and inference
                results = self.orchestrate_preprocessing_and_inference(request)
                
                # Format prediction results
                response = self.format_prediction_results(results)
                
                # Log prediction in background
                background_tasks.add_task(self.log_prediction_to_database, request, response)
                
                return response
                
            except Exception as e:
                raise self.handle_error_responses(e)
        
        @self.app.post("/predict/batch")
        async def batch_predict(request: BatchPredictionRequest, background_tasks: BackgroundTasks):
            """
            Batch prediction endpoint
            
            Args:
                request: BatchPredictionRequest object
                background_tasks: FastAPI background tasks
                
            Returns:
                List of PredictionResponse objects
            """
            try:
                if len(request.predictions) > 100:
                    raise ValueError("Batch size cannot exceed 100 predictions")
                
                results = []
                
                for pred_request in request.predictions:
                    try:
                        # Validate input
                        if not self.validate_input_parameters(pred_request):
                            continue
                            
                        # Process prediction
                        result = self.orchestrate_preprocessing_and_inference(pred_request)
                        response = self.format_prediction_results(result)
                        results.append(response)
                        
                        # Log in background
                        background_tasks.add_task(
                            self.log_prediction_to_database, 
                            pred_request, 
                            response
                        )
                        
                    except Exception as e:
                        logger.error(f"Error in batch prediction item: {e}")
                        continue
                
                return {
                    "predictions": results,
                    "total_processed": len(results),
                    "total_requested": len(request.predictions),
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                raise self.handle_error_responses(e)
        
        @self.app.get("/predictions/logs")
        async def get_prediction_logs(limit: int = 50):
            """
            Get recent prediction logs
            
            Args:
                limit: Number of logs to return
                
            Returns:
                List of recent prediction logs
            """
            try:
                recent_logs = self.prediction_logs[-limit:] if self.prediction_logs else []
                return {
                    "logs": recent_logs,
                    "total_logs": len(self.prediction_logs),
                    "showing": len(recent_logs)
                }
                
            except Exception as e:
                raise self.handle_error_responses(e)
        
        @self.app.get("/model/info")
        async def get_model_info():
            """Get model information"""
            return {
                "model_version": self.model_version,
                "model_type": type(self.model).__name__ if self.model else "None",
                "feature_count": len(self.feature_agent.feature_columns),
                "feature_columns": self.feature_agent.feature_columns,
                "model_loaded": self.model is not None
            }

# Create global instance
prediction_agent = PredictionAPIAgent()

# Export the FastAPI app
app = prediction_agent.app