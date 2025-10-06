"""
Prediction API Endpoints
Purpose: Handle HTTP requests for return predictions with user authentication support
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, BackgroundTasks, Header, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import pandas as pd
import io
import csv
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

# Import the agents
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.model_inference import get_inference_agent, ModelInferenceAgent
from agents.feature_engineering import get_feature_engineering_agent, FeatureEngineeringAgent
from agents.order_processing import get_order_processing_agent, OrderProcessingAgent
from utils.supabase_service import get_supabase_service, SupabaseService

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/predict", tags=["prediction"])

# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    price: float = Field(..., gt=0, description="Product price in USD")
    quantity: int = Field(..., gt=0, description="Order quantity")
    product_category: str = Field(..., description="Product category")
    gender: str = Field(..., description="Customer gender")
    payment_method: str = Field(..., description="Payment method used")
    age: int = Field(..., ge=0, le=120, description="Customer age")
    location: str = Field(..., description="Customer location")

    @field_validator('product_category')
    def validate_category(cls, v):
        allowed_categories = [
            'Electronics', 'Clothing', 'Books', 'Home & Garden', 
            'Sports', 'Beauty', 'Toys', 'Automotive', 'Health'
        ]
        if v not in allowed_categories:
            logger.warning(f"Unknown category: {v}, proceeding with prediction")
        return v
    
    @field_validator('gender')
    def validate_gender(cls, v):
        allowed_genders = ['Male', 'Female', 'Other']
        if v not in allowed_genders:
            raise ValueError(f"Gender must be one of: {allowed_genders}")
        return v
    
    @field_validator('payment_method')
    def validate_payment_method(cls, v):
        allowed_methods = [
            'Credit Card', 'Debit Card', 'PayPal', 
            'Bank Transfer', 'Cash', 'Digital Wallet'
        ]
        if v not in allowed_methods:
            logger.warning(f"Unknown payment method: {v}, proceeding with prediction")
        return v

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    predictions: List[PredictionRequest] = Field(..., max_items=100, description="List of predictions (max 100)")

class PredictionResponse(BaseModel):
    """Response model for prediction results"""
    success: bool
    prediction: Optional[Dict[str, Any]] = None
    model_info: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    success: bool
    results: List[PredictionResponse]
    summary: Dict[str, Any]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    status: str
    models_loaded: Dict[str, bool]
    test_prediction: Optional[str] = None
    error: Optional[str] = None
    timestamp: str

class BatchJobStatus(BaseModel):
    """Response model for batch job status"""
    job_id: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    total_records: int
    processed_records: int
    failed_records: int
    progress_percentage: float
    results_available: bool
    created_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None

class FileUploadResponse(BaseModel):
    """Response model for file upload"""
    success: bool
    job_id: str
    message: str
    total_records: int
    estimated_processing_time: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# Global storage for batch jobs (in production, use Redis or database)
batch_jobs: Dict[str, Dict[str, Any]] = {}

# Dependencies to get agents and services
def get_agent() -> ModelInferenceAgent:
    """Dependency to provide inference agent"""
    return get_inference_agent()

def get_feature_agent() -> FeatureEngineeringAgent:
    """Dependency to provide feature engineering agent"""
    return get_feature_engineering_agent()

def get_order_agent() -> OrderProcessingAgent:
    """Dependency to provide order processing agent"""
    return get_order_processing_agent()

def get_db_service() -> SupabaseService:
    """Dependency to provide Supabase service"""
    return get_supabase_service()

async def get_current_user(
    authorization: Optional[str] = Header(None),
    db_service: SupabaseService = Depends(get_db_service)
) -> Optional[Dict[str, Any]]:
    """
    Get current authenticated user from Authorization header
    
    Args:
        authorization: Authorization header with Bearer token
        db_service: Supabase service instance
        
    Returns:
        User data if authenticated, None otherwise
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None
        
    try:
        token = authorization.split("Bearer ")[1]
        user = db_service.authenticate_user(token)
        return user
    except Exception as e:
        logger.warning(f"Authentication failed: {str(e)}")
        return None

async def get_request_metadata(request: Request) -> Dict[str, Any]:
    """
    Extract request metadata for logging
    
    Args:
        request: FastAPI request object
        
    Returns:
        Request metadata
    """
    return {
        'ip_address': request.client.host if request.client else None,
        'user_agent': request.headers.get('user-agent'),
        'request_size_bytes': len(await request.body()) if hasattr(request, 'body') else None
    }

def validate_csv_file(file_content: bytes) -> tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Validate uploaded CSV file
    
    Args:
        file_content: Raw file bytes
        
    Returns:
        Tuple of (is_valid, error_message, dataframe)
    """
    try:
        # Try to read as CSV
        df = pd.read_csv(io.BytesIO(file_content))
        
        # Check if file is empty
        if df.empty:
            return False, "File is empty", None
            
        # Check row limit
        if len(df) > 10000:
            return False, "File contains more than 10,000 rows (limit exceeded)", None
            
        # Check required columns
        required_columns = ['price', 'quantity', 'product_category', 'gender', 'payment_method', 'age', 'location']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}", None
            
        # Basic data validation with better type checking
        # Clean and convert price column
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        if df['price'].isnull().any() or (df['price'] <= 0).any():
            return False, "Price column contains invalid values (must be positive numbers)", None
            
        # Clean and convert quantity column  
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        if df['quantity'].isnull().any() or (df['quantity'] <= 0).any():
            return False, "Quantity column contains invalid values (must be positive integers)", None
            
        # Clean and convert age column
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        if df['age'].isnull().any() or (df['age'] < 0).any() or (df['age'] > 120).any():
            return False, "Age column contains invalid values (must be between 0-120)", None
            
        # Clean string columns
        string_columns = ['product_category', 'gender', 'payment_method', 'location']
        for col in string_columns:
            df[col] = df[col].astype(str).str.strip()
            if df[col].isnull().any() or (df[col] == '').any() or (df[col] == 'nan').any():
                return False, f"{col} column contains empty or invalid values", None
            
        return True, "File validation successful", df
        
    except Exception as e:
        return False, f"Error reading CSV file: {str(e)}", None

def validate_excel_file(file_content: bytes) -> tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Validate uploaded Excel file
    
    Args:
        file_content: Raw file bytes
        
    Returns:
        Tuple of (is_valid, error_message, dataframe)
    """
    try:
        # Try to read as Excel
        df = pd.read_excel(io.BytesIO(file_content))
        
        # Use same validation as CSV
        return validate_csv_data(df)
        
    except Exception as e:
        return False, f"Error reading Excel file: {str(e)}", None

def validate_csv_data(df: pd.DataFrame) -> tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Validate DataFrame data regardless of source
    """
    # Check if file is empty
    if df.empty:
        return False, "File is empty", None
        
    # Check row limit
    if len(df) > 10000:
        return False, "File contains more than 10,000 rows (limit exceeded)", None
        
    # Check required columns
    required_columns = ['price', 'quantity', 'product_category', 'gender', 'payment_method', 'age', 'location']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}", None
        
    # Basic data validation with better type checking
    # Clean and convert price column
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    if df['price'].isnull().any() or (df['price'] <= 0).any():
        return False, "Price column contains invalid values (must be positive numbers)", None
        
    # Clean and convert quantity column  
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    if df['quantity'].isnull().any() or (df['quantity'] <= 0).any():
        return False, "Quantity column contains invalid values (must be positive integers)", None
        
    # Clean and convert age column
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    if df['age'].isnull().any() or (df['age'] < 0).any() or (df['age'] > 120).any():
        return False, "Age column contains invalid values (must be between 0-120)", None
        
    # Clean string columns
    string_columns = ['product_category', 'gender', 'payment_method', 'location']
    for col in string_columns:
        df[col] = df[col].astype(str).str.strip()
        if df[col].isnull().any() or (df[col] == '').any() or (df[col] == 'nan').any():
            return False, f"{col} column contains empty or invalid values", None
        
    return True, "File validation successful", df

async def process_batch_predictions(job_id: str, df: pd.DataFrame, agent: ModelInferenceAgent):
    """
    Process batch predictions in background
    
    Args:
        job_id: Unique job identifier
        df: DataFrame with prediction data
        agent: Model inference agent
    """
    try:
        # Import the required agents for batch processing
        from agents.feature_engineering import FeatureEngineeringAgent
        from agents.order_processing import OrderProcessingAgent
        
        # Initialize agents
        feature_agent = FeatureEngineeringAgent()
        order_agent = OrderProcessingAgent()
        
        # Update job status
        batch_jobs[job_id]["status"] = "processing"
        batch_jobs[job_id]["started_at"] = datetime.now().isoformat()
        
        # Convert DataFrame to list of prediction requests
        predictions = []
        failed_count = 0
        
        for index, row in df.iterrows():
            try:
                # Create prediction request with proper type handling
                # Convert pandas/numpy types to native Python types
                request_data = {
                    "price": float(pd.to_numeric(row['price'], errors='coerce')),
                    "quantity": int(pd.to_numeric(row['quantity'], errors='coerce')),
                    "product_category": str(row['product_category']).strip(),
                    "gender": str(row['gender']).strip(),
                    "payment_method": str(row['payment_method']).strip(),
                    "age": int(pd.to_numeric(row['age'], errors='coerce')),
                    "location": str(row['location']).strip()
                }
                
                # Validate that numeric conversions succeeded
                if pd.isna(request_data["price"]) or pd.isna(request_data["quantity"]) or pd.isna(request_data["age"]):
                    raise ValueError(f"Invalid numeric data in row {index + 1}: price={row['price']}, quantity={row['quantity']}, age={row['age']}")
                
                # Follow the same pipeline as single prediction
                # Step 1: Use order processing agent to create basic features
                processed_result = order_agent.process_single_order(request_data)
                
                if not processed_result['success']:
                    raise ValueError(f"Order processing failed: {processed_result.get('error', 'Unknown error')}")
                
                # Step 2: Use feature engineering agent to create advanced features
                basic_features_df = processed_result['prediction_ready_data']
                final_data = feature_agent.transform(basic_features_df)
                
                # Step 3: Make prediction with fully engineered features
                result = agent.predict_single(final_data)
                
                if result["success"]:
                    predictions.append({
                        "row_index": index + 1,
                        "input_data": request_data,
                        "prediction": result["prediction"],
                        "success": True
                    })
                else:
                    failed_count += 1
                    predictions.append({
                        "row_index": index + 1,
                        "input_data": request_data,
                        "error": result.get("error", "Unknown error"),
                        "success": False
                    })
                
                # Update progress
                processed = index + 1
                batch_jobs[job_id]["processed_records"] = processed
                batch_jobs[job_id]["failed_records"] = failed_count
                batch_jobs[job_id]["progress_percentage"] = (processed / len(df)) * 100
                
                # Small delay to prevent overwhelming the system
                if index % 100 == 0:
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                failed_count += 1
                # Create safe input data for error reporting
                try:
                    input_data_for_error = {
                        "price": str(row['price']),
                        "quantity": str(row['quantity']),
                        "product_category": str(row['product_category']),
                        "gender": str(row['gender']),
                        "payment_method": str(row['payment_method']),
                        "age": str(row['age']),
                        "location": str(row['location'])
                    }
                except:
                    input_data_for_error = {"error": "Could not extract row data"}
                
                predictions.append({
                    "row_index": index + 1,
                    "input_data": input_data_for_error,
                    "error": f"Data conversion error: {str(e)}",
                    "success": False
                })
                batch_jobs[job_id]["failed_records"] = failed_count
        
        # Job completed
        batch_jobs[job_id]["status"] = "completed"
        batch_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        batch_jobs[job_id]["results"] = predictions
        batch_jobs[job_id]["results_available"] = True
        
        # Generate summary
        successful_predictions = [p for p in predictions if p["success"]]
        batch_jobs[job_id]["summary"] = {
            "total_processed": len(predictions),
            "successful_predictions": len(successful_predictions),
            "failed_predictions": failed_count,
            "success_rate": (len(successful_predictions) / len(predictions)) * 100 if predictions else 0,
            "avg_return_probability": sum(p["prediction"]["return_probability"] for p in successful_predictions) / len(successful_predictions) if successful_predictions else 0
        }
        
    except Exception as e:
        # Job failed
        batch_jobs[job_id]["status"] = "failed"
        batch_jobs[job_id]["error_message"] = str(e)
        batch_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        logger.error(f"Batch job {job_id} failed: {str(e)}")



@router.post("/single", response_model=PredictionResponse)
async def predict_single(
    prediction_request: PredictionRequest,
    request: Request,
    agent: ModelInferenceAgent = Depends(get_agent),
    feature_agent: FeatureEngineeringAgent = Depends(get_feature_agent),
    order_agent: OrderProcessingAgent = Depends(get_order_agent),
    db_service: SupabaseService = Depends(get_db_service),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> PredictionResponse:
    """Make a single prediction with optional user authentication"""
    start_time = datetime.now()
    user_id = current_user['id'] if current_user else None
    
    try:
        # Convert request to DataFrame format expected by the model
        data = prediction_request.model_dump()
        
        # Step 1: Use order processing agent to create basic features
        processed_result = order_agent.process_single_order(data)
        
        if not processed_result['success']:
            return PredictionResponse(
                success=False,
                error=processed_result['error']
            )
        
        # Step 2: Use feature engineering agent to create advanced features
        basic_features_df = processed_result['prediction_ready_data']
        final_data = feature_agent.transform(basic_features_df)
        
        # Step 3: Make prediction with fully engineered features
        prediction_result = agent.predict_single(final_data)
        
        if not prediction_result['success']:
            return PredictionResponse(
                success=False,
                error=prediction_result.get('error', 'Prediction failed')
            )
        
        # Calculate processing time
        processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Store prediction in database if Supabase is enabled
        if db_service.is_enabled() and prediction_result.get('prediction'):
            # Merge input data with prediction results for storage
            prediction_data = prediction_result['prediction'].copy()
            prediction_data.update(data)  # Include all input data
            prediction_data['processing_time_ms'] = processing_time_ms
            
            stored_result = await db_service.store_prediction(prediction_data, user_id)
            if stored_result:
                logger.info(f"Prediction stored to database with ID: {stored_result.get('id')}")
            else:
                logger.warning("Failed to store prediction to database")
        
        # Log API usage
        if db_service.is_enabled():
            request_metadata = await get_request_metadata(request)
            await db_service.log_api_usage(
                user_id=user_id,
                endpoint="/predict/single",
                processing_time_ms=processing_time_ms,
                success=True,
                request_metadata=request_metadata
            )
        
        return PredictionResponse(
            success=True,
            prediction=prediction_result['prediction'],
            model_info=prediction_result.get('model_info'),
            feature_importance=prediction_result.get('feature_importance'),
            metadata=prediction_result.get('metadata')
        )
        
    except Exception as e:
        processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        logger.error(f"Single prediction error: {e}")
        
        # Log failed API usage
        if db_service.is_enabled():
            request_metadata = await get_request_metadata(request)
            await db_service.log_api_usage(
                user_id=user_id,
                endpoint="/predict/single",
                processing_time_ms=processing_time_ms,
                success=False,
                error_message=str(e),
                request_metadata=request_metadata
            )
        
        return PredictionResponse(
            success=False,
            error=f"Prediction failed: {str(e)}"
        )

@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    agent: ModelInferenceAgent = Depends(get_agent)
) -> BatchPredictionResponse:
    """
    Make batch return predictions
    
    Args:
        request: Batch prediction request
        agent: Model inference agent
        
    Returns:
        Batch prediction response with results
    """
    try:
        logger.info(f"Received batch prediction request for {len(request.predictions)} items")
        
        # Convert requests to feature dictionaries
        features_list = [pred.dict() for pred in request.predictions]
        
        # Make batch predictions
        results = agent.predict_batch(features_list)
        
        # Convert results to response format
        prediction_responses = []
        successful_predictions = 0
        
        for result in results:
            if result.get('success', False):
                successful_predictions += 1
                prediction_responses.append(PredictionResponse(
                    success=True,
                    prediction=result.get('prediction'),
                    model_info=result.get('model_info'),
                    feature_importance=result.get('feature_importance'),
                    metadata=result.get('metadata')
                ))
            else:
                prediction_responses.append(PredictionResponse(
                    success=False,
                    error=result.get('error')
                ))
        
        # Create summary
        summary = {
            'total_requests': len(request.predictions),
            'successful_predictions': successful_predictions,
            'failed_predictions': len(request.predictions) - successful_predictions,
            'success_rate': successful_predictions / len(request.predictions) if request.predictions else 0
        }
        
        return BatchPredictionResponse(
            success=True,
            results=prediction_responses,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=HealthCheckResponse)
async def health_check(agent: ModelInferenceAgent = Depends(get_agent)) -> HealthCheckResponse:
    """
    Check the health of the prediction service
    
    Args:
        agent: Model inference agent
        
    Returns:
        Health check response
    """
    try:
        health_result = agent.health_check()
        
        return HealthCheckResponse(
            status=health_result.get('status', 'unknown'),
            models_loaded=health_result.get('models_loaded', {}),
            test_prediction=health_result.get('test_prediction'),
            error=health_result.get('error'),
            timestamp=health_result.get('timestamp')
        )
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return HealthCheckResponse(
            status="unhealthy",
            models_loaded={},
            error=str(e),
            timestamp=datetime.now().isoformat()
        )

@router.get("/model-info")
async def get_model_info(agent: ModelInferenceAgent = Depends(get_agent)) -> Dict[str, Any]:
    """
    Get information about loaded models
    
    Args:
        agent: Model inference agent
        
    Returns:
        Model information
    """
    try:
        return agent.get_model_info()
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/upload", response_model=FileUploadResponse)
async def upload_batch_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    agent: ModelInferenceAgent = Depends(get_agent)
) -> FileUploadResponse:
    """
    Upload CSV or Excel file for batch prediction processing
    
    Args:
        file: Uploaded file (CSV or Excel)
        agent: Model inference agent
        
    Returns:
        FileUploadResponse with job details
    """
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
            
        file_extension = file.filename.lower().split('.')[-1]
        if file_extension not in ['csv', 'xlsx', 'xls']:
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Only CSV and Excel files are supported"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Validate file content based on type
        if file_extension == 'csv':
            is_valid, error_message, df = validate_csv_file(file_content)
        else:
            is_valid, error_message, df = validate_excel_file(file_content)
        
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_message)
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job tracking
        batch_jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "total_records": len(df),
            "processed_records": 0,
            "failed_records": 0,
            "progress_percentage": 0.0,
            "results_available": False,
            "created_at": datetime.now().isoformat(),
            "filename": file.filename,
            "file_type": file_extension
        }
        
        # Start background processing
        background_tasks.add_task(process_batch_predictions, job_id, df, agent)
        
        # Calculate estimated processing time (rough estimate: 100ms per record)
        estimated_seconds = len(df) * 0.1
        if estimated_seconds < 60:
            estimated_time = f"{estimated_seconds:.0f} seconds"
        else:
            estimated_time = f"{estimated_seconds/60:.1f} minutes"
        
        return FileUploadResponse(
            success=True,
            job_id=job_id,
            message=f"File uploaded successfully. Processing {len(df)} records.",
            total_records=len(df),
            estimated_processing_time=estimated_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@router.get("/batch/{job_id}", response_model=BatchJobStatus)
async def get_batch_status(job_id: str) -> BatchJobStatus:
    """
    Get status of batch prediction job
    
    Args:
        job_id: Unique job identifier
        
    Returns:
        BatchJobStatus with current job status
    """
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = batch_jobs[job_id]
    
    return BatchJobStatus(
        job_id=job_id,
        status=job_data["status"],
        total_records=job_data["total_records"],
        processed_records=job_data["processed_records"],
        failed_records=job_data["failed_records"],
        progress_percentage=job_data["progress_percentage"],
        results_available=job_data["results_available"],
        created_at=job_data["created_at"],
        completed_at=job_data.get("completed_at"),
        error_message=job_data.get("error_message")
    )

@router.get("/batch/{job_id}/results")
async def get_batch_results(job_id: str):
    """
    Get results of completed batch prediction job
    
    Args:
        job_id: Unique job identifier
        
    Returns:
        Batch prediction results
    """
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = batch_jobs[job_id]
    
    if job_data["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job is not completed. Current status: {job_data['status']}"
        )
    
    if not job_data["results_available"]:
        raise HTTPException(status_code=404, detail="Results not available")
    
    return {
        "job_id": job_id,
        "status": job_data["status"],
        "summary": job_data.get("summary", {}),
        "results": job_data.get("results", []),
        "completed_at": job_data.get("completed_at")
    }

@router.get("/batch/{job_id}/download")
async def download_batch_results(job_id: str, format: str = "csv"):
    """
    Download batch prediction results as CSV or Excel
    
    Args:
        job_id: Unique job identifier
        format: Download format ('csv' or 'excel')
        
    Returns:
        StreamingResponse with file download
    """
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = batch_jobs[job_id]
    
    if job_data["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job is not completed. Current status: {job_data['status']}"
        )
    
    if not job_data["results_available"]:
        raise HTTPException(status_code=404, detail="Results not available")
    
    # Prepare data for download
    results = job_data.get("results", [])
    
    # Create DataFrame from results
    download_data = []
    for result in results:
        row = result["input_data"].copy()
        if result["success"]:
            row.update({
                "return_probability": result["prediction"]["return_probability"],
                "risk_level": result["prediction"]["risk_level"],
                "confidence_score": result["prediction"]["confidence_score"],
                "prediction_status": "Success"
            })
        else:
            row.update({
                "return_probability": None,
                "risk_level": None,
                "confidence_score": None,
                "prediction_status": "Failed",
                "error": result.get("error", "Unknown error")
            })
        download_data.append(row)
    
    df = pd.DataFrame(download_data)
    
    # Generate file based on format
    if format.lower() == "csv":
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        response = StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=batch_predictions_{job_id}.csv"}
        )
        
    elif format.lower() == "excel":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Predictions')
        output.seek(0)
        
        response = StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=batch_predictions_{job_id}.xlsx"}
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid format. Use 'csv' or 'excel'")
    
    return response

# Example usage endpoints for testing
@router.get("/example")
async def get_example_request() -> Dict[str, Any]:
    """
    Get an example prediction request for testing
    
    Returns:
        Example request data
    """
    return {
        "example_single_request": {
            "price": 199.99,
            "quantity": 1,
            "product_category": "Electronics",
            "gender": "Female",
            "payment_method": "Credit Card",
            "age": 28,
            "location": "California"
        },
        "example_batch_request": {
            "predictions": [
                {
                    "price": 199.99,
                    "quantity": 1,
                    "product_category": "Electronics",
                    "gender": "Female",
                    "payment_method": "Credit Card",
                    "age": 28,
                    "location": "California"
                },
                {
                    "price": 49.99,
                    "quantity": 2,
                    "product_category": "Clothing",
                    "gender": "Male",
                    "payment_method": "PayPal",
                    "age": 35,
                    "location": "New York"
                }
            ]
        }
    }