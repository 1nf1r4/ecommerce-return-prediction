from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting E-commerce Return Prediction API...")
    include_routers()
    logger.info("✅ API startup complete!")
    yield
    # Shutdown
    logger.info("🛑 Shutting down API...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="E-commerce Return Prediction API",
    description="AI-powered return prediction service for e-commerce orders",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
import os
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173").split(",")

# Add common patterns for production
if any("vercel.app" in origin for origin in cors_origins):
    cors_origins.extend([
        "https://*.vercel.app",
        "https://vercel.app"
    ])

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins + [
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "https://ecommerce-return-prediction-2.vercel.app",  # Numbered Vercel domain
    ],
    allow_origin_regex=r"https://ecommerce-return-prediction.*\.vercel\.app",  # All Vercel subdomains/variants
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy load routers to avoid loading models during startup
def include_routers():
    """Include routers after app startup to avoid model loading during initialization"""
    routers_loaded = 0
    
    try:
        from api.prediction import router as prediction_router
        app.include_router(prediction_router)
        logger.info("Prediction router loaded")
        routers_loaded += 1
    except Exception as e:
        logger.warning(f"Failed to load prediction router: {str(e)}")
    
    try:
        from api.order_processing import router as order_processing_router
        app.include_router(order_processing_router)
        logger.info("Order processing router loaded")
        routers_loaded += 1
    except Exception as e:
        logger.warning(f"Failed to load order processing router: {str(e)}")
    
    try:
        from api.analytics import router as analytics_router
        app.include_router(analytics_router)
        logger.info("Analytics router loaded")
        routers_loaded += 1
    except Exception as e:
        logger.warning(f"Failed to load analytics router: {str(e)}")
    
    # If no routers loaded successfully, use minimal fallback
    if routers_loaded == 0:
        logger.warning("Loading minimal fallback router...")
        try:
            from api.minimal import router as minimal_router
            app.include_router(minimal_router)
            logger.info("Minimal fallback router loaded")
        except Exception as e:
            logger.error(f"Failed to load minimal router: {str(e)}")

    logger.info(f"Loaded {routers_loaded}/3 full routers")

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "E-commerce Return Prediction API",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/docs",
        "endpoints": {
            "health_check": "/health",
            "prediction_health": "/predict/health",
            "order_health": "/orders/health",
            "single_prediction": "/predict/single",
            "batch_prediction": "/predict/batch",
            "batch_upload": "/predict/batch/upload",
            "batch_status": "/predict/batch/{job_id}",
            "batch_results": "/predict/batch/{job_id}/results",
            "batch_download": "/predict/batch/{job_id}/download",
            "model_info": "/predict/model-info",
            "example_requests": "/predict/example",
            "order_processing": "/orders/process",
            "batch_order_processing": "/orders/batch-process",
            "validation_rules": "/orders/validation-rules",
            "order_stats": "/orders/stats",
            "analytics_health": "/api/analytics/health",
            "dashboard_data": "/api/analytics/dashboard",
            "revenue_impact": "/api/analytics/revenue-impact",
            "daily_reports": "/api/analytics/reports/{date}",
            "business_insights": "/api/analytics/insights",
            "accuracy_analysis": "/api/analytics/accuracy",
            "return_trends": "/api/analytics/trends",
            "model_performance": "/api/analytics/performance",
            "business_kpis": "/api/analytics/kpis"
        },
        "features": [
            "Single order return prediction",
            "Batch prediction processing",
            "File upload support (CSV/Excel)",
            "Real-time order processing",
            "Business intelligence analytics",
            "Product intelligence insights",
            "Model health monitoring"
        ]
    }

@app.get("/health")
def health_check():
    """General health check endpoint - lightweight version for Railway"""
    import time
    
    return {
        "status": "healthy",
        "service": "return-prediction-api",
        "version": "1.0.0",
        "timestamp": int(time.time()),
        "message": "Service is running"
    }

@app.get("/health/detailed")
def detailed_health_check():
    """Detailed health check endpoint with agent status"""
    try:
        # Test basic imports and agent availability
        from agents.model_inference import get_inference_agent
        from agents.order_processing import get_order_processing_agent
        from agents.feature_engineering import get_feature_engineering_agent
        from agents.eda_preprocess import get_preprocess_agent
        from agents.product_intelligence import get_product_intelligence_agent
        from agents.business_intelligence import get_business_intelligence_agent
        
        # Test agent initialization
        agents_status = {}
        try:
            inference_agent = get_inference_agent()
            agents_status['model_inference'] = 'healthy'
        except Exception as e:
            agents_status['model_inference'] = f'error: {str(e)}'
        
        try:
            order_agent = get_order_processing_agent()
            agents_status['order_processing'] = 'healthy'
        except Exception as e:
            agents_status['order_processing'] = f'error: {str(e)}'
        
        try:
            fe_agent = get_feature_engineering_agent()
            agents_status['feature_engineering'] = 'healthy'
        except Exception as e:
            agents_status['feature_engineering'] = f'error: {str(e)}'
        
        try:
            preprocess_agent = get_preprocess_agent()
            agents_status['eda_preprocess'] = 'healthy'
        except Exception as e:
            agents_status['eda_preprocess'] = f'error: {str(e)}'
        
        try:
            product_agent = get_product_intelligence_agent()
            agents_status['product_intelligence'] = 'healthy'
        except Exception as e:
            agents_status['product_intelligence'] = f'error: {str(e)}'
        
        try:
            bi_agent = get_business_intelligence_agent()
            agents_status['business_intelligence'] = 'healthy'
        except Exception as e:
            agents_status['business_intelligence'] = f'error: {str(e)}'
        
        # Overall status
        healthy_agents = sum(1 for status in agents_status.values() if status == 'healthy')
        total_agents = len(agents_status)
        
        overall_status = "healthy" if healthy_agents == total_agents else "degraded" if healthy_agents > 0 else "unhealthy"
        
        return {
            "status": overall_status,
            "service": "return-prediction-api",
            "agents": agents_status,
            "agent_summary": f"{healthy_agents}/{total_agents} agents healthy",
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "return-prediction-api",
            "error": str(e),
            "timestamp": "2024-01-01T00:00:00Z"
        }

# Development/testing mode
if __name__ == "__main__":
    # For development testing with sample data
    import pandas as pd
    from agents.feature_engineering import FeatureEngineeringAgent
    from agents.model_inference import ModelInferenceAgent
    
    logger.info("Running in development mode...")
    
    # Test the pipeline with sample data
    sample_data = pd.DataFrame([{
        'Product_Category': 'Electronics',
        'Product_Price': 199.99,
        'Order_Quantity': 1,
        'User_Age': 28,
        'User_Gender': 'Female',
        'Payment_Method': 'Credit Card',
        'User_Location': 'Urban',
        'Discount_Applied': 10.0,
        'Shipping_Method': 'Standard',
        'Order_Year': 2024,
        'Order_Month': 10,
        'Order_Weekday': 1
    }])
    
    try:
        # Test feature engineering
        fe_agent = FeatureEngineeringAgent()
        engineered_features = fe_agent.transform(sample_data)
        logger.info(f"Feature engineering successful. Shape: {engineered_features.shape}")
        
        # Test model inference
        inference_agent = ModelInferenceAgent()
        
        # Test to_inference integration
        prediction_result = fe_agent.to_inference(engineered_features, inference_agent)
        logger.info(f"Prediction successful: {prediction_result['success']}")
        
        if prediction_result['success']:
            pred_data = prediction_result['prediction']
            logger.info(f"Return probability: {pred_data['return_probability']:.3f}")
            logger.info(f"Will return: {pred_data['will_return']}")
        
        logger.info("Development test completed successfully!")
        
    except Exception as e:
        logger.error(f"Development test failed: {str(e)}")
    
    # Start the server for development
    logger.info("Starting FastAPI server on http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
