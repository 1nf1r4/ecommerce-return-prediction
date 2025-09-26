# E-commerce Return Prediction Backend API

## Overview

This backend provides **Feature Engineering Agent** and **Prediction API Agent** for predicting e-commerce return probability. The system uses machine learning to analyze order characteristics and provide return risk assessments.

## Architecture

### 3. Feature Engineering Agent (`agents/feature_engineering.py`)

**Purpose**: Create derived features from raw inputs

**Functions**:
- ✅ **Calculate Total_Order_Value** (price × quantity)
- ✅ **Extract temporal features** (year, month, weekday, weekend detection)
- ✅ **Generate location encoding** from user location
- ✅ **Apply discount calculations** and transformations
- ✅ **Create categorical features** (customer tier, payment method encoding)
- ✅ **Feature vector generation** (14 optimized features for model compatibility)

**Key Features**:
- Handles missing data gracefully with sensible defaults
- Supports both training (fit=True) and prediction (fit=False) modes
- Generates 14 standardized features compatible with existing models
- Comprehensive logging for debugging and monitoring

### 4. Prediction API Agent (`api/prediction.py`)

**Purpose**: Handle HTTP requests and responses

**Functions**:
- ✅ **Validate input parameters** with Pydantic models
- ✅ **Orchestrate preprocessing and inference** pipeline
- ✅ **Format prediction results** with confidence scores
- ✅ **Log predictions to database** (in-memory logging)
- ✅ **Handle error responses and timeouts** with detailed error messages

**Key Features**:
- FastAPI-based REST API with automatic documentation
- Real-time single predictions via `/predict`
- Batch processing via `/predict/batch` (up to 100 predictions)
- Comprehensive error handling and validation
- CORS support for frontend integration
- Background task processing for logging

## API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with basic info |
| `/health` | GET | Health check with system status |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/predictions/logs` | GET | Recent prediction logs |
| `/model/info` | GET | Model information and features |
| `/api/status` | GET | API status and capabilities |

### Request/Response Examples

#### Single Prediction

**Request:**
```json
POST /predict
{
  \"price\": 99.99,
  \"quantity\": 2,
  \"discount_rate\": 0.1,
  \"discount_amount\": 10.0,
  \"user_location\": \"New York\",
  \"product_category\": \"Electronics\",
  \"customer_segment\": \"Premium\",
  \"payment_method\": \"credit_card\",
  \"customer_age\": 35,
  \"days_since_last_order\": 15,
  \"order_date\": \"2024-01-15\"
}
```

**Response:**
```json
{
  \"prediction_id\": \"uuid-string\",
  \"return_probability\": 0.477,
  \"confidence_score\": 0.523,
  \"risk_category\": \"Medium\",
  \"features_used\": {
    \"total_order_value\": 199.98,
    \"price\": 99.99,
    \"quantity\": 2,
    \"discount_amount\": 10.0,
    \"final_price\": 89.99,
    \"customer_age\": 35,
    \"days_since_last_order\": 15,
    \"month\": 1,
    \"weekday\": 0,
    \"is_weekend\": 0,
    \"location_encoded\": 1,
    \"customer_tier\": 3,
    \"payment_method_encoded\": 1,
    \"has_discount\": 1
  },
  \"model_version\": \"1.0\",
  \"timestamp\": \"2025-09-26T17:02:35.123456\",
  \"processing_time_ms\": 4.13
}
```

#### Batch Prediction

**Request:**
```json
POST /predict/batch
{
  \"predictions\": [
    {
      \"price\": 99.99,
      \"quantity\": 2,
      \"product_category\": \"Electronics\"
    },
    {
      \"price\": 49.99,
      \"quantity\": 1,
      \"product_category\": \"Clothing\"
    }
  ]
}
```

## Feature Engineering Details

### Generated Features (14 total)

1. **total_order_value**: Price × Quantity
2. **price**: Original product price
3. **quantity**: Number of items ordered
4. **discount_amount**: Absolute discount value
5. **final_price**: Price after discount
6. **customer_age**: Customer age
7. **days_since_last_order**: Days since last purchase
8. **month**: Order month (1-12)
9. **weekday**: Day of week (0=Monday, 6=Sunday)
10. **is_weekend**: Weekend flag (1 if Sat/Sun)
11. **location_encoded**: Encoded customer location
12. **customer_tier**: Customer segment encoding (0-3)
13. **payment_method_encoded**: Payment method encoding
14. **has_discount**: Discount flag (1 if any discount)

### Feature Engineering Process

1. **Input Validation**: Validates required fields (price, quantity)
2. **Default Values**: Applies sensible defaults for missing optional fields
3. **Temporal Processing**: Extracts date-based features
4. **Categorical Encoding**: Encodes text categories to numbers
5. **Discount Calculations**: Computes discount-related metrics
6. **Vector Generation**: Creates consistent 14-feature vector

## Installation & Setup

### Prerequisites
- Python 3.8+
- All dependencies in `requirements.txt`

### Quick Start

1. **Install Dependencies:**
```bash
cd services
pip install -r requirements.txt
```

2. **Start the Server:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

3. **Test the API:**
```bash
python test_api.py
```

4. **Access Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing

The `test_api.py` script provides comprehensive testing:

- ✅ Health check verification
- ✅ Single prediction testing
- ✅ Batch prediction testing
- ✅ Feature engineering validation
- ✅ Error handling verification
- ✅ Model information retrieval

**Run Tests:**
```bash
python test_api.py
```

## Error Handling

The API provides detailed error responses:

- **400 Bad Request**: Invalid input parameters
- **408 Timeout**: Request processing timeout
- **500 Internal Server Error**: Model or processing errors

Example error response:
```json
{
  \"detail\": \"Invalid input: Price must be positive\"
}
```

## Performance

- **Single Prediction**: ~4-10ms processing time
- **Batch Processing**: Efficient parallel processing
- **Memory Usage**: In-memory logging (last 1000 predictions)
- **Model Loading**: Automatic model loading on startup

## Monitoring & Logging

- **Comprehensive Logging**: All operations logged with timestamps
- **Prediction Tracking**: In-memory storage of recent predictions
- **Health Monitoring**: Status endpoints for system health
- **Performance Metrics**: Processing time tracking

## Security Features

- **Input Validation**: Pydantic model validation
- **CORS Support**: Configurable cross-origin requests
- **Error Sanitization**: Safe error message exposure
- **Type Safety**: Strong typing throughout codebase

## Model Compatibility

- **Feature Count**: Fixed 14-feature input vector
- **Model Format**: Compatible with scikit-learn RandomForest
- **Version Handling**: Model version tracking and reporting
- **Backward Compatibility**: Graceful handling of model updates

## Future Enhancements

- Database persistence for prediction logs
- A/B testing framework for model versions
- Real-time model retraining pipeline
- Enhanced authentication and authorization
- Performance optimization and caching

---

## API Documentation

Full interactive API documentation is available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

The API follows OpenAPI 3.0 standards and includes comprehensive examples for all endpoints.