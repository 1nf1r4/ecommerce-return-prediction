import requests
import json
from datetime import datetime

# Base URL for the API
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_single_prediction():
    """Test single prediction endpoint"""
    print("Testing single prediction...")
    
    # Sample prediction request
    prediction_data = {
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
        "order_date": "2024-01-15"
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=prediction_data)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction ID: {result['prediction_id']}")
        print(f"Return Probability: {result['return_probability']:.3f}")
        print(f"Confidence Score: {result['confidence_score']:.3f}")
        print(f"Risk Category: {result['risk_category']}")
        print(f"Processing Time: {result['processing_time_ms']:.2f}ms")
        print(f"Features Used: {len(result['features_used'])} features")
    else:
        print(f"Error: {response.text}")
    print("-" * 50)

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("Testing batch prediction...")
    
    batch_data = {
        "predictions": [
            {
                "price": 99.99,
                "quantity": 2,
                "product_category": "Electronics",
                "user_location": "New York"
            },
            {
                "price": 49.99,
                "quantity": 1,
                "product_category": "Clothing",
                "user_location": "California"
            },
            {
                "price": 149.99,
                "quantity": 3,
                "product_category": "Home",
                "user_location": "Texas",
                "discount_amount": 30.0
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=batch_data)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Total Processed: {result['total_processed']}")
        print(f"Total Requested: {result['total_requested']}")
        
        for i, prediction in enumerate(result['predictions']):
            print(f"  Prediction {i+1}:")
            print(f"    Return Probability: {prediction['return_probability']:.3f}")
            print(f"    Risk Category: {prediction['risk_category']}")
    else:
        print(f"Error: {response.text}")
    print("-" * 50)

def test_model_info():
    """Test model info endpoint"""
    print("Testing model info...")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_api_status():
    """Test API status endpoint"""
    print("Testing API status...")
    response = requests.get(f"{BASE_URL}/api/status")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_feature_engineering():
    """Test feature engineering with different input scenarios"""
    print("Testing feature engineering with various scenarios...")
    
    test_cases = [
        {
            "name": "Minimal Input",
            "data": {"price": 25.99, "quantity": 1}
        },
        {
            "name": "High Value Order",
            "data": {
                "price": 999.99,
                "quantity": 5,
                "discount_amount": 100.0,
                "customer_segment": "VIP",
                "user_location": "Los Angeles"
            }
        },
        {
            "name": "Weekend Order",
            "data": {
                "price": 75.50,
                "quantity": 2,
                "order_date": "2024-01-13",  # Saturday
                "product_category": "Sports"
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"Testing: {test_case['name']}")
        response = requests.post(f"{BASE_URL}/predict", json=test_case['data'])
        
        if response.status_code == 200:
            result = response.json()
            print(f"  Return Probability: {result['return_probability']:.3f}")
            print(f"  Risk Category: {result['risk_category']}")
            
            # Show some interesting features (using correct feature names from memory specification)
            features = result['features_used']
            print(f"  Total Order Value: ${features.get('Total_Order_Value', 0):.2f}")
            print(f"  Has Discount: {features.get('Discount_Applied', 0)}")
            print(f"  Is Weekend: {features.get('Order_Weekday', 0) in [5, 6]}")
            print(f"  User Age: {features.get('User_Age', 0)}")
        else:
            print(f"  Error: {response.text}")
        print()
    print("-" * 50)

if __name__ == "__main__":
    print("=" * 60)
    print("E-COMMERCE RETURN PREDICTION API TEST SUITE")
    print("=" * 60)
    
    try:
        # Run all tests
        test_health_check()
        test_api_status()
        test_model_info()
        test_single_prediction()
        test_batch_prediction()
        test_feature_engineering()
        
        print("✅ All tests completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API server.")
        print("Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error during testing: {e}")