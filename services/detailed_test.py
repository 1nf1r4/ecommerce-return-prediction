import requests
import json

def test_feature_engineering_detailed():
    """Test Feature Engineering Agent with detailed output"""
    
    print("=== FEATURE ENGINEERING AGENT TEST ===")
    
    # Test data with comprehensive fields
    test_data = {
        'price': 75.99,
        'quantity': 1,
        'product_category': 'Electronics',
        'user_location': 'California',
        'customer_age': 28,
        'user_gender': 'Male',
        'payment_method': 'credit_card',
        'shipping_method': 'Express',
        'discount_amount': 5.0,
        'order_date': '2025-01-15'
    }
    
    try:
        response = requests.post('http://localhost:8000/predict', json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ SUCCESS!")
            print(f"Return Probability: {result['return_probability']:.3f}")
            print(f"Confidence Score: {result['confidence_score']:.3f}")
            print(f"Risk Category: {result['risk_category']}")
            print(f"Processing Time: {result['processing_time_ms']:.2f}ms")
            print(f"Features Generated: {len(result['features_used'])}")
            print()
            
            print("=== FEATURE BREAKDOWN ===")
            features = result['features_used']
            
            # Check memory specification compliance
            expected_features = [
                'Product_Category', 'Product_Price', 'Order_Quantity', 'Return_Reason',
                'User_Age', 'User_Gender', 'Payment_Method', 'Shipping_Method', 
                'Discount_Applied', 'Total_Order_Value', 'Order_Year', 'Order_Month',
                'Order_Weekday', 'User_Location_Num'
            ]
            
            print("✅ MEMORY SPECIFICATION COMPLIANCE CHECK:")
            missing_features = []
            for expected in expected_features:
                if expected in features:
                    print(f"  ✅ {expected:20}: {features[expected]}")
                else:
                    missing_features.append(expected)
                    print(f"  ❌ {expected:20}: MISSING")
            
            if missing_features:
                print(f"\n❌ Missing features: {missing_features}")
            else:
                print(f"\n✅ All 14 features present and correctly named!")
                
            # Validate key calculations
            expected_total = test_data['price'] * test_data['quantity']
            actual_total = features.get('Total_Order_Value', 0)
            print(f"\n=== CALCULATION VALIDATION ===")
            print(f"Expected Total Order Value: ${expected_total:.2f}")
            print(f"Actual Total Order Value: ${actual_total:.2f}")
            print(f"Calculation Correct: {'✅' if abs(expected_total - actual_total) < 0.01 else '❌'}")
            
        else:
            print(f"❌ ERROR: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")

if __name__ == "__main__":
    test_feature_engineering_detailed()