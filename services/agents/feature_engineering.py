import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineeringAgent:
    """
    Feature Engineering Agent for E-commerce Return Prediction
    
    Purpose: Create derived features from raw inputs
    Functions:
    - Calculate Total_Order_Value (price × quantity)
    - Extract temporal features (year, month, weekday)
    - Generate location encoding from user location
    - Apply discount calculations and transformations
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.location_encoder = LabelEncoder()
        # Fixed feature columns to match existing model (14 features)
        # Exact feature names expected by the trained model:
        self.feature_columns = [
            'Product_Category', 'Product_Price', 'Order_Quantity', 'Return_Reason',
            'User_Age', 'User_Gender', 'Payment_Method', 'Shipping_Method', 
            'Discount_Applied', 'Total_Order_Value', 'Order_Year', 'Order_Month',
            'Order_Weekday', 'User_Location_Num'
        ]
        
        # Pre-fit location encoder with common locations to reduce warnings
        common_locations = [
            'Unknown', 'New York', 'California', 'Texas', 'Florida', 'Illinois',
            'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia',
            'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin'
        ]
        self.location_encoder.fit(common_locations)
        
    def calculate_total_order_value(self, data: Dict[str, Any]) -> float:
        """
        Calculate Total Order Value (price × quantity)
        
        Args:
            data: Dictionary containing 'price' and 'quantity' keys
            
        Returns:
            float: Total order value
        """
        try:
            price = float(data.get('price', 0))
            quantity = int(data.get('quantity', 0))
            total_value = price * quantity
            
            logger.info(f"Calculated total order value: {total_value}")
            return total_value
            
        except (ValueError, TypeError) as e:
            logger.error(f"Error calculating total order value: {e}")
            return 0.0
    
    def extract_temporal_features(self, order_date: str) -> Dict[str, int]:
        """
        Extract temporal features (year, month, weekday) from order date
        
        Args:
            order_date: Date string in format 'YYYY-MM-DD' or datetime object
            
        Returns:
            Dict containing year, month, weekday, quarter, day_of_year
        """
        try:
            if isinstance(order_date, str):
                date_obj = datetime.strptime(order_date, '%Y-%m-%d')
            elif isinstance(order_date, datetime):
                date_obj = order_date
            else:
                date_obj = datetime.now()
                
            temporal_features = {
                'year': date_obj.year,
                'month': date_obj.month,
                'weekday': date_obj.weekday(),  # 0=Monday, 6=Sunday
                'quarter': (date_obj.month - 1) // 3 + 1,
                'day_of_year': date_obj.timetuple().tm_yday,
                'is_weekend': 1 if date_obj.weekday() >= 5 else 0,
                'is_month_start': 1 if date_obj.day <= 7 else 0,
                'is_month_end': 1 if date_obj.day >= 24 else 0
            }
            
            logger.info(f"Extracted temporal features: {temporal_features}")
            return temporal_features
            
        except Exception as e:
            logger.error(f"Error extracting temporal features: {e}")
            return {
                'year': datetime.now().year,
                'month': datetime.now().month,
                'weekday': datetime.now().weekday(),
                'quarter': 1,
                'day_of_year': 1,
                'is_weekend': 0,
                'is_month_start': 0,
                'is_month_end': 0
            }
    
    def generate_location_encoding(self, user_location: str, fit: bool = False) -> int:
        """
        Generate location encoding from user location
        
        Args:
            user_location: String representing user location (city, state, country)
            fit: Whether to fit the encoder (True for training, False for prediction)
            
        Returns:
            int: Encoded location value
        """
        try:
            if not user_location or user_location.strip() == '':
                user_location = 'Unknown'
                
            user_location = user_location.strip().title()
            
            if fit:
                # For training - fit the encoder
                if not hasattr(self.location_encoder, 'classes_'):
                    self.location_encoder.fit([user_location])
                encoded_location = self.location_encoder.transform([user_location])[0]
            else:
                # For prediction - handle unseen locations
                try:
                    encoded_location = self.location_encoder.transform([user_location])[0]
                except ValueError:
                    # Handle unseen location by assigning a default value (Unknown = 0)
                    logger.info(f"New location '{user_location}' assigned to Unknown category")
                    encoded_location = 0  # 'Unknown' is index 0 in our common_locations
                    
            logger.info(f"Encoded location '{user_location}' to: {encoded_location}")
            return int(encoded_location)
            
        except Exception as e:
            logger.error(f"Error encoding location: {e}")
            return 0
    
    def apply_discount_calculations(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Apply discount calculations and transformations
        
        Args:
            data: Dictionary containing price, discount_rate, discount_amount
            
        Returns:
            Dict containing discount features
        """
        try:
            price = float(data.get('price', 0))
            discount_rate = float(data.get('discount_rate', 0))
            discount_amount = float(data.get('discount_amount', 0))
            
            # Calculate various discount features
            discount_features = {
                'discount_rate': discount_rate,
                'discount_amount': discount_amount,
                'final_price': price - discount_amount,
                'discount_percentage': (discount_amount / price * 100) if price > 0 else 0,
                'has_discount': 1 if (discount_rate > 0 or discount_amount > 0) else 0,
                'discount_savings': discount_amount,
                'price_after_discount': max(0, price - discount_amount)
            }
            
            # Additional discount categorization
            if discount_features['discount_percentage'] >= 50:
                discount_features['discount_category'] = 3  # High discount
            elif discount_features['discount_percentage'] >= 20:
                discount_features['discount_category'] = 2  # Medium discount
            elif discount_features['discount_percentage'] > 0:
                discount_features['discount_category'] = 1  # Low discount
            else:
                discount_features['discount_category'] = 0  # No discount
                
            logger.info(f"Calculated discount features: {discount_features}")
            return discount_features
            
        except (ValueError, TypeError) as e:
            logger.error(f"Error calculating discount features: {e}")
            return {
                'discount_rate': 0.0,
                'discount_amount': 0.0,
                'final_price': 0.0,
                'discount_percentage': 0.0,
                'has_discount': 0,
                'discount_savings': 0.0,
                'price_after_discount': 0.0,
                'discount_category': 0
            }
    
    def create_categorical_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create and encode categorical features
        
        Args:
            data: Raw input data
            
        Returns:
            Dict containing encoded categorical features
        """
        try:
            categorical_features = {}
            
            # Product category encoding
            product_category = data.get('product_category', 'Unknown')
            if 'product_category' not in self.label_encoders:
                self.label_encoders['product_category'] = LabelEncoder()
                
            try:
                categorical_features['product_category_encoded'] = \
                    self.label_encoders['product_category'].transform([product_category])[0]
            except (ValueError, AttributeError):
                # Handle unseen categories
                categorical_features['product_category_encoded'] = 0
                
            # Customer segment encoding
            customer_segment = data.get('customer_segment', 'Regular')
            if customer_segment.lower() in ['premium', 'vip', 'gold']:
                categorical_features['customer_tier'] = 3
            elif customer_segment.lower() in ['silver', 'preferred']:
                categorical_features['customer_tier'] = 2
            elif customer_segment.lower() in ['bronze', 'member']:
                categorical_features['customer_tier'] = 1
            else:
                categorical_features['customer_tier'] = 0
                
            # Payment method encoding
            payment_method = data.get('payment_method', 'Unknown')
            payment_mapping = {
                'credit_card': 1, 'debit_card': 2, 'paypal': 3,
                'bank_transfer': 4, 'cash_on_delivery': 5, 'unknown': 0
            }
            categorical_features['payment_method_encoded'] = \
                payment_mapping.get(payment_method.lower(), 0)
                
            return categorical_features
            
        except Exception as e:
            logger.error(f"Error creating categorical features: {e}")
            return {
                'product_category_encoded': 0,
                'customer_tier': 0,
                'payment_method_encoded': 0
            }
    
    def create_interaction_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Create interaction features between different variables
        
        Args:
            features: Dictionary of existing features
            
        Returns:
            Dict containing interaction features
        """
        try:
            interaction_features = {}
            
            # Price and quantity interactions
            total_value = features.get('total_order_value', 0)
            quantity = features.get('quantity', 1)
            
            interaction_features['price_per_item'] = total_value / max(quantity, 1)
            
            # Temporal and discount interactions
            is_weekend = features.get('is_weekend', 0)
            has_discount = features.get('has_discount', 0)
            
            interaction_features['weekend_discount'] = is_weekend * has_discount
            
            # Customer tier and order value interaction
            customer_tier = features.get('customer_tier', 0)
            interaction_features['tier_value_ratio'] = customer_tier * total_value / 1000
            
            # Month and category interaction (seasonal effects)
            month = features.get('month', 1)
            category = features.get('product_category_encoded', 0)
            interaction_features['month_category'] = month * category
            
            return interaction_features
            
        except Exception as e:
            logger.error(f"Error creating interaction features: {e}")
            return {
                'price_per_item': 0.0,
                'weekend_discount': 0,
                'tier_value_ratio': 0.0,
                'month_category': 0
            }
    
    def process_features(self, raw_data: Dict[str, Any], fit: bool = False) -> Dict[str, Any]:
        """
        Main method to process all features from raw input data
        
        Args:
            raw_data: Raw input data dictionary
            fit: Whether to fit encoders (True for training, False for prediction)
            
        Returns:
            Dict containing the 14 engineered features with exact names expected by the model
        """
        try:
            logger.info("Starting feature engineering process")
            
            engineered_features = {}
            
            # 1. Product Category (encoded)
            product_category = raw_data.get('product_category', 'Unknown')
            if 'product_category' not in self.label_encoders:
                self.label_encoders['product_category'] = LabelEncoder()
                common_categories = ['Unknown', 'Electronics', 'Clothing', 'Home', 'Books', 'Sports', 'Beauty', 'Food']
                self.label_encoders['product_category'].fit(common_categories)
            try:
                engineered_features['Product_Category'] = \
                    self.label_encoders['product_category'].transform([product_category])[0]
            except ValueError:
                engineered_features['Product_Category'] = 0  # Unknown
            
            # 2. Product Price
            engineered_features['Product_Price'] = float(raw_data.get('price', 0))
            
            # 3. Order Quantity
            engineered_features['Order_Quantity'] = float(raw_data.get('quantity', 1))
            
            # 4. Return Reason (default to 0 for prediction, as we don't know it yet)
            engineered_features['Return_Reason'] = 0  # Default for prediction
            
            # 5. User Age
            engineered_features['User_Age'] = float(raw_data.get('customer_age', 30))
            
            # 6. User Gender (encoded)
            user_gender = raw_data.get('user_gender', 'Unknown')
            if 'user_gender' not in self.label_encoders:
                self.label_encoders['user_gender'] = LabelEncoder()
                common_genders = ['Unknown', 'Male', 'Female', 'Other']
                self.label_encoders['user_gender'].fit(common_genders)
            try:
                engineered_features['User_Gender'] = \
                    self.label_encoders['user_gender'].transform([user_gender])[0]
            except ValueError:
                engineered_features['User_Gender'] = 0  # Unknown
            
            # 7. Payment Method (encoded)
            payment_method = raw_data.get('payment_method', 'Unknown')
            if 'payment_method' not in self.label_encoders:
                self.label_encoders['payment_method'] = LabelEncoder()
                common_methods = ['Unknown', 'credit_card', 'debit_card', 'paypal', 'bank_transfer', 'cash_on_delivery']
                self.label_encoders['payment_method'].fit(common_methods)
            try:
                engineered_features['Payment_Method'] = \
                    self.label_encoders['payment_method'].transform([payment_method])[0]
            except ValueError:
                engineered_features['Payment_Method'] = 0  # Unknown
            
            # 8. Shipping Method (encoded)
            shipping_method = raw_data.get('shipping_method', 'Standard')
            if 'shipping_method' not in self.label_encoders:
                self.label_encoders['shipping_method'] = LabelEncoder()
                common_shipping = ['Standard', 'Express', 'Overnight', 'Free', 'Premium']
                self.label_encoders['shipping_method'].fit(common_shipping)
            try:
                engineered_features['Shipping_Method'] = \
                    self.label_encoders['shipping_method'].transform([shipping_method])[0]
            except ValueError:
                engineered_features['Shipping_Method'] = 0  # Standard
            
            # 9. Discount Applied (binary)
            discount_features = self.apply_discount_calculations(raw_data)
            engineered_features['Discount_Applied'] = discount_features['has_discount']
            
            # 10. Total Order Value
            engineered_features['Total_Order_Value'] = self.calculate_total_order_value(raw_data)
            
            # 11-13. Temporal features
            order_date = raw_data.get('order_date', datetime.now().strftime('%Y-%m-%d'))
            temporal_features = self.extract_temporal_features(order_date)
            engineered_features['Order_Year'] = temporal_features['year']
            engineered_features['Order_Month'] = temporal_features['month']
            engineered_features['Order_Weekday'] = temporal_features['weekday']
            
            # 14. User Location (numeric encoding)
            user_location = raw_data.get('user_location', 'Unknown')
            engineered_features['User_Location_Num'] = self.generate_location_encoding(
                user_location, fit=fit
            )
            
            # Ensure we only return the 14 expected features in the correct order
            final_features = {}
            for col in self.feature_columns:
                final_features[col] = engineered_features.get(col, 0.0)
                
            logger.info(f"Feature engineering completed. Generated {len(final_features)} features")
            return final_features
            
        except Exception as e:
            logger.error(f"Error in feature processing: {e}")
            raise
    
    def get_feature_vector(self, features: Dict[str, Any]) -> List[float]:
        """
        Convert feature dictionary to ordered feature vector
        
        Args:
            features: Dictionary of features
            
        Returns:
            List of feature values in consistent order (14 features for model compatibility)
        """
        try:
            # Fixed order of 14 features expected by the model
            feature_vector = []
            for col in self.feature_columns:
                value = features.get(col, 0)
                feature_vector.append(float(value))
                
            return feature_vector
            
        except Exception as e:
            logger.error(f"Error creating feature vector: {e}")
            return [0.0] * 14  # Return 14 zeros as fallback
    
    def fit_encoders(self, training_data: List[Dict[str, Any]]):
        """
        Fit all encoders on training data
        
        Args:
            training_data: List of training data dictionaries
        """
        try:
            logger.info("Fitting encoders on training data")
            
            # Collect all unique values for categorical features
            locations = [item.get('user_location', 'Unknown') for item in training_data]
            categories = [item.get('product_category', 'Unknown') for item in training_data]
            
            # Fit location encoder
            self.location_encoder.fit(list(set(locations)))
            
            # Fit product category encoder
            if 'product_category' not in self.label_encoders:
                self.label_encoders['product_category'] = LabelEncoder()
            self.label_encoders['product_category'].fit(list(set(categories)))
            
            logger.info("Encoders fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting encoders: {e}")
            raise

# Utility functions for the agent
def validate_input_data(data: Dict[str, Any]) -> bool:
    """
    Validate input data for feature engineering
    
    Args:
        data: Input data dictionary
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    required_fields = ['price', 'quantity']
    
    for field in required_fields:
        if field not in data:
            logger.error(f"Missing required field: {field}")
            return False
            
    try:
        float(data['price'])
        int(data['quantity'])
        return True
    except (ValueError, TypeError):
        logger.error("Invalid data types for price or quantity")
        return False

def get_default_feature_values() -> Dict[str, Any]:
    """
    Get default values for features when data is missing
    
    Returns:
        Dict with default feature values
    """
    return {
        'price': 0.0,
        'quantity': 1,
        'discount_rate': 0.0,
        'discount_amount': 0.0,
        'user_location': 'Unknown',
        'product_category': 'Unknown',
        'customer_segment': 'Regular',
        'payment_method': 'Unknown',
        'customer_age': 30,
        'days_since_last_order': 30,
        'order_date': datetime.now().strftime('%Y-%m-%d')
    }