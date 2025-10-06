"""
Order Processing Agent
Purpose: Handle real-time order ingestion, validation, and preprocessing
Functions:
- Validate incoming order data
- Process and clean order information
- Extract features for model prediction
- Integrate with ModelInferenceAgent
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderData(BaseModel):
    """Simple order data model for validation"""
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

    @field_validator('product_category')
    def validate_category(cls, v):
        allowed_categories = [
            'Electronics', 'Clothing', 'Books', 'Home & Garden', 
            'Sports', 'Beauty', 'Toys', 'Automotive', 'Health', 'Home'
        ]
        if v not in allowed_categories:
            logger.warning(f"Unusual category: {v}, but allowing it")
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
            'Bank Transfer', 'Cash', 'Digital Wallet', 'Gift Card'
        ]
        if v not in allowed_methods:
            logger.warning(f"Unusual payment method: {v}, but allowing it")
        return v

    @field_validator('shipping_method')
    def validate_shipping_method(cls, v):
        allowed_methods = ['Standard', 'Express', 'Next-Day']
        if v not in allowed_methods:
            logger.warning(f"Unusual shipping method: {v}, defaulting to Standard")
            return 'Standard'
        return v


class OrderProcessingAgent:
    """
    Simple Order Processing Agent
    
    Handles order validation, processing, and feature preparation
    for the return prediction model.
    """
    
    def __init__(self):
        """Initialize the Order Processing Agent"""
        self.processed_count = 0
        logger.info("Order Processing Agent initialized")
    
    def validate_order_data(self, order_data: Dict[str, Any]) -> Tuple[bool, Optional[OrderData], Optional[str]]:
        """
        Validate incoming order data
        
        Args:
            order_data: Raw order data dictionary
            
        Returns:
            Tuple of (is_valid, validated_data, error_message)
        """
        try:
            # Use Pydantic for validation
            validated_order = OrderData(**order_data)
            return True, validated_order, None
        
        except Exception as e:
            error_msg = f"Validation error: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def _get_basic_mappings(self) -> Dict[str, Dict[str, int]]:
        """Get basic categorical mappings (moved complex encoding to FeatureEngineeringAgent)"""
        return {
            'category': {'Electronics': 1, 'Clothing': 2, 'Books': 3, 'Home': 4, 'Toys': 5,
                        'Sports': 6, 'Beauty': 7, 'Automotive': 8, 'Health': 9, 'Home & Garden': 4},
            'gender': {'Male': 1, 'Female': 2, 'Other': 0},
            'payment': {'Credit Card': 1, 'Debit Card': 2, 'PayPal': 3, 'Bank Transfer': 4,
                       'Cash': 5, 'Digital Wallet': 6, 'Gift Card': 7},
            'shipping': {'Standard': 1, 'Express': 2, 'Next-Day': 3}
        }

    def extract_basic_features(self, order_data: OrderData) -> Dict[str, Any]:
        """
        Extract basic features from validated order data (complex feature engineering moved to FeatureEngineeringAgent)
        
        Args:
            order_data: Validated order data
            
        Returns:
            Dictionary of basic features
        """
        try:
            # Get basic categorical mappings
            mappings = self._get_basic_mappings()
            
            # Create basic feature dictionary
            features = {
                'Product_Category': mappings['category'].get(order_data.product_category, 1),
                'Product_Price': float(order_data.price),
                'Order_Quantity': int(order_data.quantity),
                'User_Age': int(order_data.age),
                'User_Gender': mappings['gender'].get(order_data.gender, 0),
                'Payment_Method': mappings['payment'].get(order_data.payment_method, 1),
                'Shipping_Method': mappings['shipping'].get(order_data.shipping_method or 'Standard', 1),
                'Discount_Applied': float(order_data.discount_applied or 0.0),
                'User_Location_Num': 1  # Simplified location encoding
            }
            
            # Add basic derived features
            features['Total_Order_Value'] = features['Product_Price'] * features['Order_Quantity']
            
            # Add temporal features if order_date provided
            if order_data.order_date:
                try:
                    order_dt = datetime.strptime(order_data.order_date, '%Y-%m-%d')
                    features['Order_Year'] = int(order_dt.year)
                    features['Order_Month'] = int(order_dt.month)
                    features['Order_Weekday'] = int(order_dt.weekday())
                except ValueError:
                    logger.warning(f"Invalid date format: {order_data.order_date}, using current date")
                    current_dt = datetime.now()
                    features['Order_Year'] = int(current_dt.year)
                    features['Order_Month'] = int(current_dt.month)
                    features['Order_Weekday'] = int(current_dt.weekday())
            else:
                current_dt = datetime.now()
                features['Order_Year'] = int(current_dt.year)
                features['Order_Month'] = int(current_dt.month)
                features['Order_Weekday'] = int(current_dt.weekday())
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting basic features: {str(e)}")
            raise
    
    def prepare_for_prediction(self, features: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare basic features as DataFrame (advanced feature engineering done by FeatureEngineeringAgent)
        
        Args:
            features: Basic extracted features dictionary
            
        Returns:
            DataFrame with basic features
        """
        try:
            # Create DataFrame with basic features (no advanced engineering here)
            df = pd.DataFrame([features])
            return df
            
        except Exception as e:
            logger.error(f"Error preparing data for prediction: {str(e)}")
            raise
    
    def process_single_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single order through the complete pipeline
        
        Args:
            order_data: Raw order data
            
        Returns:
            Dictionary with processed order and features
        """
        try:
            # Update processed count first
            self.processed_count += 1
            
            # Generate order ID if not provided
            order_id = order_data.get('order_id')
            if not order_id:
                # Generate a timestamp-based order ID
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                order_id = f"ORD_{timestamp}_{self.processed_count:04d}"
            
            # Step 1: Validate order data
            is_valid, validated_data, error_msg = self.validate_order_data(order_data)
            
            if not is_valid:
                return {
                    'success': False,
                    'error': error_msg,
                    'order_id': order_id
                }
            
            # Step 2: Extract basic features  
            features = self.extract_basic_features(validated_data)
            
            # Step 3: Prepare for prediction
            prediction_df = self.prepare_for_prediction(features)
            
            return {
                'success': True,
                'order_id': order_id,
                'validated_data': validated_data.model_dump(),
                'features': features,
                'prediction_ready_data': prediction_df,
                'processing_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Error processing order: {str(e)}"
            logger.error(error_msg)
            # Ensure we always have a valid order_id even in error cases
            order_id = order_data.get('order_id', f'error_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            return {
                'success': False,
                'error': error_msg,
                'order_id': order_id
            }
    
    def process_batch_orders(self, orders_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process multiple orders in batch
        
        Args:
            orders_list: List of raw order data dictionaries
            
        Returns:
            Dictionary with batch processing results
        """
        try:
            results = []
            successful_orders = []
            failed_orders = []
            
            for i, order_data in enumerate(orders_list):
                # Ensure order has an ID (will be generated in process_single_order if not provided)
                if 'order_id' not in order_data or not order_data['order_id']:
                    order_data['order_id'] = f'batch_{i+1:03d}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
                
                result = self.process_single_order(order_data)
                results.append(result)
                
                if result['success']:
                    successful_orders.append(result)
                else:
                    failed_orders.append(result)
            
            return {
                'success': True,
                'batch_size': len(orders_list),
                'successful_count': len(successful_orders),
                'failed_count': len(failed_orders),
                'results': results,
                'successful_orders': successful_orders,
                'failed_orders': failed_orders,
                'processing_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Error processing batch orders: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'batch_size': len(orders_list) if orders_list else 0
            }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics
        
        Returns:
            Dictionary with agent statistics
        """
        return {
            'agent_name': 'OrderProcessingAgent',
            'total_processed': self.processed_count,
            'status': 'active',
            'last_updated': datetime.now().isoformat()
        }


# Global instance for easy access
_order_processing_agent = None

def get_order_processing_agent() -> OrderProcessingAgent:
    """
    Get or create the global OrderProcessingAgent instance
    
    Returns:
        OrderProcessingAgent instance
    """
    global _order_processing_agent
    if _order_processing_agent is None:
        _order_processing_agent = OrderProcessingAgent()
    return _order_processing_agent