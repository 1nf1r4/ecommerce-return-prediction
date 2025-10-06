"""
Product Intelligence Agent
Purpose: Analyze product-specific patterns and maintain product catalog
Functions:
- Get product risk profile analysis
- Update product metrics based on return data
- Analyze category patterns and trends
- Provide seasonal adjustments for predictions
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductIntelligenceAgent:
    """
    Product Intelligence Agent for product-specific analytics
    
    This agent maintains product-level insights and patterns:
    - Product risk profiles and return rates
    - Category-level analytics
    - Seasonal trends and adjustments
    - Feature importance by product category
    """
    
    def __init__(self):
        """Initialize the Product Intelligence Agent"""
        self.product_analytics = {}
        self.category_analytics = {}
        self.seasonal_patterns = {}
        self.feature_importance_by_category = {}
        self.processed_orders = 0
        self._initialize_default_analytics()
        logger.info("Product Intelligence Agent initialized")
    
    def _initialize_default_analytics(self):
        """Initialize default analytics data"""
        # Default category risk profiles based on historical e-commerce data
        self.category_analytics = {
            'Electronics': {
                'base_return_rate': 0.15,
                'avg_return_probability': 0.68,
                'total_orders': 0,
                'total_returns': 0,
                'risk_factors': ['High value', 'Complex setup', 'Compatibility issues'],
                'seasonal_multiplier': {'Q1': 0.9, 'Q2': 1.0, 'Q3': 1.1, 'Q4': 1.2}
            },
            'Clothing': {
                'base_return_rate': 0.25,
                'avg_return_probability': 0.72,
                'total_orders': 0,
                'total_returns': 0,
                'risk_factors': ['Size issues', 'Style mismatch', 'Quality concerns'],
                'seasonal_multiplier': {'Q1': 1.0, 'Q2': 0.9, 'Q3': 1.1, 'Q4': 1.3}
            },
            'Books': {
                'base_return_rate': 0.05,
                'avg_return_probability': 0.62,
                'total_orders': 0,
                'total_returns': 0,
                'risk_factors': ['Wrong edition', 'Damage during shipping'],
                'seasonal_multiplier': {'Q1': 1.0, 'Q2': 0.8, 'Q3': 1.2, 'Q4': 0.9}
            },
            'Home & Garden': {
                'base_return_rate': 0.12,
                'avg_return_probability': 0.66,
                'total_orders': 0,
                'total_returns': 0,
                'risk_factors': ['Size/fit issues', 'Quality expectations'],
                'seasonal_multiplier': {'Q1': 0.8, 'Q2': 1.2, 'Q3': 1.1, 'Q4': 0.9}
            },
            'Sports': {
                'base_return_rate': 0.18,
                'avg_return_probability': 0.69,
                'total_orders': 0,
                'total_returns': 0,
                'risk_factors': ['Size/fit issues', 'Performance expectations'],
                'seasonal_multiplier': {'Q1': 1.3, 'Q2': 1.0, 'Q3': 0.8, 'Q4': 1.1}
            },
            'Beauty': {
                'base_return_rate': 0.20,
                'avg_return_probability': 0.70,
                'total_orders': 0,
                'total_returns': 0,
                'risk_factors': ['Skin reactions', 'Color mismatch', 'Expiration concerns'],
                'seasonal_multiplier': {'Q1': 0.9, 'Q2': 1.0, 'Q3': 1.0, 'Q4': 1.2}
            },
            'Toys': {
                'base_return_rate': 0.22,
                'avg_return_probability': 0.71,
                'total_orders': 0,
                'total_returns': 0,
                'risk_factors': ['Safety concerns', 'Age appropriateness', 'Quality issues'],
                'seasonal_multiplier': {'Q1': 0.7, 'Q2': 0.8, 'Q3': 1.0, 'Q4': 1.8}
            },
            'Automotive': {
                'base_return_rate': 0.10,
                'avg_return_probability': 0.65,
                'total_orders': 0,
                'total_returns': 0,
                'risk_factors': ['Compatibility issues', 'Wrong part'],
                'seasonal_multiplier': {'Q1': 1.0, 'Q2': 1.1, 'Q3': 1.2, 'Q4': 0.9}
            },
            'Health': {
                'base_return_rate': 0.15,
                'avg_return_probability': 0.67,
                'total_orders': 0,
                'total_returns': 0,
                'risk_factors': ['Expiration dates', 'Side effects', 'Effectiveness'],
                'seasonal_multiplier': {'Q1': 1.2, 'Q2': 0.9, 'Q3': 0.9, 'Q4': 1.1}
            },
            'Home': {
                'base_return_rate': 0.12,
                'avg_return_probability': 0.66,
                'total_orders': 0,
                'total_returns': 0,
                'risk_factors': ['Size/fit issues', 'Quality expectations'],
                'seasonal_multiplier': {'Q1': 0.8, 'Q2': 1.2, 'Q3': 1.1, 'Q4': 0.9}
            }
        }
        
        # Default feature importance by category
        self.feature_importance_by_category = {
            'Electronics': {
                'Product_Price': 0.35,
                'User_Age': 0.20,
                'Payment_Method': 0.15,
                'Order_Quantity': 0.12,
                'Discount_Applied': 0.10,
                'Shipping_Method': 0.08
            },
            'Clothing': {
                'Product_Price': 0.25,
                'User_Age': 0.30,
                'User_Gender': 0.20,
                'Order_Quantity': 0.10,
                'Discount_Applied': 0.10,
                'Shipping_Method': 0.05
            },
            'Books': {
                'Product_Price': 0.20,
                'User_Age': 0.35,
                'Payment_Method': 0.15,
                'Order_Quantity': 0.15,
                'Discount_Applied': 0.10,
                'Shipping_Method': 0.05
            }
        }
    
    def get_product_risk_profile(self, product_sku: Optional[str] = None, 
                                category: Optional[str] = None,
                                price: Optional[float] = None) -> Dict[str, Any]:
        """
        Get risk profile for a specific product or category
        
        Args:
            product_sku: Specific product SKU (optional)
            category: Product category
            price: Product price for price-based analysis
            
        Returns:
            Dictionary containing risk profile analysis
        """
        try:
            if product_sku and product_sku in self.product_analytics:
                # Return specific product analytics
                product_data = self.product_analytics[product_sku]
                return {
                    'product_sku': product_sku,
                    'category': product_data.get('category', 'Unknown'),
                    'return_rate': product_data.get('return_rate', 0.0),
                    'avg_return_probability': product_data.get('avg_return_probability', 0.7),
                    'total_orders': product_data.get('total_orders', 0),
                    'risk_level': self._determine_risk_level(product_data.get('return_rate', 0.0)),
                    'confidence': min(100, product_data.get('total_orders', 0) * 2),  # Confidence based on data volume
                    'last_updated': product_data.get('last_updated', datetime.now().isoformat())
                }
            
            elif category and category in self.category_analytics:
                # Return category-level analytics
                category_data = self.category_analytics[category]
                
                # Adjust risk based on price if provided
                adjusted_probability = category_data['avg_return_probability']
                if price:
                    price_adjustment = self._calculate_price_risk_adjustment(price, category)
                    adjusted_probability = min(0.85, max(0.55, adjusted_probability + price_adjustment))
                
                return {
                    'category': category,
                    'base_return_rate': category_data['base_return_rate'],
                    'avg_return_probability': adjusted_probability,
                    'risk_factors': category_data['risk_factors'],
                    'risk_level': self._determine_risk_level(category_data['base_return_rate']),
                    'seasonal_patterns': category_data['seasonal_multiplier'],
                    'total_category_orders': category_data['total_orders'],
                    'price_adjusted': price is not None
                }
            
            else:
                # Return default risk profile
                return {
                    'category': category or 'Unknown',
                    'base_return_rate': 0.15,
                    'avg_return_probability': 0.68,
                    'risk_level': 'MEDIUM',
                    'confidence': 0,
                    'note': 'Using default risk profile - insufficient data'
                }
                
        except Exception as e:
            logger.error(f"Error getting product risk profile: {str(e)}")
            return {
                'error': str(e),
                'risk_level': 'MEDIUM',
                'avg_return_probability': 0.68
            }
    
    def _calculate_price_risk_adjustment(self, price: float, category: str) -> float:
        """
        Calculate risk adjustment based on product price within category
        
        Args:
            price: Product price
            category: Product category
            
        Returns:
            Risk adjustment factor (-0.15 to +0.15)
        """
        # Price thresholds by category (approximate)
        price_thresholds = {
            'Electronics': {'low': 50, 'medium': 200, 'high': 500},
            'Clothing': {'low': 25, 'medium': 75, 'high': 150},
            'Books': {'low': 10, 'medium': 25, 'high': 50},
            'Home & Garden': {'low': 30, 'medium': 100, 'high': 250},
            'Sports': {'low': 25, 'medium': 100, 'high': 300},
            'Beauty': {'low': 15, 'medium': 50, 'high': 100},
            'Toys': {'low': 10, 'medium': 50, 'high': 150},
            'Automotive': {'low': 20, 'medium': 100, 'high': 300},
            'Health': {'low': 10, 'medium': 30, 'high': 75},
            'Home': {'low': 30, 'medium': 100, 'high': 250}
        }
        
        thresholds = price_thresholds.get(category, {'low': 25, 'medium': 75, 'high': 200})
        
        if price <= thresholds['low']:
            return -0.05  # Lower risk for low-priced items
        elif price <= thresholds['medium']:
            return 0.0   # Baseline risk
        elif price <= thresholds['high']:
            return 0.08  # Higher risk for expensive items
        else:
            return 0.15  # Highest risk for very expensive items
    
    def _determine_risk_level(self, return_rate: float) -> str:
        """
        Determine risk level based on return rate
        
        Args:
            return_rate: Historical return rate
            
        Returns:
            Risk level string
        """
        if return_rate <= 0.1:
            return 'LOW'
        elif return_rate <= 0.2:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def update_product_metrics(self, product_sku: str, category: str, 
                             order_data: Dict[str, Any], actual_return: Optional[bool] = None) -> bool:
        """
        Update product metrics based on new order or return data
        
        Args:
            product_sku: Product SKU
            category: Product category
            order_data: Order information
            actual_return: Whether the order was actually returned (None if unknown)
            
        Returns:
            Success status
        """
        try:
            # Initialize product entry if not exists
            if product_sku not in self.product_analytics:
                self.product_analytics[product_sku] = {
                    'category': category,
                    'total_orders': 0,
                    'total_returns': 0,
                    'return_rate': 0.0,
                    'avg_return_probability': 0.0,
                    'price_history': [],
                    'created_at': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat()
                }
            
            product_metrics = self.product_analytics[product_sku]
            
            # Update order count
            product_metrics['total_orders'] += 1
            
            # Update return count if actual return data is provided
            if actual_return is not None:
                if actual_return:
                    product_metrics['total_returns'] += 1
                
                # Recalculate return rate
                product_metrics['return_rate'] = product_metrics['total_returns'] / product_metrics['total_orders']
            
            # Update price history
            if 'price' in order_data:
                product_metrics['price_history'].append({
                    'price': order_data['price'],
                    'date': datetime.now().isoformat()
                })
                
                # Keep only last 100 price records
                if len(product_metrics['price_history']) > 100:
                    product_metrics['price_history'] = product_metrics['price_history'][-100:]
            
            # Update timestamp
            product_metrics['last_updated'] = datetime.now().isoformat()
            
            # Update category-level metrics
            self._update_category_metrics(category, actual_return)
            
            self.processed_orders += 1
            logger.info(f"Updated metrics for product {product_sku}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating product metrics: {str(e)}")
            return False
    
    def _update_category_metrics(self, category: str, actual_return: Optional[bool] = None):
        """
        Update category-level metrics
        
        Args:
            category: Product category
            actual_return: Whether the order was actually returned (None if unknown)
        """
        if category in self.category_analytics:
            self.category_analytics[category]['total_orders'] += 1
            
            if actual_return is not None and actual_return:
                self.category_analytics[category]['total_returns'] += 1
                
                # Recalculate category return rate
                total_orders = self.category_analytics[category]['total_orders']
                total_returns = self.category_analytics[category]['total_returns']
                if total_orders > 0:
                    self.category_analytics[category]['base_return_rate'] = total_returns / total_orders
    
    def analyze_category_patterns(self, category: str) -> Dict[str, Any]:
        """
        Analyze patterns and trends for a specific category
        
        Args:
            category: Product category to analyze
            
        Returns:
            Dictionary containing category analysis
        """
        try:
            if category not in self.category_analytics:
                return {
                    'error': f'Category {category} not found in analytics',
                    'available_categories': list(self.category_analytics.keys())
                }
            
            category_data = self.category_analytics[category]
            
            # Get products in this category
            category_products = {
                sku: data for sku, data in self.product_analytics.items()
                if data.get('category') == category
            }
            
            # Calculate category insights
            total_products = len(category_products)
            
            if total_products > 0:
                # Calculate average return rate across products
                product_return_rates = [
                    data.get('return_rate', 0.0) for data in category_products.values()
                    if data.get('total_orders', 0) >= 5  # Only products with sufficient data
                ]
                
                avg_product_return_rate = np.mean(product_return_rates) if product_return_rates else category_data['base_return_rate']
                
                # Calculate price distribution
                all_prices = []
                for data in category_products.values():
                    all_prices.extend([p['price'] for p in data.get('price_history', [])])
                
                price_stats = {}
                if all_prices:
                    price_stats = {
                        'avg_price': np.mean(all_prices),
                        'median_price': np.median(all_prices),
                        'min_price': np.min(all_prices),
                        'max_price': np.max(all_prices),
                        'price_std': np.std(all_prices)
                    }
            else:
                avg_product_return_rate = category_data['base_return_rate']
                price_stats = {}
            
            return {
                'category': category,
                'total_products_tracked': total_products,
                'category_return_rate': category_data['base_return_rate'],
                'avg_product_return_rate': avg_product_return_rate,
                'total_category_orders': category_data['total_orders'],
                'total_category_returns': category_data['total_returns'],
                'risk_factors': category_data['risk_factors'],
                'seasonal_patterns': category_data['seasonal_multiplier'],
                'price_statistics': price_stats,
                'feature_importance': self.feature_importance_by_category.get(category, {}),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing category patterns: {str(e)}")
            return {'error': str(e)}
    
    def get_seasonal_adjustments(self, category: str, order_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get seasonal adjustment factors for predictions
        
        Args:
            category: Product category
            order_date: Order date (uses current date if None)
            
        Returns:
            Dictionary containing seasonal adjustments
        """
        try:
            if order_date is None:
                order_date = datetime.now()
            
            # Determine quarter
            month = order_date.month
            if month in [1, 2, 3]:
                quarter = 'Q1'
            elif month in [4, 5, 6]:
                quarter = 'Q2'
            elif month in [7, 8, 9]:
                quarter = 'Q3'
            else:
                quarter = 'Q4'
            
            # Get seasonal multiplier for category
            if category in self.category_analytics:
                seasonal_multiplier = self.category_analytics[category]['seasonal_multiplier'].get(quarter, 1.0)
            else:
                seasonal_multiplier = 1.0
            
            # Calculate additional seasonal factors
            seasonal_factors = {
                'quarter': quarter,
                'seasonal_multiplier': seasonal_multiplier,
                'month': month,
                'is_holiday_season': quarter == 'Q4',
                'is_back_to_school': month in [8, 9],
                'is_spring_cleaning': month in [3, 4, 5],
                'adjusted_return_probability': None  # Will be calculated when base probability is known
            }
            
            return seasonal_factors
            
        except Exception as e:
            logger.error(f"Error getting seasonal adjustments: {str(e)}")
            return {'seasonal_multiplier': 1.0, 'quarter': 'Q1'}
    
    def get_feature_importance(self, category: str) -> Dict[str, float]:
        """
        Get feature importance weights for a specific category
        
        Args:
            category: Product category
            
        Returns:
            Dictionary of feature importance weights
        """
        return self.feature_importance_by_category.get(category, {
            'Product_Price': 0.25,
            'User_Age': 0.25,
            'Payment_Method': 0.15,
            'Order_Quantity': 0.15,
            'Discount_Applied': 0.10,
            'Shipping_Method': 0.10
        })
    
    def generate_insights_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive insights report
        
        Returns:
            Dictionary containing insights and recommendations
        """
        try:
            # Overall statistics
            total_products = len(self.product_analytics)
            total_categories = len(self.category_analytics)
            
            # Category rankings by risk
            category_risks = [
                {
                    'category': cat,
                    'return_rate': data['base_return_rate'],
                    'total_orders': data['total_orders']
                }
                for cat, data in self.category_analytics.items()
            ]
            category_risks.sort(key=lambda x: x['return_rate'], reverse=True)
            
            # Top risk categories
            high_risk_categories = [cat for cat in category_risks if cat['return_rate'] > 0.2]
            
            # Product insights
            products_with_data = {
                sku: data for sku, data in self.product_analytics.items()
                if data.get('total_orders', 0) >= 10
            }
            
            high_risk_products = [
                sku for sku, data in products_with_data.items()
                if data.get('return_rate', 0) > 0.3
            ]
            
            # Recommendations
            recommendations = []
            
            if high_risk_categories:
                recommendations.append(f"Monitor {len(high_risk_categories)} high-risk categories closely")
            
            if high_risk_products:
                recommendations.append(f"Review {len(high_risk_products)} products with high return rates")
            
            if total_products < 50:
                recommendations.append("Increase product tracking to improve prediction accuracy")
            
            return {
                'summary': {
                    'total_products_tracked': total_products,
                    'total_categories': total_categories,
                    'total_orders_processed': self.processed_orders
                },
                'category_risk_ranking': category_risks[:5],  # Top 5 by risk
                'high_risk_categories': high_risk_categories,
                'high_risk_products': high_risk_products[:10],  # Top 10
                'recommendations': recommendations,
                'report_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating insights report: {str(e)}")
            return {'error': str(e)}
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics and status
        
        Returns:
            Dictionary with agent statistics
        """
        return {
            'agent_name': 'ProductIntelligenceAgent',
            'products_tracked': len(self.product_analytics),
            'categories_tracked': len(self.category_analytics),
            'orders_processed': self.processed_orders,
            'status': 'active',
            'last_updated': datetime.now().isoformat()
        }


# Global instance for easy access
_product_intelligence_agent = None

def get_product_intelligence_agent() -> ProductIntelligenceAgent:
    """
    Get or create the global ProductIntelligenceAgent instance
    
    Returns:
        ProductIntelligenceAgent instance
    """
    global _product_intelligence_agent
    if _product_intelligence_agent is None:
        _product_intelligence_agent = ProductIntelligenceAgent()
    return _product_intelligence_agent


# Example usage and testing
if __name__ == "__main__":
    # Initialize agent
    agent = ProductIntelligenceAgent()
    
    # Test product risk profile
    risk_profile = agent.get_product_risk_profile(category='Electronics', price=299.99)
    print("Electronics Risk Profile:")
    print(json.dumps(risk_profile, indent=2))
    
    # Test category analysis
    category_analysis = agent.analyze_category_patterns('Clothing')
    print("\nClothing Category Analysis:")
    print(json.dumps(category_analysis, indent=2))
    
    # Test seasonal adjustments
    seasonal_adj = agent.get_seasonal_adjustments('Toys')
    print("\nSeasonal Adjustments for Toys:")
    print(json.dumps(seasonal_adj, indent=2))
    
    # Generate insights report
    insights = agent.generate_insights_report()
    print("\nInsights Report:")
    print(json.dumps(insights, indent=2))
