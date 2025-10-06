"""
Business Intelligence Agent
Purpose: Generate actionable insights and business reports
Functions:
- Calculate revenue impact from predictions
- Generate daily and periodic reports
- Analyze prediction accuracy over time
- Create executive dashboard data
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BusinessIntelligenceAgent:
    """
    Business Intelligence Agent for generating actionable insights
    
    This agent provides business-level analytics and reporting:
    - Revenue impact calculations
    - ROI analysis for return predictions
    - Performance tracking and trends
    - Executive reporting and KPIs
    """
    
    def __init__(self):
        """Initialize the Business Intelligence Agent"""
        self.prediction_history = deque(maxlen=10000)  # Store last 10k predictions
        self.daily_metrics = defaultdict(dict)
        self.accuracy_tracking = deque(maxlen=1000)  # Store accuracy measurements
        self.revenue_impact_history = deque(maxlen=365)  # Store daily revenue impacts
        self.processed_predictions = 0
        self.total_revenue_saved = 0.0
        self._initialize_default_metrics()
        logger.info("Business Intelligence Agent initialized")
    
    def _initialize_default_metrics(self):
        """Initialize default business metrics"""
        today = datetime.now().date()
        
        # Initialize current day metrics
        self.daily_metrics[today] = {
            'total_predictions': 0,
            'high_risk_predictions': 0,
            'medium_risk_predictions': 0,
            'low_risk_predictions': 0,
            'average_risk_score': 0.0,
            'revenue_at_risk': 0.0,
            'revenue_saved': 0.0,
            'orders_flagged_for_review': 0,
            'processing_time_avg': 0.0,
            'accuracy_score': 0.0,
            'created_at': datetime.now().isoformat()
        }
    
    def record_prediction(self, prediction_data: Dict[str, Any], 
                         order_data: Dict[str, Any],
                         processing_time_ms: Optional[float] = None) -> bool:
        """
        Record a new prediction for business analytics
        
        Args:
            prediction_data: Prediction results from model
            order_data: Original order data
            processing_time_ms: Time taken to process prediction
            
        Returns:
            Success status
        """
        try:
            timestamp = datetime.now()
            today = timestamp.date()
            
            # Extract prediction details
            return_probability = prediction_data.get('return_probability', 0.0)
            risk_level = self._determine_risk_level(return_probability)
            order_value = order_data.get('price', 0.0) * order_data.get('quantity', 1)
            
            # Create prediction record
            prediction_record = {
                'timestamp': timestamp.isoformat(),
                'order_id': order_data.get('order_id', f'pred_{self.processed_predictions}'),
                'return_probability': return_probability,
                'risk_level': risk_level,
                'order_value': order_value,
                'category': order_data.get('product_category', 'Unknown'),
                'customer_age': order_data.get('age', 0),
                'payment_method': order_data.get('payment_method', 'Unknown'),
                'processing_time_ms': processing_time_ms or 0.0,
                'revenue_at_risk': order_value * return_probability,
                'flagged_for_review': risk_level in ['HIGH', 'MEDIUM']
            }
            
            # Add to prediction history
            self.prediction_history.append(prediction_record)
            
            # Update daily metrics
            if today not in self.daily_metrics:
                self.daily_metrics[today] = {
                    'total_predictions': 0,
                    'high_risk_predictions': 0,
                    'medium_risk_predictions': 0,
                    'low_risk_predictions': 0,
                    'average_risk_score': 0.0,
                    'revenue_at_risk': 0.0,
                    'revenue_saved': 0.0,
                    'orders_flagged_for_review': 0,
                    'processing_time_avg': 0.0,
                    'accuracy_score': 0.0,
                    'created_at': datetime.now().isoformat()
                }
            
            daily_data = self.daily_metrics[today]
            
            # Update counters
            daily_data['total_predictions'] += 1
            
            if risk_level == 'HIGH':
                daily_data['high_risk_predictions'] += 1
            elif risk_level == 'MEDIUM':
                daily_data['medium_risk_predictions'] += 1
            else:
                daily_data['low_risk_predictions'] += 1
            
            if prediction_record['flagged_for_review']:
                daily_data['orders_flagged_for_review'] += 1
            
            # Update running averages
            total_preds = daily_data['total_predictions']
            daily_data['average_risk_score'] = (
                (daily_data['average_risk_score'] * (total_preds - 1) + return_probability) / total_preds
            )
            
            if processing_time_ms:
                daily_data['processing_time_avg'] = (
                    (daily_data['processing_time_avg'] * (total_preds - 1) + processing_time_ms) / total_preds
                )
            
            # Update revenue metrics
            daily_data['revenue_at_risk'] += prediction_record['revenue_at_risk']
            
            # Calculate potential revenue saved (estimated)
            if risk_level == 'HIGH':
                # Assume 50% of high-risk orders would have been returned without intervention
                potential_saved = order_value * 0.5
                daily_data['revenue_saved'] += potential_saved
                self.total_revenue_saved += potential_saved
            elif risk_level == 'MEDIUM':
                # Assume 25% of medium-risk orders would have been returned
                potential_saved = order_value * 0.25
                daily_data['revenue_saved'] += potential_saved
                self.total_revenue_saved += potential_saved
            
            self.processed_predictions += 1
            return True
            
        except Exception as e:
            logger.error(f"Error recording prediction: {str(e)}")
            return False
    
    def _determine_risk_level(self, probability: float) -> str:
        """Determine risk level based on return probability"""
        if probability <= 0.3:
            return 'LOW'
        elif probability <= 0.6:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def calculate_revenue_impact(self, time_period_days: int = 30) -> Dict[str, Any]:
        """
        Calculate revenue impact from recent predictions
        
        Args:
            time_period_days: Number of days to analyze
            
        Returns:
            Dictionary containing revenue impact analysis
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=time_period_days)
            
            # Filter predictions to time period
            recent_predictions = [
                pred for pred in self.prediction_history
                if datetime.fromisoformat(pred['timestamp']) >= cutoff_date
            ]
            
            if not recent_predictions:
                return {
                    'time_period_days': time_period_days,
                    'total_predictions': 0,
                    'total_revenue_analyzed': 0.0,
                    'estimated_revenue_saved': 0.0,
                    'roi_percentage': 0.0
                }
            
            # Calculate basic metrics
            total_predictions = len(recent_predictions)
            total_revenue = sum(pred['order_value'] for pred in recent_predictions)
            
            # Estimate revenue saved based on intervention
            high_risk_orders = [pred for pred in recent_predictions if pred['risk_level'] == 'HIGH']
            medium_risk_orders = [pred for pred in recent_predictions if pred['risk_level'] == 'MEDIUM']
            
            # Conservative estimates: 40% of high-risk, 20% of medium-risk would have returned
            estimated_saved_high = sum(pred['order_value'] for pred in high_risk_orders) * 0.4
            estimated_saved_medium = sum(pred['order_value'] for pred in medium_risk_orders) * 0.2
            total_estimated_saved = estimated_saved_high + estimated_saved_medium
            
            # Calculate ROI (assuming operational cost of $0.10 per prediction)
            operational_cost = total_predictions * 0.10
            roi_percentage = ((total_estimated_saved - operational_cost) / operational_cost * 100) if operational_cost > 0 else 0
            
            return {
                'time_period_days': time_period_days,
                'total_predictions': total_predictions,
                'total_revenue_analyzed': total_revenue,
                'estimated_revenue_saved': total_estimated_saved,
                'operational_cost': operational_cost,
                'net_benefit': total_estimated_saved - operational_cost,
                'roi_percentage': roi_percentage,
                'average_order_value': total_revenue / total_predictions if total_predictions > 0 else 0,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating revenue impact: {str(e)}")
            return {'error': str(e)}
    
    def generate_daily_report(self, target_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate daily business report
        
        Args:
            target_date: Date for report (uses current date if None)
            
        Returns:
            Dictionary containing daily report data
        """
        try:
            if target_date is None:
                target_date = datetime.now().date()
            else:
                target_date = target_date.date() if isinstance(target_date, datetime) else target_date
            
            # Get daily metrics
            daily_data = self.daily_metrics.get(target_date, {})
            
            if not daily_data:
                return {
                    'date': target_date.isoformat(),
                    'status': 'no_data',
                    'message': 'No predictions recorded for this date'
                }
            
            # Calculate additional metrics
            total_predictions = daily_data.get('total_predictions', 0)
            
            # Risk distribution percentages
            high_risk_pct = (daily_data.get('high_risk_predictions', 0) / total_predictions * 100) if total_predictions > 0 else 0
            medium_risk_pct = (daily_data.get('medium_risk_predictions', 0) / total_predictions * 100) if total_predictions > 0 else 0
            low_risk_pct = (daily_data.get('low_risk_predictions', 0) / total_predictions * 100) if total_predictions > 0 else 0
            
            # Performance metrics
            avg_processing_time = daily_data.get('processing_time_avg', 0.0)
            performance_rating = 'Excellent' if avg_processing_time < 100 else 'Good' if avg_processing_time < 200 else 'Needs Improvement'
            
            # Get trend data (compare with previous day)
            previous_date = target_date - timedelta(days=1)
            previous_data = self.daily_metrics.get(previous_date, {})
            
            trends = {}
            if previous_data:
                trends = {
                    'predictions_change': daily_data.get('total_predictions', 0) - previous_data.get('total_predictions', 0),
                    'revenue_at_risk_change': daily_data.get('revenue_at_risk', 0.0) - previous_data.get('revenue_at_risk', 0.0),
                    'revenue_saved_change': daily_data.get('revenue_saved', 0.0) - previous_data.get('revenue_saved', 0.0),
                    'avg_risk_score_change': daily_data.get('average_risk_score', 0.0) - previous_data.get('average_risk_score', 0.0)
                }
            
            return {
                'date': target_date.isoformat(),
                'summary': {
                    'total_predictions': total_predictions,
                    'orders_flagged_for_review': daily_data.get('orders_flagged_for_review', 0),
                    'average_risk_score': round(daily_data.get('average_risk_score', 0.0), 3),
                    'revenue_at_risk': round(daily_data.get('revenue_at_risk', 0.0), 2),
                    'estimated_revenue_saved': round(daily_data.get('revenue_saved', 0.0), 2),
                    'average_processing_time_ms': round(avg_processing_time, 2),
                    'performance_rating': performance_rating
                },
                'risk_distribution': {
                    'high_risk': {
                        'count': daily_data.get('high_risk_predictions', 0),
                        'percentage': round(high_risk_pct, 1)
                    },
                    'medium_risk': {
                        'count': daily_data.get('medium_risk_predictions', 0),
                        'percentage': round(medium_risk_pct, 1)
                    },
                    'low_risk': {
                        'count': daily_data.get('low_risk_predictions', 0),
                        'percentage': round(low_risk_pct, 1)
                    }
                },
                'trends': trends,
                'recommendations': self._generate_daily_recommendations(daily_data, trends),
                'report_generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating daily report: {str(e)}")
            return {'error': str(e)}
    
    def _generate_daily_recommendations(self, daily_data: Dict[str, Any], trends: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on daily performance
        
        Args:
            daily_data: Daily metrics
            trends: Trend data compared to previous day
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        total_predictions = daily_data.get('total_predictions', 0)
        high_risk_pct = (daily_data.get('high_risk_predictions', 0) / total_predictions * 100) if total_predictions > 0 else 0
        avg_processing_time = daily_data.get('processing_time_avg', 0.0)
        
        # Performance recommendations
        if avg_processing_time > 200:
            recommendations.append("Consider optimizing prediction processing - response times above target")
        
        # Risk level recommendations
        if high_risk_pct > 30:
            recommendations.append("High percentage of high-risk orders detected - review fulfillment processes")
        elif high_risk_pct < 5:
            recommendations.append("Very low high-risk orders - verify model calibration")
        
        # Trend-based recommendations
        if trends.get('predictions_change', 0) > 50:
            recommendations.append("Significant increase in prediction volume - monitor system performance")
        
        if trends.get('revenue_at_risk_change', 0) > 1000:
            recommendations.append("Revenue at risk increased significantly - investigate order patterns")
        
        # Volume recommendations
        if total_predictions < 10:
            recommendations.append("Low prediction volume - consider increasing marketing or check system health")
        elif total_predictions > 500:
            recommendations.append("High prediction volume - excellent adoption")
        
        if not recommendations:
            recommendations.append("All metrics within normal ranges - continue monitoring")
        
        return recommendations
    
    def analyze_prediction_accuracy(self, actual_returns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze prediction accuracy against actual return data
        
        Args:
            actual_returns: List of actual return data with order_ids
            
        Returns:
            Dictionary containing accuracy analysis
        """
        try:
            if not actual_returns:
                return {
                    'error': 'No actual return data provided',
                    'current_accuracy_estimate': 'N/A'
                }
            
            # Match predictions with actual outcomes
            matched_predictions = []
            
            for return_data in actual_returns:
                order_id = return_data.get('order_id')
                actual_returned = return_data.get('returned', False)
                
                # Find corresponding prediction
                for pred in self.prediction_history:
                    if pred['order_id'] == order_id:
                        matched_predictions.append({
                            'order_id': order_id,
                            'predicted_probability': pred['return_probability'],
                            'predicted_risk_level': pred['risk_level'],
                            'actual_returned': actual_returned,
                            'order_value': pred['order_value'],
                            'category': pred['category']
                        })
                        break
            
            if not matched_predictions:
                return {
                    'error': 'No matching predictions found for provided return data',
                    'total_return_records': len(actual_returns),
                    'suggestion': 'Ensure order_ids match between predictions and return data'
                }
            
            # Calculate accuracy metrics
            total_matched = len(matched_predictions)
            
            # Binary classification accuracy (threshold at 0.5)
            correct_predictions = sum(
                1 for pred in matched_predictions
                if (pred['predicted_probability'] > 0.5) == pred['actual_returned']
            )
            accuracy = correct_predictions / total_matched
            
            # Risk level accuracy
            high_risk_actual_returns = sum(
                1 for pred in matched_predictions
                if pred['predicted_risk_level'] == 'HIGH' and pred['actual_returned']
            )
            total_high_risk_predictions = sum(
                1 for pred in matched_predictions
                if pred['predicted_risk_level'] == 'HIGH'
            )
            high_risk_precision = (high_risk_actual_returns / total_high_risk_predictions) if total_high_risk_predictions > 0 else 0
            
            # Calculate by category
            category_accuracy = {}
            for category in set(pred['category'] for pred in matched_predictions):
                category_preds = [pred for pred in matched_predictions if pred['category'] == category]
                if category_preds:
                    category_correct = sum(
                        1 for pred in category_preds
                        if (pred['predicted_probability'] > 0.5) == pred['actual_returned']
                    )
                    category_accuracy[category] = category_correct / len(category_preds)
            
            # Update accuracy tracking
            accuracy_record = {
                'timestamp': datetime.now().isoformat(),
                'accuracy': accuracy,
                'sample_size': total_matched,
                'high_risk_precision': high_risk_precision
            }
            self.accuracy_tracking.append(accuracy_record)
            
            return {
                'analysis_summary': {
                    'total_predictions_analyzed': total_matched,
                    'overall_accuracy': round(accuracy * 100, 2),
                    'high_risk_precision': round(high_risk_precision * 100, 2),
                    'correct_predictions': correct_predictions
                },
                'category_accuracy': {
                    cat: round(acc * 100, 2) for cat, acc in category_accuracy.items()
                },
                'detailed_results': matched_predictions,
                'accuracy_trend': list(self.accuracy_tracking)[-10:],  # Last 10 accuracy measurements
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing prediction accuracy: {str(e)}")
            return {'error': str(e)}
    
    def create_executive_dashboard_data(self) -> Dict[str, Any]:
        """
        Create data for executive dashboard
        
        Returns:
            Dictionary containing executive-level KPIs and insights
        """
        try:
            # Calculate metrics for different time periods
            last_7_days = datetime.now() - timedelta(days=7)
            last_30_days = datetime.now() - timedelta(days=30)
            
            # Get recent predictions
            recent_predictions = [
                pred for pred in self.prediction_history
                if datetime.fromisoformat(pred['timestamp']) >= last_7_days
            ]
            
            monthly_predictions = [
                pred for pred in self.prediction_history
                if datetime.fromisoformat(pred['timestamp']) >= last_30_days
            ]
            
            # Calculate KPIs
            total_predictions_7d = len(recent_predictions)
            total_predictions_30d = len(monthly_predictions)
            
            # Revenue metrics
            revenue_at_risk_7d = sum(pred['revenue_at_risk'] for pred in recent_predictions)
            revenue_saved_7d = sum(
                pred['order_value'] * (0.4 if pred['risk_level'] == 'HIGH' else 0.2 if pred['risk_level'] == 'MEDIUM' else 0)
                for pred in recent_predictions
            )
            
            # Risk distribution
            high_risk_7d = len([pred for pred in recent_predictions if pred['risk_level'] == 'HIGH'])
            medium_risk_7d = len([pred for pred in recent_predictions if pred['risk_level'] == 'MEDIUM'])
            low_risk_7d = len([pred for pred in recent_predictions if pred['risk_level'] == 'LOW'])
            
            # Performance metrics
            avg_processing_time = np.mean([pred.get('processing_time_ms', 0) for pred in recent_predictions]) if recent_predictions else 0
            
            # Category performance
            category_stats = defaultdict(lambda: {'orders': 0, 'high_risk': 0, 'revenue_at_risk': 0.0})
            for pred in recent_predictions:
                category = pred['category']
                category_stats[category]['orders'] += 1
                if pred['risk_level'] == 'HIGH':
                    category_stats[category]['high_risk'] += 1
                category_stats[category]['revenue_at_risk'] += pred['revenue_at_risk']
            
            # Calculate category risk rates
            category_risk_rates = {}
            for category, stats in category_stats.items():
                if stats['orders'] > 0:
                    category_risk_rates[category] = {
                        'total_orders': stats['orders'],
                        'high_risk_rate': (stats['high_risk'] / stats['orders']) * 100,
                        'revenue_at_risk': stats['revenue_at_risk']
                    }
            
            # Trends (compare week-over-week)
            previous_week = [
                pred for pred in self.prediction_history
                if last_7_days - timedelta(days=7) <= datetime.fromisoformat(pred['timestamp']) < last_7_days
            ]
            
            prediction_trend = ((total_predictions_7d - len(previous_week)) / len(previous_week) * 100) if previous_week else 0
            
            # Latest accuracy if available
            latest_accuracy = self.accuracy_tracking[-1]['accuracy'] * 100 if self.accuracy_tracking else None
            
            return {
                'kpis': {
                    'total_predictions_7d': total_predictions_7d,
                    'total_predictions_30d': total_predictions_30d,
                    'revenue_at_risk_7d': round(revenue_at_risk_7d, 2),
                    'estimated_revenue_saved_7d': round(revenue_saved_7d, 2),
                    'total_revenue_saved_lifetime': round(self.total_revenue_saved, 2),
                    'average_processing_time_ms': round(avg_processing_time, 2),
                    'model_accuracy_latest': round(latest_accuracy, 1) if latest_accuracy else 'N/A',
                    'prediction_volume_trend_pct': round(prediction_trend, 1)
                },
                'risk_distribution_7d': {
                    'high_risk': {'count': high_risk_7d, 'percentage': round(high_risk_7d / total_predictions_7d * 100, 1) if total_predictions_7d > 0 else 0},
                    'medium_risk': {'count': medium_risk_7d, 'percentage': round(medium_risk_7d / total_predictions_7d * 100, 1) if total_predictions_7d > 0 else 0},
                    'low_risk': {'count': low_risk_7d, 'percentage': round(low_risk_7d / total_predictions_7d * 100, 1) if total_predictions_7d > 0 else 0}
                },
                'category_performance': category_risk_rates,
                'system_health': {
                    'status': 'Healthy' if avg_processing_time < 200 else 'Monitor',
                    'predictions_today': self.daily_metrics.get(datetime.now().date(), {}).get('total_predictions', 0),
                    'uptime_status': 'Active',
                    'data_quality': 'Good'
                },
                'insights': self._generate_executive_insights(recent_predictions, category_risk_rates),
                'dashboard_updated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating executive dashboard data: {str(e)}")
            return {'error': str(e)}
    
    def _generate_executive_insights(self, recent_predictions: List[Dict], 
                                   category_stats: Dict[str, Any]) -> List[str]:
        """
        Generate executive-level insights and recommendations
        
        Args:
            recent_predictions: Recent prediction data
            category_stats: Category performance statistics
            
        Returns:
            List of insight strings
        """
        insights = []
        
        if not recent_predictions:
            insights.append("No recent predictions to analyze")
            return insights
        
        total_predictions = len(recent_predictions)
        high_risk_pct = len([p for p in recent_predictions if p['risk_level'] == 'HIGH']) / total_predictions * 100
        
        # Volume insights
        if total_predictions > 100:
            insights.append(f"Strong adoption: {total_predictions} predictions in the last 7 days")
        elif total_predictions < 20:
            insights.append("Low prediction volume - consider increasing system promotion")
        
        # Risk insights
        if high_risk_pct > 25:
            insights.append(f"High risk alert: {high_risk_pct:.1f}% of orders flagged as high-risk")
        elif high_risk_pct < 5:
            insights.append("Very low risk orders detected - excellent order quality")
        
        # Category insights
        if category_stats:
            highest_risk_category = max(
                category_stats.items(),
                key=lambda x: x[1]['high_risk_rate']
            )
            insights.append(f"Highest risk category: {highest_risk_category[0]} ({highest_risk_category[1]['high_risk_rate']:.1f}% high-risk)")
            
            # Revenue impact
            highest_revenue_risk = max(
                category_stats.items(),
                key=lambda x: x[1]['revenue_at_risk']
            )
            insights.append(f"Highest revenue at risk: {highest_revenue_risk[0]} (${highest_revenue_risk[1]['revenue_at_risk']:.2f})")
        
        # Performance insights
        avg_processing = np.mean([p.get('processing_time_ms', 0) for p in recent_predictions])
        if avg_processing < 100:
            insights.append("Excellent system performance - sub-100ms average response time")
        elif avg_processing > 200:
            insights.append("System performance needs attention - response times above target")
        
        return insights
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics and status
        
        Returns:
            Dictionary with agent statistics
        """
        return {
            'agent_name': 'BusinessIntelligenceAgent',
            'total_predictions_processed': self.processed_predictions,
            'prediction_history_size': len(self.prediction_history),
            'days_with_data': len(self.daily_metrics),
            'accuracy_measurements': len(self.accuracy_tracking),
            'total_revenue_saved': round(self.total_revenue_saved, 2),
            'status': 'active',
            'last_updated': datetime.now().isoformat()
        }


# Global instance for easy access
_business_intelligence_agent = None

def get_business_intelligence_agent() -> BusinessIntelligenceAgent:
    """
    Get or create the global BusinessIntelligenceAgent instance
    
    Returns:
        BusinessIntelligenceAgent instance
    """
    global _business_intelligence_agent
    if _business_intelligence_agent is None:
        _business_intelligence_agent = BusinessIntelligenceAgent()
    return _business_intelligence_agent


# Example usage and testing
if __name__ == "__main__":
    # Initialize agent
    agent = BusinessIntelligenceAgent()
    
    # Test recording predictions
    sample_prediction = {
        'return_probability': 0.75,
        'confidence': 0.85
    }
    
    sample_order = {
        'order_id': 'TEST-001',
        'price': 199.99,
        'quantity': 1,
        'product_category': 'Electronics',
        'age': 28,
        'payment_method': 'Credit Card'
    }
    
    # Record a few test predictions
    for i in range(5):
        agent.record_prediction(sample_prediction, sample_order, processing_time_ms=150)
    
    # Generate daily report
    daily_report = agent.generate_daily_report()
    print("Daily Report:")
    print(json.dumps(daily_report, indent=2))
    
    # Create executive dashboard
    dashboard_data = agent.create_executive_dashboard_data()
    print("\nExecutive Dashboard:")
    print(json.dumps(dashboard_data, indent=2))
    
    # Get agent stats
    stats = agent.get_agent_stats()
    print(f"\nAgent Stats: {json.dumps(stats, indent=2)}")
