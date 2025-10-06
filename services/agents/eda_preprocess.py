"""
EDA Preprocessing Agent
Purpose: Advanced data validation and preprocessing for batch analysis
Functions:
- Advanced data quality analysis
- Batch data processing and cleaning
- Statistical validation and outlier detection
- Data quality reporting
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EDAPreprocessAgent:
    """
    EDA Preprocessing Agent for advanced data analysis and batch processing
    
    This agent focuses on:
    - Advanced data quality analysis
    - Batch data processing
    - Statistical validation
    - Data profiling and reporting
    """
    
    def __init__(self):
        """Initialize the EDA Preprocessing Agent"""
        self.processed_count = 0
        self.data_quality_thresholds = self._load_quality_thresholds()
        logger.info("EDA Preprocessing Agent initialized")
    
    def _load_quality_thresholds(self) -> Dict[str, Any]:
        """Load data quality thresholds"""
        return {
            'missing_value_threshold': 0.1,  # 10% missing values max
            'outlier_threshold': 0.05,       # 5% outliers max
            'duplicate_threshold': 0.01,     # 1% duplicates max
            'categorical_cardinality_max': 100,  # Max unique categories
            'numeric_range_validation': {
                'Product_Price': {'min': 0.01, 'max': 10000.0},
                'Order_Quantity': {'min': 1, 'max': 100},
                'User_Age': {'min': 18, 'max': 100},
                'Discount_Applied': {'min': 0.0, 'max': 100.0}
            }
        }
    
    def analyze_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data quality analysis for batch data
        
        Args:
            data: Input DataFrame to analyze
            
        Returns:
            Dictionary containing data quality metrics
        """
        try:
            if data.empty:
                return {'error': 'Input data is empty', 'quality_score': 0}
            
            quality_report = {
                'total_records': len(data),
                'total_columns': len(data.columns),
                'missing_value_analysis': {},
                'duplicate_analysis': {},
                'outlier_analysis': {},
                'data_type_analysis': {},
                'categorical_analysis': {},
                'numeric_range_analysis': {},
                'overall_quality_score': 0.0
            }
            
            # Missing value analysis
            missing_stats = {}
            for col in data.columns:
                missing_count = data[col].isnull().sum()
                missing_pct = (missing_count / len(data)) * 100
                missing_stats[col] = {
                    'missing_count': int(missing_count),
                    'missing_percentage': round(missing_pct, 2),
                    'quality_flag': 'GOOD' if missing_pct <= self.data_quality_thresholds['missing_value_threshold'] * 100 else 'WARNING'
                }
            quality_report['missing_value_analysis'] = missing_stats
            
            # Duplicate analysis
            duplicate_count = data.duplicated().sum()
            duplicate_pct = (duplicate_count / len(data)) * 100
            quality_report['duplicate_analysis'] = {
                'duplicate_count': int(duplicate_count),
                'duplicate_percentage': round(duplicate_pct, 2),
                'quality_flag': 'GOOD' if duplicate_pct <= self.data_quality_thresholds['duplicate_threshold'] * 100 else 'WARNING'
            }
            
            # Outlier analysis for numeric columns
            outlier_stats = {}
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if len(data[col].dropna()) > 0:
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
                    outlier_pct = (len(outliers) / len(data[col].dropna())) * 100
                    
                    outlier_stats[col] = {
                        'outlier_count': len(outliers),
                        'outlier_percentage': round(outlier_pct, 2),
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'quality_flag': 'GOOD' if outlier_pct <= self.data_quality_thresholds['outlier_threshold'] * 100 else 'WARNING'
                    }
            quality_report['outlier_analysis'] = outlier_stats
            
            # Categorical analysis
            categorical_stats = {}
            categorical_columns = data.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                unique_count = data[col].nunique()
                categorical_stats[col] = {
                    'unique_values': int(unique_count),
                    'most_frequent': str(data[col].mode().iloc[0]) if len(data[col].mode()) > 0 else 'N/A',
                    'quality_flag': 'GOOD' if unique_count <= self.data_quality_thresholds['categorical_cardinality_max'] else 'WARNING'
                }
            quality_report['categorical_analysis'] = categorical_stats
            
            # Calculate overall quality score
            quality_score = self._calculate_overall_quality_score(quality_report)
            quality_report['overall_quality_score'] = quality_score
            
            return quality_report
            
        except Exception as e:
            logger.error(f"Error in data quality analysis: {str(e)}")
            return {'error': str(e), 'quality_score': 0}
    
    def _calculate_overall_quality_score(self, quality_report: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-100)"""
        try:
            score = 100.0
            
            # Deduct for missing values
            missing_analysis = quality_report.get('missing_value_analysis', {})
            avg_missing_pct = np.mean([stats['missing_percentage'] for stats in missing_analysis.values()]) if missing_analysis else 0
            score -= avg_missing_pct * 2
            
            # Deduct for duplicates
            duplicate_pct = quality_report.get('duplicate_analysis', {}).get('duplicate_percentage', 0)
            score -= duplicate_pct * 5
            
            # Deduct for outliers
            outlier_analysis = quality_report.get('outlier_analysis', {})
            avg_outlier_pct = np.mean([stats['outlier_percentage'] for stats in outlier_analysis.values()]) if outlier_analysis else 0
            score -= avg_outlier_pct * 3
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {str(e)}")
            return 50.0

    def process_batch_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Process batch data with advanced analysis
        
        Args:
            data: Raw batch DataFrame
            
        Returns:
            Dictionary containing processed data and analysis
        """
        try:
            # Step 1: Analyze data quality
            quality_analysis = self.analyze_data_quality(data)
            
            # Step 2: Basic outlier detection and flagging
            outlier_report = self.detect_outliers(data)
            
            # Step 3: Generate data profiling report
            profile_report = self._generate_data_profile(data)
            
            # Update processed count
            self.processed_count += len(data)
            
            return {
                'success': True,
                'original_data': data,
                'quality_analysis': quality_analysis,
                'outlier_report': outlier_report,
                'data_profile': profile_report,
                'recommendations': self._generate_recommendations(quality_analysis),
                'processing_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_data_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data profile"""
        try:
            profile = {
                'basic_stats': {
                    'total_records': len(data),
                    'total_columns': len(data.columns),
                    'memory_usage_mb': round(data.memory_usage(deep=True).sum() / 1024 / 1024, 2)
                },
                'column_types': {
                    'numeric': list(data.select_dtypes(include=[np.number]).columns),
                    'categorical': list(data.select_dtypes(include=['object']).columns),
                    'datetime': list(data.select_dtypes(include=['datetime']).columns)
                },
                'summary_statistics': {}
            }
            
            # Generate summary statistics for numeric columns
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                profile['summary_statistics'][col] = {
                    'mean': round(data[col].mean(), 2),
                    'median': round(data[col].median(), 2),
                    'std': round(data[col].std(), 2),
                    'min': round(data[col].min(), 2),
                    'max': round(data[col].max(), 2),
                    '25th_percentile': round(data[col].quantile(0.25), 2),
                    '75th_percentile': round(data[col].quantile(0.75), 2)
                }
            
            return profile
            
        except Exception as e:
            logger.error(f"Error generating data profile: {str(e)}")
            return {}

    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            'agent_name': 'EDAPreprocessAgent',
            'total_processed': self.processed_count,
            'quality_thresholds': self.data_quality_thresholds,
            'status': 'active',
            'last_updated': datetime.now().isoformat()
        }
    
    def detect_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Advanced outlier detection for batch analysis"""
        try:
            outlier_report = {}
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                col_data = data[col].dropna()
                
                if len(col_data) > 0:
                    # IQR method
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    
                    # Z-score method (for comparison)
                    z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                    z_outliers = col_data[z_scores > 3]
                    
                    outlier_report[col] = {
                        'iqr_outlier_count': len(outliers),
                        'iqr_outlier_percentage': (len(outliers) / len(col_data)) * 100,
                        'iqr_bounds': {'lower': lower_bound, 'upper': upper_bound},
                        'zscore_outlier_count': len(z_outliers),
                        'zscore_outlier_percentage': (len(z_outliers) / len(col_data)) * 100,
                        'sample_outlier_values': outliers.head(5).tolist()
                    }
            
            return outlier_report
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {str(e)}")
            return {}


# Global instance for easy access
_preprocess_agent = None

def get_preprocess_agent() -> EDAPreprocessAgent:
    """
    Get or create the global EDAPreprocessAgent instance
    
    Returns:
        EDAPreprocessAgent instance
    """
    global _preprocess_agent
    if _preprocess_agent is None:
        _preprocess_agent = EDAPreprocessAgent()
    return _preprocess_agent


# Example usage and testing
if __name__ == "__main__":
    # Initialize agent
    agent = EDAPreprocessAgent()
    
    # Test with sample data
    sample_data = pd.DataFrame([
        {
            'price': 199.99,
            'quantity': 1,
            'product_category': 'Electronics',
            'age': 28,
            'gender': 'Female',
            'payment_method': 'Credit Card',
            'location': 'Urban',
            'discount_applied': 10.0,
            'shipping_method': 'Standard'
        },
        {
            'price': 49.99,
            'quantity': 2,
            'product_category': 'Clothing',
            'age': 35,
            'gender': 'Male',
            'payment_method': 'PayPal',
            'location': 'Suburban',
            'discount_applied': 0.0,
            'shipping_method': 'Express'
        }
    ])
    
    try:
        # Run preprocessing
        result = agent.preprocess(sample_data)
        
        if result['success']:
            print("Preprocessing successful!")
            print(f"Processed data shape: {result['processed_data'].shape}")
            print(f"Data quality score: {result['preprocessing_report']['data_quality_score']}")
            print(f"Outliers detected: {result['preprocessing_report']['outliers_detected']}")
        else:
            print(f"Preprocessing failed: {result['error']}")
        
        # Get agent stats
        stats = agent.get_preprocessing_stats()
        print(f"\nAgent stats: {json.dumps(stats, indent=2)}")
        
    except Exception as e:
        print(f"Error: {e}")