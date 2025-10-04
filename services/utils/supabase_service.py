"""
Supabase Database Service
Purpose: Handle database operations for storing and retrieving predictions with user authentication
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from supabase import create_client, Client
from dotenv import load_dotenv
import uuid
import asyncio
from functools import wraps

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

class SupabaseService:
    """Service for interacting with Supabase database with user authentication support"""
    
    def __init__(self):
        """Initialize Supabase client"""
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_KEY')  # Use service key for backend
        
        if not supabase_url or not supabase_key:
            logger.warning("Supabase credentials not found. Database features will be disabled.")
            self.client: Optional[Client] = None
            self.enabled = False
        else:
            try:
                self.client: Client = create_client(supabase_url, supabase_key)
                self.enabled = True
                logger.info("Supabase client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {str(e)}")
                self.client = None
                self.enabled = False
    
    def is_enabled(self) -> bool:
        """Check if Supabase integration is enabled"""
        return self.enabled and self.client is not None
    
    def authenticate_user(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate user using JWT token
        
        Args:
            token: JWT token from frontend
            
        Returns:
            User data if valid, None otherwise
        """
        if not self.is_enabled():
            return None
            
        try:
            # Set the auth token for the request
            self.client.auth._headers['Authorization'] = f'Bearer {token}'
            user_response = self.client.auth.get_user(token)
            
            if user_response.user:
                return {
                    'id': user_response.user.id,
                    'email': user_response.user.email,
                    'user_metadata': user_response.user.user_metadata,
                    'app_metadata': user_response.user.app_metadata
                }
            return None
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return None
    
    async def create_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create or update user profile
        
        Args:
            user_id: User ID from auth
            profile_data: Profile information
            
        Returns:
            Created profile data or None if failed
        """
        if not self.is_enabled():
            return None
            
        try:
            profile = {
                'id': user_id,
                'username': profile_data.get('username'),
                'full_name': profile_data.get('full_name'),
                'company_name': profile_data.get('company_name'),
                'industry': profile_data.get('industry'),
                'website': profile_data.get('website'),
                'avatar_url': profile_data.get('avatar_url'),
                'updated_at': datetime.now().isoformat()
            }
            
            # Remove None values
            profile = {k: v for k, v in profile.items() if v is not None}
            
            result = self.client.table('profiles').upsert(profile).execute()
            
            if result.data:
                logger.info(f"Profile created/updated for user: {user_id}")
                return result.data[0]
            return None
            
        except Exception as e:
            logger.error(f"Error creating user profile: {str(e)}")
            return None
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user profile by ID
        
        Args:
            user_id: User ID
            
        Returns:
            Profile data or None if not found
        """
        if not self.is_enabled():
            return None
            
        try:
            result = self.client.table('profiles').select('*').eq('id', user_id).single().execute()
            
            if result.data:
                return result.data
            return None
            
        except Exception as e:
            logger.error(f"Error getting user profile: {str(e)}")
            return None
    
    async def create_batch_job(self, user_id: str, job_data: Dict[str, Any]) -> Optional[str]:
        """
        Create a new batch prediction job
        
        Args:
            user_id: User ID
            job_data: Job information
            
        Returns:
            Job ID if created successfully, None otherwise
        """
        if not self.is_enabled():
            return None
            
        try:
            job_id = str(uuid.uuid4())
            
            batch_job = {
                'id': job_id,
                'user_id': user_id,
                'job_name': job_data.get('job_name', 'Batch Prediction'),
                'total_records': job_data['total_records'],
                'original_filename': job_data.get('filename'),
                'file_type': job_data.get('file_type'),
                'file_size_bytes': job_data.get('file_size_bytes'),
                'estimated_processing_time_seconds': job_data.get('estimated_processing_time_seconds'),
                'status': 'pending'
            }
            
            result = self.client.table('batch_jobs').insert(batch_job).execute()
            
            if result.data:
                logger.info(f"Batch job created: {job_id} for user: {user_id}")
                return job_id
            return None
            
        except Exception as e:
            logger.error(f"Error creating batch job: {str(e)}")
            return None
    
    async def update_batch_job(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update batch job status and progress
        
        Args:
            job_id: Job ID
            updates: Updates to apply
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_enabled():
            return False
            
        try:
            updates['updated_at'] = datetime.now().isoformat()
            
            result = self.client.table('batch_jobs').update(updates).eq('id', job_id).execute()
            
            if result.data:
                logger.debug(f"Batch job updated: {job_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error updating batch job: {str(e)}")
            return False
    
    async def get_batch_job(self, job_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get batch job by ID
        
        Args:
            job_id: Job ID
            user_id: Optional user ID for authorization
            
        Returns:
            Job data or None if not found
        """
        if not self.is_enabled():
            return None
            
        try:
            query = self.client.table('batch_jobs').select('*').eq('id', job_id)
            
            if user_id:
                query = query.eq('user_id', user_id)
            
            result = query.single().execute()
            
            if result.data:
                return result.data
            return None
            
        except Exception as e:
            logger.error(f"Error getting batch job: {str(e)}")
            return None
    
    async def get_user_batch_jobs(
        self, 
        user_id: str, 
        limit: int = 20,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get user's batch jobs
        
        Args:
            user_id: User ID
            limit: Maximum number of jobs to return
            status: Optional status filter
            
        Returns:
            List of batch jobs
        """
        if not self.is_enabled():
            return []
            
        try:
            query = self.client.table('batch_jobs').select('*').eq('user_id', user_id)
            
            if status:
                query = query.eq('status', status)
            
            query = query.order('created_at', desc=True).limit(limit)
            
            result = query.execute()
            
            if result.data:
                return result.data
            return []
            
        except Exception as e:
            logger.error(f"Error getting user batch jobs: {str(e)}")
            return []
    
    async def store_prediction(self, prediction_data: Dict[str, Any], user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Store a prediction in the database
        
        Args:
            prediction_data: Prediction data to store (merged input + output)
            user_id: Optional user ID (if authenticated)
            
        Returns:
            Stored prediction data or None if failed
        """
        if not self.is_enabled():
            logger.debug("Supabase not enabled, skipping database storage")
            return None
        
        try:
            # Prepare data for insertion - handle both input data and prediction results
            db_data = {
                'user_id': user_id,  # Can be None for anonymous predictions
                'order_id': prediction_data.get('order_id', str(uuid.uuid4())),
                
                # Input features (from request)
                'category_name': prediction_data.get('product_category', prediction_data.get('category')),
                'price': float(prediction_data.get('price', 0)),
                'quantity': int(prediction_data.get('quantity', 1)),
                'customer_age': int(prediction_data.get('age', 0)),
                'customer_gender': prediction_data.get('gender'),
                'customer_location': prediction_data.get('location'),
                'payment_method_name': prediction_data.get('payment_method'),
                'shipping_method_name': prediction_data.get('shipping_method', 'Standard'),
                'discount_applied': float(prediction_data.get('discount_applied', 0)),
                
                # Calculated values
                'total_order_value': float(prediction_data.get('total_order_value', 
                    prediction_data.get('price', 0) * prediction_data.get('quantity', 1))),
                
                # Prediction results (from prediction response)
                'predicted_return_probability': float(prediction_data.get('return_probability', 0)),
                'predicted_return_flag': prediction_data.get('will_return', False),
                'confidence_score': float(prediction_data.get('confidence_score', 
                    prediction_data.get('confidence', 0))),
                'risk_level': prediction_data.get('risk_level', 'UNKNOWN'),
                'recommendations': prediction_data.get('recommendations', []),
                'model_version': prediction_data.get('model_version', 'v1.0'),
                'processing_time_ms': prediction_data.get('processing_time_ms'),
                'status': 'completed'
            }
            
            # Insert into database
            result = self.client.table('predictions').insert(db_data).execute()
            
            if result.data:
                logger.info(f"Prediction stored successfully for order: {db_data['order_id']}")
                return result.data[0]
            else:
                logger.warning("No data returned from insert operation")
                return None
                
        except Exception as e:
            logger.error(f"Error storing prediction: {str(e)}")
            logger.error(f"Prediction data keys: {list(prediction_data.keys())}")
            return None
    
    async def get_predictions(
        self, 
        user_id: Optional[str] = None, 
        limit: int = 100,
        risk_level: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_anonymous: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve predictions from database
        
        Args:
            user_id: Optional user ID to filter by
            limit: Maximum number of records to return
            risk_level: Optional risk level filter (LOW, MEDIUM, HIGH)
            start_date: Optional start date filter (ISO format)
            end_date: Optional end date filter (ISO format)
            include_anonymous: Whether to include anonymous predictions
            
        Returns:
            List of predictions
        """
        if not self.is_enabled():
            logger.debug("Supabase not enabled, returning empty list")
            return []
        
        try:
            # Build query
            query = self.client.table('predictions').select('*')
            
            # Apply user filter
            if user_id:
                if include_anonymous:
                    # Include both user's predictions and anonymous ones
                    query = query.or_(f'user_id.eq.{user_id},user_id.is.null')
                else:
                    query = query.eq('user_id', user_id)
            else:
                # When user_id is None:
                # - If include_anonymous is True: get ALL predictions (both user and anonymous)
                # - If include_anonymous is False: get only anonymous predictions
                if not include_anonymous:
                    query = query.is_('user_id', 'null')
                # else: no filter applied - get all predictions
            
            # Apply other filters
            if risk_level:
                query = query.eq('risk_level', risk_level.upper())
            
            if start_date:
                query = query.gte('created_at', start_date)
            
            if end_date:
                query = query.lte('created_at', end_date)
            
            # Order by most recent and limit
            query = query.order('created_at', desc=True).limit(limit)
            
            # Execute query
            result = query.execute()
            
            if result.data:
                logger.info(f"Retrieved {len(result.data)} predictions from database")
                return result.data
            else:
                logger.info("No predictions found")
                return []
                
        except Exception as e:
            logger.error(f"Error retrieving predictions: {str(e)}")
            return []
    
    async def get_predictions_summary(
        self, 
        user_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get summary statistics for predictions
        
        Args:
            user_id: Optional user ID to filter by
            days: Number of days to look back
            
        Returns:
            Summary statistics
        """
        if not self.is_enabled():
            return self._get_empty_summary()
        
        try:
            # Calculate start date
            start_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Get predictions
            predictions = await self.get_predictions(
                user_id=user_id,
                start_date=start_date,
                limit=10000
            )
            
            if not predictions:
                return self._get_empty_summary()
            
            # Calculate summary
            total = len(predictions)
            high_risk = sum(1 for p in predictions if p.get('risk_level') == 'HIGH')
            medium_risk = sum(1 for p in predictions if p.get('risk_level') == 'MEDIUM')
            low_risk = sum(1 for p in predictions if p.get('risk_level') == 'LOW')
            
            avg_prob = sum(float(p.get('predicted_return_probability', 0)) for p in predictions) / total if total > 0 else 0
            
            total_revenue_at_risk = sum(
                float(p.get('total_order_value', 0)) * float(p.get('predicted_return_probability', 0))
                for p in predictions
            )
            
            return {
                'total_predictions': total,
                'high_risk_count': high_risk,
                'medium_risk_count': medium_risk,
                'low_risk_count': low_risk,
                'average_probability': round(avg_prob, 4),
                'total_revenue_at_risk': round(total_revenue_at_risk, 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting predictions summary: {str(e)}")
            return self._get_empty_summary()
    
    def _get_empty_summary(self) -> Dict[str, Any]:
        """Return empty summary statistics"""
        return {
            'total_predictions': 0,
            'high_risk_count': 0,
            'medium_risk_count': 0,
            'low_risk_count': 0,
            'average_probability': 0,
            'total_revenue_at_risk': 0
        }
    
    async def log_api_usage(
        self, 
        user_id: Optional[str],
        endpoint: str,
        processing_time_ms: Optional[int] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        request_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log API usage for analytics and quota tracking
        
        Args:
            user_id: User ID (None for anonymous)
            endpoint: API endpoint called
            processing_time_ms: Processing time in milliseconds
            success: Whether the request was successful
            error_message: Error message if failed
            request_metadata: Additional request metadata
            
        Returns:
            True if logged successfully, False otherwise
        """
        if not self.is_enabled():
            return False
            
        try:
            usage_data = {
                'user_id': user_id,
                'endpoint': endpoint,
                'processing_time_ms': processing_time_ms,
                'success': success,
                'error_message': error_message
            }
            
            # Add request metadata if provided
            if request_metadata:
                usage_data.update({
                    'ip_address': request_metadata.get('ip_address'),
                    'user_agent': request_metadata.get('user_agent'),
                    'request_size_bytes': request_metadata.get('request_size_bytes'),
                    'response_size_bytes': request_metadata.get('response_size_bytes'),
                    'cost_cents': request_metadata.get('cost_cents', 0)
                })
            
            result = self.client.table('api_usage').insert(usage_data).execute()
            
            return result.data is not None
            
        except Exception as e:
            logger.error(f"Error logging API usage: {str(e)}")
            return False
    
    async def get_user_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user preferences
        
        Args:
            user_id: User ID
            
        Returns:
            User preferences or None if not found
        """
        if not self.is_enabled():
            return None
            
        try:
            result = self.client.table('user_preferences').select('*').eq('user_id', user_id).single().execute()
            
            if result.data:
                return result.data
            return None
            
        except Exception as e:
            logger.error(f"Error getting user preferences: {str(e)}")
            return None
    
    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """
        Update user preferences
        
        Args:
            user_id: User ID
            preferences: Preferences to update
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_enabled():
            return False
            
        try:
            preferences['user_id'] = user_id
            preferences['updated_at'] = datetime.now().isoformat()
            
            result = self.client.table('user_preferences').upsert(preferences).execute()
            
            return result.data is not None
            
        except Exception as e:
            logger.error(f"Error updating user preferences: {str(e)}")
            return False
    
    async def get_user_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Get user analytics and usage statistics
        
        Args:
            user_id: User ID
            days: Number of days to analyze
            
        Returns:
            Analytics data
        """
        if not self.is_enabled():
            return {}
            
        try:
            start_date = (datetime.now() - timedelta(days=days)).date()
            
            # Get API usage analytics
            usage_result = self.client.table('api_usage')\
                .select('*')\
                .eq('user_id', user_id)\
                .gte('date_bucket', start_date.isoformat())\
                .execute()
            
            # Get prediction summary
            prediction_summary = await self.get_predictions_summary(user_id, days)
            
            # Calculate usage metrics
            usage_data = usage_result.data if usage_result.data else []
            
            total_requests = sum(u.get('request_count', 1) for u in usage_data)
            successful_requests = sum(u.get('request_count', 1) for u in usage_data if u.get('success', True))
            avg_processing_time = sum(u.get('processing_time_ms', 0) for u in usage_data) / len(usage_data) if usage_data else 0
            
            # Group by endpoint
            endpoint_usage = {}
            for usage in usage_data:
                endpoint = usage.get('endpoint', 'unknown')
                if endpoint not in endpoint_usage:
                    endpoint_usage[endpoint] = 0
                endpoint_usage[endpoint] += usage.get('request_count', 1)
            
            return {
                'prediction_summary': prediction_summary,
                'api_usage': {
                    'total_requests': total_requests,
                    'successful_requests': successful_requests,
                    'success_rate': successful_requests / total_requests if total_requests > 0 else 0,
                    'avg_processing_time_ms': round(avg_processing_time, 2),
                    'endpoint_usage': endpoint_usage
                },
                'period_days': days,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting user analytics: {str(e)}")
            return {}


# Global instance
_supabase_service: Optional[SupabaseService] = None

def get_supabase_service() -> SupabaseService:
    """Get or create the global Supabase service instance"""
    global _supabase_service
    if _supabase_service is None:
        _supabase_service = SupabaseService()
    return _supabase_service
