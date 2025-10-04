"""
Vercel serverless function entry point for FastAPI application
"""

import os
import sys
from mangum import Mangum

# Add the services directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
services_dir = os.path.join(current_dir, '..', 'services')
sys.path.insert(0, services_dir)

# Import the FastAPI app
from main import app

# Create the handler for Vercel
handler = Mangum(app, lifespan="off")
