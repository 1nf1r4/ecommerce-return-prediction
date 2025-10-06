#!/usr/bin/env python3
"""
Python-based startup script for Railway deployment
More reliable than shell script for environment variable handling
"""

import os
import sys
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_environment():
    """Validate the environment and setup"""
    logger.info("Railway Deployment Startup")
    logger.info("=" * 50)
    
    # Check Python version
    logger.info(f"Python version: {sys.version}")
    
    # Check current directory
    logger.info(f"Current directory: {os.getcwd()}")
    
    # Check required files
    required_files = ["main.py", "requirements.txt"]
    for file in required_files:
        if os.path.exists(file):
            logger.info(f"{file} exists")
        else:
            logger.error(f"{file} missing")
            return False
    
    # Check directories
    required_dirs = ["api", "agents"]
    for dir in required_dirs:
        if os.path.isdir(dir):
            logger.info(f"{dir}/ directory exists")
        else:
            logger.warning(f"{dir}/ directory missing")
    
    return True

def test_imports():
    """Test critical imports"""
    logger.info("Testing critical imports...")
    
    try:
        import fastapi
        logger.info("FastAPI")
        
        import uvicorn
        logger.info("Uvicorn")
        
        import pandas
        logger.info("Pandas")
        
        import numpy
        logger.info("Numpy")
        
        from main import app, health_check
        logger.info("Main app import")
        
        result = health_check()
        logger.info(f"Health check: {result.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Import error: {str(e)}")
        return False

def get_port():
    """Get the port number from environment or use default"""
    port_str = os.environ.get('PORT', '8000')
    
    logger.info(f"Environment PORT: {port_str}")
    
    try:
        port = int(port_str)
        if 1 <= port <= 65535:
            logger.info(f"Using port: {port}")
            return port
        else:
            logger.warning(f"Port {port} out of range, using 8000")
            return 8000
    except ValueError:
        logger.warning(f"Invalid port '{port_str}', using 8000")
        return 8000

def start_server():
    """Start the uvicorn server"""
    port = get_port()
    
    logger.info("=" * 50)
    logger.info(f"Starting application on port {port}")
    
    try:
        # Use subprocess to start uvicorn
        cmd = [
            "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", str(port), 
            "--log-level", "info"
        ]
        
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Start the server
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

def main():
    """Main startup function"""
    try:
        # Validate environment
        if not validate_environment():
            logger.error("Environment validation failed")
            sys.exit(1)
        
        # Test imports
        if not test_imports():
            logger.error("Import testing failed")
            sys.exit(1)
        
        # Start server
        start_server()
        
    except KeyboardInterrupt:
        logger.info("Startup interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
