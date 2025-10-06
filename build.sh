#!/bin/bash

# Build script for Vercel deployment
echo "Starting build process..."

# Set up environment
echo "Setting up environment..."
export NODE_ENV=production

# Navigate to frontend directory
cd frontend

# Install dependencies
echo "Installing frontend dependencies..."
npm ci

# Build the frontend
echo "Building frontend..."
npm run build

# Verify build output
if [ -d "dist" ]; then
    echo "Frontend build successful!"
    echo "Build output directory: $(pwd)/dist"
    ls -la dist/
else
    echo "Frontend build failed - dist directory not found"
    exit 1
fi

echo "Build process completed successfully!"
