FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY services/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire services directory
COPY services/ .

# Make scripts executable
RUN chmod +x startup.py

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Expose port (Railway will override PORT environment variable)
EXPOSE $PORT

# Use multiple fallback options
CMD ["sh", "-c", "python startup.py || python -c \"import os; port=int(os.environ.get('PORT', 8000)); import subprocess; subprocess.run(['uvicorn', 'main:app', '--host', '0.0.0.0', '--port', str(port), '--log-level', 'info'])\""]
