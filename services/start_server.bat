@echo off
echo Starting E-commerce Return Prediction API...
echo.
echo Choose mode:
echo [1] Development (with auto-reload)
echo [2] Production (no auto-reload)
echo [3] Direct uvicorn command
echo.
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo Starting in Development mode...
    python main.py --dev
) else if "%choice%"=="2" (
    echo Starting in Production mode...
    python main.py
) else if "%choice%"=="3" (
    echo Starting with uvicorn...
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
) else (
    echo Invalid choice. Starting in Production mode...
    python main.py
)

echo.
echo Server will be available at: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo Health Check: http://localhost:8000/health
pause