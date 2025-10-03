@echo off
echo ========================================
echo NASA Exoplanet Detection Platform
echo Quick Deployment Script
echo ========================================

echo.
echo Checking Docker installation...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not installed or not running
    echo Please install Docker Desktop and try again
    pause
    exit /b 1
)
echo ✅ Docker is available

echo.
echo Checking Docker Compose...
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker Compose is not available
    pause
    exit /b 1
)
echo ✅ Docker Compose is available

echo.
echo Building and starting services...
echo This may take 5-10 minutes on first run...
docker-compose up -d --build

echo.
echo Waiting for services to start...
timeout /t 30 /nobreak >nul

echo.
echo ========================================
echo 🚀 Deployment Complete!
echo ========================================
echo.
echo Access your application:
echo 🌐 Frontend:  http://localhost:3000
echo 🔧 Backend:   http://localhost:8000
echo 📚 API Docs:  http://localhost:8000/docs
echo 🗄️ MongoDB:   localhost:27017
echo.
echo Checking service status...
docker-compose ps

echo.
echo To view logs: docker-compose logs -f
echo To stop: docker-compose down
echo To restart: docker-compose restart
echo.
echo ========================================
echo 🏆 NASA Space Apps Challenge 2025
echo Exoplanet Detection Platform Ready!
echo ========================================

pause