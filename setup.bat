@echo off
REM NASA Space Apps Challenge 2025 - Exoplanet Detection Platform
REM Windows Setup Script

echo 🌟 NASA Space Apps Challenge 2025 - Exoplanet Detection Platform Setup
echo ==================================================================

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Compose is not installed. Please install Docker Compose first.
    pause
    exit /b 1
)

REM Create .env file from example
if not exist .env (
    echo 📝 Creating .env file from example...
    copy .env.example .env
    echo ✅ .env file created. Please review and update if needed.
)

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist data mkdir data
if not exist models mkdir models
if not exist logs mkdir logs

REM Download sample data (placeholder)
echo 📊 Setting up sample data structure...
if not exist data\nasa_archive mkdir data\nasa_archive
if not exist data\light_curves mkdir data\light_curves
if not exist data\processed mkdir data\processed

REM Build and start services
echo 🚀 Building and starting services...
docker-compose up --build -d

REM Wait for services to be ready
echo ⏳ Waiting for services to start...
timeout /t 30 /nobreak >nul

REM Check service status
echo 🔍 Checking service status...
docker-compose ps

REM Display access information
echo.
echo 🎉 Setup Complete!
echo ===================
echo 📊 Dashboard: http://localhost:3000
echo 🔗 API Docs: http://localhost:8000/docs
echo 🗄️  MongoDB: localhost:27017
echo 📡 Redis: localhost:6379
echo.
echo 🚀 Next steps:
echo 1. Visit the dashboard at http://localhost:3000
echo 2. Click 'Fetch Latest NASA Data' to load exoplanet data
echo 3. Explore the predictions and upload your own data
echo.
echo 📚 For more information, see README.md
echo 🐛 To view logs: docker-compose logs -f
echo 🛑 To stop: docker-compose down
echo.
pause