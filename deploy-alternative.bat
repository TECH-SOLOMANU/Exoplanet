@echo off
echo ========================================
echo NASA Exoplanet Detection Platform
echo Alternative Deployment Script
echo ========================================

echo.
echo Checking Docker installation...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not installed
    echo Please install Docker Desktop and try again
    goto :manual_setup
)
echo ✅ Docker is installed

echo.
echo Checking if Docker is running...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Docker is not running
    echo.
    echo Attempting to start Docker Desktop...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    echo Waiting for Docker to start (this may take 1-2 minutes)...
    
    REM Wait up to 2 minutes for Docker to start
    set /a count=0
    :wait_loop
    timeout /t 10 /nobreak >nul
    docker info >nul 2>&1
    if %errorlevel% equ 0 goto :docker_ready
    set /a count+=1
    if %count% lss 12 goto :wait_loop
    
    echo ❌ Docker failed to start automatically
    echo.
    echo Please manually start Docker Desktop and press any key to continue...
    pause
    
    docker info >nul 2>&1
    if %errorlevel% neq 0 goto :manual_setup
)

:docker_ready
echo ✅ Docker is running

echo.
echo Building and starting services...
docker-compose up -d --build

echo.
echo Waiting for services to start...
timeout /t 30 /nobreak >nul

echo.
echo Checking service status...
docker-compose ps

echo.
goto :deployment_complete

:manual_setup
echo.
echo ========================================
echo 🛠️  Manual Setup (Without Docker)
echo ========================================
echo.
echo Since Docker is not available, let's set up manually:
echo.
echo 1. Backend Setup:
echo    cd backend
echo    python -m venv venv
echo    .\venv\Scripts\activate
echo    pip install -r requirements.txt
echo.
echo 2. Frontend Setup:
echo    cd frontend
echo    npm install
echo.
echo 3. Start Services:
echo    Backend: uvicorn app.main:app --reload --port 800
echo    Frontend: npm start
echo.
echo Would you like to run the manual setup now? (y/n)
set /p choice="Enter your choice: "
if /i "%choice%"=="y" goto :run_manual
if /i "%choice%"=="yes" goto :run_manual
goto :end

:run_manual
echo.
echo Starting manual setup...
echo.

echo Setting up backend...
cd backend
if not exist venv (
    echo Creating Python virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call .\venv\Scripts\activate.bat

echo Installing Python dependencies...
pip install -r requirements.txt

echo Starting backend server on port 800...
start "Exoplanet Backend" cmd /k "uvicorn app.main:app --reload --port 800"

cd ..

echo.
echo Setting up frontend...
cd frontend

echo Installing Node.js dependencies...
npm install

echo Starting frontend development server...
start "Exoplanet Frontend" cmd /k "npm start"

cd ..

echo.
echo ========================================
echo 🚀 Manual Deployment Complete!
echo ========================================
echo.
echo Your application is starting:
echo 🌐 Frontend:  http://localhost:3000
echo 🔧 Backend:   http://localhost:800
echo 📚 API Docs:  http://localhost:800/docs
echo.
echo Note: MongoDB is not running. The app will use fallback data.
echo.
goto :end

:deployment_complete
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
echo To view logs: docker-compose logs -f
echo To stop: docker-compose down
echo To restart: docker-compose restart
echo.

:end
echo ========================================
echo 🏆 NASA Space Apps Challenge 2025
echo Exoplanet Detection Platform Ready!
echo ========================================
pause