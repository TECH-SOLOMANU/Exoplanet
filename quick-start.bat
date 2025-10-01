@echo off
REM Quick Start Script - No Docker Required
REM NASA Space Apps Challenge 2025 - Exoplanet Detection Platform

echo.
echo ========================================================================
echo 🌟 NASA Space Apps Challenge 2025 - Quick Start (No Docker)
echo ========================================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8+ first.
    echo    Download from: https://python.org/downloads
    pause
    exit /b 1
)

REM Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js is not installed. Please install Node.js 16+ first.
    echo    Download from: https://nodejs.org
    pause
    exit /b 1
)

echo ✅ Python and Node.js detected
echo.

REM Start Backend in Demo Mode
echo 🚀 Starting Backend (Demo Mode)...
start "NASA Exoplanet API" cmd /k "python demo_app.py"

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Start Frontend
echo 🎨 Starting Frontend...
cd frontend
start "NASA Exoplanet Dashboard" cmd /k "npm install && npm start"

echo.
echo 🎉 Application Starting!
echo ========================================
echo 📊 Dashboard: http://localhost:3000
echo 🔗 API Demo: http://localhost:8000/docs
echo.
echo ⚠️  DEMO MODE: Using mock data for immediate testing
echo 📚 For full features with real NASA data, see DEVELOPMENT_SETUP.md
echo.
echo 🛑 To stop: Close both command windows or press Ctrl+C in each
echo.
pause