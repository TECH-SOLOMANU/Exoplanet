# NASA Exoplanet Detection Platform - PowerShell Deployment Script
# NASA Space Apps Challenge 2025

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🚀 NASA Exoplanet Detection Platform" -ForegroundColor Green
Write-Host "PowerShell Deployment Script" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

# Function to test if a command exists
function Test-Command($command) {
    try {
        if (Get-Command $command -ErrorAction SilentlyContinue) {
            return $true
        }
    }
    catch {
        return $false
    }
    return $false
}

# Check Docker
Write-Host "`n📋 Checking Docker..." -ForegroundColor Blue
if (Test-Command "docker") {
    Write-Host "✅ Docker is installed" -ForegroundColor Green
    
    # Test if Docker is running
    try {
        docker info 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Docker is running" -ForegroundColor Green
            
            Write-Host "`n🏗️ Building and starting services..." -ForegroundColor Blue
            docker-compose up -d --build
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "`n⏳ Waiting for services to start..." -ForegroundColor Yellow
                Start-Sleep -Seconds 30
                
                Write-Host "`n📊 Checking service status..." -ForegroundColor Blue
                docker-compose ps
                
                Write-Host "`n========================================" -ForegroundColor Cyan
                Write-Host "🚀 Docker Deployment Complete!" -ForegroundColor Green
                Write-Host "========================================" -ForegroundColor Cyan
                Write-Host "`nAccess your application:" -ForegroundColor White
                Write-Host "🌐 Frontend:  http://localhost:3000" -ForegroundColor Yellow
                Write-Host "🔧 Backend:   http://localhost:8000" -ForegroundColor Yellow
                Write-Host "📚 API Docs:  http://localhost:8000/docs" -ForegroundColor Yellow
                Write-Host "🗄️ MongoDB:   localhost:27017" -ForegroundColor Yellow
                
                Write-Host "`nUseful commands:" -ForegroundColor Blue
                Write-Host "📋 View logs:     docker-compose logs -f" -ForegroundColor Gray
                Write-Host "🛑 Stop services: docker-compose down" -ForegroundColor Gray
                Write-Host "🔄 Restart:       docker-compose restart" -ForegroundColor Gray
            }
            else {
                Write-Host "❌ Docker deployment failed. Trying manual setup..." -ForegroundColor Red
                $useManual = $true
            }
        }
        else {
            Write-Host "⚠️ Docker is not running. Trying manual setup..." -ForegroundColor Yellow
            $useManual = $true
        }
    }
    catch {
        Write-Host "⚠️ Docker connection failed. Trying manual setup..." -ForegroundColor Yellow
        $useManual = $true
    }
}
else {
    Write-Host "❌ Docker not found. Using manual setup..." -ForegroundColor Red
    $useManual = $true
}

# Manual setup if Docker failed
if ($useManual) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "🛠️ Manual Setup" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    
    # Check Python
    Write-Host "`n📋 Checking Python..." -ForegroundColor Blue
    if (Test-Command "python") {
        Write-Host "✅ Python is available" -ForegroundColor Green
        
        # Setup backend
        Write-Host "`n🏗️ Setting up backend..." -ForegroundColor Blue
        Set-Location "backend"
        
        if (!(Test-Path "venv")) {
            Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
            python -m venv venv
        }
        
        Write-Host "Activating virtual environment..." -ForegroundColor Yellow
        & ".\venv\Scripts\Activate.ps1"
        
        Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
        pip install -r requirements.txt
        
        Write-Host "Starting backend server on port 800..." -ForegroundColor Yellow
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; .\venv\Scripts\Activate.ps1; uvicorn app.main:app --reload --port 800"
        
        Set-Location ".."
    }
    else {
        Write-Host "❌ Python not found. Please install Python 3.8+" -ForegroundColor Red
    }
    
    # Check Node.js
    Write-Host "`n📋 Checking Node.js..." -ForegroundColor Blue
    if (Test-Command "npm") {
        Write-Host "✅ Node.js is available" -ForegroundColor Green
        
        # Setup frontend
        Write-Host "`n🏗️ Setting up frontend..." -ForegroundColor Blue
        Set-Location "frontend"
        
        Write-Host "Installing Node.js dependencies..." -ForegroundColor Yellow
        npm install
        
        Write-Host "Starting frontend development server..." -ForegroundColor Yellow
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; npm start"
        
        Set-Location ".."
    }
    else {
        Write-Host "❌ Node.js not found. Please install Node.js 16+" -ForegroundColor Red
    }
    
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "🚀 Manual Deployment Started!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "`nYour application is starting:" -ForegroundColor White
    Write-Host "🌐 Frontend:  http://localhost:3000" -ForegroundColor Yellow
    Write-Host "🔧 Backend:   http://localhost:800" -ForegroundColor Yellow
    Write-Host "📚 API Docs:  http://localhost:800/docs" -ForegroundColor Yellow
    Write-Host "`nNote: MongoDB is not running. The app will use fallback data." -ForegroundColor Gray
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "🏆 NASA Space Apps Challenge 2025" -ForegroundColor Green
Write-Host "Exoplanet Detection Platform Ready!" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nPress any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")