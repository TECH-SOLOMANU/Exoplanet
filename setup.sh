#!/bin/bash

# NASA Space Apps Challenge 2025 - Exoplanet Detection Platform
# Quick Setup Script

echo "🌟 NASA Space Apps Challenge 2025 - Exoplanet Detection Platform Setup"
echo "=================================================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create .env file from example
if [ ! -f .env ]; then
    echo "📝 Creating .env file from example..."
    cp .env.example .env
    echo "✅ .env file created. Please review and update if needed."
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data models logs

# Download sample data (placeholder)
echo "📊 Setting up sample data structure..."
mkdir -p data/nasa_archive data/light_curves data/processed

# Build and start services
echo "🚀 Building and starting services..."
docker-compose up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check service status
echo "🔍 Checking service status..."
docker-compose ps

# Display access information
echo ""
echo "🎉 Setup Complete!"
echo "==================="
echo "📊 Dashboard: http://localhost:3000"
echo "🔗 API Docs: http://localhost:8000/docs"
echo "🗄️  MongoDB: localhost:27017"
echo "📡 Redis: localhost:6379"
echo ""
echo "🚀 Next steps:"
echo "1. Visit the dashboard at http://localhost:3000"
echo "2. Click 'Fetch Latest NASA Data' to load exoplanet data"
echo "3. Explore the predictions and upload your own data"
echo ""
echo "📚 For more information, see README.md"
echo "🐛 To view logs: docker-compose logs -f"
echo "🛑 To stop: docker-compose down"