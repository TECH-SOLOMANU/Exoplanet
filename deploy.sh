#!/bin/bash

echo "========================================"
echo "NASA Exoplanet Detection Platform"
echo "Quick Deployment Script"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo
echo -e "${BLUE}Checking Docker installation...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: Docker is not installed${NC}"
    echo "Please install Docker and try again"
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi
echo -e "${GREEN}✅ Docker is available${NC}"

echo
echo -e "${BLUE}Checking Docker Compose...${NC}"
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}ERROR: Docker Compose is not available${NC}"
    echo "Please install Docker Compose and try again"
    exit 1
fi
echo -e "${GREEN}✅ Docker Compose is available${NC}"

echo
echo -e "${BLUE}Checking if Docker is running...${NC}"
if ! docker info &> /dev/null; then
    echo -e "${RED}ERROR: Docker is not running${NC}"
    echo "Please start Docker and try again"
    exit 1
fi
echo -e "${GREEN}✅ Docker is running${NC}"

echo
echo -e "${YELLOW}Building and starting services...${NC}"
echo "This may take 5-10 minutes on first run..."
docker-compose up -d --build

echo
echo -e "${BLUE}Waiting for services to start...${NC}"
sleep 30

echo
echo "========================================"
echo -e "${GREEN}🚀 Deployment Complete!${NC}"
echo "========================================"
echo
echo -e "${BLUE}Access your application:${NC}"
echo -e "🌐 Frontend:  ${YELLOW}http://localhost:3000${NC}"
echo -e "🔧 Backend:   ${YELLOW}http://localhost:8000${NC}"
echo -e "📚 API Docs:  ${YELLOW}http://localhost:8000/docs${NC}"
echo -e "🗄️ MongoDB:   ${YELLOW}localhost:27017${NC}"
echo

echo -e "${BLUE}Checking service status...${NC}"
docker-compose ps

echo
echo -e "${BLUE}Useful commands:${NC}"
echo -e "📋 View logs:     ${YELLOW}docker-compose logs -f${NC}"
echo -e "🛑 Stop services: ${YELLOW}docker-compose down${NC}"
echo -e "🔄 Restart:       ${YELLOW}docker-compose restart${NC}"
echo -e "🔍 Service status:${YELLOW}docker-compose ps${NC}"

echo
echo "========================================"
echo -e "${GREEN}🏆 NASA Space Apps Challenge 2025${NC}"
echo -e "${GREEN}Exoplanet Detection Platform Ready!${NC}"
echo "========================================"