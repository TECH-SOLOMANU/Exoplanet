#!/bin/bash

# AWS ECS Deployment Script for NASA Exoplanet Detection Platform
# NASA Space Apps Challenge 2025

set -e

# Configuration
PROJECT_NAME="exoplanet-detection"
AWS_REGION="us-east-1"
CLUSTER_NAME="$PROJECT_NAME-cluster"
ECR_REPO_PREFIX="$PROJECT_NAME"

echo "=========================================="
echo "🚀 AWS ECS Deployment"
echo "NASA Exoplanet Detection Platform"
echo "=========================================="

# Check prerequisites
echo "📋 Checking prerequisites..."
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Please install it first."
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install it first."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "📋 AWS Account ID: $ACCOUNT_ID"

# Create ECR repositories
echo "🏗️  Creating ECR repositories..."
aws ecr describe-repositories --repository-names "$PROJECT_NAME-backend" --region $AWS_REGION 2>/dev/null || \
aws ecr create-repository --repository-name "$PROJECT_NAME-backend" --region $AWS_REGION

aws ecr describe-repositories --repository-names "$PROJECT_NAME-frontend" --region $AWS_REGION 2>/dev/null || \
aws ecr create-repository --repository-name "$PROJECT_NAME-frontend" --region $AWS_REGION

# Login to ECR
echo "🔐 Logging into ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build and push backend
echo "🏗️  Building and pushing backend..."
docker build -t $PROJECT_NAME-backend ./backend
docker tag $PROJECT_NAME-backend:latest $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$PROJECT_NAME-backend:latest
docker push $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$PROJECT_NAME-backend:latest

# Build and push frontend
echo "🏗️  Building and pushing frontend..."
docker build -f ./frontend/Dockerfile.prod -t $PROJECT_NAME-frontend ./frontend
docker tag $PROJECT_NAME-frontend:latest $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$PROJECT_NAME-frontend:latest
docker push $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$PROJECT_NAME-frontend:latest

# Create ECS cluster
echo "🏗️  Creating ECS cluster..."
aws ecs describe-clusters --clusters $CLUSTER_NAME --region $AWS_REGION 2>/dev/null || \
aws ecs create-cluster --cluster-name $CLUSTER_NAME --region $AWS_REGION

# Create task definitions and services (simplified)
echo "🚀 Deploying services..."
echo "📋 Backend Image: $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$PROJECT_NAME-backend:latest"
echo "📋 Frontend Image: $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$PROJECT_NAME-frontend:latest"

echo "=========================================="
echo "✅ Deployment completed!"
echo "=========================================="
echo "📋 Next steps:"
echo "1. Create task definitions in AWS Console"
echo "2. Create services with load balancers"
echo "3. Configure environment variables"
echo "4. Set up RDS for MongoDB (optional)"
echo "5. Configure Route 53 for custom domain"
echo ""
echo "🌐 Images are available in ECR:"
echo "   Backend:  $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$PROJECT_NAME-backend:latest"
echo "   Frontend: $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$PROJECT_NAME-frontend:latest"
echo "=========================================="