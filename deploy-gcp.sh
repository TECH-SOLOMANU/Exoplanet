#!/bin/bash

# Google Cloud Platform Deployment Script
# NASA Exoplanet Detection Platform - NASA Space Apps Challenge 2025

set -e

# Configuration
PROJECT_ID="exoplanet-detection-$(date +%s)"
REGION="us-central1"
SERVICE_NAME="exoplanet-platform"

echo "=========================================="
echo "🚀 Google Cloud Platform Deployment"
echo "NASA Exoplanet Detection Platform"
echo "=========================================="

# Check prerequisites
echo "📋 Checking prerequisites..."
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install it first."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Set project
echo "🏗️  Setting up GCP project..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and deploy backend
echo "🏗️  Building and deploying backend..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/exoplanet-backend ./backend

gcloud run deploy exoplanet-backend \
    --image gcr.io/$PROJECT_ID/exoplanet-backend \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --port 8000 \
    --set-env-vars ENVIRONMENT=production

# Get backend URL
BACKEND_URL=$(gcloud run services describe exoplanet-backend --platform managed --region $REGION --format 'value(status.url)')

# Build and deploy frontend
echo "🏗️  Building and deploying frontend..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/exoplanet-frontend ./frontend \
    --substitutions _REACT_APP_API_URL=$BACKEND_URL

gcloud run deploy exoplanet-frontend \
    --image gcr.io/$PROJECT_ID/exoplanet-frontend \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --port 80

# Get frontend URL
FRONTEND_URL=$(gcloud run services describe exoplanet-frontend --platform managed --region $REGION --format 'value(status.url)')

echo "=========================================="
echo "✅ Deployment completed!"
echo "=========================================="
echo "🌐 Your application is now available:"
echo "   Frontend: $FRONTEND_URL"
echo "   Backend:  $BACKEND_URL"
echo "   API Docs: $BACKEND_URL/docs"
echo ""
echo "📋 Project ID: $PROJECT_ID"
echo "📋 Region: $REGION"
echo "=========================================="