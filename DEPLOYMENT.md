# 🚀 Deployment Guide - NASA Exoplanet Detection Platform

## Quick Start (Local Docker)

### Prerequisites
- Docker Desktop installed
- Git repository cloned
- 8GB+ RAM recommended

### 1. Local Docker Deployment (5 minutes)

```bash
# Clone and navigate
git clone https://github.com/TECH-SOLOMANU/Exoplanet.git
cd Exoplanet

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f backend
```

**Access Points:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- MongoDB: localhost:27017

### 2. Production Docker Deployment

```bash
# Production with Nginx reverse proxy
docker-compose --profile production up -d

# Monitor services
docker-compose logs -f
```

## Cloud Deployment Options

### 3. AWS Deployment (Recommended for Competition)

#### Option A: AWS ECS Fargate (Serverless)

```bash
# Install AWS CLI and ECS CLI
aws configure

# Create ECS cluster
ecs-cli up --cluster exoplanet-cluster --launch-type FARGATE

# Deploy services
ecs-cli compose --project-name exoplanet service up --launch-type FARGATE
```

#### Option B: AWS EC2 with Docker

```bash
# Launch EC2 instance (t3.medium or larger)
# Install Docker
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# Deploy application
git clone https://github.com/TECH-SOLOMANU/Exoplanet.git
cd Exoplanet
docker-compose up -d
```

**Cost Estimate:** $20-50/month

### 4. Google Cloud Platform (GCP)

#### Cloud Run Deployment

```bash
# Install gcloud CLI
gcloud auth login

# Build and deploy backend
gcloud builds submit --tag gcr.io/YOUR-PROJECT/exoplanet-backend ./backend
gcloud run deploy exoplanet-backend --image gcr.io/YOUR-PROJECT/exoplanet-backend --platform managed

# Build and deploy frontend
gcloud builds submit --tag gcr.io/YOUR-PROJECT/exoplanet-frontend ./frontend
gcloud run deploy exoplanet-frontend --image gcr.io/YOUR-PROJECT/exoplanet-frontend --platform managed
```

**Cost Estimate:** $10-30/month

### 5. Azure Container Instances

```bash
# Install Azure CLI
az login

# Create resource group
az group create --name exoplanet-rg --location eastus

# Deploy container group
az container create --resource-group exoplanet-rg --file docker-compose.yml
```

### 6. Free Deployment Options (Perfect for Demo)

#### Render.com (Free Tier)

1. Connect GitHub repository to Render
2. Create Web Service for backend
3. Create Static Site for frontend
4. Add MongoDB Atlas free cluster

#### Railway.app

```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy
railway login
railway init
railway up
```

#### Heroku (Container Deployment)

```bash
# Install Heroku CLI
heroku login

# Create apps
heroku create exoplanet-backend
heroku create exoplanet-frontend

# Set up container deployment
heroku container:push web -a exoplanet-backend
heroku container:release web -a exoplanet-backend
```

## NASA Space Apps Challenge Specific Setup

### Competition Demo Setup (5 minutes)

```bash
# Quick demo deployment
docker-compose up -d mongodb redis backend

# Wait for services to start (30 seconds)
sleep 30

# Start frontend
cd frontend && npm start
```

### Environment Variables for Production

```env
# Backend (.env)
MONGODB_URL=mongodb://your-cluster.mongodb.net
REDIS_URL=redis://your-redis-instance
DATABASE_NAME=exoplanet_production
NASA_API_KEY=your-nasa-api-key
SECRET_KEY=your-secret-key

# Frontend (.env)
REACT_APP_API_URL=https://your-backend-domain.com
REACT_APP_ENV=production
```

## Monitoring and Maintenance

### Health Checks

```bash
# Check service health
curl http://localhost:8000/health
curl http://localhost:3000

# Monitor logs
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Backup Strategy

```bash
# Backup MongoDB
docker exec exoplanet_mongodb mongodump --db exoplanet_db --out /backup

# Backup models
docker cp exoplanet_backend:/app/models ./models_backup
```

## Performance Optimization

### Production Configuration

```yaml
# docker-compose.prod.yml
services:
  backend:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
  
  frontend:
    environment:
      - NODE_ENV=production
    command: npm run build && serve -s build -l 3000
```

### Scaling

```bash
# Scale backend services
docker-compose up -d --scale backend=3

# Load balancer with Nginx
# (Already configured in docker-compose.yml)
```

## Troubleshooting

### Common Issues

1. **Port 800 vs 8000**: Update frontend proxy in package.json
2. **MongoDB Connection**: Ensure MongoDB is running first
3. **Memory Issues**: Increase Docker memory allocation
4. **API Rate Limits**: Add NASA API key for higher limits

### Debug Commands

```bash
# Check container status
docker-compose ps

# View detailed logs
docker-compose logs backend | grep ERROR

# Restart services
docker-compose restart backend

# Clean rebuild
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

## Cost Analysis

| Platform | Free Tier | Paid Tier | Best For |
|----------|-----------|-----------|----------|
| Render | ✅ Limited | $7/month | Demo/Competition |
| Railway | ✅ $5 credit | $5/month | Quick deployment |
| AWS | ✅ 12 months | $20-50/month | Production |
| GCP | ✅ $300 credit | $10-30/month | Scalability |
| Azure | ✅ $200 credit | $15-40/month | Enterprise |

## NASA Competition Recommendations

### For Judges Demo:
1. **Local Docker** (Most reliable)
2. **Render.com** (Public accessible)
3. **GitHub Codespaces** (Browser-based)

### For Production:
1. **AWS ECS Fargate** (Scalable)
2. **GCP Cloud Run** (Cost-effective)
3. **Azure Container Apps** (Enterprise-ready)

---

## 🏆 Competition-Ready Deployment

Your NASA Space Apps Challenge project is **100% deployment-ready** with:

- ✅ Docker containerization
- ✅ Multi-service orchestration
- ✅ Production configurations
- ✅ Health monitoring
- ✅ Scalable architecture
- ✅ Cloud platform compatibility

Choose the deployment method that best fits your demo requirements!