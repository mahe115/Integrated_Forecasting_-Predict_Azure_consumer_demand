# DEPLOYMENT_GUIDE.md

# 🚀 Production Deployment Guide

## Azure Demand Forecasting Platform - Enterprise Deployment

*Complete guide for deploying the Azure Demand Forecasting Platform in production environments*

---

## 🎯 **Deployment Overview**

This guide covers three deployment scenarios:
- **Development Setup**: Local development and testing
- **Production Deployment**: Enterprise-ready configuration
- **Cloud Deployment**: Azure, AWS, GCP deployment options

**System Requirements**:
- Python 3.8+ with pip
- 8GB RAM (minimum 4GB for development)
- 5GB free disk space
- Modern web browser (Chrome, Firefox, Safari)

---

## 🏠 **Development Setup (Recommended for Evaluation)**

### Quick Start (60 seconds)
```bash
# 1. Clone the repository
git clone https://github.com/your-org/IntegratedForecasting-PredictAzuredemand.git
cd IntegratedForecasting-PredictAzuredemand

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch complete system
python start_pipeline.py
```

**Access Points**:
- Main Dashboard: `http://localhost:8501`
- API Endpoints: `http://localhost:5000/api`
- Health Check: `http://localhost:5000/api/health`

### Alternative Manual Setup
```bash
# Terminal 1: Backend API Server
python optimised_backend_app.py
# Output: Flask server running on http://localhost:5000

# Terminal 2: Frontend Dashboard  
streamlit run dashboard_app.py
# Output: Streamlit dashboard at http://localhost:8501

# Terminal 3: ML Training Pipeline (Optional)
python model_training_pipeline.py
# Output: AI models training with intelligent selection
```

---

## 🏭 **Production Deployment**

### 1. Production Dependencies
```bash
# Install production dependencies
pip install -r requirements.txt
pip install gunicorn supervisor nginx

# Optional: Install additional production packages
pip install redis celery prometheus_client
```

### 2. Production Backend Configuration

**Create `gunicorn.conf.py`**:
```python
# gunicorn.conf.py
bind = "0.0.0.0:5000"
workers = 4
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
preload_app = True
keepalive = 60

# Logging
accesslog = "/var/log/gunicorn/access.log"
errorlog = "/var/log/gunicorn/error.log"
loglevel = "info"

# Process naming
proc_name = "azure_forecasting_api"

# Worker timeout
timeout = 300
graceful_timeout = 30

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190
```

**Start Production Backend**:
```bash
# Create log directory
sudo mkdir -p /var/log/gunicorn
sudo chown $USER:$USER /var/log/gunicorn

# Start with gunicorn
gunicorn --config gunicorn.conf.py optimised_backend_app:app
```

### 3. Production Frontend Configuration

**Create `streamlit_config.toml`**:
```toml
# ~/.streamlit/config.toml
[server]
port = 8501
address = "0.0.0.0"
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
serverAddress = "your-domain.com"
serverPort = 8501

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF" 
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[logger]
level = "info"
```

**Start Production Frontend**:
```bash
streamlit run dashboard_app.py --server.port 8501 --server.address 0.0.0.0
```

### 4. Process Management with Supervisor

**Create `/etc/supervisor/conf.d/azure_forecasting.conf`**:
```ini
[program:azure_forecasting_api]
command=gunicorn --config gunicorn.conf.py optimised_backend_app:app
directory=/path/to/your/project
user=azureuser
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/supervisor/azure_api.log

[program:azure_forecasting_frontend]
command=streamlit run dashboard_app.py --server.port 8501 --server.address 0.0.0.0
directory=/path/to/your/project
user=azureuser
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/supervisor/azure_frontend.log

[program:azure_ml_training]
command=python model_training_pipeline.py
directory=/path/to/your/project
user=azureuser
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/supervisor/azure_training.log
```

**Control services**:
```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start all
sudo supervisorctl status
```

### 5. Reverse Proxy with Nginx

**Create `/etc/nginx/sites-available/azure_forecasting`**:
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL Configuration
    ssl_certificate /path/to/ssl/certificate.crt;
    ssl_certificate_key /path/to/ssl/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    # Frontend (Dashboard)
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
    
    # API Backend
    location /api/ {
        proxy_pass http://127.0.0.1:5000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Rate limiting
        limit_req zone=api burst=20 nodelay;
        
        # CORS headers
        add_header Access-Control-Allow-Origin "*" always;
        add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS" always;
        add_header Access-Control-Allow-Headers "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range" always;
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://127.0.0.1:5000/api/health;
        access_log off;
    }
    
    # Static files caching
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}

# Rate limiting zone
http {
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
}
```

**Enable site**:
```bash
sudo ln -s /etc/nginx/sites-available/azure_forecasting /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## ☁️ **Cloud Deployment Options**

### Azure Deployment

**1. Azure App Service Deployment**:
```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login and create resource group
az login
az group create --name azure-forecasting-rg --location eastus

# Create App Service plan
az appservice plan create --name azure-forecasting-plan \
    --resource-group azure-forecasting-rg \
    --sku B2 --is-linux

# Deploy backend API
az webapp create --resource-group azure-forecasting-rg \
    --plan azure-forecasting-plan \
    --name azure-forecasting-api \
    --runtime "PYTHON|3.8"

# Deploy frontend
az webapp create --resource-group azure-forecasting-rg \
    --plan azure-forecasting-plan \
    --name azure-forecasting-dashboard \
    --runtime "PYTHON|3.8"
```

**2. Azure Container Instances**:
```dockerfile
# Dockerfile.api
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "--config", "gunicorn.conf.py", "optimised_backend_app:app"]
```

```dockerfile
# Dockerfile.dashboard  
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "dashboard_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Deploy containers**:
```bash
# Build and push to Azure Container Registry
az acr create --resource-group azure-forecasting-rg \
    --name azureforecastingacr --sku Basic

# Build and deploy
docker build -t azureforecastingacr.azurecr.io/api:latest -f Dockerfile.api .
docker build -t azureforecastingacr.azurecr.io/dashboard:latest -f Dockerfile.dashboard .

az acr login --name azureforecastingacr
docker push azureforecastingacr.azurecr.io/api:latest
docker push azureforecastingacr.azurecr.io/dashboard:latest

# Create container instances
az container create --resource-group azure-forecasting-rg \
    --name azure-forecasting-api \
    --image azureforecastingacr.azurecr.io/api:latest \
    --ports 5000 \
    --cpu 2 --memory 4

az container create --resource-group azure-forecasting-rg \
    --name azure-forecasting-dashboard \
    --image azureforecastingacr.azurecr.io/dashboard:latest \
    --ports 8501 \
    --cpu 2 --memory 4
```

### AWS Deployment

**1. EC2 Deployment**:
```bash
# Launch EC2 instance
aws ec2 run-instances --image-id ami-0abcdef1234567890 \
    --count 1 --instance-type t3.large \
    --key-name your-key-pair \
    --security-groups azure-forecasting-sg

# Install dependencies and deploy
ssh -i your-key.pem ec2-user@your-ec2-ip
sudo yum update -y
sudo yum install python38 python38-pip -y
git clone your-repo
cd your-repo
pip3.8 install -r requirements.txt
python3.8 start_pipeline.py
```

**2. AWS ECS Deployment**:
```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.dashboard
    ports:
      - "8501:8501"
    depends_on:
      - api
    environment:
      - API_BASE_URL=http://api:5000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### GCP Deployment

**1. Google Cloud Run**:
```bash
# Enable APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Build and deploy API
gcloud builds submit --tag gcr.io/your-project/azure-forecasting-api
gcloud run deploy azure-forecasting-api \
    --image gcr.io/your-project/azure-forecasting-api \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated

# Build and deploy dashboard
gcloud builds submit --tag gcr.io/your-project/azure-forecasting-dashboard -f Dockerfile.dashboard
gcloud run deploy azure-forecasting-dashboard \
    --image gcr.io/your-project/azure-forecasting-dashboard \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated
```

---

## 🔒 **Security Configuration**

### 1. API Authentication
```python
# Add to optimised_backend_app.py
from flask_jwt_extended import JWTManager, create_access_token, jwt_required

app.config['JWT_SECRET_KEY'] = 'your-secret-key-change-in-production'
jwt = JWTManager(app)

@app.route('/api/auth/login', methods=['POST'])
def login():
    # Implement your authentication logic
    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token)

@app.route('/api/forecast/predict')
@jwt_required()
def forecast_predict():
    # Protected endpoint
    pass
```

### 2. Environment Variables
```bash
# .env file
FLASK_ENV=production
SECRET_KEY=your-super-secret-key
DATABASE_URL=postgresql://user:password@localhost/azure_forecasting
API_RATE_LIMIT=1000
JWT_SECRET_KEY=your-jwt-secret
SSL_CERT_PATH=/path/to/ssl/cert.pem
SSL_KEY_PATH=/path/to/ssl/key.pem
```

### 3. Firewall Configuration
```bash
# Ubuntu/Debian
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=22/tcp
sudo firewall-cmd --permanent --add-port=80/tcp  
sudo firewall-cmd --permanent --add-port=443/tcp
sudo firewall-cmd --reload
```

---

## 📊 **Monitoring & Observability**

### 1. Health Checks
```bash
# API health check
curl -f http://localhost:5000/api/health || exit 1

# Dashboard health check  
curl -f http://localhost:8501/_stcore/health || exit 1
```

### 2. Log Configuration
```python
# logging_config.py
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }
    },
    'handlers': {
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/azure_forecasting/app.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'default'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['file']
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
```

### 3. Performance Monitoring
```python
# Add to optimised_backend_app.py
from prometheus_client import Counter, Histogram, generate_latest
import time

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')

@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request  
def after_request(response):
    request_latency = time.time() - request.start_time
    REQUEST_LATENCY.observe(request_latency)
    REQUEST_COUNT.labels(method=request.method, endpoint=request.endpoint).inc()
    return response

@app.route('/metrics')
def metrics():
    return generate_latest()
```

---

## 🧪 **Testing & Validation**

### 1. Production Readiness Checklist
```bash
#!/bin/bash
# production_check.sh

echo "🔍 Production Readiness Check"

# Check API endpoints
echo "Testing API endpoints..."
curl -f http://localhost:5000/api/health || echo "❌ API health check failed"
curl -f http://localhost:5000/api/kpis || echo "❌ KPIs endpoint failed"

# Check dashboard
echo "Testing dashboard..."
curl -f http://localhost:8501/_stcore/health || echo "❌ Dashboard health check failed"

# Check ML models
echo "Testing ML models..."
curl -f http://localhost:5000/api/forecast/models || echo "❌ ML models check failed"

# Check disk space
df -h | grep -E '^/dev/' | awk '{ print $5 " " $1 }' | while read output;
do
  percentage=$(echo $output | awk '{ print $1}' | sed 's/%//g')
  partition=$(echo $output | awk '{ print $2 }')
  if [ $percentage -gt 90 ]; then
    echo "❌ Disk space critical: $partition ($percentage%)"
  fi
done

echo "✅ Production check complete"
```

### 2. Load Testing
```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Test API endpoints
ab -n 1000 -c 10 http://localhost:5000/api/kpis
ab -n 100 -c 5 http://localhost:5000/api/forecast/predict?days=30

# Test with different scenarios
curl -X POST http://localhost:5000/api/training/intelligent/trigger & # Background training
ab -n 500 -c 5 http://localhost:5000/api/health # While training
```

### 3. Backup & Recovery
```bash
#!/bin/bash
# backup.sh - Daily backup script

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/azure_forecasting"

# Create backup directory
mkdir -p $BACKUP_DIR/$DATE

# Backup database
sqlite3 model_performance.db ".backup $BACKUP_DIR/$DATE/model_performance.db"

# Backup models
tar -czf $BACKUP_DIR/$DATE/models.tar.gz models/ user_models/ storage_models/

# Backup logs
tar -czf $BACKUP_DIR/$DATE/logs.tar.gz /var/log/azure_forecasting/

# Clean old backups (keep 30 days)
find $BACKUP_DIR -type d -mtime +30 -exec rm -rf {} \;

echo "Backup completed: $BACKUP_DIR/$DATE"
```

---

## 🚨 **Troubleshooting**

### Common Issues & Solutions

**1. Port Already in Use**:
```bash
# Check what's using the port
sudo lsof -i :5000
sudo lsof -i :8501

# Kill process if needed
sudo kill -9 $(lsof -t -i:5000)
```

**2. Memory Issues**:
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Increase swap if needed
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile  
sudo mkswap /swapfile
sudo swapon /swapfile
```

**3. ML Models Not Loading**:
```bash
# Check model files exist
ls -la models/
ls -la user_models/
ls -la storage_models/

# Check permissions
sudo chown -R $USER:$USER models/ user_models/ storage_models/
chmod -R 755 models/ user_models/ storage_models/
```

**4. Database Connection Issues**:
```bash
# Check database file
ls -la model_performance.db
sqlite3 model_performance.db ".schema"

# Reset database if corrupted
rm model_performance.db
python model_training_pipeline.py  # Will recreate database
```

---

## 📋 **Maintenance**

### Daily Tasks
```bash
# Check system status
sudo supervisorctl status
curl http://localhost:5000/api/health

# Review logs
tail -100 /var/log/azure_forecasting/app.log
tail -100 /var/log/supervisor/azure_api.log

# Check disk usage
df -h
du -sh /path/to/project/*
```

### Weekly Tasks  
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Backup models and database
./backup.sh

# Review performance metrics
curl http://localhost:5000/metrics

# Test load performance
ab -n 100 -c 5 http://localhost:5000/api/kpis
```

### Monthly Tasks
```bash
# Review log files and rotate if needed
sudo logrotate -f /etc/logrotate.d/azure_forecasting

# Update Python dependencies (test in staging first)
pip list --outdated
pip install -r requirements.txt --upgrade

# Security updates
sudo apt list --upgradable | grep -i security
sudo apt upgrade -y
```

---

## 📞 **Support & Maintenance**

### Log Locations
- Application logs: `/var/log/azure_forecasting/`
- Supervisor logs: `/var/log/supervisor/`
- Nginx logs: `/var/log/nginx/`
- System logs: `/var/log/syslog`

### Key Configuration Files
- Application config: `config.py`
- Nginx config: `/etc/nginx/sites-available/azure_forecasting`
- Supervisor config: `/etc/supervisor/conf.d/azure_forecasting.conf`
- Gunicorn config: `gunicorn.conf.py`

### Emergency Procedures
```bash
# Emergency restart
sudo supervisorctl restart all
sudo systemctl restart nginx

# Rollback deployment
git checkout previous-stable-tag
sudo supervisorctl restart all

# Emergency maintenance mode
# Create maintenance.html and configure Nginx to serve it
```

---

*This deployment guide provides comprehensive coverage for production deployment of the Azure Demand Forecasting Platform. For additional support, refer to the [API Documentation](API_DOCUMENTATION.md) and [Technical Support](README.md#support).*