# 🔧 Production Configuration Guide

This guide covers all the mock/placeholder values that need to be updated for a fully functional RAG Argo Pipeline.

## 🔑 Required API Keys (CRITICAL)

### 1. OpenAI API Key
- **File**: `.env`
- **Variable**: `OPENAI_API_KEY`
- **Current**: `"sk-your-openai-api-key-here"`
- **Get from**: https://platform.openai.com/api-keys
- **Required for**: Text embeddings, GPT models
- **Format**: `sk-proj-...` or `sk-...`

### 2. Google Gemini API Key
- **File**: `.env` 
- **Variable**: `GEMINI_API_KEY`
- **Current**: `"your-gemini-api-key-here"`
- **Get from**: https://makersuite.google.com/app/apikey
- **Required for**: Alternative embeddings (optional if using OpenAI)
- **Format**: `AIzaSy...`

### 3. Groq API Key
- **File**: `.env`
- **Variable**: `GROQ_API_KEY`
- **Current**: `"gsk_your-groq-api-key-here"`
- **Get from**: https://console.groq.com/keys
- **Required for**: Fast LLM inference
- **Format**: `gsk_...`

### 4. Secret Key for JWT
- **File**: `.env`
- **Variable**: `SECRET_KEY`
- **Current**: `"your-super-secret-key-change-this-in-production"`
- **Generate**: Use `openssl rand -hex 32` or Python:
```python
import secrets
print(secrets.token_hex(32))
```

## 🗄️ Database Configurations

### 1. Qdrant Vector Database
- **Local Development**: `http://localhost:6333` (works out of the box)
- **Production Options**:
  - **Qdrant Cloud**: Get from https://cloud.qdrant.io/
    - Update `QDRANT_URL` to your cloud URL
    - Set `QDRANT_API_KEY` to your API key
  - **Self-hosted**: Update to your server IP/domain

### 2. Redis Cache
- **Local Development**: `redis://localhost:6379/0` (works with Docker)
- **Production Options**:
  - **Redis Cloud**: https://redis.com/try-free/
  - **AWS ElastiCache**: Redis endpoint
  - **Self-hosted**: Update `REDIS_URL` with your Redis server

## 🌐 Environment-Specific Settings

### Development Environment
```bash
ENVIRONMENT="development"
DEBUG=True
RELOAD=True
LOG_LEVEL="DEBUG"
```

### Production Environment
```bash
ENVIRONMENT="production"
DEBUG=False
RELOAD=False
LOG_LEVEL="INFO"
HOST="0.0.0.0"  # or your specific IP
PORT=8000
```

## 📁 File Paths (Update for Production)

Current paths are relative - update for production:
```bash
# Current (local development)
DATA_DIR="./data"
PDF_DIR="./data/pdfs"
OUTPUT_DIR="./outputs"
LOGS_DIR="./logs"

# Production example
DATA_DIR="/var/app/data"
PDF_DIR="/var/app/data/pdfs"
OUTPUT_DIR="/var/app/outputs"
LOGS_DIR="/var/app/logs"
```

## 🔒 Security Settings

### 1. Update Grafana Credentials
**File**: `docker-compose.yml`
```yaml
environment:
  - GF_SECURITY_ADMIN_USER=admin
  - GF_SECURITY_ADMIN_PASSWORD=admin123  # CHANGE THIS!
```

### 2. Rate Limiting (Adjust for your needs)
```bash
RATE_LIMIT_REQUESTS=100  # requests per period
RATE_LIMIT_PERIOD=60     # seconds
```

## 🚀 Domain and SSL Configuration

### 1. Update nginx.conf for your domain
**File**: `nginx.conf`
- Replace `localhost` with your actual domain
- Add SSL certificates in `/ssl` directory

### 2. Update CORS settings
**File**: `app/main.py`
- Update `allow_origins` for your frontend domains

## 📊 Monitoring Credentials

### Grafana Dashboard Access
- **URL**: `http://your-domain:3000`
- **Default**: admin/admin123
- **Change in**: `docker-compose.yml` under grafana service

### Prometheus Metrics
- **URL**: `http://your-domain:9090`
- No authentication by default

## 🔧 Step-by-Step Setup

### 1. Copy and Configure Environment
```bash
# Copy the example file
cp .env.example .env

# Edit with your actual values
nano .env  # or use your preferred editor
```

### 2. Update Required Variables
**Minimum required changes in `.env`:**
```bash
# API Keys (MUST UPDATE)
OPENAI_API_KEY="sk-your-actual-openai-key"
GROQ_API_KEY="gsk_your-actual-groq-key"
SECRET_KEY="your-generated-64-char-secret"

# Environment
ENVIRONMENT="production"
DEBUG=False
```

### 3. Database URLs (if using cloud services)
```bash
# If using Qdrant Cloud
QDRANT_URL="https://your-cluster.qdrant.tech:6333"
QDRANT_API_KEY="your-qdrant-api-key"

# If using Redis Cloud
REDIS_URL="redis://username:password@redis-server:port/0"
REDIS_PASSWORD="your-redis-password"
```

### 4. Update docker-compose.yml
```yaml
# Change Grafana password
- GF_SECURITY_ADMIN_PASSWORD=your-secure-password

# Add environment-specific settings if needed
```

## ⚠️ Security Checklist

- [ ] Updated all API keys with real values
- [ ] Changed SECRET_KEY from default
- [ ] Updated Grafana admin password
- [ ] Set ENVIRONMENT="production"
- [ ] Set DEBUG=False
- [ ] Configured proper CORS origins
- [ ] Set up SSL certificates (for production)
- [ ] Updated Redis password (if exposed)

## 🧪 Testing Configuration

After updating configuration:
```bash
# Test with Docker
docker-compose up -d

# Check health
curl http://localhost/health

# View API docs
# Visit: http://localhost/docs
```

## 📝 Configuration Validation

The application will validate configuration on startup and show errors for:
- Missing required API keys
- Invalid model configurations
- Database connection issues
- Missing directories

Check logs for configuration errors:
```bash
docker-compose logs rag-api
```

## 🆘 Quick Start Checklist

**Minimum changes to get running:**
1. Get OpenAI API key → Update `OPENAI_API_KEY`
2. Get Groq API key → Update `GROQ_API_KEY`
3. Generate secret → Update `SECRET_KEY`
4. Run: `docker-compose up -d`
5. Test: `curl http://localhost/health`

**Everything else can use defaults for initial testing!**
