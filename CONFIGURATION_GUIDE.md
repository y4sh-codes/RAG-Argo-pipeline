# 🔧 Configuration Guide

This guide covers all the placeholder values that need to be updated for a fully functional RAG Argo Pipeline.

## 🔑 Required API Keys (CRITICAL)

### 1. OpenAI API Key
- **File**: `.env`
- **Variable**: `OPENAI_API_KEY`
- **Current**: `"sk-your-openai-api-key-here"`
- **Get from**: https://platform.openai.com/api-keys
- **Required for**: Text embeddings, GPT models
- **Format**: `sk-proj-...` or `sk-...`

### 2. Groq API Key
- **File**: `.env`
- **Variable**: `GROQ_API_KEY`
- **Current**: `"gsk_your-groq-api-key-here"`
- **Get from**: https://console.groq.com/keys
- **Required for**: Fast LLM inference
- **Format**: `gsk_...`

## 🗄️ Database Configurations

### 1. Qdrant Vector Database
- **Local Development**: `http://localhost:6333` (works out of the box)
- **Cloud Options**:
  - **Qdrant Cloud**: Get from https://cloud.qdrant.io/
    - Update `QDRANT_URL` to your cloud URL
    - Set `QDRANT_API_KEY` to your API key
  - **Self-hosted**: Update to your server IP/domain

### 2. Redis Cache
- **Local Development**: `redis://localhost:6379/0` (works with local Redis)
- **Cloud Options**:
  - **Redis Cloud**: https://redis.com/try-free/
  - **AWS ElastiCache**: Redis endpoint
  - **Self-hosted**: Update `REDIS_URL` with your Redis server

## 📁 File Paths

Current paths are relative - update for your needs:
```bash
# Default (local development)
DATA_DIR="./data"
PDF_DIR="./data/pdfs"
OUTPUT_DIR="./outputs"
LOGS_DIR="./logs"

# Custom example
DATA_DIR="/your/custom/data"
PDF_DIR="/your/custom/data/pdfs"
OUTPUT_DIR="/your/custom/outputs"
LOGS_DIR="/your/custom/logs"
```

##  Step-by-Step Setup

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

# Environment
ENVIRONMENT="development"  # or "production"
DEBUG=True  # set to False for production
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

## 🧪 Testing Configuration

After updating configuration:
```bash
# Install dependencies
pip install -r requirements.txt

# Start the application
python -m uvicorn app.main:app --reload

# Check health
curl http://localhost:8000/health

# View API docs
# Visit: http://localhost:8000/docs
```

## 📝 Configuration Validation

The application will validate configuration on startup and show errors for:
- Missing required API keys
- Invalid model configurations
- Database connection issues
- Missing directories

Check logs for configuration errors:
```bash
# View application logs
tail -f logs/app.log

# Or if running directly
python -m uvicorn app.main:app --log-level debug
```

## 🆘 Quick Start Checklist

**Minimum changes to get running:**
1. Get OpenAI API key → Update `OPENAI_API_KEY`
2. Get Groq API key → Update `GROQ_API_KEY`
3. Run: `pip install -r requirements.txt`
4. Run: `python -m uvicorn app.main:app --reload`
5. Test: `curl http://localhost:8000/health`

**Everything else can use defaults for initial testing!**
