# RAG Argo Pipeline

A comprehensive, industry-ready RAG (Retrieval-Augmented Generation) pipeline for analyzing Argo oceanographic data. This system processes PDF research papers, creates intelligent vector embeddings, and provides AI-powered query responses with proper citations.

## 🌊 Features

- **Multi-Modal AI Integration**: OpenAI, Gemini for embeddings and chunking, Groq for response generation
- **Intelligent Document Processing**: PDF extraction with metadata parsing and semantic chunking
- **Vector Search**: Qdrant-powered similarity search with advanced filtering
- **Professional Report Generation**: PDF reports with multiple templates (scientific, business, academic)
- **Citation Management**: Automatic citation extraction and APA formatting
- **Enterprise Security**: JWT authentication, rate limiting, API key management
- **Production Ready**: Docker deployment with monitoring, logging, and health checks

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- API Keys for OpenAI, Gemini, and Groq
- Python 3.11+ (for local development)

### 1. Environment Setup

```bash
# Clone and navigate to project
git clone <your-repo-url>
cd rag_argo_pipeline

# Copy and configure environment
cp .env.example .env
```

Edit `.env` with your API keys:

```env
OPENAI_API_KEY="sk-your-openai-key"
GEMINI_API_KEY="your-gemini-key"
GROQ_API_KEY="gsk_your-groq-key"
SECRET_KEY="your-secure-secret-key"
```

### 2. Deploy with Docker

```bash
# Make deployment script executable (Linux/Mac)
chmod +x deploy.sh

# Deploy full stack
./deploy.sh deploy

# Or manually:
docker-compose up -d
```

### 3. Verify Deployment

```bash
# Check health
curl http://localhost/health

# View API documentation
# Visit: http://localhost/docs
```

## 📚 API Usage

### Upload Documents

```bash
curl -X POST "http://localhost/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_paper.pdf"
```

### Semantic Search

```bash
curl -X POST "http://localhost/search/semantic" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Argo float temperature measurements",
    "limit": 10,
    "similarity_threshold": 0.7
  }'
```

### RAG Query with Citations

```bash
curl -X POST "http://localhost/search/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do Argo floats measure ocean temperature?",
    "response_format": "detailed",
    "include_citations": true
  }'
```

### Generate Professional Report

```bash
curl -X POST "http://localhost/reports/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Deep water formation in the North Atlantic",
    "title": "North Atlantic Deep Water Analysis",
    "template": "scientific",
    "include_citations": true,
    "include_charts": true
  }'
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │   Qdrant Vector  │    │   Document      │
│   (Port 8000)   │◄──►│   Database       │◄──►│   Processing    │
│                 │    │   (Port 6333)    │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ▲                        ▲                        ▲
         │                        │                        │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Nginx Proxy   │    │   Redis Cache    │    │   AI Services   │
│   (Port 80/443) │    │   (Port 6379)    │    │   OpenAI/Gemini │
│                 │    │                  │    │   Groq          │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🛠️ Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start development server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 📋 Configuration

### Core Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings | Required |
| `GEMINI_API_KEY` | Google Gemini API key | Required |
| `GROQ_API_KEY` | Groq API key for responses | Required |
| `QDRANT_URL` | Qdrant database URL | `http://localhost:6333` |
| `CHUNK_SIZE` | Text chunk size | `1000` |
| `MAX_FILE_SIZE_MB` | Max PDF file size | `50` |

### AI Model Configuration

```env
# Choose embedding provider: "openai" or "gemini"
EMBEDDING_MODEL="openai"

# Choose chunking provider: "openai" or "gemini"
CHUNKING_MODEL="openai"

# Groq model for responses
GROQ_MODEL="llama3-70b-8192"
```

## 🔒 Security Features

- JWT-based authentication
- API key validation
- Rate limiting (configurable per endpoint)
- Input sanitization and validation
- HTTPS support (production)
- Security headers and CSP
- IP blocking capabilities
- Audit logging

### Enable Authentication

```python
from app.utils.security import require_api_key

@app.get("/secure-endpoint")
async def secure_endpoint(api_key: str = Depends(require_api_key)):
    return {"message": "Authenticated access"}
```

## 📊 Monitoring

### Built-in Endpoints

- Health Check: `/health`
- Metrics: `/metrics` (Prometheus format)
- System Info: `/info`

### Grafana Dashboards

Access Grafana at `http://localhost:3000`:
- Username: `admin`
- Password: `admin123`

### Log Management

```bash
# View all logs
docker-compose logs -f

# View specific service
docker-compose logs -f rag-api
docker-compose logs -f qdrant
```

## 📖 API Documentation

Interactive API documentation available at:
- Swagger UI: `http://localhost/docs`
- ReDoc: `http://localhost/redoc`

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/documents/upload` | POST | Upload and process PDF |
| `/search/semantic` | POST | Vector similarity search |
| `/search/query` | POST | RAG query with AI response |
| `/reports/generate` | POST | Generate PDF report |
| `/health` | GET | System health check |

## 🚀 Production Deployment

### Environment Variables

```env
ENVIRONMENT=production
SECRET_KEY=your-secure-production-key
QDRANT_URL=http://qdrant:6333
REDIS_URL=redis://redis:6379/0
```

### SSL Configuration

1. Place SSL certificates in `./ssl/` directory
2. Update `nginx.conf` with SSL settings
3. Configure domain name

### Scaling

```bash
# Scale API instances
docker-compose up -d --scale rag-api=3

# Use external load balancer for production
```

## 🧪 Testing

```bash
# Run tests (when implemented)
pytest tests/

# Test API endpoints
python scripts/test_api.py

# Load testing
# Use tools like Apache Bench or k6
```

## 🔧 Maintenance

### Backup Data

```bash
./deploy.sh backup
```

### Update Application

```bash
./deploy.sh update
```

### Monitor Resources

```bash
# Check container stats
docker stats

# Check disk usage
df -h
docker system df
```

## 🐛 Troubleshooting

### Common Issues

1. **API Keys Not Working**
   - Verify keys in `.env` file
   - Check key permissions and quotas

2. **Qdrant Connection Issues**
   - Ensure Qdrant is running: `docker-compose ps`
   - Check logs: `docker-compose logs qdrant`

3. **Memory Issues**
   - Increase Docker memory limits
   - Monitor with `docker stats`

4. **PDF Processing Fails**
   - Check file size limits
   - Verify PDF is not corrupted
   - Check logs for specific errors

### Debug Mode

```bash
# Enable debug logging
export DEBUG=True
export LOG_LEVEL=DEBUG

# Restart services
docker-compose restart rag-api
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- Create an issue for bugs or feature requests
- Check documentation at `/docs` endpoint
- Monitor system health at `/health` endpoint

---

**Built for Production** • **Enterprise Ready** • **Highly Scalable**
