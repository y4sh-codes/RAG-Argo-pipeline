"""
Main FastAPI application for the RAG Argo Pipeline.
Industry-ready API with proper error handling, middleware, and documentation.
"""

import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import structlog

from .config import settings, validate_api_keys, create_directories
from .routers import documents, search, reports
from .services.qdrant_service import qdrant_service
from .services.groq_service import groq_service

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer() if settings.log_format == "json" else structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting RAG Argo Pipeline API")
    
    try:
        # Validate configuration
        validate_api_keys()
        create_directories()
        
        # Initialize services
        await startup_checks()
        
        logger.info("✅ Application started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down RAG Argo Pipeline API")


async def startup_checks():
    """Perform startup health checks for all services."""
    services_status = {}
    
    # Check Qdrant connection
    try:
        qdrant_healthy = await qdrant_service.check_connection()
        services_status["qdrant"] = "healthy" if qdrant_healthy else "unhealthy"
        if qdrant_healthy:
            await qdrant_service.ensure_collection_exists()
    except Exception as e:
        logger.error(f"Qdrant startup check failed: {str(e)}")
        services_status["qdrant"] = "error"
    
    # Check Groq connection
    try:
        groq_healthy = await groq_service.check_connection()
        services_status["groq"] = "healthy" if groq_healthy else "unhealthy"
    except Exception as e:
        logger.error(f"Groq startup check failed: {str(e)}")
        services_status["groq"] = "error"
    
    logger.info("Service health check completed", services=services_status)
    
    # Fail startup if critical services are down
    if services_status.get("qdrant") == "error":
        raise Exception("Qdrant database is not available")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    # RAG Argo Pipeline API
    
    A comprehensive research assistant API for Argo oceanographic data analysis.
    
    ## Features
    
    - **Document Processing**: Upload and process PDF research papers with intelligent chunking
    - **Semantic Search**: Vector-based similarity search across document collections  
    - **RAG Queries**: Natural language queries with AI-powered responses and citations
    - **Report Generation**: Professional PDF reports with multiple templates
    - **Citation Management**: Automatic citation extraction and formatting
    - **Multi-Model Support**: OpenAI, Gemini, and Groq integration
    
    ## Quick Start
    
    1. Upload PDF documents using `/documents/upload`
    2. Search for information using `/search/semantic` or `/search/query`
    3. Generate reports using `/reports/generate`
    
    ## Authentication
    
    All endpoints require proper API configuration. See documentation for setup details.
    """,
    contact={
        "name": "RAG Argo Pipeline Support",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment == "development" else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Custom middleware
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log all requests with timing information."""
    start_time = time.time()
    
    # Log request
    logger.info(
        "HTTP request started",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else "unknown"
    )
    
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log response
    logger.info(
        "HTTP request completed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=f"{process_time:.3f}s"
    )
    
    # Add timing header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    """Global error handling middleware."""
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(
            "Unhandled exception",
            error=str(e),
            method=request.method,
            url=str(request.url),
            exc_info=True
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred",
                "request_id": getattr(request.state, "request_id", "unknown")
            }
        )


# Include routers
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(reports.router)


# Root endpoint
@app.get("/", tags=["root"])
async def read_root():
    """API root endpoint with basic information."""
    return {
        "message": "Welcome to the RAG Argo Pipeline API",
        "version": settings.app_version,
        "environment": settings.environment,
        "status": "operational",
        "docs_url": "/docs" if settings.environment != "production" else None,
        "endpoints": {
            "documents": "/documents",
            "search": "/search", 
            "reports": "/reports",
            "health": "/health"
        }
    }


# Health check endpoint
@app.get("/health", tags=["system"])
@limiter.limit("60/minute")
async def health_check(request: Request):
    """
    Comprehensive health check for all system components.
    
    Returns the health status of the API and all connected services.
    """
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.app_version,
        "environment": settings.environment,
        "services": {}
    }
    
    # Check Qdrant
    try:
        qdrant_healthy = await qdrant_service.check_connection()
        collection_info = await qdrant_service.get_collection_info() if qdrant_healthy else {}
        
        health_status["services"]["qdrant"] = {
            "status": "healthy" if qdrant_healthy else "unhealthy",
            "url": settings.qdrant_url,
            "collection": settings.qdrant_collection_name,
            "documents": collection_info.get("points_count", 0)
        }
    except Exception as e:
        health_status["services"]["qdrant"] = {
            "status": "error",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check Groq
    try:
        groq_healthy = await groq_service.check_connection()
        health_status["services"]["groq"] = {
            "status": "healthy" if groq_healthy else "unhealthy",
            "model": settings.groq_model
        }
    except Exception as e:
        health_status["services"]["groq"] = {
            "status": "error", 
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check file system
    try:
        import os
        health_status["services"]["filesystem"] = {
            "status": "healthy",
            "pdf_dir_exists": os.path.exists(settings.pdf_dir),
            "output_dir_exists": os.path.exists(settings.output_dir),
            "data_dir_writable": os.access(settings.data_dir, os.W_OK)
        }
    except Exception as e:
        health_status["services"]["filesystem"] = {
            "status": "error",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Set overall status
    unhealthy_services = [
        service for service, info in health_status["services"].items()
        if info.get("status") != "healthy"
    ]
    
    if unhealthy_services:
        health_status["status"] = "degraded"
        health_status["issues"] = unhealthy_services
    
    # Return appropriate HTTP status
    status_code = 200 if health_status["status"] == "healthy" else 503
    
    return JSONResponse(content=health_status, status_code=status_code)


# System information endpoint
@app.get("/info", tags=["system"])
@limiter.limit("30/minute") 
async def system_info(request: Request):
    """Get system information and statistics."""
    try:
        collection_info = await qdrant_service.get_collection_info()
        
        info = {
            "application": {
                "name": settings.app_name,
                "version": settings.app_version,
                "environment": settings.environment,
                "python_version": "3.x",  # You could get actual version
            },
            "configuration": {
                "embedding_model": settings.embedding_model,
                "chat_model": settings.chat_model,
                "chunk_size": settings.chunk_size,
                "max_file_size_mb": settings.max_file_size_mb,
            },
            "database": {
                "collection_name": settings.qdrant_collection_name,
                "total_documents": collection_info.get("points_count", 0),
                "vector_dimension": settings.vector_dimension,
            },
            "features": {
                "pdf_processing": True,
                "semantic_search": True,
                "rag_queries": True,
                "report_generation": True,
                "citations": True,
            }
        }
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving system information")


# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=settings.app_name,
        version=settings.app_version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom schema properties
    openapi_schema["info"]["x-logo"] = {
        "url": "https://example.com/logo.png"
    }
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "APIKeyHeader": {
            "type": "apiKey",
            "in": "header", 
            "name": "X-API-Key"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured logging."""
    logger.warning(
        "HTTP exception occurred",
        status_code=exc.status_code,
        detail=exc.detail,
        url=str(request.url),
        method=request.method
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Exception",
            "status_code": exc.status_code,
            "detail": exc.detail,
            "timestamp": time.time()
        }
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle value errors."""
    logger.error(
        "Value error occurred",
        error=str(exc),
        url=str(request.url),
        method=request.method
    )
    
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid input",
            "message": str(exc),
            "timestamp": time.time()
        }
    )


# Development server startup
def start_server():
    """Start the development server."""
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
        access_log=True
    )


if __name__ == "__main__":
    start_server()
