"""
Configuration management for the RAG Argo Pipeline.
Handles environment variables and application settings.
"""

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = Field(default="RAG Argo Pipeline", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    reload: bool = Field(default=False, env="RELOAD")
    
    # Security
    secret_key: str = Field(env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    groq_api_key: Optional[str] = Field(default=None, env="GROQ_API_KEY")
    
    # Model Configuration
    embedding_model: str = Field(default="openai", env="EMBEDDING_MODEL")
    chunking_model: str = Field(default="openai", env="CHUNKING_MODEL")
    chat_model: str = Field(default="groq", env="CHAT_MODEL")
    
    # OpenAI
    openai_embedding_model: str = Field(default="text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    openai_chunking_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_CHUNKING_MODEL")
    openai_max_tokens: int = Field(default=1000, env="OPENAI_MAX_TOKENS")
    
    # Gemini
    gemini_embedding_model: str = Field(default="models/embedding-001", env="GEMINI_EMBEDDING_MODEL")
    gemini_chunking_model: str = Field(default="gemini-pro", env="GEMINI_CHUNKING_MODEL")
    
    # Groq
    groq_model: str = Field(default="llama3-70b-8192", env="GROQ_MODEL")
    groq_max_tokens: int = Field(default=8192, env="GROQ_MAX_TOKENS")
    groq_temperature: float = Field(default=0.1, env="GROQ_TEMPERATURE")
    
    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(default="argo_papers", env="QDRANT_COLLECTION_NAME")
    vector_dimension: int = Field(default=1536, env="VECTOR_DIMENSION")
    
    # Chunking
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    min_chunk_size: int = Field(default=100, env="MIN_CHUNK_SIZE")
    max_chunks_per_document: int = Field(default=100, env="MAX_CHUNKS_PER_DOCUMENT")
    
    # PDF Processing
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    allowed_extensions: str = Field(default=".pdf", env="ALLOWED_EXTENSIONS")
    extraction_method: str = Field(default="pymupdf", env="EXTRACTION_METHOD")
    
    # Search
    default_search_limit: int = Field(default=10, env="DEFAULT_SEARCH_LIMIT")
    max_search_limit: int = Field(default=50, env="MAX_SEARCH_LIMIT")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(default=60, env="RATE_LIMIT_PERIOD")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    
    # File Paths
    data_dir: str = Field(default="./data", env="DATA_DIR")
    pdf_dir: str = Field(default="./data/pdfs", env="PDF_DIR")
    output_dir: str = Field(default="./outputs", env="OUTPUT_DIR")
    logs_dir: str = Field(default="./logs", env="LOGS_DIR")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    log_file: str = Field(default="app.log", env="LOG_FILE")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # Citation
    max_citations: int = Field(default=10, env="MAX_CITATIONS")
    citation_style: str = Field(default="apa", env="CITATION_STYLE")
    
    # Report Generation
    report_template: str = Field(default="scientific", env="REPORT_TEMPLATE")
    include_charts: bool = Field(default=True, env="INCLUDE_CHARTS")
    max_report_pages: int = Field(default=50, env="MAX_REPORT_PAGES")
    
    @field_validator("embedding_model", "chunking_model")
    @classmethod
    def validate_ai_models(cls, v):
        if v not in ["openai", "gemini"]:
            raise ValueError("AI model must be 'openai' or 'gemini'")
        return v
    
    @field_validator("chat_model")
    @classmethod
    def validate_chat_model(cls, v):
        if v != "groq":
            raise ValueError("Currently only 'groq' is supported for chat model")
        return v
    
    @field_validator("extraction_method")
    @classmethod
    def validate_extraction_method(cls, v):
        if v not in ["pymupdf", "pdfplumber", "pypdf2"]:
            raise ValueError("Extraction method must be 'pymupdf', 'pdfplumber', or 'pypdf2'")
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        if v not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError("Log level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
        return v
    
    model_config = {"env_file": ".env", "case_sensitive": False}


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Create settings instance
settings = get_settings()


def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        settings.data_dir,
        settings.pdf_dir,
        settings.output_dir,
        settings.logs_dir,
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


# Validate required API keys based on configuration
def validate_api_keys():
    """Validate that required API keys are present."""
    errors = []
    
    if settings.embedding_model == "openai" and not settings.openai_api_key:
        errors.append("OPENAI_API_KEY is required when using OpenAI for embeddings")
    
    if settings.chunking_model == "openai" and not settings.openai_api_key:
        errors.append("OPENAI_API_KEY is required when using OpenAI for chunking")
    
    if settings.embedding_model == "gemini" and not settings.gemini_api_key:
        errors.append("GEMINI_API_KEY is required when using Gemini for embeddings")
    
    if settings.chunking_model == "gemini" and not settings.gemini_api_key:
        errors.append("GEMINI_API_KEY is required when using Gemini for chunking")
    
    if not settings.groq_api_key:
        errors.append("GROQ_API_KEY is required")
    
    if not settings.secret_key or settings.secret_key == "your-super-secret-key-change-this-in-production":
        errors.append("SECRET_KEY must be set and changed from default value")
    
    if errors:
        raise ValueError(f"Configuration errors: {'; '.join(errors)}")


if __name__ == "__main__":
    # Test configuration loading
    try:
        validate_api_keys()
        create_directories()
        print("✅ Configuration loaded successfully")
        print(f"📁 Data directory: {settings.data_dir}")
        print(f"🤖 Embedding model: {settings.embedding_model}")
        print(f"🤖 Chat model: {settings.chat_model}")
        print(f"🗄️ Vector database: {settings.qdrant_url}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
