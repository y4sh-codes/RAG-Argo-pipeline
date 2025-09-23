"""
Pydantic models for the RAG Argo Pipeline.
Defines data structures for API requests, responses, and internal data handling.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, field_validator


class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""
    OPENAI = "openai"


class ChatProvider(str, Enum):
    """Supported chat providers."""
    GROQ = "groq"


# Document Models
class DocumentMetadata(BaseModel):
    """Metadata extracted from a document."""
    title: Optional[str] = None
    authors: List[str] = []
    publication_date: Optional[datetime] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    keywords: List[str] = []
    page_count: Optional[int] = None
    file_size: Optional[int] = None
    language: Optional[str] = None


class DocumentChunk(BaseModel):
    """A chunk of text from a document."""
    chunk_id: str
    document_id: str
    text: str
    page_number: Optional[int] = None
    chunk_index: int
    metadata: Dict[str, Any] = {}
    embedding: Optional[List[float]] = None
    
    model_config = {"arbitrary_types_allowed": True}


class Document(BaseModel):
    """A processed document."""
    document_id: str
    filename: str
    file_path: str
    content: str
    metadata: DocumentMetadata
    chunks: List[DocumentChunk] = []
    status: DocumentStatus = DocumentStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None


# API Request Models
class DocumentUploadResponse(BaseModel):
    """Response for document upload."""
    document_id: str
    filename: str
    status: DocumentStatus
    message: str


class ChunkingRequest(BaseModel):
    """Request for document chunking."""
    document_id: str
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200
    chunking_strategy: Optional[str] = "semantic"


class EmbeddingRequest(BaseModel):
    """Request for generating embeddings."""
    text: str
    provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    model: Optional[str] = None


class SearchRequest(BaseModel):
    """Request for semantic search."""
    query: str
    limit: int = Field(default=10, ge=1, le=50)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    filters: Optional[Dict[str, Any]] = None


class QueryRequest(BaseModel):
    """Request for RAG query."""
    query: str
    search_limit: int = Field(default=10, ge=1, le=50)
    include_citations: bool = True
    response_format: str = Field(default="detailed", pattern="^(brief|detailed|comprehensive)$")
    
    @field_validator("response_format")
    @classmethod
    def validate_response_format(cls, v):
        if v not in ["brief", "detailed", "comprehensive"]:
            raise ValueError("response_format must be 'brief', 'detailed', or 'comprehensive'")
        return v


class ReportRequest(BaseModel):
    """Request for generating a PDF report."""
    query: str
    title: str
    include_citations: bool = True
    include_charts: bool = True
    template: str = Field(default="scientific", pattern="^(scientific|business|academic)$")
    max_pages: int = Field(default=20, ge=1, le=100)


# API Response Models
class Citation(BaseModel):
    """A citation with source information."""
    document_id: str
    title: str
    authors: List[str]
    publication_date: Optional[datetime]
    doi: Optional[str]
    page_number: Optional[int]
    chunk_text: str
    relevance_score: float
    citation_text: str


class QueryResponse(BaseModel):
    """Response for RAG query."""
    query: str
    response: str
    citations: List[Citation]
    search_results_count: int
    processing_time: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SearchResult(BaseModel):
    """A search result item."""
    document_id: str
    chunk_id: str
    title: str
    text: str
    similarity_score: float
    page_number: Optional[int]
    authors: List[str]
    publication_date: Optional[datetime]


class SearchResponse(BaseModel):
    """Response for semantic search."""
    query: str
    results: List[SearchResult]
    total_results: int
    processing_time: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ReportResponse(BaseModel):
    """Response for report generation."""
    report_id: str
    filename: str
    file_path: str
    pages: int
    size_bytes: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# System Status Models
class SystemHealth(BaseModel):
    """System health status."""
    status: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    services: Dict[str, str]
    version: str
    uptime: float


class DatabaseStats(BaseModel):
    """Database statistics."""
    total_documents: int
    total_chunks: int
    total_vectors: int
    collection_size: int
    last_indexed: Optional[datetime]


class ServiceStats(BaseModel):
    """Service statistics."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    uptime: float


# Error Models
class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    message: str
    code: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: Optional[Dict[str, Any]] = None


class ValidationError(BaseModel):
    """Validation error details."""
    field: str
    message: str
    value: Any


# Configuration Models
class ProcessingConfig(BaseModel):
    """Configuration for document processing."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    embedding_model: str = "text-embedding-3-small"
    max_chunks_per_document: int = 100


class SearchConfig(BaseModel):
    """Configuration for search functionality."""
    default_limit: int = 10
    max_limit: int = 50
    similarity_threshold: float = 0.7
    rerank_results: bool = True


# Batch Processing Models
class BatchProcessingJob(BaseModel):
    """Batch processing job information."""
    job_id: str
    document_ids: List[str]
    status: DocumentStatus
    progress: float  # 0.0 to 1.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class BatchProcessingRequest(BaseModel):
    """Request for batch processing."""
    document_ids: List[str]
    processing_options: ProcessingConfig = ProcessingConfig()
    priority: int = Field(default=0, ge=0, le=10)
