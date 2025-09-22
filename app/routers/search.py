"""
FastAPI routers for search and query endpoints.
Handles semantic search and RAG query processing.
"""

import time
from typing import List, Dict, Any, Optional
import logging

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse

from ..models import (
    SearchRequest, SearchResponse, QueryRequest, QueryResponse,
    EmbeddingRequest
)
from ..services.qdrant_service import qdrant_service
from ..services.chunking_service import chunking_service
from ..services.groq_service import groq_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/semantic", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """
    Perform semantic search across indexed documents.
    
    This endpoint generates embeddings for the query and searches
    for similar chunks in the vector database.
    """
    try:
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = await chunking_service.generate_embedding(request.query)
        
        # Search for similar chunks
        search_results = await qdrant_service.search_similar(
            query_embedding=query_embedding,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold,
            filters=request.filters
        )
        
        processing_time = time.time() - start_time
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Process a natural language query using RAG (Retrieval-Augmented Generation).
    
    This endpoint performs semantic search and generates a comprehensive
    response using the Groq language model with proper citations.
    """
    try:
        response = await groq_service.process_query(request)
        return response
    
    except Exception as e:
        logger.error(f"Error in query processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.get("/suggestions")
async def get_search_suggestions(
    q: str = Query(..., description="Partial query to get suggestions for"),
    limit: int = Query(5, description="Number of suggestions to return")
):
    """
    Get search suggestions based on indexed content.
    
    This endpoint provides autocomplete suggestions based on
    document content and common research topics.
    """
    try:
        # For now, return common Argo-related search suggestions
        # In a real implementation, this could use indexed terms
        
        suggestions = [
            "Argo float temperature measurements",
            "Ocean salinity profiles",
            "Deep water formation processes",
            "Meridional overturning circulation",
            "Mixed layer depth variability",
            "Thermohaline circulation patterns",
            "Ocean heat content changes",
            "Water mass characteristics",
            "Seasonal temperature variations",
            "Global ocean monitoring"
        ]
        
        # Filter suggestions based on query
        query_lower = q.lower()
        filtered_suggestions = [
            s for s in suggestions 
            if query_lower in s.lower()
        ][:limit]
        
        if not filtered_suggestions:
            # Return general suggestions if no matches
            filtered_suggestions = suggestions[:limit]
        
        return {
            "query": q,
            "suggestions": filtered_suggestions
        }
    
    except Exception as e:
        logger.error(f"Error getting search suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating suggestions")


@router.post("/embedding")
async def generate_embedding(request: EmbeddingRequest):
    """
    Generate embeddings for text using the configured embedding model.
    
    This endpoint is useful for testing and external integrations
    that need to generate compatible embeddings.
    """
    try:
        embedding = await chunking_service.generate_embedding(request.text)
        
        return {
            "text": request.text,
            "embedding": embedding,
            "dimension": len(embedding),
            "provider": request.provider
        }
    
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


@router.get("/similar/{document_id}")
async def find_similar_documents(
    document_id: str,
    limit: int = Query(10, description="Number of similar documents to return"),
    threshold: float = Query(0.7, description="Similarity threshold")
):
    """
    Find documents similar to a specific document.
    
    This endpoint uses the first chunk of the specified document
    to find other similar documents in the collection.
    """
    try:
        # Get chunks for the source document
        source_chunks = await qdrant_service.get_document_chunks(document_id)
        
        if not source_chunks:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Use the first chunk's text to find similar documents
        first_chunk_text = source_chunks[0].get("text", "")
        
        if not first_chunk_text:
            raise HTTPException(status_code=400, detail="Document has no content")
        
        # Generate embedding for the first chunk
        query_embedding = await chunking_service.generate_embedding(first_chunk_text)
        
        # Search for similar chunks
        search_results = await qdrant_service.search_similar(
            query_embedding=query_embedding,
            limit=limit * 2,  # Get more results to filter out same document
            similarity_threshold=threshold
        )
        
        # Filter out chunks from the same document
        filtered_results = [
            result for result in search_results 
            if result.document_id != document_id
        ][:limit]
        
        # Group by document_id to get unique documents
        unique_documents = {}
        for result in filtered_results:
            if result.document_id not in unique_documents:
                unique_documents[result.document_id] = result
        
        similar_docs = list(unique_documents.values())
        
        return {
            "source_document_id": document_id,
            "similar_documents": similar_docs,
            "total_found": len(similar_docs)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Error finding similar documents")


@router.get("/topics")
async def get_trending_topics():
    """
    Get trending topics based on indexed documents.
    
    This endpoint analyzes the content to identify common topics
    and research areas in the document collection.
    """
    try:
        # For now, return predefined Argo research topics
        # In a real implementation, this could analyze document metadata
        # and extract trending topics using NLP techniques
        
        topics = [
            {
                "topic": "Ocean Temperature Monitoring",
                "documents": 45,
                "relevance": 0.95
            },
            {
                "topic": "Salinity Measurements",
                "documents": 38,
                "relevance": 0.92
            },
            {
                "topic": "Deep Water Formation",
                "documents": 29,
                "relevance": 0.88
            },
            {
                "topic": "Climate Change Impact",
                "documents": 33,
                "relevance": 0.87
            },
            {
                "topic": "Ocean Circulation Patterns",
                "documents": 26,
                "relevance": 0.85
            },
            {
                "topic": "Mixed Layer Dynamics",
                "documents": 22,
                "relevance": 0.83
            },
            {
                "topic": "Data Quality Assessment",
                "documents": 19,
                "relevance": 0.80
            },
            {
                "topic": "Seasonal Variability",
                "documents": 17,
                "relevance": 0.78
            }
        ]
        
        return {
            "topics": topics,
            "total_topics": len(topics),
            "last_updated": "2024-01-01T00:00:00Z"
        }
    
    except Exception as e:
        logger.error(f"Error getting trending topics: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving trending topics")


@router.post("/batch-query")
async def batch_query(
    queries: List[str],
    response_format: str = "brief",
    include_citations: bool = True
):
    """
    Process multiple queries in batch.
    
    This endpoint allows processing multiple queries efficiently
    and returns responses for all queries.
    """
    try:
        if len(queries) > 10:
            raise HTTPException(
                status_code=400, 
                detail="Maximum 10 queries allowed per batch"
            )
        
        results = []
        
        for i, query in enumerate(queries):
            try:
                request = QueryRequest(
                    query=query,
                    response_format=response_format,
                    include_citations=include_citations
                )
                response = await groq_service.process_query(request)
                results.append({
                    "query_index": i,
                    "query": query,
                    "response": response.model_dump()
                })
            except Exception as e:
                logger.error(f"Error processing query {i}: {str(e)}")
                results.append({
                    "query_index": i,
                    "query": query,
                    "error": str(e)
                })
        
        return {
            "batch_results": results,
            "total_queries": len(queries),
            "successful_queries": len([r for r in results if "error" not in r])
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch query processing: {str(e)}")
        raise HTTPException(status_code=500, detail="Batch query processing failed")


@router.get("/analytics")
async def get_search_analytics():
    """
    Get search and usage analytics.
    
    This endpoint provides insights into search patterns,
    popular queries, and system usage statistics.
    """
    try:
        # In a real implementation, this would query analytics data
        # For now, return mock analytics data
        
        collection_info = await qdrant_service.get_collection_info()
        
        analytics = {
            "database_stats": {
                "total_documents": collection_info.get("points_count", 0),
                "total_vectors": collection_info.get("vectors_count", 0),
                "collection_size_mb": collection_info.get("disk_data_size", 0) / (1024 * 1024)
            },
            "popular_queries": [
                {"query": "Argo float temperature", "count": 45},
                {"query": "Ocean salinity measurements", "count": 38},
                {"query": "Deep water formation", "count": 29}
            ],
            "search_patterns": {
                "avg_query_length": 8.5,
                "avg_results_returned": 12.3,
                "avg_response_time_ms": 1250
            },
            "usage_trends": {
                "daily_queries": 156,
                "weekly_growth": 0.15,
                "most_active_hour": 14
            }
        }
        
        return analytics
    
    except Exception as e:
        logger.error(f"Error getting analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving analytics")
