"""
Qdrant vector database service for storing and searching document embeddings.
Handles vector storage, similarity search, and metadata filtering.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import json

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, CreateCollection, PointStruct,
    Filter, FieldCondition, Match, SearchRequest
)

from ..config import settings
from ..models import DocumentChunk, SearchResult, SearchResponse

logger = logging.getLogger(__name__)


class QdrantService:
    """Qdrant vector database service for document embeddings."""
    
    def __init__(self):
        self.client = None
        self.collection_name = settings.qdrant_collection_name
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Qdrant client."""
        try:
            # Parse URL to extract host and port
            if settings.qdrant_url.startswith("http://") or settings.qdrant_url.startswith("https://"):
                import urllib.parse
                parsed = urllib.parse.urlparse(settings.qdrant_url)
                host = parsed.hostname
                port = parsed.port or 6333
                prefer_grpc = False
            else:
                host = settings.qdrant_url
                port = 6333
                prefer_grpc = True
            
            self.client = QdrantClient(
                host=host,
                port=port,
                api_key=settings.qdrant_api_key,
                prefer_grpc=prefer_grpc,
                timeout=30.0
            )
            
            logger.info(f"Qdrant client initialized: {settings.qdrant_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            self.client = None
    
    async def ensure_collection_exists(self) -> bool:
        """Ensure the collection exists, create if necessary."""
        try:
            if not self.client:
                raise Exception("Qdrant client not initialized")
            
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name in collection_names:
                logger.info(f"Collection '{self.collection_name}' already exists")
                return True
            
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.vector_dimension,
                    distance=Distance.COSINE
                )
            )
            
            logger.info(f"Created collection '{self.collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            return False
    
    async def index_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """
        Index document chunks with their embeddings.
        
        Args:
            chunks: List of document chunks with embeddings
            
        Returns:
            Success status
        """
        try:
            if not self.client:
                raise Exception("Qdrant client not initialized")
            
            if not await self.ensure_collection_exists():
                raise Exception("Failed to ensure collection exists")
            
            # Prepare points for indexing
            points = []
            for chunk in chunks:
                if not chunk.embedding:
                    logger.warning(f"Chunk {chunk.chunk_id} has no embedding, skipping")
                    continue
                
                # Prepare metadata for storage
                payload = {
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "chunk_length": len(chunk.text)
                }
                
                # Add document metadata
                if chunk.metadata:
                    for key, value in chunk.metadata.items():
                        # Convert lists to strings for Qdrant compatibility
                        if isinstance(value, list):
                            payload[f"metadata_{key}"] = json.dumps(value)
                        elif value is not None:
                            payload[f"metadata_{key}"] = str(value)
                
                point = PointStruct(
                    id=hash(chunk.chunk_id) & 0x7FFFFFFFFFFFFFFF,  # Ensure positive int
                    vector=chunk.embedding,
                    payload=payload
                )
                points.append(point)
            
            if not points:
                logger.warning("No points to index")
                return True
            
            # Index points in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                logger.info(f"Indexed batch {i//batch_size + 1}, points: {len(batch)}")
            
            logger.info(f"Successfully indexed {len(points)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing chunks: {str(e)}")
            return False
    
    async def search_similar(
        self, 
        query_embedding: List[float], 
        limit: int = 10,
        similarity_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query_embedding: Query vector embedding
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            filters: Optional metadata filters
            
        Returns:
            List of search results
        """
        try:
            if not self.client:
                raise Exception("Qdrant client not initialized")
            
            # Build filter conditions
            filter_conditions = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    if isinstance(value, list):
                        # Handle list values (e.g., multiple authors)
                        for v in value:
                            conditions.append(
                                FieldCondition(
                                    key=f"metadata_{key}",
                                    match=Match(value=str(v))
                                )
                            )
                    else:
                        conditions.append(
                            FieldCondition(
                                key=f"metadata_{key}" if key.startswith("metadata_") else key,
                                match=Match(value=str(value))
                            )
                        )
                
                if conditions:
                    filter_conditions = Filter(must=conditions)
            
            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=similarity_threshold,
                query_filter=filter_conditions
            )
            
            # Convert to SearchResult objects
            results = []
            for result in search_results:
                payload = result.payload
                
                # Parse authors from metadata
                authors = []
                if "metadata_authors" in payload:
                    try:
                        authors = json.loads(payload["metadata_authors"])
                    except (json.JSONDecodeError, TypeError):
                        authors = [str(payload["metadata_authors"])]
                
                # Parse publication date
                publication_date = None
                if "metadata_publication_date" in payload:
                    try:
                        publication_date = datetime.fromisoformat(
                            payload["metadata_publication_date"].replace("Z", "+00:00")
                        )
                    except (ValueError, TypeError):
                        pass
                
                search_result = SearchResult(
                    document_id=payload.get("document_id", ""),
                    chunk_id=payload.get("chunk_id", ""),
                    title=payload.get("metadata_title", "Untitled"),
                    text=payload.get("text", ""),
                    similarity_score=result.score,
                    page_number=payload.get("page_number"),
                    authors=authors,
                    publication_date=publication_date
                )
                results.append(search_result)
            
            logger.info(f"Found {len(results)} similar chunks")
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {str(e)}")
            return []
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Success status
        """
        try:
            if not self.client:
                raise Exception("Qdrant client not initialized")
            
            # Delete points by document_id filter
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="document_id",
                                match=Match(value=document_id)
                            )
                        ]
                    )
                )
            )
            
            logger.info(f"Deleted document {document_id} from vector database")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information and statistics."""
        try:
            if not self.client:
                raise Exception("Qdrant client not initialized")
            
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                "name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "disk_data_size": collection_info.disk_data_size,
                "ram_data_size": collection_info.ram_data_size,
                "config": {
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": collection_info.config.params.vectors.distance.value
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {}
    
    async def check_connection(self) -> bool:
        """Check if Qdrant connection is healthy."""
        try:
            if not self.client:
                return False
            
            # Try to get collections list
            collections = self.client.get_collections()
            return True
            
        except Exception as e:
            logger.error(f"Qdrant connection check failed: {str(e)}")
            return False
    
    async def create_backup(self, backup_path: str) -> bool:
        """Create a backup snapshot of the collection."""
        try:
            if not self.client:
                raise Exception("Qdrant client not initialized")
            
            # Create snapshot
            snapshot_name = f"{self.collection_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.client.create_snapshot(collection_name=self.collection_name)
            
            logger.info(f"Created backup snapshot: {snapshot_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating backup: {str(e)}")
            return False
    
    async def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        try:
            if not self.client:
                raise Exception("Qdrant client not initialized")
            
            # Search for all chunks of the document
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=Match(value=document_id)
                        )
                    ]
                ),
                limit=10000  # Large limit to get all chunks
            )
            
            chunks = []
            for point in results[0]:  # results is (points, next_page_offset)
                payload = point.payload
                chunks.append({
                    "chunk_id": payload.get("chunk_id"),
                    "text": payload.get("text"),
                    "chunk_index": payload.get("chunk_index"),
                    "page_number": payload.get("page_number"),
                    "created_at": payload.get("created_at")
                })
            
            # Sort by chunk index
            chunks.sort(key=lambda x: x.get("chunk_index", 0))
            
            logger.info(f"Retrieved {len(chunks)} chunks for document {document_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting document chunks: {str(e)}")
            return []
    
    async def update_chunk_metadata(
        self, 
        chunk_id: str, 
        metadata: Dict[str, Any]
    ) -> bool:
        """Update metadata for a specific chunk."""
        try:
            if not self.client:
                raise Exception("Qdrant client not initialized")
            
            # Find the point
            search_results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="chunk_id",
                            match=Match(value=chunk_id)
                        )
                    ]
                ),
                limit=1
            )
            
            if not search_results[0]:
                logger.warning(f"Chunk {chunk_id} not found for metadata update")
                return False
            
            point = search_results[0][0]
            
            # Update payload
            updated_payload = point.payload.copy()
            for key, value in metadata.items():
                if isinstance(value, list):
                    updated_payload[f"metadata_{key}"] = json.dumps(value)
                else:
                    updated_payload[f"metadata_{key}"] = str(value)
            
            # Update point
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point.id,
                        vector=point.vector,
                        payload=updated_payload
                    )
                ]
            )
            
            logger.info(f"Updated metadata for chunk {chunk_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating chunk metadata: {str(e)}")
            return False


# Create global instance
qdrant_service = QdrantService()
