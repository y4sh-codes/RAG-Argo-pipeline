"""
FastAPI routers for document management endpoints.
Handles PDF upload, processing, and document management.
"""

import os
import uuid
import shutil
import asyncio
from typing import List, Optional
import logging
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from ..config import settings
from ..models import (
    DocumentUploadResponse, Document, DocumentStatus, ChunkingRequest,
    BatchProcessingJob, BatchProcessingRequest
)
from ..services.pdf_processor import pdf_processor
from ..services.chunking_service import chunking_service
from ..services.qdrant_service import qdrant_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to upload and process")
):
    """
    Upload and process a PDF document.
    
    This endpoint accepts a PDF file, validates it, stores it, and initiates
    background processing including text extraction and vector indexing.
    """
    try:
        # Validate file
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        
        if file.size > settings.max_file_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum of {settings.max_file_size_mb}MB"
            )
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(settings.pdf_dir, filename)
        
        # Ensure directory exists
        os.makedirs(settings.pdf_dir, exist_ok=True)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Validate the saved PDF
        is_valid, error_message = pdf_processor.validate_pdf(file_path)
        if not is_valid:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=error_message)
        
        # Start background processing
        background_tasks.add_task(process_document_background, file_path, file.filename)
        
        logger.info(f"Document uploaded: {filename}")
        
        return DocumentUploadResponse(
            document_id=file_id,
            filename=file.filename,
            status=DocumentStatus.PENDING,
            message="Document uploaded successfully and processing started"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during upload")


async def process_document_background(file_path: str, original_filename: str):
    """Background task to process uploaded document."""
    try:
        # Process PDF
        document = await pdf_processor.process_pdf(file_path, original_filename)
        
        if document.status == DocumentStatus.FAILED:
            logger.error(f"Failed to process document: {document.error_message}")
            return
        
        # Create semantic chunks
        chunks = await chunking_service.create_semantic_chunks(
            text=document.content,
            document_id=document.document_id,
            metadata=document.metadata.model_dump()
        )
        
        document.chunks = chunks
        
        # Index chunks in vector database
        success = await qdrant_service.index_chunks(chunks)
        
        if success:
            document.status = DocumentStatus.COMPLETED
            logger.info(f"Successfully processed and indexed document: {document.document_id}")
        else:
            document.status = DocumentStatus.FAILED
            document.error_message = "Failed to index chunks in vector database"
            logger.error(f"Failed to index document: {document.document_id}")
    
    except Exception as e:
        logger.error(f"Error in background processing: {str(e)}")


@router.get("/")
async def list_documents():
    """
    List all processed documents with basic information.
    """
    try:
        # For now, return collection info since we don't have a document database
        collection_info = await qdrant_service.get_collection_info()
        
        return {
            "total_documents": collection_info.get("points_count", 0),
            "collection_info": collection_info,
            "message": "Document listing from vector database. Implement proper document storage for detailed listing."
        }
    
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving document list")


@router.get("/{document_id}")
async def get_document(document_id: str):
    """
    Get detailed information about a specific document.
    """
    try:
        # Get document chunks from vector database
        chunks = await qdrant_service.get_document_chunks(document_id)
        
        if not chunks:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "document_id": document_id,
            "total_chunks": len(chunks),
            "chunks": chunks[:10],  # Return first 10 chunks for preview
            "message": f"Found {len(chunks)} chunks for document {document_id}"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving document")


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and all its associated chunks.
    """
    try:
        success = await qdrant_service.delete_document(document_id)
        
        if success:
            return {"message": f"Document {document_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found or could not be deleted")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error deleting document")


@router.post("/batch-process")
async def batch_process_documents(
    background_tasks: BackgroundTasks,
    request: BatchProcessingRequest
):
    """
    Process multiple documents in batch.
    """
    try:
        job_id = str(uuid.uuid4())
        
        # In a real implementation, you'd store this job in a database
        job = BatchProcessingJob(
            job_id=job_id,
            document_ids=request.document_ids,
            status=DocumentStatus.PENDING,
            progress=0.0
        )
        
        # Start background processing
        background_tasks.add_task(batch_process_background, job)
        
        return {
            "job_id": job_id,
            "status": "started",
            "document_count": len(request.document_ids),
            "message": "Batch processing job started"
        }
    
    except Exception as e:
        logger.error(f"Error starting batch processing: {str(e)}")
        raise HTTPException(status_code=500, detail="Error starting batch processing")


async def batch_process_background(job: BatchProcessingJob):
    """Background task for batch processing."""
    try:
        job.status = DocumentStatus.PROCESSING
        
        for i, document_id in enumerate(job.document_ids):
            # Process each document
            # This is a placeholder - implement actual batch processing logic
            await asyncio.sleep(1)  # Simulate processing
            
            job.progress = (i + 1) / len(job.document_ids)
        
        job.status = DocumentStatus.COMPLETED
        logger.info(f"Batch job {job.job_id} completed successfully")
    
    except Exception as e:
        job.status = DocumentStatus.FAILED
        job.error_message = str(e)
        logger.error(f"Batch job {job.job_id} failed: {str(e)}")


@router.post("/{document_id}/rechunk")
async def rechunk_document(
    document_id: str,
    request: ChunkingRequest,
    background_tasks: BackgroundTasks
):
    """
    Re-chunk a document with different parameters.
    """
    try:
        # Check if document exists
        chunks = await qdrant_service.get_document_chunks(document_id)
        if not chunks:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Start background re-chunking
        background_tasks.add_task(rechunk_document_background, document_id, request)
        
        return {
            "message": f"Re-chunking started for document {document_id}",
            "parameters": {
                "chunk_size": request.chunk_size or settings.chunk_size,
                "chunk_overlap": request.chunk_overlap or settings.chunk_overlap,
                "strategy": request.chunking_strategy or "semantic"
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting re-chunking for document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error starting re-chunking process")


async def rechunk_document_background(document_id: str, request: ChunkingRequest):
    """Background task for document re-chunking."""
    try:
        # This would involve:
        # 1. Getting original document content
        # 2. Re-chunking with new parameters
        # 3. Deleting old chunks
        # 4. Indexing new chunks
        
        logger.info(f"Re-chunking document {document_id} with new parameters")
        
        # Placeholder implementation
        await asyncio.sleep(2)  # Simulate processing
        
        logger.info(f"Re-chunking completed for document {document_id}")
    
    except Exception as e:
        logger.error(f"Error in re-chunking background task: {str(e)}")


@router.get("/{document_id}/summary")
async def get_document_summary(document_id: str):
    """
    Get an AI-generated summary of a document.
    """
    try:
        from ..services.groq_service import groq_service
        
        summary = await groq_service.summarize_document(document_id)
        
        return {
            "document_id": document_id,
            "summary": summary,
            "generated_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error generating summary for document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating document summary")
