"""
PDF processing service for extracting text, metadata, and structure from PDF documents.
Supports multiple extraction methods and intelligent text cleaning.
"""

import hashlib
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import logging

import fitz  # PyMuPDF
import pdfplumber
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..config import settings
from ..models import Document, DocumentMetadata, DocumentChunk, DocumentStatus

logger = logging.getLogger(__name__)


class PDFProcessor:
    """PDF processing service with multiple extraction methods."""
    
    def __init__(self):
        self.extraction_methods = {
            "pymupdf": self._extract_with_pymupdf,
            "pdfplumber": self._extract_with_pdfplumber,
            "pypdf2": self._extract_with_pypdf2
        }
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    async def process_pdf(self, file_path: str, filename: str) -> Document:
        """
        Process a PDF file and extract text, metadata, and create document chunks.
        
        Args:
            file_path: Path to the PDF file
            filename: Original filename
            
        Returns:
            Document object with extracted content and metadata
        """
        try:
            # Generate document ID
            document_id = self._generate_document_id(file_path)
            
            # Extract content and metadata
            content, metadata, page_contents = await self._extract_content_and_metadata(file_path)
            
            # Create document object
            document = Document(
                document_id=document_id,
                filename=filename,
                file_path=file_path,
                content=content,
                metadata=metadata,
                status=DocumentStatus.PROCESSING
            )
            
            # Create chunks
            chunks = await self._create_chunks(document, page_contents)
            document.chunks = chunks
            
            # Update status
            document.status = DocumentStatus.COMPLETED
            document.processed_at = datetime.now(timezone.utc)
            
            logger.info(f"Successfully processed PDF: {filename}, chunks: {len(chunks)}")
            return document
            
        except Exception as e:
            logger.error(f"Error processing PDF {filename}: {str(e)}")
            return Document(
                document_id=self._generate_document_id(file_path),
                filename=filename,
                file_path=file_path,
                content="",
                metadata=DocumentMetadata(),
                status=DocumentStatus.FAILED,
                error_message=str(e)
            )
    
    async def _extract_content_and_metadata(self, file_path: str) -> Tuple[str, DocumentMetadata, List[Dict]]:
        """Extract text content and metadata from PDF."""
        method = self.extraction_methods.get(settings.extraction_method, self._extract_with_pymupdf)
        return await method(file_path)
    
    async def _extract_with_pymupdf(self, file_path: str) -> Tuple[str, DocumentMetadata, List[Dict]]:
        """Extract content using PyMuPDF (fitz)."""
        doc = fitz.open(file_path)
        full_content = []
        page_contents = []
        
        try:
            # Extract metadata
            metadata_dict = doc.metadata
            metadata = DocumentMetadata(
                title=self._clean_text(metadata_dict.get("title", "")),
                authors=self._extract_authors(metadata_dict.get("author", "")),
                page_count=doc.page_count,
                file_size=os.path.getsize(file_path)
            )
            
            # Extract text from each page
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                cleaned_text = self._clean_text(text)
                
                if cleaned_text.strip():
                    full_content.append(cleaned_text)
                    page_contents.append({
                        "page_number": page_num + 1,
                        "text": cleaned_text
                    })
            
            # Try to extract additional metadata from first few pages
            first_pages_text = " ".join(full_content[:3])
            metadata = self._enhance_metadata_from_text(metadata, first_pages_text)
            
        finally:
            doc.close()
        
        return "\n\n".join(full_content), metadata, page_contents
    
    async def _extract_with_pdfplumber(self, file_path: str) -> Tuple[str, DocumentMetadata, List[Dict]]:
        """Extract content using pdfplumber."""
        full_content = []
        page_contents = []
        
        with pdfplumber.open(file_path) as pdf:
            # Basic metadata
            metadata = DocumentMetadata(
                page_count=len(pdf.pages),
                file_size=os.path.getsize(file_path)
            )
            
            # Extract text from each page
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    cleaned_text = self._clean_text(text)
                    if cleaned_text.strip():
                        full_content.append(cleaned_text)
                        page_contents.append({
                            "page_number": i + 1,
                            "text": cleaned_text
                        })
            
            # Enhance metadata from content
            first_pages_text = " ".join(full_content[:3])
            metadata = self._enhance_metadata_from_text(metadata, first_pages_text)
        
        return "\n\n".join(full_content), metadata, page_contents
    
    async def _extract_with_pypdf2(self, file_path: str) -> Tuple[str, DocumentMetadata, List[Dict]]:
        """Extract content using PyPDF2."""
        full_content = []
        page_contents = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Basic metadata
            metadata = DocumentMetadata(
                page_count=len(pdf_reader.pages),
                file_size=os.path.getsize(file_path)
            )
            
            # Extract metadata if available
            if pdf_reader.metadata:
                metadata.title = self._clean_text(str(pdf_reader.metadata.get('/Title', '')))
                metadata.authors = self._extract_authors(str(pdf_reader.metadata.get('/Author', '')))
            
            # Extract text from each page
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    cleaned_text = self._clean_text(text)
                    if cleaned_text.strip():
                        full_content.append(cleaned_text)
                        page_contents.append({
                            "page_number": i + 1,
                            "text": cleaned_text
                        })
            
            # Enhance metadata from content
            first_pages_text = " ".join(full_content[:3])
            metadata = self._enhance_metadata_from_text(metadata, first_pages_text)
        
        return "\n\n".join(full_content), metadata, page_contents
    
    async def _create_chunks(self, document: Document, page_contents: List[Dict]) -> List[DocumentChunk]:
        """Create text chunks from document content."""
        chunks = []
        chunk_index = 0
        
        # Process each page
        for page_info in page_contents:
            page_text = page_info["text"]
            page_number = page_info["page_number"]
            
            # Skip if page is too short
            if len(page_text) < settings.min_chunk_size:
                continue
            
            # Split page into chunks
            page_chunks = self.text_splitter.split_text(page_text)
            
            for chunk_text in page_chunks:
                if len(chunk_text.strip()) >= settings.min_chunk_size:
                    chunk = DocumentChunk(
                        chunk_id=f"{document.document_id}_{chunk_index}",
                        document_id=document.document_id,
                        text=chunk_text.strip(),
                        page_number=page_number,
                        chunk_index=chunk_index,
                        metadata={
                            "title": document.metadata.title,
                            "authors": document.metadata.authors,
                            "page_number": page_number,
                            "chunk_length": len(chunk_text)
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    
                    # Respect max chunks limit
                    if len(chunks) >= settings.max_chunks_per_document:
                        logger.warning(f"Reached max chunks limit for document {document.document_id}")
                        break
            
            if len(chunks) >= settings.max_chunks_per_document:
                break
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
        text = re.sub(r'\f', ' ', text)  # Remove form feeds
        text = re.sub(r'(?:\r\n|\r|\n){3,}', '\n\n', text)  # Normalize line breaks
        
        # Remove headers/footers patterns (common artifacts)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def _extract_authors(self, author_str: str) -> List[str]:
        """Extract and clean author names from metadata."""
        if not author_str or author_str == "None":
            return []
        
        # Split by common delimiters
        authors = re.split(r'[,;&]', author_str)
        
        # Clean and filter authors
        cleaned_authors = []
        for author in authors:
            author = author.strip()
            if author and len(author) > 2:  # Filter out initials only
                cleaned_authors.append(author)
        
        return cleaned_authors[:10]  # Limit to 10 authors
    
    def _enhance_metadata_from_text(self, metadata: DocumentMetadata, text: str) -> DocumentMetadata:
        """Enhance metadata by extracting information from document text."""
        if not text:
            return metadata
        
        # Extract title if not already present
        if not metadata.title:
            title_match = re.search(r'^(.+?)(?:\n|Abstract|ABSTRACT)', text[:1000], re.IGNORECASE)
            if title_match:
                potential_title = self._clean_text(title_match.group(1))
                if 10 <= len(potential_title) <= 200:  # Reasonable title length
                    metadata.title = potential_title
        
        # Extract DOI
        doi_match = re.search(r'doi[:\s]*(10\.\d+/[^\s]+)', text, re.IGNORECASE)
        if doi_match:
            metadata.doi = doi_match.group(1)
        
        # Extract keywords
        keywords_match = re.search(r'(?:keywords?|key\s*words?)[:\s]*(.+?)(?:\n\n|\n[A-Z]|\d+\.)', 
                                  text, re.IGNORECASE | re.DOTALL)
        if keywords_match:
            keywords_text = keywords_match.group(1)
            keywords = [kw.strip() for kw in re.split(r'[,;]', keywords_text) 
                       if kw.strip() and len(kw.strip()) > 2]
            metadata.keywords = keywords[:20]  # Limit to 20 keywords
        
        # Extract abstract
        abstract_match = re.search(r'(?:abstract|summary)[:\s]*(.+?)(?:\n\n|introduction|keywords)', 
                                  text[:3000], re.IGNORECASE | re.DOTALL)
        if abstract_match:
            abstract_text = self._clean_text(abstract_match.group(1))
            if 50 <= len(abstract_text) <= 2000:  # Reasonable abstract length
                metadata.abstract = abstract_text
        
        # Try to extract year from text
        year_match = re.search(r'(?:19|20)\d{2}', text[:2000])
        if year_match:
            try:
                year = int(year_match.group())
                if 1950 <= year <= datetime.now().year:
                    metadata.publication_date = datetime(year, 1, 1)
            except ValueError:
                pass
        
        return metadata
    
    def _generate_document_id(self, file_path: str) -> str:
        """Generate a unique document ID based on file content."""
        # Use file path and size for ID generation
        file_stat = os.stat(file_path)
        content = f"{file_path}_{file_stat.st_size}_{file_stat.st_mtime}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def extract_text_only(self, file_path: str) -> str:
        """Quick text extraction without full processing."""
        try:
            method = self.extraction_methods.get(settings.extraction_method, self._extract_with_pymupdf)
            content, _, _ = await method(file_path)
            return content
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            return ""
    
    def validate_pdf(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate PDF file.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not os.path.exists(file_path):
                return False, "File does not exist"
            
            # Check file size
            file_size = os.path.getsize(file_path)
            max_size = settings.max_file_size_mb * 1024 * 1024
            
            if file_size == 0:
                return False, "File is empty"
            
            if file_size > max_size:
                return False, f"File size exceeds limit of {settings.max_file_size_mb}MB"
            
            # Check if file can be opened
            try:
                doc = fitz.open(file_path)
                if doc.page_count == 0:
                    doc.close()
                    return False, "PDF has no pages"
                doc.close()
            except Exception as e:
                return False, f"Cannot open PDF: {str(e)}"
            
            return True, ""
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"


# Create global instance
pdf_processor = PDFProcessor()
