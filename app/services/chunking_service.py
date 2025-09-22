"""
Intelligent chunking servic        try:
            # Initialize OpenAI
            if settings.openai_api_key:
                openai.api_key = settings.openai_api_key
                self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
                logger.info("OpenAI client initialized")
            
            # Initialize SentenceTransformer for semantic chunkings.
Provides semantic-aware text chunking and embedding generation.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple
import json
import re

import openai
from sentence_transformers import SentenceTransformer

from ..config import settings
from ..models import DocumentChunk, EmbeddingProvider

logger = logging.getLogger(__name__)


class ChunkingService:
    """Intelligent document chunking service with AI-powered optimization."""
    
    def __init__(self):
        self.openai_client = None
        self.sentence_transformer = None
        
        # Initialize based on configuration
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI models based on configuration."""
        try:
            # Initialize OpenAI
            if settings.openai_api_key and (
                settings.chunking_model == "openai" or 
                settings.embedding_model == "openai"
            ):
                openai.api_key = settings.openai_api_key
                self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
                logger.info("OpenAI client initialized")
            
            # Initialize sentence transformer for fallback embeddings
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer initialized")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
    
    async def create_semantic_chunks(
        self, 
        text: str, 
        document_id: str,
        metadata: Dict[str, Any] = None
    ) -> List[DocumentChunk]:
        """
        Create semantically coherent chunks using AI models.
        
        Args:
            text: Full document text
            document_id: Unique document identifier
            metadata: Document metadata
            
        Returns:
            List of semantic chunks
        """
        try:
            # First, create logical sections
            sections = await self._identify_sections(text)
            
            # Then create chunks from sections
            chunks = []
            chunk_index = 0
            
            for section in sections:
                section_chunks = await self._chunk_section(
                    section, 
                    document_id, 
                    chunk_index,
                    metadata or {}
                )
                chunks.extend(section_chunks)
                chunk_index += len(section_chunks)
            
            # Post-process chunks
            chunks = await self._optimize_chunks(chunks)
            
            logger.info(f"Created {len(chunks)} semantic chunks for document {document_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in semantic chunking: {str(e)}")
            # Fallback to simple chunking
            return await self._create_simple_chunks(text, document_id, metadata or {})
    
    async def _identify_sections(self, text: str) -> List[Dict[str, Any]]:
        """Identify logical sections in the document."""
        try:
            # Use AI to identify section boundaries
            if settings.chunking_model == "openai" and self.openai_client:
                return await self._identify_sections_openai(text)
            else:
                # Fallback to rule-based section identification
                return self._identify_sections_heuristic(text)
        except Exception as e:
            logger.error(f"Error identifying sections: {str(e)}")
            return self._identify_sections_heuristic(text)
    
    async def _identify_sections_openai(self, text: str) -> List[Dict[str, Any]]:
        """Use OpenAI to identify document sections."""
        try:
            # Truncate text if too long
            max_length = 8000  # Leave room for prompt
            if len(text) > max_length:
                text = text[:max_length] + "..."
            
            prompt = f"""
            Analyze the following academic paper text and identify logical sections.
            Return a JSON array of sections with their titles and approximate start positions.
            
            Each section should have:
            - title: The section heading
            - start_position: Approximate character position in the text
            - section_type: One of [abstract, introduction, methodology, results, discussion, conclusion, references, other]
            
            Text:
            {text}
            
            Return only the JSON array, no other text.
            """
            
            response = await self.openai_client.chat.completions.acreate(
                model=settings.openai_chunking_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            sections_data = json.loads(content)
            
            # Convert to internal format
            sections = []
            for i, section in enumerate(sections_data):
                start_pos = section.get("start_position", 0)
                end_pos = sections_data[i + 1].get("start_position", len(text)) if i + 1 < len(sections_data) else len(text)
                
                sections.append({
                    "title": section.get("title", f"Section {i + 1}"),
                    "text": text[start_pos:end_pos],
                    "section_type": section.get("section_type", "other"),
                    "start_position": start_pos
                })
            
            return sections
            
        except Exception as e:
            logger.error(f"OpenAI section identification failed: {str(e)}")
            return self._identify_sections_heuristic(text)
    
    def _identify_sections_heuristic(self, text: str) -> List[Dict[str, Any]]:
        """Fallback heuristic-based section identification."""
        sections = []
        
        # Common section patterns in academic papers
        section_patterns = [
            (r'\b(?:abstract|summary)\b', 'abstract'),
            (r'\b(?:introduction|intro)\b', 'introduction'), 
            (r'\b(?:method|methodology|methods|approach)\b', 'methodology'),
            (r'\b(?:result|results|finding|findings)\b', 'results'),
            (r'\b(?:discussion|analysis)\b', 'discussion'),
            (r'\b(?:conclusion|conclusions|summary)\b', 'conclusion'),
            (r'\b(?:reference|references|bibliography)\b', 'references'),
            (r'\b(?:appendix|appendices)\b', 'appendix')
        ]
        
        # Find potential section headers
        potential_sections = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if len(line) > 0 and len(line) < 100:  # Reasonable header length
                for pattern, section_type in section_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        char_position = sum(len(lines[j]) + 1 for j in range(i))
                        potential_sections.append({
                            "title": line,
                            "section_type": section_type,
                            "start_position": char_position,
                            "line_number": i
                        })
                        break
        
        # If no sections found, create arbitrary sections
        if not potential_sections:
            section_length = len(text) // 4
            for i in range(4):
                start_pos = i * section_length
                end_pos = min((i + 1) * section_length, len(text))
                sections.append({
                    "title": f"Section {i + 1}",
                    "text": text[start_pos:end_pos],
                    "section_type": "other",
                    "start_position": start_pos
                })
        else:
            # Create sections from identified headers
            for i, section in enumerate(potential_sections):
                start_pos = section["start_position"]
                end_pos = potential_sections[i + 1]["start_position"] if i + 1 < len(potential_sections) else len(text)
                
                sections.append({
                    "title": section["title"],
                    "text": text[start_pos:end_pos],
                    "section_type": section["section_type"],
                    "start_position": start_pos
                })
        
        return sections
    
    async def _chunk_section(
        self, 
        section: Dict[str, Any], 
        document_id: str, 
        start_index: int,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Create optimized chunks from a document section."""
        chunks = []
        text = section["text"].strip()
        
        if len(text) < settings.min_chunk_size:
            return chunks
        
        # If section is small enough, use as single chunk
        if len(text) <= settings.chunk_size * 1.2:  # Allow slight overflow
            chunk = DocumentChunk(
                chunk_id=f"{document_id}_{start_index}",
                document_id=document_id,
                text=text,
                chunk_index=start_index,
                metadata={
                    **metadata,
                    "section_title": section["title"],
                    "section_type": section["section_type"],
                    "chunk_length": len(text)
                }
            )
            chunks.append(chunk)
            return chunks
        
        # Use AI-guided chunking for larger sections
        if settings.chunking_model == "openai" and self.openai_client:
            section_chunks = await self._chunk_with_ai(text, section["title"])
        else:
            # Fallback to semantic splitting
            section_chunks = self._chunk_by_sentences(text)
        
        # Create chunk objects
        for i, chunk_text in enumerate(section_chunks):
            if len(chunk_text.strip()) >= settings.min_chunk_size:
                chunk = DocumentChunk(
                    chunk_id=f"{document_id}_{start_index + i}",
                    document_id=document_id,
                    text=chunk_text.strip(),
                    chunk_index=start_index + i,
                    metadata={
                        **metadata,
                        "section_title": section["title"],
                        "section_type": section["section_type"],
                        "chunk_length": len(chunk_text)
                    }
                )
                chunks.append(chunk)
        
        return chunks
    
    async def _chunk_with_ai(self, text: str, section_title: str) -> List[str]:
        """Use AI to create semantically coherent chunks."""
        try:
            prompt = f"""
            Break the following text from the "{section_title}" section into semantically coherent chunks.
            Each chunk should be {settings.chunk_size} characters or less.
            Maintain context and meaning within each chunk.
            
            Return the chunks separated by "---CHUNK---".
            
            Text:
            {text}
            """
            
            response = await self.openai_client.chat.completions.acreate(
                model=settings.openai_chunking_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            chunks = [chunk.strip() for chunk in content.split("---CHUNK---") if chunk.strip()]
            
            return chunks if chunks else self._chunk_by_sentences(text)
            
        except Exception as e:
            logger.error(f"AI chunking failed: {str(e)}")
            return self._chunk_by_sentences(text)
    
    def _chunk_by_sentences(self, text: str) -> List[str]:
        """Chunk text by sentence boundaries."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + " " + sentence) <= settings.chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def _optimize_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Post-process and optimize chunks."""
        optimized_chunks = []
        
        for chunk in chunks:
            # Skip very short chunks
            if len(chunk.text) < settings.min_chunk_size:
                continue
            
            # Clean up text
            chunk.text = self._clean_chunk_text(chunk.text)
            
            # Generate embeddings if needed
            if settings.embedding_model == "openai":
                try:
                    chunk.embedding = await self.generate_embedding(chunk.text)
                except Exception as e:
                    logger.error(f"Failed to generate embedding for chunk {chunk.chunk_id}: {str(e)}")
            
            optimized_chunks.append(chunk)
        
        return optimized_chunks
    
    def _clean_chunk_text(self, text: str) -> str:
        """Clean and normalize chunk text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove incomplete sentences at start/end
        sentences = text.split('.')
        if len(sentences) > 1:
            # Check if first sentence seems incomplete
            if len(sentences[0]) < 20 or not sentences[0][0].isupper():
                sentences = sentences[1:]
            
            # Check if last sentence seems incomplete
            if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
                sentences = sentences[:-1]
            
            text = '.'.join(sentences) + ('.' if sentences else '')
        
        return text.strip()
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using configured provider."""
        try:
            if settings.embedding_model == "openai" and self.openai_client:
                return await self._generate_openai_embedding(text)
            else:
                # Fallback to sentence transformer
                return self.sentence_transformer.encode(text).tolist()
        
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Fallback to sentence transformer
            return self.sentence_transformer.encode(text).tolist()
    
    async def _generate_openai_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI."""
        response = await self.openai_client.embeddings.acreate(
            model=settings.openai_embedding_model,
            input=text[:8000]  # Limit input length
        )
        return response.data[0].embedding
    
    async def _create_simple_chunks(
        self, 
        text: str, 
        document_id: str, 
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Fallback simple chunking method."""
        chunks = []
        chunk_size = settings.chunk_size
        overlap = settings.chunk_overlap
        
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + chunk_size
            if end > len(text):
                end = len(text)
            
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) >= settings.min_chunk_size:
                chunk = DocumentChunk(
                    chunk_id=f"{document_id}_{chunk_index}",
                    document_id=document_id,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    metadata={
                        **metadata,
                        "chunk_length": len(chunk_text),
                        "chunking_method": "simple"
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
            
            start = end - overlap
        
        return chunks


# Create global instance
chunking_service = ChunkingService()
