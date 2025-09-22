"""
Groq-powered RAG query service for generating responses with citations.
Handles query processing, context retrieval, and response generation.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re

import groq
from groq import Groq

from ..config import settings
from ..models import QueryRequest, QueryResponse, Citation, SearchResult
from .qdrant_service import qdrant_service
from .chunking_service import chunking_service

logger = logging.getLogger(__name__)


class GroqService:
    """Groq-powered RAG service for query processing and response generation."""
    
    def __init__(self):
        self.client = None
        self._initialize_client()
        
        # Response templates
        self.response_templates = {
            "brief": self._brief_response_prompt,
            "detailed": self._detailed_response_prompt,
            "comprehensive": self._comprehensive_response_prompt
        }
    
    def _initialize_client(self):
        """Initialize Groq client."""
        try:
            if not settings.groq_api_key:
                raise Exception("GROQ_API_KEY not configured")
            
            self.client = Groq(api_key=settings.groq_api_key)
            logger.info("Groq client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {str(e)}")
            self.client = None
    
    async def process_query(self, query_request: QueryRequest) -> QueryResponse:
        """
        Process a user query and generate a comprehensive response with citations.
        
        Args:
            query_request: Query request with parameters
            
        Returns:
            QueryResponse with answer and citations
        """
        start_time = time.time()
        
        try:
            if not self.client:
                raise Exception("Groq client not initialized")
            
            # Step 1: Generate query embedding for similarity search
            query_embedding = await chunking_service.generate_embedding(query_request.query)
            
            # Step 2: Search for relevant chunks
            search_results = await qdrant_service.search_similar(
                query_embedding=query_embedding,
                limit=query_request.search_limit,
                similarity_threshold=settings.similarity_threshold
            )
            
            if not search_results:
                return QueryResponse(
                    query=query_request.query,
                    response="I couldn't find any relevant information in the knowledge base to answer your question.",
                    citations=[],
                    search_results_count=0,
                    processing_time=time.time() - start_time
                )
            
            # Step 3: Generate response using Groq
            response_text = await self._generate_response(
                query=query_request.query,
                search_results=search_results,
                response_format=query_request.response_format
            )
            
            # Step 4: Generate citations
            citations = []
            if query_request.include_citations:
                citations = self._generate_citations(search_results, response_text)
            
            # Step 5: Post-process response
            final_response = self._post_process_response(response_text, citations)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Processed query in {processing_time:.2f}s, found {len(search_results)} results")
            
            return QueryResponse(
                query=query_request.query,
                response=final_response,
                citations=citations,
                search_results_count=len(search_results),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return QueryResponse(
                query=query_request.query,
                response=f"I encountered an error while processing your query: {str(e)}",
                citations=[],
                search_results_count=0,
                processing_time=time.time() - start_time
            )
    
    async def _generate_response(
        self, 
        query: str, 
        search_results: List[SearchResult],
        response_format: str = "detailed"
    ) -> str:
        """Generate response using Groq based on search results."""
        try:
            # Prepare context from search results
            context_chunks = []
            for i, result in enumerate(search_results[:15]):  # Limit context size
                context_chunks.append(
                    f"[{i+1}] Title: {result.title}\n"
                    f"Authors: {', '.join(result.authors) if result.authors else 'Unknown'}\n"
                    f"Content: {result.text}\n"
                    f"Relevance: {result.similarity_score:.3f}\n"
                )
            
            context = "\n---\n".join(context_chunks)
            
            # Get appropriate prompt template
            prompt_template = self.response_templates.get(response_format, self._detailed_response_prompt)
            
            # Generate system prompt
            system_prompt = self._create_system_prompt()
            
            # Generate user prompt
            user_prompt = prompt_template(query, context)
            
            # Generate response
            chat_completion = await asyncio.to_thread(
                self.client.chat.completions.create,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=settings.groq_model,
                max_tokens=settings.groq_max_tokens,
                temperature=settings.groq_temperature,
                top_p=0.95,
                stream=False
            )
            
            response = chat_completion.choices[0].message.content.strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for Groq model."""
        return """You are an expert research assistant specializing in Argo oceanographic data and marine science. 
        
Your role is to:
1. Analyze scientific literature about Argo floats and oceanographic data
2. Provide accurate, evidence-based answers using the provided context
3. Cite sources appropriately using [1], [2], etc. format
4. Acknowledge limitations when information is insufficient
5. Use scientific terminology appropriately while remaining accessible

Guidelines:
- Base your answers strictly on the provided context
- Use citation numbers [1], [2], etc. to reference specific sources
- If the context doesn't contain sufficient information, clearly state this
- Maintain scientific accuracy and precision
- Structure your response clearly with appropriate sections
- Include quantitative data when available
- Explain technical concepts when necessary

Context will be provided as numbered sources with titles, authors, and content excerpts."""
    
    def _brief_response_prompt(self, query: str, context: str) -> str:
        """Generate brief response prompt."""
        return f"""
Based on the following research context about Argo data, provide a concise answer to the user's question.
Keep your response to 2-3 sentences and include the most relevant citations.

Question: {query}

Research Context:
{context}

Instructions:
- Provide a direct, concise answer
- Include 1-2 key citations using [1], [2] format
- Focus on the most important finding or information
"""
    
    def _detailed_response_prompt(self, query: str, context: str) -> str:
        """Generate detailed response prompt."""
        return f"""
Based on the following research context about Argo data, provide a comprehensive answer to the user's question.
Structure your response with clear explanations and appropriate citations.

Question: {query}

Research Context:
{context}

Instructions:
- Provide a well-structured, detailed answer
- Include relevant background information
- Use citations [1], [2], [3], etc. throughout your response
- Explain technical concepts clearly
- Include quantitative data when available
- Address different aspects of the question
"""
    
    def _comprehensive_response_prompt(self, query: str, context: str) -> str:
        """Generate comprehensive response prompt."""
        return f"""
Based on the following research context about Argo data, provide a thorough, comprehensive analysis of the user's question.
Create a detailed response that covers multiple perspectives and includes extensive citations.

Question: {query}

Research Context:
{context}

Instructions:
- Provide an extensive, multi-paragraph response
- Include detailed background and context
- Cover different aspects and perspectives of the topic
- Use extensive citations [1], [2], [3], etc. throughout
- Include methodological details when relevant
- Discuss implications and significance
- Address potential limitations or uncertainties
- Structure with clear sections if appropriate
"""
    
    def _generate_citations(
        self, 
        search_results: List[SearchResult], 
        response_text: str
    ) -> List[Citation]:
        """Generate citations based on search results and response text."""
        citations = []
        
        # Find all citation numbers in the response
        citation_numbers = re.findall(r'\[(\d+)\]', response_text)
        cited_numbers = set(map(int, citation_numbers))
        
        # Generate citations for referenced results
        for i, result in enumerate(search_results):
            if (i + 1) in cited_numbers:
                citation = Citation(
                    document_id=result.document_id,
                    title=result.title,
                    authors=result.authors,
                    publication_date=result.publication_date,
                    doi=None,  # Would need to be extracted from metadata
                    page_number=result.page_number,
                    chunk_text=result.text[:500] + "..." if len(result.text) > 500 else result.text,
                    relevance_score=result.similarity_score,
                    citation_text=self._format_citation_text(result)
                )
                citations.append(citation)
        
        # Sort citations by citation number appearance in text
        citations.sort(key=lambda c: search_results.index(
            next(r for r in search_results if r.document_id == c.document_id)
        ))
        
        return citations[:settings.max_citations]
    
    def _format_citation_text(self, search_result: SearchResult) -> str:
        """Format citation text in academic style."""
        authors_text = "Unknown"
        if search_result.authors:
            if len(search_result.authors) == 1:
                authors_text = search_result.authors[0]
            elif len(search_result.authors) == 2:
                authors_text = f"{search_result.authors[0]} & {search_result.authors[1]}"
            else:
                authors_text = f"{search_result.authors[0]} et al."
        
        year_text = ""
        if search_result.publication_date:
            year_text = f" ({search_result.publication_date.year})"
        
        title_text = search_result.title if search_result.title != "Untitled" else "Document"
        
        page_text = ""
        if search_result.page_number:
            page_text = f", p. {search_result.page_number}"
        
        return f"{authors_text}{year_text}. {title_text}{page_text}."
    
    def _post_process_response(self, response_text: str, citations: List[Citation]) -> str:
        """Post-process response to improve formatting and add citation list."""
        # Clean up response
        response = response_text.strip()
        
        # Add citation list if citations exist
        if citations:
            response += "\n\n**References:**\n"
            for i, citation in enumerate(citations):
                response += f"[{i+1}] {citation.citation_text}\n"
        
        return response
    
    async def generate_follow_up_questions(self, query: str, response: str) -> List[str]:
        """Generate follow-up questions based on the query and response."""
        try:
            if not self.client:
                return []
            
            prompt = f"""
Based on the following question and answer about Argo oceanographic data, 
generate 3 relevant follow-up questions that users might want to ask.

Original Question: {query}

Answer: {response[:1000]}...

Generate 3 concise follow-up questions that would naturally extend the conversation:
"""
            
            chat_completion = await asyncio.to_thread(
                self.client.chat.completions.create,
                messages=[{"role": "user", "content": prompt}],
                model=settings.groq_model,
                max_tokens=200,
                temperature=0.7
            )
            
            response_text = chat_completion.choices[0].message.content.strip()
            
            # Extract questions (assuming they're numbered or bulleted)
            questions = []
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and ('?' in line):
                    # Clean up question formatting
                    question = re.sub(r'^[\d\.\-\*\s]+', '', line).strip()
                    if question and len(question) > 10:
                        questions.append(question)
            
            return questions[:3]
            
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {str(e)}")
            return []
    
    async def summarize_document(self, document_id: str) -> str:
        """Generate a summary of a specific document."""
        try:
            if not self.client:
                return "Summary service not available."
            
            # Get all chunks for the document
            chunks = await qdrant_service.get_document_chunks(document_id)
            
            if not chunks:
                return "Document not found or has no content."
            
            # Combine chunk texts
            full_text = " ".join([chunk.get("text", "") for chunk in chunks[:20]])  # Limit for API
            
            prompt = f"""
Provide a comprehensive summary of the following Argo oceanographic research document.
Include key findings, methodology, and significance.

Document Content:
{full_text[:4000]}...

Create a structured summary with:
1. Main topic and research focus
2. Key findings
3. Methodology (if described)
4. Significance and implications
"""
            
            chat_completion = await asyncio.to_thread(
                self.client.chat.completions.create,
                messages=[{"role": "user", "content": prompt}],
                model=settings.groq_model,
                max_tokens=1000,
                temperature=0.3
            )
            
            return chat_completion.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error summarizing document: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    async def check_connection(self) -> bool:
        """Check if Groq service is available."""
        try:
            if not self.client:
                return False
            
            # Test with a simple query
            chat_completion = await asyncio.to_thread(
                self.client.chat.completions.create,
                messages=[{"role": "user", "content": "Hello"}],
                model=settings.groq_model,
                max_tokens=10
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Groq connection check failed: {str(e)}")
            return False


# Create global instance
groq_service = GroqService()
