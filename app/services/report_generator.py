"""
PDF report generation service for creating professional research reports.
Supports multiple templates and includes charts, citations, and formatting.
"""

import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import uuid
import io
import base64

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT, TA_RIGHT
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from ..config import settings
from ..models import ReportRequest, ReportResponse, Citation, QueryResponse
from .groq_service import groq_service

logger = logging.getLogger(__name__)


class PDFReportGenerator:
    """Professional PDF report generator with scientific templates."""
    
    def __init__(self):
        # Set matplotlib backend for headless operation
        plt.switch_backend('Agg')
        
        # Configure styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        # Report templates
        self.templates = {
            "scientific": self._scientific_template,
            "business": self._business_template,
            "academic": self._academic_template
        }
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2E86AB'),
            alignment=TA_CENTER
        ))
        
        # Heading styles
        self.styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#2E86AB'),
            borderWidth=1,
            borderColor=colors.HexColor('#2E86AB'),
            borderPadding=5
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            textColor=colors.HexColor('#A23B72')
        ))
        
        # Body text
        self.styles.add(ParagraphStyle(
            name='CustomNormal',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            alignment=TA_JUSTIFY,
            leftIndent=0,
            rightIndent=0
        ))
        
        # Citation style
        self.styles.add(ParagraphStyle(
            name='Citation',
            parent=self.styles['Normal'],
            fontSize=9,
            leftIndent=20,
            rightIndent=20,
            spaceAfter=6,
            textColor=colors.grey
        ))
        
        # Footer style
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            alignment=TA_CENTER,
            textColor=colors.grey
        ))
    
    async def generate_report(self, request: ReportRequest) -> ReportResponse:
        """
        Generate a comprehensive PDF report based on query and research.
        
        Args:
            request: Report generation request
            
        Returns:
            ReportResponse with file information
        """
        try:
            # Generate unique report ID
            report_id = str(uuid.uuid4())
            filename = f"argo_report_{report_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            file_path = os.path.join(settings.output_dir, filename)
            
            # Ensure output directory exists
            os.makedirs(settings.output_dir, exist_ok=True)
            
            # Process query to get research content
            query_response = await groq_service.process_query(request)
            
            # Generate additional analysis if needed
            research_data = await self._gather_research_data(request, query_response)
            
            # Select template
            template_func = self.templates.get(request.template, self._scientific_template)
            
            # Generate PDF
            await template_func(file_path, request, query_response, research_data)
            
            # Get file stats
            file_size = os.path.getsize(file_path)
            page_count = await self._count_pdf_pages(file_path)
            
            logger.info(f"Generated report: {filename}, size: {file_size} bytes, pages: {page_count}")
            
            return ReportResponse(
                report_id=report_id,
                filename=filename,
                file_path=file_path,
                pages=page_count,
                size_bytes=file_size
            )
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise Exception(f"Report generation failed: {str(e)}")
    
    async def _gather_research_data(
        self, 
        request: ReportRequest, 
        query_response: QueryResponse
    ) -> Dict[str, Any]:
        """Gather additional research data for the report."""
        research_data = {
            "query_response": query_response,
            "charts": [],
            "statistics": {},
            "related_topics": []
        }
        
        try:
            # Generate charts if requested
            if request.include_charts:
                charts = await self._generate_charts(query_response)
                research_data["charts"] = charts
            
            # Generate statistics
            research_data["statistics"] = {
                "total_sources": len(query_response.citations),
                "search_results": query_response.search_results_count,
                "processing_time": query_response.processing_time,
                "query_length": len(request.query),
                "response_length": len(query_response.response)
            }
            
            # Generate related topics
            related_topics = await self._extract_related_topics(query_response)
            research_data["related_topics"] = related_topics
            
        except Exception as e:
            logger.error(f"Error gathering research data: {str(e)}")
        
        return research_data
    
    async def _scientific_template(
        self, 
        file_path: str, 
        request: ReportRequest, 
        query_response: QueryResponse,
        research_data: Dict[str, Any]
    ):
        """Generate PDF using scientific template."""
        doc = SimpleDocTemplate(
            file_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        story = []
        
        # Title Page
        story.extend(self._create_title_page(request, research_data))
        story.append(PageBreak())
        
        # Abstract/Executive Summary
        story.extend(self._create_abstract(query_response, research_data))
        story.append(Spacer(1, 12))
        
        # Main Content
        story.extend(self._create_main_content(query_response, research_data))
        
        # Charts and Figures
        if request.include_charts and research_data.get("charts"):
            story.append(PageBreak())
            story.extend(self._create_charts_section(research_data["charts"]))
        
        # References
        if request.include_citations and query_response.citations:
            story.append(PageBreak())
            story.extend(self._create_references_section(query_response.citations))
        
        # Appendices
        story.extend(self._create_appendices(research_data))
        
        # Build PDF
        doc.build(story)
    
    async def _business_template(
        self, 
        file_path: str, 
        request: ReportRequest, 
        query_response: QueryResponse,
        research_data: Dict[str, Any]
    ):
        """Generate PDF using business template."""
        doc = SimpleDocTemplate(file_path, pagesize=letter)
        story = []
        
        # Executive Summary
        story.extend(self._create_business_header(request))
        story.extend(self._create_executive_summary(query_response))
        
        # Key Findings
        story.extend(self._create_key_findings(query_response, research_data))
        
        # Detailed Analysis
        story.extend(self._create_business_analysis(query_response))
        
        # Recommendations
        story.extend(self._create_recommendations(query_response))
        
        # Supporting Data
        if request.include_charts:
            story.extend(self._create_charts_section(research_data.get("charts", [])))
        
        doc.build(story)
    
    async def _academic_template(
        self, 
        file_path: str, 
        request: ReportRequest, 
        query_response: QueryResponse,
        research_data: Dict[str, Any]
    ):
        """Generate PDF using academic template."""
        doc = SimpleDocTemplate(file_path, pagesize=A4)
        story = []
        
        # Title and Abstract
        story.extend(self._create_academic_header(request))
        story.extend(self._create_abstract(query_response, research_data))
        
        # Introduction
        story.extend(self._create_introduction(query_response))
        
        # Literature Review
        story.extend(self._create_literature_review(query_response))
        
        # Analysis
        story.extend(self._create_analysis(query_response, research_data))
        
        # Conclusion
        story.extend(self._create_conclusion(query_response))
        
        # References
        if query_response.citations:
            story.extend(self._create_references_section(query_response.citations))
        
        doc.build(story)
    
    def _create_title_page(self, request: ReportRequest, research_data: Dict[str, Any]) -> List:
        """Create report title page."""
        elements = []
        
        # Main title
        title = Paragraph(request.title, self.styles['CustomTitle'])
        elements.append(title)
        elements.append(Spacer(1, 30))
        
        # Subtitle with query
        subtitle = Paragraph(f"Research Query: <i>{request.query}</i>", self.styles['Heading2'])
        elements.append(subtitle)
        elements.append(Spacer(1, 20))
        
        # Report metadata
        metadata_data = [
            ["Report ID:", research_data.get("query_response", {}).get("timestamp", datetime.now()).strftime("%Y%m%d-%H%M%S")],
            ["Generated:", datetime.now().strftime("%B %d, %Y at %H:%M")],
            ["Sources:", str(research_data.get("statistics", {}).get("total_sources", "N/A"))],
            ["Processing Time:", f"{research_data.get('statistics', {}).get('processing_time', 0):.2f} seconds"],
            ["Template:", request.template.title()]
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 3*inch])
        metadata_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        elements.append(Spacer(1, 40))
        elements.append(metadata_table)
        elements.append(Spacer(1, 40))
        
        # Disclaimer
        disclaimer = Paragraph(
            "<b>Disclaimer:</b> This report is generated automatically based on available research data. "
            "Please verify all information and consult original sources for critical decisions.",
            self.styles['Citation']
        )
        elements.append(disclaimer)
        
        return elements
    
    def _create_abstract(self, query_response: QueryResponse, research_data: Dict[str, Any]) -> List:
        """Create abstract/executive summary."""
        elements = []
        
        elements.append(Paragraph("Abstract", self.styles['CustomHeading1']))
        
        # Generate abstract from response
        abstract_text = self._extract_abstract(query_response.response)
        abstract = Paragraph(abstract_text, self.styles['CustomNormal'])
        elements.append(abstract)
        
        return elements
    
    def _create_main_content(self, query_response: QueryResponse, research_data: Dict[str, Any]) -> List:
        """Create main content sections."""
        elements = []
        
        # Introduction
        elements.append(Paragraph("Introduction", self.styles['CustomHeading1']))
        intro_text = f"This report addresses the research question: \"{query_response.query}\". " \
                    f"The analysis is based on {len(query_response.citations)} scientific sources " \
                    f"related to Argo oceanographic data."
        elements.append(Paragraph(intro_text, self.styles['CustomNormal']))
        elements.append(Spacer(1, 12))
        
        # Main findings
        elements.append(Paragraph("Findings and Analysis", self.styles['CustomHeading1']))
        
        # Split response into paragraphs
        response_paragraphs = query_response.response.split('\n\n')
        for para in response_paragraphs:
            if para.strip():
                # Clean up citation references for ReportLab
                cleaned_para = para.replace('[', '<super>[').replace(']', ']</super>')
                elements.append(Paragraph(cleaned_para, self.styles['CustomNormal']))
                elements.append(Spacer(1, 6))
        
        return elements
    
    def _create_charts_section(self, charts: List[Dict[str, Any]]) -> List:
        """Create charts and figures section."""
        elements = []
        
        elements.append(Paragraph("Figures and Analysis", self.styles['CustomHeading1']))
        
        for i, chart_data in enumerate(charts):
            # Chart title
            chart_title = f"Figure {i+1}: {chart_data.get('title', 'Data Visualization')}"
            elements.append(Paragraph(chart_title, self.styles['CustomHeading2']))
            
            # Add chart image
            if 'image_path' in chart_data:
                try:
                    img = Image(chart_data['image_path'], width=6*inch, height=4*inch)
                    elements.append(img)
                except Exception as e:
                    logger.error(f"Error adding chart image: {str(e)}")
                    elements.append(Paragraph("Chart could not be displayed", self.styles['Citation']))
            
            # Chart description
            if 'description' in chart_data:
                elements.append(Spacer(1, 6))
                elements.append(Paragraph(chart_data['description'], self.styles['CustomNormal']))
            
            elements.append(Spacer(1, 12))
        
        return elements
    
    def _create_references_section(self, citations: List[Citation]) -> List:
        """Create references section."""
        elements = []
        
        elements.append(Paragraph("References", self.styles['CustomHeading1']))
        
        for i, citation in enumerate(citations):
            citation_text = f"[{i+1}] {citation.citation_text}"
            elements.append(Paragraph(citation_text, self.styles['Citation']))
            elements.append(Spacer(1, 4))
        
        return elements
    
    def _create_appendices(self, research_data: Dict[str, Any]) -> List:
        """Create appendices section."""
        elements = []
        
        elements.append(PageBreak())
        elements.append(Paragraph("Appendices", self.styles['CustomHeading1']))
        
        # Appendix A: Search Statistics
        elements.append(Paragraph("Appendix A: Search and Processing Statistics", self.styles['CustomHeading2']))
        
        stats = research_data.get("statistics", {})
        stats_data = [
            ["Total Sources Found:", str(stats.get("total_sources", "N/A"))],
            ["Search Results:", str(stats.get("search_results", "N/A"))],
            ["Processing Time:", f"{stats.get('processing_time', 0):.2f} seconds"],
            ["Query Length:", f"{stats.get('query_length', 0)} characters"],
            ["Response Length:", f"{stats.get('response_length', 0)} characters"]
        ]
        
        stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
        stats_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ]))
        
        elements.append(stats_table)
        elements.append(Spacer(1, 12))
        
        # Appendix B: Related Topics
        related_topics = research_data.get("related_topics", [])
        if related_topics:
            elements.append(Paragraph("Appendix B: Related Research Topics", self.styles['CustomHeading2']))
            for topic in related_topics[:10]:  # Limit to 10 topics
                elements.append(Paragraph(f"• {topic}", self.styles['CustomNormal']))
        
        return elements
    
    async def _generate_charts(self, query_response: QueryResponse) -> List[Dict[str, Any]]:
        """Generate charts based on query response."""
        charts = []
        
        try:
            # Chart 1: Source relevance distribution
            if query_response.citations:
                relevance_scores = [c.relevance_score for c in query_response.citations]
                
                plt.figure(figsize=(10, 6))
                plt.hist(relevance_scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
                plt.xlabel('Relevance Score')
                plt.ylabel('Number of Sources')
                plt.title('Distribution of Source Relevance Scores')
                plt.grid(True, alpha=0.3)
                
                chart_path = os.path.join(settings.output_dir, f"relevance_dist_{uuid.uuid4().hex[:8]}.png")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                charts.append({
                    "title": "Source Relevance Distribution",
                    "image_path": chart_path,
                    "description": "Distribution of relevance scores for sources used in this analysis."
                })
            
            # Chart 2: Publication timeline (if date info available)
            pub_dates = [c.publication_date.year for c in query_response.citations 
                        if c.publication_date]
            
            if len(pub_dates) > 1:
                plt.figure(figsize=(10, 6))
                plt.hist(pub_dates, bins=min(len(set(pub_dates)), 10), alpha=0.7, 
                        color='lightcoral', edgecolor='black')
                plt.xlabel('Publication Year')
                plt.ylabel('Number of Publications')
                plt.title('Timeline of Source Publications')
                plt.grid(True, alpha=0.3)
                
                chart_path = os.path.join(settings.output_dir, f"timeline_{uuid.uuid4().hex[:8]}.png")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                charts.append({
                    "title": "Publication Timeline",
                    "image_path": chart_path,
                    "description": "Timeline showing when the source publications were released."
                })
        
        except Exception as e:
            logger.error(f"Error generating charts: {str(e)}")
        
        return charts
    
    async def _extract_related_topics(self, query_response: QueryResponse) -> List[str]:
        """Extract related topics from the response."""
        topics = []
        
        # Simple keyword extraction from response
        response_text = query_response.response.lower()
        
        # Common Argo-related terms
        argo_terms = [
            "temperature profiles", "salinity measurements", "ocean currents",
            "mixed layer depth", "water masses", "thermohaline circulation",
            "meridional overturning", "deep water formation", "upwelling",
            "climate variability", "ocean heat content", "sea level rise"
        ]
        
        for term in argo_terms:
            if term in response_text:
                topics.append(term.title())
        
        return topics[:10]
    
    def _extract_abstract(self, response_text: str) -> str:
        """Extract or generate abstract from response."""
        # Take first 2-3 sentences as abstract
        sentences = response_text.split('. ')
        abstract_sentences = sentences[:3]
        abstract = '. '.join(abstract_sentences)
        
        if not abstract.endswith('.'):
            abstract += '.'
        
        return abstract
    
    async def _count_pdf_pages(self, file_path: str) -> int:
        """Count pages in generated PDF."""
        try:
            import PyPDF2
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                return len(reader.pages)
        except Exception:
            return 1  # Default to 1 page if can't count
    
    # Business template methods
    def _create_business_header(self, request: ReportRequest) -> List:
        """Create business report header."""
        elements = []
        elements.append(Paragraph(request.title, self.styles['CustomTitle']))
        elements.append(Spacer(1, 20))
        return elements
    
    def _create_executive_summary(self, query_response: QueryResponse) -> List:
        """Create executive summary for business template."""
        elements = []
        elements.append(Paragraph("Executive Summary", self.styles['CustomHeading1']))
        
        summary = self._extract_abstract(query_response.response)
        elements.append(Paragraph(summary, self.styles['CustomNormal']))
        elements.append(Spacer(1, 12))
        
        return elements
    
    def _create_key_findings(self, query_response: QueryResponse, research_data: Dict[str, Any]) -> List:
        """Create key findings section."""
        elements = []
        elements.append(Paragraph("Key Findings", self.styles['CustomHeading1']))
        
        # Extract key points from response
        points = query_response.response.split('\n')
        key_points = [p.strip() for p in points if p.strip() and len(p.strip()) > 50]
        
        for i, point in enumerate(key_points[:5]):  # Limit to 5 key findings
            elements.append(Paragraph(f"{i+1}. {point}", self.styles['CustomNormal']))
            elements.append(Spacer(1, 6))
        
        return elements
    
    def _create_business_analysis(self, query_response: QueryResponse) -> List:
        """Create detailed analysis for business template."""
        elements = []
        elements.append(Paragraph("Detailed Analysis", self.styles['CustomHeading1']))
        elements.append(Paragraph(query_response.response, self.styles['CustomNormal']))
        return elements
    
    def _create_recommendations(self, query_response: QueryResponse) -> List:
        """Create recommendations section."""
        elements = []
        elements.append(Paragraph("Recommendations", self.styles['CustomHeading1']))
        
        # Generate generic recommendations
        recommendations = [
            "Continue monitoring relevant research developments in this area",
            "Consider implementing findings in operational procedures",
            "Validate results with additional data sources when possible",
            "Share insights with relevant stakeholders and teams"
        ]
        
        for rec in recommendations:
            elements.append(Paragraph(f"• {rec}", self.styles['CustomNormal']))
            elements.append(Spacer(1, 4))
        
        return elements
    
    # Academic template methods
    def _create_academic_header(self, request: ReportRequest) -> List:
        """Create academic report header."""
        elements = []
        elements.append(Paragraph(request.title, self.styles['CustomTitle']))
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("Research Report", self.styles['Heading2']))
        elements.append(Spacer(1, 12))
        return elements
    
    def _create_introduction(self, query_response: QueryResponse) -> List:
        """Create introduction section."""
        elements = []
        elements.append(Paragraph("Introduction", self.styles['CustomHeading1']))
        
        intro_text = f"This research report examines the question: {query_response.query}. " \
                    "The analysis draws upon current scientific literature related to " \
                    "Argo oceanographic data and marine science research."
        
        elements.append(Paragraph(intro_text, self.styles['CustomNormal']))
        return elements
    
    def _create_literature_review(self, query_response: QueryResponse) -> List:
        """Create literature review section."""
        elements = []
        elements.append(Paragraph("Literature Review", self.styles['CustomHeading1']))
        
        if query_response.citations:
            review_text = f"This review is based on {len(query_response.citations)} " \
                         "relevant scientific sources. The following analysis synthesizes " \
                         "key findings from the literature."
        else:
            review_text = "This analysis is based on available research data."
        
        elements.append(Paragraph(review_text, self.styles['CustomNormal']))
        return elements
    
    def _create_analysis(self, query_response: QueryResponse, research_data: Dict[str, Any]) -> List:
        """Create analysis section."""
        elements = []
        elements.append(Paragraph("Analysis", self.styles['CustomHeading1']))
        elements.append(Paragraph(query_response.response, self.styles['CustomNormal']))
        return elements
    
    def _create_conclusion(self, query_response: QueryResponse) -> List:
        """Create conclusion section."""
        elements = []
        elements.append(Paragraph("Conclusion", self.styles['CustomHeading1']))
        
        # Extract conclusion from response or generate one
        conclusion_text = "Based on the analysis of available research data, " \
                         "this study provides insights into the research question. " \
                         "Further investigation may be warranted to validate these findings."
        
        elements.append(Paragraph(conclusion_text, self.styles['CustomNormal']))
        return elements


# Create global instance
pdf_generator = PDFReportGenerator()
