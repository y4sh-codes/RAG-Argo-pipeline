"""
FastAPI routers for report generation endpoints.
Handles PDF report generation with different templates and formats.
"""

import os
from typing import Optional
import logging

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from datetime import datetime

from ..models import ReportRequest, ReportResponse
from ..services.report_generator import pdf_generator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reports", tags=["reports"])


@router.post("/generate", response_model=ReportResponse)
async def generate_report(request: ReportRequest):
    """
    Generate a comprehensive PDF report based on a research query.
    
    This endpoint creates a professional PDF report that includes:
    - Research findings based on the query
    - Proper citations and references
    - Charts and visualizations (if requested)
    - Multiple template options (scientific, business, academic)
    """
    try:
        report_response = await pdf_generator.generate_report(request)
        return report_response
    
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {str(e)}"
        )


@router.get("/{report_id}/download")
async def download_report(report_id: str):
    """
    Download a generated PDF report.
    
    This endpoint serves the generated PDF file for download.
    """
    try:
        # Find the report file
        # In a real implementation, you'd store report metadata in a database
        # For now, we'll look for files with the report_id pattern
        
        from ..config import settings
        import glob
        
        pattern = os.path.join(settings.output_dir, f"*{report_id}*.pdf")
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            raise HTTPException(status_code=404, detail="Report not found")
        
        file_path = matching_files[0]
        filename = os.path.basename(file_path)
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/pdf'
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading report {report_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error downloading report")


@router.get("/")
async def list_reports():
    """
    List all generated reports.
    
    Returns a list of available reports with their metadata.
    """
    try:
        from ..config import settings
        import glob
        
        # Get all PDF files in the output directory
        pattern = os.path.join(settings.output_dir, "*.pdf")
        report_files = glob.glob(pattern)
        
        reports = []
        for file_path in report_files:
            try:
                filename = os.path.basename(file_path)
                file_stats = os.stat(file_path)
                
                # Extract report ID from filename
                report_id = filename.split('_')[2] if '_' in filename else "unknown"
                
                reports.append({
                    "report_id": report_id,
                    "filename": filename,
                    "size_bytes": file_stats.st_size,
                    "created_at": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                })
            except Exception as e:
                logger.error(f"Error processing report file {file_path}: {str(e)}")
                continue
        
        # Sort by creation time (newest first)
        reports.sort(key=lambda x: x["created_at"], reverse=True)
        
        return {
            "reports": reports,
            "total_reports": len(reports)
        }
    
    except Exception as e:
        logger.error(f"Error listing reports: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving report list")


@router.delete("/{report_id}")
async def delete_report(report_id: str):
    """
    Delete a generated report.
    
    This endpoint removes the PDF file and any associated metadata.
    """
    try:
        from ..config import settings
        import glob
        
        pattern = os.path.join(settings.output_dir, f"*{report_id}*.pdf")
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            raise HTTPException(status_code=404, detail="Report not found")
        
        file_path = matching_files[0]
        os.remove(file_path)
        
        logger.info(f"Deleted report: {report_id}")
        
        return {"message": f"Report {report_id} deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting report {report_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error deleting report")


@router.get("/templates")
async def get_report_templates():
    """
    Get available report templates and their descriptions.
    
    Returns information about available report templates and their features.
    """
    try:
        templates = {
            "scientific": {
                "name": "Scientific Research Report",
                "description": "Professional scientific format with abstract, methodology, findings, and comprehensive references",
                "features": ["Abstract", "Literature Review", "Detailed Analysis", "Charts", "References", "Appendices"],
                "best_for": "Academic research, scientific publications, detailed analysis"
            },
            "business": {
                "name": "Business Report",
                "description": "Executive summary format focused on key findings and recommendations",
                "features": ["Executive Summary", "Key Findings", "Recommendations", "Supporting Charts"],
                "best_for": "Business presentations, executive briefings, decision support"
            },
            "academic": {
                "name": "Academic Paper",
                "description": "Traditional academic paper format with introduction, literature review, and conclusion",
                "features": ["Introduction", "Literature Review", "Analysis", "Conclusion", "References"],
                "best_for": "Academic assignments, thesis work, formal research papers"
            }
        }
        
        return {
            "templates": templates,
            "total_templates": len(templates),
            "default_template": "scientific"
        }
    
    except Exception as e:
        logger.error(f"Error getting report templates: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving templates")


@router.post("/preview")
async def preview_report(request: ReportRequest):
    """
    Generate a preview of what the report would contain without creating the full PDF.
    
    This endpoint returns a structured preview of the report content,
    including sections, estimated page count, and citation list.
    """
    try:
        from ..services.groq_service import groq_service
        
        # Process the query to get content
        query_response = await groq_service.process_query(request)
        
        # Generate preview structure
        preview = {
            "title": request.title,
            "template": request.template,
            "estimated_sections": [],
            "estimated_pages": 0,
            "citation_count": len(query_response.citations),
            "word_count": len(query_response.response.split()),
            "processing_time": query_response.processing_time,
            "content_preview": query_response.response[:500] + "..." if len(query_response.response) > 500 else query_response.response
        }
        
        # Estimate sections based on template
        if request.template == "scientific":
            preview["estimated_sections"] = [
                "Title Page", "Abstract", "Introduction", "Findings and Analysis",
                "Figures and Charts" if request.include_charts else None,
                "References" if request.include_citations else None,
                "Appendices"
            ]
            preview["estimated_pages"] = 8 + (2 if request.include_charts else 0)
        elif request.template == "business":
            preview["estimated_sections"] = [
                "Executive Summary", "Key Findings", "Detailed Analysis",
                "Recommendations", "Supporting Data"
            ]
            preview["estimated_pages"] = 5 + (2 if request.include_charts else 0)
        else:  # academic
            preview["estimated_sections"] = [
                "Title Page", "Abstract", "Introduction", "Literature Review",
                "Analysis", "Conclusion", "References"
            ]
            preview["estimated_pages"] = 7 + (1 if request.include_charts else 0)
        
        # Remove None values
        preview["estimated_sections"] = [s for s in preview["estimated_sections"] if s is not None]
        
        return preview
    
    except Exception as e:
        logger.error(f"Error generating report preview: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating preview")


@router.get("/{report_id}/metadata")
async def get_report_metadata(report_id: str):
    """
    Get metadata for a specific report.
    
    Returns detailed information about the report including size, creation date,
    and other relevant metadata.
    """
    try:
        from ..config import settings
        import glob
        
        pattern = os.path.join(settings.output_dir, f"*{report_id}*.pdf")
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            raise HTTPException(status_code=404, detail="Report not found")
        
        file_path = matching_files[0]
        filename = os.path.basename(file_path)
        file_stats = os.stat(file_path)
        
        # Try to count pages (requires PyPDF2)
        page_count = 1
        try:
            import PyPDF2
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                page_count = len(reader.pages)
        except Exception:
            pass  # Use default page count
        
        metadata = {
            "report_id": report_id,
            "filename": filename,
            "file_path": file_path,
            "size_bytes": file_stats.st_size,
            "size_mb": round(file_stats.st_size / (1024 * 1024), 2),
            "pages": page_count,
            "created_at": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            "file_extension": "pdf",
            "mime_type": "application/pdf"
        }
        
        return metadata
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting report metadata {report_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving report metadata")
