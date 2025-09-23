#!/usr/bin/env python3
"""
Batch PDF Processing Script
Upload and process multiple PDFs at once
"""
import os
import requests
import time
from pathlib import Path

# Configuration
API_BASE_URL = "http://127.0.0.1:8000"
PDF_DIRECTORY = "data/pdfs"

def upload_pdf(file_path):
    """Upload a single PDF file."""
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'application/pdf')}
            response = requests.post(f"{API_BASE_URL}/documents/upload", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Uploaded: {os.path.basename(file_path)}")
            print(f"   Document ID: {result.get('document_id')}")
            print(f"   Status: {result.get('status')}")
            return True
        else:
            print(f"❌ Failed to upload {os.path.basename(file_path)}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error uploading {os.path.basename(file_path)}: {str(e)}")
        return False

def process_pdfs_batch():
    """Process all PDFs in the directory."""
    pdf_dir = Path(PDF_DIRECTORY)
    
    if not pdf_dir.exists():
        print(f"❌ Directory {PDF_DIRECTORY} does not exist!")
        print("Please create it and add your PDF files.")
        return
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in {PDF_DIRECTORY}")
        return
    
    print(f"📄 Found {len(pdf_files)} PDF files to process...")
    
    successful = 0
    failed = 0
    
    for pdf_file in pdf_files:
        print(f"\n📤 Processing: {pdf_file.name}")
        if upload_pdf(pdf_file):
            successful += 1
            # Small delay to avoid overwhelming the server
            time.sleep(1)
        else:
            failed += 1
    
    print(f"\n📊 Processing Summary:")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📄 Total: {len(pdf_files)}")

def check_server_health():
    """Check if the server is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Server is running and healthy!")
            return True
        else:
            print("❌ Server responded but may have issues")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running?")
        print("Start the server with: python -m uvicorn app.main:app --reload")
        return False

if __name__ == "__main__":
    print("🚀 RAG Pipeline - PDF Batch Processor")
    print("=" * 50)
    
    # Check server health first
    if not check_server_health():
        exit(1)
    
    # Process PDFs
    process_pdfs_batch()
    
    print("\n🎉 Batch processing complete!")
    print("Visit http://localhost:8000/docs to interact with your processed documents.")