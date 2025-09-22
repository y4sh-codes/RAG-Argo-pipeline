# PDF Documents Directory

This directory is where you should place PDF documents for processing by the RAG Argo Pipeline.

## 📄 How to Add PDFs

### Method 1: Direct File Copy
Simply copy your PDF files into this directory:
```bash
# Example:
cp your_research_paper.pdf /path/to/rag_argo_pipeline/data/pdfs/
```

### Method 2: API Upload
Use the REST API to upload PDFs:
```bash
curl -X POST "http://localhost/documents/upload" \
  -F "file=@your_paper.pdf"
```

## ✅ Supported Files

- **Format**: PDF files only (`.pdf` extension)
- **Size**: Maximum 50MB per file
- **Content**: Text-based PDFs (scanned images may not process well)
- **Languages**: Primarily optimized for English content

## 🔍 What Happens Next?

Once you add PDFs here:

1. **Auto-Detection**: The system monitors this directory for new files
2. **Text Extraction**: Content is extracted using PyMuPDF, pdfplumber, or PyPDF2
3. **Metadata Extraction**: Title, authors, abstract, DOI, etc. are identified
4. **Chunking**: Documents are split into semantic chunks
5. **Embedding**: Text is converted to vector embeddings
6. **Search Ready**: Documents become searchable via the API

## 📋 File Naming

For best results, use descriptive filenames:

✅ **Good Examples**:
- `marine_biology_argo_2024.pdf`
- `ocean_temperature_analysis_smith_2023.pdf`
- `deep_sea_research_methodology.pdf`

❌ **Avoid**:
- `document1.pdf`
- `untitled.pdf`
- `scan001.pdf`

## 🔧 Processing Status

Check if your PDFs are processed:
```bash
# Check system health
curl http://localhost/health

# List processed documents
curl http://localhost/documents/
```

## 🚨 Troubleshooting

**PDF not processing?**
- Check file size (must be < 50MB)
- Ensure it's a text-based PDF, not a scanned image
- Look at logs: `docker-compose logs rag-api`

**Processing failed?**
- Try a different PDF extraction method in settings
- Check the error logs for specific issues
- Ensure the PDF is not password-protected

---

**Ready to get started?** Just drop your PDF files in this directory and they'll be automatically processed!
