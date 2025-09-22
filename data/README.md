# Data Directory

This directory is used to store PDF documents for processing by the RAG Argo Pipeline.

## 📁 Directory Structure

```
data/
├── README.md          # This file
├── pdfs/             # PDF documents go here
│   └── README.md     # PDF-specific instructions
└── processed/        # Processed documents metadata (auto-created)
```

## 📄 PDF Storage

- **Location**: Place your PDF files in the `pdfs/` subdirectory
- **Supported Formats**: `.pdf` files only
- **File Size Limit**: Maximum 50MB per file (configurable via `MAX_FILE_SIZE_MB`)
- **File Types**: Research papers, academic documents, reports, etc.

## 🚀 Usage

1. **Manual Upload**: Copy PDF files directly to `data/pdfs/`
2. **API Upload**: Use the `/documents/upload` endpoint
3. **Batch Processing**: The system will automatically process new files

## 📋 Examples

```bash
# Example file structure after adding PDFs:
data/pdfs/
├── oceanography_paper_2024.pdf
├── argo_float_data_analysis.pdf
└── marine_research_report.pdf
```

## ⚙️ Configuration

The data directory path can be configured in your `.env` file:

```env
DATA_DIR="./data"
PDF_DIR="./data/pdfs"
```

## 🔄 Processing Pipeline

When PDFs are added to this directory:

1. **Extraction**: Text and metadata are extracted
2. **Chunking**: Documents are split into semantic chunks
3. **Embedding**: Text chunks are converted to vector embeddings
4. **Storage**: Vectors are stored in Qdrant database
5. **Indexing**: Documents become searchable via API

## 📊 Monitoring

- Check processing status via `/health` endpoint
- View document statistics in Grafana dashboard
- Monitor logs in `logs/` directory

---

**Note**: This directory is automatically created when the application starts. The processed document metadata and temporary files may also be stored in subdirectories here.
