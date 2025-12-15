
# README.md

# Semantic-Chunking RAG API (FastAPI)

This project is a **FastAPI backend for a Retrieval-Augmented Generation (RAG) system**.
It enables uploading documents, semantic chunking, embedding, querying using an LLM (Gemini), and rechunking previously uploaded documents. It also supports retrieving all uploaded documents.

---

## Table of Contents

1. [Features](#features)
2. [Directory Structure](#directory-structure)
3. [Document Upload & Storage](#document-upload--storage)
4. [Document IDs & Querying](#document-ids--querying)
5. [Rechunking Documents](#rechunking-documents)
6. [Get All Documents](#get-all-documents)
7. [Environment Variables](#environment-variables)
8. [Requirements](#requirements)
9. [Usage Examples](#usage-examples)

---

## Features

* Upload `.txt`, `.md`, `.pdf`, `.docx`, `.html` documents
* Semantic chunking with configurable chunk size and overlap
* Embedding using **SentenceTransformer**
* Store chunks in **ChromaDB** collection
* Query a specific document using `document_id`
* Rechunk and re-embed documents
* Retrieve metadata and list of all uploaded documents

---

## Directory Structure

```
.
├── main.py                # FastAPI app with endpoints
├── rag_data/              # Directory where uploaded documents are stored
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables
└── README.md
```

* All uploaded files are stored in the `RAG_DATA_DIR` directory (`./rag_data` by default).
* Each file is prefixed with a **unique `document_id`** generated on upload:

```
<document_id>_<original_filename>
```

This ensures **unique identification** and enables precise querying and rechunking.

---

## Document Upload & Storage

**Endpoint:** `POST /upload`

* Upload a file via `multipart/form-data`.
* Generates a **unique `document_id`**.
* Extracts text using a **unified helper function** `extract_text_from_file()`.
* Splits text into semantic chunks.
* Embeds each chunk and stores it in ChromaDB with metadata:

```json
{
    "document_id": "12345abcdef",
    "source": "Guide-to-Rice.pdf",
    "chunk_index": 0,
    "file_type": ".pdf"
}
```

* Returns:

```json
{
    "file": "Guide-to-Rice.pdf",
    "num_chunks": 43,
    "status": "ingested",
    "document_id": "12345abcdef"
}
```

> This `document_id` is **critical** for querying, rechunking, or fetching document metadata.

---

## Document IDs & Querying

**Endpoint:** `POST /prompt`

* Requires `document_id` in the request body to **filter queries to a specific document**:

```json
{
  "question": "Is rice farmed in Nigeria?",
  "top_k": 5,
  "document_id": "12345abcdef"
}
```

* Embeds the question and retrieves the **top_k most relevant chunks** from ChromaDB for that document.
* Passes retrieved chunks to Gemini LLM to generate a contextual answer.
* Returns both the **answer** and **source chunks** with metadata.

> **Important:** Always copy the `document_id` returned from `/upload` when querying, otherwise you may get results from unrelated documents.

---

## Rechunking Documents

**Endpoint:** `POST /rechunk/{document_id}`

* Allows updating or recomputing the chunks for a previously uploaded document.
* Deletes existing chunks in ChromaDB for the document.
* Re-extracts text from the original file stored in `RAG_DATA_DIR`.
* Re-chunks, re-embeds, and stores new chunks.
* Returns:

```json
{
  "document_id": "12345abcdef",
  "num_chunks": 43,
  "status": "rechunked"
}
```

> Copy the `document_id` from the original upload when calling rechunk.
> Useful if you update chunk size, overlap, or need to fix extraction errors.

---

## Get All Documents

**Endpoint:** `GET /getAllDocuments`

* Returns metadata for all uploaded documents:

```json
{
  "num_documents": 2,
  "documents": [
    {
      "document_id": "12345abcdef",
      "source": "Guide-to-Rice.pdf",
      "file_type": ".pdf",
      "num_chunks": 43
    },
    {
      "document_id": "67890ghijk",
      "source": "Guide-to-Beans.docx",
      "file_type": ".docx",
      "num_chunks": 37
    }
  ]
}
```

* Useful for listing all documents available for querying or rechunking.
* `num_chunks` indicates how many chunks exist in ChromaDB for that document.

---

## Environment Variables

Set in `.env`:

```
HF_API_KEY=your_huggingface_api_key
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
GEMINI_API_KEY=your_gemini_api_key
LLM_MODEL_NAME=gemini-2.5-flash
CHROMA_DB_HOST=http://localhost:8000
RAG_DATA_DIR=./rag_data
CHUNK_LENGTH=1000
TOP_K=5
PORT=8000
```

---

## Requirements

**requirements.txt**

```
fastapi==0.111.0
uvicorn==0.23.2
python-dotenv==1.0.0
sentence-transformers==2.2.2
chromadb==0.4.3
pydantic==2.5.1
requests==2.31.0
rich==13.5.1
python-docx==0.8.12
pdfminer.six==20221105
```

> Optional dependencies for HTML, PDF, and DOCX parsing are included.

---

## Usage Examples

### Upload a document

```bash
curl -X POST "http://localhost:8000/upload" \
-F "file=@Guide-to-Rice.pdf"
```

* Returns `document_id`. **Copy this for queries and rechunking.**

### Query a document

```bash
curl -X POST "http://localhost:8000/prompt" \
-H "Content-Type: application/json" \
-d '{
  "question": "Is rice farmed in Nigeria?",
  "top_k": 5,
  "document_id": "12345abcdef"
}'
```

### Rechunk a document

```bash
curl -X POST "http://localhost:8000/rechunk/12345abcdef"
```

* Deletes old chunks and reprocesses the document.

### List all documents

```bash
curl -X GET "http://localhost:8000/getAllDocuments"
```

---

## Notes

* Always use the `document_id` returned from `/upload` for consistent querying.
* Uploaded files are stored in `RAG_DATA_DIR` for persistent access.
* Rechunking does **not change the document_id**, only refreshes its chunks in ChromaDB.

---

This README gives a **full developer guide** for using your semantic-chunking RAG API, covering **uploads, querying, rechunking, and document management**.

---

