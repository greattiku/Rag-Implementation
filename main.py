# main.py
import os
import json
import glob
import uuid
from typing import List, Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import requests
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from rich import print as rprint

# ---------- Load .env ----------
load_dotenv()  # loads from .env

print("EMBED_MODEL_NAME =", os.getenv("EMBED_MODEL_NAME"))

HF_API_KEY = os.getenv("HF_API_KEY", "")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")
CHROMA_DB_HOST = os.getenv("CHROMA_DB_HOST", "http://localhost:8000")
RAG_DATA_DIR = os.getenv("RAG_DATA_DIR", "./rag_data")
CHUNK_LENGTH = int(os.getenv("CHUNK_LENGTH", "1000"))  # approximate characters per chunk
PORT = int(os.getenv("PORT", "8000"))
TOP_K = int(os.getenv("TOP_K", "5"))

# Set huggingface token envs used by libraries
# SentenceTransformer/huggingface_hub expects HUGGINGFACEHUB_API_TOKEN in many setups
if HF_API_KEY:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_API_KEY
    os.environ["HF_API_KEY"] = HF_API_KEY

# ---------- FastAPI app ----------
app = FastAPI(title="Semantic-Chunking RAG (FastAPI)")

# ---------- Initialize components ----------
rprint("[bold]Initializing embedding model and Chroma client...[/bold]")

# Embedding model (SentenceTransformer)


try:
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
except Exception as e:
    embed_model = None
    rprint(f"[red]Failed to load embedding model: {e}[/red]")
    raise RuntimeError(f"Embedding model not available: {e}")

# ChromaDB client — using HTTP host (self-hosted Chroma server) if provided.
# If CHROMA_DB_HOST is left default "http://localhost:8000" we still attempt connect.
try:
    chroma_client = chromadb.Client(Settings(
        chroma_api_impl="rest",
        chroma_server_host=CHROMA_DB_HOST.replace("http://", "").replace("https://", "").split(":")[0],
        chroma_server_http_port=int(CHROMA_DB_HOST.split(":")[-1]) if ":" in CHROMA_DB_HOST else 8000
    ))
    rprint(f"Chroma client configured to host {CHROMA_DB_HOST}")
except Exception:
    # fallback to in-memory / local chroma if no host available
    rprint("[yellow]Falling back to in-process Chroma client (no remote host configured).[/yellow]")
    chroma_client = chromadb.Client()

# Create or get a collection
COLLECTION_NAME = "rag_collection"
try:
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
except Exception:
    collection = chroma_client.create_collection(name=COLLECTION_NAME)
rprint(f"Using chroma collection: {COLLECTION_NAME}")

# ---------- Utility functions ----------



def extract_text_from_file(path: Path, raw_bytes: bytes | None = None) -> str:
    suffix = path.suffix.lower()

    try:
        # TXT / MD
        if suffix in [".txt", ".md"]:
            return (
                raw_bytes.decode("utf-8", errors="ignore")
                if raw_bytes
                else path.read_text(encoding="utf-8", errors="ignore")
            )

        # PDF
        if suffix == ".pdf":
            from pdfminer.high_level import extract_text
            return extract_text(path)

        # DOCX
        if suffix == ".docx":
            from docx import Document
            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs)

        # HTML
        if suffix in [".html", ".htm"]:
            from bs4 import BeautifulSoup
            html = (
                raw_bytes.decode("utf-8", errors="ignore")
                if raw_bytes
                else path.read_text(encoding="utf-8", errors="ignore")
            )
            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text(separator="\n")

    except Exception as e:
        rprint(f"[red]Text extraction failed for {path.name}: {e}[/red]")

    return ""






def semantic_chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Basic semantic chunker:
    - Splits text by sentences/linebreaks and accumulates into chunks roughly chunk_size characters long
    - Adds small overlap to preserve context
    """
    if not text:
        return []
    # Normalize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = text.split("\n")
    chunks = []
    current = ""
    for part in parts:
        if not part.strip():
            # preserve paragraph break as single newline
            if len(current) > 0 and len(current) + 1 <= chunk_size:
                current += "\n"
            continue
        candidate = (current + "\n" + part).strip() if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            # if current is non-empty, flush current as chunk
            if current:
                chunks.append(current.strip())
            # If part itself is longer than chunk_size, break it directly
            if len(part) > chunk_size:
                # split long paragraph into fixed-size windows with overlap
                start = 0
                while start < len(part):
                    end = min(start + chunk_size, len(part))
                    chunks.append(part[start:end].strip())
                    start = end - overlap if end < len(part) else end
                current = ""
            else:
                current = part
    if current:
        chunks.append(current.strip())

    # apply overlap merging to add preceding overlap characters to chunk metadata if desired
    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    if embed_model is None:
        raise RuntimeError("Embedding model is not initialized")
    if not texts:
        return []
    embeddings = embed_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return [emb.tolist() for emb in embeddings]

# # Gemini / Google Generative API wrapper


def call_gemini_generate(
    prompt: str,
    model_name: str = LLM_MODEL_NAME,
    api_key: str = GEMINI_API_KEY,
    max_tokens: int = 512
) -> Dict[str, Any]:

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    url = (
        f"https://generativelanguage.googleapis.com/v1/models/"
        f"{model_name}:generateContent"
        f"?key={api_key}"
    )

    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.2
        }
    }

    resp = requests.post(url, json=body, timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Gemini API error {resp.status_code}: {resp.text}"
        )

    return resp.json()


# ---------- API Models ----------

class IngestResult(BaseModel):
    file: str
    num_chunks: int
    status: str
    document_id: str  # 🔑 ADD THIS



class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    document_id: str  # 🔑 REQUIRED


class QueryResult(BaseModel):
    answer: str
    source_chunks: List[Dict[str, Any]]
    metadata: Dict[str, Any]

# ---------- Endpoints ----------

@app.get("/status")
def status():
    return {
        "status": "ok",
        "embed_model": EMBED_MODEL_NAME,
        "chroma_host": CHROMA_DB_HOST,
        "collection": COLLECTION_NAME
    }


@app.get("/getAllDocuments")
def get_status():
    """
    Return all uploaded documents and basic info
    """
    try:
        # Pull only metadata (cheap)
        data = collection.get(include=["metadatas"])

        docs = {}

        for md in data.get("metadatas", []):
            doc_id = md.get("document_id")
            if not doc_id:
                continue

            if doc_id not in docs:
                docs[doc_id] = {
                    "document_id": doc_id,
                    "source": md.get("source"),
                    "file_type": md.get("file_type"),
                    "num_chunks": 0
                }

            docs[doc_id]["num_chunks"] += 1

        return {
            "num_documents": len(docs),
            "documents": list(docs.values())
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





# @app.post("/ingest", response_model=List[IngestResult])
def ingest_all_files():
    data_dir = Path(RAG_DATA_DIR)
    if not data_dir.exists():
        raise HTTPException(status_code=400, detail="RAG_DATA_DIR not found")

    results = []
    files = list(data_dir.glob("*"))

    if not files:
        raise HTTPException(status_code=404, detail="No files found")

    for file_path in files:
        try:
            text = extract_text_from_file(file_path)

            if not text.strip():
                results.append(IngestResult(
                    file=file_path.name,
                    num_chunks=0,
                    status="skipped (no text)",
                ))
                continue

            chunks = semantic_chunk_text(
                text,
                chunk_size=CHUNK_LENGTH,
                overlap=max(100, int(CHUNK_LENGTH * 0.1)),
            )

            embeddings = embed_texts(chunks)

            ids, metadatas, documents = [], [], []

            for i, chunk in enumerate(chunks):
                cid = f"{file_path.name}_chunk_{i}_{uuid.uuid4().hex[:8]}"
                ids.append(cid)
                documents.append(chunk)
                metadatas.append({
                    "source": file_path.name,
                    "chunk_index": i,
                    "file_type": file_path.suffix.lower(),
                })

            collection.add(
                documents=documents,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
            )

            results.append(IngestResult(
                file=file_path.name,
                num_chunks=len(chunks),
                status="ingested",
            ))

        except Exception as e:
            results.append(IngestResult(
                file=file_path.name,
                num_chunks=0,
                status=f"error: {e}",
            ))

    return results




@app.post("/upload", response_model=IngestResult)
async def upload_file(file: UploadFile = File(...)):
    try:
        os.makedirs(RAG_DATA_DIR, exist_ok=True)

        document_id = uuid.uuid4().hex  # 🔑 unique per document
        out_path = Path(RAG_DATA_DIR) / f"{document_id}_{file.filename}"

        contents = await file.read()
        out_path.write_bytes(contents)

        # ---------- TEXT EXTRACTION (updated to use helper) ----------
        text = extract_text_from_file(out_path, contents)

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="No extractable text found in document"
            )

        # ---------- CHUNK + EMBED ----------
        chunks = semantic_chunk_text(
            text,
            CHUNK_LENGTH,
            overlap=max(100, int(CHUNK_LENGTH * 0.1))
        )

        embeddings = embed_texts(chunks)

        ids, metadatas, documents = [], [], []

        for i, chunk in enumerate(chunks):
            ids.append(f"{document_id}_{i}")
            documents.append(chunk)
            metadatas.append({
                "document_id": document_id,   # ✅ This enables filtering
                "source": file.filename,
                "chunk_index": i,
                "file_type": out_path.suffix.lower()
            })

        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

        return IngestResult(
            file=file.filename,
            num_chunks=len(chunks),
            status="ingested",
            document_id=document_id  # return to client
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



    
@app.post("/prompt", response_model=QueryResult)
def query_rag(req: QueryRequest):
    """
    Query RAG:
    - Embed the question
    - Retrieve top_k chunks from Chroma
    - Compose a prompt with retrieved chunks and the user question
    - Call Gemini to generate an answer
    - Return the answer + source chunks
    """
    q = req.question
    top_k = req.top_k or TOP_K
    if top_k < 1:
        top_k = 1  # ensure at least 1 result
    if not q:
        raise HTTPException(status_code=400, detail="question required")

    # 1) Embed the question
    try:
        q_emb = embed_texts([q])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed query: {e}")

    # 2) Retrieve top_k chunks from Chroma
    try:
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where={"document_id": req.document_id},  # ✅ FILTER
            include=["documents", "metadatas", "distances"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chroma query failed: {e}")

    # 3) Process Chroma results
    docs = []
    doc_texts = results["documents"][0] if results.get("documents") else []
    metadatas_list = results["metadatas"][0] if results.get("metadatas") else [{}]*len(doc_texts)
    distances = results["distances"][0] if results.get("distances") else [None]*len(doc_texts)

    for i, dt in enumerate(doc_texts):
        md = metadatas_list[i] if i < len(metadatas_list) else {}
        # Ensure metadata is a dict
        if isinstance(md, list):
            md = md[0] if md else {}
        docs.append({
            "id": f"doc_{i}",
            "text": dt,
            "metadata": md,
            "distance": distances[i] if i < len(distances) else None
        })

    # 4) Compose prompt for Gemini
    context_blocks = []
    for i, d in enumerate(docs):
        context_blocks.append(
            f"Source [{i}] (source={d['metadata'].get('source', 'unknown')}, idx={d['metadata'].get('chunk_index', 'n/a')}):\n{d['text']}\n"
        )

    context_text = "\n\n---\n\n".join(context_blocks) if context_blocks else ""
    prompt = f"""You are an assistant that answers user questions using the provided context. Use the context to answer the question. If the answer is not in the context, say 'I don't know'.

Context:
{context_text}

User question:
{q}

Answer concisely and cite sources by index (e.g., [Source 0]) where useful.
"""

    # 5) Call Gemini to generate an answer
    try:
        gen_resp = call_gemini_generate(
            prompt=prompt,
            model_name=LLM_MODEL_NAME,
            api_key=GEMINI_API_KEY,
            max_tokens=512
        )
        # Extract text from common response patterns
   
        answer_text = "No answer returned from LLM."
        if isinstance(gen_resp, dict):
            candidates = gen_resp.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                texts = [p.get("text", "") for p in parts if "text" in p]
                if texts:
                    answer_text = "\n".join(texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")


    return QueryResult(
        answer=answer_text,
        source_chunks=docs,
        metadata={"model_used": LLM_MODEL_NAME, "num_retrieved": len(docs)}
    )


@app.post("/rechunk/{document_id}")
def rechunk_document(document_id: str):
    """
    Re-chunk and re-embed a previously uploaded document
    """
    try:
        # 1️⃣ Delete old chunks
        collection.delete(where={"document_id": document_id})

        # 2️⃣ Find the original file
        files = list(Path(RAG_DATA_DIR).glob(f"{document_id}_*"))
        if not files:
            raise HTTPException(404, "Original file not found")

        file_path = files[0]
        suffix = file_path.suffix.lower()

        # 3️⃣ Re-extract text (same logic as upload)
        text = extract_text_from_file(file_path)

        # 4️⃣ Re-chunk (you can tweak params here)
        chunks = semantic_chunk_text(
            text,
            CHUNK_LENGTH,
            overlap=max(100, int(CHUNK_LENGTH * 0.1))
        )

        embeddings = embed_texts(chunks)

        # 5️⃣ Store again
        ids, docs, mds = [], [], []

        for i, chunk in enumerate(chunks):
            ids.append(f"{document_id}_{i}")
            docs.append(chunk)
            mds.append({
                "document_id": document_id,
                "source": file_path.name.split("_", 1)[1],
                "chunk_index": i,
                "file_type": suffix
            })

        collection.add(
            documents=docs,
            embeddings=embeddings,
            ids=ids,
            metadatas=mds
        )

        return {
            "document_id": document_id,
            "num_chunks": len(chunks),
            "status": "rechunked"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
