"""
Knowledge Base API Routes
stuff for uploading docs and doing RAG search on them.
"""

import os
import shutil
from fastapi import APIRouter, HTTPException, UploadFile, File, Request
from pydantic import BaseModel, Field
from typing import Optional

from api.limiter import limiter  
from rag.retriever import retrieve_context
from rag.document_loader import process_and_store_documents
from config.logger import get_logger

logger = get_logger("api.knowledge")
router = APIRouter(prefix="/api/knowledge", tags=["Knowledge Base"])

UPLOAD_DIR = "data/knowledge_base"
os.makedirs(UPLOAD_DIR, exist_ok=True)


ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


class SearchRequest(BaseModel):  
    query: str = Field(..., min_length=1, max_length=500)
    top_k: Optional[int] = Field(default=3, ge=1, le=10)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    upload a doc to the KB.
    it automatically chunks it up and throws it into pgvector.
    """
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Accepted: .txt, .md, .pdf, .docx"
        )

    save_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        logger.info(f"File uploaded: {file.filename}")

        
        process_and_store_documents(UPLOAD_DIR)

        return {
            "status": "success",
            "filename": file.filename,
            "message": f"Document '{file.filename}' uploaded and indexed successfully."
        }
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
@limiter.limit("60/minute")  
async def search_knowledge(request: Request, req: SearchRequest):
    """does a search in pgvector using the query string."""
    
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        context = retrieve_context(req.query, top_k=req.top_k)
        return {
            "query": req.query,
            "results": context if context else "No relevant documents found.",
            "found": bool(context)
        }
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def list_documents():
    """lists all the docs we have so far"""
    
    files = [
        f for f in os.listdir(UPLOAD_DIR)
        if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS
    ]
    return {
        "documents": files,
        "total": len(files),
        "directory": UPLOAD_DIR
    }
