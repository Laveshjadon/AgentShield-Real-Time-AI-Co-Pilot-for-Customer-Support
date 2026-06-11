"""
hybrid retriever code
combines pgvector and bm25 search together
and merges the results using RRF.
"""

import time
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from rank_bm25 import BM25Okapi

from db.connection import get_session
from db.models import KnowledgeChunk
from config.settings import Settings
from config.logger import get_logger
from rag.rrf import reciprocal_rank_fusion

logger = get_logger("rag.hybrid_retriever")
settings = Settings()


_embedding_model = None
_bm25_index = None
_corpus_chunks = []


def _init_dense_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        t0 = time.perf_counter()
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info(f"Dense model loaded in {(time.perf_counter() - t0) * 1000:.0f}ms")


def _init_bm25_index():
    """Load knowledge chunks from the database and build the BM25 index in memory."""
    global _bm25_index, _corpus_chunks
    if _bm25_index is not None:
        return

    logger.info("Initializing in-memory BM25 index from database...")
    t0 = time.perf_counter()
    
    with get_session() as session:
        records = session.query(KnowledgeChunk).all()
        
    _corpus_chunks = [
        {
            "chunk_index": r.chunk_index,
            "source_file": r.source_file,
            "content": r.content
        } for r in records
    ]
    
    
    tokenized_corpus = [chunk["content"].lower().split() for chunk in _corpus_chunks]
    
    if tokenized_corpus:
        _bm25_index = BM25Okapi(tokenized_corpus)
    else:
        _bm25_index = None
        
    logger.info(f"BM25 index built with {len(_corpus_chunks)} chunks in {(time.perf_counter() - t0) * 1000:.0f}ms")


def retrieve_hybrid(query: str, top_dense: int = 20, top_sparse: int = 20, top_final: int = 5) -> tuple[List[Dict[str, Any]], dict]:
    """
    does the actual hybrid search and ranks everything.
    returns the chunks and also how long each part took.
    """
    _init_dense_model()
    _init_bm25_index()
    
    metrics = {}
    
    
    sparse_candidates = []
    t_sparse = time.perf_counter()
    if _bm25_index and _corpus_chunks:
        tokenized_query = query.lower().split()
        scores = _bm25_index.get_scores(tokenized_query)
        
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_sparse]
        
        for idx in top_indices:
            if scores[idx] > 0:
                chunk = dict(_corpus_chunks[idx])
                chunk["bm25_score"] = scores[idx]
                sparse_candidates.append(chunk)
                
    metrics["bm25_ms"] = (time.perf_counter() - t_sparse) * 1000


    
    dense_candidates = []
    t_dense = time.perf_counter()
    query_vector = _embedding_model.encode(query).tolist()
    
    sql_query = text(
        """
        SELECT content, source_file, chunk_index, 1 - (embedding <=> :vector) AS similarity
        FROM knowledge_chunks
        ORDER BY embedding <=> :vector
        LIMIT :top_dense
        """
    )
    
    with get_session() as session:
        vector_str = "[" + ",".join(map(str, query_vector)) + "]"
        result = session.execute(sql_query, {"vector": vector_str, "top_dense": top_dense}).fetchall()
        
        for row in result:
            dense_candidates.append({
                "content": row[0],
                "source_file": row[1],
                "chunk_index": row[2],
                "similarity": row[3]
            })
            
    metrics["dense_ms"] = (time.perf_counter() - t_dense) * 1000


    
    t_rrf = time.perf_counter()
    final_chunks = reciprocal_rank_fusion(
        dense_results=dense_candidates,
        sparse_results=sparse_candidates,
        k=60,
        top_n=top_final
    )
    metrics["rrf_ms"] = (time.perf_counter() - t_rrf) * 1000
    
    return final_chunks, metrics
