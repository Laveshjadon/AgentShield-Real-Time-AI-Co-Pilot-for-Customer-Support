# --- FROM: retrieval/schemas.py ---
"""
RAG Schemas

Pydantic models for structured LLM generation and retrieval.
"""

from pydantic import BaseModel, Field


class AgentSuggestionResponse(BaseModel):
    """Structured response expected from the LLM support copilot."""
    
    suggestion: str = Field(
        ..., 
        description="The exact words the agent should say to the customer. Should be concise and polite."
    )
    policy_reference: str = Field(
        ..., 
        description="A direct quote or summary from the company policy that justifies the suggestion. If no policy matches, state 'None'."
    )
    confidence: float = Field(
        ..., 
        description="A confidence score between 0.0 and 1.0 indicating how certain the LLM is that the suggestion is correct based on the retrieved context."
    )
    next_action: str = Field(
        ..., 
        description="A short recommended next step for the agent (e.g., 'Process refund', 'Escalate to Tier 2', 'Ask for serial number')."
    )


# --- FROM: retrieval/rrf.py ---
"""
rrf scoring script
this fuses the lists from bm25 and the vector search together.
googled the formula for this: score = sum(1 / (k + rank)).
"""

from typing import List, Dict, Any, Tuple
from config.logger import get_logger

logger = get_logger("rag.rrf")


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]], 
    sparse_results: List[Dict[str, Any]], 
    k: int = 60,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    does the actual fusion math on the two lists.
    needs the chunks to have a source_file and chunk_index so i can tell what's what.
    returns the top ones after combining.
    """
    rrf_scores: Dict[Tuple[str, int], float] = {}
    chunk_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
    
    
    def _process_ranked_list(ranked_list: List[Dict[str, Any]]):
        for rank, chunk in enumerate(ranked_list, start=1):
            unique_id = (chunk["source_file"], chunk["chunk_index"])
            
            if unique_id not in chunk_map:
                
                chunk_map[unique_id] = chunk
                rrf_scores[unique_id] = 0.0
                
            rrf_scores[unique_id] += 1.0 / (k + rank)

    
    _process_ranked_list(dense_results)
    _process_ranked_list(sparse_results)
    
    
    sorted_fused = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    
    
    final_results = []
    for unique_id, score in sorted_fused[:top_n]:
        chunk = chunk_map[unique_id]
        chunk["rrf_score"] = score
        final_results.append(chunk)
        
    logger.debug(f"[RRF] Fused {len(rrf_scores)} unique chunks down to top {top_n}")
    return final_results


# --- FROM: retrieval/retriever.py ---
"""
old dense-only retriever. keeping it around as a baseline to compare against
the hybrid pipeline, and as a fallback if something breaks.
"""

import glob
import os
import re

from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from rank_bm25 import BM25Okapi

from src.core.db import get_session
from config.settings import Settings
from config.logger import get_logger

logger = get_logger("rag.retriever")
settings = Settings()


_legacy_embedding_model = None


def _get_legacy_embedding_model():
    global _legacy_embedding_model
    if _legacy_embedding_model is None:
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        _legacy_embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _legacy_embedding_model


def _tokenize(text_value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text_value.lower())


def _chunk_text(text_value: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    text_value = re.sub(r"\s+", " ", text_value).strip()
    if not text_value:
        return []
    chunks = []
    start = 0
    while start < len(text_value):
        end = min(start + chunk_size, len(text_value))
        chunks.append(text_value[start:end].strip())
        if end == len(text_value):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _retrieve_from_local_files(query: str, top_k: int) -> str:
    """if postgres isn't running, just read the files directly and do bm25 on them locally. good enough for demos."""
    kb_dir = os.path.join("data", "knowledge_base")
    files = []
    for pattern in ("*.txt", "*.md"):
        files.extend(glob.glob(os.path.join(kb_dir, pattern)))
    documents = []

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                text_value = handle.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as handle:
                text_value = handle.read()

        for chunk_index, chunk in enumerate(_chunk_text(text_value)):
            documents.append(
                {
                    "source_file": os.path.basename(file_path),
                    "chunk_index": chunk_index,
                    "content": chunk,
                }
            )

    if not documents:
        logger.warning("[RETRIEVER] Local KB fallback found no .txt or .md documents.")
        return ""

    tokenized_docs = [_tokenize(doc["content"]) for doc in documents]
    query_tokens = _tokenize(query)
    if not query_tokens:
        return ""

    index = BM25Okapi(tokenized_docs)
    scores = index.get_scores(query_tokens)
    boosted_scores = []
    for idx, score in enumerate(scores):
        doc = documents[idx]
        source_tokens = set(_tokenize(doc["source_file"]))
        content_tokens = _tokenize(doc["content"])
        unique_content_tokens = set(content_tokens)
        source_overlap = len(source_tokens.intersection(query_tokens))
        content_overlap = len(unique_content_tokens.intersection(query_tokens))
        boosted_scores.append(
            score
            + (source_overlap * 6.0)
            + (content_overlap * 0.15)
        )

    ranked_indices = sorted(
        range(len(boosted_scores)),
        key=lambda i: boosted_scores[i],
        reverse=True,
    )

    context_pieces = []
    for idx in ranked_indices[:top_k]:
        if boosted_scores[idx] <= 0:
            continue
        doc = documents[idx]
        context_pieces.append(
            f"[Source: {doc['source_file']}]\n{doc['content']}"
        )

    if not context_pieces:
        logger.info("[RETRIEVER] Local KB fallback found no lexical match for '%s'.", query)
        return ""

    logger.warning(
        "[RETRIEVER] PostgreSQL unavailable; using local KB fallback with %d chunks.",
        len(context_pieces),
    )
    return "\n\n---\n\n".join(context_pieces)


def retrieve_context(query: str, top_k: int = None) -> str:
    """
    queries the db to find similar stuff and returns it all as one big string.
    """
    if top_k is None:
        top_k = settings.RAG_TOP_K

    logger.info(f"Searching knowledge base for: '{query}'")

    
    query_vector = _get_legacy_embedding_model().encode(query).tolist()

    
    
    sql_query = text(
        """
        SELECT content, source_file, 1 - (embedding <=> :vector) AS similarity
        FROM knowledge_chunks
        ORDER BY embedding <=> :vector
        LIMIT :top_k
        """
    )

    try:
        with get_session() as session:
            
            vector_str = "[" + ",".join(map(str, query_vector)) + "]"
            
            result = session.execute(
                sql_query, 
                {"vector": vector_str, "top_k": top_k}
            ).fetchall()
    except Exception as exc:
        logger.error("[RETRIEVER] PostgreSQL retrieval failed: %s", exc)
        return _retrieve_from_local_files(query, top_k)

    if not result:
        logger.warning(f"[RETRIEVER] No chunks found in DB for query: '{query}'")
        return ""

    
    context_pieces = []
    low_score_count = 0
    for row in result:
        content = row[0]
        source = row[1]
        similarity = row[2]

        
        if similarity > 0.3:
            context_pieces.append(f"[Source: {source}]\n{content}")
            logger.debug(f"Match found in {source} (score: {similarity:.2f})")
        else:
            low_score_count += 1
            logger.debug(f"Chunk skipped — low similarity {similarity:.2f} in {source}")

    if not context_pieces:
        
        
        top_score = result[0][2] if result else 0
        logger.warning(
            f"[RETRIEVER] Silent failure: {len(result)} chunks found but ALL below "
            f"similarity threshold (0.3). Best score={top_score:.3f}. "
            f"Query: '{query}'. "
            f"LLM will be skipped — agent gets no suggestion."
        )
        return ""

    final_context = "\n\n---\n\n".join(context_pieces)
    logger.info(f"[RETRIEVER] Returning {len(context_pieces)} chunks (filtered {low_score_count} low-score).")
    return final_context


if __name__ == "__main__":
    
    print("\n--- Testing RAG Retriever ---")
    
    q1 = "How long do I have to get a refund?"
    print(f"\nQuery: {q1}")
    print("Result:\n" + retrieve_context(q1))
    
    q2 = "My internet is broken and the light is blinking red."
    print(f"\nQuery: {q2}")
    print("Result:\n" + retrieve_context(q2))


# --- FROM: retrieval/hybrid_retriever.py ---
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

from src.core.db import get_session
from src.core.db import KnowledgeChunk
from config.settings import Settings
from config.logger import get_logger

logger = get_logger("rag.hybrid_retriever")
settings = Settings()


_embedding_model = None
_bm25_index = None
_corpus_chunks = []


def _hybrid_tokenize(text_value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text_value.lower())


def _source_text(source_file: str) -> str:
    stem = os.path.splitext(os.path.basename(source_file or ""))[0]
    return stem.replace("_", " ").replace("-", " ")


def _expanded_sparse_text(chunk: Dict[str, Any]) -> str:
    source = _source_text(chunk.get("source_file", ""))
    content = chunk.get("content", "")
    # Repeat source/title terms so document identity can compete with generic FAQ text.
    return " ".join([source] * 4 + [content])


def _source_specificity(source_file: str) -> float:
    source = (source_file or "").lower()
    if source.endswith(".docx"):
        return 0.85
    if re.match(r"^\d+_", os.path.basename(source)):
        return 1.15
    return 1.0


def _query_source_affinity(query_terms: set[str], source_file: str) -> float:
    """
    boosts chunks where the source filename actually overlaps with what the user asked.
    e.g. if they ask about "refund" and the file is called refund_policy.txt, that chunk
    should rank higher. each matching term adds 0.10, capped at 1.30 so it doesn't go crazy.
    no hardcoded filenames — works for any kb.
    """
    if not query_terms or not source_file:
        return 1.0
    source_terms = set(_hybrid_tokenize(_source_text(source_file)))
    overlap = len(query_terms.intersection(source_terms))
    return min(1.0 + overlap * 0.10, 1.30)


def _init_dense_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        t0 = time.perf_counter()
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info(f"Dense model loaded in {(time.perf_counter() - t0) * 1000:.0f}ms")


def _init_bm25_index():
    """pulls all chunks from the db and builds the bm25 index in memory. only runs once on first call."""
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
    
    
    tokenized_corpus = [_hybrid_tokenize(_expanded_sparse_text(chunk)) for chunk in _corpus_chunks]
    
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
    query_terms: set = set()  # initialised here so fusion loop always has it
    t_sparse = time.perf_counter()
    if _bm25_index and _corpus_chunks:
        tokenized_query = _hybrid_tokenize(query)
        scores = _bm25_index.get_scores(tokenized_query)
        query_terms = set(tokenized_query)
        boosted_scores = []
        for idx, score in enumerate(scores):
            source_terms = set(_hybrid_tokenize(_source_text(_corpus_chunks[idx].get("source_file", ""))))
            source_overlap = len(query_terms.intersection(source_terms))
            boosted_scores.append(float(score) + (source_overlap * 6.0))
        
        
        top_indices = sorted(range(len(boosted_scores)), key=lambda i: boosted_scores[i], reverse=True)[:top_sparse]
        
        for idx in top_indices:
            if boosted_scores[idx] > 0:
                chunk = dict(_corpus_chunks[idx])
                chunk["bm25_score"] = boosted_scores[idx]
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
    merged: Dict[tuple[str, int], Dict[str, Any]] = {}
    max_bm25 = max((chunk.get("bm25_score", 0.0) for chunk in sparse_candidates), default=0.0) or 1.0
    max_similarity = max((chunk.get("similarity", 0.0) for chunk in dense_candidates), default=0.0) or 1.0

    for rank, chunk in enumerate(sparse_candidates, start=1):
        key = (chunk["source_file"], chunk["chunk_index"])
        entry = merged.setdefault(key, dict(chunk))
        entry["bm25_score"] = chunk.get("bm25_score", 0.0)
        entry["sparse_rank"] = rank

    for rank, chunk in enumerate(dense_candidates, start=1):
        key = (chunk["source_file"], chunk["chunk_index"])
        entry = merged.setdefault(key, dict(chunk))
        entry["similarity"] = chunk.get("similarity", 0.0)
        entry["dense_rank"] = rank

    for entry in merged.values():
        bm25_norm = max(float(entry.get("bm25_score", 0.0)), 0.0) / max_bm25
        dense_norm = max(float(entry.get("similarity", 0.0)), 0.0) / max_similarity
        rank_bonus = 0.0
        if entry.get("sparse_rank") == 1:
            rank_bonus += 0.15
        if entry.get("dense_rank") == 1:
            rank_bonus += 0.05
        entry["hybrid_score"] = (
            ((0.7 * bm25_norm) + (0.3 * dense_norm) + rank_bonus)
            * _source_specificity(entry.get("source_file", ""))
            * _query_source_affinity(query_terms, entry.get("source_file", ""))
        )

    final_chunks = sorted(
        merged.values(),
        key=lambda chunk: chunk.get("hybrid_score", 0.0),
        reverse=True,
    )[:top_final]
    metrics["fusion_ms"] = (time.perf_counter() - t_rrf) * 1000
    
    return final_chunks, metrics


