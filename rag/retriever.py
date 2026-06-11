"""Legacy pgvector retriever retained for fallback and comparison."""

import glob
import os
import re

from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from rank_bm25 import BM25Okapi

from db.connection import get_session
from config.settings import Settings
from config.logger import get_logger

logger = get_logger("rag.retriever")
settings = Settings()


logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)


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
    """Fallback retrieval for local demos when PostgreSQL is unavailable."""
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

    
    query_vector = embedding_model.encode(query).tolist()

    
    
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
