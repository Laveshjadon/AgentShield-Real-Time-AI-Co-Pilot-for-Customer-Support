"""
Load supported knowledge-base documents, split them into chunks, embed them,
and store them in PostgreSQL.

"""

import os
import glob
from sentence_transformers import SentenceTransformer
from src.core.db import get_session
from src.core.db import KnowledgeChunk
from config.settings import Settings
from config.logger import get_logger

from src.ingestion.parser import get_loader_for_file, DocumentValidationError

logger = get_logger("rag.document_loader")
settings = Settings()



_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _embedding_model


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks for embedding."""
    if len(text) <= chunk_size:
        return [text]
        
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
            
        
        break_point = text.rfind('\n', start, end)
        if break_point == -1 or break_point < start + (chunk_size // 2):
            break_point = text.rfind(' ', start, end)
            
        if break_point != -1 and break_point > start:
            end = break_point
            
        chunks.append(text[start:end].strip())
        start = end - chunk_overlap
        
    return chunks

def process_and_store_documents(directory: str = "data/knowledge_base"):
    """
    Load supported files, chunk them, embed them, and persist them.
    """
    if not os.path.exists(directory):
        logger.error(f"Directory {directory} does not exist!")
        return

    
    files = []
    for ext in ['*.txt', '*.md', '*.pdf', '*.docx', '*.html', '*.htm']:
        files.extend(glob.glob(os.path.join(directory, ext)))

    if not files:
        logger.warning(f"No supported files found in {directory}")
        return

    logger.info(f"Found {len(files)} files to process.")

    with get_session() as session:
        for file_path in files:
            filename = os.path.basename(file_path)
            logger.info(f"Processing {filename}...")

            
            try:
                loader = get_loader_for_file(file_path)
            except ValueError as e:
                logger.error(f"Skipping {filename}: {e}")
                continue

            
            try:
                doc_chunks = loader.load(file_path)
            except DocumentValidationError as e:
                logger.warning(f"Validation failed for {filename}: {e}")
                continue
            except Exception as e:
                logger.error(f"Failed to load {filename}: {e}")
                continue

            logger.info(f"  -> Extracted {len(doc_chunks)} logical sections/pages.")

            
            total_db_chunks = 0
            for doc_chunk in doc_chunks:
                
                sub_chunks = chunk_text(doc_chunk.text, settings.RAG_CHUNK_SIZE, settings.RAG_CHUNK_OVERLAP)
                
                for sub_text in sub_chunks:
                    
                    embedding_vector = _get_embedding_model().encode(sub_text).tolist()

                    
                    db_chunk = KnowledgeChunk(
                        content=sub_text,
                        source_file=doc_chunk.source_file,
                        document_type=doc_chunk.document_type,
                        page_number=doc_chunk.page_number,
                        chunk_index=total_db_chunks,
                        embedding=embedding_vector
                    )
                    session.add(db_chunk)
                    total_db_chunks += 1

            
            session.commit()
            logger.info(f"  -> Saved {total_db_chunks} vector chunks from {filename} to database.")

    logger.info("All documents processed successfully!")


if __name__ == "__main__":
    process_and_store_documents()
