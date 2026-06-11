"""Normalize transcript text into a compact retrieval query."""

import spacy
from typing import List, Optional
from config.logger import get_logger

logger = get_logger("rag.query_builder")


try:
    nlp = spacy.load("en_core_web_lg", disable=["parser", "ner", "textcat"])
except OSError:
    logger.warning("Spacy model 'en_core_web_lg' not found. Query builder will fall back to basic splitting.")
    nlp = None


TERMINOLOGY_MAP = {
    "money back": "refund",
    "broke": "defective",
    "broken": "defective",
    "dead": "defective",
    "doesn't work": "defective",
    "replace": "replacement",
    "send back": "return",
    "swap": "exchange",
    "fix": "repair",
}

class QueryBuilder:
    """Convert transcript text into a search query."""

    def __init__(self):
        pass

    def build_query(self, transcript_chunk: str, conversation_history: Optional[str] = None) -> str:
        """
        Normalize the latest customer text into searchable keywords.
        """
        
        text_to_process = transcript_chunk.lower()
        
        
        for raw_term, normalized_term in TERMINOLOGY_MAP.items():
            if raw_term in text_to_process:
                text_to_process = text_to_process.replace(raw_term, normalized_term)

        if not nlp:
            
            return text_to_process

        
        doc = nlp(text_to_process)
        
        keywords: List[str] = []
        for token in doc:
            if token.is_stop or token.is_punct or len(token.text) < 3:
                continue
            
            
            if token.pos_ in {"NOUN", "PROPN", "VERB", "ADJ"}:
                keywords.append(token.lemma_)
                
        if not keywords:
            return transcript_chunk
            
        optimized_query = " ".join(set(keywords))
        
        logger.debug(f"[QueryBuilder] Original: {transcript_chunk.strip()}")
        logger.debug(f"[QueryBuilder] Optimized: {optimized_query}")
        
        return optimized_query
