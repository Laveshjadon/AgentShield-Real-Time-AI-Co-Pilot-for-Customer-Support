"""Coordinate query building, retrieval, and LLM response generation."""

import time
import json
from config.settings import Settings
from config.logger import get_logger

from rag.query_builder import QueryBuilder
from rag.hybrid_retriever import retrieve_hybrid
from rag.prompt_builder import PromptBuilder
from rag.llm_client import LLMClient
from analysis.pii_service import pii_service

logger = get_logger("rag.generator")


query_builder = QueryBuilder()
prompt_builder = PromptBuilder()
llm_client = LLMClient()


def generate_answer(
    transcript: str,
    conversation_history: str = "",
    context: str | None = None,
    latest_utterance: str | None = None,
) -> str:
    """
    Run the RAG pipeline and return an agent suggestion.
    """
    t_start = time.perf_counter()
    metrics = {}
    
    
    t_redact = time.perf_counter()
    safe_transcript = pii_service.redact(transcript)
    safe_latest_utterance = pii_service.redact(latest_utterance or transcript)
    safe_history = pii_service.redact(conversation_history or transcript)
    metrics["pii_ms"] = (time.perf_counter() - t_redact) * 1000

    try:
        if context:
            retrieved_chunks = [
                {
                    "source_file": "retrieved_context",
                    "content": context,
                }
            ]
            metrics["query_builder_ms"] = 0
            metrics["bm25_ms"] = 0
            metrics["dense_ms"] = 0
            metrics["rrf_ms"] = 0
        else:
            
            t_qb = time.perf_counter()
            query = query_builder.build_query(safe_latest_utterance, safe_history)
            metrics["query_builder_ms"] = (time.perf_counter() - t_qb) * 1000
            
            
            retrieved_chunks, ret_metrics = retrieve_hybrid(query, top_dense=20, top_sparse=20, top_final=5)
            metrics.update(ret_metrics)
        
        
        t_pb = time.perf_counter()
        messages = prompt_builder.build_messages(
            transcript_chunk=safe_latest_utterance,
            retrieved_chunks=retrieved_chunks,
            conversation_history=safe_history
        )
        metrics["prompt_builder_ms"] = (time.perf_counter() - t_pb) * 1000
        
        
        response_obj, llm_metrics = llm_client.generate_suggestion(messages)
        metrics.update(llm_metrics)
        
        
        total_ms = (time.perf_counter() - t_start) * 1000
        metrics["total_pipeline_ms"] = total_ms
        
        
        logger.info(
            f"[RAG_PIPELINE] Latencies: "
            f"PII={metrics.get('pii_ms', 0):.0f}ms | "
            f"QB={metrics.get('query_builder_ms', 0):.0f}ms | "
            f"BM25={metrics.get('bm25_ms', 0):.0f}ms | "
            f"Dense={metrics.get('dense_ms', 0):.0f}ms | "
            f"RRF={metrics.get('rrf_ms', 0):.0f}ms | "
            f"Prompt={metrics.get('prompt_builder_ms', 0):.0f}ms | "
            f"LLM={metrics.get('llm_ms', 0):.0f}ms || "
            f"TOTAL={total_ms:.0f}ms"
        )
        
        
        formatted_suggestion = (
            f"{response_obj.suggestion}\n\n"
            f"🔹 Next Action: {response_obj.next_action}\n"
            f"📖 Reference: {response_obj.policy_reference} (Confidence: {response_obj.confidence:.2f})"
        )
        return formatted_suggestion
        
    except Exception as e:
        logger.error(f"[RAG_PIPELINE] Unhandled exception: {e}")
        return "Suggestion unavailable — please refer to knowledge base manually."

if __name__ == "__main__":
    
    test_transcript = (
        "Agent: Thank you for calling TechNova. How can I help?\n"
        "Customer: Yeah hi, I bought a router 10 days ago but it's completely dead. I want my money back!"
    )
    print("\nGenerating AI Suggestion...\n")
    suggestion = generate_answer(test_transcript)
    print("AGENT SUGGESTION:")
    print(suggestion)
