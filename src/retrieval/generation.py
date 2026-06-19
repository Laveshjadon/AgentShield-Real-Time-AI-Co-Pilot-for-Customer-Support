# --- FROM: retrieval/llm_client.py ---
"""
my llm client wrapper
handles both openai and groq APIs so we can switch easily.
added some retries and timeouts because the api keeps flaking on me.
"""

import time
import json
import logging
from typing import Dict, Any, Optional

from openai import OpenAI
from groq import Groq

from config.settings import Settings
from config.logger import get_logger
from src.retrieval.hybrid import AgentSuggestionResponse

logger = get_logger("rag.llm_client")
settings = Settings()

class LLMClient:
    """wrapper to get json back from the llm"""

    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        self.model = settings.LLM_MODEL
        self.client = None
        
        try:
            if self.provider == "openai":
                if not settings.OPENAI_API_KEY:
                    logger.error("OPENAI_API_KEY is not set.")
                else:
                    self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
                    
            elif self.provider == "groq":
                if not settings.GROQ_API_KEY:
                    logger.error("GROQ_API_KEY is not set.")
                else:
                    self.client = Groq(api_key=settings.GROQ_API_KEY)
            else:
                logger.error(f"Unsupported provider: {self.provider}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM Client: {e}")

    def generate_suggestion(
        self, 
        messages: list[Dict[str, str]], 
        retries: int = 2,
        timeout: float = 3.0
    ) -> tuple[AgentSuggestionResponse, dict]:
        """
        hits the api and retries if it fails.
        also times it so we know if it's lagging.
        """
        metrics = {}
        fallback = AgentSuggestionResponse(
            suggestion="Suggestion unavailable — please refer to knowledge base manually.",
            policy_reference="None",
            confidence=0.0,
            next_action="Manual review required"
        )
        
        if not self.client:
            logger.error("[LLMClient] Client is not initialized.")
            return fallback, metrics

        attempt = 0
        while attempt <= retries:
            t0 = time.perf_counter()
            try:
                
                
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    response_format={"type": "json_object"},
                    timeout=timeout
                )
                
                raw_json = completion.choices[0].message.content
                metrics["llm_ms"] = (time.perf_counter() - t0) * 1000
                
                
                parsed_response = AgentSuggestionResponse.model_validate_json(raw_json)
                return parsed_response, metrics
                
            except Exception as e:
                attempt += 1
                latency = (time.perf_counter() - t0) * 1000
                logger.warning(
                    f"[LLMClient] Attempt {attempt} failed | latency={latency:.0f}ms | error={type(e).__name__}: {e}"
                )
                if attempt > retries:
                    logger.error("[LLMClient] All retries exhausted. Returning fallback.")
                    metrics["llm_ms"] = latency
                    return fallback, metrics
                time.sleep(0.5) 

        return fallback, metrics


# --- FROM: retrieval/query_builder.py ---
"""Normalize transcript text into a compact retrieval query."""

import spacy
from typing import List, Optional
from config.logger import get_logger

logger = get_logger("rag.query_builder")


nlp = None
_nlp_loaded = False


def _get_nlp():
    global nlp, _nlp_loaded
    if not _nlp_loaded:
        try:
            nlp = spacy.load("en_core_web_lg", disable=["parser", "ner", "textcat"])
        except OSError:
            logger.warning(
                "Spacy model 'en_core_web_lg' not found. Query builder will fall back to basic splitting."
            )
            nlp = None
        _nlp_loaded = True
    return nlp


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

        nlp_model = _get_nlp()
        if not nlp_model:
            
            return text_to_process

        
        doc = nlp_model(text_to_process)
        
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


# --- FROM: retrieval/prompt_builder.py ---
"""Build grounded prompts with deterministic response-language selection."""

import re
from typing import List, Dict, Any


DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
HINGLISH_MARKERS = {
    "aap", "abhi", "account se", "bahut", "batao", "chahiye", "gaya",
    "gayi", "hai", "hain", "hoon", "karunga", "karungi", "kat gaye",
    "kat gaya", "kya", "kyun", "main", "mera", "mere", "meri", "mujhe",
    "nahi", "paise", "paisa", "raha", "rahi", "refund do", "tumhari",
}


def detect_response_language(text: str) -> tuple[str, str]:
    """Return a stable language label and output instruction."""
    if DEVANAGARI_RE.search(text):
        return (
            "Hindi (Devanagari)",
            "Write the entire suggestion and next_action in Hindi using Devanagari script.",
        )

    normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    marker_hits = sum(marker in normalized for marker in HINGLISH_MARKERS)
    if marker_hits >= 2:
        return (
            "Hinglish (Roman script)",
            "Write the entire suggestion and next_action in natural Hinglish using only the Latin alphabet.",
        )

    return (
        "English",
        "Write the entire suggestion and next_action in English only. Do not use Hindi or Devanagari.",
    )


class PromptBuilder:
    """helper to make the final prompt string"""

    def __init__(self):
        
        
        self.system_prompt_template = """You are an AI support copilot assisting a live call center agent.
Your job is to read the latest transcript and the provided company policy context, and suggest exactly what the agent should say to solve the problem.

RULES:
1. Be concise. The agent has to read this while talking. Use bullet points if needed.
2. ONLY use the information provided in the KNOWLEDGE BASE. Do not invent answers.
3. If the knowledge base does not contain the answer, say "I cannot find this in our policies. Please escalate."
4. If the customer is angry, add a brief empathy statement first.

LANGUAGE & COMMUNICATION RULES:
1. Native Multilingual Support: You are fully fluent in English, Hindi (Devanagari script), and Hinglish (Hindi written in the English alphabet).
2. Follow the explicit TARGET RESPONSE LANGUAGE supplied with the live transcript. It is authoritative.
   - If they type in English, reply in English.
   - If they type in Hindi (हिंदी), reply in Hindi (हिंदी).
   - If they type in Hinglish (e.g., "Mere account se paise kat gaye"), reply in Hinglish.
3. Never Force Translation: Do not reply in English if the user asked in Hindi, and do not force Devanagari script if the user typed in Hinglish.
4. Tone in Hindi: Maintain a highly respectful, professional, and empathetic banking tone. Use formal/respectful Hindi pronouns (always use "आप" (Aap) for the customer, never "तुम" (Tum) or "तू" (Tu)). Use standard banking terms in English if there is no common Hindi equivalent (e.g., use "Transaction", "Account", "Loan EMI", "Credit Card" instead of forcing pure Hindi translations like "लेनदेन" or "ऋण").
5. Empathy First: If a customer is frustrated in Hindi, acknowledge their frustration politely in Hindi before providing the solution.

KNOWLEDGE BASE:
{retrieved_chunks}

CONVERSATION HISTORY:
{conversation_history}

Return valid JSON only matching the following structure:
{{
    "suggestion": "string (the exact words the agent should say)",
    "policy_reference": "string (citation of the relevant policy, or 'None')",
    "confidence": float (0.0 to 1.0),
    "next_action": "string (brief next step recommendation)"
}}"""

    def build_messages(
        self, 
        transcript_chunk: str, 
        retrieved_chunks: List[Dict[str, Any]], 
        conversation_history: str = ""
    ) -> List[Dict[str, str]]:
        """
        puts the messages in that list format openai wants
        """
        
        formatted_chunks = []
        for chunk in retrieved_chunks:
            source = chunk.get("source_file", "Unknown")
            content = chunk.get("content", "")
            formatted_chunks.append(f"[Source: {source}]\n{content}")
            
        context_str = "\n\n---\n\n".join(formatted_chunks) if formatted_chunks else "No relevant knowledge found."
        
        
        system_content = self.system_prompt_template.format(
            retrieved_chunks=context_str,
            conversation_history=conversation_history if conversation_history else "No prior history."
        )
        
        language_label, language_instruction = detect_response_language(
            transcript_chunk
        )

        
        
        messages = [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": (
                    f"TARGET RESPONSE LANGUAGE: {language_label}\n"
                    f"MANDATORY OUTPUT RULE: {language_instruction}\n\n"
                    f"LIVE CALL TRANSCRIPT:\n{transcript_chunk}"
                ),
            },
        ]
        
        return messages


# --- FROM: retrieval/generator.py ---
"""Coordinate query building, retrieval, and LLM response generation."""

import time
import json
from config.settings import Settings
from config.logger import get_logger

from src.retrieval.hybrid import retrieve_hybrid
from src.analysis.pii import get_pii_service

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
    pii_service = get_pii_service()
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


