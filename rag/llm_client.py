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
from rag.schemas import AgentSuggestionResponse

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
