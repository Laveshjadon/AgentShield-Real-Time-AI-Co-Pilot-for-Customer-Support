"""
PII Redaction
Redacts sensitive identifiers from transcripts before LLM processing.

Entities it blocks:
- EMAIL_ADDRESS (built-in)
- PHONE_NUMBER (built-in)
- CREDIT_CARD (built-in)
- LOCATION (built-in)
- AADHAAR (custom regex)
- PAN (custom regex)
- BANK_ACCOUNT (custom regex)
"""

import asyncio
import time
from typing import Dict, List, Any

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from config.logger import get_logger

logger = get_logger("analysis.pii")


class PIIService:
    def __init__(self):
        """
        sets up presidio with our custom recognizers for indian id types.
        presidio handles email, phone, credit card out of the box.
        i had to add aadhaar, pan, and bank account myself since those aren't built in.
        """
        t0 = time.perf_counter()
        
        
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers()
        
        
        aadhaar_pattern = Pattern(
            name="aadhaar_pattern",
            regex=r"\b\d{4}\s?\d{4}\s?\d{4}\b",
            score=1.0
        )
        aadhaar_recognizer = PatternRecognizer(
            supported_entity="AADHAAR",
            patterns=[aadhaar_pattern]
        )
        registry.add_recognizer(aadhaar_recognizer)
        
        
        pan_pattern = Pattern(
            name="pan_pattern",
            regex=r"\b[A-Z]{5}\d{4}[A-Z]{1}\b",
            score=1.0
        )
        pan_recognizer = PatternRecognizer(
            supported_entity="PAN",
            patterns=[pan_pattern]
        )
        registry.add_recognizer(pan_recognizer)
        
        
        # Require a banking keyword before the number to avoid false positives
        # on reference IDs, order numbers, timestamps, etc.
        bank_pattern = Pattern(
            name="bank_account_pattern",
            regex=r"(?i)(?:account|acc|a\/c|acct)[^\d]{0,10}\d{9,18}\b",
            score=0.85
        )
        bank_recognizer = PatternRecognizer(
            supported_entity="BANK_ACCOUNT",
            patterns=[bank_pattern]
        )
        registry.add_recognizer(bank_recognizer)
        
        
        self.analyzer = AnalyzerEngine(registry=registry, supported_languages=["en"])
        self.anonymizer = AnonymizerEngine()
        
        
        self.entities = [
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "CREDIT_CARD",
            "LOCATION",      
            "AADHAAR",
            "PAN",
            "BANK_ACCOUNT"
        ]
        
        logger.info(f"PIIService initialized in {(time.perf_counter() - t0) * 1000:.1f}ms")

    def redact(self, text: str) -> str:
        """
        analyzes text and removes the PII.
        returns the clean string.
        """
        if not text or not text.strip():
            return text
            
        
        results = self.analyzer.analyze(
            text=text,
            entities=self.entities,
            language="en"
        )
        
        if not results:
            return text
            
        
        entity_counts: Dict[str, int] = {}
        for r in results:
            entity_counts[r.entity_type] = entity_counts.get(r.entity_type, 0) + 1
            
        if entity_counts:
            logger.info(f"[PII] Detected and redacting: {entity_counts}")
            
        
        operators = {
            entity: OperatorConfig("replace", {"new_value": f"[{entity}_REDACTED]"})
            for entity in self.entities
        }
        
        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=operators
        )
        
        return anonymized_result.text

    async def aredact(self, text: str) -> str:
        """
        async version of redact. runs in a thread pool so it doesn't block.
        hopefully finishes under 50ms.
        """
        t0 = time.perf_counter()
        
        
        redacted_text = await asyncio.to_thread(self.redact, text)
        
        latency_ms = (time.perf_counter() - t0) * 1000
        if latency_ms > 100:
            logger.warning(f"[PII] Redaction took longer than expected: {latency_ms:.1f}ms")
            
        return redacted_text


_pii_service = None


def get_pii_service() -> PIIService:
    global _pii_service
    if _pii_service is None:
        _pii_service = PIIService()
    return _pii_service
