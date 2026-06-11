import pytest
from rag.prompt_builder import PromptBuilder, detect_response_language
import json

def test_prompt_builder_structure():
    pb = PromptBuilder()
    
    retrieved_chunks = [
        {"source_file": "policy.pdf", "content": "Refunds are allowed within 14 days."}
    ]
    transcript = "I want a refund for my router."
    history = "Agent: Hello."
    
    messages = pb.build_messages(
        transcript_chunk=transcript,
        retrieved_chunks=retrieved_chunks,
        conversation_history=history
    )
    
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    
    
    assert "policy.pdf" in messages[0]["content"]
    assert "Refunds are allowed" in messages[0]["content"]
    assert "Agent: Hello." in messages[0]["content"]
    
    
    assert "Return valid JSON only matching the following structure" in messages[0]["content"]
    
    
    assert transcript in messages[1]["content"]
    assert "TARGET RESPONSE LANGUAGE: English" in messages[1]["content"]
    assert "Do not use Hindi or Devanagari" in messages[1]["content"]


def test_language_detection_english():
    label, _ = detect_response_language(
        "My account was debited but the beneficiary did not receive the payment."
    )
    assert label == "English"


def test_language_detection_hinglish():
    label, instruction = detect_response_language(
        "Mere account se paise kat gaye lekin refund nahi aaya."
    )
    assert label == "Hinglish (Roman script)"
    assert "Latin alphabet" in instruction


def test_language_detection_devanagari():
    label, instruction = detect_response_language(
        "मेरे खाते से पैसे कट गए लेकिन भुगतान नहीं हुआ।"
    )
    assert label == "Hindi (Devanagari)"
    assert "Devanagari" in instruction


def test_latest_english_turn_overrides_hinglish_history():
    pb = PromptBuilder()

    messages = pb.build_messages(
        transcript_chunk="My friend has not received the money I transferred.",
        retrieved_chunks=[],
        conversation_history=(
            "Customer: Mere account se paise kat gaye hain.\n"
            "Agent: Main transaction check kar raha hoon."
        ),
    )

    assert "TARGET RESPONSE LANGUAGE: English" in messages[1]["content"]
    assert "Do not use Hindi or Devanagari" in messages[1]["content"]
