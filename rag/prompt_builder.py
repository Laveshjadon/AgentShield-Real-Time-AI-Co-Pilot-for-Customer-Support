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
