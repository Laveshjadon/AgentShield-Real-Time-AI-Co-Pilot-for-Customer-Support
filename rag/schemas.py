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
