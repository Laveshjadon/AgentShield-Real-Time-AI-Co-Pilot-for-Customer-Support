"""RAG pipeline adapters used by evaluation runs."""

from __future__ import annotations

from typing import Any, Callable


RagResult = dict[str, Any]
RagPipeline = Callable[[str], RagResult]


def mock_hybrid_rag(question: str) -> RagResult:
    """Tiny executable RAG stub showing the required return shape."""

    contexts = [
        {
            "id": "refund_policy.txt:0",
            "source_file": "refund_policy.txt",
            "chunk_index": 0,
            "content": "Refund requests are handled according to the refund policy and eligibility window.",
        },
        {
            "id": "billing_faq.txt:0",
            "source_file": "billing_faq.txt",
            "chunk_index": 0,
            "content": "Billing agents should verify account ownership before discussing payment details.",
        },
    ]
    return {
        "answer": f"Mock answer for: {question}",
        "contexts": contexts,
        "metadata": {"pipeline": "mock_hybrid"},
    }


def agent_shield_hybrid_rag(question: str, top_k: int = 5) -> RagResult:
    """Adapter for this repo's current Hybrid RAG retriever.

    Replace the answer generation block with your production chain if you want
    generated-answer metrics to exercise the full copilot instead of this
    context-grounded placeholder.
    """

    from src.retrieval.hybrid import retrieve_hybrid

    contexts, retrieval_metadata = retrieve_hybrid(
        question,
        top_dense=20,
        top_sparse=20,
        top_final=top_k,
    )
    joined_context = "\n\n".join(context["content"] for context in contexts)
    answer = (
        "Based on the retrieved policy context: "
        + (joined_context[:900] if joined_context else "No relevant context was retrieved.")
    )
    return {
        "answer": answer,
        "contexts": contexts,
        "metadata": {"pipeline": "agent_shield_hybrid", **retrieval_metadata},
    }


def baseline_dense_rag(question: str, top_k: int = 5) -> RagResult:
    """Baseline adapter using the legacy dense retriever."""

    from src.retrieval.hybrid import retrieve_context

    context_text = retrieve_context(question, top_k=top_k)
    contexts = [
        {
            "source_file": "baseline_dense_context",
            "chunk_index": 0,
            "content": context_text,
        }
    ]
    return {
        "answer": "Based on the dense retrieval context: " + (context_text[:900] or "No context."),
        "contexts": contexts,
        "metadata": {"pipeline": "baseline_dense"},
    }
