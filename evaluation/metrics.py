"""Deterministic retrieval metrics for ranked RAG contexts."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def context_id(context: dict[str, Any] | str) -> str:
    """Normalize a retrieved context to a stable id.

    Preferred ids are source_file:chunk_index. If your labels are only files,
    matching still works because source_file is also considered below.
    """

    if isinstance(context, str):
        return context
    source = str(context.get("source_file") or context.get("source") or "")
    chunk_index = context.get("chunk_index")
    if source and chunk_index is not None:
        return f"{source}:{chunk_index}"
    return source or str(context.get("id") or context.get("doc_id") or "")


def _matches(retrieved: dict[str, Any] | str, expected_ids: set[str]) -> bool:
    rid = context_id(retrieved)
    if rid in expected_ids:
        return True
    if isinstance(retrieved, dict):
        source = str(retrieved.get("source_file") or retrieved.get("source") or "")
        return source in expected_ids
    return False


def recall_at_k(
    retrieved_contexts: Iterable[dict[str, Any] | str],
    ground_truth_context: Iterable[str],
    k: int,
) -> float:
    expected = set(ground_truth_context)
    if not expected:
        return 0.0
    hits = {
        expected_id
        for context in list(retrieved_contexts)[:k]
        for expected_id in expected
        if _matches(context, {expected_id})
    }
    return len(hits) / len(expected)


def hit_at_k(
    retrieved_contexts: Iterable[dict[str, Any] | str],
    ground_truth_context: Iterable[str],
    k: int,
) -> float:
    expected = set(ground_truth_context)
    return float(any(_matches(context, expected) for context in list(retrieved_contexts)[:k]))


def mrr(
    retrieved_contexts: Iterable[dict[str, Any] | str],
    ground_truth_context: Iterable[str],
    k: int | None = None,
) -> float:
    expected = set(ground_truth_context)
    ranked = list(retrieved_contexts)
    if k is not None:
        ranked = ranked[:k]
    for index, context in enumerate(ranked, start=1):
        if _matches(context, expected):
            return 1.0 / index
    return 0.0


def retrieval_scores(
    retrieved_contexts: list[dict[str, Any] | str],
    ground_truth_context: list[str],
    k_values: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for k in k_values:
        scores[f"recall@{k}"] = recall_at_k(retrieved_contexts, ground_truth_context, k)
        scores[f"hit@{k}"] = hit_at_k(retrieved_contexts, ground_truth_context, k)
        scores[f"mrr@{k}"] = mrr(retrieved_contexts, ground_truth_context, k)
    return scores
