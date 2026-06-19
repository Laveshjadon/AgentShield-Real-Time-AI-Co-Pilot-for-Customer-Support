"""Golden dataset loading and LangSmith dataset setup."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GoldenExample:
    """One labeled RAG evaluation example."""

    id: str
    question: str
    ground_truth: str
    ground_truth_context: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


def load_golden_dataset(path: str | Path) -> list[GoldenExample]:
    """Load JSONL examples with question, expected answer, and expected sources."""

    examples: list[GoldenExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append(
                GoldenExample(
                    id=row.get("id") or f"example-{line_number}",
                    question=row["question"],
                    ground_truth=row["ground_truth"],
                    ground_truth_context=list(row["ground_truth_context"]),
                    metadata=dict(row.get("metadata", {})),
                )
            )
    return examples


def upsert_langsmith_dataset(
    dataset_name: str,
    examples: list[GoldenExample],
    description: str = "Golden dataset for Hybrid RAG evaluation.",
) -> str:
    """Create or update a LangSmith dataset and return its id.

    Requires LANGCHAIN_API_KEY and LANGCHAIN_TRACING_V2=true.
    """

    from langsmith import Client

    client = Client()
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
    except Exception:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=description,
        )

    existing = {
        example.id
        for example in client.list_examples(dataset_id=dataset.id)
        if example.id
    }
    for example in examples:
        if example.id in existing:
            continue
        client.create_example(
            dataset_id=dataset.id,
            inputs={
                "question": example.question,
                "ground_truth_context": example.ground_truth_context,
            },
            outputs={"ground_truth": example.ground_truth},
            metadata={"example_id": example.id, **example.metadata},
        )
    return str(dataset.id)
