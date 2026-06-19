"""Run RAG evaluations and push metrics to LangSmith."""

from __future__ import annotations

import argparse
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from langsmith import Client

from evaluation.dataset import load_golden_dataset, upsert_langsmith_dataset
from evaluation.metrics import retrieval_scores
from evaluation.rag_pipelines import agent_shield_hybrid_rag, baseline_dense_rag, mock_hybrid_rag


PIPELINES = {
    "mock": mock_hybrid_rag,
    "hybrid": agent_shield_hybrid_rag,
    "baseline": baseline_dense_rag,
}


def _log_feedback(client: Client, run_id: str, scores: dict[str, float]) -> None:
    for key, value in scores.items():
        client.create_feedback(run_id=run_id, key=key, score=value)


def evaluate_pipeline(
    pipeline_name: str,
    dataset_path: str | Path,
    langsmith_dataset: str,
    k_values: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    examples = load_golden_dataset(dataset_path)
    upsert_langsmith_dataset(langsmith_dataset, examples)
    client = Client()
    all_scores: list[dict[str, float]] = []

    for example in examples:
        run_id = uuid.uuid4()
        client.create_run(
            id=run_id,
            name=f"{pipeline_name}_rag_eval_case",
            run_type="chain",
            inputs={"question": example.question},
            project_name=os.environ.get("LANGCHAIN_PROJECT"),
            start_time=datetime.now(timezone.utc),
            extra={
                "metadata": {
                    "experiment": f"{pipeline_name}-rag-eval",
                    "example_id": example.id,
                    "ground_truth_context": example.ground_truth_context,
                }
            },
        )
        result = PIPELINES[pipeline_name](example.question)
        client.update_run(
            run_id,
            outputs={
                "answer": result["answer"],
                "contexts": result["contexts"],
                "ground_truth": example.ground_truth,
                "metadata": result.get("metadata", {}),
            },
            end_time=datetime.now(timezone.utc),
        )

        scores = retrieval_scores(result["contexts"], example.ground_truth_context, k_values=k_values)
        all_scores.append(scores)
        _log_feedback(client, str(run_id), scores)

    summary = {
        metric: mean(score_row[metric] for score_row in all_scores)
        for metric in all_scores[0]
    }
    print(f"{pipeline_name} summary")
    for metric, value in summary.items():
        print(f"{metric}: {value:.3f}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=sorted(PIPELINES), default="mock")
    parser.add_argument("--dataset", default="evaluation/golden_dataset.jsonl")
    parser.add_argument("--langsmith-dataset", default="AgentShield Golden RAG Eval")
    args = parser.parse_args()

    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", f"agentshield-{args.pipeline}-rag-eval")
    evaluate_pipeline(args.pipeline, args.dataset, args.langsmith_dataset)


if __name__ == "__main__":
    main()
