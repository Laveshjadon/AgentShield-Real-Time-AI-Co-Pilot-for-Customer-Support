"""
Faithfulness / Hallucination Evaluator
======================================
Measures whether generated agent suggestions are fully grounded in the
retrieved policy context.  A suggestion is faithful if every factual
claim it makes is directly entailed by at least one retrieved chunk.

Why this matters for AgentShield
---------------------------------
The system is grounded by design — it should say "I cannot find this in
our policies" when the KB doesn't cover a question.  But there is no test
that the LLM actually respects that instruction.  In a banking call
centre, a hallucinated refund window, wrong fee amount, or invented
escalation procedure is a compliance risk and a direct agent liability.

Approach: LLM-as-judge (two-step)
-----------------------------------
1. Decompose the suggestion into atomic, independently verifiable claims.
2. For each claim, ask the judge LLM: is this claim directly supported by
   the retrieved context?  Faithfulness = supported_claims / total_claims.

No external eval framework is required — the same OpenAI/Groq client
already configured in settings is reused.

Usage
-----
# Run standalone against the hybrid pipeline:
python -m evaluation.faithfulness --pipeline hybrid

# Import from eval_suite:
from evaluation.faithfulness import evaluate_faithfulness
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from config.settings import Settings
from config.logger import get_logger

logger = get_logger("eval.faithfulness")
settings = Settings()


# ---------------------------------------------------------------------------
# LLM client (reuses the provider already configured in .env)
# ---------------------------------------------------------------------------

def _build_client():
    """Return (client, model_name) using the same provider as the main app."""
    provider = settings.LLM_PROVIDER.lower()
    if provider == "groq":
        from groq import Groq
        if not settings.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set.")
        return Groq(api_key=settings.GROQ_API_KEY), settings.LLM_MODEL
    elif provider == "openai":
        from openai import OpenAI
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set.")
        return OpenAI(api_key=settings.OPENAI_API_KEY), settings.LLM_MODEL
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


# ---------------------------------------------------------------------------
# Step 1: decompose suggestion into atomic claims
# ---------------------------------------------------------------------------

_DECOMPOSE_PROMPT = """\
You are a fact-extraction assistant for a banking call centre audit.

Given an agent suggestion, extract every atomic, independently verifiable
factual claim — statements about policy rules, time windows, fees, procedures,
eligibility criteria, or any domain fact that could be checked against a
policy document.

DO NOT include:
- Greetings or empathy statements ("I understand your frustration")
- Procedural filler ("Let me look into that for you")
- Next-action labels that are purely instructional

Return a JSON object: {"claims": ["claim1", "claim2", ...]}
If there are no verifiable claims, return {"claims": []}.

Suggestion:
{suggestion}"""


def decompose_into_claims(suggestion: str, client, model: str) -> list[str]:
    """Break a suggestion into atomic verifiable claims via LLM."""
    # Strip emoji-decorated metadata lines added by generate_answer
    # e.g. "🔹 Next Action: ..." and "📖 Reference: ..."
    lines = [
        ln for ln in suggestion.splitlines()
        if not ln.strip().startswith(("🔹", "📖", "Next Action:", "Reference:"))
    ]
    clean_suggestion = "\n".join(lines).strip()
    if not clean_suggestion:
        return []

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": _DECOMPOSE_PROMPT.format(suggestion=clean_suggestion),
            }],
            temperature=0.0,
            response_format={"type": "json_object"},
            timeout=15.0,
        )
        raw = resp.choices[0].message.content
        parsed = json.loads(raw)
        # Accept both {"claims": [...]} and a bare list
        if isinstance(parsed, list):
            return [str(c) for c in parsed]
        return [str(c) for c in parsed.get("claims", [])]
    except Exception as exc:
        logger.warning("[FAITHFULNESS] decompose_into_claims failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Step 2: check whether each claim is supported by retrieved context
# ---------------------------------------------------------------------------

_SUPPORT_PROMPT = """\
You are a compliance auditor for a banking call centre AI system.

RETRIEVED POLICY CONTEXT:
{context}

CLAIM TO VERIFY:
{claim}

Is this claim directly supported by the policy context above?
- Answer TRUE only if the claim is explicitly stated or unambiguously implied.
- Answer FALSE if the claim contradicts the context, extends beyond it, or
  cannot be verified from it.

Return JSON: {{"supported": true}} or {{"supported": false}}"""


def _check_claim(claim: str, context: str, client, model: str) -> bool:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": _SUPPORT_PROMPT.format(context=context, claim=claim),
            }],
            temperature=0.0,
            response_format={"type": "json_object"},
            timeout=10.0,
        )
        raw = resp.choices[0].message.content
        return bool(json.loads(raw).get("supported", False))
    except Exception as exc:
        logger.warning("[FAITHFULNESS] _check_claim failed: %s", exc)
        # Conservative: treat parse failures as unsupported
        return False


# ---------------------------------------------------------------------------
# Per-example faithfulness score
# ---------------------------------------------------------------------------

def faithfulness_score(
    suggestion: str,
    retrieved_chunks: list[dict[str, Any]],
    client,
    model: str,
) -> dict[str, Any]:
    """
    Compute faithfulness for one (suggestion, retrieved_chunks) pair.

    Returns
    -------
    {
        "faithfulness": float,          # supported / total, or 1.0 if no claims
        "supported_count": int,
        "total_count": int,
        "claims": [{"claim": str, "supported": bool}, ...],
        "note": str | None,
    }
    """
    if not suggestion:
        return {"faithfulness": 0.0, "supported_count": 0, "total_count": 0,
                "claims": [], "note": "empty_suggestion"}

    if not retrieved_chunks:
        return {"faithfulness": 0.0, "supported_count": 0, "total_count": 0,
                "claims": [], "note": "no_context"}

    combined_context = "\n\n---\n\n".join(
        f"[Source: {c.get('source_file', 'unknown')}]\n{c.get('content', '')}"
        for c in retrieved_chunks
    )

    claims = decompose_into_claims(suggestion, client, model)
    if not claims:
        # No verifiable claims → not hallucinating, but also not asserting anything
        return {"faithfulness": 1.0, "supported_count": 0, "total_count": 0,
                "claims": [], "note": "no_verifiable_claims"}

    results = []
    for claim in claims:
        supported = _check_claim(claim, combined_context, client, model)
        results.append({"claim": claim, "supported": supported})

    supported_count = sum(1 for r in results if r["supported"])
    score = supported_count / len(results)

    return {
        "faithfulness": round(score, 4),
        "supported_count": supported_count,
        "total_count": len(results),
        "claims": results,
        "note": None,
    }


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_faithfulness(
    examples: list[dict[str, Any]],
    pipeline_fn,
) -> dict[str, Any]:
    """
    Run faithfulness evaluation across a list of golden examples.

    Parameters
    ----------
    examples:
        List of dicts with at least {"question": str}.  The pipeline_fn is
        called with the question and must return:
            {"answer": str, "contexts": list[dict]}
    pipeline_fn:
        Callable matching the RagPipeline signature in rag_pipelines.py.

    Returns
    -------
    Aggregate summary + per-example breakdown.
    """
    client, model = _build_client()

    all_faith_scores: list[float] = []
    per_example: list[dict] = []
    hallucinated: list[dict] = []

    for example in examples:
        question = example["question"]
        example_id = example.get("id", question[:30])

        t0 = time.perf_counter()
        result = pipeline_fn(question)
        pipeline_ms = round((time.perf_counter() - t0) * 1000, 1)

        suggestion = result.get("answer", "")
        contexts = result.get("contexts", [])

        scores = faithfulness_score(suggestion, contexts, client, model)

        row = {
            "id": example_id,
            "question": question,
            "suggestion_preview": suggestion[:200],
            "pipeline_ms": pipeline_ms,
            **scores,
        }
        per_example.append(row)
        all_faith_scores.append(scores["faithfulness"])

        if scores["faithfulness"] < 1.0 and scores["total_count"] > 0:
            hallucinated.append({
                "id": example_id,
                "question": question,
                "faithfulness": scores["faithfulness"],
                "unsupported_claims": [
                    r["claim"] for r in scores["claims"] if not r["supported"]
                ],
            })

        logger.info(
            "[FAITHFULNESS] id=%s | %.3f (%d/%d claims supported)",
            example_id, scores["faithfulness"],
            scores["supported_count"], scores["total_count"],
        )

    n = len(all_faith_scores)
    mean_f = sum(all_faith_scores) / n if n else 0.0
    min_f = min(all_faith_scores) if n else 0.0

    summary = {
        "mean_faithfulness": round(mean_f, 4),
        "min_faithfulness": round(min_f, 4),
        "hallucination_rate": round(len(hallucinated) / n, 4) if n else 0.0,
        "examples_with_hallucinations": len(hallucinated),
        "total_examples": n,
        "hallucinated_examples": hallucinated,
        "per_example": per_example,
        # Pass/fail gate: production target ≥ 0.95
        "passed": mean_f >= 0.95,
    }

    _print_faithfulness_report(summary)
    return summary


def _print_faithfulness_report(summary: dict) -> None:
    status = "PASS ✓" if summary["passed"] else "FAIL ✗"
    print("\n" + "=" * 60)
    print(f"  FAITHFULNESS / HALLUCINATION  [{status}]")
    print("=" * 60)
    print(f"  Mean Faithfulness:      {summary['mean_faithfulness']:.3f}  (target ≥ 0.950)")
    print(f"  Min Faithfulness:       {summary['min_faithfulness']:.3f}")
    print(f"  Hallucination Rate:     {summary['hallucination_rate']:.1%}")
    print(f"  Examples with issues:   {summary['examples_with_hallucinations']}/{summary['total_examples']}")

    if summary["hallucinated_examples"]:
        print("\n  Unsupported claims detected:")
        for ex in summary["hallucinated_examples"]:
            print(f"\n    [{ex['id']}] faithfulness={ex['faithfulness']:.3f}")
            for claim in ex["unsupported_claims"]:
                print(f"      ✗ {claim}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run faithfulness evaluation against a RAG pipeline."
    )
    parser.add_argument(
        "--pipeline",
        choices=["mock", "hybrid", "baseline"],
        default="hybrid",
        help="Which pipeline adapter to evaluate.",
    )
    parser.add_argument(
        "--dataset",
        default="evaluation/golden_dataset.jsonl",
        help="Path to JSONL golden dataset.",
    )
    args = parser.parse_args()

    from evaluation.dataset import load_golden_dataset
    from evaluation.rag_pipelines import (
        agent_shield_hybrid_rag,
        baseline_dense_rag,
        mock_hybrid_rag,
    )

    pipelines = {
        "mock": mock_hybrid_rag,
        "hybrid": agent_shield_hybrid_rag,
        "baseline": baseline_dense_rag,
    }

    examples = [
        {"id": ex.id, "question": ex.question, "ground_truth": ex.ground_truth}
        for ex in load_golden_dataset(args.dataset)
    ]
    evaluate_faithfulness(examples, pipelines[args.pipeline])
