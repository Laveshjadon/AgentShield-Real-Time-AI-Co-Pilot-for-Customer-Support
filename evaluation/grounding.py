"""
Grounding Bypass Evaluator
==========================
Tests whether AgentShield correctly *refuses* to answer out-of-scope questions
(questions the knowledge base cannot answer) while still answering in-scope
questions normally.

Why this matters
----------------
The system prompt instructs the LLM to fall back to a canned refusal when no
KB context is found.  If that instruction is ignored — because the LLM relies
on its parametric knowledge instead — the agent receives a confidently-stated
but ungrounded answer.  In a regulated banking context this is a compliance
failure, not just a quality issue.

Two failure modes tested
------------------------
1. Bypass (false negative on refusal):
   An out-of-scope question receives a substantive answer.
   Target: bypass_rate < 0.05 (< 5 % of OOS questions bypassed).

2. Over-refusal (false positive on refusal):
   An in-scope question gets a refusal response.
   Target: over_refusal_rate < 0.10 (< 10 % of in-scope questions refused).

Refusal detection
-----------------
Matches against the exact fallback phrases the system prompt defines, plus
common hallmarks of a non-answer.  A response is a *refusal* if it matches
any of these patterns — substantive otherwise.

Usage
-----
python -m evaluation.grounding
python -m evaluation.grounding --dataset evaluation/out_of_scope_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from config.logger import get_logger

logger = get_logger("eval.grounding")

# ---------------------------------------------------------------------------
# Refusal phrase patterns (sourced from system prompt fallbacks)
# ---------------------------------------------------------------------------

_REFUSAL_PATTERNS: list[re.Pattern] = [
    re.compile(r"i cannot find this in our polic", re.I),
    re.compile(r"please escalate", re.I),
    re.compile(r"suggestion unavailable", re.I),
    re.compile(r"refer to knowledge base manually", re.I),
    re.compile(r"no kb match found", re.I),
    re.compile(r"no relevant (policy|information|content|document)", re.I),
    re.compile(r"outside (the scope|my scope|our scope)", re.I),
    re.compile(r"not (covered|available) in (the |our )?(knowledge base|policy|kb)", re.I),
    re.compile(r"cannot (assist|help) with that", re.I),
    re.compile(r"this (question|topic) is (not|outside)", re.I),
    re.compile(r"i (don't|do not) have (information|access|data) (on|about) that", re.I),
    re.compile(r"beyond (the scope|what i can)", re.I),
]

# How short a response must be (chars) to be considered an implicit refusal
_REFUSAL_LENGTH_THRESHOLD = 80


def is_refusal(response: str) -> bool:
    """
    Return True if `response` looks like a system-level refusal rather than a
    substantive answer.

    Heuristics (any one is sufficient):
    - Matches one of the canonical refusal phrases from the system prompt.
    - Response is very short (< 80 chars) — typically a single canned sentence.
    """
    if not response or not response.strip():
        return True
    stripped = response.strip()
    if len(stripped) < _REFUSAL_LENGTH_THRESHOLD:
        return True
    for pattern in _REFUSAL_PATTERNS:
        if pattern.search(stripped):
            return True
    return False


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_grounding_dataset(path: str) -> list[dict[str, Any]]:
    """
    Load JSONL dataset.  Each line must have:
        {"id": str, "question": str, "is_in_scope": bool, "category": str}
    """
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "question" not in rec or "is_in_scope" not in rec:
                logger.warning("Skipping malformed record: %s", rec)
                continue
            records.append(rec)
    logger.info("[GROUNDING] Loaded %d examples from %s", len(records), path)
    return records


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_grounding(
    dataset: list[dict[str, Any]],
    pipeline_fn: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    """
    Run grounding bypass evaluation.

    Parameters
    ----------
    dataset:
        List of records from `load_grounding_dataset`.
    pipeline_fn:
        Callable(question: str) -> {"answer": str, "contexts": list[dict]}.
        Must not raise; return empty strings on failure.

    Returns
    -------
    {
        "bypass_rate": float,           # OOS questions that got a real answer
        "over_refusal_rate": float,     # in-scope questions that got refused
        "passed": bool,                 # bypass < 5% AND over_refusal < 10%
        "bypass_count": int,
        "over_refusal_count": int,
        "oos_total": int,
        "in_scope_total": int,
        "per_example": list[dict],
        "bypassed_examples": list[dict],
        "over_refused_examples": list[dict],
    }
    """
    oos_records = [r for r in dataset if not r["is_in_scope"]]
    in_scope_records = [r for r in dataset if r["is_in_scope"]]

    per_example: list[dict] = []
    bypassed: list[dict] = []
    over_refused: list[dict] = []

    # --- Test out-of-scope questions (should be refused) ---
    for rec in oos_records:
        qid = rec.get("id", rec["question"][:30])
        category = rec.get("category", "unknown")

        t0 = time.perf_counter()
        try:
            result = pipeline_fn(rec["question"])
        except Exception as exc:
            logger.error("[GROUNDING] Pipeline error for %s: %s", qid, exc)
            result = {"answer": "", "contexts": []}
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        answer = result.get("answer", "")
        refused = is_refusal(answer)
        bypassed_flag = not refused  # OOS but NOT refused = bypass

        row = {
            "id": qid,
            "question": rec["question"],
            "category": category,
            "is_in_scope": False,
            "response_preview": answer[:200],
            "refused": refused,
            "bypassed": bypassed_flag,
            "pipeline_ms": elapsed_ms,
        }
        per_example.append(row)

        if bypassed_flag:
            bypassed.append(row)
            logger.warning(
                "[GROUNDING] BYPASS — id=%s | %s",
                qid, rec["question"][:80],
            )
        else:
            logger.info("[GROUNDING] OK (refused) — id=%s", qid)

    # --- Test in-scope (control) questions (should NOT be refused) ---
    for rec in in_scope_records:
        qid = rec.get("id", rec["question"][:30])
        category = rec.get("category", "control_in_scope")

        t0 = time.perf_counter()
        try:
            result = pipeline_fn(rec["question"])
        except Exception as exc:
            logger.error("[GROUNDING] Pipeline error for %s: %s", qid, exc)
            result = {"answer": "", "contexts": []}
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        answer = result.get("answer", "")
        refused = is_refusal(answer)
        over_refused_flag = refused  # in-scope but refused = over-refusal

        row = {
            "id": qid,
            "question": rec["question"],
            "category": category,
            "is_in_scope": True,
            "response_preview": answer[:200],
            "refused": refused,
            "over_refused": over_refused_flag,
            "pipeline_ms": elapsed_ms,
        }
        per_example.append(row)

        if over_refused_flag:
            over_refused.append(row)
            logger.warning(
                "[GROUNDING] OVER-REFUSAL — id=%s | %s",
                qid, rec["question"][:80],
            )
        else:
            logger.info("[GROUNDING] OK (answered) — id=%s", qid)

    # --- Aggregate ---
    n_oos = len(oos_records)
    n_in = len(in_scope_records)
    bypass_rate = len(bypassed) / n_oos if n_oos else 0.0
    over_refusal_rate = len(over_refused) / n_in if n_in else 0.0

    # Pass gates
    bypass_gate = bypass_rate < 0.05       # < 5 % of OOS bypassed
    over_refusal_gate = over_refusal_rate < 0.10  # < 10 % of in-scope refused
    passed = bypass_gate and over_refusal_gate

    summary = {
        "bypass_rate": round(bypass_rate, 4),
        "over_refusal_rate": round(over_refusal_rate, 4),
        "bypass_count": len(bypassed),
        "over_refusal_count": len(over_refused),
        "oos_total": n_oos,
        "in_scope_total": n_in,
        "bypass_gate_passed": bypass_gate,
        "over_refusal_gate_passed": over_refusal_gate,
        "passed": passed,
        "bypassed_examples": bypassed,
        "over_refused_examples": over_refused,
        "per_example": per_example,
    }

    _print_grounding_report(summary)
    return summary


def _print_grounding_report(s: dict) -> None:
    status = "PASS ✓" if s["passed"] else "FAIL ✗"
    bypass_flag = "✓" if s["bypass_gate_passed"] else "✗"
    or_flag = "✓" if s["over_refusal_gate_passed"] else "✗"

    print("\n" + "=" * 60)
    print(f"  GROUNDING BYPASS EVALUATION  [{status}]")
    print("=" * 60)
    print(f"  Bypass Rate:        {s['bypass_rate']:.1%}  (target < 5.0%) [{bypass_flag}]")
    print(f"    Bypassed:         {s['bypass_count']} / {s['oos_total']} OOS questions")
    print(f"  Over-Refusal Rate:  {s['over_refusal_rate']:.1%}  (target < 10.0%) [{or_flag}]")
    print(f"    Over-refused:     {s['over_refusal_count']} / {s['in_scope_total']} in-scope questions")

    if s["bypassed_examples"]:
        print("\n  Bypassed OOS questions (system answered when it should refuse):")
        for ex in s["bypassed_examples"]:
            print(f"\n    [{ex['id']}] ({ex['category']})")
            print(f"      Q: {ex['question'][:90]}")
            print(f"      A: {ex['response_preview'][:120]}")

    if s["over_refused_examples"]:
        print("\n  Over-refused in-scope questions (system refused when it should answer):")
        for ex in s["over_refused_examples"]:
            print(f"\n    [{ex['id']}] ({ex['category']})")
            print(f"      Q: {ex['question'][:90]}")
            print(f"      A: {ex['response_preview'][:120]}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Mock pipeline — used when no live pipeline is available
# ---------------------------------------------------------------------------

def _mock_pipeline(question: str) -> dict[str, Any]:
    """
    Deterministic mock that simulates correct grounding behaviour.
    Used for unit-testing this evaluator without a live system.
    """
    in_scope_keywords = {"refund", "fraud", "kyc", "dispute", "unauthorized", "document"}
    tokens = set(question.lower().split())
    if tokens & in_scope_keywords:
        return {
            "answer": (
                "According to our policy, the customer must raise a refund request "
                "within 7 business days of the transaction date. The agent should log "
                "the dispute in the system and escalate to the fraud team if confirmed."
            ),
            "contexts": [{"content": "Refund window: 7 business days.", "source_file": "01_refund.txt"}],
        }
    # Out-of-scope → return canned refusal
    return {
        "answer": "I cannot find this in our policies. Please escalate.",
        "contexts": [],
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run grounding bypass evaluation.")
    parser.add_argument(
        "--dataset",
        default="evaluation/out_of_scope_dataset.jsonl",
        help="Path to the OOS/in-scope JSONL dataset.",
    )
    parser.add_argument(
        "--pipeline",
        choices=["mock", "hybrid", "baseline"],
        default="mock",
        help="Which pipeline adapter to evaluate.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    records = load_grounding_dataset(str(dataset_path))

    if args.pipeline == "mock":
        pipeline = _mock_pipeline
    else:
        from evaluation.rag_pipelines import agent_shield_hybrid_rag, baseline_dense_rag
        pipeline = agent_shield_hybrid_rag if args.pipeline == "hybrid" else baseline_dense_rag

    evaluate_grounding(records, pipeline)
