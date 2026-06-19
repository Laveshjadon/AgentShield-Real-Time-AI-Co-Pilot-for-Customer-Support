"""
AgentShield Unified Evaluation Suite
=====================================
Runs all three evaluators in sequence and prints a consolidated pass/fail
report.  Designed to be run as a pre-deployment CI gate.

Evaluators
----------
1. Faithfulness   — hallucination rate of RAG suggestions
2. Grounding      — bypass rate on out-of-scope questions
3. PII Coverage   — false negative rate per entity type

Exit code
---------
0  → all evaluators passed
1  → one or more evaluators failed

Usage
-----
# Run against mock pipelines (no API keys required):
python -m evaluation.eval_suite --pipeline mock

# Run against the live hybrid pipeline:
python -m evaluation.eval_suite --pipeline hybrid

# Skip one evaluator:
python -m evaluation.eval_suite --skip pii
python -m evaluation.eval_suite --skip faithfulness grounding
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

from config.logger import get_logger

logger = get_logger("eval.suite")

EVALUATORS = ("faithfulness", "grounding", "pii")


# ---------------------------------------------------------------------------
# Individual runners (each catches its own errors so one failure ≠ suite crash)
# ---------------------------------------------------------------------------

def _run_faithfulness(pipeline_fn, dataset_path: str) -> dict[str, Any]:
    from evaluation.faithfulness import evaluate_faithfulness
    from evaluation.dataset import load_golden_dataset

    examples = [
        {"id": ex.id, "question": ex.question, "ground_truth": ex.ground_truth}
        for ex in load_golden_dataset(dataset_path)
    ]
    return evaluate_faithfulness(examples, pipeline_fn)


def _run_grounding(pipeline_fn, dataset_path: str) -> dict[str, Any]:
    from evaluation.grounding import evaluate_grounding, load_grounding_dataset

    records = load_grounding_dataset(dataset_path)
    return evaluate_grounding(records, pipeline_fn)


def _run_pii(verbose: bool) -> dict[str, Any]:
    from evaluation.pii_eval import evaluate_pii

    return evaluate_pii(verbose=verbose)


# ---------------------------------------------------------------------------
# Main suite runner
# ---------------------------------------------------------------------------

def run_suite(
    pipeline: str = "mock",
    skip: list[str] | None = None,
    golden_dataset: str = "evaluation/golden_dataset.jsonl",
    grounding_dataset: str = "evaluation/out_of_scope_dataset.jsonl",
    verbose: bool = False,
) -> bool:
    """
    Run all (non-skipped) evaluators.  Returns True iff all passed.
    """
    skip = skip or []
    run_order = [e for e in EVALUATORS if e not in skip]

    # Resolve pipeline function
    if pipeline == "mock":
        from evaluation.grounding import _mock_pipeline

        def _mock_rag(question: str) -> dict[str, Any]:
            """Shared mock: returns plausible answer for in-scope, refusal otherwise."""
            return _mock_pipeline(question)

        pipeline_fn = _mock_rag
    elif pipeline == "hybrid":
        from evaluation.rag_pipelines import agent_shield_hybrid_rag
        pipeline_fn = agent_shield_hybrid_rag
    elif pipeline == "baseline":
        from evaluation.rag_pipelines import baseline_dense_rag
        pipeline_fn = baseline_dense_rag
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")

    results: dict[str, dict] = {}
    timings: dict[str, float] = {}
    errors: dict[str, str] = {}

    for name in run_order:
        t0 = time.perf_counter()
        try:
            if name == "faithfulness":
                results[name] = _run_faithfulness(pipeline_fn, golden_dataset)
            elif name == "grounding":
                results[name] = _run_grounding(pipeline_fn, grounding_dataset)
            elif name == "pii":
                results[name] = _run_pii(verbose)
        except Exception as exc:
            logger.error("[SUITE] %s evaluator crashed: %s", name, exc, exc_info=True)
            errors[name] = str(exc)
            results[name] = {"passed": False}
        timings[name] = round((time.perf_counter() - t0), 2)

    _print_suite_report(results, timings, errors, pipeline, skip)

    all_passed = all(results[n].get("passed", False) for n in run_order)
    return all_passed


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def _print_suite_report(
    results: dict[str, dict],
    timings: dict[str, float],
    errors: dict[str, str],
    pipeline: str,
    skipped: list[str],
) -> None:
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " AGENTSHIELD EVALUATION SUITE — SUMMARY".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print(f"  Pipeline: {pipeline}")
    if skipped:
        print(f"  Skipped:  {', '.join(skipped)}")
    print()

    labels = {
        "faithfulness": "Faithfulness / Hallucination",
        "grounding":    "Grounding Bypass",
        "pii":          "PII Redaction Coverage",
    }
    gate_details = {
        "faithfulness": lambda r: f"mean_faith={r.get('mean_faithfulness', '?'):.3f}  (≥0.95)",
        "grounding":    lambda r: (
            f"bypass={r.get('bypass_rate', '?'):.1%}  (<5%)  "
            f"over_refusal={r.get('over_refusal_rate', '?'):.1%}  (<10%)"
        ),
        "pii":          lambda r: (
            f"{r.get('total_cases', '?')} cases  "
            f"{r.get('failed_cases', '?')} failed  "
            f"{len(r.get('false_positive_failures', []))} FP failures"
        ),
    }

    all_ok = True
    for name, res in results.items():
        passed = res.get("passed", False)
        if not passed:
            all_ok = False
        icon = "✓ PASS" if passed else "✗ FAIL"
        label = labels.get(name, name)
        elapsed = timings.get(name, 0)

        print(f"  [{icon}]  {label}")
        if name in errors:
            print(f"           ERROR: {errors[name]}")
        elif name in gate_details:
            try:
                print(f"           {gate_details[name](res)}")
            except Exception:
                pass
        print(f"           Time: {elapsed:.1f}s")
        print()

    overall = "ALL PASSED ✓" if all_ok else "ONE OR MORE FAILED ✗"
    print("  " + "─" * 40)
    print(f"  Overall: {overall}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AgentShield unified evaluation suite."
    )
    parser.add_argument(
        "--pipeline",
        choices=["mock", "hybrid", "baseline"],
        default="mock",
        help="Pipeline adapter to use for faithfulness and grounding evals.",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=list(EVALUATORS),
        default=[],
        metavar="EVAL",
        help="Evaluator(s) to skip.",
    )
    parser.add_argument(
        "--golden-dataset",
        default="evaluation/golden_dataset.jsonl",
        help="Path to golden Q&A dataset for faithfulness eval.",
    )
    parser.add_argument(
        "--grounding-dataset",
        default="evaluation/out_of_scope_dataset.jsonl",
        help="Path to OOS/in-scope dataset for grounding eval.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print all PII test cases, not just failures.",
    )
    args = parser.parse_args()

    passed = run_suite(
        pipeline=args.pipeline,
        skip=args.skip,
        golden_dataset=args.golden_dataset,
        grounding_dataset=args.grounding_dataset,
        verbose=args.verbose,
    )
    sys.exit(0 if passed else 1)
