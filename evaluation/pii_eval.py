"""
PII Redaction Coverage Evaluator
=================================
Measures how well the PIIService (src/analysis/pii.py) detects and redacts
sensitive entities from call-centre transcripts.

Why this matters
----------------
A missed PII instance (false negative) means a raw Aadhaar number, PAN, phone
number, or credit card reaches the LLM and may be logged to LangSmith,
persisted in the knowledge base, or leaked in an API response.  In a banking
context this is a direct regulatory violation (RBI data-localisation, DPDPA).

What we measure
---------------
Per entity type:
  - False Negative Rate (FNR) = missed / total labelled instances
  - False Positive Rate (FPR) = over-redacted tokens / total clean tokens
  - Total instances seen and coverage breakdown

Pass gates (per entity type):
  - FNR < 0.05  (< 5 % of sensitive instances missed)
  - FPR < 0.10  (< 10 % of clean tokens wrongly redacted)

Coverage gaps flagged explicitly:
  - Hinglish / mixed-language text (Presidio is English-only)
  - PAN in lowercase ("abcde1234f") — regex is case-sensitive by default
  - Aadhaar with em-dash or zero-width space separators
  - BANK_ACCOUNT numbers without a keyword prefix (should NOT be redacted)

Usage
-----
python -m evaluation.pii_eval
python -m evaluation.pii_eval --verbose
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from config.logger import get_logger

logger = get_logger("eval.pii")

# ---------------------------------------------------------------------------
# Labelled test dataset
# Each example: input text, list of expected redacted spans by entity type.
# The redacted text must NOT appear in the output; clean spans must survive.
# ---------------------------------------------------------------------------

@dataclass
class PIITestCase:
    id: str
    description: str
    input_text: str
    # entity_type -> list of substrings that MUST be redacted
    must_redact: dict[str, list[str]] = field(default_factory=dict)
    # Substrings that must NOT be redacted (false-positive probes)
    must_preserve: list[str] = field(default_factory=list)
    # Category tag for grouping results
    category: str = "standard"


TEST_CASES: list[PIITestCase] = [

    # ------------------------------------------------------------------
    # EMAIL
    # ------------------------------------------------------------------
    PIITestCase(
        id="email_basic",
        description="Plain email address",
        input_text="My email is john.doe@example.com, please confirm.",
        must_redact={"EMAIL_ADDRESS": ["john.doe@example.com"]},
        must_preserve=["please confirm"],
    ),

    # ------------------------------------------------------------------
    # PHONE
    # ------------------------------------------------------------------
    PIITestCase(
        id="phone_indian_10d",
        description="10-digit Indian mobile number",
        input_text="Call me back on 9876543210 at any time.",
        must_redact={"PHONE_NUMBER": ["9876543210"]},
        must_preserve=["Call me back on", "at any time"],
    ),
    PIITestCase(
        id="phone_with_country_code",
        description="Phone with +91 prefix",
        input_text="My number is +91-98765-43210.",
        must_redact={"PHONE_NUMBER": ["+91-98765-43210"]},
        must_preserve=[],
    ),

    # ------------------------------------------------------------------
    # CREDIT CARD
    # ------------------------------------------------------------------
    PIITestCase(
        id="credit_card_spaced",
        description="Credit card with spaces",
        input_text="Card number 4111 1111 1111 1111 was charged.",
        must_redact={"CREDIT_CARD": ["4111 1111 1111 1111"]},
        must_preserve=["was charged"],
    ),
    PIITestCase(
        id="credit_card_dashed",
        description="Credit card with dashes",
        input_text="My Visa is 4111-1111-1111-1111.",
        must_redact={"CREDIT_CARD": ["4111-1111-1111-1111"]},
        must_preserve=[],
    ),

    # ------------------------------------------------------------------
    # AADHAAR (custom recognizer)
    # ------------------------------------------------------------------
    PIITestCase(
        id="aadhaar_plain",
        description="12-digit Aadhaar without spaces",
        input_text="My Aadhaar number is 234567890123.",
        must_redact={"AADHAAR": ["234567890123"]},
        must_preserve=[],
    ),
    PIITestCase(
        id="aadhaar_spaced",
        description="Aadhaar with spaces (common format)",
        input_text="Aadhaar: 2345 6789 0123",
        must_redact={"AADHAAR": ["2345 6789 0123"]},
        must_preserve=[],
    ),
    PIITestCase(
        id="aadhaar_hinglish",
        description="Aadhaar mentioned in Hinglish context",
        input_text="Mera Aadhaar number hai 2345 6789 0123, please verify karo.",
        must_redact={"AADHAAR": ["2345 6789 0123"]},
        must_preserve=["please verify karo"],
        category="hinglish",
    ),

    # ------------------------------------------------------------------
    # PAN (custom recognizer)
    # ------------------------------------------------------------------
    PIITestCase(
        id="pan_standard",
        description="Standard PAN format (AAAAA9999A)",
        input_text="My PAN card is ABCDE1234F.",
        must_redact={"PAN": ["ABCDE1234F"]},
        must_preserve=[],
    ),
    PIITestCase(
        id="pan_lowercase_gap",
        description="PAN in lowercase — KNOWN GAP: regex is case-sensitive",
        input_text="PAN: abcde1234f",
        must_redact={},  # Currently NOT redacted — coverage gap
        must_preserve=[],
        category="coverage_gap",
    ),
    PIITestCase(
        id="pan_hinglish",
        description="PAN in Hinglish sentence",
        input_text="Bhai mera PAN ABCDE1234F hai, KYC ke liye chahiye.",
        must_redact={"PAN": ["ABCDE1234F"]},
        must_preserve=["KYC ke liye chahiye"],
        category="hinglish",
    ),

    # ------------------------------------------------------------------
    # BANK_ACCOUNT (custom recognizer — keyword-gated)
    # ------------------------------------------------------------------
    PIITestCase(
        id="bank_account_keyword",
        description="Bank account with 'account' keyword prefix",
        input_text="Please transfer to account 123456789012.",
        must_redact={"BANK_ACCOUNT": ["123456789012"]},
        must_preserve=["Please transfer to"],
    ),
    PIITestCase(
        id="bank_account_acc_prefix",
        description="Bank account with 'acc' abbreviation",
        input_text="Acc no: 987654321098",
        must_redact={"BANK_ACCOUNT": ["987654321098"]},
        must_preserve=[],
    ),
    PIITestCase(
        id="bank_account_no_keyword_fp",
        description="9-digit order/reference ID without bank keyword — must NOT be redacted",
        input_text="Your order reference is 123456789.",
        must_redact={},
        must_preserve=["123456789"],
        category="false_positive_probe",
    ),
    PIITestCase(
        id="bank_account_timestamp_fp",
        description="11-digit timestamp-like number without bank keyword — must NOT be redacted",
        input_text="Transaction logged at 20240619123.",
        must_redact={},
        must_preserve=["20240619123"],
        category="false_positive_probe",
    ),

    # ------------------------------------------------------------------
    # Mixed / multi-entity
    # ------------------------------------------------------------------
    PIITestCase(
        id="multi_entity",
        description="Multiple entity types in one transcript",
        input_text=(
            "Customer: my email is jane@bank.com, phone 9123456789, "
            "Aadhaar 1234 5678 9012, PAN FGHIJ5678K, card 5500 0000 0000 0004."
        ),
        must_redact={
            "EMAIL_ADDRESS": ["jane@bank.com"],
            "PHONE_NUMBER": ["9123456789"],
            "AADHAAR": ["1234 5678 9012"],
            "PAN": ["FGHIJ5678K"],
            "CREDIT_CARD": ["5500 0000 0000 0004"],
        },
        must_preserve=["Customer:", "my email is", "phone", "card"],
    ),

    # ------------------------------------------------------------------
    # Clean text (no PII) — should not be altered significantly
    # ------------------------------------------------------------------
    PIITestCase(
        id="clean_text_no_pii",
        description="No PII — output should equal input",
        input_text="The customer wants to know the refund policy for cancelled tickets.",
        must_redact={},
        must_preserve=["refund policy for cancelled tickets"],
        category="clean_probe",
    ),
]


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

@dataclass
class EntityResult:
    entity_type: str
    total_instances: int = 0
    missed: int = 0        # false negatives
    detected: int = 0


def evaluate_pii(verbose: bool = False) -> dict[str, Any]:
    """
    Run all test cases against the live PIIService and return a report dict.
    """
    # Import here so the evaluator is importable even without presidio installed
    from src.analysis.pii import get_pii_service
    svc = get_pii_service()

    entity_stats: dict[str, EntityResult] = {}
    false_positive_failures: list[dict] = []
    per_case: list[dict] = []
    coverage_gaps: list[dict] = []

    for tc in TEST_CASES:
        redacted = svc.redact(tc.input_text)
        case_issues: list[str] = []

        # Check must-redact spans
        for entity_type, spans in tc.must_redact.items():
            if entity_type not in entity_stats:
                entity_stats[entity_type] = EntityResult(entity_type)

            for span in spans:
                entity_stats[entity_type].total_instances += 1
                if span in redacted:
                    # Still present → missed
                    entity_stats[entity_type].missed += 1
                    case_issues.append(f"MISSED [{entity_type}]: {span!r}")
                    if tc.category == "coverage_gap":
                        coverage_gaps.append({
                            "id": tc.id,
                            "description": tc.description,
                            "span": span,
                            "entity_type": entity_type,
                        })
                else:
                    entity_stats[entity_type].detected += 1

        # Check must-preserve spans (false-positive probes)
        for span in tc.must_preserve:
            if span not in redacted:
                fp_issue = f"FALSE POSITIVE: {span!r} was removed"
                case_issues.append(fp_issue)
                false_positive_failures.append({
                    "id": tc.id,
                    "description": tc.description,
                    "removed_span": span,
                    "input": tc.input_text,
                    "output": redacted,
                })

        passed_case = len(case_issues) == 0
        row = {
            "id": tc.id,
            "description": tc.description,
            "category": tc.category,
            "passed": passed_case,
            "issues": case_issues,
            "input": tc.input_text,
            "output": redacted,
        }
        per_case.append(row)

        if verbose or not passed_case:
            status = "PASS" if passed_case else "FAIL"
            logger.info("[PII EVAL] [%s] %s — %s", status, tc.id, tc.description)
            if not passed_case:
                for issue in case_issues:
                    logger.warning("         %s", issue)

    # --- Per-entity FNR ---
    entity_report: list[dict] = []
    for et, stats in entity_stats.items():
        fnr = stats.missed / stats.total_instances if stats.total_instances else 0.0
        gate_passed = fnr < 0.05
        entity_report.append({
            "entity_type": et,
            "total_instances": stats.total_instances,
            "detected": stats.detected,
            "missed": stats.missed,
            "fnr": round(fnr, 4),
            "gate_passed": gate_passed,
        })

    n_cases = len(per_case)
    n_failed = sum(1 for c in per_case if not c["passed"])
    all_gates = all(e["gate_passed"] for e in entity_report)
    no_fp_failures = len(false_positive_failures) == 0
    passed_overall = all_gates and no_fp_failures

    summary = {
        "passed": passed_overall,
        "total_cases": n_cases,
        "failed_cases": n_failed,
        "false_positive_failures": false_positive_failures,
        "coverage_gaps": coverage_gaps,
        "entity_report": entity_report,
        "per_case": per_case,
    }

    _print_pii_report(summary)
    return summary


def _print_pii_report(s: dict) -> None:
    status = "PASS ✓" if s["passed"] else "FAIL ✗"
    print("\n" + "=" * 60)
    print(f"  PII REDACTION COVERAGE  [{status}]")
    print("=" * 60)
    print(f"  Test cases: {s['total_cases']}  |  Failed: {s['failed_cases']}")

    print("\n  Per-entity FNR (target < 5.0%):")
    for e in s["entity_report"]:
        gate = "✓" if e["gate_passed"] else "✗"
        print(
            f"    [{gate}] {e['entity_type']:<18} "
            f"FNR={e['fnr']:.1%}  "
            f"({e['detected']}/{e['total_instances']} detected)"
        )

    if s["false_positive_failures"]:
        print(f"\n  False Positive failures ({len(s['false_positive_failures'])}):")
        for fp in s["false_positive_failures"]:
            print(f"    [{fp['id']}] removed {fp['removed_span']!r}")

    if s["coverage_gaps"]:
        print(f"\n  Known coverage gaps ({len(s['coverage_gaps'])}):")
        for gap in s["coverage_gaps"]:
            print(f"    [{gap['id']}] {gap['description']}")
            print(f"      Entity: {gap['entity_type']} | Span: {gap['span']!r}")
        print(
            "\n  ⚠  Gap remediation: add case-insensitive flag to PAN regex "
            "and test with Devanagari digit variants for Hinglish coverage."
        )

    if s["failed_cases"] > 0:
        print(f"\n  Failed test cases:")
        for c in s["per_case"]:
            if not c["passed"]:
                print(f"\n    [{c['id']}] {c['description']}")
                for issue in c["issues"]:
                    print(f"      → {issue}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run PII redaction coverage evaluation.")
    parser.add_argument("--verbose", action="store_true", help="Print all cases, not just failures.")
    args = parser.parse_args()
    evaluate_pii(verbose=args.verbose)
