"""
Multi-Agent Document Intelligence System
------------------------------------------
Coordinates three agents across multiple documents:
  Agent 1 (Extraction) -> Agent 2 (Validation, conditional) -> Agent 3 (Synthesis)

Reuses extraction_pipeline.py's Agent 1 unchanged. See ARCHITECTURE.md for
the design rationale behind the fan-out/fan-in pattern and why validation
only runs on low-confidence fields.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

# Agent 1 is the existing pipeline, reused as-is.
from extraction_pipeline import (
    run_field_extraction,
    get_claude_client,
    call_claude_safely,
    _call_claude,
    MODEL,
)

CONFIDENCE_THRESHOLD = 0.7  # fields below this get sent to Agent 2


# --------------------------------------------------------------------------
# Agent 2: Validation
# --------------------------------------------------------------------------

VALIDATION_SYSTEM_PROMPT = """You are a strict fact-checker.

You will be given a claimed field value and the exact sentence it was
supposedly drawn from. Your only job is to judge whether that sentence
actually, unambiguously supports the claimed value — not to re-extract
or improve it.

Return valid JSON only, no other text, in this schema:
{
  "supported": true/false,
  "corrected_value": "..." ,
  "reasoning": "one short sentence"
}

If the source sentence clearly supports the claim, set supported to true
and corrected_value to the same value. If it does NOT support the claim
(wrong, ambiguous, or the sentence doesn't actually contain it), set
supported to false and corrected_value to your best correction based
ONLY on the sentence given — or null if the sentence doesn't contain the
answer at all.
"""


def validate_field(client, field_name: str, claimed_value: str, source_text: str) -> Dict:
    """Agent 2's real Claude call. Deliberately minimal input — just the
    claim and its source, not the whole document — since that's all a
    fact-check needs."""
    content = [{
        "type": "text",
        "text": f'Field: "{field_name}"\nClaimed value: "{claimed_value}"\nSource sentence: "{source_text}"',
    }]
    return _call_claude(client, VALIDATION_SYSTEM_PROMPT, content)


def mock_validate_field(field_name: str, claimed_value: str, source_text: str) -> Dict:
    """Deterministic stand-in: 'validates' by checking the claimed value's
    words actually appear in the source sentence — a crude but zero-cost
    proxy for what a real fact-check call would reason about properly."""
    if claimed_value and source_text and str(claimed_value).lower() in source_text.lower():
        return {"supported": True, "corrected_value": claimed_value, "reasoning": "(mock) value found verbatim in source"}
    return {"supported": False, "corrected_value": claimed_value, "reasoning": "(mock) value not verbatim in source — needs real check"}


def run_validation_agent(extraction_result: Dict, mock: bool) -> Dict:
    """Walks every field in an extraction result; only calls Agent 2 for
    fields below CONFIDENCE_THRESHOLD. High-confidence fields pass through
    untouched — this is the 'conditional, not automatic' design decision
    from ARCHITECTURE.md."""
    client = None if mock else get_claude_client()
    validated = {}

    for field_name, data in extraction_result.items():
        confidence = data.get("confidence", 0)
        if confidence >= CONFIDENCE_THRESHOLD or not data.get("value"):
            validated[field_name] = data
            continue

        print(f"[validate] '{field_name}' confidence {confidence:.2f} < {CONFIDENCE_THRESHOLD} — checking with Agent 2...")
        if mock:
            check = mock_validate_field(field_name, data.get("value"), data.get("source_text"))
        else:
            check = call_claude_safely(
                validate_field, client, field_name, data.get("value"), data.get("source_text")
            ) or {"supported": False, "corrected_value": data.get("value"), "reasoning": "validation call failed"}

        validated[field_name] = {
            **data,
            "value": check.get("corrected_value", data.get("value")),
            "validated": check.get("supported", False),
            "validation_note": check.get("reasoning"),
        }

    return validated


# --------------------------------------------------------------------------
# Agent 3: Synthesis
# --------------------------------------------------------------------------

SYNTHESIS_SYSTEM_PROMPT = """You are a synthesis assistant that combines
structured extraction results from multiple documents into one report.

You will receive a JSON list, one entry per document, each with the
document name and its extracted/validated fields. Write a short combined
report in plain text (not JSON) that:
- Notes how many documents were processed
- Summarizes each document's key fields briefly
- Flags any numeric totals across documents (e.g. sum of amounts) if the
  fields suggest it makes sense
- Flags any fields that failed validation, by document name
- Notes any documents that are missing from the input (they will be
  listed separately as failed, not included in the JSON you receive)

Keep it concise — a few short paragraphs, not a page.
"""


def run_synthesis_agent(client, document_results: List[Dict]) -> str:
    content = [{"type": "text", "text": json.dumps(document_results, indent=2)}]
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=[{"type": "text", "text": SYNTHESIS_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": content}],
    )
    return response.content[0].text


def mock_synthesis(document_results: List[Dict]) -> str:
    lines = [f"(mock synthesis) Processed {len(document_results)} document(s):\n"]
    for doc in document_results:
        name = doc["document"]
        fields = doc["fields"]
        summary = ", ".join(f"{k}={v.get('value')}" for k, v in fields.items())
        lines.append(f"- {name}: {summary}")
        for k, v in fields.items():
            if v.get("validated") is False:
                lines.append(f"    ⚠ '{k}' failed validation: {v.get('validation_note')}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

def extract_one_document(pdf_path: str, fields: List[str], mock: bool, top_n: int) -> Dict:
    """Runs Agent 1 for a single document. Wrapped so ThreadPoolExecutor
    can run several of these concurrently, and so any exception here is
    caught per-document rather than killing the whole batch."""
    try:
        result = run_field_extraction(pdf_path, fields, mock=mock, top_n=top_n)
        return {"document": os.path.basename(pdf_path), "status": "ok", "fields": result}
    except Exception as e:
        return {"document": os.path.basename(pdf_path), "status": "failed", "error": str(e)}


def run_orchestrator(pdf_paths: List[str], fields: List[str], mock: bool = False, top_n: int = 5) -> Dict:
    print(f"=== Agent 1: Extraction (parallel across {len(pdf_paths)} document(s)) ===")
    extraction_results = []
    with ThreadPoolExecutor(max_workers=min(4, len(pdf_paths))) as pool:
        futures = {pool.submit(extract_one_document, path, fields, mock, top_n): path for path in pdf_paths}
        for future in as_completed(futures):
            extraction_results.append(future.result())

    succeeded = [r for r in extraction_results if r["status"] == "ok"]
    failed = [r for r in extraction_results if r["status"] == "failed"]

    if failed:
        print(f"[warn] {len(failed)} document(s) failed extraction and will be excluded: "
              f"{[f['document'] for f in failed]}")

    print(f"\n=== Agent 2: Validation (only fields below confidence {CONFIDENCE_THRESHOLD}) ===")
    for doc in succeeded:
        doc["fields"] = run_validation_agent(doc["fields"], mock=mock)

    print(f"\n=== Agent 3: Synthesis (combining {len(succeeded)} validated document(s)) ===")
    if mock:
        report = mock_synthesis(succeeded)
    else:
        client = get_claude_client()
        report = call_claude_safely(run_synthesis_agent, client, succeeded) or "(synthesis call failed)"

    return {
        "report": report,
        "documents_processed": [d["document"] for d in succeeded],
        "documents_failed": [{"document": f["document"], "error": f["error"]} for f in failed],
        "per_document_results": succeeded,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Document Intelligence System")
    parser.add_argument("pdf_paths", nargs="+", help="One or more PDF files to process")
    parser.add_argument("--fields", required=True, help="Comma-separated list of fields to extract")
    parser.add_argument("--top-n", type=int, default=5, help="Max candidate pages per document for Agent 1")
    parser.add_argument("--mock", action="store_true", help="Force mock mode (no API calls)")
    args = parser.parse_args()

    mock = args.mock or not os.environ.get("ANTHROPIC_API_KEY")
    if mock and not args.mock:
        print("[info] No ANTHROPIC_API_KEY found — running in mock mode.\n")

    fields = [f.strip() for f in args.fields.split(",")]
    result = run_orchestrator(args.pdf_paths, fields, mock=mock, top_n=args.top_n)

    print("\n=== Final Combined Report ===")
    print(result["report"])
    print("\n=== Run Summary ===")
    print(json.dumps(
        {"documents_processed": result["documents_processed"], "documents_failed": result["documents_failed"]},
        indent=2,
    ))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"\n[error] {e}", file=sys.stderr)
        sys.exit(1)
