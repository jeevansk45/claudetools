# `orchestrator.py` — Explained Line by Line

Same format as the extraction pipeline's explanation file: code, then
plain-language meaning, top to bottom.

---

## 1. Imports and setup - Agent import

```python
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
```
Mostly the same tools as before, plus one new one:
`concurrent.futures` — this is what lets several documents get processed
**at the same time** instead of one after another. `ThreadPoolExecutor`
manages a small pool of worker threads; `as_completed` lets us collect
results as each one finishes, in whatever order they happen to finish in
(not necessarily the order we started them).

```python
from extraction_pipeline import (
    run_field_extraction,
    get_claude_client,
    call_claude_safely,
    _call_claude,
    MODEL,
)
```
This is the key line proving reuse: rather than rewriting Agent 1, this
file **imports** five specific things directly from the earlier
`extraction_pipeline.py` file — the function that runs a full extraction,
the safe-client-builder, the error-handling wrapper, the low-level "call
Claude and parse JSON" helper, and the model name constant. Nothing about
Agent 1 was duplicated or rewritten.

```python
CONFIDENCE_THRESHOLD = 0.7
```
A constant, defined once, used by the validation logic below — any field
extracted with less than 70% confidence gets a second look from Agent 2.
Keeping it here (instead of buried in a function) makes it an easy knob
to find and tune later.

---

## 2. Agent 2: Validation

```python
VALIDATION_SYSTEM_PROMPT = """You are a strict fact-checker.
...
"""
```
Another system prompt string, same idea as before — but notice the
*framing* is deliberately different from the extraction prompt: "fact-
checker" judging whether a sentence supports a claim, not "extractor"
finding a value. This is the design decision from `ARCHITECTURE.md` about
not letting Agent 2 just rubber-stamp Agent 1's own reasoning.

```python
def validate_field(client, field_name: str, claimed_value: str, source_text: str) -> Dict:
    content = [{
        "type": "text",
        "text": f'Field: "{field_name}"\nClaimed value: "{claimed_value}"\nSource sentence: "{source_text}"',
    }]
    return _call_claude(client, VALIDATION_SYSTEM_PROMPT, content)
```
Builds the message Claude will see — just three pieces of information:
which field, what was claimed, and the one sentence it was supposedly
based on. Notice what's *missing*: the whole document, the page number,
anything else. That's intentional — Agent 2 doesn't need it, so it isn't
sent (cheaper and faster). Then hands off to `_call_claude`, the same
low-level function Agent 1 uses — proving both agents share the same
"talk to Claude and parse JSON back" plumbing.

```python
def mock_validate_field(field_name: str, claimed_value: str, source_text: str) -> Dict:
    if claimed_value and source_text and str(claimed_value).lower() in source_text.lower():
        return {"supported": True, "corrected_value": claimed_value, "reasoning": "(mock) value found verbatim in source"}
    return {"supported": False, "corrected_value": claimed_value, "reasoning": "(mock) value not verbatim in source — needs real check"}
```
The free, fake version. `str(claimed_value).lower() in source_text.lower()`
is a simple substring check — "does the exact claimed text appear
somewhere in the source sentence?" `str(...)` guards against a crash if
`claimed_value` happens to not already be text. This is obviously cruder
than real reasoning (it can't tell "the total is $48,500" actually
supports a claim of "$48,500" phrased differently) — but it's a
reasonable free stand-in for demo purposes, and its limitation is
documented right there in the reasoning message.

```python
def run_validation_agent(extraction_result: Dict, mock: bool) -> Dict:
    client = None if mock else get_claude_client()
    validated = {}
```
This is the function that decides **whether** Agent 2 runs at all for
each field. `client = None if mock else get_claude_client()` — only
bother building a real Claude connection if we're not in mock mode (no
point creating one we'll never use).

```python
    for field_name, data in extraction_result.items():
        confidence = data.get("confidence", 0)
        if confidence >= CONFIDENCE_THRESHOLD or not data.get("value"):
            validated[field_name] = data
            continue
```
Loop through every field Agent 1 returned. If its confidence is already
high enough (`>= 0.7`), **or** there's no value to check in the first
place (nothing was found), just copy it through unchanged and move to
the next field (`continue` skips the rest of this loop iteration). This
is the "conditional, not automatic" rule in code form.

```python
        print(f"[validate] '{field_name}' confidence {confidence:.2f} < {CONFIDENCE_THRESHOLD} — checking with Agent 2...")
        if mock:
            check = mock_validate_field(field_name, data.get("value"), data.get("source_text"))
        else:
            check = call_claude_safely(
                validate_field, client, field_name, data.get("value"), data.get("source_text")
            ) or {"supported": False, "corrected_value": data.get("value"), "reasoning": "validation call failed"}
```
Only fields that didn't get skipped above reach this point. `{:.2f}`
formats the confidence number to two decimal places for a cleaner log
line. Then either call the mock or the real validator — note the real
call goes through `call_claude_safely` (the same error-handling wrapper
Agent 1 uses), and if that call fails or returns nothing, `or {...}`
supplies a safe fallback result instead of crashing.

```python
        validated[field_name] = {
            **data,
            "value": check.get("corrected_value", data.get("value")),
            "validated": check.get("supported", False),
            "validation_note": check.get("reasoning"),
        }

    return validated
```
Build the final entry for this field: `{**data, ...}` copies everything
Agent 1 originally said about this field (its old value, confidence,
source text) and then *overwrites* a few keys on top — the (possibly
corrected) value, whether it was validated, and why. This way nothing
from Agent 1's original result is silently lost, it's just updated.

---

## 3. Agent 3: Synthesis

```python
SYNTHESIS_SYSTEM_PROMPT = """You are a synthesis assistant...
"""
```
A third distinct system prompt — this one's job is explicitly to combine
*already-structured* JSON from multiple documents into one readable
report, not to look at any raw document text at all.

```python
def run_synthesis_agent(client, document_results: List[Dict]) -> str:
    content = [{"type": "text", "text": json.dumps(document_results, indent=2)}]
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=[{"type": "text", "text": SYNTHESIS_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": content}],
    )
    return response.content[0].text
```
Turns the whole list of per-document results into one JSON string
(`json.dumps`) and sends that as the entire input. This is a much smaller
and cheaper API call than re-reading any actual PDFs — by this point, all
the expensive document-reading work is already done, and Agent 3 is just
reasoning over a summary. Note this call is made directly here (not
through `_call_claude`), because the response this time is meant to be
plain text, not JSON to parse — `response.content[0].text` is returned as-is.

```python
def mock_synthesis(document_results: List[Dict]) -> str:
    lines = [f"(mock synthesis) Processed {len(document_results)} document(s):\n"]
    for doc in document_results:
        name = doc["document"]
        fields = doc["fields"]
        summary = ", ".join(f"{k}={v.get('value')}" for k, v in fields.items())
        lines.append(f"- {name}: {summary}")
```
The mock version builds up a report line by line instead of asking
Claude to write it. `", ".join(f"{k}={v.get('value')}" for k, v in
fields.items())` is a compact way of writing "for every field name and
its data, make a `name=value` string, and join them all with commas" —
turning `{"recipient": {"value": "Acme"}}` into `"recipient=Acme"`.

```python
        for k, v in fields.items():
            if v.get("validated") is False:
                lines.append(f"    ⚠ '{k}' failed validation: {v.get('validation_note')}")
    return "\n".join(lines)
```
For each field, specifically check `is False` (not just "falsy") — this
matters because a field that was *never* sent to Agent 2 has no
`"validated"` key at all, so `.get("validated")` returns `None`, and
`None is False` is `False` — so untouched fields are correctly not
flagged. Only fields Agent 2 actually checked *and* rejected get the
warning line. `"\n".join(lines)` glues all the report lines into one
final string, one per line.

---

## 4. The Orchestrator itself

```python
def extract_one_document(pdf_path: str, fields: List[str], mock: bool, top_n: int) -> Dict:
    try:
        result = run_field_extraction(pdf_path, fields, mock=mock, top_n=top_n)
        return {"document": os.path.basename(pdf_path), "status": "ok", "fields": result}
    except Exception as e:
        return {"document": os.path.basename(pdf_path), "status": "failed", "error": str(e)}
```
This wraps Agent 1's extraction for exactly *one* document, with a
`try/except` around the whole thing. `os.path.basename(pdf_path)` strips
any folder path and keeps just the filename (e.g.
`"/some/folder/doc.pdf"` becomes `"doc.pdf"`) for cleaner reporting.
Critically: if extraction throws *any* exception — a broken PDF, a
missing file, anything — this function catches it and returns a normal
`"status": "failed"` dictionary instead of letting the crash propagate
upward. This is what makes failure isolation possible: each document's
outcome, good or bad, becomes a plain piece of data to handle later,
never a program-ending crash.

```python
def run_orchestrator(pdf_paths: List[str], fields: List[str], mock: bool = False, top_n: int = 5) -> Dict:
    print(f"=== Agent 1: Extraction (parallel across {len(pdf_paths)} document(s)) ===")
    extraction_results = []
    with ThreadPoolExecutor(max_workers=min(4, len(pdf_paths))) as pool:
        futures = {pool.submit(extract_one_document, path, fields, mock, top_n): path for path in pdf_paths}
        for future in as_completed(futures):
            extraction_results.append(future.result())
```
This is the parallel fan-out. `ThreadPoolExecutor(max_workers=min(4,
len(pdf_paths)))` creates a pool of worker threads — capped at 4, or
fewer if there are fewer than 4 documents (no point creating unused
workers). `pool.submit(...)` hands each document off to the pool and
immediately gets back a "future" — a placeholder for a result that isn't
ready yet. Doing this inside a dictionary comprehension (`{pool.submit(...):
path for path in pdf_paths}`) starts *all* the documents processing at
once, rather than waiting for each one before starting the next.
`as_completed(futures)` then lets us collect each result as soon as it's
actually done, in whatever order that happens to be — `future.result()`
retrieves the actual returned dictionary from `extract_one_document`.

```python
    succeeded = [r for r in extraction_results if r["status"] == "ok"]
    failed = [r for r in extraction_results if r["status"] == "failed"]

    if failed:
        print(f"[warn] {len(failed)} document(s) failed extraction and will be excluded: "
              f"{[f['document'] for f in failed]}")
```
Splits every result into two lists based on its `status`. If anything
failed, print a warning listing exactly which document names failed —
visible immediately, rather than silently disappearing from the rest of
the pipeline.

```python
    print(f"\n=== Agent 2: Validation (only fields below confidence {CONFIDENCE_THRESHOLD}) ===")
    for doc in succeeded:
        doc["fields"] = run_validation_agent(doc["fields"], mock=mock)
```
Only the **successful** documents move on to Agent 2 — failed ones are
already set aside. For each surviving document, replace its `"fields"`
entry with the validated version (some fields unchanged, some corrected —
whatever `run_validation_agent` decided).

```python
    print(f"\n=== Agent 3: Synthesis (combining {len(succeeded)} validated document(s)) ===")
    if mock:
        report = mock_synthesis(succeeded)
    else:
        client = get_claude_client()
        report = call_claude_safely(run_synthesis_agent, client, succeeded) or "(synthesis call failed)"
```
Same mock/real branching pattern as before. Note `succeeded` at this
point contains every document's *validated* fields, not the original raw
extraction — Agent 3 always sees the most up-to-date version of the data.

```python
    return {
        "report": report,
        "documents_processed": [d["document"] for d in succeeded],
        "documents_failed": [{"document": f["document"], "error": f["error"]} for f in failed],
        "per_document_results": succeeded,
    }
```
Packages everything the caller might want: the final written report, a
simple list of which documents made it all the way through, a list of
which failed and why, and the full detailed per-document data for anyone
who wants to inspect it further.

---

## 5. Command-line entry point

```python
def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Document Intelligence System")
    parser.add_argument("pdf_paths", nargs="+", help="One or more PDF files to process")
```
`nargs="+"` is the key difference from the single-document pipeline's
`main()` — it means "accept one *or more* values here," so
`pdf_paths` becomes a list even if you only pass one file, and you can
list as many PDFs as you want on the command line.

```python
    parser.add_argument("--fields", required=True, help="Comma-separated list of fields to extract")
    parser.add_argument("--top-n", type=int, default=5, help="Max candidate pages per document for Agent 1")
    parser.add_argument("--mock", action="store_true", help="Force mock mode (no API calls)")
    args = parser.parse_args()
```
Same style of options as the single-document version — required fields
list, an optional page-count limit, and a mock-mode switch. Note there's
no `--query` mode here — this orchestrator is currently built only around
field extraction, not open-ended questions (that would be a reasonable
future addition).

```python
    mock = args.mock or not os.environ.get("ANTHROPIC_API_KEY")
    if mock and not args.mock:
        print("[info] No ANTHROPIC_API_KEY found — running in mock mode.\n")

    fields = [f.strip() for f in args.fields.split(",")]
    result = run_orchestrator(args.pdf_paths, fields, mock=mock, top_n=args.top_n)
```
Identical logic to before: decide mock mode, parse the comma-separated
field list, then hand everything off to `run_orchestrator` to actually do
the work.

```python
    print("\n=== Final Combined Report ===")
    print(result["report"])
    print("\n=== Run Summary ===")
    print(json.dumps(
        {"documents_processed": result["documents_processed"], "documents_failed": result["documents_failed"]},
        indent=2,
    ))
```
Prints the human-readable report first, then a compact JSON summary of
just which documents succeeded and failed — useful for a quick glance
without reading the whole report, or for another program to parse this
output automatically later.

```python
if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"\n[error] {e}", file=sys.stderr)
        sys.exit(1)
```
Same top-level safety net as the extraction pipeline — if anything raises
a `RuntimeError` anywhere in the whole three-agent run (like a missing
API key), it's caught here and printed as one clean line instead of a
full crash trace, and the program exits with status code 1 (failure).
