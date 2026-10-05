# Multi-Agent Document Intelligence System — Architecture

## Problem Statement
Process multiple documents into one trustworthy, combined report — using
three cooperating agents instead of one big prompt — while keeping cost
and failure handling sane as the number of documents grows.

## Why three agents instead of one
A single Claude call *could* try to "extract, double-check, and summarize"
all in one prompt. Splitting it into three agents buys three things a
single call can't:
1. **Independent scaling** — extraction runs in parallel across documents;
   validation only runs on the subset that actually needs it; synthesis
   runs once at the end. A monolithic prompt can't selectively spend more
   effort on the uncertain 20% of results.
2. **Failure isolation** — one document's extraction failing doesn't take
   down the whole run.
3. **Different framings for different jobs** — extraction is "find this,"
   validation is "fact-check this claim against its source," synthesis is
   "combine these already-structured results." Mixing all three into one
   prompt makes each one worse at its job.

## Agent Roles

### Agent 1 — Extraction (reused, unchanged)
This is `extraction_pipeline.py` exactly as built in the document
extraction project. No changes — it's used here as a callable component,
proving the earlier project was built as a reusable piece, not a one-off
script.

### Agent 2 — Validation
**Input:** only the fields Agent 1 returned with confidence below a
threshold (default 0.7) — not the whole document, not the high-confidence
fields. **Job:** given a claimed value and the exact sentence Agent 1
cited as its source, answer one question: does that sentence actually,
unambiguously support that value? This is a fact-check framing, not a
second extraction attempt — deliberately different wording from Agent 1's
prompt so it isn't just confirming Agent 1's own reasoning.

### Agent 3 — Synthesis
**Input:** the validated, structured results from every document that
succeeded (JSON, not raw PDF text — cheap, since the heavy lifting already
happened). **Job:** produce one combined narrative report across all
documents — totals, common fields, discrepancies between documents,
explicit notes on any document that failed and was excluded.

## Orchestration Flow

```
documents = [doc1.pdf, doc2.pdf, ...]
    │
    ▼
┌─────────────────────────────────────────┐
│  FAN-OUT: Agent 1 runs in parallel        │
│  across all documents (ThreadPoolExecutor)│
│  — one document's failure doesn't block   │
│    the others                             │
└──────────────────┬────────────────────────┘
                   │
                   ▼
        for each successful extraction:
        ┌────────────────────────────┐
        │ any field confidence < 0.7? │
        │  yes → Agent 2 (validation) │
        │  no  → pass through as-is   │
        └────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  FAN-IN: Agent 3 runs once, given every   │
│  successfully-processed document's        │
│  validated results                        │
└──────────────────┬────────────────────────┘
                   │
                   ▼
         Final combined report
    (+ list of any documents that failed)
```

## Key Decisions

### 1. Parallel extraction, not sequential
Documents don't depend on each other for Agent 1's work, so
`ThreadPoolExecutor` runs several extractions concurrently. This is the
first genuinely new problem multi-agent work introduces: with a single
document, you never had to think about "what if one of several
concurrent operations fails while the others succeed?"

### 2. Validation is conditional, not automatic
Running Agent 2 on every field would double the API cost for no benefit
on fields Agent 1 was already confident about. The 0.7 threshold is a
tunable knob — lower it for stricter fact-checking, raise it to save cost
and only catch the worst cases.

### 3. Context handed to each agent is minimal, not cumulative
Agent 2 never sees the full document — just the claim and its cited
source sentence. Agent 3 never sees any raw document text — just
structured JSON from Agents 1 and 2. Each handoff is the smallest slice
of context that agent needs to do its job, not "everything so far."

### 4. Failure handling is per-document, not all-or-nothing
The orchestrator tracks which documents succeeded and which failed at
each stage. Agent 3's report explicitly lists excluded documents rather
than silently proceeding as if they didn't exist — a design choice that
matters more as document count grows.

## Tradeoffs Made
- **Fixed 0.7 confidence threshold**, not adaptive — a production version
  might tune this per field type (e.g. dates need higher certainty than
  free-text descriptions).
- **No retry logic on transient failures** — a failed extraction is
  logged and excluded, not automatically retried. Retry-with-backoff is a
  natural next addition, not included here to keep the orchestration
  pattern itself legible.
- **Threading, not a task queue** — fine for a handful of documents in a
  demo; a production version processing thousands would use a real queue
  (Celery, SQS) instead of an in-process thread pool.
