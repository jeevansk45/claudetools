# Multi-Agent Document Intelligence System

Three Claude agents coordinated by one orchestrator: **Extraction** (reused
from the [document extraction pipeline](../doc-extraction-pipeline)) →
**Validation** (conditional fact-checking) → **Synthesis** (combined
report across all documents).

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full design rationale —
especially *why* three agents instead of one big prompt, and how failures
are isolated per-document.

## Quickstart

```bash
pip install anthropic pdfplumber pdf2image scikit-learn python-dotenv

# Mock mode — no API key needed, see the full flow for free:
python orchestrator.py sample_document.pdf sample_document_2.pdf \
    --fields "recipient name,contract date,total amount" --mock

# Real Claude API:
export ANTHROPIC_API_KEY=sk-ant-...
python orchestrator.py sample_document.pdf sample_document_2.pdf \
    --fields "recipient name,contract date,total amount"
```

Two synthetic sample contracts are included, deliberately similar in
structure but different in values (different recipients, dates, amounts)
so you can see Agent 3 actually combine and compare across them.

## What each demo run proves

- **Parallel extraction**: watch the log output interleave between the two
  documents — Agent 1 runs concurrently, not one-at-a-time.
- **Conditional validation**: Agent 2 only fires on fields below the
  confidence threshold (0.7). In mock mode, run
  `python -c "from orchestrator import mock_validate_field; print(mock_validate_field(...))"`
  directly to see it catch an unsupported claim.
- **Failure isolation**: pass a nonexistent file alongside a real one —
  the orchestrator excludes it and keeps going, and the final report
  explicitly lists it as failed rather than silently ignoring it:
  ```bash
  python orchestrator.py sample_document.pdf does_not_exist.pdf --fields "recipient name" --mock
  ```

## Project structure

```
orchestrator.py            # the multi-agent coordination logic (this project's core)
extraction_pipeline.py     # Agent 1, reused unchanged from the earlier project
ARCHITECTURE.md            # design decisions — read this first
sample_document.pdf        # test document 1
sample_document_2.pdf      # test document 2 (different values, same structure)
```

## Next steps (natural extensions, not yet built)

- Retry-with-backoff for transient extraction failures, instead of
  excluding on the first error
- A real task queue (Celery/SQS) in place of `ThreadPoolExecutor`, for
  processing hundreds/thousands of documents rather than a handful
- A fourth agent: a **router** that decides which fields even need
  extraction per document type, before Agent 1 runs — useful once
  documents aren't all the same template
