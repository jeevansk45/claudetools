# Smart Document Extraction Pipeline

A cost-optimized architecture for extracting user-specified fields from PDF
documents using Claude — built to demonstrate real system design tradeoffs
(not just a wrapper around an API call).

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full design rationale.

## What it does

Given a PDF and a list of fields (e.g. `"recipient name,contract date,total
amount"`), the pipeline:

1. Detects which pages are text-based vs. scanned
2. Scores every page for relevance — **without** calling Claude — and only
   sends the top candidates. Two ranking strategies, depending on mode:
   - `--fields`: keyword match against the field names
   - `--query`: TF-IDF/cosine similarity between the question and page text
3. Calls Claude once per candidate page, with instructions cached
   (`cache_control: ephemeral`) so repeat calls are cheaper. Pages flagged
   as scanned are sent as images (vision); everything else is sent as text.
4. Returns a confidence score and the exact source sentence behind every
   value or answer — so hallucinations are visible, not hidden

## Quickstart

```bash
pip install -r requirements.txt

# Field extraction — no API key needed to see the full flow:
python extraction_pipeline.py sample_document.pdf --fields "recipient name,contract date,total amount" --mock

# Open-ended question — same flag pattern, different mode:
python extraction_pipeline.py sample_document.pdf --query "What are the termination conditions?" --mock
```

## Running against the real Claude API

```bash
cp .env.example .env
# edit .env and paste in your real ANTHROPIC_API_KEY

python extraction_pipeline.py sample_document.pdf --fields "recipient name,contract date,total amount"
# (drop --mock — the pipeline auto-detects the key via python-dotenv)
```

This is the actual production code path — `get_claude_client()` builds a
real `anthropic.Anthropic()` client and every call goes through
`call_claude_safely()`, which catches and reports (rather than crashing on):

- **`AuthenticationError`** — bad/expired key → clean error message, exits
  immediately rather than burning through every page first
- **`RateLimitError`** — logs a warning, skips that page's result, keeps
  processing the rest of the document
- **`APIConnectionError`** / **`APIStatusError`** — same graceful-skip
  behavior, so one bad page doesn't take down the whole run

If you don't have a key handy, `--mock` runs the identical control flow
(ranking, merging, output shape) with a rule-based stand-in for the Claude
call — useful for demoing the architecture without incurring any cost.

`sample_document.pdf` is a synthetic 4-page service agreement included for
testing — it has a header page, two filler/terms pages, and a signature
page, so you can see the candidate-page ranking actually skip low-relevance
pages on a longer document.

## Project structure

```
extraction_pipeline.py   # the pipeline itself (intake → chunk → extract → merge)
ARCHITECTURE.md           # design decisions and tradeoffs
sample_document.pdf       # synthetic test document
requirements.txt
.env.example              # copy to .env and add your ANTHROPIC_API_KEY
.gitignore                # excludes .env from version control
```

## Why this is portfolio-worthy (not just an API wrapper)

- **Cost lever is real**: candidate-page selection is a genuine architectural
  choice with a measurable tradeoff (fewer tokens sent vs. risk of missing a
  field on a skipped page) — not a cosmetic feature.
- **Trust is designed in**: every extracted value carries its own evidence
  (`source_text`) and a confidence score, so the output is auditable.
- **Runs without spending money**: mock mode mirrors the real code path
  exactly (same ranking, same merge logic), so anyone can clone this and see
  it work before deciding to add an API key.

## Next steps (natural extensions, not yet built)

- Swap the TF-IDF ranker for a real embedding model (Voyage AI or OpenAI) —
  the architecture doc explains exactly where that swap happens
- Batch mode: run the same cached prompt across many documents in one pass
- A second Claude call that only re-checks low-confidence fields/answers
  (targeted validation instead of blanket multi-pass extraction)
- Wire this into the **multi-agent workflow** project as the "extraction
  agent" — a natural next step once this piece is solid on its own
