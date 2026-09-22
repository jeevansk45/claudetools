# Smart Document Extraction Pipeline — Architecture

## Problem Statement
Extract user-specified fields (e.g., "recipient name", "contract date") from PDF
documents — text-based or scanned — while minimizing Claude API cost and
keeping hallucination risk visible via confidence scoring.

## Design Goals (in priority order)
1. **Cost efficiency** — don't pay to process pages that don't matter
2. **Flexibility** — any field, any document, no hardcoded schemas
3. **Trustworthy output** — every extracted value is traceable to source text

## High-Level Flow

```
PDF Upload
    │
    ▼
[1] Document Intake ──────► detect: text-based or scanned?
    │
    ▼
[2] Smart Chunking ───────► split into page-level chunks,
    │                       keep 1-page overlap for cross-page fields
    ▼
[3] Candidate Page Selection ► lightweight keyword/embedding pass
    │                          to rank which chunks likely hold the
    │                          requested fields (skip the rest)
    ▼
[4] Extraction Engine ────► Claude API call per candidate chunk,
    │                       system prompt is cached (prompt caching),
    │                       only chunk content + field list varies
    ▼
[5] Merge & Validate ─────► combine per-chunk results, resolve
    │                       conflicts, attach confidence + source_text
    ▼
Structured JSON Output
```

## Key Architectural Decisions

### 1. Intake: text vs. scanned detection
`pdfplumber` extracts text per page. If a page yields near-zero extractable
characters, it's flagged `likely_scanned`. Only flagged pages are rendered
to an image (via `pdf2image`) and sent to Claude as a vision input — every
other page is sent as plain text. This keeps the (slower, pricier) vision
path limited to pages that actually need it, rather than applying it
uniformly across the document.

### 2. Chunking: page-level with overlap
Chunking at the page boundary (rather than fixed token windows) keeps
document structure legible to the model and simplifies "source page"
citations in the output. A 1-page overlap catches fields that straddle a
page break (e.g., a signature block starting on the last line of page 4).

### 3. Candidate selection before extraction (the cost lever)
This is the highest-leverage optimization: instead of sending the whole
document to Claude, a cheap local pass (regex + keyword proximity, no LLM
call) ranks pages by likelihood of containing each requested field, and only
the top-N candidate pages are sent to Claude. For a 100-page contract where
you want "effective date" and "signatory," this can cut input tokens by
80-90%.

### 4. Prompt caching
The system prompt (extraction instructions + output schema) is identical
across every call for a given field-set. It's marked `cache_control:
ephemeral`, so only the *first* call in a session pays full price — every
subsequent call against the same field-set gets the ~90% cache discount on
that portion of input tokens. This matters most at volume, but the pattern
is worth building correctly from the start.

### 5. Two extraction modes, one shared pipeline
Field extraction (`--fields`) and open-ended Q&A (`--query`) are different
problems at the ranking step — a fixed field name can be keyword-matched,
but a question like "what happens if the client terminates early?" rarely
shares vocabulary with the answer. So query mode swaps the keyword ranker
for TF-IDF + cosine similarity between the question and each page's text —
a local, dependency-light stand-in for a real embedding search (Voyage AI
or OpenAI embeddings, in production). Everything downstream — the cached
system prompt, the confidence/source_text output shape, the vision routing
for scanned pages — is identical between the two modes. Only the ranking
function and the prompt template differ.

### 6. Confidence scoring instead of multi-pass validation
Rather than a second Claude call to "check" the first one (2x cost), the
extraction prompt itself requires the model to return a confidence score and
the exact source sentence it drew from. Low-confidence or missing-source
answers are flagged for human review — cheap, and it surfaces likely
hallucinations without a second API call.

## Tradeoffs Made
- **Candidate selection is heuristic, not semantic** (no embeddings/vector
  DB) to keep this dependency-light for a portfolio project. A production
  version would swap the keyword ranker for embedding similarity search.
- **No human-in-the-loop UI** — confidence scores are surfaced in the JSON
  output; wiring them to a review queue is a natural next step.
- **Batch mode is stubbed, not built** — the single-document path is fully
  implemented; batching many documents through the same cached prompt is
  the next extension (see README "Next Steps").

## What This Demonstrates
- Cost-aware system design (candidate selection + prompt caching working
  together, not just one or the other)
- Structured, verifiable LLM output (confidence + source citation as
  first-class fields, not an afterthought)
- Sensible handling of a real messy input (mixed text/scanned PDFs)
