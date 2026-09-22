"""
Smart Document Extraction Pipeline
-----------------------------------
Extracts user-specified fields from a PDF (text-based or scanned) using
Claude, with cost-optimizing candidate-page selection and prompt caching.

Usage:
    python extraction_pipeline.py sample_document.pdf --fields "recipient name,contract date,total amount"

If ANTHROPIC_API_KEY is not set, the pipeline runs in --mock mode
automatically so you can see the full flow without spending anything.
"""

import argparse
import base64
import io
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None

try:
    from dotenv import load_dotenv
    load_dotenv()  # pulls ANTHROPIC_API_KEY from a local .env file if present
except ImportError:
    pass  # dotenv is a convenience, not a hard requirement — env vars still work without it


MODEL = "claude-sonnet-4-5"  # swap for whichever model your account has access to


def get_claude_client():
    """Builds the Anthropic client and fails with a clear, actionable error
    rather than a raw stack trace if the SDK or key is missing/invalid."""
    if anthropic is None:
        raise RuntimeError(
            "The 'anthropic' package isn't installed. Run: pip install anthropic"
        )
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Either export it in your shell, or "
            "copy .env.example to .env and fill it in. Use --mock to run "
            "without a key."
        )
    try:
        return anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Anthropic client: {e}")


def call_claude_safely(fn, *args, **kwargs) -> Dict:
    """Wraps a Claude API call with the error handling a real deployment
    needs: auth failures, rate limits, and transient errors all fail
    loudly and specifically instead of crashing the whole batch on one
    bad page."""
    try:
        return fn(*args, **kwargs)
    except anthropic.AuthenticationError:
        raise RuntimeError("Anthropic API rejected the key — check ANTHROPIC_API_KEY is correct and active.")
    except anthropic.RateLimitError:
        print("[warn] Rate limited by Anthropic API — skipping this page's result.", file=sys.stderr)
        return {}
    except anthropic.APIStatusError as e:
        print(f"[warn] Anthropic API error ({e.status_code}) on this page — skipping. {e.message}", file=sys.stderr)
        return {}
    except anthropic.APIConnectionError:
        print("[warn] Network error reaching Anthropic API — skipping this page's result.", file=sys.stderr)
        return {}


# --------------------------------------------------------------------------
# Step 1: Document Intake
# --------------------------------------------------------------------------

@dataclass
class Page:
    number: int
    text: str
    likely_scanned: bool


def load_pdf(path: str) -> List[Page]:
    """Extract per-page text; flag pages with near-zero text as 'likely scanned'."""
    if pdfplumber is None:
        raise RuntimeError("pdfplumber not installed. pip install pdfplumber")

    pages = []
    with pdfplumber.open(path) as pdf:
        for i, p in enumerate(pdf.pages):
            text = p.extract_text() or ""
            # Heuristic: a normal text page has hundreds of chars.
            # Near-empty extraction on a non-trivial page area suggests
            # the content is an image (scanned) rather than embedded text.
            likely_scanned = len(text.strip()) < 20
            pages.append(Page(number=i + 1, text=text, likely_scanned=likely_scanned))
    return pages


# --------------------------------------------------------------------------
# Step 2/3: Candidate Page Selection (the cost lever)
# --------------------------------------------------------------------------

def rank_candidate_pages(pages: List[Page], fields: List[str], top_n: int = 5) -> List[Page]:
    """
    Cheap, local (no LLM call) heuristic ranking of which pages are most
    likely to contain the requested fields, so we only pay to send the
    top_n pages to Claude instead of the whole document.

    Approach: score each page by how many field-related keywords appear
    near each other, plus a small bonus for early/late pages (headers,
    signature blocks) since key fields cluster there in most documents.
    """
    def keywords_for(f: str) -> List[str]:
        # naive keyword expansion — a production version would use an
        # embedding similarity search instead of hand-rolled synonyms
        base = f.lower().split()
        synonyms = {
            "date": ["date", "dated", "effective", "signed"],
            "name": ["name", "recipient", "party", "signatory"],
            "amount": ["amount", "total", "sum", "$", "usd", "price"],
            "signature": ["signature", "signed", "witness"],
        }
        expanded = set(base)
        for word in base:
            for key, syns in synonyms.items():
                if key in word:
                    expanded.update(syns)
        return list(expanded)

    all_keywords = set()
    for f in fields:
        all_keywords.update(keywords_for(f))

    scored = []
    for page in pages:
        text_lower = page.text.lower()
        score = sum(text_lower.count(kw) for kw in all_keywords)
        # boundary bonus: first/last 2 pages often hold header/signature info
        if page.number <= 2 or page.number > len(pages) - 2:
            score += 2
        # scanned pages can't be scored by keyword — always include as
        # candidates since we can't rule them out cheaply
        if page.likely_scanned:
            score += 1
        scored.append((score, page))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_n]]


def rank_pages_by_query(pages: List[Page], query: str, top_n: int = 5) -> List[Page]:
    """
    Semantic-ish candidate ranking for open-ended questions, where a fixed
    keyword list (rank_candidate_pages) doesn't work because the question's
    wording rarely matches the document's wording directly.

    Uses TF-IDF + cosine similarity as a *local, free* stand-in for a real
    embedding search (e.g. Voyage AI or OpenAI embeddings) — same role in
    the architecture (rank pages before paying for an LLM call), cheaper
    dependency for a portfolio project. Swapping in real embeddings later
    means changing this one function, nothing else in the pipeline.

    Text-only: pages with (nearly) no extracted text score ~0 here and are
    always included as candidates instead, since we can't judge relevance
    of an image without actually looking at it (that's what vision does).
    """
    if TfidfVectorizer is None:
        raise RuntimeError("scikit-learn not installed. pip install scikit-learn")

    text_pages = [p for p in pages if not p.likely_scanned and p.text.strip()]
    scanned_pages = [p for p in pages if p.likely_scanned]

    if not text_pages:
        return scanned_pages[:top_n]

    corpus = [p.text for p in text_pages] + [query]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    query_vec = matrix[-1]
    page_vecs = matrix[:-1]
    similarities = cosine_similarity(query_vec, page_vecs)[0]

    scored = list(zip(similarities, text_pages))
    scored.sort(key=lambda x: x[0], reverse=True)

    ranked_text_pages = [p for _, p in scored[:top_n]]
    # always give scanned pages a chance too, up to remaining budget
    remaining = max(0, top_n - len(ranked_text_pages))
    return ranked_text_pages + scanned_pages[:remaining]


# --------------------------------------------------------------------------
# Step 4: Extraction Engine
# --------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT_TEMPLATE = """You are a precise document field extractor.

Extract ONLY the following fields from the provided document excerpt:
{field_list}

Rules:
- Return valid JSON only, no other text.
- For each field, return: value, confidence (0.0-1.0), source_text (the exact
  sentence or line you drew the value from).
- If a field is not present in this excerpt, set value to null, confidence to
  0.0, and source_text to null. Do NOT guess or infer values that aren't
  explicitly stated.
- confidence should reflect how directly and unambiguously the text states
  the value — not your general certainty.

Output schema:
{{
  "field_name": {{"value": ..., "confidence": ..., "source_text": ...}},
  ...
}}
"""


QUERY_SYSTEM_PROMPT = """You are a precise document question-answering assistant.

Answer the user's question using ONLY the provided document excerpt. Do not
use outside knowledge and do not guess at anything the excerpt doesn't state.

Return valid JSON only, no other text, in this exact schema:
{
  "answer": "...",
  "confidence": 0.0-1.0,
  "source_text": "the exact sentence(s) you drew the answer from",
  "found": true/false
}

If the excerpt does not contain information relevant to the question, set
found to false, answer to null, confidence to 0.0, and source_text to null.
confidence should reflect how directly the excerpt supports the answer, not
your general certainty about the topic.
"""


def build_system_prompt(fields: List[str]) -> str:
    field_list = "\n".join(f"- {f}" for f in fields)
    return EXTRACTION_SYSTEM_PROMPT_TEMPLATE.format(field_list=field_list)


def render_page_as_image_b64(pdf_path: str, page_number: int) -> Optional[str]:
    """Render a single PDF page to a base64 PNG for vision-based extraction.
    Used only for pages flagged likely_scanned, so text-based pages never
    pay the extra cost/latency of a vision call."""
    if convert_from_path is None:
        return None
    images = convert_from_path(
        pdf_path, first_page=page_number, last_page=page_number, dpi=150
    )
    if not images:
        return None
    buf = io.BytesIO()
    images[0].save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_user_content(pdf_path: str, page: Page, instruction: str) -> list:
    """Builds the message content for a page — text block for normal pages,
    an image block for pages flagged likely_scanned (vision path)."""
    if page.likely_scanned:
        image_b64 = render_page_as_image_b64(pdf_path, page.number)
        if image_b64 is None:
            # Vision rendering unavailable — fall back to whatever text
            # extraction found (likely near-empty), rather than crashing.
            return [{"type": "text", "text": f"{instruction} (page {page.number}, text-only fallback):\n\n{page.text}"}]
        return [
            {"type": "text", "text": f"{instruction} (page {page.number}, image below):"},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": image_b64},
            },
        ]
    return [{"type": "text", "text": f"{instruction} (page {page.number}):\n\n{page.text}"}]


def _call_claude(client, system_prompt: str, content: list) -> Dict:
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": content}],
    )
    raw = response.content[0].text
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {}


def extract_from_page(client, system_prompt: str, page: Page, pdf_path: str) -> Dict:
    """Single Claude call for one candidate page — field extraction mode.
    System prompt is cached across calls for the same field-set. Routes to
    vision automatically if the page was flagged likely_scanned."""
    content = _build_user_content(pdf_path, page, "Document excerpt")
    return _call_claude(client, system_prompt, content)


def answer_query_from_page(client, system_prompt: str, page: Page, pdf_path: str, query: str) -> Dict:
    """Single Claude call for one candidate page — open-ended query mode."""
    content = _build_user_content(pdf_path, page, f"Question: {query}\n\nDocument excerpt")
    return _call_claude(client, system_prompt, content)


def mock_extract_from_page(fields: List[str], page: Page) -> Dict:
    """Deterministic stand-in for extract_from_page so the pipeline can be
    demoed with zero API cost. Simulates finding fields via keyword match."""
    result = {}
    for f in fields:
        key = f.strip().lower()
        found = None
        for line in page.text.splitlines():
            if any(word in line.lower() for word in key.split()):
                found = line.strip()
                break
        if found:
            result[f] = {"value": found, "confidence": 0.8, "source_text": found}
        else:
            result[f] = {"value": None, "confidence": 0.0, "source_text": None}
    return result


def mock_answer_query_from_page(page: Page, query: str) -> Dict:
    """Deterministic stand-in for answer_query_from_page. Simulates relevance
    by checking whether any query word appears in the page text, and if so
    returns the surrounding sentence as a fake 'answer'. A real Claude call
    would actually reason over the text instead of just matching words."""
    query_words = [w for w in query.lower().split() if len(w) > 3]
    for line in page.text.splitlines():
        line_lower = line.lower()
        if any(w in line_lower for w in query_words):
            return {
                "answer": f"(mock) Found relevant text on page {page.number}: {line.strip()}",
                "confidence": 0.6,
                "source_text": line.strip(),
                "found": True,
            }
    return {"answer": None, "confidence": 0.0, "source_text": None, "found": False}


# --------------------------------------------------------------------------
# Step 5: Merge & Validate
# --------------------------------------------------------------------------

def merge_results(per_page_results: List[Dict]) -> Dict:
    """For each field, keep the highest-confidence non-null answer found
    across all candidate pages."""
    merged: Dict[str, Dict] = {}
    for page_result in per_page_results:
        for field_name, data in page_result.items():
            if field_name not in merged or data.get("confidence", 0) > merged[field_name].get("confidence", 0):
                merged[field_name] = data
    return merged


def merge_query_results(per_page_answers: List[Dict]) -> Dict:
    """For open-ended queries, a question might genuinely be answered by
    more than one page (e.g. 'what are all the payment terms?' spanning
    two sections). Keep the highest-confidence 'found' answer as primary,
    but surface any other found answers as supporting evidence rather than
    silently discarding them."""
    found = [a for a in per_page_answers if a.get("found")]
    if not found:
        return {"answer": None, "confidence": 0.0, "source_text": None, "found": False, "supporting": []}

    found.sort(key=lambda a: a.get("confidence", 0), reverse=True)
    best = found[0]
    supporting = found[1:]
    return {**best, "supporting": supporting}


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_field_extraction(pdf_path: str, fields: List[str], mock: bool, top_n: int) -> Dict:
    pages = load_pdf(pdf_path)
    candidates = rank_candidate_pages(pages, fields, top_n=top_n)

    print(f"[intake]   {len(pages)} total pages "
          f"({sum(1 for p in pages if p.likely_scanned)} likely scanned)")
    print(f"[chunking] selected {len(candidates)} candidate page(s): "
          f"{[p.number for p in candidates]} (skipped {len(pages) - len(candidates)})")

    system_prompt = build_system_prompt(fields)
    per_page_results = []

    if mock:
        print("[extract]  running in MOCK mode (no API calls, no cost)")
        for page in candidates:
            per_page_results.append(mock_extract_from_page(fields, page))
    else:
        client = get_claude_client()
        for page in candidates:
            mode = "vision" if page.likely_scanned else "text"
            print(f"[extract]  calling Claude for page {page.number} ({mode})...")
            per_page_results.append(
                call_claude_safely(extract_from_page, client, system_prompt, page, pdf_path)
            )

    return merge_results(per_page_results)


def run_query(pdf_path: str, query: str, mock: bool, top_n: int) -> Dict:
    pages = load_pdf(pdf_path)
    candidates = rank_pages_by_query(pages, query, top_n=top_n)

    print(f"[intake]   {len(pages)} total pages "
          f"({sum(1 for p in pages if p.likely_scanned)} likely scanned)")
    print(f"[chunking] TF-IDF-ranked {len(candidates)} candidate page(s): "
          f"{[p.number for p in candidates]} (skipped {len(pages) - len(candidates)})")

    per_page_answers = []

    if mock:
        print("[extract]  running in MOCK mode (no API calls, no cost)")
        for page in candidates:
            per_page_answers.append(mock_answer_query_from_page(page, query))
    else:
        client = get_claude_client()
        for page in candidates:
            mode = "vision" if page.likely_scanned else "text"
            print(f"[extract]  calling Claude for page {page.number} ({mode})...")
            per_page_answers.append(
                call_claude_safely(answer_query_from_page, client, QUERY_SYSTEM_PROMPT, page, pdf_path, query)
                or {"found": False}
            )

    return merge_query_results(per_page_answers)


def main():
    parser = argparse.ArgumentParser(description="Smart Document Extraction Pipeline")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--fields", help="Comma-separated list of fields to extract")
    mode_group.add_argument("--query", help="An open-ended natural-language question to answer")
    parser.add_argument("--top-n", type=int, default=5, help="Max candidate pages to send to Claude")
    parser.add_argument("--mock", action="store_true", help="Force mock mode (no API calls)")
    args = parser.parse_args()

    mock = args.mock or not os.environ.get("ANTHROPIC_API_KEY")
    if mock and not args.mock:
        print("[info] No ANTHROPIC_API_KEY found — running in mock mode.\n")

    if args.fields:
        fields = [f.strip() for f in args.fields.split(",")]
        result = run_field_extraction(args.pdf_path, fields, mock=mock, top_n=args.top_n)
        print("\n=== Extraction Result ===")
    else:
        result = run_query(args.pdf_path, args.query, mock=mock, top_n=args.top_n)
        print("\n=== Query Answer ===")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"\n[error] {e}", file=sys.stderr)
        sys.exit(1)
