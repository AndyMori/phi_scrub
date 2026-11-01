"""
pdf_to_chunks.py
----------------
Converts a regulatory PDF (e.g., 45 CFR 164.514) into structured JSON chunks
that are ready to embed into a Chroma vector database.

Each chunk contains:
  - id          : unique stable ID (source_filename:chunk_index)
  - text        : the chunk text that will be embedded
  - metadata    : source, page numbers, section heading, char offset

Usage:
    python pdf_to_chunks.py <input.pdf> <output.json> [--chunk-size 800] [--overlap 100]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pdfplumber


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def extract_pages(pdf_path: str) -> list[dict]:
    """Return a list of {page_num, text} dicts from the PDF."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=2, y_tolerance=2)
            if text:
                pages.append({"page_num": i, "text": text.strip()})
    return pages


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

# Boilerplate header that repeats on every eCFR page:
#   "45 CFR 164.514 (up to date as of 10/28/2026)\n
#    45 CFR 164.514(b)(2)(i)(C)\n
#    Other requirements relating to uses and disclosures of protected health...\n"
_ECFR_HEADER_RE = re.compile(
    r"45 CFR \d+\.\d+[^\n]*\n"   # line 1: "45 CFR 164.514 (up to date …)"
    r"(?:45 CFR \d+\.\d+[^\n]*\n)?"  # line 2 (optional): section ref
    r"Other requirements[^\n]+\n",   # line 3: "Other requirements …"
    re.MULTILINE,
)

# Footer lines: "45 CFR 164.514(b)(2) (enhanced display) page 3 of 8"
_ECFR_FOOTER_RE = re.compile(
    r"45 CFR \d+\.\d+\([^\n]+ \(enhanced display\) page \d+ of \d+\n?",
    re.MULTILINE,
)


def clean_text(text: str) -> str:
    """Normalise whitespace and remove eCFR boilerplate artefacts."""
    # Fix hyphenated line-breaks (word split across lines)
    text = re.sub(r"-\n(\S)", r"\1", text)
    # Collapse multiple spaces / tabs
    text = re.sub(r"[ \t]+", " ", text)
    # Remove repeating eCFR headers and footers
    text = _ECFR_HEADER_RE.sub("", text)
    text = _ECFR_FOOTER_RE.sub("", text)
    # Collapse 3+ consecutive newlines to two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

# Patterns that look like CFR section headings, e.g.  "§ 164.514", "(b)(1)", "(2)(i)"
_SECTION_RE = re.compile(
    r"^(?:§\s*[\d.]+|"                            # § 164.514
    r"\([a-zA-Z0-9]+\)(?:\([a-zA-Z0-9]+\))*"      # (b)(2)(i)
    r")\s*",
    re.MULTILINE,
)


def detect_heading(text: str) -> str:
    """Return the first section marker found in the text, or '' if none."""
    m = _SECTION_RE.search(text)
    return m.group(0).strip() if m else ""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    full_text: str,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[dict]:
    """
    Split text into overlapping chunks that respect paragraph boundaries.

    Paragraphs are separated by one or more blank lines.  When a single
    paragraph exceeds chunk_size it is split at sentence boundaries.

    Returns a list of {text, char_start, char_end, heading}.
    """
    # Split into paragraphs (blank-line separated)
    raw_paragraphs = re.split(r"\n{2,}", full_text)

    # Further split oversized paragraphs at sentence boundaries
    paragraphs: list[str] = []
    for para in raw_paragraphs:
        if len(para) <= chunk_size:
            paragraphs.append(para)
        else:
            # Split at ". " or ";\n" boundaries
            sentences = re.split(r"(?<=[.;])\s+", para)
            current = ""
            for sent in sentences:
                if current and len(current) + len(sent) + 1 > chunk_size:
                    paragraphs.append(current.strip())
                    current = sent
                else:
                    current = (current + " " + sent).strip() if current else sent
            if current:
                paragraphs.append(current.strip())

    chunks: list[dict] = []
    current_parts: list[str] = []
    current_len = 0
    char_pos = 0          # approximate running position in full_text
    chunk_start = 0

    def flush() -> None:
        body = "\n\n".join(current_parts).strip()
        if body:
            chunks.append(
                {
                    "text": body,
                    "char_start": chunk_start,
                    "char_end": chunk_start + len(body),
                    "heading": detect_heading(body),
                }
            )

    for para in paragraphs:
        para_len = len(para)

        if current_len + para_len > chunk_size and current_parts:
            flush()
            # Seed next chunk with overlap from the tail of the current one
            overlap_text = "\n\n".join(current_parts)[-overlap:].strip()
            current_parts = [overlap_text] if overlap_text else []
            current_len = len(overlap_text)
            chunk_start = max(0, char_pos - overlap)

        current_parts.append(para)
        current_len += para_len + 2   # +2 for "\n\n" separator
        char_pos += para_len + 2

    flush()
    return chunks


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def pdf_to_chunks(
    pdf_path: str,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[dict]:
    """Full pipeline: PDF → pages → clean text → chunks with metadata."""
    source_name = Path(pdf_path).name

    pages = extract_pages(pdf_path)
    if not pages:
        raise ValueError(f"No text could be extracted from {pdf_path}")

    # Join all pages and clean once (so cross-page boilerplate is caught)
    combined_parts: list[tuple[int, str]] = []
    for p in pages:
        combined_parts.append((p["page_num"], p["text"]))

    full_raw = "\n\n".join(t for _, t in combined_parts)
    full_text = clean_text(full_raw)

    # Build a rough char→page mapping based on original page text lengths
    page_boundaries: list[tuple[int, int]] = []
    offset = 0
    for page_num, text in combined_parts:
        page_boundaries.append((offset, page_num))
        offset += len(text) + 2

    def char_to_page(char_pos: int) -> int:
        page = 1
        for start, pnum in page_boundaries:
            if char_pos >= start:
                page = pnum
        return page

    raw_chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)

    result: list[dict] = []
    for idx, chunk in enumerate(raw_chunks):
        start_page = char_to_page(chunk["char_start"])
        end_page = char_to_page(chunk["char_end"])
        result.append(
            {
                "id": f"{source_name}:{idx:04d}",
                "text": chunk["text"],
                "metadata": {
                    "source": source_name,
                    "chunk_index": idx,
                    "start_page": start_page,
                    "end_page": end_page,
                    "heading": chunk["heading"],
                    "char_start": chunk["char_start"],
                    "char_end": chunk["char_end"],
                },
            }
        )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a regulatory PDF to Chroma-ready JSON chunks."
    )
    parser.add_argument("input_pdf", help="Path to the source PDF file")
    parser.add_argument("output_json", help="Path for the output JSON file")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Target character length per chunk (default: 800)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="Overlap characters between consecutive chunks (default: 100)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_pdf)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting text from: {input_path}")
    chunks = pdf_to_chunks(
        str(input_path),
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(chunks)} chunks → {output_path}")
    print(f"  chunk_size={args.chunk_size}, overlap={args.overlap}")

    # Quick preview
    if chunks:
        print(f"\nFirst chunk preview (id={chunks[0]['id']}):")
        print(f"  pages {chunks[0]['metadata']['start_page']}–{chunks[0]['metadata']['end_page']}")
        print(f"  heading: '{chunks[0]['metadata']['heading']}'")
        print(f"  text[:300]:\n{chunks[0]['text'][:300]}")


if __name__ == "__main__":
    main()
