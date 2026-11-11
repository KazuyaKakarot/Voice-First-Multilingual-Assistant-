"""Text helpers for scheme documents: normalization and section splitting.

Used for loose files (.md / .txt / .pdf) dropped into data/docs/. Curated scheme JSON files
(src/schemes.py) are already sectioned and skip this step.
"""
import re
import unicodedata

# Canonical section types. Retrieval works better when chunks know which part of a scheme they are.
SECTION_KEYWORDS = [
    ("eligibility", ["eligib", "who can apply", "who is eligible", "beneficiar", "पात्र"]),
    ("benefits", ["benefit", "assistance", "subsidy", "amount", "लाभ"]),
    ("documents", ["document", "required", "proof", "दस्तावेज"]),
    ("application", ["how to apply", "application", "apply", "process", "procedure", "आवेदन"]),
]
_PLAIN_ABOUT_HEADINGS = {"overview", "details", "description", "about", "about the scheme"}
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿"), None)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def classify_heading(heading: str) -> str:
    h = heading.lower()
    for name, keys in SECTION_KEYWORDS:
        if any(k in h for k in keys):
            return name
    return "about"


def split_sections(markdown: str) -> list[dict]:
    """Split on markdown headings (#, ##, ...). Text before the first heading is 'about'."""
    sections, heading, buf = [], "Overview", []

    def flush():
        body = normalize("\n".join(buf))
        if body:
            sections.append({"type": classify_heading(heading), "heading": heading, "text": body})

    for line in markdown.splitlines():
        m = re.match(r"^#{1,6}\s+(.*)", line)
        if m:
            flush()
            heading, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    flush()
    return sections


def is_plain_heading(line: str) -> bool:
    """A short line like 'Eligibility Criteria:' or 'Documents Required' in plain (PDF/text) content.
    Kept strict (3 words max, no sentence punctuation) so ordinary sentences aren't mistaken for headings."""
    s = line.strip().rstrip(":").strip()
    if not s or len(s.split()) > 3 or s.endswith((".", ",", ";")):
        return False
    return s.lower() in _PLAIN_ABOUT_HEADINGS or classify_heading(s) != "about"


def split_plain_sections(text: str) -> list[dict]:
    """Section-split text that has no markdown, by recognising heading-like lines."""
    lines = [("## " + ln.strip().rstrip(":")) if is_plain_heading(ln) else ln for ln in text.splitlines()]
    return split_sections("\n".join(lines))


def pdf_to_text(path) -> str:
    from pypdf import PdfReader  # lazy

    return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
