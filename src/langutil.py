"""Small language helpers (no dependencies)."""
import re

# Unicode script ranges -> default language for that script. Several languages share a script
# (Devanagari: hi/mr), so this is a best-effort default; let the user override with --lang.
_SCRIPTS = [
    ("hi", 0x0900, 0x097F),  # Devanagari
    ("bn", 0x0980, 0x09FF),
    ("pa", 0x0A00, 0x0A7F),  # Gurmukhi
    ("gu", 0x0A80, 0x0AFF),
    ("or", 0x0B00, 0x0B7F),
    ("ta", 0x0B80, 0x0BFF),
    ("te", 0x0C00, 0x0C7F),
    ("kn", 0x0C80, 0x0CFF),
    ("ml", 0x0D00, 0x0D7F),
]


def detect_lang(text: str) -> str:
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for lang, lo, hi in _SCRIPTS:
            if lo <= cp <= hi:
                counts[lang] = counts.get(lang, 0) + 1
                break
    return max(counts, key=counts.get) if counts else "en"


def strip_citations(text: str) -> str:
    """Remove [1], [2] markers (and markdown) so the text reads naturally when spoken."""
    text = re.sub(r"\s*\[\d+\]", "", text)
    return re.sub(r"[*_`#]+", "", text).strip()


def split_sentences(text: str, max_chars: int = 400) -> list[str]:
    """Group sentences (., ?, !, danda) into pieces under max_chars, for APIs with input limits."""
    sentences = [s for s in re.split(r"(?<=[.!?।])\s+", text.strip()) if s]
    pieces, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > max_chars:
            pieces.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        pieces.append(cur)
    return pieces
