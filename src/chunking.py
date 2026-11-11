"""Section-aware chunking. Pure functions, no heavy dependencies."""
from src import config


def split_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = size - overlap
    pieces = []
    for i in range(0, len(words), step):
        pieces.append(" ".join(words[i : i + size]))
        if i + size >= len(words):
            break
    return pieces


def chunk_document(doc: dict, start_id: int = 0) -> list[dict]:
    """One chunk per section (split further if long). Each chunk is prefixed with the scheme title and
    section type, so a chunk saying '18 years or above' still knows which scheme and which section it is."""
    chunks = []
    for sec in doc["sections"]:
        for j, piece in enumerate(split_words(sec["text"], config.CHUNK_WORDS, config.CHUNK_OVERLAP)):
            chunks.append({
                "id": start_id + len(chunks),
                "text": f"{doc['title']} | {sec['type']}: {piece}",
                "url": doc["url"],
                "title": doc["title"],
                "state": doc["state"],
                "category": doc["category"],
                "section": sec["type"],
                "verified_at": doc["verified_at"],
                "scheme_id": doc.get("scheme_id", ""),
                "source": doc["url"],
                "chunk": j,
            })
    return chunks
