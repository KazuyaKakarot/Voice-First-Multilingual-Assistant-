"""Validate the scheme data, chunk it, embed it once, persist to disk.  Run: python main.py ingest"""
import json
from datetime import date

from src import config, schemes
from src.chunking import chunk_document
from src.cleaner import normalize, pdf_to_text, split_plain_sections, split_sections


def load_loose_files(start_id: int) -> list[dict]:
    """Optional extras: .md / .txt / .pdf files in data/docs/ (e.g. scheme PDFs from a public dataset).
    They carry no verified date or state, so they are treated as all-India, dated by file modification time."""
    docs: list[dict] = []
    if not config.DOCS_DIR.exists():
        return docs
    for p in sorted(config.DOCS_DIR.rglob("*")):
        suffix = p.suffix.lower()
        if suffix not in {".txt", ".md", ".pdf"}:
            continue
        text = pdf_to_text(p) if suffix == ".pdf" else p.read_text(encoding="utf-8", errors="ignore")
        text = normalize(text)
        sections = split_sections(text) if suffix == ".md" else split_plain_sections(text)
        if not sections:
            continue
        docs.append({
            "id": start_id + len(docs), "scheme_id": p.stem, "url": p.name,
            "title": p.stem.replace("_", " ").replace("-", " ").strip(), "state": "all-india", "category": "",
            "verified_at": date.fromtimestamp(p.stat().st_mtime).isoformat(), "sections": sections,
        })
    return docs


def load_documents() -> tuple[list[dict], list[str], list[str]]:
    """Return (documents, errors, warnings) from curated scheme JSON plus any loose files."""
    valid, errors, warnings = schemes.load_all()
    docs = [schemes.to_document(s, i) for i, s in enumerate(valid)]
    loose = load_loose_files(start_id=len(docs))
    if loose:
        warnings.append(f"{len(loose)} loose file(s) from data/docs/ loaded as all-India with no verified date")
    return docs + loose, errors, warnings


def main():
    docs, errors, warnings = load_documents()
    for w in warnings:
        print("warning:", w)
    if errors:
        print()
        for e in errors:
            print("ERROR:", e)
        raise SystemExit(f"\n{len(errors)} problem(s) in the scheme data. Fix them, then run ingest again.")
    if not docs:
        raise SystemExit("No schemes found. Copy data/schemes/_TEMPLATE.json to a new file "
                         "(e.g. data/schemes/pm-kisan.json), fill it in, and run again.")

    chunks: list[dict] = []
    for d in docs:
        chunks += chunk_document(d, start_id=len(chunks))
    print(f"{len(docs)} schemes -> {len(chunks)} chunks")

    from fastembed import TextEmbedding
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    embedder = TextEmbedding(config.EMBED_MODEL)
    vectors = list(embedder.embed([c["text"] for c in chunks], batch_size=32))

    client = QdrantClient(path=config.QDRANT_PATH)
    if client.collection_exists(config.COLLECTION):
        client.delete_collection(config.COLLECTION)
    client.create_collection(
        config.COLLECTION,
        vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
    )
    client.upsert(
        config.COLLECTION,
        points=[PointStruct(id=c["id"], vector=v.tolist(), payload=c) for c, v in zip(chunks, vectors)],
    )
    config.CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False))
    print(f"Indexed -> {config.QDRANT_PATH}")
