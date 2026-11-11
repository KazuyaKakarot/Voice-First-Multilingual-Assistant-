"""Hybrid retrieval over multiple rewritten queries.

For each query: dense (Qdrant) + BM25 rankings. All rankings are fused with Reciprocal Rank Fusion,
optionally filtered by state, then a cross-encoder reranks the candidates against the user's
ORIGINAL question (the rewrites are only for recall; the original expresses the real intent).
"""
import json
import re

from src import config


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank)."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


def filter_by_state(chunks: list[dict], ids: list[int], state: str | None, min_keep: int) -> list[int]:
    """Keep chunks for the user's state plus all-India schemes. If that leaves too few, don't filter."""
    if not state:
        return ids
    state = state.lower()
    kept = [i for i in ids if chunks[i].get("state", "all-india") in (state, "all-india", "")]
    return kept if len(kept) >= min_keep else ids


class HybridRetriever:
    def __init__(self):
        from fastembed import TextEmbedding
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        from qdrant_client import QdrantClient
        from rank_bm25 import BM25Okapi

        self.chunks = json.loads(config.CHUNKS_PATH.read_text())
        self.bm25 = BM25Okapi([_tokenize(c["text"]) for c in self.chunks])
        self.embedder = TextEmbedding(config.EMBED_MODEL)
        self.reranker = TextCrossEncoder(config.RERANK_MODEL)
        self.qdrant = QdrantClient(path=config.QDRANT_PATH)

    def _dense(self, query: str, n: int) -> list[int]:
        vec = next(iter(self.embedder.embed([query]))).tolist()
        res = self.qdrant.query_points(config.COLLECTION, query=vec, limit=n)
        return [p.id for p in res.points]

    def _sparse(self, query: str, n: int) -> list[int]:
        scores = self.bm25.get_scores(_tokenize(query))
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]

    def search(self, queries: list[str], question: str, k: int = config.TOP_K,
               candidates: int = config.CANDIDATES, state: str | None = None,
               dense: bool = True, sparse: bool = True, rerank: bool = True) -> list[dict]:
        """The dense / sparse / rerank switches exist so the evaluation can measure what each stage adds."""
        if not dense and not sparse:
            raise ValueError("enable at least one of dense or sparse retrieval")
        rankings = []
        for q in queries:
            if dense:
                rankings.append(self._dense(q, candidates))
            if sparse:
                rankings.append(self._sparse(q, candidates))
        fused = rrf(rankings)
        top_ids = sorted(fused, key=fused.get, reverse=True)[:candidates]
        top_ids = filter_by_state(self.chunks, top_ids, state, min_keep=k)
        docs = [self.chunks[i] for i in top_ids]
        if not rerank:
            return [{**self.chunks[i], "score": fused[i]} for i in top_ids][:k]

        scores = list(self.reranker.rerank(question, [d["text"] for d in docs]))
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:k]
        return [{**d, "score": float(s)} for d, s in ranked]
