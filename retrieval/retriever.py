"""Hybrid retrieval: dense (bge-m3/Chroma) ∪ BM25, fused with reciprocal-rank
fusion, then reranked by a cross-encoder.

No gates: either signal alone can surface a chunk, and the reranker decides
final order. This replaces the legacy TF-IDF-gated scoring.
"""
import logging
import math
import re

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

import config
from ingestion.index import get_collection, get_embedder

log = logging.getLogger(__name__)

RRF_K = 60  # standard reciprocal-rank fusion constant


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    def __init__(self, rerank: bool = True):
        self.collection = get_collection()
        data = self.collection.get()
        self.ids = data["ids"]
        self.chunks = {
            cid: {"chunk_id": cid, "text": doc, **meta}
            for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
        }
        self.bm25 = BM25Okapi([_tokenize(self.chunks[cid]["text"]) for cid in self.ids])
        self.embedder = get_embedder()
        self.reranker = CrossEncoder(config.RERANKER_MODEL) if rerank else None
        log.info(f"HybridRetriever ready: {len(self.ids)} chunks, rerank={'on' if rerank else 'off'}")

    def _dense_ids(self, query: str, pdfs: list[str] = None) -> list[str]:
        embedding = self.embedder.encode([query], normalize_embeddings=True)
        n = min(config.DENSE_TOP_K, len(self.ids))
        where = {"pdf": {"$in": list(pdfs)}} if pdfs else None
        ids = self.collection.query(
            query_embeddings=embedding.tolist(), n_results=n, where=where,
        )["ids"][0]
        # the live collection may contain chunks upserted after this snapshot
        return [cid for cid in ids if cid in self.chunks]

    def _bm25_ids(self, query: str, pdfs: list[str] = None) -> list[str]:
        allowed = set(pdfs) if pdfs else None
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(self.ids)), key=lambda i: scores[i], reverse=True)
        ids = (i for i in ranked
               if scores[i] > 0 and (allowed is None or self.chunks[self.ids[i]]["pdf"] in allowed))
        return [self.ids[i] for _, i in zip(range(config.BM25_TOP_K), ids)]

    def search(self, query: str, top_k: int = config.RERANK_TOP_N, pdfs: list[str] = None,
               rerank: bool = True) -> list[dict]:
        """Hybrid search; pass pdfs to restrict every stage to those documents' chunks.
        rerank=False returns fusion (RRF) order — the cross-encoder dominates CPU
        latency (~minutes per pass), so callers making many narrow passes skip it."""
        fused: dict[str, float] = {}
        for id_list in (self._dense_ids(query, pdfs), self._bm25_ids(query, pdfs)):
            for rank, cid in enumerate(id_list, 1):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        candidates = sorted(fused, key=fused.get, reverse=True)

        if rerank and self.reranker is not None:
            pairs = [(query, self.chunks[cid]["text"]) for cid in candidates]
            rerank_scores = self.reranker.predict(pairs)
            blended = [
                1.0 / (1.0 + math.exp(-float(r))) + config.RERANK_BLEND_LAMBDA * fused[cid]
                for r, cid in zip(rerank_scores, candidates)
            ]
            candidates = [cid for _, cid in sorted(zip(blended, candidates), key=lambda x: -x[0])]

        return [self.chunks[cid] for cid in candidates[:top_k]]
