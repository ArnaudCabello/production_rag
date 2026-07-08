"""Dense-only retrieval over the new Chroma index (Phase 2 component baseline).

Phase 3 adds BM25 + fusion + reranking on top; this adapter measures the
bge-m3 index alone.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from ingestion.index import get_collection, get_embedder  # noqa: E402


class DenseRetriever:
    def __init__(self):
        self.collection = get_collection()
        self.embedder = get_embedder()

    def search(self, query: str, top_k: int = 10):
        embedding = self.embedder.encode([query], normalize_embeddings=True)
        result = self.collection.query(query_embeddings=embedding.tolist(), n_results=top_k)
        return [
            {"text": doc, **meta}
            for doc, meta in zip(result["documents"][0], result["metadatas"][0])
        ]
