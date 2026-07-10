"""Embed chunks with bge-m3 and store them in a persistent Chroma collection.

Incremental by content hash: an unchanged PDF is skipped entirely; a changed or
renamed PDF has its old chunks deleted and new ones added. Chunk IDs are stable
({doc_hash}-{seq}), so re-ingesting the same corpus is a no-op.
"""
import logging

import chromadb
from sentence_transformers import SentenceTransformer

import config

log = logging.getLogger(__name__)

_embedder = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        log.info(f"Loading embedding model {config.EMBEDDING_MODEL}...")
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder


def get_collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def doc_is_indexed(collection, doc_hash: str) -> bool:
    return len(collection.get(where={"doc_hash": doc_hash}, limit=1)["ids"]) > 0


def delete_doc(collection, pdf_name: str, keep_hash: str):
    """Remove chunks of any previous version of this PDF (different content hash)."""
    stale = collection.get(where={"$and": [{"pdf": pdf_name}, {"doc_hash": {"$ne": keep_hash}}]})
    if stale["ids"]:
        collection.delete(ids=stale["ids"])
        log.info(f"Deleted {len(stale['ids'])} stale chunks of {pdf_name}")


def index_chunks(collection, chunks: list[dict], batch_size: int = 32):
    embedder = get_embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    collection.upsert(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[
            {"doc_hash": c["doc_hash"], "pdf": c["pdf"], "headings": c["headings"]}
            | ({"figures": c["figures"]} if c.get("figures") else {})
            | ({"prov": c["prov"]} if c.get("prov") else {})
            for c in chunks
        ],
    )
    log.info(f"Indexed {len(chunks)} chunks (collection now {collection.count()})")
