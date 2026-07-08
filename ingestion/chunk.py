"""DoclingDocument → chunks via docling's HybridChunker.

Chunks carry stable IDs ({doc_hash}-{seq}) and heading context. Tables are
serialized into the chunk text by the chunker itself, so table content is
searchable and visible to the LLM. Only reference/acknowledgment sections are
excluded (exact heading match) — the abstract stays in.
"""
import logging

from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument
from transformers import AutoTokenizer

import config

log = logging.getLogger(__name__)

_chunker = None


def get_chunker() -> HybridChunker:
    global _chunker
    if _chunker is None:
        tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)
        _chunker = HybridChunker(tokenizer=tokenizer, max_tokens=config.CHUNK_MAX_TOKENS, merge_peers=True)
    return _chunker


def _excluded(headings) -> bool:
    return any(h.strip().lower() in config.EXCLUDED_HEADINGS for h in headings or [])


def chunk_document(doc_hash: str, doc_name: str, doc: DoclingDocument) -> list[dict]:
    chunker = get_chunker()
    chunks = []
    seq = 0
    for chunk in chunker.chunk(doc):
        headings = list(chunk.meta.headings or [])
        if _excluded(headings):
            continue
        text = chunker.contextualize(chunk)  # heading-prefixed text, tables serialized
        if not text.strip():
            continue
        chunks.append({
            "chunk_id": f"{doc_hash}-{seq:04d}",
            "doc_hash": doc_hash,
            "pdf": doc_name,
            "headings": " > ".join(headings),
            "text": text,
        })
        seq += 1
    log.info(f"{doc_name}: {len(chunks)} chunks")
    return chunks
