"""DoclingDocument → chunks via docling's HybridChunker.

Chunks carry stable IDs ({doc_hash}-{seq}), heading context, and provenance
(page number + bounding boxes, PDF points, origin bottom-left) so a citation
can be highlighted in the original PDF. Tables are serialized into the chunk
text by the chunker itself, so table content is searchable and visible to the
LLM. Sections in config.EXCLUDED_HEADINGS are dropped (exact heading match) —
the abstract and acknowledgments stay in. A per-document card chunk states
file name, title, and byline as sentences.
"""
import json
import logging
import re

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


_PREAMBLE_STOP = re.compile(r"^(abstract|(\d+\.?\s*)?introduction)\b", re.IGNORECASE)


def build_document_card(doc_hash: str, doc_name: str, doc: DoclingDocument) -> dict:
    """Synthetic chunk stating document-level facts (file name, title, page-1 byline)
    as sentences, so questions about the document as an object can retrieve it."""
    title = None
    preamble = []
    for item in doc.texts:
        if not item.prov or item.prov[0].page_no != 1:
            continue
        if item.label in ("page_header", "page_footer", "footnote"):
            continue
        text = item.text.strip()
        if not text:
            continue
        # stop at the first real section (ABSTRACT/INTRODUCTION), a body
        # paragraph, or a safety cap — everything before is the title block
        if _PREAMBLE_STOP.match(text) or len(text) > 300 or len(preamble) >= 8:
            break
        if title is None and item.label in ("title", "section_header"):
            title = text
        else:
            preamble.append(text)
    parts = [f"Document: {doc_name}."]
    if title:
        parts.append(f"Title: {title}.")
    if preamble:
        parts.append("The header and byline of this document reads: " + " ".join(preamble))
    return {
        "chunk_id": f"{doc_hash}-card",
        "doc_hash": doc_hash,
        "pdf": doc_name,
        "headings": "Document card",
        "text": " ".join(parts),
    }


def figure_map_by_caption_ref(doc_hash: str, doc: DoclingDocument) -> dict[str, str]:
    """Map each figure caption's item ref to its saved image file, so chunks that
    contain a caption can carry the figure. Numbering matches convert.save_figures."""
    mapping = {}
    for n, picture in enumerate(doc.pictures, 1):
        path = config.FIGURES_DIR / f"{doc_hash}-fig{n:03d}.png"
        if not path.exists():
            continue
        rel_path = str(path.relative_to(config.REPO_ROOT))
        mapping[picture.self_ref] = rel_path  # chunker may emit the picture item or its caption
        for caption_ref in picture.captions:
            mapping[caption_ref.cref] = rel_path
    return mapping


def chunk_provenance(chunk) -> str:
    """JSON [[page, l, t, r, b], ...] for every item in the chunk — the rectangles
    a PDF viewer needs to highlight the passage. Docling coordinates: PDF points,
    origin bottom-left."""
    boxes = []
    for item in chunk.meta.doc_items or []:
        for prov in getattr(item, "prov", None) or []:
            bbox = prov.bbox
            boxes.append([prov.page_no, round(bbox.l, 1), round(bbox.t, 1), round(bbox.r, 1), round(bbox.b, 1)])
    return json.dumps(boxes) if boxes else ""


def chunk_document(doc_hash: str, doc_name: str, doc: DoclingDocument) -> list[dict]:
    chunker = get_chunker()
    figure_map = figure_map_by_caption_ref(doc_hash, doc)
    chunks = [build_document_card(doc_hash, doc_name, doc)]
    seq = 0
    for chunk in chunker.chunk(doc):
        headings = list(chunk.meta.headings or [])
        if _excluded(headings):
            continue
        text = chunker.contextualize(chunk)  # heading-prefixed text, tables serialized
        if not text.strip():
            continue
        figures = [figure_map[item.self_ref] for item in chunk.meta.doc_items or []
                   if item.self_ref in figure_map]
        prov = chunk_provenance(chunk)
        record = {
            "chunk_id": f"{doc_hash}-{seq:04d}",
            "doc_hash": doc_hash,
            "pdf": doc_name,
            "headings": " > ".join(headings),
            "text": text,
        }
        if figures:
            record["figures"] = ",".join(dict.fromkeys(figures))  # Chroma metadata must be scalar
        if prov:
            record["prov"] = prov
        chunks.append(record)
        seq += 1
    log.info(f"{doc_name}: {len(chunks)} chunks "
             f"({sum(1 for c in chunks if c.get('figures'))} with figures)")
    return chunks
