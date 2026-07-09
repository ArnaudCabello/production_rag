"""PDF → DoclingDocument, cached on disk keyed by file content hash.

The JSON cache is the lossless intermediate: chunking works from docling's
structured document (headings, tables as objects), never from re-parsed markdown.
"""
import hashlib
import json
import logging
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument

import config

log = logging.getLogger(__name__)


def content_hash(pdf_path: Path) -> str:
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:12]


def convert_pdf(pdf_path: Path) -> tuple[str, DoclingDocument]:
    """Convert a PDF (or load from cache). Returns (doc_hash, document)."""
    doc_hash = content_hash(pdf_path)
    cache_file = config.DOCLING_CACHE / f"{doc_hash}.json"

    if cache_file.exists():
        log.info(f"Cache hit for {pdf_path.name} ({doc_hash})")
        return doc_hash, DoclingDocument.model_validate_json(cache_file.read_text(encoding="utf-8"))

    log.info(f"Converting {pdf_path.name} ({doc_hash})...")
    pipeline_options = PdfPipelineOptions(
        do_ocr=config.OCR_ENABLED,
        generate_picture_images=True,
        images_scale=2.0,
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    result = converter.convert(pdf_path)
    doc = result.document
    save_figures(doc_hash, doc)

    config.DOCLING_CACHE.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(doc.export_to_dict()), encoding="utf-8")
    log.info(f"Converted {pdf_path.name}: {doc.num_pages()} pages, cached at {cache_file}")
    return doc_hash, doc


def save_figures(doc_hash: str, doc: DoclingDocument):
    """Write each figure to data/figures/{doc_hash}-fig{n}.png (n = position in
    doc.pictures, the same numbering chunking uses), then drop the in-memory
    image payloads so the JSON cache stays lean."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    for n, picture in enumerate(doc.pictures, 1):
        image = picture.get_image(doc)
        if image is None:
            continue
        image.save(config.FIGURES_DIR / f"{doc_hash}-fig{n:03d}.png")
        picture.image = None
        saved += 1
    if saved:
        log.info(f"Saved {saved} figure(s) for {doc_hash}")
