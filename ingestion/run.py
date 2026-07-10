"""Ingest a directory of PDFs end-to-end: convert → chunk → embed → index.

Usage (repo root):
    python -m ingestion.run              # ingests config.PDF_DIR
    python -m ingestion.run path/to/dir
"""
import logging
import sys
from pathlib import Path

import config
from ingestion.chunk import chunk_document
from ingestion.convert import content_hash, convert_pdf
from ingestion.index import delete_doc, doc_is_indexed, get_collection, index_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def ingest_pdf(collection, pdf_path: Path) -> int:
    doc_hash = content_hash(pdf_path)
    delete_doc(collection, pdf_path.name, keep_hash=doc_hash)
    if doc_is_indexed(collection, doc_hash):  # before conversion/cache-load — unchanged files cost a hash only
        log.info(f"⏭️  {pdf_path.name} unchanged ({doc_hash}), skipping")
        return 0
    doc_hash, doc = convert_pdf(pdf_path)
    chunks = chunk_document(doc_hash, pdf_path.name, doc)
    index_chunks(collection, chunks)
    return len(chunks)


def main():
    pdf_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else config.PDF_DIR
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {pdf_dir}")

    collection = get_collection()
    total = 0
    for pdf in pdfs:
        total += ingest_pdf(collection, pdf)
    log.info(f"✅ Done: {len(pdfs)} PDF(s), {total} new chunks, collection size {collection.count()}")


if __name__ == "__main__":
    main()
