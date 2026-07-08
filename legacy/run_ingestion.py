"""Rebuild the legacy pipeline's index over a folder of PDFs.

The legacy scripts (markdown.py -> chunks.py -> embedding.py) are preserved
verbatim from the pre-rewrite pipeline and use paths relative to their own
directory, so everything they produce stays under legacy/:
    legacy/outputs/            per-PDF markdown + extracted images
    legacy/all_chunks.json     chunked corpus
    legacy/vector_store.faiss  MiniLM FAISS index
    legacy/metadata1.json      chunk metadata (what eval/legacy_adapter.py reads)

Usage (repo root):
    python legacy/run_ingestion.py                 # ingest data/pdfs/
    python legacy/run_ingestion.py /path/to/pdfs   # ingest another folder

markdown.py checkpoints per file (legacy/outputs/checkpoint.txt), so re-running
skips PDFs that were already converted.
"""
import shutil
import subprocess
import sys
from pathlib import Path

LEGACY_DIR = Path(__file__).resolve().parent


def main():
    pdf_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else LEGACY_DIR.parent / "data" / "pdfs"
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {pdf_dir}")

    temp = LEGACY_DIR / "temp_pdfs"
    temp.mkdir(exist_ok=True)
    for pdf in pdfs:
        shutil.copy(pdf, temp / pdf.name)
    print(f"Staged {len(pdfs)} PDF(s) from {pdf_dir}")

    for script in ("markdown.py", "chunks.py", "embedding.py"):
        print(f"\n=== legacy/{script} ===", flush=True)
        result = subprocess.run([sys.executable, script], cwd=LEGACY_DIR)
        if result.returncode != 0:
            sys.exit(f"legacy/{script} failed with exit code {result.returncode}")

    print("\nLegacy index built: legacy/vector_store.faiss + legacy/metadata1.json")


if __name__ == "__main__":
    main()
