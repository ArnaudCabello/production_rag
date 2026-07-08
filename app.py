"""Gradio app over the RAG pipeline: upload PDFs → ask cited questions.

Usage (repo root, GPU recommended):
    python app.py            # local
    python app.py --share    # public gradio.live link (needed on Colab)

The existing Chroma index is used at startup — no reprocessing on launch.
Models load lazily on the first question.
"""
import argparse
import logging
import shutil
from pathlib import Path

import gradio as gr

import config
from ingestion.index import get_collection
from ingestion.run import ingest_pdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class RAGApp:
    def __init__(self):
        self.graph = None

    def _ensure_ready(self):
        if self.graph is None:
            from generation.llm import get_llm
            from generation.pipeline import build_graph
            from retrieval.retriever import HybridRetriever

            log.info("Loading models (first question or corpus changed)...")
            self.graph = build_graph(HybridRetriever(), get_llm())

    def ask(self, question: str):
        question = question.strip()
        if not question:
            return "Please enter a question.", ""
        if get_collection().count() == 0:
            return "⚠️ No documents indexed yet — upload PDFs first (or run `python -m ingestion.run`).", ""

        self._ensure_ready()
        result = self.graph.invoke({"question": question})
        sources = "\n\n".join(
            f"**[{i}] {chunk.get('pdf', '?')}** — {chunk.get('headings', '')}\n\n{chunk['text'][:600]}"
            for i, chunk in enumerate(result["chunks"], 1)
        )
        return result["answer"], sources

    def upload(self, files, progress=gr.Progress()):
        if not files:
            return "No files uploaded."
        config.PDF_DIR.mkdir(parents=True, exist_ok=True)
        collection = get_collection()
        added = 0
        for i, f in enumerate(files, 1):
            src = Path(f.name)
            dest = config.PDF_DIR / src.name
            shutil.copy(src, dest)
            progress(i / len(files), desc=f"Ingesting {src.name}")
            added += ingest_pdf(collection, dest)
        self.graph = None  # retriever caches the corpus; rebuild on next question
        return f"✅ Ingested {len(files)} PDF(s): {added} new chunks. Collection size: {collection.count()}."


def build_interface(app: RAGApp) -> gr.Blocks:
    with gr.Blocks(title="Production RAG") as interface:
        gr.Markdown("# 📚 Production RAG\nUpload PDFs, then ask questions — answers cite their sources.")

        with gr.Row():
            pdf_upload = gr.File(label="Upload PDFs", file_count="multiple", file_types=[".pdf"], scale=3)
            process_btn = gr.Button("🚀 Ingest", variant="primary", scale=1)
        status = gr.Textbox(label="Ingestion status", lines=2, interactive=False)

        question = gr.Textbox(label="Your question", lines=2, placeholder="Ask about the documents...")
        ask_btn = gr.Button("Ask", variant="primary")
        answer = gr.Textbox(label="Answer", lines=8)
        sources = gr.Markdown(label="Sources")

        process_btn.click(app.upload, inputs=[pdf_upload], outputs=[status])
        ask_btn.click(app.ask, inputs=[question], outputs=[answer, sources])
        question.submit(app.ask, inputs=[question], outputs=[answer, sources])
    return interface


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="publish a public gradio.live link (Colab)")
    args = parser.parse_args()

    build_interface(RAGApp()).launch(share=args.share)
