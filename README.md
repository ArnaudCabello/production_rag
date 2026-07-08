# Production RAG

Local, open-source RAG pipeline over PDF documents. Upload PDFs, ask questions, get
answers with inline source citations. Every model runs locally (HuggingFace weights) —
no API keys, no data leaves the machine.

## Architecture

```
PDF ──► docling (layout + tables) ──► HybridChunker ──► bge-m3 embeddings ──► Chroma
                                          │ per-document "card" chunk (title/byline)
Query ──► dense (bge-m3) ∪ BM25 ──► reciprocal-rank fusion ──► bge-reranker-v2-m3
      ──► top-5 chunks as numbered sources ──► generator LLM (LangGraph) ──► cited answer
```

| Component | Default | Where set |
|---|---|---|
| Generator | Qwen/Qwen2.5-14B-Instruct | `config.GENERATOR_MODEL` (any LangChain provider via `GENERATOR_PROVIDER`) |
| Embeddings | BAAI/bge-m3 | `config.EMBEDDING_MODEL` |
| Reranker | BAAI/bge-reranker-v2-m3 | `config.RERANKER_MODEL` |
| Vector store | Chroma (persistent, `chroma_db/`) | `config.CHROMA_DIR` |
| Chunking | docling HybridChunker, 512 tokens | `config.CHUNK_MAX_TOKENS` |

Rerank order is `sigmoid(cross_encoder) + λ·RRF` with λ=2, chosen by a measured sweep
(see `eval/results/`). Ingestion is incremental by content hash: unchanged PDFs are
skipped, changed PDFs replace their old chunks, chunk IDs (`{doc_hash}-{seq}`) are stable.

## Quickstart

Needs a GPU for the 14B generator (A100 recommended); ingestion and retrieval also run on CPU.

```bash
pip install -r requirements.txt
# put your PDFs in data/pdfs/
python -m ingestion.run       # convert + chunk + embed + index (incremental)
python app.py                 # Gradio UI; add --share on Colab
```

### Google Colab

```python
from google.colab import userdata
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")          # HF secret (model downloads)
token = userdata.get("GITHUB_TOKEN")                        # GitHub secret (push access)

!git clone -b revision https://{token}@github.com/ArnaudCabello/production_rag.git
%cd production_rag
!pip install -q -r requirements.txt
# upload PDFs to data/pdfs/ (Colab file browser), then:
!python -m ingestion.run
!python app.py --share
```

## Evaluation

A golden Q/A set with evidence-based retrieval metrics and an LLM judge lives in `eval/`
(see `eval/README.md`). Every pipeline change is measured against it:

| | Legacy pipeline | Current |
|---|---|---|
| retrieval hit@5 | 0.77 | **0.90** |
| retrieval hit@10 | 0.80 | **1.00** |
| answers judged correct | 7/30 | **22/30** |
| answers judged incorrect | 7/30 | **0/30** |

```bash
python eval/retrieval_eval.py --retriever hybrid    # retrieval metrics (CPU-friendly)
python eval/generate_answers_v2.py                  # answer the golden set (GPU)
python eval/judge_answers.py eval/results/answers_v2.jsonl
```

## Repo layout

```
config.py          all models, paths, and retrieval parameters
ingestion/         convert.py (PDF→docling), chunk.py (chunks + document cards), index.py (Chroma)
retrieval/         hybrid retriever: dense ∪ BM25 → RRF → blended cross-encoder rerank
generation/        model-agnostic LLM loader + LangGraph retrieve→generate pipeline
app.py             Gradio UI (loads the existing index at startup)
eval/              golden set, retrieval metrics, generation judge, recorded results
data/pdfs/         your source PDFs (gitignored; folder is tracked)
```

Scanned PDFs: set `OCR_ENABLED = True` in `config.py` (requires EasyOCR model download).
