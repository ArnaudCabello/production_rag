"""FastAPI backend for the desktop app: ingest, scoped ask, settings, PDF serving.

Run (repo root):
    python -m uvicorn backend.main:app --port 8642

Heavy resources load lazily and are cached with separate locks and lifetimes:
the retriever (invalidated when the corpus changes), the LLM client (invalidated
when provider/model settings change), and the graph (depends on both).
"""
import json
import logging
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import config
from backend import settings as app_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

def _available_ram_gb() -> float | None:
    try:  # Linux; on macOS/Windows return None and let warm-up proceed
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return None


def _warm_up():
    """Pre-load the embedding model only — it is needed by both ingest and ask.
    The full retriever (reranker + BM25) is another ~2.3GB and is deliberately
    NOT warmed: on small-RAM machines it must not sit in memory while docling
    converts PDFs (measured OOM on an 8GB laptop). The first ask loads it.
    On machines with little free RAM, skip warming entirely — trading first-use
    latency for not being the process the OOM killer picks."""
    try:
        available = _available_ram_gb()
        if available is not None and available < 6:
            log.info(f"Warm-up skipped: {available:.1f}GB RAM available — models load on first use")
            return
        from ingestion.index import get_embedder

        get_embedder()
        log.info("Warm-up complete: embedder ready")
    except Exception:
        log.exception("Warm-up failed (the first use will load models instead)")


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=_warm_up, daemon=True).start()
    yield


app = FastAPI(title="production-rag backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",   # vite dev
        "http://localhost:4173", "http://127.0.0.1:4173",   # vite preview
        "tauri://localhost", "https://tauri.localhost", "http://tauri.localhost",  # packaged shell
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

_retriever_lock = threading.Lock()
_llm_lock = threading.Lock()
_graph_lock = threading.Lock()
_ingest_lock = threading.Lock()

_retriever = None
_files_cache: list[dict] | None = None
_llm = None
_llm_key = None    # (provider, model) the cached client was built for
_graph = None
_graph_key = None  # (id(retriever), provider, model) the cached graph was built from
_ingest = {"running": False, "processed": 0, "total": 0, "error": None}


def get_retriever():
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                from retrieval.retriever import HybridRetriever

                _retriever = HybridRetriever()
    return _retriever


def get_llm_client():
    global _llm, _llm_key
    s = app_settings.load_settings()
    key = (s["provider"], s["model"])
    if _llm is None or _llm_key != key:
        with _llm_lock:
            if _llm is None or _llm_key != key:
                if s["provider"] != "huggingface" and not app_settings.export_key_to_env(s["provider"]):
                    raise HTTPException(status_code=400, detail=f"No API key configured for {s['provider']}")
                from generation.llm import get_llm

                _llm = get_llm(s["model"], s["provider"])
                _llm_key = key
    return _llm, key


def get_graph():
    global _graph, _graph_key
    retriever = get_retriever()        # heavy loads happen before the graph lock
    llm, (provider, model) = get_llm_client()
    key = (id(retriever), provider, model)
    if _graph is None or _graph_key != key:
        with _graph_lock:
            if _graph is None or _graph_key != key:
                from generation.pipeline import build_graph

                _graph = build_graph(retriever, llm, provider=provider)
                _graph_key = key
    return _graph


def _corpus_changed():
    """After an ingest: retriever and graph are stale, the LLM client is not."""
    global _retriever, _files_cache, _graph, _graph_key
    _retriever, _files_cache, _graph, _graph_key = None, None, None, None


def _settings_changed():
    """After a settings change: LLM client and graph are stale, the retriever is not."""
    global _llm, _llm_key, _graph, _graph_key
    _llm, _llm_key, _graph, _graph_key = None, None, None, None


# ---------- corpus ----------

class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    files: list[str] | None = None  # restrict to these document names


def _file_list() -> list[dict]:
    global _files_cache
    if _files_cache is None:
        from ingestion.index import get_collection

        counts: dict[str, int] = {}
        for meta in get_collection().get(include=["metadatas"])["metadatas"]:
            counts[meta["pdf"]] = counts.get(meta["pdf"], 0) + 1
        _files_cache = sorted(({"name": n, "chunks": c} for n, c in counts.items()),
                              key=lambda f: f["name"])
    return _files_cache


@app.get("/api/status")
def status():
    file_list = _file_list()
    return {
        "documents": len(file_list),
        "chunks": sum(f["chunks"] for f in file_list),
        "corpus_dir": str(config.PDF_DIR),
        "ingest": _ingest,
    }


@app.get("/api/files")
def files():
    return _file_list()


def _run_ingest():
    import gc

    from ingestion.index import get_collection
    from ingestion.run import ingest_pdf

    try:
        # Docling peaks at gigabytes per document; drop the retriever (reranker +
        # chunk snapshot) for the duration so the two never coexist in RAM. Asks
        # are 409-blocked during ingest anyway, and the ask after rebuilds it.
        _corpus_changed()
        gc.collect()
        pdfs = sorted(config.PDF_DIR.glob("*.pdf"))
        _ingest.update(processed=0, total=len(pdfs), error=None)
        collection = get_collection()
        for pdf in pdfs:
            try:
                ingest_pdf(collection, pdf)
            except Exception as exc:  # one bad PDF must not abort the rest
                log.exception(f"Failed to ingest {pdf.name}")
                _ingest["error"] = f"{pdf.name}: {exc}"
            _ingest["processed"] += 1
            gc.collect()  # release this document's conversion before the next
        _corpus_changed()
    except Exception as exc:  # surfaced via /api/status, not lost in a thread
        log.exception("Ingest failed")
        _ingest["error"] = str(exc)
    finally:
        _ingest["running"] = False


@app.post("/api/ingest")
def ingest():
    with _ingest_lock:  # claim the running flag before the thread starts, so the guard holds
        if _ingest["running"]:
            raise HTTPException(status_code=409, detail="Ingest already running")
        _ingest["running"] = True
    threading.Thread(target=_run_ingest, daemon=True).start()
    return {"started": True}


def _safe_corpus_path(name: str) -> Path | None:
    """Resolve a client-supplied name to a file inside the corpus dir, or None."""
    try:
        path = (config.PDF_DIR / Path(name).name).resolve()
        if path.parent == config.PDF_DIR.resolve() and path.suffix.lower() == ".pdf":
            return path
    except (ValueError, OSError):  # null bytes, over-long names, etc.
        pass
    return None


@app.post("/api/upload")
async def upload(file: UploadFile):
    path = _safe_corpus_path(file.filename or "")
    if path is None:
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    config.PDF_DIR.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")  # not *.pdf, so a concurrent ingest never globs it
    partial.write_bytes(await file.read())
    partial.replace(path)
    return {"saved": path.name}


@app.get("/api/pdf/{name}")
def pdf(name: str):
    path = _safe_corpus_path(name)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type="application/pdf")


@app.delete("/api/document/{name}")
def delete_document(name: str):
    """Remove a document everywhere: PDF file, its chunks, figures, docling cache."""
    if _ingest["running"]:
        raise HTTPException(status_code=409, detail="Ingest in progress — try again when it finishes")
    path = _safe_corpus_path(name)
    if path is None:
        raise HTTPException(status_code=400, detail="Invalid document name")
    from ingestion.index import get_collection

    collection = get_collection()
    chunks = collection.get(where={"pdf": path.name}, include=["metadatas"])
    if not chunks["ids"] and not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    doc_hashes = {meta["doc_hash"] for meta in chunks["metadatas"]}
    if chunks["ids"]:
        collection.delete(ids=chunks["ids"])
    path.unlink(missing_ok=True)
    for doc_hash in doc_hashes:  # derived artifacts, keyed by content hash
        (config.DOCLING_CACHE / f"{doc_hash}.json").unlink(missing_ok=True)
        for fig in config.FIGURES_DIR.glob(f"{doc_hash}-fig*.png"):
            fig.unlink(missing_ok=True)
    _corpus_changed()
    return {"deleted": path.name, "chunks_removed": len(chunks["ids"])}


@app.get("/api/figure/{name}")
def figure(name: str):
    try:
        path = (config.FIGURES_DIR / Path(name).name).resolve()
        if path.parent != config.FIGURES_DIR.resolve() or path.suffix.lower() != ".png" or not path.is_file():
            raise HTTPException(status_code=404, detail="Not found")
    except (ValueError, OSError):
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type="image/png")


# ---------- ask ----------

@app.post("/api/ask")
def ask(req: AskRequest):
    if _ingest["running"]:
        raise HTTPException(status_code=409, detail="Ingest in progress — try again when it finishes")
    known = {f["name"] for f in _file_list()}
    if not known:
        raise HTTPException(status_code=400, detail="No documents ingested yet")
    scope = req.files or None
    if scope:
        unknown = [f for f in scope if f not in known]
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unknown documents: {unknown}")
    try:
        result = get_graph().invoke({"question": req.question, "scope": scope})
    except HTTPException:
        raise
    except Exception as exc:
        provider_status = getattr(exc, "status_code", None)
        if provider_status == 401:
            raise HTTPException(status_code=401,
                                detail="The LLM provider rejected the configured API key — update it in settings")
        if provider_status == 404:
            raise HTTPException(status_code=400,
                                detail="The configured model name was not found at the provider — check settings")
        if provider_status == 429:
            raise HTTPException(status_code=429,
                                detail="The LLM provider reports a rate/quota limit — check the account's billing or retry later")
        raise
    return {
        "answer": result["answer"],
        # basenames only; the client turns them into /api/figure/{name} URLs
        "figures": [Path(p).name for p in result.get("figures", [])],
        "sources": [
            {
                "n": i,
                "chunk_id": c["chunk_id"],
                "pdf": c["pdf"],
                "headings": c.get("headings", ""),
                "text": c["text"],
                "boxes": json.loads(c["prov"]) if c.get("prov") else [],
            }
            for i, c in enumerate(result["chunks"], 1)
        ],
    }


# ---------- settings ----------

class SettingsRequest(BaseModel):
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None


@app.get("/api/settings")
def get_settings():
    s = app_settings.load_settings()
    return {
        "provider": s["provider"],
        "model": s["model"],
        "has_key": app_settings.get_api_key(s["provider"]) is not None,
        "providers": sorted(app_settings.PROVIDER_ENV),
    }


@app.put("/api/settings")
def put_settings(req: SettingsRequest):
    s = app_settings.load_settings()
    if req.provider is not None:
        if req.provider not in app_settings.PROVIDER_ENV:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {req.provider}")
        s["provider"] = req.provider
    if req.model is not None:
        s["model"] = req.model
    app_settings.save_settings(s)
    if req.api_key:
        app_settings.set_api_key(s["provider"], req.api_key)
    _settings_changed()  # next ask rebuilds the LLM client + graph; retriever survives
    return get_settings()


# ---------- packaged frontend ----------

if getattr(sys, "frozen", False):  # packaged executable: dist is bundled next to the code
    FRONTEND_DIST = Path(sys._MEIPASS) / "frontend" / "dist"
else:
    FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    from fastapi.staticfiles import StaticFiles

    # Registered last: the /api/* routes above always win; everything else
    # serves the built app, so one process is the whole application (run.sh).
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
