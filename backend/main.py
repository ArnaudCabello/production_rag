"""FastAPI backend for the desktop app: ingest, scoped ask, settings, PDF serving.

Run (repo root):
    uvicorn backend.main:app --port 8642

The heavy pieces (retriever models, the LLM client, the graph) load lazily on
first use and are cached; finishing an ingest invalidates the retriever so new
documents become searchable without a restart.
"""
import logging
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import config
from backend import settings as app_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="production-rag backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "tauri://localhost"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_lock = threading.Lock()
_retriever = None
_graph = None
_graph_key = None  # (provider, model) the cached graph was built with
_ingest = {"running": False, "processed": 0, "total": 0, "error": None}


def get_retriever():
    global _retriever
    with _lock:
        if _retriever is None:
            from retrieval.retriever import HybridRetriever

            _retriever = HybridRetriever()
        return _retriever


def get_graph():
    """Graph for the currently configured provider/model (rebuilt on change)."""
    global _graph, _graph_key
    s = app_settings.load_settings()
    key = (s["provider"], s["model"])
    with _lock:
        if _graph is None or _graph_key != key:
            if s["provider"] != "huggingface" and not app_settings.export_key_to_env(s["provider"]):
                raise HTTPException(status_code=400, detail=f"No API key configured for {s['provider']}")
            from generation.llm import get_llm
            from generation.pipeline import build_graph

            config.GENERATOR_PROVIDER = s["provider"]  # vision routing reads this
            _graph = build_graph(get_retriever(), get_llm(s["model"], s["provider"]))
            _graph_key = key
        return _graph


def _invalidate():
    global _retriever, _graph, _graph_key
    with _lock:
        _retriever, _graph, _graph_key = None, None, None


# ---------- corpus ----------

class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    files: list[str] | None = None  # restrict to these document names


@app.get("/api/status")
def status():
    from ingestion.index import get_collection

    collection = get_collection()
    names = {m["pdf"] for m in collection.get(include=["metadatas"])["metadatas"]}
    return {
        "documents": len(names),
        "chunks": collection.count(),
        "corpus_dir": str(config.PDF_DIR),
        "ingest": _ingest,
    }


@app.get("/api/files")
def files():
    from ingestion.index import get_collection

    counts: dict[str, int] = {}
    for meta in get_collection().get(include=["metadatas"])["metadatas"]:
        counts[meta["pdf"]] = counts.get(meta["pdf"], 0) + 1
    return sorted(({"name": n, "chunks": c} for n, c in counts.items()), key=lambda f: f["name"])


def _run_ingest():
    from ingestion.index import get_collection
    from ingestion.run import ingest_pdf

    try:
        pdfs = sorted(config.PDF_DIR.glob("*.pdf"))
        _ingest.update(running=True, processed=0, total=len(pdfs), error=None)
        collection = get_collection()
        for pdf in pdfs:
            ingest_pdf(collection, pdf)
            _ingest["processed"] += 1
        _invalidate()
    except Exception as exc:  # surfaced via /api/status, not lost in a thread
        log.exception("Ingest failed")
        _ingest["error"] = str(exc)
    finally:
        _ingest["running"] = False


@app.post("/api/ingest")
def ingest():
    if _ingest["running"]:
        raise HTTPException(status_code=409, detail="Ingest already running")
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
    path.write_bytes(await file.read())
    return {"saved": path.name}


@app.get("/api/pdf/{name}")
def pdf(name: str):
    path = _safe_corpus_path(name)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type="application/pdf")


# ---------- ask ----------

@app.post("/api/ask")
def ask(req: AskRequest):
    if _ingest["running"]:
        raise HTTPException(status_code=409, detail="Ingest in progress — try again when it finishes")
    known = {f["name"] for f in files()}
    if not known:
        raise HTTPException(status_code=400, detail="No documents ingested yet")
    scope = req.files or None
    if scope:
        unknown = [f for f in scope if f not in known]
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unknown documents: {unknown}")
    result = get_graph().invoke({"question": req.question, "scope": scope})
    return {
        "answer": result["answer"],
        "sources": [
            {
                "n": i,
                "chunk_id": c["chunk_id"],
                "pdf": c["pdf"],
                "headings": c.get("headings", ""),
                "text": c["text"],
                "prov": c.get("prov", ""),
                "figures": c.get("figures", ""),
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
    _invalidate()  # next ask rebuilds the graph with the new provider/model
    return get_settings()
