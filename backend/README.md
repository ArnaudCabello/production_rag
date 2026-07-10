# Backend (desktop app API)

Requires **Python 3.11 or 3.12** — newer versions lack prebuilt wheels for
several dependencies (lxml, torch, chromadb) and fall back to source builds
that fail. On the API-generation path, install CPU-only torch first — torch and
torchvision must come from the same index or torchvision fails at import
(`operator torchvision::nms does not exist`):
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
(~250MB instead of the ~6GB CUDA bundle).

```bash
python -m uvicorn backend.main:app --port 8642
```

Endpoints: `/api/status`, `/api/files`, `/api/ingest` (background, poll status),
`/api/upload`, `/api/pdf/{name}`, `/api/ask` (optional `files` scope),
`/api/settings` (provider/model; API keys go to the OS credential store and are
never returned).

Tests (plain scripts; `test_api.py` needs an ingested corpus):

```bash
python tests/test_pipeline.py && python tests/test_scope.py && python tests/test_api.py
```

## Notes and known v1 trade-offs

- **Indexes created before the provenance change must be rebuilt** (delete
  `chroma_db/` + `data/docling/` and re-ingest) or citation highlighting has no
  boxes to draw — old chunks lack `prov` metadata and nothing backfills it.
- Provenance is stored as JSON in chunk metadata. Fine at lab scale; at
  thousands of documents it inflates full-collection reads and retriever RAM —
  revisit (sidecar storage) during the packaging phase.
- An ask that is already retrieving when an ingest starts may see a slightly
  stale index (never a crash — unknown chunk ids are filtered).
- Multi-document fan-out queries every document in scope; above
  `MULTI_DOC_MAX_DOCS` it falls back to the documents the main retrieval
  surfaced. A real document-selection stage is the planned follow-up for large
  corpora.
- Fan-out searches run sequentially (one embed + rerank per document). Cheap at
  lab scale; batch if profiling ever says otherwise.
