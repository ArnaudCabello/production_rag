"""Scope + fan-out cap tests with stubs (no models)."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import config
from generation.pipeline import build_graph

def chunk(cid, pdf):
    return {"chunk_id": cid, "pdf": pdf, "text": "t", "headings": ""}

class StubRetriever:
    def __init__(self, docs):
        self.chunks = {f"{d}-1": chunk(f"{d}-1", d) for d in docs}
        self.calls = []
    def search(self, q, top_k=5, pdfs=None, rerank=True):
        self.calls.append((top_k, tuple(pdfs) if pdfs else None))
        if pdfs:
            return [chunk(f"{p}-{i}", p) for p in pdfs for i in (1, 2)][:top_k]
        return [chunk("a.pdf-1", "a.pdf")]

class StubLLM:
    def invoke(self, messages):
        class R: content = "ANSWER"
        return R()

# scoped ask: main retrieval filtered to scope; fan-out over scope only
ret = StubRetriever(["a.pdf", "b.pdf", "c.pdf"])
out = build_graph(ret, StubLLM()).invoke({"question": "compare these studies", "scope": ["a.pdf", "b.pdf"]})
assert ret.calls == [(5, ("a.pdf", "b.pdf")), (2, ("a.pdf",)), (2, ("b.pdf",))], ret.calls
pdfs_in_answer = {c["pdf"] for c in out["chunks"]}
assert pdfs_in_answer == {"a.pdf", "b.pdf"}, pdfs_in_answer
print("scoped multi-doc: filter + fan-out limited to scope: OK")

# scoped single-doc question: one filtered retrieval, no fan-out
ret = StubRetriever(["a.pdf", "b.pdf"])
out = build_graph(ret, StubLLM()).invoke({"question": "what is X", "scope": ["b.pdf"]})
assert ret.calls == [(5, ("b.pdf",))], ret.calls
print("scoped single-doc: one filtered call: OK")

# unscoped ask without scope key at all (eval/app path) still works
ret = StubRetriever(["a.pdf", "b.pdf"])
out = build_graph(ret, StubLLM()).invoke({"question": "what is X"})
assert ret.calls == [(5, None)]
print("no scope key: unchanged behavior: OK")

# fan-out cap: >MULTI_DOC_MAX_DOCS docs -> only docs surfaced by main retrieval
docs = [f"d{i}.pdf" for i in range(12)]
ret = StubRetriever(docs)
out = build_graph(ret, StubLLM()).invoke({"question": "compare these studies"})
fanout_targets = [c[1] for c in ret.calls[1:]]
assert fanout_targets == [("a.pdf",)], fanout_targets  # only the doc the main retrieval surfaced
print("fan-out capped to surfaced documents on large corpora: OK")
