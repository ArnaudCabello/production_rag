"""Backend API tests against the real (ReAct-only) index; graph stubbed for /ask."""
import os
import sys
import tempfile

os.environ["RAG_APP_DIR"] = tempfile.mkdtemp()  # isolate settings/keys from any real config
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

import backend.main as backend_main

client = TestClient(backend_main.app)

# status + files reflect the real index (any corpus with ReAct.pdf ingested)
s = client.get("/api/status").json()
assert s["documents"] >= 1 and s["chunks"] > 50, s
f = client.get("/api/files").json()
assert {x["name"] for x in f} >= {"ReAct.pdf"}, f
assert sum(x["chunks"] for x in f) == s["chunks"], (f, s)
print("status/files: OK", s["documents"], "docs,", s["chunks"], "chunks")

# pdf serving + traversal attempts
assert client.get("/api/pdf/ReAct.pdf").status_code == 200
assert client.get("/api/pdf/ReAct.pdf").headers["content-type"] == "application/pdf"
for evil in ("..%2F..%2Fconfig.py", "..%2Fdocling%2Fx.json", "nope.pdf", "ReAct.pdf%00.txt"):
    code = client.get(f"/api/pdf/{evil}").status_code
    assert code == 404, (evil, code)
print("pdf serving + traversal guard: OK")

# settings: defaults, masking, unknown provider rejected, key never echoed
s = client.get("/api/settings").json()
assert s["provider"] == "anthropic" and s["has_key"] in (False, True) and "api_key" not in s
assert client.put("/api/settings", json={"provider": "not-a-provider"}).status_code == 400
r = client.put("/api/settings", json={"provider": "openai", "model": "gpt-test", "api_key": "sk-secret-123"}).json()
assert r["provider"] == "openai" and r["has_key"] is True and "sk-secret" not in str(r)
print("settings + key masking: OK")

# ask validation: unknown scope file rejected; empty question rejected
assert client.post("/api/ask", json={"question": "hi", "files": ["ghost.pdf"]}).status_code == 400
assert client.post("/api/ask", json={"question": ""}).status_code == 422

# ask with stubbed graph: response shape includes prov for highlighting
class StubGraph:
    def invoke(self, state):
        assert state["scope"] == ["ReAct.pdf"]
        return {"answer": "stub [1]", "chunks": [
            {"chunk_id": "x-0001", "pdf": "ReAct.pdf", "headings": "H", "text": "T",
             "prov": "[[2, 1.0, 2.0, 3.0, 4.0]]"}]}

backend_main.get_graph = lambda: StubGraph()
r = client.post("/api/ask", json={"question": "what is ReAct?", "files": ["ReAct.pdf"]}).json()
src = r["sources"][0]
assert r["answer"] == "stub [1]" and src["n"] == 1
assert src["boxes"] == [[2, 1.0, 2.0, 3.0, 4.0]]  # decoded, not stringly-typed
print("ask: scope validation + structured boxes: OK")

# regression: get_retriever must not deadlock when called from a caller holding no lock twice over
import threading
done = threading.Event()
def _load_twice():
    backend_main.get_retriever()
    backend_main.get_retriever()
    done.set()
t = threading.Thread(target=_load_twice, daemon=True)
t.start()
t.join(timeout=300)
assert done.is_set(), "get_retriever deadlocked or took >300s"
print("retriever caching: no deadlock, idempotent: OK")

# upload: non-pdf rejected, pdf accepted with sanitized name
r = client.post("/api/upload", files={"file": ("evil.txt", b"x", "text/plain")})
assert r.status_code == 400
r = client.post("/api/upload", files={"file": ("../../sneaky.pdf", b"%PDF-1.4 test", "application/pdf")})
assert r.status_code == 200 and r.json()["saved"] == "sneaky.pdf"
import config
assert (config.PDF_DIR / "sneaky.pdf").exists() and not (config.PDF_DIR.parent.parent / "sneaky.pdf").exists()
(config.PDF_DIR / "sneaky.pdf").unlink()
print("upload: type + filename sanitization: OK")
