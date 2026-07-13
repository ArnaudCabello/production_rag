"""Multi-document fan-out tests with stubs (no models)."""
import sys
from pathlib import Path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from generation.pipeline import build_graph, MULTI_DOC_QUESTION

def chunk(cid, pdf):
    return {"chunk_id": cid, "pdf": pdf, "text": "t", "headings": ""}

class StubRetriever:
    def __init__(self):
        self.chunks = {"a-1": chunk("a-1", "alpha.pdf"), "b-1": chunk("b-1", "beta.pdf")}
        self.calls = []
    def search(self, q, top_k=5, pdfs=None):
        self.calls.append((q, top_k, tuple(pdfs) if pdfs else None))
        if pdfs == ["alpha.pdf"]:
            return [chunk("a-1", "alpha.pdf"), chunk("a-2", "alpha.pdf")]
        if pdfs == ["beta.pdf"]:
            return [chunk("b-2", "beta.pdf"), chunk("b-3", "beta.pdf")]
        return [chunk("a-1", "alpha.pdf")]

class StubLLM:
    def invoke(self, messages):
        class R: content = "ANSWER"
        return R()

# multi-doc wording -> per-document fan-out, dedup, original first
ret = StubRetriever()
out = build_graph(ret, StubLLM()).invoke({"question": "compare these studies"})
assert [c["chunk_id"] for c in out["chunks"]] == ["a-1", "a-2", "b-2", "b-3"]
assert ret.calls == [("compare these studies", 5, None),
                     ("compare these studies", 2, ("alpha.pdf",)),
                     ("compare these studies", 2, ("beta.pdf",))], ret.calls
print("multi-doc fan-out: every document contributes, deduped: OK")

# single-doc wording -> plain retrieval only
ret = StubRetriever()
out = build_graph(ret, StubLLM()).invoke({"question": "what is X"})
assert [c["chunk_id"] for c in out["chunks"]] == ["a-1"] and len(ret.calls) == 1
print("single-doc question -> no fan-out: OK")

import config
config.MULTI_DOC_FANOUT = False
ret = StubRetriever()
out = build_graph(ret, StubLLM()).invoke({"question": "compare these studies"})
assert [c["chunk_id"] for c in out["chunks"]] == ["a-1"] and len(ret.calls) == 1
print("MULTI_DOC_FANOUT=False: OK")

import json
qs = json.load(open(Path(__file__).resolve().parent.parent / "eval" / "golden_set.json"))["questions"]
assert [q["id"] for q in qs if MULTI_DOC_QUESTION.search(q["question"])] == ["x01", "x02", "x03"]
print("trigger still matches exactly x01-x03: OK")

# vision route returns the figures it was shown (A/B/C order); text route returns none
import generation.vision as vision
vision.answer_with_figures_api = lambda llm, q, chunks, fmt: "VISION ANSWER"
config.MULTI_DOC_FANOUT = True

class FigRetriever(StubRetriever):
    def search(self, q, top_k=5, pdfs=None):
        return [dict(chunk("a-1", "alpha.pdf"), figures="data/figures/x-fig002.png,data/figures/x-fig001.png")]

out = build_graph(FigRetriever(), StubLLM(), provider="anthropic").invoke({"question": "what do the EDS maps show"})
assert out["answer"] == "VISION ANSWER"
assert out["figures"] == ["data/figures/x-fig002.png", "data/figures/x-fig001.png"], out["figures"]
out = build_graph(FigRetriever(), StubLLM(), provider="anthropic").invoke({"question": "what is the hardness"})
assert out["answer"] == "ANSWER" and not out.get("figures")
print("vision route returns figures in prompt order; text route returns none: OK")
