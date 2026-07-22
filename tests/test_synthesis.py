"""M5 synthesis tests: citation parsing/validation, context capping (stubs, no models)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

from agentic.citations import extract_citations
from agentic.graph import MAX_SYNTH_CHUNKS, build_agentic_graph
from generation.pipeline import USER_TEMPLATE, format_context

Q = "what is X"


def chunk(cid, pdf="alpha.pdf"):
    return {"chunk_id": cid, "pdf": pdf, "text": f"text-{cid}", "headings": ""}


class StubRetriever:
    def __init__(self, chunks=None):
        self.results = chunks or [chunk("a-1"), chunk("b-1", "beta.pdf"), chunk("c-1")]
        self.calls = []

    def search(self, q, top_k=5, pdfs=None, rerank=True):
        self.calls.append((q, top_k, tuple(pdfs) if pdfs else None, rerank))
        return self.results


class ScriptedLLM:
    """Returns scripted responses in invoke order (plan, check, ..., synthesis)."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        content = self.responses[len(self.messages) - 1]
        class R: pass
        R.content = content
        return R()


def run_agentic(retriever, llm, question=Q, **kw):
    graph = build_agentic_graph(retriever, llm, **kw)
    state = {"question": question, "llm_calls": 0, "retrieval_calls": 0,
             "chunks": [], "rounds": 0, "pending_queries": [], "queries_run": [],
             "gaps": [], "new_chunks": 0}
    if kw.get("trace"):
        state["trace"] = []
    return graph.invoke(state)


SUFFICIENT = json.dumps({"sufficient": True, "missing": [], "queries": []})


# --- extract_citations ---

# 1. single, adjacent, and comma-grouped markers
c = extract_citations("A [1]. B [2][3]. C [1, 2].", 3)
assert c["markers"] == [1, 2, 3, 1, 2]
assert c["valid"] == [1, 2, 3]
assert c["invalid"] == []
print("extract: [1] / [2][3] / [1, 2] forms: OK")

# 2. out-of-range and zero are invalid; valid/invalid deduped, markers keep dupes
c = extract_citations("X [1][9]. Y [0]. Z [9].", 3)
assert c["markers"] == [1, 9, 0, 9]
assert c["valid"] == [1]
assert c["invalid"] == [9, 0]
print("extract: out-of-range/zero invalid, dedup in valid/invalid: OK")

# 3. no markers → all empty
c = extract_citations("This answer cites nothing.", 3)
assert c == {"markers": [], "valid": [], "invalid": []}
print("extract: no markers → empty: OK")

# 4. non-citation brackets ignored
c = extract_citations("See [see Fig. 2] and [a] and [] and [1.5].", 3)
assert c == {"markers": [], "valid": [], "invalid": []}
print("extract: non-numeric brackets ignored: OK")


# --- graph integration ---

# 5. scripted cited answer over 3 chunks → valid/invalid split, chunk_ids resolve
#    positionally (chunks[n-1] in retrieval order)
ret = StubRetriever()
llm = ScriptedLLM(["PLAN-GARBAGE", SUFFICIENT, "X is 5 [1][3]. Y unknown [7]."])
ag = run_agentic(ret, llm)
assert ag["citations"]["valid"] == [1, 3]
assert ag["citations"]["invalid"] == [7]
assert ag["citations"]["chunk_ids"] == ["a-1", "c-1"]
print("graph: citations parsed and resolved to chunk_ids: OK")

# 6. capping: >MAX_SYNTH_CHUNKS retrieved → prompt numbers stop at the cap,
#    result chunks reflect what the LLM saw, trace records dropped count
big = [chunk(f"x-{i}") for i in range(MAX_SYNTH_CHUNKS + 5)]
ret = StubRetriever(big)
llm = ScriptedLLM(["PLAN-GARBAGE", SUFFICIENT, "ANSWER"])
ag = run_agentic(ret, llm, trace=True)
synth_user = llm.messages[-1][1].content
assert f"[{MAX_SYNTH_CHUNKS}]" in synth_user
assert f"[{MAX_SYNTH_CHUNKS + 1}]" not in synth_user
assert len(ag["chunks"]) == MAX_SYNTH_CHUNKS
assert [c["chunk_id"] for c in ag["chunks"]] == [f"x-{i}" for i in range(MAX_SYNTH_CHUNKS)]
synth_ev = [e for e in ag["trace"] if e["node"] == "synthesize"][0]
assert synth_ev["context_chunks"] == MAX_SYNTH_CHUNKS
assert synth_ev["dropped_chunks"] == 5
assert synth_ev["context_rounds"] == {1: MAX_SYNTH_CHUNKS}  # T2: all round-1 here
print("graph: context capped deterministically, dropped count traced: OK")

# 7. no-cap parity: ≤cap chunks → synthesis prompt byte-identical to baseline,
#    uncited answer → empty citations, nothing dropped
ret = StubRetriever()
llm = ScriptedLLM(["PLAN-GARBAGE", SUFFICIENT, "ANSWER"])
ag = run_agentic(ret, llm, trace=True)
assert llm.messages[-1][1].content == USER_TEMPLATE.format(
    context=format_context(ag["chunks"]), question=Q)
assert ag["citations"] == {"markers": [], "valid": [], "invalid": [], "chunk_ids": []}
synth_ev = [e for e in ag["trace"] if e["node"] == "synthesize"][0]
assert synth_ev["dropped_chunks"] == 0
assert synth_ev["citations_valid"] == 0 and synth_ev["citations_invalid"] == 0
print("graph: under cap → baseline prompt intact, empty citations: OK")

# 8. budget: llm_calls unchanged by M5 (plan + check + synthesis = 3)
assert ag["llm_calls"] == 3
print("budget: llm_calls == 3, M5 adds none: OK")

# 9. citation validity resolves against the CAPPED list — a marker inside the
#    cap is valid, one beyond it is invalid
ret = StubRetriever(big)
llm = ScriptedLLM(["PLAN-GARBAGE", SUFFICIENT,
                   f"A [1]. B [{MAX_SYNTH_CHUNKS}]. C [{MAX_SYNTH_CHUNKS + 3}]."])
ag = run_agentic(ret, llm)
assert ag["citations"]["valid"] == [1, MAX_SYNTH_CHUNKS]
assert ag["citations"]["invalid"] == [MAX_SYNTH_CHUNKS + 3]
assert ag["citations"]["chunk_ids"] == ["x-0", f"x-{MAX_SYNTH_CHUNKS - 1}"]
print("graph: validity measured against the capped context: OK")


# --- scorer metric ---

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from score_benchmark import citation_validity  # noqa: E402

assert citation_validity("A [1]. B [2].", [chunk("a-1"), chunk("b-1")]) == 1.0
assert citation_validity("A [1]. B [9].", [chunk("a-1"), chunk("b-1")]) == 0.5
assert citation_validity("no citations here", [chunk("a-1")]) is None
assert citation_validity("this information is not available in the corpus", []) is None
print("scorer: citation_validity 1.0 / 0.5 / None cases: OK")
