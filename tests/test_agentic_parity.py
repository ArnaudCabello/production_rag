"""Parity tests: the agentic graph == baseline, with stubs (no models).

M2 redefinition: the StubLLM's "ANSWER" output is unparseable planner JSON, so
the planner falls back to [question] — behavioural parity (same retrieval, same
final prompt) holds, but llm_calls == 2 (planner + synthesis) and the baseline
prompt is the LAST llm call.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic.graph import build_agentic_graph
from generation.pipeline import build_graph


def chunk(cid, pdf):
    return {"chunk_id": cid, "pdf": pdf, "text": f"text-{cid}", "headings": ""}


class StubRetriever:
    def __init__(self):
        self.chunks = {"a-1": chunk("a-1", "alpha.pdf"), "b-1": chunk("b-1", "beta.pdf")}
        self.calls = []

    def search(self, q, top_k=5, pdfs=None, rerank=True):
        self.calls.append((q, top_k, tuple(pdfs) if pdfs else None, rerank))
        if pdfs == ["alpha.pdf"]:
            return [chunk("a-1", "alpha.pdf"), chunk("a-2", "alpha.pdf")]
        if pdfs == ["beta.pdf"]:
            return [chunk("b-2", "beta.pdf"), chunk("b-3", "beta.pdf")]
        return [chunk("a-1", "alpha.pdf"), chunk("b-1", "beta.pdf")]


class StubLLM:
    def __init__(self):
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        class R: content = "ANSWER"
        return R()


def run_agentic(retriever, llm, question):
    graph = build_agentic_graph(retriever, llm)
    return graph.invoke({"question": question, "llm_calls": 0, "retrieval_calls": 0})


# 1. parity on a plain (non-fanout, non-vision) question: same answer, same chunks
base_ret, base_llm = StubRetriever(), StubLLM()
base = build_graph(base_ret, base_llm).invoke({"question": "what is X"})
ag_ret, ag_llm = StubRetriever(), StubLLM()
ag = run_agentic(ag_ret, ag_llm, "what is X")
assert ag["answer"] == base["answer"]
assert [c["chunk_id"] for c in ag["chunks"]] == [c["chunk_id"] for c in base["chunks"]]
print("parity: same answer and chunk sequence as baseline: OK")

# 2. prompt parity: the final (synthesis) call has identical system + user text;
# the extra first call is the planner
assert len(ag_llm.messages) == 2 and len(base_llm.messages) == 1
assert [m.content for m in ag_llm.messages[-1]] == [m.content for m in base_llm.messages[0]]
print("parity: identical synthesis prompts (system + user): OK")

# 3. retriever-call parity: one search with baseline defaults; counters 2 (planner
# + synthesis) / 1
assert ag_ret.calls == [("what is X", 5, None, True)], ag_ret.calls
assert ag["llm_calls"] == 2 and ag["retrieval_calls"] == 1
print("parity: single search with baseline defaults, counters 2/1: OK")

# 4. deliberate divergence: no multi-doc fan-out in the M1 skeleton — the M2
# planner + M3 loop replace it with targeted sub-queries
ag_ret, ag_llm = StubRetriever(), StubLLM()
ag = run_agentic(ag_ret, ag_llm, "compare these studies")
assert len(ag_ret.calls) == 1 and ag["retrieval_calls"] == 1
print("divergence (intentional): fanout-worded question does one retrieval: OK")

# 5. dedup by chunk_id across sub-queries (future-proofing for multi-round retrieval)
class DupRetriever(StubRetriever):
    def search(self, q, top_k=5, pdfs=None, rerank=True):
        self.calls.append((q, top_k, tuple(pdfs) if pdfs else None, rerank))
        return [chunk("a-1", "alpha.pdf"), chunk("a-1", "alpha.pdf")]

ag = run_agentic(DupRetriever(), StubLLM(), "what is X")
assert [c["chunk_id"] for c in ag["chunks"]] == ["a-1"]
print("dedup by chunk_id: OK")

# 6. trace toggle: off by default (no trace in result), on records per-node events
ag = run_agentic(StubRetriever(), StubLLM(), "what is X")
assert not ag.get("trace"), ag.get("trace")
graph = build_agentic_graph(StubRetriever(), StubLLM(), trace=True)
ag = graph.invoke({"question": "what is X", "llm_calls": 0,
                   "retrieval_calls": 0, "trace": []})
assert [e["node"] for e in ag["trace"]] == ["plan", "retrieve", "synthesize"], ag["trace"]
assert ag["trace"][0]["sub_queries"] == ["what is X"]
assert ag["trace"][1] == {"node": "retrieve", "query": "what is X",
                          "chunk_ids": ["a-1", "b-1"]}
assert ag["trace"][2]["context_chunks"] == 2
print("trace: off by default, on records plan/retrieve/synthesize events: OK")
