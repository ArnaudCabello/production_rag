"""M3 retrieval-loop tests: round cap, termination, dedup, aggregation knob (stubs, no models)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

from agentic.graph import (AGG_SUBQUERY_TOP_K, MAX_PENDING_PER_ROUND, MAX_ROUNDS,
                           build_agentic_graph)

Q = "what is X"


def chunk(cid, pdf="alpha.pdf"):
    return {"chunk_id": cid, "pdf": pdf, "text": f"text-{cid}", "headings": ""}


class StubRetriever:
    def __init__(self):
        self.calls = []

    def search(self, q, top_k=5, pdfs=None, rerank=True):
        self.calls.append((q, top_k, tuple(pdfs) if pdfs else None, rerank))
        return [chunk("a-1"), chunk("b-1", "beta.pdf")]


class PlannerLLM:
    """First invoke returns the planner JSON, later ones the answer."""
    def __init__(self, plan_json="ANSWER"):  # default: fallback plan [question]
        self.plan_json = plan_json
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        content = self.plan_json if len(self.messages) == 1 else "ANSWER"
        class R: pass
        R.content = content
        return R()


def run_agentic(retriever, llm, question, **kw):
    graph = build_agentic_graph(retriever, llm, **kw)
    state = {"question": question, "llm_calls": 0, "retrieval_calls": 0,
             "chunks": [], "rounds": 0, "pending_queries": [], "queries_run": [],
             "gaps": []}
    if kw.get("trace"):
        state["trace"] = []
    return graph.invoke(state)


# 1. default path (no check injection): the M4 LLM check runs; PlannerLLM's "ANSWER"
#    is unparseable → fail-safe sufficient, 1 round, llm_calls 3 (plan+check+synth)
ret = StubRetriever()
ag = run_agentic(ret, PlannerLLM(), Q, trace=True)
assert ag["rounds"] == 1
assert ag["llm_calls"] == 3 and ag["retrieval_calls"] == 1
assert [e["node"] for e in ag["trace"]] == ["plan", "retrieve", "check", "synthesize"]
check_ev = ag["trace"][2]
assert check_ev["sufficient"] is True and check_ev["rounds"] == 1
assert ag["answer"] == "ANSWER"
print("default: 1 round, fail-safe LLM check, counters 3/1: OK")

# 2. termination at cap: check always insufficient with fresh queries → exactly MAX_ROUNDS
ret = StubRetriever()
always_more = lambda s: {"sufficient": False, "pending_queries": [f"r{s['rounds']}"]}
ag = run_agentic(ret, PlannerLLM(), Q, check=always_more)
assert ag["rounds"] == MAX_ROUNDS, ag["rounds"]
assert [c[0] for c in ret.calls] == [Q, "r1", "r2", "r3"]
assert ag["retrieval_calls"] == MAX_ROUNDS
assert ag["answer"] == "ANSWER"  # still synthesized after cap
print("termination: hard cap at MAX_ROUNDS despite insufficient verdicts: OK")

# 3. no re-retrieval: re-enqueued query (case/space variant) never searched again
ret = StubRetriever()
renq = lambda s: {"sufficient": False, "pending_queries": ["  WHAT is x "]}
ag = run_agentic(ret, PlannerLLM(), Q, check=renq)
assert [c[0] for c in ret.calls] == [Q], ret.calls
assert ag["retrieval_calls"] == 1
print("dedup: already-run query variant not re-searched, loop terminates: OK")

# 4. cross-round chunk dedup: overlapping ids across rounds → unique union, first-seen order
class OverlapRetriever(StubRetriever):
    def search(self, q, top_k=5, pdfs=None, rerank=True):
        self.calls.append((q, top_k, tuple(pdfs) if pdfs else None, rerank))
        return [chunk("a-1"), chunk(f"new-{len(self.calls)}")]

ret = OverlapRetriever()
ag = run_agentic(ret, PlannerLLM(), Q, check=always_more)
ids = [c["chunk_id"] for c in ag["chunks"]]
assert ids == ["a-1", "new-1", "new-2", "new-3", "new-4"], ids
print("dedup: chunk union unique across rounds, first-seen order: OK")

# 5. exhausted termination: insufficient but no pending queries → stop after 1 round
ret = StubRetriever()
ag = run_agentic(ret, PlannerLLM(), Q, check=lambda s: {"sufficient": False})
assert ag["rounds"] == 1 and ag["answer"] == "ANSWER"
print("termination: insufficient + no pending queries stops: OK")

# 6. pending cap: 5 enqueued → only MAX_PENDING_PER_ROUND searched in round 2
ret = StubRetriever()
def flood(s):
    if s["rounds"] == 1:
        return {"sufficient": False, "pending_queries": [f"p{i}" for i in range(5)]}
    return {"sufficient": True}
ag = run_agentic(ret, PlannerLLM(), Q, check=flood)
assert [c[0] for c in ret.calls] == [Q, "p0", "p1", "p2"], ret.calls
assert ag["rounds"] == 2
print("pending cap: only MAX_PENDING_PER_ROUND per refinement round: OK")
assert MAX_PENDING_PER_ROUND == 3

# 7. aggregation knob: question reranked with defaults, sub-queries broad un-reranked;
#    refinement queries for aggregation are broad too
agg_plan = json.dumps({"category": "aggregation", "sub_queries": ["ZrB2 melting", "HfB2 melting"]})
ret = StubRetriever()
ag = run_agentic(ret, PlannerLLM(agg_plan), Q, check=lambda s: (
    {"sufficient": False, "pending_queries": ["refined"]} if s["rounds"] == 1
    else {"sufficient": True}))
assert ret.calls[0] == (Q, 5, None, True), ret.calls[0]
assert ret.calls[1] == ("ZrB2 melting", AGG_SUBQUERY_TOP_K, None, False)
assert ret.calls[2] == ("HfB2 melting", AGG_SUBQUERY_TOP_K, None, False)
assert ret.calls[3] == ("refined", AGG_SUBQUERY_TOP_K, None, False)
print("aggregation: question reranked, sub/refinement queries broad (8, no rerank): OK")

# 8. comparative: all searches use defaults
cmp_plan = json.dumps({"category": "comparative", "sub_queries": ["A hardness", "B hardness"]})
ret = StubRetriever()
run_agentic(ret, PlannerLLM(cmp_plan), Q)
assert all(c[1] == 5 and c[3] is True for c in ret.calls), ret.calls
print("comparative: all searches use reranked defaults: OK")

# 9. trace: retrieve events carry round/broad; check events sufficient/rounds
ret = StubRetriever()
ag = run_agentic(ret, PlannerLLM(agg_plan), Q, trace=True, check=lambda s: (
    {"sufficient": False, "pending_queries": ["refined"]} if s["rounds"] == 1
    else {"sufficient": True}))
ev = [e for e in ag["trace"] if e["node"] == "retrieve"]
assert [(e["round"], e["broad"]) for e in ev] == \
    [(1, False), (1, True), (1, True), (2, True)], ev
checks = [e for e in ag["trace"] if e["node"] == "check"]
assert [(e["sufficient"], e["rounds"]) for e in checks] == [(False, 1), (True, 2)]
print("trace: retrieve round/broad + check sufficient/rounds: OK")
