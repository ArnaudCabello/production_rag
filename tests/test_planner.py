"""M2 planner tests: JSON parsing/fallback, graph integration, trace (stubs, no models)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

from agentic.planner import CATEGORIES, MAX_SUB_QUERIES, parse_plan
from agentic.graph import build_agentic_graph

Q = "what is X"

# 1. valid JSON → parsed category + sub_queries, no fallback
plan = parse_plan('{"category": "comparative", "sub_queries": ["ZrB2 hardness", "HfB2 hardness"]}', Q)
assert plan["category"] == "comparative"
assert plan["sub_queries"] == ["ZrB2 hardness", "HfB2 hardness"]
assert not plan.get("fallback")
print("parse: valid JSON: OK")

# 2. JSON wrapped in prose / code fences → still parsed
wrapped = 'Sure, here is the plan:\n```json\n{"category": "aggregation", "sub_queries": ["ZrB2 melting point"]}\n```\nDone.'
plan = parse_plan(wrapped, Q)
assert plan["category"] == "aggregation" and plan["sub_queries"] == ["ZrB2 melting point"]
print("parse: JSON in prose/fences: OK")

# 3. failures → deterministic fallback, flagged
for bad in [
    "ANSWER",                                                    # garbage
    '{"category": "comparative", "sub_queries": ["a", ',         # truncated
    '{"category": "comparative", "sub_queries": "not a list"}',  # wrong type
    '{"category": "banana", "sub_queries": ["a"]}',              # unknown category
    '{"category": "simple", "sub_queries": []}',                 # empty list
    '{"category": "simple", "sub_queries": ["", "  "]}',         # all-empty strings
    '{"sub_queries": ["a"]}',                                    # missing category
    '{"category": "simple"}',                                    # missing sub_queries
    '{"category": "simple", "sub_queries": [1, 2]}',             # non-str items
]:
    plan = parse_plan(bad, Q)
    assert plan == {"category": "simple", "sub_queries": [Q], "fallback": True}, (bad, plan)
print("parse: fallback on garbage/truncated/invalid: OK")

# 4. sub_queries stripped, empties dropped, capped at MAX_SUB_QUERIES
plan = parse_plan(json.dumps({"category": "aggregation",
                              "sub_queries": [" a ", "", "b", "c", "d", "e", "f"]}), Q)
assert plan["sub_queries"] == ["a", "b", "c", "d"] and len(plan["sub_queries"]) == MAX_SUB_QUERIES
assert not plan.get("fallback")
print("parse: strip/drop-empty/cap: OK")

# 5. label set is the PRD's five
assert CATEGORIES == {"simple", "comparative", "multi_hop", "aggregation", "unanswerable_maybe"}
print("categories: PRD label set: OK")


# --- graph integration (stubs mirror tests/test_agentic_parity.py) ---

def chunk(cid, pdf):
    return {"chunk_id": cid, "pdf": pdf, "text": f"text-{cid}", "headings": ""}


class StubRetriever:
    def __init__(self):
        self.calls = []

    def search(self, q, top_k=5, pdfs=None, rerank=True):
        self.calls.append((q, top_k, tuple(pdfs) if pdfs else None, rerank))
        return [chunk("a-1", "alpha.pdf"), chunk("b-1", "beta.pdf")]


class PlannerLLM:
    """First invoke returns the planner JSON, later ones the answer."""
    def __init__(self, plan_json):
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


# 6. planner sub_queries drive retrieval: question always first, one search per query
ret = StubRetriever()
llm = PlannerLLM('{"category": "comparative", "sub_queries": ["ZrB2 hardness", "HfB2 hardness"]}')
ag = run_agentic(ret, llm, Q)
assert [c[0] for c in ret.calls] == [Q, "ZrB2 hardness", "HfB2 hardness"], ret.calls
assert ag["sub_queries"][0] == Q
assert ag["category"] == "comparative"
assert ag["llm_calls"] == 3 and ag["retrieval_calls"] == 3  # M4: +1 check call
assert [c["chunk_id"] for c in ag["chunks"]] == ["a-1", "b-1"]  # deduped union
assert ag["answer"] == "ANSWER"
print("graph: planner queries drive retrieval, question first, counters 3/3: OK")

# 7. sub-query duplicating the question (case/space-insensitive) is not re-run
ret = StubRetriever()
ag = run_agentic(ret, PlannerLLM('{"category": "simple", "sub_queries": ["  What is x "]}'), Q)
assert [c[0] for c in ret.calls] == [Q], ret.calls
assert ag["retrieval_calls"] == 1
print("graph: question-duplicate sub-query deduped: OK")

# 8. total queries capped at 1 + MAX_SUB_QUERIES
ret = StubRetriever()
many = json.dumps({"category": "aggregation", "sub_queries": [f"q{i}" for i in range(9)]})
ag = run_agentic(ret, PlannerLLM(many), Q)
assert len(ret.calls) == 1 + MAX_SUB_QUERIES, ret.calls
print("graph: retrieval capped at 1 + MAX_SUB_QUERIES: OK")

# 9. unparseable planner output → fallback == M1 behaviour (single retrieval)
ret = StubRetriever()
ag = run_agentic(ret, PlannerLLM("ANSWER"), Q)
assert [c[0] for c in ret.calls] == [Q]
assert ag["category"] == "simple" and ag["llm_calls"] == 3  # M4: +1 check call
print("graph: fallback degrades to single-query retrieval: OK")

# 10. trace: plan event carries category/sub_queries/fallback/raw
raw = '{"category": "comparative", "sub_queries": ["ZrB2 hardness"]}'
ag = run_agentic(StubRetriever(), PlannerLLM(raw), Q, trace=True)
ev = ag["trace"][0]
assert ev["node"] == "plan" and ev["category"] == "comparative"
assert ev["sub_queries"] == [Q, "ZrB2 hardness"]
assert ev["fallback"] is False and ev["raw"] == raw
ag = run_agentic(StubRetriever(), PlannerLLM("garbage"), Q, trace=True)
assert ag["trace"][0]["fallback"] is True
print("trace: plan event has category/sub_queries/fallback/raw: OK")
