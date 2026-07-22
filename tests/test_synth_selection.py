"""T2 tests: query-interleave cap selection + aggregation cap (stubs, no models)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

from agentic.graph import (MAX_SYNTH_CHUNKS, MAX_SYNTH_CHUNKS_AGG,
                           build_agentic_graph, select_synth_chunks)

Q = "what is X"


def chunk(cid, q_idx=None, rnd=None):
    c = {"chunk_id": cid, "pdf": "alpha.pdf", "text": f"text-{cid}", "headings": ""}
    if q_idx is not None:
        c["q_idx"] = q_idx
    if rnd is not None:
        c["round"] = rnd
    return c


class QueryRetriever:
    """Returns a distinct scripted chunk list per query string."""
    def __init__(self, by_query):
        self.by_query = by_query

    def search(self, q, top_k=5, pdfs=None, rerank=True):
        return self.by_query[q]


class ScriptedLLM:
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


# --- select_synth_chunks (pure) ---

# 1. under cap → identity (same objects, same order)
few = [chunk(f"a-{i}", q_idx=0, rnd=1) for i in range(5)]
assert select_synth_chunks(few, 20) is few or select_synth_chunks(few, 20) == few
assert select_synth_chunks(few, 5) == few
print("select: under/at cap → identity: OK")

# 2. over cap → round-robin across q_idx groups; every group represented,
#    question-group top hit first, within-group order kept, exactly cap picks
union = ([chunk(f"q0-{i}", 0, 1) for i in range(10)]
         + [chunk(f"q1-{i}", 1, 1) for i in range(10)]
         + [chunk(f"q2-{i}", 2, 1) for i in range(10)]
         + [chunk(f"q3-{i}", 3, 2) for i in range(3)]
         + [chunk(f"q4-{i}", 4, 3) for i in range(2)])
sel = select_synth_chunks(union, 20)
assert len(sel) == 20
assert sel[0]["chunk_id"] == "q0-0"                       # question group's top hit stays first
assert {c["q_idx"] for c in sel} == {0, 1, 2, 3, 4}       # every group represented
assert [c["chunk_id"] for c in sel if c["q_idx"] == 3] == ["q3-0", "q3-1", "q3-2"]  # late rounds survive
assert [c["chunk_id"] for c in sel if c["q_idx"] == 4] == ["q4-0", "q4-1"]
for g in range(5):  # within-group relevance order preserved
    ids = [c["chunk_id"] for c in sel if c["q_idx"] == g]
    assert ids == sorted(ids, key=lambda s: int(s.split("-")[1]))
assert select_synth_chunks(union, 20) == sel              # deterministic
print("select: interleave over cap, late-round groups represented: OK")

# 3. chunks missing q_idx fall into one group → degrades to first-N
plain = [chunk(f"p-{i}") for i in range(25)]
assert [c["chunk_id"] for c in select_synth_chunks(plain, 20)] == [f"p-{i}" for i in range(20)]
print("select: missing q_idx degrades to first-N: OK")


# --- graph integration ---

AGG_PLAN = json.dumps({"category": "aggregation", "sub_queries": ["sq one", "sq two"]})
SIMPLE_PLAN = json.dumps({"category": "simple", "sub_queries": ["sq one", "sq two"]})
SUFFICIENT = {"sufficient": True, "missing": [], "pending_queries": []}

# 3 round-1 queries × 12 distinct chunks = 36-chunk union
by_query = {Q: [chunk(f"g0-{i}") for i in range(12)],
            "sq one": [chunk(f"g1-{i}") for i in range(12)],
            "sq two": [chunk(f"g2-{i}") for i in range(12)]}

# 4. aggregation category → cap 30; retrieve annotates q_idx/round on copies
llm = ScriptedLLM([AGG_PLAN, "ANSWER [1]."])
ag = run_agentic(QueryRetriever(by_query), llm, check=lambda s: SUFFICIENT)
assert len(ag["chunks"]) == MAX_SYNTH_CHUNKS_AGG == 30
assert ag["chunks"][0]["chunk_id"] == "g0-0"
assert ag["chunks"][0]["q_idx"] == 0 and ag["chunks"][0]["round"] == 1
assert {c["q_idx"] for c in ag["chunks"]} == {0, 1, 2}
assert not any("q_idx" in c for c in by_query[Q])  # retriever dicts never mutated
print("graph: aggregation cap 30, q_idx/round annotated on copies: OK")

# 5. other categories keep cap 20 on the same over-cap union
llm = ScriptedLLM([SIMPLE_PLAN, "ANSWER"])
ag = run_agentic(QueryRetriever(by_query), llm, check=lambda s: SUFFICIENT)
assert len(ag["chunks"]) == MAX_SYNTH_CHUNKS == 20
print("graph: non-aggregation stays at cap 20: OK")

# 6. citations validate against the capped (interleaved) list; [1] resolves to
#    the question group's top hit; a marker beyond the cap is invalid
llm = ScriptedLLM([SIMPLE_PLAN, f"A [1]. B [{MAX_SYNTH_CHUNKS + 1}]."])
ag = run_agentic(QueryRetriever(by_query), llm, check=lambda s: SUFFICIENT)
assert ag["citations"]["valid"] == [1]
assert ag["citations"]["invalid"] == [MAX_SYNTH_CHUNKS + 1]
assert ag["citations"]["chunk_ids"] == ["g0-0"]
print("graph: citations validate against the capped interleaved list: OK")

# 7. trace: synthesize event carries context_rounds and late rounds are non-zero
by_query2 = {Q: [chunk(f"g0-{i}") for i in range(10)],
             "sq one": [chunk(f"g1-{i}") for i in range(10)],
             "sq two": [chunk(f"g2-{i}") for i in range(10)],
             "more": [chunk(f"g3-{i}") for i in range(5)]}
verdicts = [{"sufficient": False, "missing": ["m"], "pending_queries": ["more"]}, SUFFICIENT]
llm = ScriptedLLM([AGG_PLAN, "ANSWER"])
ag = run_agentic(QueryRetriever(by_query2), llm, trace=True,
                 check=lambda s, v=iter(verdicts): next(v))
synth_ev = [e for e in ag["trace"] if e["node"] == "synthesize"][0]
assert len(ag["chunks"]) == 30  # 35-chunk union capped at 30 (aggregation)
assert synth_ev["dropped_chunks"] == 5
assert synth_ev["context_rounds"][1] == 25 and synth_ev["context_rounds"][2] == 5
assert sum(synth_ev["context_rounds"].values()) == 30
print("trace: context_rounds histogram shows round-2 chunks in context: OK")

print("ALL T2 SELECTION TESTS PASSED")
