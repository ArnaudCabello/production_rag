"""T1 round-efficiency tests: no-new-chunks early stop, stalled-check stop,
factual checker rule (stubs, no models) — see agent/plans/T1_round_efficiency.md."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic.checker import CHECK_SYSTEM
from agentic.graph import GAP_NOTE, MAX_ROUNDS, build_agentic_graph

Q = "what is X"


def chunk(cid, pdf="alpha.pdf"):
    return {"chunk_id": cid, "pdf": pdf, "text": f"text-{cid}", "headings": ""}


class StubRetriever:
    """Always returns the same two chunks: round 2+ adds nothing new."""
    def __init__(self):
        self.calls = []

    def search(self, q, top_k=5, pdfs=None, rerank=True):
        self.calls.append((q, top_k, tuple(pdfs) if pdfs else None, rerank))
        return [chunk("a-1"), chunk("b-1", "beta.pdf")]


class FreshRetriever(StubRetriever):
    """Every search returns a never-seen chunk: early stop must not fire."""
    def search(self, q, top_k=5, pdfs=None, rerank=True):
        self.calls.append((q, top_k, tuple(pdfs) if pdfs else None, rerank))
        return [chunk("a-1"), chunk(f"new-{len(self.calls)}")]


class EmptyRetriever(StubRetriever):
    def search(self, q, top_k=5, pdfs=None, rerank=True):
        self.calls.append((q, top_k, tuple(pdfs) if pdfs else None, rerank))
        return []


class StubLLM:
    def __init__(self):
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        class R: content = "ANSWER"
        return R()


class CountingCheck:
    """Wraps a verdict function; counts how often the check node ran it."""
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def __call__(self, state):
        self.calls += 1
        return self.fn(state)


def run_agentic(retriever, llm, question, **kw):
    graph = build_agentic_graph(retriever, llm, **kw)
    state = {"question": question, "llm_calls": 0, "retrieval_calls": 0,
             "chunks": [], "rounds": 0, "pending_queries": [], "queries_run": [],
             "gaps": [], "new_chunks": 0}
    if kw.get("trace"):
        state["trace"] = []
    return graph.invoke(state)


# 1. early stop: round 2 adds 0 new chunks → retrieve routes straight to
#    synthesize; check ran once; round-1 gaps still drive GAP_NOTE
check = CountingCheck(lambda s: {"sufficient": False, "pending_queries": ["r1"],
                                 "missing": ["part A"]})
ret, llm = StubRetriever(), StubLLM()
ag = run_agentic(ret, llm, Q, check=check, trace=True)
assert ag["rounds"] == 2 and check.calls == 1, (ag["rounds"], check.calls)
assert ag["gaps"] == ["part A"]
synth_user = llm.messages[-1][-1].content
assert synth_user.startswith(GAP_NOTE.format(gaps="- part A")), synth_user[:100]
skip = [e for e in ag["trace"] if e.get("skipped")]
assert len(skip) == 1 and skip[0] == {"node": "check", "skipped": "no_new_chunks",
                                      "rounds": 2}, skip
print("early stop: no-new-chunks round skips check, gaps drive GAP_NOTE: OK")

# 2. no early stop when new chunks arrive: check runs every round to the cap
check = CountingCheck(lambda s: {"sufficient": False,
                                 "pending_queries": [f"r{s['rounds']}"],
                                 "missing": [f"m{s['rounds']}"]})
ag = run_agentic(FreshRetriever(), StubLLM(), Q, check=check)
assert ag["rounds"] == MAX_ROUNDS and check.calls == MAX_ROUNDS, \
    (ag["rounds"], check.calls)
print("no stop with fresh chunks: check runs every round, MAX_ROUNDS backstop: OK")

# 3. round 1 never skips the check, even with zero chunks retrieved
check = CountingCheck(lambda s: {"sufficient": True})
ag = run_agentic(EmptyRetriever(), StubLLM(), Q, check=check)
assert ag["rounds"] == 1 and check.calls == 1
print("round 1 always checks (only unanswerable detector), even on 0 chunks: OK")

# 4. stalled stop: identical `missing` two rounds running (normalized compare)
#    forces pending_queries=[] despite fresh queries and fresh chunks
def stalled(s):
    return {"sufficient": False, "pending_queries": [f"q{s['rounds']}"],
            "missing": ["  Part A "] if s["rounds"] > 1 else ["part a"]}
check = CountingCheck(stalled)
ag = run_agentic(FreshRetriever(), StubLLM(), Q, check=check, trace=True)
assert ag["rounds"] == 2 and check.calls == 2, (ag["rounds"], check.calls)
stall_ev = [e for e in ag["trace"] if e.get("stalled")]
assert len(stall_ev) == 1 and stall_ev[0]["node"] == "check" \
    and stall_ev[0]["rounds"] == 2, stall_ev
print("stalled stop: repeated `missing` verdict ends the loop after round 2: OK")

# 5. differing `missing` keeps looping to the cap
check = CountingCheck(lambda s: {"sufficient": False,
                                 "pending_queries": [f"q{s['rounds']}"],
                                 "missing": [f"m{s['rounds']}"]})
ag = run_agentic(FreshRetriever(), StubLLM(), Q, check=check)
assert ag["rounds"] == MAX_ROUNDS and check.calls == MAX_ROUNDS
print("no stall on changing `missing`: loop runs to MAX_ROUNDS: OK")

# 6. round-1 immunity: insufficient with empty missing == empty init gaps must
#    NOT count as stalled (loop continues on pending queries)
check = CountingCheck(lambda s: {"sufficient": False,
                                 "pending_queries": [f"q{s['rounds']}"],
                                 "missing": []})
ag = run_agentic(FreshRetriever(), StubLLM(), Q, check=check)
assert ag["rounds"] == MAX_ROUNDS, ag["rounds"]
print("empty missing never counts as stalled: OK")

# 7. parity path unchanged: default LLM check (unparseable → fail-safe
#    sufficient), 1 round, llm_calls 3, node order intact
ret, llm = StubRetriever(), StubLLM()
ag = run_agentic(ret, llm, Q, trace=True)
assert ag["rounds"] == 1 and ag["llm_calls"] == 3 and ag["retrieval_calls"] == 1
assert [e["node"] for e in ag["trace"]] == ["plan", "retrieve", "check", "synthesize"]
print("parity: fallback path still 1 round / 3 llm calls: OK")

# 8. factual calibration rule present in CHECK_SYSTEM (wording lever, T1 §3)
assert "single specific fact" in CHECK_SYSTEM, "factual round-1 rule missing"
print("CHECK_SYSTEM carries the single-fact sufficiency rule: OK")
