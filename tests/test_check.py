"""M4 evidence-check tests: parse/fallback, loop integration, gap-driven refusal (stubs, no models)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

from agentic.checker import FALLBACK_VERDICT, parse_check
from agentic.graph import MAX_PENDING_PER_ROUND, MAX_ROUNDS, build_agentic_graph
from generation.pipeline import USER_TEMPLATE, format_context

Q = "what is X"


def chunk(cid, pdf="alpha.pdf"):
    return {"chunk_id": cid, "pdf": pdf, "text": f"text-{cid}", "headings": ""}


class StubRetriever:
    def __init__(self):
        self.calls = []

    def search(self, q, top_k=5, pdfs=None, rerank=True):
        self.calls.append((q, top_k, tuple(pdfs) if pdfs else None, rerank))
        return [chunk("a-1"), chunk("b-1", "beta.pdf")]


class FreshRetriever(StubRetriever):
    """T1: refinement rounds adding no new chunks skip the check, so multi-round
    check tests need every search to yield a fresh chunk."""
    def search(self, q, top_k=5, pdfs=None, rerank=True):
        self.calls.append((q, top_k, tuple(pdfs) if pdfs else None, rerank))
        return [chunk("a-1"), chunk(f"new-{len(self.calls)}")]


class ScriptedLLM:
    """Returns scripted responses in invoke order (plan, check, check, ..., synthesis)."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        content = self.responses[len(self.messages) - 1]
        class R: pass
        R.content = content
        return R()


def run_agentic(retriever, llm, question, **kw):
    graph = build_agentic_graph(retriever, llm, **kw)
    state = {"question": question, "llm_calls": 0, "retrieval_calls": 0,
             "chunks": [], "rounds": 0, "pending_queries": [], "queries_run": [],
             "gaps": [], "new_chunks": 0}
    if kw.get("trace"):
        state["trace"] = []
    return graph.invoke(state)


def verdict_json(sufficient, missing=(), queries=()):
    return json.dumps({"sufficient": sufficient, "missing": list(missing),
                       "queries": list(queries)})


# --- parse_check ---

# 1. valid JSON
v = parse_check(verdict_json(False, ["melting point of HfB2"], ["HfB2 melting point"]))
assert v["sufficient"] is False
assert v["missing"] == ["melting point of HfB2"]
assert v["queries"] == ["HfB2 melting point"]
assert not v.get("fallback")
print("parse: valid JSON: OK")

# 2. JSON in prose/fences
wrapped = 'Verdict:\n```json\n{"sufficient": true, "missing": [], "queries": []}\n```'
v = parse_check(wrapped)
assert v["sufficient"] is True and v["missing"] == [] and v["queries"] == []
print("parse: JSON in prose/fences: OK")

# 3. failures → fail-safe fallback (sufficient=True: degrade to M3, never burn rounds)
for bad in [
    "ANSWER",                                              # garbage
    '{"sufficient": false, "missing": [',                  # truncated
    '{"missing": [], "queries": []}',                      # missing 'sufficient'
    '{"sufficient": "yes", "missing": [], "queries": []}', # non-bool sufficient
    '{"sufficient": false, "missing": "x", "queries": []}',# non-list missing
    '{"sufficient": false, "missing": [], "queries": "q"}',# non-list queries
    '{"sufficient": false, "missing": [1], "queries": []}',# non-str items
]:
    v = parse_check(bad)
    assert v == FALLBACK_VERDICT, (bad, v)
assert FALLBACK_VERDICT["sufficient"] is True and FALLBACK_VERDICT["fallback"] is True
print("parse: fail-safe fallback on garbage/invalid: OK")

# 4. sufficient=true forces queries empty; strings stripped/empties dropped; cap at 3
v = parse_check(verdict_json(True, [], ["stale query"]))
assert v["queries"] == []
v = parse_check(verdict_json(False, [" m1 ", ""], [" q1 ", "", "q2", "q3", "q4"]))
assert v["missing"] == ["m1"]
assert v["queries"] == ["q1", "q2", "q3"] and len(v["queries"]) == MAX_PENDING_PER_ROUND
print("parse: sufficient forces no queries; strip/drop-empty/cap: OK")


# --- graph integration (default check = LLM check) ---

# 5. insufficient round 1 → refinement round runs the check's queries; llm_calls
#    counts plan + each check + synthesis
ret = FreshRetriever()
llm = ScriptedLLM(["PLAN-GARBAGE",                       # plan → fallback [Q]
                   verdict_json(False, ["m1"], ["q2"]),  # check round 1
                   verdict_json(True),                   # check round 2
                   "ANSWER"])                            # synthesis
ag = run_agentic(ret, llm, Q)
assert [c[0] for c in ret.calls] == [Q, "q2"], ret.calls
assert ag["rounds"] == 2
assert ag["llm_calls"] == 4 and ag["retrieval_calls"] == 2
assert ag["answer"] == "ANSWER"
assert ag["gaps"] == []  # last verdict wins: sufficient cleared the gaps
print("loop: insufficient verdict drives refinement round, counters 4/2: OK")

# 6. sufficient round 1 → single round, llm_calls == 3 (plan + check + synthesis)
ret = StubRetriever()
ag = run_agentic(ret, ScriptedLLM(["PLAN-GARBAGE", verdict_json(True), "ANSWER"]), Q)
assert ag["rounds"] == 1 and ag["llm_calls"] == 3 and ag["retrieval_calls"] == 1
print("loop: sufficient round 1, counters 3/1: OK")

# 7. unparseable check output → fail-safe: one round, synthesis proceeds, trace flags fallback
ret = StubRetriever()
ag = run_agentic(ret, ScriptedLLM(["PLAN-GARBAGE", "CHECK-GARBAGE", "ANSWER"]), Q,
                 trace=True)
assert ag["rounds"] == 1 and ag["answer"] == "ANSWER" and ag["llm_calls"] == 3
check_ev = [e for e in ag["trace"] if e["node"] == "check"][0]
assert check_ev["sufficient"] is True and check_ev["fallback"] is True
print("loop: unparseable check fails safe to synthesize: OK")

# 8. budget: every check insufficient with fresh queries → MAX_ROUNDS, llm_calls ≤ 6
#    (T1: missing must vary per round or the stalled stop ends the loop early)
ret = FreshRetriever()
responses = ["PLAN-GARBAGE"] + [verdict_json(False, [f"m{i}"], [f"r{i}"])
                                for i in range(MAX_ROUNDS)] + ["ANSWER"]
ag = run_agentic(ret, ScriptedLLM(responses), Q)
assert ag["rounds"] == MAX_ROUNDS
assert ag["llm_calls"] == 1 + MAX_ROUNDS + 1 and ag["llm_calls"] <= 6
print("budget: worst case 1 + 4 checks + 1 = 6 LLM calls: OK")

# 9. gaps flow: insufficient with missing but NO queries → loop stops, synthesis
#    prompt carries the gap note + refusal instruction
ret = StubRetriever()
llm = ScriptedLLM(["PLAN-GARBAGE",
                   verdict_json(False, ["boiling point of X"]),
                   "ANSWER"])
ag = run_agentic(ret, llm, Q)
assert ag["rounds"] == 1 and ag["gaps"] == ["boiling point of X"]
synth_user = llm.messages[-1][1].content
assert "boiling point of X" in synth_user
assert "not available in the corpus" in synth_user  # refusal wording (scorer REFUSAL regex)
baseline_user = USER_TEMPLATE.format(context=format_context(ag["chunks"]), question=Q)
assert synth_user.endswith(baseline_user)  # baseline prompt intact, note prepended
print("gaps: uncovered points reach the synthesis prompt with refusal instruction: OK")

# 10. no gaps → synthesis prompt byte-identical to baseline (parity preserved)
ret = StubRetriever()
llm = ScriptedLLM(["PLAN-GARBAGE", verdict_json(True), "ANSWER"])
ag = run_agentic(ret, llm, Q)
assert llm.messages[-1][1].content == USER_TEMPLATE.format(
    context=format_context(ag["chunks"]), question=Q)
print("gaps: empty gaps leave the baseline synthesis prompt unchanged: OK")

# 11. check prompt sees the question and the retrieved evidence
llm = ScriptedLLM(["PLAN-GARBAGE", verdict_json(True), "ANSWER"])
run_agentic(StubRetriever(), llm, Q)
check_user = llm.messages[1][1].content
assert Q in check_user and "text-a-1" in check_user and "a-1" in check_user
print("check prompt: question + chunk ids/text present: OK")

# 12. injected check= still overrides the LLM default (M3 seam intact) — no check LLM call
ret = StubRetriever()
llm = ScriptedLLM(["PLAN-GARBAGE", "ANSWER"])
ag = run_agentic(ret, llm, Q, check=lambda s: {"sufficient": True})
assert ag["llm_calls"] == 2
print("seam: injected check bypasses LLM check: OK")
