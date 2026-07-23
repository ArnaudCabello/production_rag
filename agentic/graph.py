"""Agentic RAG pipeline: plan → retrieve ⇄ check → synthesize (see PRD.md).

M2: the planner classifies the question and emits sub-queries (agentic/planner.py);
on unparseable planner output it falls back to [question], matching the M1
pass-through. M3: retrieval loops — round 1 runs the planner sub-queries, the
check node may enqueue refined queries for further rounds (hard cap MAX_ROUNDS,
cross-round dedup of queries and chunks). M4: the check is an LLM sufficiency
judgment (agentic/checker.py) that enqueues refinement queries and records
uncovered gaps; remaining gaps make synthesize prepend a refuse/hedge
instruction. The check= hook still injects stubs in tests.
M5: synthesize caps the multi-round chunk union (MAX_SYNTH_CHUNKS) and validates
the answer's [n] citations deterministically (agentic/citations.py). T2: chunks
carry round/q_idx provenance; the cap selects by query-interleave
(select_synth_chunks) and aggregation questions get MAX_SYNTH_CHUNKS_AGG. No multi-doc
fan-out or vision routing: the planner/loop modules replace the former, and
vision is out of scope for the agentic pipeline.
"""
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agentic.checker import make_check
from agentic.citations import extract_citations
from agentic.planner import MAX_SUB_QUERIES, make_plan
from generation.pipeline import SYSTEM_PROMPT, USER_TEMPLATE, format_context

MAX_ROUNDS = 4              # hard cap on retrieval rounds (PRD §4)
AGG_SUBQUERY_TOP_K = 8      # broad, un-reranked recall for aggregation sub-queries
MAX_PENDING_PER_ROUND = 3   # refinement queries per round (the check fills pending_queries)
MAX_SYNTH_CHUNKS = 20       # cap on the multi-round union shown to the generator (M5);
                            # T2: query-interleave selection (select_synth_chunks)
MAX_SYNTH_CHUNKS_AGG = 30   # T2: aggregation questions overflow 20 in round 1 alone

# Prepended to the synthesis user message when the check left uncovered gaps.
# "not available in the corpus" wording matches eval/score_benchmark.py's REFUSAL regex.
GAP_NOTE = """Note: retrieval found no evidence in the corpus for the following point(s):
{gaps}
Answer the question from the context below using the evidence that IS there — do not refuse \
just because the listed point(s) are uncovered; briefly note what is not covered instead. \
Never guess or use outside knowledge for the missing points. Only if the context contains \
no relevant evidence for the question at all, reply that this information is not available \
in the corpus.

"""


# T4: appended to the synthesis system message on every question (planner labels
# are too unreliable to gate on). Targets cross_document/multi_chunk misses where
# the evidence is in context but the answer hedges or drops the asked relation.
SYNTH_GUIDE = """

Additionally:
- Answer exactly what the question asks. If it asks how things compare or relate, \
state the comparison/relationship explicitly and cover EACH side with its own \
citation — never reply with a generic similarity.
- Address every part of the question, and give the sources' specific values \
(numbers, sizes, temperatures, compositions) rather than qualitative summaries — \
but only values the sources state for the asked subject; never substitute a \
value reported for a different material, composition, or condition."""


def _norm(q):
    return q.strip().lower()


def select_synth_chunks(chunks: list[dict], cap: int) -> list[dict]:
    """T2 cap selection: round-robin across q_idx query groups so every
    sub-query and late-round targeted query keeps representation; within-group
    relevance order preserved. Identity when the union fits the cap. Chunks
    without q_idx share one group (degrades to first-N)."""
    if len(chunks) <= cap:
        return chunks
    groups: dict = {}
    order = []
    for c in chunks:
        k = c.get("q_idx")
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(c)
    picked, i = [], 0
    while len(picked) < cap:
        for k in order:
            if i < len(groups[k]):
                picked.append(groups[k][i])
                if len(picked) == cap:
                    break
        i += 1
    return picked


class AgentState(TypedDict):
    question: str
    category: str  # M2 planner label: simple|comparative|multi_hop|aggregation|unanswerable_maybe
    sub_queries: list[str]  # M2 planner output; question always first
    chunks: list[dict]  # union across retrievals, deduped by chunk_id
    answer: str
    llm_calls: int
    retrieval_calls: int
    rounds: int  # retrieval rounds completed
    pending_queries: list[str]  # enqueued for the next round; M4's check populates
    queries_run: list[str]  # normalized queries already searched (cross-round dedup)
    gaps: list[str]  # uncovered parts per the last check verdict; drives refusal/hedging
    new_chunks: int  # chunks added by the latest retrieve round (T1 early stop)
    citations: dict  # M5: {"markers", "valid", "invalid", "chunk_ids"} parsed from the answer
    trace: list[dict]  # per-node events when trace=True; init to [] in the invoke input


def build_agentic_graph(retriever, llm, trace: bool = False, check=None):
    check_fn = check or (lambda state: make_check(llm, state, MAX_ROUNDS))
    def plan(state: AgentState):
        p = make_plan(llm, state["question"])
        queries = [state["question"]]  # question always first — sub-queries only add recall
        for q in p["sub_queries"]:
            if q.strip().lower() != state["question"].strip().lower() and q not in queries:
                queries.append(q)
        update = {"sub_queries": queries[:1 + MAX_SUB_QUERIES],
                  "category": p["category"],
                  "llm_calls": state["llm_calls"] + 1}
        if trace:
            update["trace"] = state["trace"] + [
                {"node": "plan", "category": p["category"],
                 "sub_queries": update["sub_queries"],
                 "fallback": p.get("fallback", False), "raw": p["raw"][:500]}]
        return update

    def retrieve(state: AgentState):
        first = state["rounds"] == 0
        queries = (state["sub_queries"] if first
                   else state["pending_queries"][:MAX_PENDING_PER_ROUND])
        run = set(state["queries_run"])
        chunks = list(state["chunks"])
        seen = {c["chunk_id"] for c in chunks}
        calls, events, ran = state["retrieval_calls"], [], []
        for i, query in enumerate(queries):
            if _norm(query) in run:  # never re-search across rounds
                continue
            run.add(_norm(query))
            ran.append(_norm(query))  # queries_run holds normalized forms
            q_idx = calls  # global per-question query counter (T2 selection groups)
            calls += 1
            # aggregation needs recall beyond top-5: sub/refinement queries go
            # broad and skip the cross-encoder; the question keeps the full pass
            broad = state["category"] == "aggregation" and not (first and i == 0)
            hits = (retriever.search(query, top_k=AGG_SUBQUERY_TOP_K, rerank=False)
                    if broad else retriever.search(query))
            added = 0
            for chunk in hits:
                if chunk["chunk_id"] not in seen:
                    seen.add(chunk["chunk_id"])
                    # copy — never mutate retriever-returned dicts (T2 provenance)
                    chunks.append({**chunk, "round": state["rounds"] + 1,
                                   "q_idx": q_idx})
                    added += 1
            if trace:
                events.append({"node": "retrieve", "query": query,
                               "chunk_ids": [c["chunk_id"] for c in hits],
                               "round": state["rounds"] + 1, "broad": broad,
                               "new_chunks": added})
        update = {"chunks": chunks, "retrieval_calls": calls,
                  "rounds": state["rounds"] + 1, "pending_queries": [],
                  "queries_run": state["queries_run"] + ran,
                  "new_chunks": len(chunks) - len(state["chunks"])}
        if trace:
            if update["rounds"] > 1 and update["new_chunks"] == 0:
                events.append({"node": "check", "skipped": "no_new_chunks",
                               "rounds": update["rounds"]})
            update["trace"] = state["trace"] + events
        return update

    def route_after_retrieve(state: AgentState):
        # T1 early stop: a refinement round that added nothing new has nothing
        # to judge — skip the check LLM call. Round 1 always checks (the check
        # is the only unanswerable detector). Gaps from the previous verdict
        # persist, so GAP_NOTE still fires on the skip path.
        if state["rounds"] > 1 and state["new_chunks"] == 0:
            return "synthesize"
        return "check"

    def check(state: AgentState):
        verdict = check_fn(state)  # LLM judgment by default; tests inject stubs
        # T1 stalled stop: an insufficient verdict whose `missing` repeats the
        # previous round's gaps verbatim means re-querying is not converging.
        stalled = (not verdict["sufficient"] and verdict.get("missing")
                   and sorted(_norm(m) for m in verdict["missing"])
                   == sorted(_norm(g) for g in state["gaps"]))
        update = {}
        if stalled:
            update["pending_queries"] = []
        elif "pending_queries" in verdict:
            update["pending_queries"] = verdict["pending_queries"]
        if "missing" in verdict:  # last verdict wins — a later round may close gaps
            update["gaps"] = verdict["missing"]
        if verdict.get("llm_calls"):
            update["llm_calls"] = state["llm_calls"] + verdict["llm_calls"]
        if trace:
            event = {"node": "check", "sufficient": verdict["sufficient"],
                     "rounds": state["rounds"]}
            if stalled:
                event["stalled"] = True
            for key in ("missing", "fallback"):
                if key in verdict:
                    event[key] = verdict[key]
            if verdict.get("pending_queries"):
                event["queries"] = verdict["pending_queries"]
            update["trace"] = state["trace"] + [event]
        return update

    def route_after_check(state: AgentState):
        if state["rounds"] >= MAX_ROUNDS:
            return "synthesize"  # hard cap — single choke point
        return "retrieve" if state["pending_queries"] else "synthesize"

    def synthesize(state: AgentState):
        cap = (MAX_SYNTH_CHUNKS_AGG if state["category"] == "aggregation"
               else MAX_SYNTH_CHUNKS)
        capped = select_synth_chunks(state["chunks"], cap)
        user = USER_TEMPLATE.format(
            context=format_context(capped), question=state["question"])
        if state["gaps"]:  # evidence-driven refusal/hedging (M4)
            user = GAP_NOTE.format(gaps="\n".join(f"- {g}" for g in state["gaps"])) + user
        messages = [
            SystemMessage(content=SYSTEM_PROMPT + SYNTH_GUIDE),  # T4
            HumanMessage(content=user),
        ]
        answer = llm.invoke(messages).content
        # M5: deterministic citation validation — [n] resolves to capped[n-1]
        cites = extract_citations(answer, len(capped))
        cites["chunk_ids"] = [capped[n - 1]["chunk_id"] for n in cites["valid"]]
        update = {"answer": answer,
                  "llm_calls": state["llm_calls"] + 1,
                  "chunks": capped,  # record exactly what the generator saw
                  "citations": cites}
        if trace:
            context_rounds: dict = {}
            for c in capped:
                r = c.get("round")
                context_rounds[r] = context_rounds.get(r, 0) + 1
            update["trace"] = state["trace"] + [
                {"node": "synthesize", "context_chunks": len(capped),
                 "dropped_chunks": len(state["chunks"]) - len(capped),
                 "context_rounds": context_rounds,
                 "citations_valid": len(cites["valid"]),
                 "citations_invalid": len(cites["invalid"])}]
        return update

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan)
    graph.add_node("retrieve", retrieve)
    graph.add_node("check", check)
    graph.add_node("synthesize", synthesize)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_conditional_edges("retrieve", route_after_retrieve)
    graph.add_conditional_edges("check", route_after_check)
    graph.add_edge("synthesize", END)
    return graph.compile()
