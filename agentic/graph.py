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
the answer's [n] citations deterministically (agentic/citations.py). No multi-doc
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
                            # first-N of retrieval order — question's reranked hits come first

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


def _norm(q):
    return q.strip().lower()


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
            calls += 1
            # aggregation needs recall beyond top-5: sub/refinement queries go
            # broad and skip the cross-encoder; the question keeps the full pass
            broad = state["category"] == "aggregation" and not (first and i == 0)
            hits = (retriever.search(query, top_k=AGG_SUBQUERY_TOP_K, rerank=False)
                    if broad else retriever.search(query))
            if trace:
                events.append({"node": "retrieve", "query": query,
                               "chunk_ids": [c["chunk_id"] for c in hits],
                               "round": state["rounds"] + 1, "broad": broad})
            for chunk in hits:
                if chunk["chunk_id"] not in seen:
                    seen.add(chunk["chunk_id"])
                    chunks.append(chunk)
        update = {"chunks": chunks, "retrieval_calls": calls,
                  "rounds": state["rounds"] + 1, "pending_queries": [],
                  "queries_run": state["queries_run"] + ran}
        if trace:
            update["trace"] = state["trace"] + events
        return update

    def check(state: AgentState):
        verdict = check_fn(state)  # LLM judgment by default; tests inject stubs
        update = {}
        if "pending_queries" in verdict:
            update["pending_queries"] = verdict["pending_queries"]
        if "missing" in verdict:  # last verdict wins — a later round may close gaps
            update["gaps"] = verdict["missing"]
        if verdict.get("llm_calls"):
            update["llm_calls"] = state["llm_calls"] + verdict["llm_calls"]
        if trace:
            event = {"node": "check", "sufficient": verdict["sufficient"],
                     "rounds": state["rounds"]}
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
        capped = state["chunks"][:MAX_SYNTH_CHUNKS]
        user = USER_TEMPLATE.format(
            context=format_context(capped), question=state["question"])
        if state["gaps"]:  # evidence-driven refusal/hedging (M4)
            user = GAP_NOTE.format(gaps="\n".join(f"- {g}" for g in state["gaps"])) + user
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
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
            update["trace"] = state["trace"] + [
                {"node": "synthesize", "context_chunks": len(capped),
                 "dropped_chunks": len(state["chunks"]) - len(capped),
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
    graph.add_edge("retrieve", "check")
    graph.add_conditional_edges("check", route_after_check)
    graph.add_edge("synthesize", END)
    return graph.compile()
