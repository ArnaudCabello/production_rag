"""Agentic RAG pipeline: plan → retrieve ⇄ check → synthesize (see PRD.md).

M2: the planner classifies the question and emits sub-queries (agentic/planner.py);
on unparseable planner output it falls back to [question], matching the M1
pass-through. M3: retrieval loops — round 1 runs the planner sub-queries, the
check node may enqueue refined queries for further rounds (hard cap MAX_ROUNDS,
cross-round dedup of queries and chunks). The M3 check is an always-sufficient
stub; M4 supplies the real sufficiency judgment via the check= hook.
M4-M5 replace node internals without changing this structure. No multi-doc
fan-out or vision routing: the planner/loop modules replace the former, and
vision is out of scope for the agentic pipeline.
"""
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agentic.planner import MAX_SUB_QUERIES, make_plan
from generation.pipeline import SYSTEM_PROMPT, USER_TEMPLATE, format_context

MAX_ROUNDS = 4              # hard cap on retrieval rounds (PRD §4)
AGG_SUBQUERY_TOP_K = 8      # broad, un-reranked recall for aggregation sub-queries
MAX_PENDING_PER_ROUND = 3   # refinement queries per round (M4 fills pending_queries)


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
    trace: list[dict]  # per-node events when trace=True; init to [] in the invoke input


def build_agentic_graph(retriever, llm, trace: bool = False, check=None):
    check_fn = check or (lambda state: {"sufficient": True})  # M4 replaces
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
        verdict = check_fn(state)  # M3: always sufficient, no LLM; M4 judges
        update = {}  # M4 may add llm_calls / pending_queries
        if "pending_queries" in verdict:
            update["pending_queries"] = verdict["pending_queries"]
        if trace:
            update["trace"] = state["trace"] + [
                {"node": "check", "sufficient": verdict["sufficient"],
                 "rounds": state["rounds"]}]
        return update

    def route_after_check(state: AgentState):
        if state["rounds"] >= MAX_ROUNDS:
            return "synthesize"  # hard cap — single choke point
        return "retrieve" if state["pending_queries"] else "synthesize"

    def synthesize(state: AgentState):
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=USER_TEMPLATE.format(
                context=format_context(state["chunks"]), question=state["question"])),
        ]
        update = {"answer": llm.invoke(messages).content,
                  "llm_calls": state["llm_calls"] + 1}
        if trace:
            update["trace"] = state["trace"] + [
                {"node": "synthesize", "context_chunks": len(state["chunks"])}]
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
