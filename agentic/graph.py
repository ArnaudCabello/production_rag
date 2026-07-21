"""Agentic RAG pipeline skeleton: plan → retrieve → synthesize (see PRD.md).

M2: the planner classifies the question and emits sub-queries (agentic/planner.py);
on unparseable planner output it falls back to [question], matching the M1
pass-through. Retrieval runs once per query; synthesis uses baseline prompts.
M3-M5 replace node internals without changing this structure. No multi-doc
fan-out or vision routing: the planner/loop modules replace the former, and
vision is out of scope for the agentic pipeline.
"""
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agentic.planner import MAX_SUB_QUERIES, make_plan
from generation.pipeline import SYSTEM_PROMPT, USER_TEMPLATE, format_context


class AgentState(TypedDict):
    question: str
    category: str  # M2 planner label: simple|comparative|multi_hop|aggregation|unanswerable_maybe
    sub_queries: list[str]  # M2 planner output; question always first
    chunks: list[dict]  # union across retrievals, deduped by chunk_id
    answer: str
    llm_calls: int
    retrieval_calls: int
    trace: list[dict]  # per-node events when trace=True; init to [] in the invoke input


def build_agentic_graph(retriever, llm, trace: bool = False):
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
        chunks, seen, events = [], set(), []
        calls = state["retrieval_calls"]
        for query in state["sub_queries"]:
            calls += 1
            hits = retriever.search(query)
            if trace:
                events.append({"node": "retrieve", "query": query,
                               "chunk_ids": [c["chunk_id"] for c in hits]})
            for chunk in hits:
                if chunk["chunk_id"] not in seen:
                    seen.add(chunk["chunk_id"])
                    chunks.append(chunk)
        update = {"chunks": chunks, "retrieval_calls": calls}
        if trace:
            update["trace"] = state["trace"] + events
        return update

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
    graph.add_node("synthesize", synthesize)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()
