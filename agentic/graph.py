"""Agentic RAG pipeline skeleton: plan → retrieve → synthesize (see PRD.md).

M1: every node is a trivial pass-through so behaviour matches the baseline
linear pipeline on plain questions — the planner emits the question as the
single sub-query, retrieval runs once, synthesis uses the baseline prompts.
M2-M5 replace node internals without changing this structure. No multi-doc
fan-out or vision routing: the planner/loop modules replace the former, and
vision is out of scope for the agentic pipeline.
"""
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from generation.pipeline import SYSTEM_PROMPT, USER_TEMPLATE, format_context


class AgentState(TypedDict):
    question: str
    sub_queries: list[str]  # M2 planner output; M1: just [question]
    chunks: list[dict]  # union across retrievals, deduped by chunk_id
    answer: str
    llm_calls: int
    retrieval_calls: int
    trace: list[dict]  # per-node events when trace=True; init to [] in the invoke input


def build_agentic_graph(retriever, llm, trace: bool = False):
    def plan(state: AgentState):
        update = {"sub_queries": [state["question"]]}
        if trace:
            update["trace"] = state["trace"] + [
                {"node": "plan", "sub_queries": update["sub_queries"]}]
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
