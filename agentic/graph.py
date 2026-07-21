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


def build_agentic_graph(retriever, llm):
    def plan(state: AgentState):
        return {"sub_queries": [state["question"]]}

    def retrieve(state: AgentState):
        chunks, seen = [], set()
        calls = state["retrieval_calls"]
        for query in state["sub_queries"]:
            calls += 1
            for chunk in retriever.search(query):
                if chunk["chunk_id"] not in seen:
                    seen.add(chunk["chunk_id"])
                    chunks.append(chunk)
        return {"chunks": chunks, "retrieval_calls": calls}

    def synthesize(state: AgentState):
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=USER_TEMPLATE.format(
                context=format_context(state["chunks"]), question=state["question"])),
        ]
        return {"answer": llm.invoke(messages).content,
                "llm_calls": state["llm_calls"] + 1}

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan)
    graph.add_node("retrieve", retrieve)
    graph.add_node("synthesize", synthesize)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()
