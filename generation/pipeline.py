"""Query pipeline as a LangGraph graph: retrieve → generate | generate_vision.

The generator sees the top reranked chunks as numbered sources and must cite
them; multi-chunk and cross-document questions are answerable by construction.
When retrieved chunks carry figures (and config.VISION_ENABLED), generation
routes to the vision model, which sees the figure images alongside the sources.
"""
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

import config

SYSTEM_PROMPT = """You are a precise document question-answering assistant.
Answer using ONLY the numbered sources provided. Rules:
- Cite the sources you use inline, e.g. [1] or [2][3].
- If the sources do not contain the answer, say so plainly — never guess.
- Quote exact numbers and names from the sources; do not round or paraphrase figures."""

USER_TEMPLATE = """Sources:

{context}

Question: {question}

Answer (with [n] citations):"""


class RAGState(TypedDict):
    question: str
    chunks: list[dict]
    answer: str


def format_context(chunks: list[dict]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[{i}] {chunk.get('pdf', 'unknown')}"
        if chunk.get("headings"):
            header += f" — {chunk['headings']}"
        blocks.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(blocks)


def needs_vision(state: RAGState) -> str:
    if config.VISION_ENABLED and any(c.get("figures") for c in state["chunks"]):
        return "generate_vision"
    return "generate"


def build_graph(retriever, llm):
    def retrieve(state: RAGState):
        return {"chunks": retriever.search(state["question"])}

    def generate(state: RAGState):
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=USER_TEMPLATE.format(
                context=format_context(state["chunks"]), question=state["question"])),
        ]
        return {"answer": llm.invoke(messages).content}

    def generate_vision(state: RAGState):
        from generation.vision import answer_with_figures  # lazy: loads the VLM on first use

        return {"answer": answer_with_figures(state["question"], state["chunks"], format_context)}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("generate_vision", generate_vision)
    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges("retrieve", needs_vision)
    graph.add_edge("generate", END)
    graph.add_edge("generate_vision", END)
    return graph.compile()
