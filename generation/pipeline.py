"""Query pipeline as a LangGraph graph: retrieve → generate.

The generator sees the top reranked chunks as numbered sources and must cite
them; multi-chunk and cross-document questions are answerable by construction.
"""
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

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

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
