"""Query pipeline as a LangGraph graph: decompose → retrieve → generate | generate_vision.

The generator sees the top reranked chunks as numbered sources and must cite
them; multi-chunk and cross-document questions are answerable by construction.
A planner node first splits questions that span multiple documents into
per-document sub-queries; their top chunks are appended to the main retrieval
so a comparison across papers has sources from every paper. When retrieved
chunks carry figures (and config.VISION_ENABLED), generation routes to the
vision model, which sees the figure images alongside the sources.
"""
import re
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

import config

# The vision model (7B, 4-bit) is weaker on pure text than the main generator,
# so route to it only when the question itself asks about visual content AND
# the retrieved chunks actually carry figures.
VISUAL_QUESTION = re.compile(
    r"\b(fig(?:ure)?s?\.?|images?|photos?|photographs?|pictures?|micrographs?"
    r"|optical|maps?|diagrams?|plots?|graphs?)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You are a precise document question-answering assistant.
Answer using ONLY the numbered sources provided. Rules:
- Cite the sources you use inline, e.g. [1] or [2][3].
- If the sources do not contain the answer, say so plainly — never guess.
- Quote exact numbers and names from the sources; do not round or paraphrase figures."""

USER_TEMPLATE = """Sources:

{context}

Question: {question}

Answer (with [n] citations):"""

DECOMPOSE_PROMPT = """You are the retrieval planner for a document Q/A system.
The corpus contains these documents:
{documents}

Question: {question}

Decide from the SHAPE of the question alone — do not judge whether the documents can
answer it. If the question compares, contrasts, or combines findings across more than
one paper/study/document, reply with one short search query per compared item (2-4
lines, nothing else — no explanations). Build each query from the question's own terms
for that item. Otherwise reply with exactly NONE."""


class RAGState(TypedDict):
    question: str
    planner_reply: str
    sub_queries: list[str]
    chunks: list[dict]
    answer: str


def parse_sub_queries(reply: str) -> list[str]:
    lines = [re.sub(r"^[\s\-*\d.)]+", "", line).strip() for line in reply.splitlines()]
    lines = [line for line in lines if line]
    if not lines or lines[0].upper().startswith("NONE"):  # NONE + trailing prose is still NONE
        return []
    if len(lines) < 2:  # a comparison needs at least two sides; anything less is noise
        return []
    return lines[: config.MAX_SUBQUERIES]


def format_context(chunks: list[dict]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[{i}] {chunk.get('pdf', 'unknown')}"
        if chunk.get("headings"):
            header += f" — {chunk['headings']}"
        blocks.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(blocks)


def needs_vision(state: RAGState) -> str:
    if (
        config.VISION_ENABLED
        and VISUAL_QUESTION.search(state["question"])
        and any(c.get("figures") for c in state["chunks"])
    ):
        return "generate_vision"
    return "generate"


def build_graph(retriever, llm):
    # Document cards (file name + title + byline) tell the planner what each paper is
    # actually about; bare file names made it wrongly conclude comparisons were unanswerable.
    doc_names = sorted(
        c["text"][:220] for c in retriever.chunks.values() if c["chunk_id"].endswith("-card")
    ) or sorted({c["pdf"] for c in retriever.chunks.values()})

    def decompose(state: RAGState):
        if not config.DECOMPOSE_ENABLED:
            return {"planner_reply": "", "sub_queries": []}
        reply = llm.invoke([HumanMessage(content=DECOMPOSE_PROMPT.format(
            documents="\n".join(f"- {name}" for name in doc_names),
            question=state["question"]))]).content
        return {"planner_reply": reply, "sub_queries": parse_sub_queries(reply)}

    def retrieve(state: RAGState):
        chunks = retriever.search(state["question"])
        seen = {c["chunk_id"] for c in chunks}
        for sub_query in state["sub_queries"]:
            for chunk in retriever.search(sub_query, top_k=config.SUBQUERY_TOP_K):
                if chunk["chunk_id"] not in seen:
                    seen.add(chunk["chunk_id"])
                    chunks.append(chunk)
        return {"chunks": chunks}

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
    graph.add_node("decompose", decompose)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("generate_vision", generate_vision)
    graph.add_edge(START, "decompose")
    graph.add_edge("decompose", "retrieve")
    graph.add_conditional_edges("retrieve", needs_vision)
    graph.add_edge("generate", END)
    graph.add_edge("generate_vision", END)
    return graph.compile()
