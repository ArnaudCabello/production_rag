"""Query pipeline as a LangGraph graph: retrieve → generate | generate_vision.

The generator sees the top reranked chunks as numbered sources and must cite
them; multi-chunk and cross-document questions are answerable by construction.
For questions worded across multiple documents ("these three studies", "the
papers"), retrieval fans out: every document contributes its own best chunks,
so a comparison always has sources from every paper. When retrieved chunks
carry figures (and config.VISION_ENABLED), generation routes to the vision
model, which sees the figure images alongside the sources.
"""
import re
from typing import NotRequired, TypedDict

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

# Cross-document questions announce themselves in their wording. An LLM planner
# proved unreliable at both this judgment and at picking which documents matter,
# so the fan-out is deterministic: match the wording, then query EVERY document.
MULTI_DOC_QUESTION = re.compile(
    r"\b(papers|studies|documents|articles|reports"
    r"|these (?:two|three|four|five) \w+|both (?:papers|studies|documents))\b",
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


class RAGState(TypedDict):
    question: str
    scope: NotRequired[list[str] | None]  # restrict to these document names
    chunks: list[dict]
    answer: str
    figures: NotRequired[list[str]]  # figure image paths the vision model was shown, A/B/C order


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


def build_graph(retriever, llm, provider: str = None):
    provider = provider or config.GENERATOR_PROVIDER  # captured at build time, with the llm it describes
    doc_names = sorted({c["pdf"] for c in retriever.chunks.values()})

    def retrieve(state: RAGState):
        scope = state.get("scope") or None
        chunks = retriever.search(state["question"], pdfs=scope)
        if config.MULTI_DOC_FANOUT and MULTI_DOC_QUESTION.search(state["question"]):
            docs = scope or doc_names
            if len(docs) > config.MULTI_DOC_MAX_DOCS:
                # corpus too large to query every document: fall back to the documents
                # the main retrieval surfaced (at most RERANK_TOP_N of them) — a real
                # document-selection stage is the known follow-up for large corpora
                docs = list(dict.fromkeys(c["pdf"] for c in chunks))[: config.MULTI_DOC_MAX_DOCS]
            seen = {c["chunk_id"] for c in chunks}
            for pdf in docs:
                for chunk in retriever.search(state["question"], top_k=config.PER_DOC_TOP_K, pdfs=[pdf]):
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
        from generation.vision import collect_figures

        figures = collect_figures(state["chunks"])  # same list, same order as the prompt's A/B/C labels
        if provider == "huggingface":
            from generation.vision import answer_with_figures  # lazy: loads the local VLM on first use

            answer = answer_with_figures(state["question"], state["chunks"], format_context)
        else:
            from generation.vision import answer_with_figures_api  # API providers are multimodal already

            answer = answer_with_figures_api(llm, state["question"], state["chunks"], format_context)
        return {"answer": answer, "figures": figures}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("generate_vision", generate_vision)
    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges("retrieve", needs_vision)
    graph.add_edge("generate", END)
    graph.add_edge("generate_vision", END)
    return graph.compile()
