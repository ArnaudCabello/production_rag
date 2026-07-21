"""M4 evidence check: judge sufficiency, emit refinement queries and gap notes
(see agent/plans/M4_evidence_check.md).

Refusal is evidence-driven, never label-driven (the planner's unanswerable_maybe
never fires — Colab shape report). Any parse/validation failure fails SAFE to
{"sufficient": True}: degrade to the M3 always-sufficient behaviour rather than
burn retrieval rounds on garbage output.
"""
import json

from langchain_core.messages import HumanMessage, SystemMessage

MAX_CHUNK_CHARS = 300   # per-chunk text budget in the check prompt
MAX_CHECK_QUERIES = 3   # == graph.MAX_PENDING_PER_ROUND (no import: graph imports us)

FALLBACK_VERDICT = {"sufficient": True, "missing": [], "queries": [], "fallback": True}

CHECK_SYSTEM = """You are an evidence auditor for a question-answering system over a corpus \
of scientific papers on ultra-high-temperature ceramics (UHTCs).
Given a question, the search queries already run, and the text snippets retrieved so far, \
decide whether the snippets contain enough evidence to answer every part of the question. \
Respond with ONLY a JSON object:
{"sufficient": <true|false>, "missing": ["<part of the question the snippets do not cover>", ...], "queries": ["<new search query>", ...]}
Rules:
- sufficient=true if the snippets contain enough evidence for a useful, grounded answer — \
perfection is not required, and broad questions (ranges, trends, comparisons across studies) \
are sufficiently covered by a representative sample. Then missing and queries must be [].
- Most questions ARE answerable from the snippets; declare insufficiency only when a core \
part of the question has NO relevant evidence in any snippet.
- If a core part is uncovered AND a differently-worded search could plausibly find it, emit \
1-3 new queries (keyword-rich search strings, not questions; do not repeat queries already run).
- If the snippets for a part came back off-topic, the corpus likely lacks it: set \
sufficient=false, list it in missing, and set queries=[] — do not keep inventing queries.
- JSON only, no commentary."""

CHECK_USER = """Question: {question}

Queries already run:
{queries_run}

Retrieved snippets ({n_chunks}):
{evidence}

Rounds used: {rounds} of {max_rounds}."""


def parse_check(text: str) -> dict:
    """Parse check LLM output; on any failure return the fail-safe fallback."""
    try:
        obj = json.loads(text[text.index("{"):text.rindex("}") + 1])
        if not isinstance(obj["sufficient"], bool):
            return dict(FALLBACK_VERDICT)
        lists = {}
        for key, cap in (("missing", None), ("queries", MAX_CHECK_QUERIES)):
            val = obj[key]
            if not isinstance(val, list) or any(not isinstance(s, str) for s in val):
                return dict(FALLBACK_VERDICT)
            items = [s.strip() for s in val if s.strip()]
            lists[key] = items[:cap] if cap else items
        if obj["sufficient"]:
            lists["queries"] = []
        return {"sufficient": obj["sufficient"], **lists}
    except (ValueError, KeyError, TypeError):
        return dict(FALLBACK_VERDICT)


def make_check(llm, state, max_rounds: int) -> dict:
    """One LLM call judging evidence sufficiency; returns a check-node verdict."""
    evidence = "\n".join(
        f"[{c['chunk_id']}] {c['text'][:MAX_CHUNK_CHARS]}" for c in state["chunks"]
    ) or "(nothing retrieved)"
    raw = llm.invoke([
        SystemMessage(content=CHECK_SYSTEM),
        HumanMessage(content=CHECK_USER.format(
            question=state["question"],
            queries_run="\n".join(f"- {q}" for q in state["queries_run"]),
            n_chunks=len(state["chunks"]), evidence=evidence,
            rounds=state["rounds"], max_rounds=max_rounds)),
    ]).content
    verdict = parse_check(raw)
    verdict["pending_queries"] = verdict.pop("queries")
    verdict["llm_calls"] = 1
    verdict["raw"] = raw
    return verdict
