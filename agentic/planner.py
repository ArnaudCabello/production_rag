"""M2 planner: classify the question and emit sub-queries (see agent/plans/M2_planner.md).

The planner never picks documents — it only rewrites queries — and any
parse/validation failure falls back to {"category": "simple",
"sub_queries": [question]}, i.e. the M1 pass-through behaviour.
"""
import json

from langchain_core.messages import HumanMessage, SystemMessage

CATEGORIES = {"simple", "comparative", "multi_hop", "aggregation", "unanswerable_maybe"}
MAX_SUB_QUERIES = 4

PLANNER_SYSTEM = """You are a retrieval planner for a question-answering system over a corpus \
of scientific papers on ultra-high-temperature ceramics (UHTCs).
Classify the question and write focused search queries. Respond with ONLY a JSON object:
{"category": "<simple|comparative|multi_hop|aggregation|unanswerable_maybe>", "sub_queries": ["...", ...]}
Rules:
- simple: one fact from one place -> 1 sub-query (a cleaned-up form of the question).
- comparative: distinct entities/papers/methods compared -> one sub-query per entity (2-4).
- multi_hop: the answer to one part is needed to ask the next -> sub-queries for the \
FIRST hop plus the overall question; later hops are handled downstream.
- aggregation: a value/range collected across many studies -> 2-4 sub-queries with \
varied phrasings and synonyms.
- unanswerable_maybe: likely outside the corpus -> still emit 1 sub-query to check.
- Queries are keyword-rich search strings, not questions to a person.
- At most 4 sub-queries. JSON only, no commentary."""

PLANNER_USER = """Question: How do the sintering temperatures compare between the hot-pressed \
ZrB2-SiC composites of Smith et al. and the spark-plasma-sintered HfB2 ceramics of Lee et al.?
{{"category": "comparative", "sub_queries": ["Smith ZrB2-SiC hot pressing sintering temperature", "Lee HfB2 spark plasma sintering temperature"]}}

Question: {question}"""


def parse_plan(text: str, question: str) -> dict:
    """Parse planner LLM output; on any failure return the deterministic fallback."""
    fallback = {"category": "simple", "sub_queries": [question], "fallback": True}
    try:
        obj = json.loads(text[text.index("{"):text.rindex("}") + 1])
        category = obj["category"]
        if not isinstance(obj["sub_queries"], list):
            return fallback
        sub_queries = [q.strip() for q in obj["sub_queries"]
                       if isinstance(q, str) and q.strip()]
        if category not in CATEGORIES or not sub_queries:
            return fallback
        return {"category": category, "sub_queries": sub_queries[:MAX_SUB_QUERIES]}
    except (ValueError, KeyError, TypeError):
        return fallback


def make_plan(llm, question: str) -> dict:
    raw = llm.invoke([SystemMessage(content=PLANNER_SYSTEM),
                      HumanMessage(content=PLANNER_USER.format(question=question))]).content
    plan = parse_plan(raw, question)
    plan["raw"] = raw
    return plan
