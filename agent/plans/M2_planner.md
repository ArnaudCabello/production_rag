# M2 — Planner (question classification + sub-query emission)

## Context

Second module of the agentic pipeline (PRD.md §4/§6). Replace the pass-through
`plan` node in `agentic/graph.py` with an LLM planner that classifies the
question and emits 1–N sub-queries for M3's retrieval loop and M4's evidence
check. Graph structure (START→plan→retrieve→synthesize→END) and existing
`AgentState` keys stay; only node internals plus one new state key change.

Hard constraints:
- Budget: pipeline ≤ 6 LLM calls; planner spends exactly 1 (`llm_calls += 1`).
- Prior-art warning (`generation/pipeline.py` MULTI_DOC_QUESTION comment: "An
  LLM planner proved unreliable at both this judgment and at picking which
  documents matter"). Mitigations: (a) planner never picks documents — only
  rewrites queries; (b) strict structural validation of the JSON output with a
  deterministic fallback `{"category": "simple", "sub_queries": [question]}`
  on any parse/validation failure — worst case degrades to M1 behaviour;
  (c) the original question is ALWAYS the first retrieval query, sub-queries
  only add recall.
- Golden set v2 is FROZEN — shape-only spot checks, never answers.

Decisions confirmed with the user:
- Label set = PRD's five: `simple | comparative | multi_hop | aggregation |
  unanswerable_maybe` (retrieval-strategy labels, not eval categories).
  Stored as `state["category"]`; M3 branches on it (aggregation → broader
  recall, multi_hop → chained re-query), M4 uses `unanswerable_maybe`.
- Spot-check fixture = stratified golden-set slice (5 ids per category,
  ids+question+category only), shape report on Colab — allowed, not tuning.
- Parity test redefined: parity = fallback behaviour (same retrieval + same
  final prompt as baseline, `llm_calls == 2`). No planner bypass flag.

## Design

**`agentic/planner.py`** (new, ~80 lines):

```python
CATEGORIES = {"simple", "comparative", "multi_hop", "aggregation", "unanswerable_maybe"}
MAX_SUB_QUERIES = 4

PLANNER_SYSTEM = """You are a retrieval planner for a scientific-paper QA system...
Respond with ONLY a JSON object:
{"category": "<simple|comparative|multi_hop|aggregation|unanswerable_maybe>",
 "sub_queries": ["...", ...]}
Rules:
- simple: one fact from one place -> 1 sub-query (cleaned-up question).
- comparative: distinct entities/papers compared -> one sub-query per entity (2-4).
- multi_hop: sub-queries for the FIRST hop plus the overall question;
  later hops handled downstream (M3).
- aggregation: 2-4 sub-queries with varied phrasings/synonyms.
- unanswerable_maybe: likely outside the UHTC corpus -> still 1 sub-query.
- Queries are keyword-rich search strings. At most 4. JSON only."""
# + one-shot example (comparative) in the user message

def parse_plan(text: str, question: str) -> dict:
    # slice first '{'..last '}', json.loads; validate category in CATEGORIES,
    # sub_queries = list of 1..N non-empty strs (strip, drop empties,
    # truncate to MAX_SUB_QUERIES). Any failure ->
    # {"category": "simple", "sub_queries": [question], "fallback": True}

def make_plan(llm, question: str) -> dict:
    raw = llm.invoke([SystemMessage(PLANNER_SYSTEM),
                      HumanMessage(PLANNER_USER.format(question=question))]).content
    plan = parse_plan(raw, question)
    plan["raw"] = raw
    return plan
```

**`agentic/graph.py`** — edits inside `plan` only; add `category: str` to
`AgentState`:

```python
def plan(state):
    p = make_plan(llm, state["question"])
    queries = [state["question"]]          # question always first
    for q in p["sub_queries"]:             # dedup vs question (case-insensitive)
        if q.strip().lower() != state["question"].strip().lower() and q not in queries:
            queries.append(q)
    update = {"sub_queries": queries[: 1 + MAX_SUB_QUERIES],
              "category": p["category"],
              "llm_calls": state["llm_calls"] + 1}
    if trace:
        update["trace"] = state["trace"] + [{"node": "plan",
            "category": p["category"], "sub_queries": update["sub_queries"],
            "fallback": p.get("fallback", False), "raw": p["raw"][:500]}]
    return update
```

`retrieve` (loops sub_queries, dedups by chunk_id) and `synthesize` unchanged.
Budget: 1 plan + 1 synthesize = 2 LLM calls ≤ 6; ≤ 5 retrieval searches.

**Parity impact:** the M1 parity test's StubLLM returns "ANSWER" → planner
falls back to `[question]`; behaviour parity holds but `llm_calls == 2` and
the baseline-prompt assertion must target the LAST llm call. Minimal edit,
documented in PROGRESS.md (redefinition, not weakening).

**`eval/planner_shapes.py`** (new, ~50 lines, Colab GPU): loads
`agent/plans/fixtures/planner_slice.json`, runs `make_plan(get_llm(...), q)`
per question, prints per-golden-category: predicted-label counts, sub-query
count mean/min/max, fallback %. Soft expectations printed (cross_document /
multi_hop / aggregation → ≥2 sub-queries; factual → 1) — a report, not an
assert gate. `--model` flag like run_benchmark.

## Steps (validation-first)

1. Save this plan as `agent/plans/M2_planner.md`; set M2 in-progress in
   `agent/PROGRESS.md`.
2. Build fixture `agent/plans/fixtures/planner_slice.json` — first 5 ids per
   golden category (≈45 entries; id/question/category only, deterministic
   selection, commit the output). Verify: file has 45 entries, no answer keys.
3. Write `tests/test_planner.py` (plain script, assert+print, local StubLLM
   variants mirroring test_agentic_parity.py):
   - valid JSON → parsed category + sub_queries; question kept first;
   - JSON in prose/code fences → parsed;
   - garbage / truncated / wrong types / unknown category / empty or >4
     sub_queries → fallback or cap, `fallback` flagged;
   - graph integration: one `retriever.search` per query, dedup, `llm_calls
     == 2`, `category` in final state;
   - trace: plan event has category/sub_queries/fallback/raw.
   Verify: fails with ImportError (no agentic/planner.py yet).
4. Implement `agentic/planner.py`. Verify: parse-level tests pass.
5. Edit `agentic/graph.py` plan node + AgentState. Verify:
   `python tests/test_planner.py` fully green.
6. Update `tests/test_agentic_parity.py` (planner-aware parity). Verify: passes.
7. Write `eval/planner_shapes.py`. Local verify: imports + `--help`; real run
   deferred to Colab (record in PROGRESS.md as the M2 GPU follow-up; human
   reviews the shape report before M3 relies on labels).
8. Regressions: `python tests/test_pipeline.py`,
   `python eval/run_benchmark.py --help`.
9. Update `agent/PROGRESS.md` (done-pending-Colab-shapes), commit + push to
   `agentic_pipeline`.

## Files
- new: `agentic/planner.py`, `tests/test_planner.py`, `eval/planner_shapes.py`,
  `agent/plans/fixtures/planner_slice.json`, `agent/plans/M2_planner.md`
- edit: `agentic/graph.py`, `tests/test_agentic_parity.py`, `agent/PROGRESS.md`
- read-only reuse: `generation/llm.py`, `generation/pipeline.py`,
  `eval/golden_set_v2.json`

## Verification
```
python tests/test_planner.py            # M2 validation set — no GPU
python tests/test_agentic_parity.py     # updated parity
python tests/test_pipeline.py           # baseline regression
python eval/run_benchmark.py --help     # CLI intact
# Colab only:
python eval/planner_shapes.py --model Qwen/Qwen3-14B   # shape report → human review
```
