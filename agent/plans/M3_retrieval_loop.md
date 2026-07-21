# M3 — Retrieval loop (round cap + dedup + category-driven recall)

## Context

Third module (PRD §4/§6): turn the single-pass `retrieve` node into the
tool-use loop — hard 4-round cap, cross-round dedup, and the graph edges
M4's evidence check will drive. Deliverable per PRD §6: "tool-use loop with
round cap + dedup; tests for coverage/termination". The sufficiency judgment
("are all sub-questions covered? → loop") is explicitly M4.

Empirical input (Colab planner shape report, 45 golden questions): labels
`aggregation` and `comparative` are reliable; `multi_hop` and
`unanswerable_maybe` were NEVER predicted. Therefore M3 must not key
refinement on labels — refinement is driven by gap detection (M4). Only the
`aggregation` label drives a recall knob in M3.

Decisions confirmed with the user:
- **Structure only**: M3 installs `retrieve → check → (retrieve | synthesize)`
  conditional edges, `rounds` counter, `pending_queries` plumbing, cross-round
  query/chunk dedup — with a trivial always-sufficient `check`. M4 replaces
  the check with the LLM sufficiency judgment + refined-query generation
  (same judgment, likely same LLM call). M3 adds 0 LLM calls; fully testable
  with stubs. `check` is injectable: `build_agentic_graph(..., check=None)` —
  the exact seam M4 fills, not speculative flexibility.
- **Aggregation recall knob**: for `category == "aggregation"`, the original
  question keeps the full reranked `search(q)` (top_k=5); planner sub-queries
  use `rerank=False, top_k=8` (RRF order, skips the cross-encoder that
  dominates latency — mirrors baseline fan-out economics). All other
  categories unchanged. No `pdfs=` restriction anywhere (planner never picks
  documents).

Budget: llm_calls unchanged from M2 (plan 1 + synth 1 = 2 ≤ 6). Refinement
rounds (M4-driven) capped at `MAX_PENDING_PER_ROUND = 3` queries each; hard
cap `MAX_ROUNDS = 4` enforced at a single choke point (the conditional edge).

## Design

**`agentic/graph.py`** — all edits here; structure grows one node.

```python
MAX_ROUNDS = 4              # hard cap on retrieval rounds (PRD §4)
AGG_SUBQUERY_TOP_K = 8      # broad, un-reranked recall for aggregation sub-queries
MAX_PENDING_PER_ROUND = 3   # refinement queries per round (M4 fills pending_queries)

class AgentState(TypedDict):
    ...existing keys...
    rounds: int                 # retrieval rounds completed
    pending_queries: list[str]  # enqueued for the next round; M4 populates
    queries_run: list[str]      # normalized queries already searched (cross-round dedup)

def _norm(q): return q.strip().lower()

def build_agentic_graph(retriever, llm, trace=False, check=None):
    check_fn = check or (lambda state: {"sufficient": True})  # M4 replaces

    def retrieve(state):
        first = state["rounds"] == 0
        queries = state["sub_queries"] if first else state["pending_queries"][:MAX_PENDING_PER_ROUND]
        run = set(state["queries_run"])
        chunks = list(state["chunks"]); seen = {c["chunk_id"] for c in chunks}
        calls = state["retrieval_calls"]; ran = []
        for i, query in enumerate(queries):
            if _norm(query) in run:          # never re-search across rounds
                continue
            run.add(_norm(query)); ran.append(query); calls += 1
            broad = state["category"] == "aggregation" and not (first and i == 0)
            hits = (retriever.search(query, top_k=AGG_SUBQUERY_TOP_K, rerank=False)
                    if broad else retriever.search(query))
            # dedup-append into chunks/seen as today
            # trace event: {"node": "retrieve", "query", "chunk_ids",
            #               "round": state["rounds"] + 1, "broad": broad}
        return {"chunks": chunks, "retrieval_calls": calls,
                "rounds": state["rounds"] + 1, "pending_queries": [],
                "queries_run": state["queries_run"] + ran, ...trace}

    def check(state):
        verdict = check_fn(state)   # M3: {"sufficient": True}, no LLM
        update = {}                 # M4 may add llm_calls / pending_queries
        if "pending_queries" in verdict:
            update["pending_queries"] = verdict["pending_queries"]
        # trace event: {"node": "check", "sufficient": ..., "rounds": state["rounds"]}
        return update

    def route_after_check(state):
        if state["rounds"] >= MAX_ROUNDS:
            return "synthesize"     # hard cap — single choke point
        return "retrieve" if state["pending_queries"] else "synthesize"

    # edges: plan→retrieve→check; add_conditional_edges("check", route_after_check);
    # synthesize→END
```

Routing subtlety: "insufficient but no new queries" still terminates
(nothing to run) — matches PRD "if exhausted → note gaps" (gap noting is
M4/M5). `chunks`/`retrieval_calls` now accumulate across rounds, so retrieve
reads prior state; invoke inputs need `"chunks": [], "rounds": 0,
"pending_queries": [], "queries_run": []`.

**`eval/run_benchmark.py`** — `build_agentic`'s `answer()` invoke input gains
the new init keys.

**Tests** — `tests/test_retrieval_loop.py`, plain script, stubs per
tests/test_planner.py (StubRetriever records `(q, top_k, pdfs, rerank)`;
PlannerLLM returns planner JSON then "ANSWER"; injected `check=` callables
drive the loop).

## Steps (validation-first)

1. Save plan as `agent/plans/M3_retrieval_loop.md`; PROGRESS.md → M3
   in-progress.
2. Write `tests/test_retrieval_loop.py`:
   - default path (no check injection): simple question → 1 round, counters
     as M2 (`llm_calls==2`), check trace `sufficient: True`, straight to
     synthesize;
   - termination at cap: `check=lambda s: {"sufficient": False,
     "pending_queries": [f"r{s['rounds']}"]}` → exactly MAX_ROUNDS rounds,
     then synthesize, `rounds == 4`;
   - no re-retrieval: check re-enqueues an already-run query (case/space
     variant) → not searched again; loop terminates;
   - cross-round chunk dedup: overlapping chunk_ids across rounds → unique
     union, first-seen order;
   - exhausted termination: `sufficient: False` with empty pending → 1 round;
   - pending cap: 5 pending → only MAX_PENDING_PER_ROUND searched;
   - aggregation knob: category=aggregation → question call `(5, rerank=True)`
     via defaults, sub-query calls `(top_k=8, rerank=False)`; comparative →
     all defaults;
   - trace: retrieve events carry `round`/`broad`; check events carry
     `sufficient`/`rounds`.
   Verify: fails for the right reason (no `check=` kwarg / no loop).
3. Implement `agentic/graph.py` (AgentState keys, constants, retrieve
   rewrite, check node, conditional edge). Verify: loop tests green.
4. Update `eval/run_benchmark.py` invoke input. Verify: `--help` OK.
5. Regressions: `tests/test_planner.py` and `tests/test_agentic_parity.py`
   (their `run_agentic` helpers may need the new init keys — allowed edit;
   assertions unchanged), `tests/test_pipeline.py`.
6. PROGRESS.md (M3 done; note check is the M4 seam; no Colab run needed —
   no LLM behaviour change), commit + push to `agentic_pipeline`.

## Files
- new: `tests/test_retrieval_loop.py`, `agent/plans/M3_retrieval_loop.md`
- edit: `agentic/graph.py`, `eval/run_benchmark.py` (invoke input),
  `tests/test_planner.py` + `tests/test_agentic_parity.py` (init keys only),
  `agent/PROGRESS.md`
- read-only reuse: `retrieval/retriever.py` (unchanged per PRD),
  `agentic/planner.py`, `config.py`

## Verification
```
python tests/test_retrieval_loop.py     # M3 validation set — no GPU
python tests/test_planner.py            # M2 regression
python tests/test_agentic_parity.py     # M1 parity regression
python tests/test_pipeline.py           # baseline regression
python eval/run_benchmark.py --help     # CLI intact
# No Colab work for M3 — no LLM behaviour change; M4 owns judgment validation.
```
