# Agentic pipeline — build progress

Every agent MUST append an entry here when it completes (or abandons) a task,
so a fresh agent can pick up exactly where the last one left off. Newest entry
FIRST. Keep entries short and factual — state, not narrative.

Entry template:
```
## [date] MX <module> — <status: planned | in-progress | done | blocked>
- What was done:
- Files touched:
- Tests: <command to run them> — <passing?>
- Next step for the following agent:
- Gotchas discovered:
```

Module status board (update the table too):

| Module | Status | Plan | Notes |
|--------|--------|------|-------|
| M1 skeleton + adapter | done | agent/plans/M1_skeleton.md | parity test green |
| M2 planner | done | agent/plans/M2_planner.md | Colab shape report pending (eval/planner_shapes.py) |
| M3 retrieval loop | done | agent/plans/M3_retrieval_loop.md | check node = M4 seam |
| M4 evidence check + refusal | done | agent/plans/M4_evidence_check.md | Colab shape report pending (eval/check_shapes.py); GPU smoke pending |
| M5 synthesis | not started | — | |
| M6 benchmark run + tuning | not started | — | baseline run in progress on Colab (results land in Drive eval_v2/results) |

---

## 2026-07-21 M4 evidence check + refusal — done (Colab shape report + GPU smoke pending)
- What was done: LLM evidence check replaces the always-sufficient stub —
  `agentic/checker.py` (CHECK_SYSTEM/CHECK_USER prompts, `parse_check` with
  strict validation + fail-safe fallback {"sufficient": True}, `make_check`
  building a compact evidence view of chunk_id + text[:300]). Check is now the
  DEFAULT check_fn in build_agentic_graph (check= still injects test stubs);
  check node threads llm_calls and gaps (verdict "missing", last verdict wins)
  through state; synthesize prepends GAP_NOTE (refuse/hedge instruction whose
  "not available in the corpus" wording trips score_benchmark's REFUSAL regex)
  when gaps remain, byte-identical baseline prompt otherwise. run_benchmark
  agentic adapter inits + records "gaps". eval/check_shapes.py (mirrors
  planner_shapes.py, reuses planner_slice.json fixture) for the Colab shape
  report. Budget worst case: 1 plan + 4 checks + 1 synth = 6 = PRD cap.
- Files touched: agentic/checker.py (new), agentic/graph.py,
  eval/run_benchmark.py, eval/check_shapes.py (new), tests/test_check.py (new),
  tests/test_retrieval_loop.py + tests/test_planner.py +
  tests/test_agentic_parity.py (gaps init key; intentional counter
  redefinition 2→3: every question now includes one check call),
  agent/plans/M4_evidence_check.md
- Tests: `python tests/test_check.py` — passing (12 checks); also passing:
  test_retrieval_loop.py, test_planner.py, test_agentic_parity.py,
  test_pipeline.py. NOT run (no GPU here): 3-question benchmark smoke
  (`python eval/run_benchmark.py --pipeline agentic --limit 3`) and
  `python eval/check_shapes.py --model Qwen/Qwen3-14B` — human runs both on
  Colab before M5 relies on gap notes.
- Next step for the following agent: human reviews check_shapes report
  (watch: unanswerable → sufficient=false + missing, factual → sufficient=true,
  and the fallback rate — a high fallback rate silently degrades to M3); then
  plan M5 (synthesis with citations) with the human. M5 gets gap notes via
  state["gaps"].
- Gotchas discovered: LangGraph nodes reading state["gaps"] need it in every
  invoke init dict (M3 gotcha holds). Trace check events now carry
  missing/fallback (and queries when non-empty) — exact-dict trace asserts in
  older tests had to add those keys. checker.py can't import
  MAX_PENDING_PER_ROUND from graph.py (graph imports checker) — local
  MAX_CHECK_QUERIES=3 mirrors it.
- What was done: plan written with the human (agent/plans/M4_evidence_check.md).
  Decisions confirmed: check runs on EVERY question (labels unreliable except
  aggregation/comparative; check is the only unanswerable detector); minimal
  gap injection — check writes state["gaps"], synthesize prepends a
  refuse/hedge instruction when gaps remain (full cited synthesis stays M5);
  parse failure ⇒ fail-safe sufficient=true. Budget worst case 6 LLM calls
  (plan 1 + 4 checks + synth 1) = PRD cap.
- Files touched: agent/plans/M4_evidence_check.md (new), agent/PROGRESS.md
- Tests: none yet — /build M4 writes tests/test_check.py FIRST.
- Next step for the following agent: `/build M4` per the plan. Note: parity
  test counters intentionally become 3/1 (check adds 1 call).
- Gotchas discovered: graph.py's check node currently drops verdict keys
  other than pending_queries — llm_calls/gaps threading must be added there.

## 2026-07-21 M3 retrieval loop — done
- What was done: retrieve is now a loop — graph is plan → retrieve ⇄ check →
  synthesize. Round 1 runs the planner sub-queries; the check node may enqueue
  `pending_queries` (cap MAX_PENDING_PER_ROUND=3/round) for further rounds;
  hard cap MAX_ROUNDS=4 enforced in the conditional edge; cross-round dedup of
  queries (normalized, `queries_run`) and chunks (chunk_id union). M3's check
  is an always-sufficient stub injected via `build_agentic_graph(..., check=)`
  — the exact seam M4's LLM sufficiency judgment fills. Aggregation recall
  knob (confirmed with human): question keeps full reranked search, agg
  sub/refinement queries use top_k=8, rerank=False (per Colab shape report:
  only aggregation/comparative labels are reliable; multi_hop and
  unanswerable_maybe never fire, so refinement must come from M4 gap
  detection, not labels). 0 new LLM calls.
- Files touched: agentic/graph.py, eval/run_benchmark.py (invoke init keys),
  tests/test_retrieval_loop.py (new), tests/test_planner.py +
  tests/test_agentic_parity.py (init keys + check node in trace),
  agent/plans/M3_retrieval_loop.md (new)
- Tests: `python tests/test_retrieval_loop.py` — passing;
  test_planner.py / test_agentic_parity.py / test_pipeline.py — passing.
  No Colab run needed (no LLM behaviour change).
- Next step for the following agent: plan M4 (evidence check + refusal) with
  the human — replace the stub check with an LLM sufficiency judgment that
  emits `{"sufficient", "pending_queries"}` (and gap notes for M5); refusal
  driven by evidence, NOT the unanswerable_maybe label. Budget: plan 1 +
  synth 1 leaves ≤4 LLM calls for check rounds.
- Gotchas discovered: `queries_run` must store NORMALIZED queries (raw
  strings broke re-enqueue dedup). LangGraph raises KeyError if invoke input
  lacks any state key a node reads — all invoke sites need the full init
  dict. Trace node order now plan/retrieve/check/synthesize; retrieve events
  carry round/broad.

## 2026-07-21 M2 planner — done (Colab shape report pending)
- What was done: LLM planner replaces the pass-through plan node —
  `agentic/planner.py` (PLANNER_SYSTEM/USER prompts, `parse_plan` with strict
  validation + deterministic fallback to {"category":"simple",
  "sub_queries":[question]}, `make_plan`); plan node in graph.py builds
  queries = [question] + sub_queries (dedup, cap 1+4), sets
  state["category"], llm_calls += 1, trace event with
  category/sub_queries/fallback/raw. Labels are the PRD's five
  (simple|comparative|multi_hop|aggregation|unanswerable_maybe), confirmed
  with the human. Parity test redefined (confirmed): parity = fallback
  behaviour, llm_calls == 2, baseline prompt is the LAST llm call.
  Fixture `agent/plans/fixtures/planner_slice.json` (5 ids/category, no
  answers) + `eval/planner_shapes.py` shape report (Colab, not run here —
  no GPU locally).
- Files touched: agentic/planner.py (new), agentic/graph.py,
  tests/test_planner.py (new), tests/test_agentic_parity.py,
  eval/planner_shapes.py (new), agent/plans/fixtures/planner_slice.json (new),
  agent/plans/M2_planner.md (new)
- Tests: `python tests/test_planner.py` — passing;
  `python tests/test_agentic_parity.py` — passing;
  `python tests/test_pipeline.py` — passing. GPU smoke + shape report NOT
  run (no GPU here).
- Next step for the following agent: human runs
  `python eval/planner_shapes.py --model Qwen/Qwen3-14B` on Colab and
  reviews label/sub-query shapes BEFORE M3 branches on state["category"];
  then plan M3 (retrieval loop) with the human.
- Gotchas discovered: a string value for sub_queries iterates as characters —
  parse_plan requires isinstance(list). PLANNER_USER contains literal JSON
  braces, escaped as {{ }} for .format. StubLLM "ANSWER" is deliberately
  unparseable JSON, which is what makes the parity test exercise the
  fallback path.

## 2026-07-21 tracing + groundedness docs — done
- What was done: toggleable agent trace — `build_agentic_graph(..., trace=True)`
  makes each node append events to state["trace"] (plan: sub_queries;
  retrieve: query + returned chunk_ids per call; synthesize: context size);
  `run_benchmark.py --trace` writes it into the JSONL. Off by default — full
  306-q runs unaffected. Documented groundedness verification protocol
  (PRD success criteria + eval/README.md): counters, evidence_recall red
  flags, closed-book control run (to build in M6), M5 citation validity.
- Files touched: agentic/graph.py, eval/run_benchmark.py,
  tests/test_agentic_parity.py (trace test), eval/README.md, PRD.md
- Tests: `python tests/test_agentic_parity.py` — passing;
  `python tests/test_pipeline.py` — passing.
- Next step for the following agent: unchanged — plan M2 (planner). M2-M5
  nodes should append their own trace events (loop rounds, evidence-check
  verdicts) following the same pattern.
- Gotchas discovered: human's Gemini API key (.env) is capped at 20
  calls/day — spot checks only, never a full judging pass. `--top-k 0` does
  nothing (0 is falsy) — the closed-book control needs its own adapter.

## 2026-07-21 M1 skeleton + adapter — done
- What was done: `agentic/` package with the target plan → retrieve →
  synthesize LangGraph (all nodes trivial pass-throughs, == baseline on plain
  questions; no fanout/vision by design — see plan). `build_agentic()` wired in
  eval/run_benchmark.py; parity test written first, then code.
- Files touched: agentic/__init__.py, agentic/graph.py (new),
  tests/test_agentic_parity.py (new), eval/run_benchmark.py (build_agentic
  body), agent/plans/M1_skeleton.md (new)
- Tests: `python tests/test_agentic_parity.py` — passing;
  `python tests/test_pipeline.py` regression — passing.
- Next step for the following agent: plan M2 (planner) with the human into
  agent/plans/M2_planner.md — question classification + sub-query emission,
  validation set of question → expected-plan-shape pairs first.
- Gotchas discovered: counters live in AgentState (init to 0 in the invoke
  input); tests/test_pipeline.py executes on import, so stubs must be
  redefined locally, not imported. Generator on Colab is Qwen3-14B via
  --model; config.GENERATOR_MODEL default is stale (Qwen2.5) — leave it.

## 2026-07-21 project scaffolding — done
- What was done: branch `agentic_pipeline` created from main; PRD.md,
  agent/PROGRESS.md, agent/plans/, /onboard and /build commands added.
  Benchmark harness (golden_set_v2 306q, run/score/compare) already on main.
- Files touched: PRD.md, agent/PROGRESS.md, agent/plans/README.md,
  .claude/commands/onboard.md, .claude/commands/build.md, CLAUDE.md
- Tests: n/a (docs/scaffolding)
- Next step for the following agent: /onboard, then plan M1 with the human
  (plan mode) into agent/plans/M1_skeleton.md.
- Gotchas discovered: baseline benchmark run (Qwen3-14B) is running on the
  human's Colab; its results are the comparison target — do not touch
  eval/results/ (legacy) or overwrite eval/results_v2/bench_baseline*.
