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
| M3 retrieval loop | not started | — | |
| M4 evidence check + refusal | not started | — | |
| M5 synthesis | not started | — | |
| M6 benchmark run + tuning | not started | — | baseline run in progress on Colab (results land in Drive eval_v2/results) |

---

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
