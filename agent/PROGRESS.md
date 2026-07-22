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
| M4 evidence check + refusal | done | agent/plans/M4_evidence_check.md | shape report + smoke passed after prompt recalibration |
| M5 synthesis | done | agent/plans/M5_synthesis.md | smoke: cite✓ 100%, invalid=0, ev_recall 19.4% held |
| M6 benchmark run + tuning | in-progress | agent/plans/M6_benchmark_tuning.md | Phases 0-3 done (first full run measured, M6_report.md); Phase 4 = tuning modules below (PRD_TUNING.md) |
| T0 diagnosis harness + validation slice | not started | — | see PRD_TUNING.md §4 |
| T1 round efficiency (latency) | not started | — | after T0 |
| T2 aggregation recall + synthesis | not started | — | after T0 |
| T3 refusal + ambiguous calibration | not started | — | after T0 |
| T4 synthesis conversion (cross_doc, multi_chunk) | not started | — | after T0 |
| T5 final full run + close-out | not started | — | last; ONE full 306-q run |

---

## 2026-07-22 M6 Phases 1-3 — done (full runs measured; tuning next)
- What was done: human ran full 306-q agentic + closed_book runs + judges on
  Colab; agent pulled, ran compare.py, wrote eval/results_v2/M6_report.md.
  Headline: key_match +6.2 (p=0.005 SIGNIFICANT); Tier B judge +5.7
  (26.7→32.4, target +10 NOT met); multi_hop +16.7 and table +17.1 are real
  wins; aggregation flat at 8% for both is the blocker. Latency 6.2× baseline
  (cap 5×). Unanswerable refusal 22/25 vs 23/25. Contamination floor low
  (Tier B 6.7%); only 5 red-flag ids (see report); citation validity 99.4%.
- Files touched: eval/results_v2/M6_report.md (new), agent/PROGRESS.md
  (result files pulled from human's Colab commit 5d110cf).
- Tests: n/a (analysis). compare.py paired on 305 (v2q251 absent from
  baseline run only).
- Next step for the following agent: Phase 4 tuning in priority order:
  (1) aggregation recall/synthesis, (2) latency (factual wasted round,
  early-stop on no-new-chunks rounds), (3) refusal/ambiguous calibration,
  (4) cross_document/multi_chunk synthesis (evidence found, judge not
  converting) — inspect with --trace --ids incl. dropped_chunks. Pick the
  fixed ~20-30 id validation slice BEFORE tuning starts.
- Gotchas discovered: closed-book answers 23/25 unanswerable questions —
  never use closed-book refusal behaviour as a floor for refusal metrics.

## 2026-07-21 M6 benchmark run + tuning — in-progress (Phase 0 done, Colab runs pending)
- What was done: plan written with the human (agent/plans/M6_benchmark_tuning.md,
  approved). Decisions: measure FIRST (full 306-q agentic run of the pipeline
  exactly as M5 left it, before any tuning), closed-book control over all 306,
  tuning validated on shape reports + a fixed ~20-30 id slice with ONE final
  full run for the headline compare. Phase 0 implemented test-first:
  `--pipeline closed_book` adapter in eval/run_benchmark.py (question-only
  prompt mirroring baseline SYSTEM_PROMPT minus sources/citation rules;
  chunks=[], llm_calls=1, retrieval_calls=0; no retriever import). Scorer needs
  no changes (handles chunks=[]: ev_recall 0, citation_validity None).
  eval/README.md closed-book note updated from "to build" to "to run".
- Files touched: eval/run_benchmark.py (CLOSED_BOOK_SYSTEM/USER,
  build_closed_book, PIPELINES), tests/test_closed_book.py (new),
  eval/README.md, agent/plans/M6_benchmark_tuning.md (new), agent/PROGRESS.md
- Tests: `python tests/test_closed_book.py` — passing (4 checks); full suite
  (test_synthesis, test_check, test_retrieval_loop, test_planner,
  test_agentic_parity, test_pipeline) — passing.
- Next step for the following agent: human runs on Colab (all resumable):
  (1) `python eval/run_benchmark.py --pipeline agentic --model Qwen/Qwen3-14B`
  (NO --trace; hours; copy JSONL to Drive eval_v2/results between sessions),
  (2) `python eval/score_benchmark.py eval/results_v2/bench_agentic.jsonl --judge`,
  (3) `python eval/run_benchmark.py --pipeline closed_book --model Qwen/Qwen3-14B`
  + score with --judge,
  (4) `python eval/compare.py eval/results_v2/bench_baseline_scored.json
  eval/results_v2/bench_agentic_scored.json`. Then Phase 3: agent writes
  eval/results_v2/M6_report.md against every PRD §3 criterion, then the
  Phase 4 tuning loop (factual sufficient=false wastes a round; unanswerable
  burns 4 rounds; also eyeball dropped_chunks in a --trace --ids run).
- Gotchas discovered: none new. Standing: never re-score baseline without
  `--judge-file eval/results_v2/bench_baseline_judge.jsonl`; don't overwrite
  bench_baseline*; Gemini key capped at 20 calls/day (spot checks only).

## 2026-07-21 M5 synthesis with citations — done (Colab 3-q smoke pending)
- What was done: plan (agent/plans/M5_synthesis.md, approved) then build.
  Deterministic, ZERO new LLM calls (budget already at the cap of 6):
  `agentic/citations.py` — `extract_citations(answer, n_chunks)` parses [n] /
  [2][3] / [1, 2] markers (regex), splits into valid (1..n_chunks) / invalid;
  synthesize node caps the multi-round union at MAX_SYNTH_CHUNKS=20 (first-N of
  retrieval order), writes back `chunks: capped` (record = what the LLM saw),
  validates citations against the CAPPED list and stores
  state["citations"]={markers,valid,invalid,chunk_ids}; trace event gains
  dropped_chunks/citations_valid/citations_invalid. Prompt unchanged when
  gaps==[] (baseline already demands [n] citations) — parity intact. Runner
  persists `gaps` + `citations` in the JSONL (trace-style, only when present).
  Scorer: `citation_validity` (valid/total markers, None when uncited) shared
  via agentic.citations; summary gains cite✓ column + "cited answers" line;
  works on old result files (computed from answer+chunks, no new keys needed).
- Files touched: agentic/citations.py (new), agentic/graph.py,
  eval/run_benchmark.py, eval/score_benchmark.py, tests/test_synthesis.py (new),
  agent/plans/M5_synthesis.md (new). NO existing-test changes needed: no node
  reads state["citations"], and the parity trace assert only indexes
  context_chunks.
- Tests: `python tests/test_synthesis.py` — passing (10 checks); full suite
  (test_check, test_retrieval_loop, test_planner, test_agentic_parity,
  test_pipeline) — passing. Scorer verified against
  eval/results_v2/bench_baseline.jsonl: cite✓ 98.6%, cited answers 80.1%
  (baseline_scored.json regenerated WITH --judge-file to keep judge columns —
  plain re-scoring drops them; judge✓ 49.2% confirmed intact).
- Colab 3-q smoke (human, accepted): cite✓ 100% (all markers valid, invalid=0),
  cited answers 2/3 (v2q002 is a scoped no-evidence statement — nothing to
  cite), ev_recall 19.4% and key_match 0% both identical to the M4 smoke on
  the same aggregation ids — no regression from the context cap. Not yet
  eyeballed: dropped_chunks per question (check the synthesize trace event on
  the next Colab session). Mean latency 54s on these aggregation questions.
- Next step for the following agent: plan M6 (full benchmark run + tuning)
  with the human. M6 carry-overs: factual 4/5
  sufficient=false wastes a round; unanswerable burns all 4 rounds
  (~3 queries/round); closed-book control run adapter still to build.
- Gotchas discovered: re-running score_benchmark.py without --judge/--judge-file
  overwrites <run>_scored.json WITHOUT judge columns — always pass
  --judge-file eval/results_v2/bench_baseline_judge.jsonl when re-scoring the
  baseline. citation_validity is marker-level (dupes counted); citations_total
  uses extract_citations(answer, 0) just for the marker count.

## 2026-07-21 M4 prompt recalibration — done (Colab re-run pending)
- What was done: first Colab shape report + 3-q smoke exposed over-strictness:
  sufficient=false on ~90% of questions incl. 4/5 factual, and ALL 3 smoke
  answers were refusals ("not available in the corpus") with 0% key match at
  19% evidence recall — the "every part directly supported" rule made the
  check a paranoid judge and the GAP_NOTE read as permission to refuse
  wholesale. Recalibrated CHECK_SYSTEM (sufficient = enough for a useful
  grounded answer; representative sample suffices for broad questions;
  insufficiency only when a CORE part has no relevant evidence) and GAP_NOTE
  (answer from the evidence that IS there, refuse only when nothing relevant
  at all). Parse fallback rate was 0 — JSON structure is solid.
- Files touched: agentic/checker.py (CHECK_SYSTEM), agentic/graph.py (GAP_NOTE)
- Tests: all local suites passing (wording-independent). Colab re-run DONE:
  shape report improved (semantic 3/5, table 3/5 sufficient=true; unanswerable
  held at 5/5 false) and the 3-q smoke now answers instead of refusing
  (v2q001/003 substantive, v2q002 a scoped no-evidence statement). Smoke
  key_match 0% == baseline on the same 3 aggregation questions, but agentic
  ev_recall 19.4% vs baseline 0.0% — accepted; M4 CLOSED.
- Next step for the following agent: plan M5 (synthesis with citations) with
  the human. Carry-over tuning targets for M6: factual still 4/5
  sufficient=false (wastes a round + latency on easy questions);
  unanswerable emits ~3 queries/round instead of settling on queries=[]
  (runs the full 4 rounds, 60-90s). If over-refusal ever reappears, the
  structural lever is gating GAP_NOTE on gaps covering the WHOLE question.
- Gotchas discovered: unanswerable questions emit ~3 queries/round (burning
  rounds, 60-90s latency) instead of settling on queries=[]; acceptable under
  the cap but a target for M5/M6 tuning.

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
