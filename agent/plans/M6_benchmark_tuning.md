# M6 — Full benchmark run + tuning

## Context

M1–M5 are done: the agentic pipeline (plan → retrieve loop → check → synthesize
with citations) passes all local suites and Colab smokes, but has never run on
the full v2 benchmark (306 q). M6 produces the headline result the PRD demands:
agentic vs baseline via `eval/compare.py` (Tier B ≥ +10 pts judge-correct,
p < 0.05; Tier A regression ≤ 3 pts; refusal ≥ baseline; ≤ 6 LLM calls and
≤ 5× baseline latency median), plus the closed-book contamination control.
Decisions confirmed with the human: **measure first** (full run of the pipeline
exactly as M5 left it, before any tuning), **closed-book over all 306**, and
**tuning validated on slices, one final full run** for the headline compare.

On approval, this plan is saved to `agent/plans/M6_benchmark_tuning.md` and
PROGRESS.md gets a "planned" entry.

## Phase 0 — Code: closed-book adapter (local, before any Colab run)

The only new code. In `eval/run_benchmark.py`:

- Add `build_closed_book(model, provider, top_k, trace)` to the `PIPELINES`
  dict (L91): reuses `get_llm` like `build_baseline` (L41), but prompts with
  the question ONLY — no retriever import, no context chunks. Returns
  `{"answer", "chunks": [], "llm_calls": 1, "retrieval_calls": 0}`.
  Keep the answer-style instruction parallel to the baseline prompt (minus the
  context/citation parts) so the judge compares like with like.
- No scorer changes: `score_benchmark.py` already handles `chunks=[]`
  (evidence_recall 0, citation_validity None) and `--judge` works unchanged.
- Test first (`tests/test_closed_book.py`, StubLLM pattern from
  test_agentic_parity.py): 1 llm call, 0 retrieval calls, empty chunks, no
  question text lost, JSONL fields written. Known gotcha this fixes:
  `--top-k 0` is falsy and cannot serve as closed-book (README L34-42).

Verify: new test + full local suite green.

## Phase 1 — Full agentic run + judge + compare (Colab, human-driven)

Commands for the human (all resumable — run_benchmark skips done ids, judge
checkpoints to `*_judge.jsonl`):

1. `python eval/run_benchmark.py --pipeline agentic --model Qwen/Qwen3-14B`
   → `eval/results_v2/bench_agentic.jsonl`. NO `--trace` (README: off for full
   runs). Expect hours (smoke was ~54 s/q on aggregation; factual should be
   faster). Copy to Drive `eval_v2/results` as sessions die.
2. `python eval/score_benchmark.py eval/results_v2/bench_agentic.jsonl --judge`
   (local Qwen2.5-14B judge) → `bench_agentic_judge.jsonl` + `_scored.json`.
3. `python eval/compare.py eval/results_v2/bench_baseline_scored.json
   eval/results_v2/bench_agentic_scored.json` → per-category deltas + paired
   bootstrap.

Gotchas to respect: never re-score baseline without
`--judge-file eval/results_v2/bench_baseline_judge.jsonl`; don't overwrite
`bench_baseline*`.

## Phase 2 — Closed-book control (Colab)

1. `python eval/run_benchmark.py --pipeline closed_book --model Qwen/Qwen3-14B`
   → `bench_closed_book.jsonl` (fast: 1 call/q, no retrieval).
2. Score with `--judge` (same judge). Its judge-correct rate per category is
   the contamination floor: agentic gains are only credible above it.
   Flag any question judged correct with evidence_recall 0 in the agentic run
   AND correct closed-book — contamination red flags per PRD §3(b,c).

## Phase 3 — Analysis against success criteria

Agent writes `eval/results_v2/M6_report.md` (numbers, not vibes):

- Tier B (cross_document, multi_hop, aggregation): judge-correct delta ≥ +10,
  p < 0.05? Tier A (factual, semantic, table): regression ≤ 3?
- Unanswerable refusal rate ≥ baseline; false-answer rate ≤ baseline.
- Cost: llm_calls ≤ 6 all questions; median latency ≤ 5× baseline.
- Groundedness: retrieval_calls ≥ 1 on answered; ev_recall↔correctness;
  closed-book floor; citation validity.
- Per-category diagnosis feeding Phase 4 (where are the losses?).

## Phase 4 — Tuning loop (slices → one final full run)

Targets from the carry-overs + whatever Phase 3 surfaces:

- **Factual wastes a round**: 4/5 factual get sufficient=false. Lever:
  `CHECK_SYSTEM` (agentic/checker.py L24-33). Validate with
  `eval/check_shapes.py` on Colab.
- **Unanswerable burns 4 rounds** (~3 queries/round, 60-90 s): lever is the
  checker rule "off-topic → sufficient=false, queries=[]" (checker.py L32-33)
  and/or an early-stop in `route_after_check` (graph.py L137) when a round
  returns no new chunks. Keep refusal evidence-driven (no label gating).
- Other knobs if Phase 3 points at them: `MAX_SYNTH_CHUNKS=20`,
  `AGG_SUBQUERY_TOP_K=8`, `MAX_PENDING_PER_ROUND=3` (graph.py L26-29).

Discipline per iteration: local tests green → shape report
(`check_shapes.py` / `planner_shapes.py`) → fixed validation slice via
`--ids` (~20-30 q spanning categories, chosen ONCE before tuning starts and
distinct from the ids eyeballed during diagnosis; never tune against golden-set
answers, only shapes/costs) → only when a change survives the slice, ONE final
full 306 run → re-judge → final compare. Also eyeball `dropped_chunks` in a
`--trace --ids` run (M5 leftover).

## Phase 5 — Close-out

- Final `compare.py` report + M6_report.md conclusions vs every PRD §3
  criterion, stated pass/fail.
- PROGRESS.md entry + status board update; commit + push `agentic_pipeline`.

## Files touched

- `eval/run_benchmark.py` (closed_book adapter), `tests/test_closed_book.py`
  (new), `agent/plans/M6_benchmark_tuning.md` (this plan),
  `eval/results_v2/bench_agentic*`, `bench_closed_book*`, `M6_report.md` (new
  artifacts), tuning edits limited to `agentic/checker.py` / `agentic/graph.py`
  as justified by Phase 3, `agent/PROGRESS.md`.

## Verification

- Phase 0: `python tests/test_closed_book.py` + existing 5 suites all pass.
- Phases 1-2: JSONL row counts == 306; scored tables print judge columns.
- Phase 3-5: every PRD §3 criterion has a measured number in M6_report.md;
  compare.py bootstrap printed with p-values. Judge = local Qwen only
  (Gemini capped at 20 calls/day — spot checks at most).
