# T5 — Final full run + close-out

## Context

T0–T3 are accepted; T4 was attempted twice and reverted — the pipeline is
frozen at the T3c configuration (code paths == commit 92a6f32). T5 is the last
PRD_TUNING module: ONE full 306-question agentic run of the tuned pipeline,
judged and compared against `bench_baseline_scored.json`, producing the final
PRD §3 scoreboard in `eval/results_v2/M6_report.md`. No pipeline code changes —
if anything in the run looks broken, T5 stops and reports; it does not tune.

## Phase 1 — Pre-flight (local, agent)

1. Verify frozen state: `git diff 92a6f32 -- agentic/` is empty and
   `git diff 92a6f32 -- eval/run_benchmark.py eval/score_benchmark.py eval/compare.py`
   is empty (only docs/artifacts may differ). Any diff → stop and report.
2. Full 11-file suite green with `.venv/bin/python`.
3. No code edits. The untracked `eval_run.log` is left alone (human's call).

## Phase 2 — Colab (human-driven, all resumable)

1. Full run — explicit `--output` is MANDATORY (default would clobber the
   untuned `bench_agentic.jsonl`, which must be preserved as the tuning-delta
   reference):
   `python eval/run_benchmark.py --pipeline agentic --model Qwen/Qwen3-14B --output eval/results_v2/bench_agentic_T5.jsonl`
   (NO `--trace`; hours; copy JSONL to Drive between sessions.)
2. Judge + score (local Qwen judge only — Gemini stays at spot-check volume):
   `python eval/score_benchmark.py eval/results_v2/bench_agentic_T5.jsonl --judge`
   → `bench_agentic_T5_judge.jsonl` + `bench_agentic_T5_scored.json`.
3. Commit + push the three artifacts.

No closed-book re-run: the closed-book control is question-only (same model,
prompt untouched since M6 Phase 2) — the contamination floor from
`bench_closed_book_scored.json` carries over unchanged.

## Phase 3 — Analysis (agent)

1. Sanity gates before any comparison: 306 rows; `llm_calls` max ≤ 6;
   every answered question has `retrieval_calls ≥ 1`.
2. Headline compare:
   `python eval/compare.py eval/results_v2/bench_baseline_scored.json eval/results_v2/bench_agentic_T5_scored.json`
   (paired bootstrap p-values printed; expect pairing on 305 — v2q251 absent
   from baseline).
3. Secondary compare — tuning delta: same command with
   `bench_agentic_scored.json` (untuned) as baseline, to attribute what
   T1–T3 bought on the full set.
4. Update `eval/results_v2/M6_report.md` with a final section:
   - PRD §3 scoreboard, every criterion measured, pass/fail stated:
     Tier B ≥ +10 (p<0.05); Tier A regression ≤ 3; unanswerable refusal
     ≥ 23/25 and false-answer rate ≤ baseline; llm_calls ≤ 6; median latency
     ≤ 5× baseline (≤ ~46.5 s); groundedness (retrieval ≥ 1, ev_recall↔
     correctness, closed-book floor, citation validity, red-flag ids).
   - Per-category table (closed_book / baseline / untuned agentic / T5).
   - Tuning trajectory summary (what each of T1–T3 changed; T4 closed
     negative) and honest residual-gap diagnosis (aggregation expected to
     remain the blocker — say so with numbers if it does).

## Phase 4 — Close-out

- PROGRESS.md entry + status board: T5 done, M6 done (or "done — criteria
  partially met", stated honestly).
- Commit + push `agentic_pipeline`.

## Files touched

- `agent/plans/T5_final_run.md` (this plan), `agent/PROGRESS.md`,
  `eval/results_v2/M6_report.md` (final section),
  `eval/results_v2/bench_agentic_T5{.jsonl,_judge.jsonl,_scored.json}` (new
  artifacts, human's Colab commit). NO changes under `agentic/` or `eval/*.py`.

## Verification

- Phase 1: diffs vs 92a6f32 empty; 11-file suite green.
- Phase 2: JSONL row count == 306; scored table prints judge columns.
- Phase 3: every PRD §3 criterion has a measured number + pass/fail in
  M6_report.md; compare.py bootstrap p-values quoted verbatim.
- Standing gotchas respected: never overwrite `bench_baseline*` or the
  untuned `bench_agentic*`; re-scoring baseline requires
  `--judge-file eval/results_v2/bench_baseline_judge.jsonl`; judge = local
  Qwen only.
