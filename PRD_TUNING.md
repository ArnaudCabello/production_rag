# PRD — Agentic pipeline tuning (M6 Phase 4+)

Continuation of `PRD.md` (which still defines the overall goal and success
criteria). This document breaks the tuning work surfaced by the first full
benchmark run (`eval/results_v2/M6_report.md`, 2026-07-22) into modules that
follow the same workflow: plan into `agent/plans/TX_<name>.md` → human review →
`/build TX` (validation first) → test → PROGRESS.md entry → commit.

## 1. Where we are (measured)

First untuned full run vs baseline (305 paired, judge = local Qwen2.5-14B):

- key_match +6.2 pts, p = 0.005 (SIGNIFICANT) — the pipeline is real.
- Tier B judge-correct +5.7 (26.7 → 32.4) — **target is ≥ +10, NOT met**.
  multi_hop +16.7 and table +17.1 are genuine wins; aggregation flat at 8%
  for both pipelines is the blocker; cross_document ev_recall doubled
  (17 → 34%) but judge only +2 — evidence found, synthesis not converting.
- Median latency 6.2× baseline — **cap is 5×, NOT met**. Known causes:
  factual burns a sufficient=false round; unanswerable burns all 4 rounds.
- Unanswerable refusal 22/25 vs baseline 23/25 (**criterion ≥ baseline,
  marginally NOT met**); ambiguous 0/14 judge (baseline 1/14).
- Groundedness clean: 0 answered questions without retrieval, citation
  validity 99.4%, contamination floor 6.7% on Tier B, 5 red-flag ids.

## 2. Goal

Close the three failed PRD §3 criteria without breaking the ones that pass:

1. Tier B judge-correct ≥ +10 pts over baseline, p < 0.05.
2. Median latency ≤ 5× baseline (≤ ~46.5 s).
3. Unanswerable refusal ≥ baseline (≥ 23/25); ambiguous ≥ baseline.

Hold: Tier A regression ≤ 3 pts, llm_calls ≤ 6, groundedness checks.

## 3. Ground rules (all modules)

- The final number comes from ONE full 306-q run at the end (T5) — everything
  before it is validated on shape reports and the fixed slice only.
- **Fixed validation slice**: chosen once in T0, ~20-30 ids spanning all
  categories, disjoint from the ids eyeballed during diagnosis. Never tune
  against golden-set answers — only shapes, costs, and slice-level metrics.
- Budget stays ≤ 6 LLM calls worst case. Any lever that adds a call must
  remove one elsewhere.
- Baseline artifacts (`bench_baseline*`) are read-only. Re-scoring anything
  judge-scored requires `--judge-file`. Gemini stays at spot-check volume
  (≤ 15 calls, 20/day cap).
- Local tests green before any Colab time; Colab validation before any full
  run.

## 4. Module breakdown

| # | Module | Deliverable | Targets |
|---|--------|-------------|---------|
| T0 | Diagnosis harness + validation slice | Fixed slice committed (`eval/tuning_slice.json`); `--trace --ids` diagnosis runs on aggregation / cross_document / unanswerable failures incl. `dropped_chunks`; short findings note per category | Evidence for T1-T4 designs; no pipeline changes |
| T1 | Round efficiency (latency) | Checker calibration so factual settles in round 1; early-stop in `route_after_check` when a round adds no new chunks; unanswerable settles on queries=[] | Median latency ≤ 5×; llm_calls mean down; no Tier A/B metric drop on slice |
| T2 | Aggregation recall + synthesis | Informed by T0: likely broader/multi-query recall (AGG_SUBQUERY_TOP_K, MAX_PENDING_PER_ROUND), MAX_SYNTH_CHUNKS budget for agg, and/or an aggregation-aware synthesis instruction | Aggregation judge-correct meaningfully above 8% on slice/shape evidence |
| T3 | Refusal + ambiguous calibration | GAP_NOTE / CHECK_SYSTEM adjustments: refuse when nothing relevant (recover 23/25+), acknowledge multiple interpretations on ambiguous; guard against M4-style over-refusal | Unanswerable refusal ≥ baseline; ambiguous ack ≥ baseline; factual answers unaffected |
| T4 | Synthesis conversion (cross_document, multi_chunk) | Informed by T0: evidence reaches the prompt but judge score doesn't follow — likely chunk-cap ordering (first-N loses late-round evidence), context formatting, or synthesis instruction | cross_document judge delta > +2; multi_chunk regression erased, on slice evidence |
| T5 | Final full run + close-out | ONE full 306-q agentic run + judge + compare + updated M6_report.md; PRD §3 scoreboard pass/fail; PROGRESS + commit | The headline numbers |

Order: T0 → T1 → (T2, T3, T4 in any order, informed by T0) → T5. Modules
T1-T4 are independent enough to plan/build one at a time; T5 only runs when
everything else is merged.

## 5. Levers inventory (where the knobs live)

- `agentic/checker.py`: `CHECK_SYSTEM` / `CHECK_USER`, `MAX_CHECK_QUERIES=3`,
  `MAX_CHUNK_CHARS=300`, fail-safe fallback.
- `agentic/graph.py`: `GAP_NOTE`, `route_after_check` (early-stop seam),
  `MAX_ROUNDS=4`, `MAX_PENDING_PER_ROUND=3`, `AGG_SUBQUERY_TOP_K=8`,
  `MAX_SYNTH_CHUNKS=20` (first-N of retrieval order — suspect for T4).
- `agentic/planner.py`: `PLANNER_SYSTEM` / `PLANNER_USER`, `MAX_SUB_QUERIES=4`
  (only aggregation/comparative labels are reliable — see M3 notes).
- Shape reports: `eval/check_shapes.py`, `eval/planner_shapes.py` (Colab).

## 6. Success criteria for the tuning effort as a whole

T5's compare vs `bench_baseline_scored.json` shows, with p-values printed:
Tier B ≥ +10 (p < 0.05), Tier A regression ≤ 3, unanswerable refusal ≥ 23/25,
median latency ≤ 5×, llm_calls ≤ 6, groundedness checks clean. Anything short
of that gets written up honestly in M6_report.md with the residual gap and
its diagnosed cause.

## 7. Process

Identical to `PRD.md` §7: `/onboard` to resume, plan mode with the human →
`agent/plans/TX_<name>.md`, `/build TX` (validation set / diagnosis evidence
FIRST, then code), tests green, PROGRESS.md entry, commit + push to
`agentic_pipeline`.
