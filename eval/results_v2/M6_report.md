# M6 report — first full-run measurement (untuned, as M5 left it)

Runs: 2026-07-22 Colab A100, Qwen/Qwen3-14B, judge = local Qwen2.5-14B-Instruct.
Files: bench_agentic{,_judge.jsonl,_scored.json}, bench_closed_book{...}, vs
bench_baseline{...}. 306/306 rows each; paired on 305 (v2q251 missing from the
baseline run only).

## PRD §3 scoreboard

| Criterion | Target | Measured | Pass |
|---|---|---|---|
| Tier B judge-correct gain | ≥ +10 pts, p<0.05 | +5.7 (26.7→32.4); overall judge Δ +2.6, p=0.187 | NO |
| Tier A regression | ≤ 3 pts | factual +1.7, semantic +2.5, table +17.1 | YES |
| Unanswerable refusal ≥ baseline | ≥ 23/25 | 22/25 | NO (marginal) |
| LLM calls ≤ 6 | all q | max 6, mean 5.1 | YES |
| Median latency ≤ 5× baseline | ≤ 46.5s | 57.5s = 6.2× | NO |
| Groundedness (a) retrieval ≥ 1 on answered | all | 0 violations | YES |
| Groundedness (d) citation validity | high | 99.4% over 244 cited answers | YES |

Overall key_match: +6.2 pts, bootstrap p = 0.005 (SIGNIFICANT).

## Per-category (judge-correct %, closed_book = contamination floor)

| category | n | closed_book | baseline | agentic | Δ (ag−bl) |
|---|---|---|---|---|---|
| aggregation | 25 | 0.0 | 8.0 | 8.0 | 0.0 |
| ambiguous | 14 | 0.0 | 7.1 | 0.0 | −7.1 |
| cross_document | 50 | 14.0 | 24.0 | 26.0 | +2.0 |
| factual | 60 | 3.3 | 80.0 | 81.7 | +1.7 |
| multi_chunk | 26 | 7.7 | 73.1 | 65.4 | −7.7 |
| multi_hop | 30 | 0.0 | 46.7 | 63.3 | +16.7 |
| semantic | 40 | 7.5 | 30.0 | 32.5 | +2.5 |
| table | 36 | 5.6 | 55.6 | 75.0 | +17.1 (paired) |
| unanswerable | 25 | 48.0 | 88.0 | 76.0 | −12.0 |

Tier B aggregate: closed_book 6.7 / baseline 26.7 / agentic 32.4.

## Findings

1. **Real wins where iteration matters**: multi_hop +16.7 judge (ev_recall
   30→51.7%) and table +17.1. Cross_document key_match +20 with ev_recall
   doubled (17→34%) but judge only +2 — evidence is found, synthesis isn't
   fully converting it.
2. **Aggregation is the blocker**: 8% judge for BOTH pipelines (km 16%).
   The AGG top_k=8/no-rerank knob is not enough. This alone keeps Tier B
   below +10.
3. **Gains are credible, not contamination**: floor is 6.7% on Tier B, 0% on
   multi_hop, 3.3% factual. Only 5 red flags (agentic judge-correct +
   ev_recall 0 + closed-book correct): v2q079, v2q080 (cross_document),
   v2q154 (multi_chunk), v2q211 (semantic), v2q249 (table). Closed-book
   answered 23/25 unanswerable questions — the pipeline's refusal behaviour
   is evidence-driven, as designed.
4. **Regressions**: unanswerable 88→76 judge (check sometimes deems partial
   evidence sufficient), multi_chunk −7.7, ambiguous 0/14 (small n; GAP_NOTE
   may suppress the "acknowledge multiple interpretations" style).
5. **Latency 6.2× > 5× cap**, mean 54.8s — matches the known carry-overs:
   factual burning a sufficient=false round, unanswerable burning all 4
   rounds (~3 queries each).

## Phase 4 tuning targets (priority order)

1. Aggregation: recall + synthesis (largest Tier B headroom).
2. Latency: checker calibration so factual settles in round 1; early-stop
   when a round adds no new chunks (unanswerable path).
3. Refusal calibration: recover unanswerable ≥ 23/25 and ambiguous hedging
   without re-triggering M4-style over-refusal.
4. Cross_document + multi_chunk synthesis: evidence present, judge score not
   following — inspect answers/dropped_chunks via --trace --ids.

Discipline per plan: local tests → shape reports → fixed ~20-30 id slice →
ONE final full run + judge + compare.

---

# T5 — final full run, tuning outcome, and revert (2026-07-23)

The full tuning phase (T1 latency, T2 aggregation cap, T3 refusal calibration;
T4 synthesis conversion was attempted and reverted on the slice) was validated
on a frozen 26-id slice throughout. T5 ran the tuned config (T3c state) on the
full 306 q and judged it. **The tuning did not survive the full set: it traded
accuracy for latency and net-regressed quality.** We reverted the pipeline to
the untuned **baseline agentic** configuration (commit `64ea7ba`, the M5 close
state), which is the strongest config we have measured on the full benchmark.

## Three-pipeline comparison — judge-correct % (paired on 305 q; v2q251 absent from legacy)

| Category | Tier | n | Legacy (linear) | Baseline agentic | Tuned (T5) | ag−lg | tn−ag |
|---|:---:|--:|--:|--:|--:|--:|--:|
| factual        | A | 60 | 80.0 | 81.7 | 73.3 | +1.7  | −8.3  |
| semantic       | A | 40 | 30.0 | 32.5 | 35.0 | +2.5  | +2.5  |
| table          | A | 35 | 57.1 | 74.3 | 68.6 | +17.1 | −5.7  |
| cross_document | B | 50 | 24.0 | 26.0 | 22.0 | +2.0  | −4.0  |
| multi_hop      | B | 30 | 46.7 | 63.3 | 46.7 | +16.7 | −16.7 |
| aggregation    | B | 25 |  8.0 |  8.0 |  8.0 |  0.0  |  0.0  |
| multi_chunk    | — | 26 | 73.1 | 65.4 | 61.5 | −7.7  | −3.8  |
| ambiguous      | C | 14 |  7.1 |  0.0 |  0.0 | −7.1  |  0.0  |
| unanswerable   | C | 25 | 88.0 | 76.0 | 68.0 | −12.0 | −8.0  |
| **Tier A**     | A | 135 | 59.3 | 65.2 | 60.7 | +5.9 | −4.4 |
| **Tier B**     | B | 105 | 26.7 | 32.4 | 25.7 | +5.7 | −6.7 |
| **OVERALL**    |   | 305 | 49.2 | 51.8 | 46.6 | +2.6 | −5.2 |

Cost (mean): legacy 10.8 s / 1.0 call · baseline agentic 54.8 s / 5.1 calls ·
tuned 46.8 s / 4.2 calls.

Overall judge-correct **tuned vs baseline agentic: −5.2 pts, bootstrap
p = 0.007 (SIGNIFICANT)**. Tuned vs legacy: −2.6 (n.s.); baseline agentic vs
legacy: +2.6 (n.s.).

## PRD §3 scoreboard — tuned (T5) vs legacy baseline

| Criterion | Target | Tuned T5 | Pass |
|---|---|---|---|
| Tier B judge gain | ≥ +10, p<0.05 | −1.0 (26.7→25.7) | NO |
| Tier A regression | ≤ 3 | factual −6.7 | NO |
| Unanswerable refusal ≥ baseline | ≥ 23/25 | 17/25 | NO |
| LLM calls ≤ 6 | all q | max 6, mean 4.2 | YES |
| Median latency ≤ 5× | ≤ ~46.5 s | 46.8 s ≈ 4.3× | YES |

## What we tried, why it failed, what we learned

- **T1 (round efficiency)** added a no-new-chunks early stop, a stalled-`missing`
  stop, and a "single stated fact = sufficient" checker rule. It hit the latency
  target (6.2× → 4.3×, under the 5× cap for the first time) — the one durable
  win. But the same stops that cut wasted rounds also **cut productive
  paraphrase-drift retrieval**: multi_hop judge collapsed 63.3 → 46.7 (−16.7,
  ev_recall 51.7 → 48.3). This was the exact risk flagged in the T1 close note
  ("the stalled stop can kill productive drift retrieval; revisit if
  cross_doc/multi_hop look evidence-starved"). The single-fact rule cut factual
  short (81.7 → 73.3).
- **T3 (refusal calibration)** passed its trap slice 3/3 but did not generalize:
  full-set unanswerable refusal kept declining (88 → 76 → 68).
- **Root lesson — the slice was too small per category to see this.** The frozen
  26-id validation slice held ~2 multi_hop and ~4 unanswerable ids; category-
  level regressions of this size are invisible at n=2. Slice-validated gains
  (traps refusing, latency down, aggregation nudged) were real *on the slice*
  and misleading about the full set. **A tuning change that touches retrieval
  termination or refusal must be validated on a per-category n large enough to
  detect a 10-pt swing before it is trusted — or confirmed on a full run.**
- **Aggregation never moved** (8% for all three configs, both key-match 16%).
  It is retrieval/synthesis-limited in a way none of T2's cap knobs touched;
  it remains the structural blocker keeping Tier B below the +10 target.

## Decision and final state

Reverted `agentic/graph.py` + `agentic/checker.py` to `64ea7ba` (baseline
agentic). T1/T2-only tests removed; the T1 `new_chunks` init key dropped from
`run_benchmark.py` (closed-book adapter and T0 slice infra kept). Suite green.

**Final pipeline = baseline agentic.** Against the legacy linear baseline it is
+2.6 overall / +5.7 Tier B judge (neither significant at p<0.05), with real,
credible wins on multi_hop (+16.7) and table (+17.1) above a low contamination
floor. It does **not** meet the PRD's primary bar (Tier B +10, p<0.05) and it
costs 5× the latency. The honest headline: iterative retrieval helps where
iteration structurally matters (multi_hop, table), the gains are not yet large
or significant enough to clear the PRD target, aggregation is unsolved, and the
latency/refusal tuning that would make it cheaper cost more accuracy than it
saved.

All tuning artifacts are preserved (`slice_T1..T4b`, `trap_T3..T4b`,
`bench_agentic_T5*`) and the T-module plans in `agent/plans/` record each
attempt. Tuning attempts (T1–T4) are documented in `agent/PROGRESS.md`.
