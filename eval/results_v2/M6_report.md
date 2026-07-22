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
