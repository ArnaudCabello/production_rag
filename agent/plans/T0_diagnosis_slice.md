# T0 — Diagnosis harness + validation slice

## Context

M6's first full run (eval/results_v2/M6_report.md) left three failed PRD §3
criteria: Tier B judge +5.7 (< +10), latency 6.2× (> 5×), unanswerable refusal
22/25 (< 23/25). PRD_TUNING.md breaks the fix into T1-T4, all gated on T0:
a **fixed validation slice** (chosen once, never tuned against) and **trace
diagnosis evidence** for the failing categories. T0 makes **no pipeline
changes** — its deliverables are data + findings that T1-T4 designs cite.

Decisions confirmed with the human: slice is **stratified by outcome**
(seeded random per category, mixing currently-correct and currently-wrong
ids, weighted toward failing categories); trace diagnosis covers **all 5
failing categories** (aggregation, cross_document, unanswerable, ambiguous,
multi_chunk).

On approval this plan is saved to `agent/plans/T0_diagnosis_slice.md` and
PROGRESS.md gets a "planned" entry, per the project workflow.

## Step 1 — Selection script (local, deterministic)

New `eval/make_tuning_slice.py`: reads `eval/golden_set_v2.json` +
`eval/results_v2/bench_agentic_scored.json` (both local), fixed seed, and emits:

1. **Diagnosis ids** — per failing category, ~5 ids where the agentic run
   failed (judge incorrect / not-refused / no-ack respectively; for
   cross_document prefer ev_recall > 0 but judge-incorrect — the
   "evidence found, not converted" cases; for multi_chunk prefer ids that
   regressed vs baseline). Exclude the 5 contamination red-flag ids
   (v2q079, v2q080, v2q154, v2q211, v2q249) and v2q251 (unpaired).
   Written to `eval/results_v2/T0_diagnosis_ids.json` (ids + one-line reason
   each) — these are the "eyeballed" ids.
2. **Tuning slice** — `eval/tuning_slice.json`, ~26 ids, **disjoint from the
   diagnosis ids and red-flag ids**, format mirroring
   `agent/plans/fixtures/planner_slice.json` (`{description, questions:[{id,
   question, category}]}` — no answers). Composition (correct/wrong mix per
   current agentic scored results): aggregation 4, cross_document 4,
   unanswerable 4, ambiguous 2, multi_chunk 3, multi_hop 2, factual 3,
   semantic 2, table 2 = 26. Where a category has both outcomes, split
   ~half correct / half wrong (regression guard + headroom).

Test-first per /build: `tests/test_tuning_slice.py` validates the committed
artifacts (not the script run): slice size 24-30, all ids in golden set,
category coverage as specced, disjointness from diagnosis + red-flag ids,
no answers/reference fields in tuning_slice.json.

## Step 2 — Diagnosis trace run (Colab, human)

One resumable command over the ~25 diagnosis ids (comma-list from
T0_diagnosis_ids.json):

```
python eval/run_benchmark.py --pipeline agentic --model Qwen/Qwen3-14B \
  --trace --ids <diagnosis ids> --output eval/results_v2/trace_diagnosis.jsonl
```

Explicit `--output` keeps `bench_agentic.jsonl` untouched (default naming
would clobber it — run_benchmark.py L140-141). No scoring/judge needed for
diagnosis; the existing full-run scored/judge files already say pass/fail
per id. Human commits/pushes the JSONL (or drops it in Drive) for the agent.

## Step 3 — Analysis + findings (local, agent)

Read `trace_diagnosis.jsonl` traces (plan / retrieve / check / synthesize
events — synthesize carries `dropped_chunks`, `citations_*`; retrieve carries
`round`/`broad`; check carries `sufficient`/`missing`/`queries`) alongside the
scored rows, and write `eval/results_v2/T0_findings.md` with one section per
failing category answering the T1-T4 design questions:

- **aggregation**: does evidence ever reach the prompt (ev_recall vs
  dropped_chunks vs MAX_SYNTH_CHUNKS=20)? are broad top_k=8 queries firing
  (`broad` events)? is the miss recall or synthesis?
- **cross_document / multi_chunk**: evidence in context but judge-incorrect —
  dropped_chunks ordering (first-N loses late rounds)? context formatting?
  answer text vs reference eyeball.
- **unanswerable**: which 3 answered instead of refusing; round burn
  (queries per round, rounds used) for T1's latency case.
- **ambiguous**: does anything in plan/check surface multiple
  interpretations; what the answers do instead.
- **latency appendix for T1**: rounds/llm_calls distribution from traces +
  full-run latency by category (factual wasted round, unanswerable 4-round
  burn) — quantified.

Each section ends with "implication for TX" lines the T1-T4 plans will cite.

## Step 4 — Close out

- `agent/PROGRESS.md`: T0 entry + status board (T0 done; note the slice is
  now FROZEN — later modules must not edit tuning_slice.json).
- Commit + push `agentic_pipeline`: eval/make_tuning_slice.py,
  eval/tuning_slice.json, eval/results_v2/T0_diagnosis_ids.json,
  trace_diagnosis.jsonl, T0_findings.md, tests/test_tuning_slice.py.

## Files touched

- `eval/make_tuning_slice.py` (new), `eval/tuning_slice.json` (new),
  `eval/results_v2/T0_diagnosis_ids.json` (new),
  `eval/results_v2/trace_diagnosis.jsonl` (new, Colab),
  `eval/results_v2/T0_findings.md` (new),
  `tests/test_tuning_slice.py` (new), `agent/PROGRESS.md`.
- **No changes** to agentic/, run_benchmark.py, scorers, or baseline artifacts.

## Verification

- `python tests/test_tuning_slice.py` green; full local suite still green
  (nothing imported by it changes, but run anyway per ground rules).
- trace_diagnosis.jsonl row count == diagnosis id count; every row has a
  `trace` field with a synthesize event.
- T0_findings.md has a section per failing category, each with at least one
  quantified observation and an "implication for TX" line.
- Gemini not used; baseline files untouched (`git status` shows only new files).
