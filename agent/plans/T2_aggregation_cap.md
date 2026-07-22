# T2 — Aggregation recall → synthesis (cap policy)

## Context

M6's full run has aggregation flat at judge 8% for both pipelines — the Tier-B
blocker. T0's trace diagnosis (eval/results_v2/T0_findings.md §1) proved the
retrieval loop is structurally disconnected from synthesis: on 8/8
aggregation-labeled traces, round 1 (5 broad sub-queries × top_k=8) alone
overflows `MAX_SYNTH_CHUNKS=20`, so the first-N cap drops 100% of rounds-2-4
chunks (v2q020: check re-queried the 3 missing systems for 3 rounds, fetched
32 new chunks, and the answer still said they "are not covered"). Findings §2
shows milder cap loss on cross_document/multi_chunk, where first-N throws away
precisely the late-round TARGETED chunks. T1's slice confirmed aggregation
unmoved (judge 2→1, ev_recall 0.19).

Decisions confirmed with the human:
- **Cap selection = query-interleave**: round-robin across query groups
  (each query's #1 hit, then #2s, …) until the cap. Every sub-query and every
  late-round targeted query gets representation; within-query relevance order
  preserved; no tuned reserve constants.
- **Aggregation cap raised to 30** (`MAX_SYNTH_CHUNKS_AGG=30`; ≈15k tokens,
  fine for Qwen3-14B). Other categories stay at 20.

No new LLM calls; deterministic; synthesis prompt wording untouched (T4 owns
synthesis instructions; T3 owns refusal rules).

## Changes (agentic/graph.py only)

### 1. Provenance annotation in `retrieve`

Each appended chunk gets copied with two keys: `{"round": rounds+1, "q_idx": n}`
where `q_idx` is a global per-question query counter (increment per executed
search). Copy (`{**chunk, ...}`) — never mutate retriever-returned dicts.
Downstream is unaffected: runner records only chunk_id+text; scorer uses
chunk_id/text; baseline chunks lack the keys harmlessly.

### 2. `select_synth_chunks(chunks, cap)` — pure function

- `len(chunks) <= cap` → return unchanged (parity: 1-round questions and all
  under-cap cases byte-identical to today).
- Else: group by `q_idx` (groups in first-seen order — the question's reranked
  query is group 0, so its top hit is still selected first and stays chunk [1]),
  then round-robin: pick each group's next unpicked chunk in group order until
  `cap` picks. Output in pick order. Chunks missing `q_idx` (impossible in the
  graph, but stubs) fall into one group — degrades to first-N.

### 3. `synthesize` uses it

- `cap = MAX_SYNTH_CHUNKS_AGG (30) if state["category"] == "aggregation" else
  MAX_SYNTH_CHUNKS (20)`; `capped = select_synth_chunks(state["chunks"], cap)`.
- Everything downstream unchanged: `chunks: capped` record, citation validation
  against the capped list (numbering follows the new order — already computed
  at synthesize time, no scorer change).
- Trace synthesize event gains `context_rounds` (histogram round→count of the
  capped list) alongside the existing dropped_chunks/citations fields, so
  diagnosis runs can verify late rounds now reach synthesis.

## Tests (first, per /build)

New `tests/test_synth_selection.py` (stub patterns from test_synthesis.py):

- Under cap: selection is identity; synthesis context/record byte-identical to
  pre-T2 (parity incl. the fallback/baseline path).
- Over cap: build a union of round-1 flood (3 groups × 10) + round-2/3 targeted
  groups (2-3 chunks each) → every group represented, late-round chunks present
  in the capped list, question-group top hit first, within-group order kept,
  exactly `cap` chunks, deterministic.
- Aggregation cap: category=aggregation with 35-chunk union → 30 kept;
  other categories → 20.
- Citations still validate against the capped (reordered) list.
- Trace: synthesize event carries context_rounds with late rounds non-zero.

Existing suites: test_synthesis's exact synthesize-trace assert gains
`context_rounds`; retrieve-chunk annotation may surface in tests comparing
chunk dicts (parity tests compare chunk_id lists only — expected no change,
verify). Full 10-file suite green.

## Validation (frozen slice, per PRD_TUNING ground rules)

1. Local: new test file + full suite green.
2. Colab (human): slice run →
   `python eval/run_benchmark.py --pipeline agentic --model Qwen/Qwen3-14B
   --ids $(python -c "import json;print(','.join(q['id'] for q in json.load(open('eval/tuning_slice.json'))['questions']))")
   --output eval/results_v2/slice_T2.jsonl`
   then `python eval/score_benchmark.py eval/results_v2/slice_T2.jsonl --judge`.
3. Agent compares slice_T2 vs slice_T1 (and the untuned run) on the same 26
   ids: aggregation ev_recall/key_match/judge up (primary); cross_document/
   multi_chunk not worse (they gain late-round chunks too); latency ~unchanged
   (no new calls; slightly larger agg prompts); refusals hold (4/4).
4. Optional 1-id spot check (T0 backlog): re-run v2q020 with --trace and
   confirm context_rounds shows rounds 2-4 chunks in context and the answer
   stops claiming the re-queried systems are "not covered".
5. Accept → PROGRESS entry (T2 done) + commit + push. Full 306-q run stays T5.

## Files touched

- `agentic/graph.py` (retrieve annotation, select_synth_chunks,
  synthesize cap + trace), `tests/test_synth_selection.py` (new),
  `tests/test_synthesis.py` (trace assert key), `agent/plans/T2_aggregation_cap.md`
  (this plan, saved on approval), `agent/PROGRESS.md`.
- NOT touched: checker/refusal rules (T3), synthesis prompt wording (T4),
  planner, retriever, scorers, eval/tuning_slice.json (frozen), baseline
  artifacts.
