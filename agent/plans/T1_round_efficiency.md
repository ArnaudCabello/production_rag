# T1 — Round efficiency (latency)

## Context

M6's full run measured median latency 6.2× baseline against a 5× cap
(~46.5 s). T0's trace diagnosis (eval/results_v2/T0_findings.md §5) showed the
excess is concentrated in redundant late rounds: the check re-emits paraphrase
queries that return ≤2 (often 0) new chunks and a byte-identical `missing`
list for 2-3 straight rounds (v2q288 90s, v2q150 96s, v2q089 93s), and — per
findings §1 — those rounds' chunks are currently 100% dropped by the synthesis
cap anyway. Separately, M6's shape report showed 4/5 factual questions get
`sufficient=false`, wasting a round on the easiest category.

T1 lands the two data-backed structural stops plus the factual checker
calibration. Targets (PRD_TUNING §4): median latency ≤ 5×, mean llm_calls
down, no Tier A/B metric drop on the frozen slice. No changes to synthesis,
cap policy (T2), or refusal rules (T3).

On approval this plan is saved to `agent/plans/T1_round_efficiency.md` and
PROGRESS.md gets a "planned" entry.

## Changes (all in `agentic/graph.py` + `agentic/checker.py`)

### 1. No-new-chunks early stop (retrieve → synthesize edge)

Strongest signal (findings §5 #1), and placing it on the retrieve edge also
skips that round's check LLM call (~seconds + 1 llm_call):

- `retrieve` records `new_chunks` (count added this round) in state
  (new `AgentState` key, init in all invoke sites — known LangGraph gotcha).
- New conditional edge `route_after_retrieve`: if `rounds > 1` and
  `new_chunks == 0` → `synthesize`; else → `check`. Round 1 always checks
  (the check is the only unanswerable detector — M4 decision must hold).
- Threshold is strictly 0: zero quality risk (nothing new to judge); the
  ≤2-drip cases are covered by stop #2. Gaps from the PREVIOUS check verdict
  persist in state, so GAP_NOTE still fires correctly on the skip path.
- Trace: retrieve event gains `new_chunks`; a skip emits
  `{"node": "check", "skipped": "no_new_chunks"}`-style event so diagnosis
  runs can count firings.

### 2. Stalled-check stop (in the `check` node)

Findings §5 #2 (fires on v2q288/286/007/020 where paraphrase queries keep
dredging drift chunks so stop #1 never triggers):

- In the `check` node: if the verdict is insufficient and `verdict["missing"]`
  equals the previous round's `state["gaps"]` (compare normalized — strip/
  lower), force `pending_queries = []` → `route_after_check` synthesizes.
  Uses the existing `gaps` state key ("last verdict wins" already in place);
  no new prompt text, no new state.
- Round 1 immune by construction (`gaps` inits to `[]` and an insufficient
  round-1 verdict has non-empty missing).
- Trace: check event gains `stalled: true` when it fires.

### 3. Factual settles in round 1 (CHECK_SYSTEM calibration)

The one prompt edit, targeting M6's "4/5 factual sufficient=false":

- Add one rule to `CHECK_SYSTEM` (agentic/checker.py): a question asking for
  a single specific fact/value is sufficient as soon as any snippet directly
  states it — do not request more evidence to corroborate an already-found
  fact. (Wording tuned during build; guard: must not weaken the off-topic /
  insufficiency rules — T3 owns those.)
- Validated on Colab via the existing `eval/check_shapes.py` (watch: factual
  flips to mostly sufficient=true; unanswerable stays sufficient=false).

Budget: no lever adds an LLM call; stops #1/#2 strictly reduce calls
(worst case unchanged at 6).

## Tests (first, per /build)

New `tests/test_round_efficiency.py` (StubLLM/stub-retriever patterns from
tests/test_retrieval_loop.py / test_check.py):

- Early stop: scripted retriever returns duplicates in round 2 → graph goes
  retrieve→synthesize, check called once, llm_calls reflects the saved call,
  gaps from round-1 verdict still drive GAP_NOTE.
- No early stop when new chunks arrive; round 1 never skips check.
- Stalled stop: stub check returns identical `missing` twice → loop ends
  after round 2 despite non-empty queries; differing `missing` keeps looping.
- MAX_ROUNDS cap still the final backstop; parity path (fallback plan,
  2-llm-call baseline behaviour) unchanged.

Existing suites: `new_chunks` init key added to every invoke-site dict
(run_benchmark.py + all test files' init dicts — same M3 pattern); exact-dict
trace asserts in test_retrieval_loop/test_agentic_parity gain the new event
keys. Full 8-suite run must stay green.

## Validation (shape report + frozen slice, per PRD_TUNING ground rules)

1. Local: new test file + full suite green.
2. Colab (human): `python eval/check_shapes.py --model Qwen/Qwen3-14B` —
   factual mostly sufficient=true; unanswerable still 5/5 sufficient=false.
3. Colab (human): slice run —
   `python eval/run_benchmark.py --pipeline agentic --model Qwen/Qwen3-14B
   --ids <26 slice ids from eval/tuning_slice.json> --output
   eval/results_v2/slice_T1.jsonl` (+ score with `--judge`).
4. Agent compares slice_T1 vs the SAME 26 ids in the untuned full run
   (bench_agentic_scored.json): latency and llm_calls down (esp. the 4-round
   burners), key_match / ev_recall / judge-correct / refusal not worse.
   compare.py works on the id intersection directly.
5. Accept → PROGRESS entry (T1 done) + commit + push. The full 306-q re-run
   stays in T5.

## Files touched

- `agentic/graph.py` (route_after_retrieve, new_chunks, stalled check,
  trace events), `agentic/checker.py` (one CHECK_SYSTEM rule),
  `eval/run_benchmark.py` (init dict key only),
  `tests/test_round_efficiency.py` (new), existing test init dicts/trace
  asserts, `agent/plans/T1_round_efficiency.md` (new), `agent/PROGRESS.md`.
- NOT touched: MAX_SYNTH_CHUNKS / cap policy (T2), GAP_NOTE + refusal rules
  (T3), planner, scorers, baseline artifacts, eval/tuning_slice.json (frozen).
