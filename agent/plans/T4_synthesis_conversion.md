# T4 — Synthesis conversion (cross_document, multi_chunk)


## Context

M6 full run: cross_document/multi_chunk find evidence the judge doesn't credit.
T0 findings §2: half the misses have **full evidence in the capped context**;
answers hedge or answer the wrong relation (v2q089: "both involve enhanced
resistance" — a non-answer to the asked relation). T2 already fixed lever (a)
from T0 §2 (cap ordering → query-interleave), so T4 is lever (b): the
synthesis instruction. Fresh evidence from the T3c slice confirms two failure
modes at ev_recall 1.0:

- **Relation not answered directly / one side dropped**: v2q083 answers the
  Ta/Nb-reduction side, trails off on the thermal-conductivity side that the
  reference requires.
- **Reference-level specifics omitted**: v2q158 (partial 3 runs straight,
  ev 1.0, key_match true) gives the right mechanism but drops the specific
  values (agglomerate ~10 μm, densities 6.51 vs 10.5 g/cm3) the reference
  contains.
- v2q127 was checked and is judge churn, not synthesis (answer matches the
  reference) — not a T4 target.

Structure is NOT the problem: `format_context` already headers every chunk
with `[n] <pdf> — <headings>`, so doc identity is visible to the generator.

Decisions confirmed with the human:
- Instruction applies to **all questions** (unconditional) — planner labels
  are unreliable (M3 shape report), so gating would miss mislabeled
  cross_document questions. Accepted consequence: the M1 parity test's
  "baseline prompt is the last llm call" assert is intentionally redefined
  (precedent: M2 planner, M4 check-call count).
- **Both levers** in the instruction: direct-relation answering AND
  every-part-with-specifics.

Zero new LLM calls. Checker untouched (T3 just closed — do not disturb the
accepted CHECK_SYSTEM). GAP_NOTE wording untouched (REFUSAL regex
load-bearing). Cap policy untouched (T2). Baseline `generation/pipeline.py`
untouched.

## Change (`agentic/graph.py` only)

New module constant, appended to the system message in `synthesize` (around
graph.py L200-210 where `SystemMessage(content=SYSTEM_PROMPT)` is built):

```python
SYNTH_GUIDE = """
Additionally:
- Answer exactly what the question asks. If it asks how things compare or
  relate, state the comparison/relationship explicitly and cover EACH side
  with its own citation — never reply with a generic similarity.
- Address every part of the question, and give the sources' specific values
  (numbers, sizes, temperatures, compositions) rather than qualitative
  summaries."""
```

(Exact wording tunable at build time; system message becomes
`SYSTEM_PROMPT + SYNTH_GUIDE`.) GAP_NOTE continues to prepend to the USER
message unchanged — the two compose without interaction.

## Validation first (per /build)

1. **Local tests before code**: extend `tests/test_synthesis.py` with asserts
   that (a) the synthesize LLM call's system message ends with SYNTH_GUIDE,
   (b) GAP_NOTE behaviour is unchanged when gaps present. Update
   `tests/test_agentic_parity.py`: the "last call uses the baseline prompt"
   assert becomes "last call uses baseline prompt + SYNTH_GUIDE" —
   intentional redefinition, noted inline (per the no-silent-test-weakening
   rule). Full 11-file suite green.
2. **Colab slice run** (human): same frozen 26-id `--ids` command →
   `eval/results_v2/slice_T4.jsonl` + `score --judge`. No shape report needed
   (checker untouched).
3. Optional trap re-check `--ids v2q283,v2q284,v2q299 --trace` →
   `trap_T4.jsonl`: the instruction must not push synthesis to fill gaps with
   neighbor specifics (v2q284's hedge must survive; "specific values" demand
   is the risk — watch this).

## Acceptance gates (compare slice_T4 vs slice_T3c, T1/T2/T3 conventions)

- Primary: cross_document + multi_chunk judge up (T3c: 1/4 and 1/3); v2q158
  partial→correct is the named conversion target; v2q083 covers both sides.
- Refusals hold: unanswerable ≥ 3/4 judge-correct AND refusal-regex count
  ≥ 3; traps still refuse/hedge if run.
- No regression: aggregation/factual/semantic/table judge not worse (robust,
  ev>0 flips only; ev-0.0 churn discounted symmetrically as always).
- Cost: latency ~flat, llm_calls ~4.0 (instruction adds no calls; slightly
  longer prompt is noise).
- If the specifics demand causes near-miss leakage on unanswerables (traps
  answering again), fallback is scoping the specifics bullet with "only
  values the sources state for the asked subject" — a build-time revision,
  documented in PROGRESS if used.

## Revision T4.1 (fallback triggered)

The v1 slice/trap validation hit exactly the pre-registered risk: trap v2q284
answered with the false 99.96% density value that T3c's synthesis hedged away
(traps 2/3). Meanwhile v2q083 converted partial→correct (named target) and
v2q127 recovered; v2q158 unchanged; refusal + cost gates passed. Per the
fallback clause above, the specifics bullet is scoped: "only values the
sources state for the asked subject; never substitute a value reported for a
different material, composition, or condition" (second clause mirrors T3's
accepted checker language). Re-validate: trap trio + slice → trap_T4b/
slice_T4b, same gates; v2q284 must hedge again, v2q083 must stay correct.

## Files touched

- `agentic/graph.py` (SYNTH_GUIDE + one line in synthesize),
  `tests/test_synthesis.py`, `tests/test_agentic_parity.py` (intentional
  redefinition), `agent/plans/T4_synthesis_conversion.md` (this plan),
  `agent/PROGRESS.md`.
- NOT touched: `agentic/checker.py`, GAP_NOTE, `generation/pipeline.py`,
  cap policy/select_synth_chunks, `eval/tuning_slice.json` (frozen),
  baseline artifacts.

## Outcome — CLOSED AS ATTEMPTED (reverted)

T4.1 re-validation (slice_T4b/trap_T4b vs slice_T3c) failed both ways:
v2q284 still answered the false 99.96% density value (traps 2/3 — the scoping
clause did not stop the substitution), judge 11/26 (T3c 12), the v1 v2q083
gain did not reproduce, paranoia canary v2q105 dropped correct→partial at
ev 1.0, and cost drifted up monotonically across T3c→T4→T4b (llm 4.08→4.15→
4.27, latency med 48.0→54.5→52.8s). v2q158 never converted under either
wording (5 runs partial at ev 1.0). Decision with the human: full revert of
SYNTH_GUIDE and the test redefinitions to the T3c state (commit 92a6f32);
no T4.2. T5 proceeds from the T3c configuration.
