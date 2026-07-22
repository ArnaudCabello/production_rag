# T0 findings — trace diagnosis of the 5 failing categories

Source: `eval/results_v2/trace_diagnosis.jsonl` (25 ids from
`T0_diagnosis_ids.json`, agentic pipeline, Qwen3-14B, `--trace`), read against
`bench_agentic_scored.json` / `bench_baseline_scored.json` and the golden set.
All counts below are from these 25 traces; ids cited inline.

## 1. Aggregation (judge 8%, the Tier-B blocker)

**The retrieval loop is structurally disconnected from synthesis: on every
aggregation-labeled question, round 1 alone overflows `MAX_SYNTH_CHUNKS=20`,
so 100% of chunks retrieved in rounds 2-4 are dropped.**

- 8/8 traces the planner labeled `aggregation` (v2q002/007/018/020/035/037/
  038/286...) have context rounds `{1: 20}` — the capped first-N is filled
  entirely by round 1 (5 broad sub-queries × top_k=8 un-reranked ≈ 30-37
  unique chunks). Everything the check-driven rounds 2-4 fetch is discarded:
  v2q020 dropped 47 of 67 chunks (32 of them from rounds 2-4), v2q007 dropped
  26 of 46.
- Smoking gun: on v2q020 the check correctly identified the three missing
  composite systems, re-queried them for 3 straight rounds (16+6+10 new chunks
  retrieved), and the final answer still says those systems "are not covered
  in the provided sources" — the synthesis never saw what the loop fetched.
  Same shape on v2q007 (gaps repeated verbatim all 4 rounds).
- ev_recall 0.0 on 4/5 aggregation ids is measured against the CAPPED list
  (record = what the LLM saw, M5 design), so part of the "recall" miss may
  actually be cap loss; dropped-chunk text isn't in the record, so whether the
  dropped rounds contained golden evidence needs a 1-2 id Colab spot check —
  but the structural waste is proven regardless.

**Implication for T2**: the first-N cap policy is the primary lever, not more
recall — e.g. round-aware/interleaved selection (reserve budget for later
rounds or per-sub-query quotas) and/or raising `MAX_SYNTH_CHUNKS` for
aggregation. Broad `AGG_SUBQUERY_TOP_K=8` round 1 already floods the budget;
adding recall without fixing the cap cannot help.
**Implication for T1**: rounds 2-4 on aggregation currently buy nothing but
latency (v2q007: 4 rounds, 60s) — once T2 makes later rounds visible to
synthesis they pay for themselves; until then aggregation is also a latency
sink.

## 2. cross_document / multi_chunk (evidence found, not converted)

**The check never says "sufficient" (all-F through 4 rounds on 4/5
cross_document traces) while re-emitting near-paraphrase queries that return
almost nothing new; synthesis then answers from a context whose tail was
truncated mid-loop.**

- New-chunks-per-round decays fast: v2q087 rounds 2-4 add 3/2/1 chunks,
  v2q068 adds 5/2/1, yet the check re-queries to the round cap with lightly
  reworded variants of the same 2-3 queries (see v2q058/089 check events).
  4-round burn puts cross_document at 64-93s latency (vs ~46.5s target).
- Cap loss hits here too, milder than aggregation: v2q058 dropped 12 chunks
  (all from rounds 3-4), v2q051 dropped 9 (round 3), v2q170 (multi_chunk)
  dropped 10 (rounds 3-4). The late rounds are precisely the targeted-gap
  retrievals — first-N ordering throws away the most question-specific
  evidence while keeping round-1 generics.
- multi_chunk splits in two: ids that finish in 1 round (v2q151/165/171,
  15-40s, judge partial — synthesis quality, chunks all present) vs ids that
  burn 4 rounds for nothing (v2q150: round 4 adds 0 new chunks; v2q170).
- ev_recall is 0.5 on all 5 cross_document ids — half the golden evidence is
  in the (capped) context and the judge still scores partial/incorrect:
  answers hedge or genuinely synthesize the wrong relation (v2q089 answers
  "both involve enhanced resistance" — a non-answer to the asked relation).

**Implication for T4**: two levers, in order — (a) cap selection must not be
first-N in retrieval order (late-round targeted chunks are the ones being
dropped); (b) synthesis instruction for comparative questions (answer the
relation, per-source structure) since half the misses have full evidence in
context. **Implication for T1**: early-stop when a round adds ~no new chunks
(v2q150 round 4 = 0 new; v2q068/087 rounds 3-4 ≤ 2) — this alone removes 1-2
rounds ≈ 20-40s from the worst cross_document/multi_chunk cases without
touching quality (those rounds contributed nothing to context anyway under
the current cap).

## 3. Unanswerable (22/25 refused vs baseline 23/25)

**The 3 false answers are near-miss traps: the check flips to `sufficient`
once adjacent-but-different-condition evidence shows up, and the synthesis
then states it confidently.**

- v2q284: check goes F→T after round 2; answer asserts "99.6% relative
  density [3]" for a composition/condition the corpus reports differently —
  the classic near-neighbor trap. v2q283 same shape (F,F→T; conductivity of a
  related composite presented as the asked one). v2q299: check accepts after
  round 2 but the answer itself half-hedges ("not explicitly detailed...
  however") — the GAP_NOTE never fired because gaps=[] once sufficient=true.
- The 2 correctly-refused traces (v2q286/288) show the T1 latency case: 4
  rounds × 3 near-identical paraphrase queries (v2q288 re-runs the same 3
  queries 4 times, rounds 2-4 adding 3/7/8 chunks of drift), 73-90s, never
  settling on `queries=[]` despite `missing` staying byte-identical each round.
- Planner never emits `unanswerable_maybe` (0/25 traces here, consistent with
  the M3 shape report) — refusal rests entirely on the check, as designed.

**Implication for T3**: the check needs a "same property, DIFFERENT
material/condition ≠ evidence" rule (both false answers are condition
mismatches, not hallucinations). **Implication for T1**: stop condition when
`missing` is unchanged between rounds (or no new chunks arrive) — the
refusal traces burn 2-3 redundant rounds each.

## 4. Ambiguous (0/14 judge-correct)

**Not an acknowledgment problem — 12/14 answers already acknowledge multiple
values. It is a retrieval-targeting miss: the answers enumerate values from
the WRONG papers.**

- All 5 traces: ev_recall 0.0, key_match False, judge partial. v2q027 answers
  with generic HP 1600-2300°C / PS ≥2000°C / SPS values; the reference (and
  answer_keys) want the two specific studies (1300°C PIP, 1750-1950°C SPS).
  v2q031/035 same: right shape ("varies by system..."), wrong specifics.
- Mechanism: 3/5 ambiguous questions get labeled `aggregation` → 5 broad
  sub-queries → 30+ generic survey chunks fill the cap; 2/5 get `simple` →
  2 queries, check satisfied round 2. Either way the intended source docs
  never surface.
- The judge's "partial" (13/14) is arguably fair — the answers are true but
  unspecific. Baseline is 1/14; the realistic gain here is small and comes
  from recall precision, not prompt phrasing.

**Implication for T3**: acknowledge-multiple prompt work is already done
(12/14); the lever is retrieval specificity for these under-specified
questions — and expectations should stay modest (this is 14 questions and
the misses are recall, the hardest kind). De-prioritize vs unanswerable.

## 5. Latency appendix (for T1)

Full-run medians: agentic 6.2× baseline (cap 5×, ≈46.5s). Trace-level
structure of the excess, by dominant cause:

| Pattern | Trace evidence | Wasted cost |
|---|---|---|
| Redundant late rounds (check can't be satisfied, re-queries paraphrases, few/no new chunks) | v2q288 (4×3 same queries, 90s), v2q150 (round 4: 0 new, 96s), v2q087 (rounds 3-4: 2+1 new, 73s), v2q068, v2q089 (93s) | 1-3 rounds ≈ 15-40s/q on cross_document, multi_chunk, unanswerable |
| Aggregation multi-round retrieval that synthesis never sees (§1) | v2q007/011/020 (60-71s, all context from round 1) | 1-3 rounds/q, pure waste until T2 lands |
| 1-round questions are already fine | v2q151 15s, v2q035 16s, v2q171 19s | — |

Concrete T1 stop conditions this data supports, strongest first:
1. **No-new-chunks early stop** in `route_after_check`: if a round's queries
   returned 0 (or ≤1) chunks not already in the union → synthesize. Fires on
   v2q150/068/087-style tails; zero quality risk under the current cap.
2. **Stalled-check stop**: `missing` unchanged from previous round → stop
   re-querying (fires on v2q288/286/007/020; complements #1 when paraphrase
   queries keep dredging up drift chunks).
3. Factual round-1 settling is a CHECK_SYSTEM calibration item (known from
   M6; factual not in this trace set — validate via `check_shapes.py`).

## Colab spot-check backlog (cheap, optional)

- 1-2 aggregation ids (v2q020, v2q007): fetch the DROPPED chunk texts and
  check whether golden `evidence_any` strings were in them — quantifies how
  much of aggregation's ev_recall-0 is cap loss vs true recall miss (shapes
  T2's split between cap policy and recall knobs).
