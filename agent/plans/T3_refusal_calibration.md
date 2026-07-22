# T3 — Refusal calibration (near-miss traps) + factual round-1 settling

## Context

M6's full run: agentic refuses 22/25 unanswerable vs baseline 23/25. T0 trace
diagnosis (eval/results_v2/T0_findings.md §3) showed the 3 false answers
(v2q283/284/299) are all **near-miss traps**: the check flips `sufficient=true`
once evidence about the same property for a *different* material/composition/
condition arrives, gaps go empty, GAP_NOTE never fires, and synthesis states
the neighbor's value confidently. Separately, T1's shape report left factual
still 4/5 `sufficient=false` in round 1 (the existing single-fact rule is too
weak) — a wasted check round on the easiest category.

Decisions confirmed with the human:
- **Refusal only** — ambiguous is out of scope (T0 §4: misses are
  retrieval-targeting; acknowledgment already 12/14; de-prioritized).
- **Fold in the strengthened factual rule** — same file (CHECK_SYSTEM), same
  Colab validation pass; acceptance gates cover both directions
  (factual flips sufficient WITHOUT unanswerable regressing).

Zero new LLM calls; prompt-only change. Cap policy untouched (T2 done);
synthesis wording untouched (T4); GAP_NOTE wording untouched (its "not
available in the corpus" phrasing is load-bearing for the scorer's REFUSAL
regex).

## Changes (`agentic/checker.py`, CHECK_SYSTEM only)

1. **Near-miss rule** (new): evidence reporting the asked property for a
   DIFFERENT material, composition, or test condition does NOT cover the
   question. If the snippets only contain such near-neighbor values, the part
   is still missing — name the exact asked material/composition/condition in
   `missing` (so GAP_NOTE fires with a precise gap). Do not present a
   neighbor's value as the answer.
2. **Strengthened factual rule** (replaces the T1 wording): make it the FIRST
   decision — if the question asks for a single specific fact/value and any
   snippet directly states it *for the asked material and condition*, reply
   `sufficient=true` immediately; corroboration is never required. (The
   "for the asked material and condition" clause keeps rules 1 and 2 from
   fighting each other.)

No parser/graph/state changes: `parse_check`, verdict keys, trace events all
unchanged. Local suite is wording-independent — `tests/test_check.py` and the
full 11-file suite must stay green as-is (no test edits expected; if any test
asserts prompt text, that's a finding to surface, not to patch around).

## Validation (existing harnesses; no new local test file — prompt-only)

Per /build, the validation set comes first — here it already exists:
`eval/check_shapes.py` over `agent/plans/fixtures/planner_slice.json` (its 5
unanswerable ids include trap ids v2q283/284) + the frozen 26-id
`eval/tuning_slice.json`. Gates:

1. Local: full suite green, unchanged.
2. Colab (human) shape report: `python eval/check_shapes.py --model Qwen/Qwen3-14B`
   - factual: ≥4/5 `sufficient=true` round 1 (the flip target),
   - unanswerable: 5/5 `sufficient=false` HOLDS (hard gate),
   - semantic/table: not worse than the T1-era report (3/5 true each).
3. Colab trap spot check: `python eval/run_benchmark.py --pipeline agentic
   --model Qwen/Qwen3-14B --trace --ids v2q283,v2q284,v2q299
   --output eval/results_v2/trap_T3.jsonl` — all 3 should now refuse or hedge
   with gaps fired (check stays insufficient / missing names the asked
   condition). These are diagnosis ids (disjoint from the slice) — fair game.
4. Colab slice run → `eval/results_v2/slice_T3.jsonl` (+ score `--judge`),
   same `--ids` command as T2. Agent compares vs slice_T2:
   - unanswerable 4/4 judge-correct holds (hard gate),
   - factual: llm_calls/latency down or flat, judge not worse,
   - no over-refusal: answered categories must not start refusing
     (M4-recalibration failure mode — count REFUSAL-regex hits per category),
   - overall judge within churn of 14 (zero-evidence flips discounted, both
     directions, per the T1/T2 convention).
5. Accept → PROGRESS entry (T3 done) + commit + push. Full 306-q run stays T5.

If over-refusal appears, the documented structural lever (from M4) is gating
GAP_NOTE on gaps covering the whole question — that would be a plan revision,
not an improvised edit.

## Revision T3.2 (after first slice run failed the gates)

slice_T3 vs slice_T2: judge 14→9 raw; hard gates failed (unanswerable 3/4,
factual slower 3.7→4.0 llm + judge 2→1, over-refusal v2q041; cost up across
the board llm 3.96→4.50). Trace/answer analysis isolated ONE mechanism behind
the robust losses: the v1 near-miss rule demanded verification "for the asked
material and condition" that the 300-char evidence view structurally cannot
provide — chunks are excerpts and rarely carry their own author/year or full
conditions (v2q105: snippet states the exact asked values, gap stayed open,
answer self-contradicts; v2q041: Hu half refused despite ev 0.5). Trap wins
(v2q283/299 refuse, gaps name the asked subject) were real; v2q284 is a
partial-match trap (condition 1800°C matches, composition ZVC differs) that
also failed in T2 — not chased further. v2q012/169/250/264 flips are churn or
ev-0.0 discounts per convention.

Fix (CHECK_SYSTEM only, again): reframe near-miss as an ABSENCE test — fires
only when NO snippet mentions the asked subject at all; explicit release
valve that excerpts need not repeat attribution/conditions. Factual rule
drops "for the asked material and condition" (the paranoia vector); rule
conflict now prevented by the absence framing itself. Same validation gates;
outputs → trap_T3b.jsonl / slice_T3b.jsonl (never overwrite the T3 run).

## Revision T3.3 (after T3.2 slice run)

slice_T3b vs slice_T2: collateral fully recovered (llm 4.04, latency flat,
factual 3.7 calls, v2q105 + v2q041 correct; robust flips net 0, judge 13≈14)
BUT the target regressed: traps 1/3 (v2q283 answers again — its subject is
PARTIALLY present: snippets mention graphene composites, just not the 5 vol%
comparison, so the absence test never fires) and slice unanswerable 2/4
(v2q304 now answers with the neighbor's Mo value). Lesson: trap subjects are
never entirely absent; "mentioned at all" is the wrong discriminator.

T3.2 changed two things at once (framing verification→absence AND the
attribution release valve). T3.3 separates them: v1's strict exclusion
framing ("different material/composition/condition does not cover") PLUS the
release valve (a snippet stating the property for the asked subject counts
without authors/year/details). Factual rule keeps its T3.2 form (measured
well). Outputs → trap_T3c.jsonl / slice_T3c.jsonl. Decision rule: traps
≥2/3 + unanswerable 4/4 + v2q105/041 stay correct + llm ~4.0 → accept; if
attribution paranoia returns (v2q105 refuses) → the valve is insufficient,
revert checker.py to pre-T3 (git show b7248c1:agentic/checker.py) and close
T3 as attempted.

## Files touched

- `agentic/checker.py` (CHECK_SYSTEM two rules), this plan,
  `agent/PROGRESS.md` (entries + board).
- NOT touched: agentic/graph.py (GAP_NOTE incl.), planner, synthesis prompts
  (T4), scorers, eval/tuning_slice.json (frozen), baseline artifacts.
