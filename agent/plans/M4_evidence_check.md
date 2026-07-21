# M4 — Evidence check + refusal

## Goal

Replace the always-sufficient check stub (M3 seam) with an LLM sufficiency
judgment that (a) decides whether retrieved evidence covers the question's
sub-questions, (b) emits refinement queries for the retrieve loop when it
doesn't, and (c) records uncovered gaps that drive refusal/hedging at
synthesis time. Refusal must be evidence-driven — NOT driven by the planner's
`unanswerable_maybe` label (Colab shape report: it never fires).

Decisions confirmed with the human:
- The check runs on EVERY question (no gating on `category=="simple"`) —
  labels are only reliable for aggregation/comparative, and the check is
  the only unanswerable detector.
- Minimal gap injection in M4: check writes `state["gaps"]`; if gaps remain
  at synthesis, a short instruction is prepended telling the model to
  refuse/hedge on uncovered parts. Full cited synthesis stays M5.
- Parse failure fail-safe: unparseable check output ⇒ `sufficient: true`
  (degrade to M3 behaviour, never burn rounds on garbage).

## Budget

Worst case: 1 plan + 4 checks (one per round, MAX_ROUNDS=4) + 1 synth =
6 LLM calls — exactly the PRD cap. `MAX_PENDING_PER_ROUND=3` unchanged.

## Design

### agentic/checker.py (new — mirrors planner.py exactly)

- `CHECK_SYSTEM` / `CHECK_USER` prompts. Input: question, planner
  sub-queries, compact view of retrieved chunks so far (chunk_id + text,
  truncated), rounds used/remaining. Output: ONLY a JSON object
  `{"sufficient": bool, "missing": [<uncovered sub-question strings>],
  "queries": [<new search queries, ≤3>]}`.
- `parse_check(text)` — same slice-braces + json.loads + strict structural
  validation as `parse_plan`; on any failure returns the fail-safe
  `{"sufficient": True, "missing": [], "queries": [], "fallback": True}`.
  Validation: `sufficient` must be bool; `missing`/`queries` must be lists
  of non-empty strings (stripped, capped at MAX_PENDING_PER_ROUND for
  queries); `sufficient==True` forces `queries=[]`.
- `make_check(llm, state)` — builds messages, invokes, parses, attaches
  `raw`.

### agentic/graph.py changes

- `AgentState` += `gaps: list[str]`.
- Default `check_fn` becomes the LLM check (`make_check`); the `check=`
  parameter stays as the test-injection seam (M3 tests keep passing their
  lambdas).
- Check node: propagate `pending_queries`, set `gaps` from verdict
  `missing` (overwrite each round — last verdict wins), and increment
  `llm_calls` when the verdict came from an LLM call (checker returns
  `llm_calls: 1` in its verdict; stub lambdas that omit it add 0 — this is
  the existing dropped-key gap in the node, now threaded through).
  Trace event += `missing`, `queries`, `fallback`.
- Synthesize node: if `state["gaps"]` non-empty, prepend a short
  instruction block to the user message (before the baseline
  USER_TEMPLATE content is unchanged otherwise): evidence for the listed
  points was not found — say so explicitly rather than guessing; if
  nothing is covered, answer that the information is not available in the
  corpus. Wording must trip `eval/score_benchmark.py`'s REFUSAL regex
  ("not available" etc.) when refusing.
- Termination logic unchanged (route_after_check already handles
  rounds-cap and empty-pending).

### eval/run_benchmark.py

- Add `gaps: []` to the invoke init dict (all invoke sites need every
  state key — M3 gotcha). Record `gaps` in the output row.
- Same init-key addition in tests' `run_agentic` helpers.

## Validation set FIRST (per /build)

`tests/test_check.py` (new), no GPU — StubLLM pattern from
test_retrieval_loop.py:

1. parse_check: valid JSON; JSON in prose/fences; garbage/truncated ⇒
   fail-safe fallback; sufficient=true forces queries empty; queries
   capped at 3; missing/queries must be lists of strings.
2. Graph with scripted check-LLM (plan call → check call(s) → "ANSWER"):
   - insufficient verdict with queries ⇒ second retrieve round runs those
     queries; llm_calls counts each check call (plan 1 + checks N + synth 1).
   - sufficient on round 1 ⇒ single round, llm_calls == 3.
   - unparseable check output ⇒ fail-safe: one round, no crash, trace
     shows fallback.
   - gaps flow: verdict missing=[...] ⇒ synthesize prompt contains the
     refusal instruction and the gap text; empty gaps ⇒ synthesis prompt
     identical to baseline (parity preserved).
   - budget: MAX_ROUNDS insufficient verdicts ⇒ total llm_calls ≤ 6.
3. Existing suites stay green: test_retrieval_loop.py (injected lambdas
   still override the default), test_planner.py, test_agentic_parity.py
   (parity test now stubs check to always-sufficient OR relies on
   fail-safe with StubLLM's unparseable "ANSWER" — verify llm_calls
   expectation: parity now includes 1 check call ⇒ counters become 3/1;
   update the test's documented expectation, this is an intentional
   redefinition like M2's).

GPU/Colab (human, later): spot-check the check's JSON shape on a handful
of unanswerable + multi_hop golden questions (fixture slice, no answers),
analogous to eval/planner_shapes.py — `eval/check_shapes.py` reusing that
script's structure.

## Files

- agentic/checker.py (new)
- agentic/graph.py (state key, default check, llm_calls/gaps threading,
  synthesize gap injection, trace)
- eval/run_benchmark.py (init key, record gaps)
- eval/check_shapes.py (new, Colab shape report)
- tests/test_check.py (new); tests/test_retrieval_loop.py,
  tests/test_agentic_parity.py, tests/test_planner.py (init keys /
  counter expectations)

## Verify

`python tests/test_check.py && python tests/test_retrieval_loop.py &&
python tests/test_planner.py && python tests/test_agentic_parity.py &&
python tests/test_pipeline.py` — all green, no GPU needed. Colab shape
report before M5 relies on gap notes.
