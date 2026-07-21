# /build — implement a planned module, validation-first

Argument: the module id, e.g. `/build M2`. If omitted, build the lowest-number
module whose status is `planned` in `agent/PROGRESS.md`.

You must have onboarded already (/onboard). If you haven't, do that first.

## Order of operations — validation BEFORE code

1. Read the module's plan in `agent/plans/MX_*.md` end to end. If any step is
   ambiguous or contradicts PRD.md/CLAUDE.md, STOP and ask the human — do not
   improvise scope.
2. Set the module to `in-progress` in agent/PROGRESS.md (status board).
3. **Create the test/validation set FIRST**, exactly as the plan's
   "Validation first" section specifies, before writing any implementation
   code. Typical sources: slices of `eval/golden_set_v2.json`, hand-written
   cases, synthetic edge cases. Put tests under `tests/` (pytest, matching
   the repo's existing test style) and data under `agent/plans/fixtures/` if
   needed. Run them: they must FAIL (or skip) for the right reason — that
   proves they test something real.
4. Implement the plan's steps in order. After each step, run its verify check
   from the plan. Follow CLAUDE.md: minimum code that solves the problem,
   surgical changes, no speculative flexibility.
5. When all tests pass: run the full relevant test suite
   (`python -m pytest tests/ -x -q`) to catch regressions, and if the module
   touches the answer path, run a 3-question smoke through the benchmark
   runner (`python eval/run_benchmark.py --pipeline agentic --limit 3` —
   needs GPU; if unavailable, say so in PROGRESS instead of faking it).
6. Update `agent/PROGRESS.md`: status board row → `done` (or `blocked` with
   why), and append a full entry (template is in that file) including the
   exact test command and the next step for the following agent.
7. Commit with a clear message and push to `origin agentic_pipeline`.

## Hard rules

- No implementation before the validation set exists. If the plan's
  validation section is missing or vague, stop and ask.
- Never weaken a test to make it pass; if a test is wrong, say so in
  PROGRESS.md and fix it visibly.
- Stay inside the module's "Out of scope" fence. Adjacent improvements go in
  PROGRESS.md "Gotchas/ideas", not in the diff.
- If you run out of context/tokens mid-build, update PROGRESS.md FIRST
  (status + exactly where you stopped), commit and push whatever is coherent.
