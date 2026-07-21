# /onboard — get a fresh agent fully up to speed

You are joining the agentic RAG pipeline project mid-flight. Do the following
IN ORDER, then report. Do not start implementing anything.

1. Read `PRD.md` — what we are building, success criteria, module breakdown.
2. Read `agent/PROGRESS.md` — the module status board and the newest entries.
   The top entry is where the last agent stopped.
3. Read `CLAUDE.md` — engineering ground rules; they override your defaults.
4. Read every plan in `agent/plans/` for modules that are `planned` or
   `in-progress` on the status board (skip `done` ones).
5. Check the working tree: `git status`, `git log --oneline -5`, and confirm
   you are on branch `agentic_pipeline`. If there are uncommitted changes,
   list them — a previous agent may have stopped mid-task.
6. If a module is `in-progress`, run its test command from the plan/PROGRESS
   entry and report pass/fail — that is the ground truth of where things stand.

Then report back in this exact shape, and STOP:
- **State**: one paragraph — which modules are done / in progress / next.
- **Last agent stopped at**: from the newest PROGRESS entry + git status.
- **Tests**: what you ran and the result.
- **Proposed next action**: either "resume MX at step N" or "no plan exists
  for the next module (MX) — ready to plan it with you".

Wait for the human. They will either enter plan mode with you to write the
next module plan (which you save to `agent/plans/`), or tell you to `/build`.
