# Agentic pipeline — build progress

Every agent MUST append an entry here when it completes (or abandons) a task,
so a fresh agent can pick up exactly where the last one left off. Newest entry
FIRST. Keep entries short and factual — state, not narrative.

Entry template:
```
## [date] MX <module> — <status: planned | in-progress | done | blocked>
- What was done:
- Files touched:
- Tests: <command to run them> — <passing?>
- Next step for the following agent:
- Gotchas discovered:
```

Module status board (update the table too):

| Module | Status | Plan | Notes |
|--------|--------|------|-------|
| M1 skeleton + adapter | not started | — | |
| M2 planner | not started | — | |
| M3 retrieval loop | not started | — | |
| M4 evidence check + refusal | not started | — | |
| M5 synthesis | not started | — | |
| M6 benchmark run + tuning | not started | — | baseline run in progress on Colab (results land in Drive eval_v2/results) |

---

## 2026-07-21 project scaffolding — done
- What was done: branch `agentic_pipeline` created from main; PRD.md,
  agent/PROGRESS.md, agent/plans/, /onboard and /build commands added.
  Benchmark harness (golden_set_v2 306q, run/score/compare) already on main.
- Files touched: PRD.md, agent/PROGRESS.md, agent/plans/README.md,
  .claude/commands/onboard.md, .claude/commands/build.md, CLAUDE.md
- Tests: n/a (docs/scaffolding)
- Next step for the following agent: /onboard, then plan M1 with the human
  (plan mode) into agent/plans/M1_skeleton.md.
- Gotchas discovered: baseline benchmark run (Qwen3-14B) is running on the
  human's Colab; its results are the comparison target — do not touch
  eval/results/ (legacy) or overwrite eval/results_v2/bench_baseline*.
