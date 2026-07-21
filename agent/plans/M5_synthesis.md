# M5 — Synthesis with citations

## Context

M1–M4 are done: the agentic graph (plan → retrieve ⇄ check → synthesize) answers with the baseline prompt, prepending GAP_NOTE when the evidence check leaves gaps. The synthesize node is still the baseline pass-through; PRD M5 requires cited answers from multi-round context plus citation validity ("every citation must resolve to an actually-retrieved chunk" — PRD groundedness criterion d). The baseline SYSTEM_PROMPT already demands `[n]` citations and `format_context` numbers chunks positionally, so M5's job is not a new prompt — it is deterministic post-synthesis citation validation, deterministic context capping for the multi-round union, and a citation-validity metric in the scorer. Hard constraint: worst-case LLM budget is already at the PRD cap of 6 (1 plan + 4 checks + 1 synth), so M5 adds ZERO LLM calls.

Decisions confirmed with the human:
1. Citation validity lives BOTH in the pipeline (deterministic parse/resolve, recorded in state + JSONL) AND as a score_benchmark.py metric.
2. Context capping is deterministic: stable retrieval-union order, cap before `format_context`, dropped count logged in trace.

## Design decisions

- **Prompt unchanged when `gaps==[]`** — baseline already asks for `[n]` citations; changing wording breaks the parity test and re-opens M4-style Colab recalibration with no measured deficiency. If M6 shows poor citation coverage, tune then. GAP_NOTE path untouched.
- **Parser in new `agentic/citations.py`** (~25 lines) so graph.py and score_benchmark.py share one marker grammar. Regex `\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]` covers `[1]`, `[2][3]`, `[1, 2]`; ranges out of scope. One pure function: `extract_citations(answer, n_chunks) -> {"markers": [ints in appearance order, dupes kept], "valid": [unique in-range], "invalid": [unique out-of-range]}`; `n` resolves to `chunks[n-1]`; `0` and `> n_chunks` invalid.
- **Capping**: `MAX_SYNTH_CHUNKS = 20` constant in `agentic/graph.py` next to `MAX_ROUNDS`. Count cap only (chunks ~1–2k chars; 20 ≈ 30k chars fits the generator; a char budget is a second knob with no evidence it's needed). Cap = first-N of the existing union order (question's reranked hits first). synthesize writes back `chunks: capped` so evidence_recall and the JSONL record reflect exactly what the LLM saw.
- **JSONL**: agentic adapter also returns `citations`; runner persists `citations` and `gaps` (mirroring the `trace` handling — only when present, so baseline records are unaffected).
- **Scorer metric**: `objective_row` computes from `answer` + record `chunks` via `extract_citations` (works on baseline and old result files — no dependency on new record keys). Per row: `citations_total`, `citation_validity = valid markers / total markers` (marker-level, dupes counted), `None` when no markers. Summary adds a `cite✓` column (existing `pct()` skips None) and a `cited answers: X%` line (rows with ≥1 marker, non-refusal only).

## Budget

Citation validation is regex + indexing; capping is a slice. Worst case stays 1 plan + 4 checks + 1 synth = 6 = PRD cap.

## Changes

### agentic/citations.py (new)
`CITATION_RE`, `extract_citations(answer, n_chunks)` as above.

### agentic/graph.py
- `AgentState` += `citations: dict`.
- `MAX_SYNTH_CHUNKS = 20`.
- `synthesize`: `capped = state["chunks"][:MAX_SYNTH_CHUNKS]`; prompt built from `capped` (byte-identical to today when ≤20 chunks and no gaps); after invoke, `cites = extract_citations(answer, len(capped))`, `chunk_ids = [capped[i-1]["chunk_id"] for i in cites["valid"]]`; update = `answer`, `llm_calls`, `chunks: capped`, `citations: {**cites, "chunk_ids": chunk_ids}`; trace event `{"node": "synthesize", "context_chunks": len(capped), "dropped_chunks": dropped, "citations_valid": ..., "citations_invalid": ...}`.

### eval/run_benchmark.py
Agentic adapter: `"citations": {}` in the invoke init dict (M3 gotcha: every state key at every invoke site); `citations` added to `out`; runner copies `citations` and `gaps` into the record when present (same pattern as `trace`).

### eval/score_benchmark.py
Import `extract_citations` (REPO_ROOT already on sys.path); `objective_row` += `citations_total` / `citation_validity`; summary `cite✓` column + `cited answers` line.

### Existing tests (init-key ripple)
All invoke/`INIT` dicts gain `"citations": {}`: tests/test_agentic_parity.py, test_check.py, test_retrieval_loop.py, test_planner.py. Parity's exact synthesize trace-event assert updates to the new event dict.

## Validation set FIRST (per /build)

`tests/test_synthesis.py` (new, no GPU, ScriptedLLM/StubLLM patterns from test_check.py):
1. `extract_citations` unit cases: `[1]`; `[2][3]`; `[1, 2]`; out-of-range `[9]`/`[0]` invalid; no markers → empty; duplicates kept in `markers`, deduped in `valid`; `[see Fig. 2]`/`[a]` ignored.
2. Graph with scripted answer `"X is 5 [1][3]. Y unknown [7]."` over 3 chunks → `valid == [1, 3]`, `invalid == [7]`, `chunk_ids` = ids of 1st and 3rd chunk.
3. Capping: 25 retrieved chunks → prompt contains `[20]` not `[21]`; `result["chunks"]` has 20; trace `dropped_chunks == 5`; deterministic across runs.
4. No-cap parity: ≤20 chunks + "ANSWER" stub → synthesis prompt byte-identical to baseline; empty citations dict; `dropped_chunks == 0`.
5. Budget: llm_calls unchanged (3 on the parity path).
6. Scorer: citation_validity 1.0 / 0.5 / None cases; refusal with no markers → None, not 0.

## Steps

0. Save this plan to `agent/plans/M5_synthesis.md`.
1. Write `tests/test_synthesis.py` (red).
2. `agentic/citations.py` → citation unit tests pass.
3. `agentic/graph.py` changes → `python tests/test_synthesis.py` passes.
4. Init-key/trace updates in existing tests → full suite green:
   `python tests/test_synthesis.py && python tests/test_check.py && python tests/test_retrieval_loop.py && python tests/test_planner.py && python tests/test_agentic_parity.py && python tests/test_pipeline.py`
5. `eval/run_benchmark.py` adapter/record → suite still green.
6. `eval/score_benchmark.py` metric → verify by scoring an existing `eval/results_v2/bench_*.jsonl` (old files must not crash; `cite✓` computed from answers/chunks alone).
7. Update `agent/PROGRESS.md` (entry + status board), commit to `agentic_pipeline`.

## Colab handoff (human)

3-question smoke (`python eval/run_benchmark.py --pipeline agentic --limit 3 --trace` or `--ids`) on Colab: answers carry `[n]` markers, `citations.valid` non-empty on substantive answers, `invalid ≈ 0`, `dropped_chunks` visible on aggregation. No shape-report script needed — validation is deterministic; full-set citation_validity lands with M6.

## Files

`agentic/citations.py` (new), `agentic/graph.py`, `eval/run_benchmark.py`, `eval/score_benchmark.py`, `tests/test_synthesis.py` (new), `tests/test_agentic_parity.py`, `tests/test_check.py`, `tests/test_retrieval_loop.py`, `tests/test_planner.py`, `agent/plans/M5_synthesis.md` (new), `agent/PROGRESS.md`.
