# PRD — Agentic RAG Pipeline

## 1. What we are building

An agentic RAG pipeline over the 500-paper UHTC corpus that measurably beats
the current linear pipeline (hybrid retrieval → single generation) on the
v2 benchmark (`eval/golden_set_v2.json`, 306 questions), especially on the
categories a single retrieval pass structurally cannot win:
**cross_document, multi_hop, aggregation** (Tier B), plus honest behaviour on
**unanswerable / ambiguous** (Tier C).

The agent is a LangGraph loop in which the SAME local model used by the
baseline (**Qwen/Qwen3-14B**, thinking disabled) plans, retrieves iteratively,
self-checks, and answers with citations.

## 2. Why (evidence from the benchmark design)

The baseline retrieves once and generates once. Tier-B questions need:
- multiple targeted queries (cross_document: one per paper/system),
- query chains where hop-2 depends on hop-1's answer (multi_hop),
- broad recall beyond top-5 (aggregation),
- the option to say "not in the corpus" after genuinely looking (unanswerable).

## 3. Success criteria (measured, not vibes)

- Primary: judge-correct rate on Tier B ≥ +10 pts over baseline
  (`eval/compare.py`, paired bootstrap p < 0.05).
- No regression > 3 pts on Tier A (factual, semantic, table).
- Unanswerable: correct-refusal rate ≥ baseline; false-answer rate ≤ baseline.
- Cost ceiling: ≤ 6 LLM calls and ≤ 5× baseline latency per question (median).
- Every module lands with its own test/validation set BEFORE implementation
  (see /build workflow in .claude/commands/build.md).
- Groundedness: answers must come from retrieval, not the model's parametric
  knowledge (see "Tool-use / groundedness verification" in eval/README.md):
  (a) `retrieval_calls ≥ 1` on every answered question, (b) evidence_recall
  correlates with correctness (a correct answer with 0 evidence recall is a
  contamination red flag), (c) a closed-book control run (retrieval disabled)
  scored with the same judge — its correct rate is the contamination floor and
  the agentic pipeline's gain is only credible above it, (d) M5 citation
  validity: every citation must resolve to an actually-retrieved chunk.

## 4. Architecture (target)

```
question
  → planner node        (classify: simple | comparative | multi_hop | aggregation | maybe-unanswerable;
                         emit 1-N sub-queries)
  → retrieve loop       (HybridRetriever as a tool; agent may re-query with refined
                         terms based on what came back; hard cap: 4 retrieval rounds)
  → evidence check      (are all sub-questions covered? if not and rounds remain → loop;
                         if exhausted → note gaps)
  → synthesize          (cited answer from the union of retrieved chunks;
                         must refuse/hedge when evidence is missing/conflicting)
```

Constraints:
- Reuse `retrieval/retriever.py` (HybridRetriever) unchanged — the index and
  retrieval stack are IDENTICAL to baseline; only orchestration differs.
- Same generator model/config as baseline (`get_llm`, Qwen3-14B, deterministic).
- Expose as `answer(question) -> {"answer", "chunks", "llm_calls",
  "retrieval_calls"}` and wire into `build_agentic()` in eval/run_benchmark.py.
  "chunks" = union of all chunks retrieved across rounds (evidence recall is
  measured against it).
- Runs on Colab A100 exactly like the baseline; no API keys.

## 5. Scope

IN: planner, iterative retrieval loop, evidence-sufficiency check, synthesis
with citations, run_benchmark adapter, per-module tests.
OUT (for now): re-ingestion/index changes, UI/backend integration, vision
path, new models, multi-turn conversations.

## 6. Module breakdown (each becomes a plan in agent/plans/)

| # | Module | Deliverable |
|---|--------|-------------|
| M1 | Skeleton + adapter | `agentic/` package, trivial pass-through agent == baseline behaviour, wired into run_benchmark; parity test |
| M2 | Planner | question classification + sub-query emission; validation set of questions → expected plan shapes |
| M3 | Retrieval loop | tool-use loop with round cap + dedup; tests for coverage/termination |
| M4 | Evidence check + refusal | sufficiency judgment, unanswerable/ambiguous behaviour; tests from golden set slices |
| M5 | Synthesis | cited answer from multi-round context; citation-validity tests |
| M6 | Full benchmark run + tuning | bench_agentic on Colab, compare.py report, iterate |

## 7. Process (how agents work on this — see CLAUDE.md)

- `/onboard` — read PRD.md, agent/PROGRESS.md, CLAUDE.md, latest plan; report status.
- Plan mode with the human → produces `agent/plans/MX_<name>.md`.
- `/build MX` — implement the plan: FIRST create the module's test/validation
  set, then code until it passes, then update agent/PROGRESS.md.
- Every module ends with: tests green, PROGRESS.md updated, commit + push to
  branch `agentic_pipeline`.
