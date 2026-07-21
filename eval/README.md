# Evaluation harness

## v2 benchmark (306 questions, baseline vs agentic) — current

`golden_set_v2.json`: 306 questions over the 500-paper UHTC corpus in 9
categories (factual, semantic, table / cross_document, multi_hop, aggregation,
multi_chunk / unanswerable, ambiguous). Built by `golden_v2_work/` (see its
PROGRESS.md); results go to `eval/results_v2/`, NOT `eval/results/`.

```bash
python eval/run_benchmark.py --pipeline baseline --model Qwen/Qwen3-14B  # answers (GPU, resumable)
python eval/score_benchmark.py eval/results_v2/bench_baseline.jsonl --judge  # metrics + judge (resumable)
python eval/compare.py eval/results_v2/bench_baseline_scored.json eval/results_v2/bench_agentic_scored.json
```

## Tool-use / groundedness verification

We must be able to show the pipeline's answers come from retrieval (tool
calls), not the LLM's parametric knowledge. What exists today:

- `run_benchmark.py` records `retrieval_calls` / `llm_calls` per question —
  any answered question with `retrieval_calls == 0` means the agent skipped
  the tool and answered from memory. Check this when scoring agentic runs.
- `--trace` (agentic only) records a per-node trace in each JSONL record:
  planner sub-queries, every retrieval query with the chunk_ids it returned,
  and synthesis context size — so you can see the agent actually calling its
  tools. OFF by default; keep it off for the full 306-question run and turn it
  on for small targeted runs, e.g.
  `python eval/run_benchmark.py --pipeline agentic --trace --ids f01,mh03 --output eval/results_v2/trace_check.jsonl`
- `score_benchmark.py` computes `evidence_recall` over the chunks the
  generator actually saw. A judge-correct answer with evidence_recall 0 is a
  contamination red flag — inspect those questions, don't celebrate them.

What still needs to be run/built (target: M6):

- **Closed-book control run**: the same generator answering the golden set
  with retrieval disabled (no chunks in the prompt), scored with the same
  judge. Its correct rate is the parametric-knowledge floor; baseline/agentic
  gains only count above it. Needs a small `closed_book` adapter in
  `run_benchmark.py` (`--top-k 0` currently does nothing — 0 is falsy).
- **Citation validity** (M5 tests): every citation in an answer must resolve
  to a chunk that was actually retrieved in that run.

## External judge (Gemini)

A Gemini API key is available in `.env` for testing/judging, but it is hard
capped at **20 calls/day**. Never burn it on a full 306-question pass — use it
for spot checks (≤ ~15 questions per run, leave headroom). Full-set judging
stays on the local Qwen judge.

Everything below this line documents the LEGACY 30/54-question harness; its
recorded results stay in `eval/results/`.

---

Golden Q/A set: 30 questions over the ReAct paper corpus in `golden_set.json` —
categories: `factual`, `table`, `abstract`, `semantic` (paraphrase, low keyword overlap),
`multi_chunk` (evidence spans multiple chunks).

Each question carries:
- `evidence_any` / `evidence_all` — substrings of the source document that identify the
  supporting passage(s). A retrieval hit means a top-k chunk contains them.
- `answer_keys` — substrings a correct answer must contain (each inner list is an OR group).
- `reference_answer` — ground truth for the LLM judge.

## Retrieval eval (runs on CPU, no GPU needed)

```bash
python eval/retrieval_eval.py --verify              # sanity-check golden set vs corpus
python eval/retrieval_eval.py --retriever hybrid    # full stack: fusion + blended rerank
python eval/retrieval_eval.py --retriever hybrid-norerank   # fusion only
python eval/retrieval_eval.py --retriever dense             # dense only
```

Reports hit@1/3/5/10 and MRR, overall and per category. Requires an ingested corpus
(`python -m ingestion.run`). `--verify` also flags evidence that exists in the source
documents but not in the chunk corpus — an ingestion gap and a guaranteed retrieval miss.

## Legacy baseline (pre-rewrite pipeline)

The legacy pipeline (TF-IDF-first retrieval over a MiniLM/FAISS index, single-chunk
LLaVA generation) is kept runnable on the `legacy-test` branch — `legacy/` holds the
verbatim old code and the eval gains a `--retriever legacy` option there. Its recorded
results live in `eval/results/` on every branch (`*_legacy*` files).

## Generation eval (needs GPU — run on Colab)

```bash
python eval/generate_answers_v2.py                        # answer golden set via the pipeline
python eval/judge_answers.py eval/results/answers_v2.jsonl # key_match + local LLM judge
python eval/judge_answers.py eval/results/answers_v2.jsonl --no-judge  # key_match only, no GPU
```

The judge model (`Qwen/Qwen2.5-14B-Instruct`) grades each answer against the reference:
correct / partial / incorrect. `key_match` is the objective substring check and runs anywhere.

## Recorded results (eval/results/)

- `baseline_retrieval.json`, `answers_legacy.jsonl`, `answers_legacy_scored.json` — the
  pre-rewrite pipeline on the 30-question ReAct set (hit@5 0.77, judge-correct 7/30,
  7 incorrect). The runnable legacy code lives on the `legacy-test` branch.
- `phase6_uhtc_retrieval_legacy.json`, `answers_legacy_54q*` — the legacy pipeline
  re-measured on the 54-question / 4-document corpus (hit@5 0.70, key_match 24/54)
- `phase2_dense_retrieval.json` — new corpus, dense-only
- `phase3_fusion_retrieval.json`, `phase3_hybrid_retrieval.json` — fusion, then + blended rerank
- `answers_v2*` — current pipeline generation: key_match 28/30, judge 22 correct / 0 incorrect
