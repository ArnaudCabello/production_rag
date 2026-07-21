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
