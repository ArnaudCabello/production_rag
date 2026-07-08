# Evaluation harness

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
LLaVA generation) is preserved verbatim in `legacy/` so the baseline can be re-measured
on any corpus. Extra deps: `pip install -r legacy/requirements.txt`.

```bash
python legacy/run_ingestion.py                              # build legacy index over data/pdfs/
python eval/retrieval_eval.py --retriever legacy            # retrieval metrics (CPU)
python eval/generate_answers_legacy.py                      # answers (GPU: LLaVA 4-bit)
python eval/judge_answers.py eval/results/answers_legacy_54q.jsonl
```

The golden set is verified against the new pipeline's document conversion; passages the
legacy conversion drops (e.g. its chunker excludes abstracts) count as retrieval misses —
that is part of what the baseline measures.

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
  7 incorrect). The code was deleted in Phase 5 and later restored under `legacy/` for
  re-runs on new corpora.
- `phase2_dense_retrieval.json` — new corpus, dense-only
- `phase3_fusion_retrieval.json`, `phase3_hybrid_retrieval.json` — fusion, then + blended rerank
- `answers_v2*` — current pipeline generation: key_match 28/30, judge 22 correct / 0 incorrect
