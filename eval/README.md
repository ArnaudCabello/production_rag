# Evaluation harness

Golden Q/A set: 54 questions in `golden_set.json` over a 4-paper corpus (the ReAct
paper + 3 UHTC materials-science papers) — categories: `factual`, `table`, `abstract`,
`semantic` (paraphrase, low keyword overlap), `multi_chunk` (evidence spans multiple
chunks), `cross_document` (evidence must span 2+ papers; uses `evidence_all`), and
`multimodal` (answer requires reading a figure — expected to fail until the vision
phase; the caption chunks must still be retrieved).

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

## Generation eval (needs GPU — run on Colab)

```bash
python eval/generate_answers_v2.py                        # answer golden set via the pipeline
python eval/judge_answers.py eval/results/answers_v2.jsonl # key_match + local LLM judge
python eval/judge_answers.py eval/results/answers_v2.jsonl --no-judge  # key_match only, no GPU
```

The judge model (`Qwen/Qwen2.5-14B-Instruct`) grades each answer against the reference:
correct / partial / incorrect. `key_match` is the objective substring check and runs anywhere.

## Recorded results (eval/results/)

- `baseline_retrieval.json`, `answers_legacy*` — the pre-rewrite pipeline (code deleted in
  Phase 5; numbers kept for history: hit@5 0.77, judge-correct 7/30, 7 incorrect)
- `phase2_dense_retrieval.json` — new corpus, dense-only
- `phase3_fusion_retrieval.json`, `phase3_hybrid_retrieval.json` — fusion, then + blended rerank
- `answers_v2*` — current pipeline generation, ReAct 30-q set: key_match 28/30,
  judge 22 correct / 0 incorrect
- `phase6_uhtc_retrieval.json` — 54-q set on the 4-paper corpus: hit@1 0.61,
  hit@5 0.87, hit@10 0.93, MRR 0.73; misses are q19, h05, x01, x02
  (cross_document is the weak category: hit@10 0.33)
- `answers_v2_54q*` — generation on the 54-q set: key_match 48/54, judge
  33 correct / 21 partial / 0 incorrect. cross_document 0/3 and multimodal 0/4
  correct (the latter by design — text-only pipeline; vision phase pending)
