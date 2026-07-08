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
python eval/retrieval_eval.py --verify        # sanity-check the golden set itself
python eval/retrieval_eval.py --retriever legacy --output eval/results/baseline_retrieval.json
```

Reports hit@1/3/5/10 and MRR, overall and per category. `legacy` is a verbatim copy of the
current `rag.py` scoring (see `legacy_adapter.py`) extended to return a ranked list.
New retrievers get added as choices here so every change is measured against the same set.

## Generation eval (needs GPU — run on Colab)

```bash
python eval/generate_answers_legacy.py                         # answers with current pipeline
python eval/judge_answers.py eval/results/answers_legacy.jsonl # key_match + local LLM judge
python eval/judge_answers.py eval/results/answers_legacy.jsonl --no-judge  # key_match only, no GPU
```

The judge model (`Qwen/Qwen2.5-14B-Instruct`) grades each answer against the reference:
correct / partial / incorrect. `key_match` is the objective substring check and runs anywhere.

## Known corpus gaps (baseline)

`--verify` flags evidence that exists in the source PDF markdown but not in the chunk corpus.
Currently q04 (partially), q19, q20 — the abstract is excluded by `chunks.py`. These are
guaranteed retrieval misses until ingestion is fixed, and they are in the set on purpose.
