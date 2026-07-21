# Golden set v2 — generation progress

Protocol (for any session resuming this work):

1. Source papers: Google Drive folder `RAG_UHTC/data` (folder id
   `1iKr6zP-zvTbkI5j_vCDjOIICXRpVmHYJ`), descriptive filenames match the
   index's `pdf` metadata. `papers_manifest.tsv` = id \t title \t bytes.
2. Question generation: subagents (Haiku for factual/semantic/table/
   multi_chunk/unanswerable/ambiguous; Sonnet for cross_document/multi_hop/
   aggregation) read assigned PDFs via the Drive connector
   (`read_file_content`) and write `batch_<category>_<nn>.json` here —
   schema documented in assemble.py. Each candidate must include the
   `excerpts` it was written from (validation checks evidence verbatim
   against them).
3. After each wave: `python eval/golden_v2_work/assemble.py`, then commit +
   push branch `claude/rag-uhtc-v2-folder-tryfnw`. The batch files ARE the
   checkpoint — whatever is pushed is preserved; resume by generating only
   the categories still under target (assemble.py prints counts/targets).
4. When all targets met: human spot-check tier B (cross_document, multi_hop,
   aggregation), run `eval/retrieval_eval.py --verify` on Colab against the
   real index to repair/drop evidence spans that don't match indexed chunks,
   then freeze `eval/golden_set_v2.json`.

## Status (2026-07-21 ~12:00 UTC — session limit hit, resumes 22:40 UTC)

- [x] papers_manifest.tsv
- [x] factual 60/60, unanswerable 25/25
- [ ] semantic 26/40 (need ~2 more batches of 8)
- [ ] table 15/40 (need ~3 more batches of 9; batch_table_03 never landed)
- [ ] multi_chunk 0/30 (batch 01 agent died mid-write — relaunch 3 batches of 11)
- [ ] cross_document 0/50 (Sonnet, 5 batches of 11)
- [ ] multi_hop 0/30 (Sonnet, 3 batches of 11)
- [ ] aggregation 0/25 (Sonnet, 3 batches of 9, 4-5 papers each)
- [ ] ambiguous 0/15 (Haiku, 2 batches of 8, 3-4 papers each)
- [ ] assemble + verify + freeze

Resume: launch agents per AGENT_PROMPT_TEMPLATE.md with unused papers from
papers_manifest.tsv (see used list below / batch source_docs), run
assemble.py after each wave, commit + push.

Update the checkboxes and paper-assignment notes below as waves complete, so
the next session doesn't reuse the same papers.

## Papers used so far

(none yet)
