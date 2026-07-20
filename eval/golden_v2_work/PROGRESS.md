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

## Status

- [x] papers_manifest.tsv
- [ ] Wave 1: factual (60), semantic (40)
- [ ] Wave 2: table (40), multi_chunk (30), unanswerable (25)
- [ ] Wave 3: cross_document (50), multi_hop (30)
- [ ] Wave 4: aggregation (25), ambiguous (15)
- [ ] assemble + verify + freeze

Update the checkboxes and paper-assignment notes below as waves complete, so
the next session doesn't reuse the same papers.

## Papers used so far

(none yet)
