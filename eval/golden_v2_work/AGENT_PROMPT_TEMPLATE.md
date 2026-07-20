# Batch-agent prompt template (golden set v2 generation)

Model: haiku for factual/semantic/table/multi_chunk/unanswerable/ambiguous;
sonnet for cross_document/multi_hop/aggregation.

---

You are generating benchmark questions for a RAG evaluation over ultra-high-
temperature-ceramics (UHTC) research papers.

Step 1 — load the Drive reader: call ToolSearch with query
"select:mcp__Google_Drive__read_file_content", then read EACH assigned PDF:
{PAPER_LIST: fileId + exact pdf filename}

Step 2 — write {N} questions of category "{CATEGORY}".
{CATEGORY_INSTRUCTIONS}

Hard rules for every question:
- The question MUST stand alone: name the material system, composition,
  process, or first-author-and-year explicitly. NEVER "this study", "the
  paper", "the authors", "the sample".
- evidence_any: spans of 8-20 words copied EXACTLY character-for-character
  from the PDF text you read. Plain prose only — no formulas, subscripts,
  math symbols, or table pipes inside a span.
- answer_keys: 2-3 groups; each group 1-3 short lowercase substrings (word
  stems ok). A correct answer must contain >=1 substring from EVERY group.
  Every substring must appear in your own reference_answer.
- reference_answer: 1-3 sentences, factual, self-contained.
- excerpts: for each question include the 150-400 word passage(s) of PDF text
  (copied verbatim) that contain your evidence spans and support the answer.

Step 3 — write the file /home/user/production_rag/eval/golden_v2_work/
batch_{CATEGORY}_{NN}.json : a JSON array where each element is
{"category": "{CATEGORY}", "question": ..., "reference_answer": ...,
 "answer_keys": [[...],...], "evidence_any": [...],
 "source_docs": [<exact pdf filename(s) with .pdf>],
 "excerpts": [<verbatim passage(s)>]}

Return only: how many questions you wrote and any papers you could not read.

---

Category instructions:

- factual: answerable from one passage alone; prefer specific reported
  values, mechanisms, or processing conditions. 1 evidence span.
- semantic: requires the passage to answer BUT shares almost no vocabulary
  with it — paraphrase every key term (flexural strength -> resistance to
  bending loads; oxidation -> reaction with air at temperature). Keep the
  material system named. 1 evidence span.
- table: answer is a specific numeric value or ranking read from a table in
  the paper; reference_answer states the number with units; evidence spans
  from surrounding prose (no pipe characters). 1-2 evidence spans.
- multi_chunk: combine information from two DIFFERENT sections of the SAME
  paper (e.g. processing condition from Experimental + property from
  Results). One evidence span from EACH section; excerpts = both passages.
- cross_document: comparison requiring TWO of your assigned papers (compare
  values, contrast routes, reconcile findings). Name both material systems
  or both works. One evidence span from EACH paper.
- multi_hop: two-hop question across your assigned papers: a fact from one
  paper (hop 1) determines what to look up in another (hop 2) — not
  answerable by one retrieval query. One evidence span per paper used.
- aggregation: asks for the range/spread/typical values of one property
  ACROSS >=3 of your assigned papers; reference_answer names >=3 reported
  values with units. Evidence spans from >=2 different papers.
- unanswerable: SOUNDS answerable by a UHTC corpus, closely related to your
  papers' topics, but the answer is NOT in any of them (different
  composition, temperature regime, property, or test condition).
  reference_answer exactly: "This information is not available in the
  corpus." answer_keys [["not", "no ", "unable", "cannot", "doesn't", "does not"]],
  evidence_any [], excerpts [].
- ambiguous: deliberately underspecified — plausibly matches SEVERAL of your
  papers at once (e.g. "What is the flexural strength of ZrB2-SiC?" when
  several papers report different values). reference_answer says multiple
  papers report different values and gives >=2 of them. answer_keys must
  include a hedge group [["multiple", "several", "vary", "varies", "range", "depend"]]
  plus 1-2 value groups. Evidence spans from 2 different papers.
