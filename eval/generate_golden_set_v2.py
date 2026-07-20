"""Generate golden_set_v2.json for the 500-paper UHTC corpus.

Reads chunks from the existing Chroma index (so evidence strings are verbatim
indexed text), generates candidate Q/A via the Anthropic API, validates
locally, and writes eval/golden_set_v2.json in the exact golden_set schema.

Nine categories:
  Tier A (baseline competence): factual, semantic, table
  Tier B (agentic-discriminating): cross_document, multi_hop, aggregation, multi_chunk
  Tier C (robustness): unanswerable, ambiguous

Cost design: Haiku 4.5 for everything except cross_document / multi_hop /
aggregation, which need Sonnet-level composition. ~$3-5 for the full run.

Checkpointing: every completed generation call is appended to
eval/golden_v2_checkpoint.jsonl (validated candidates included). Re-running
the script skips completed calls and continues — safe to interrupt at any
point (token limits, crashes). The final golden set is (re)assembled from the
checkpoint on every run, so a finished checkpoint just rewrites the output.

Usage (repo root, after ingestion):
    python eval/generate_golden_set_v2.py            # full run (resumes if interrupted)
    python eval/generate_golden_set_v2.py --dry-run  # 2 questions/category smoke test
"""
import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

import anthropic  # noqa: E402
from ingestion.index import get_collection  # noqa: E402

CHEAP_MODEL = "claude-haiku-4-5"
SMART_MODEL = "claude-sonnet-4-6"   # cross_document / multi_hop / aggregation only
SMART_CATEGORIES = {"cross_document", "multi_hop", "aggregation"}
SEED = 42
TARGETS = {  # category -> count of ACCEPTED questions to aim for
    "factual": 60, "semantic": 40, "table": 40,
    "cross_document": 50, "multi_hop": 30, "aggregation": 25, "multi_chunk": 30,
    "unanswerable": 25, "ambiguous": 15,
}
OVERGEN = 1.5  # generate 50% extra candidates; validation prunes

client = anthropic.Anthropic()

COMMON_RULES = """Return ONLY a JSON array (no prose, no markdown fences).
Each element: {"question": str, "reference_answer": str,
"answer_keys": [[str,...],...], "evidence_any": [str,...]}

Hard rules:
- The question MUST stand alone: name the material system, composition, process,
  or first-author-and-year explicitly. NEVER write "this study", "the paper",
  "the authors", or "the sample".
- evidence_any: 1-2 spans of 8-20 words copied EXACTLY, character-for-character,
  from the provided text. Plain prose only: no formulas, subscripts, math
  symbols, or table pipes inside the span.
- answer_keys: 2-3 groups; each group is 1-3 short lowercase substrings
  (word stems ok). A correct answer must contain at least one substring from
  EVERY group. Every substring must appear in your own reference_answer.
- reference_answer: 1-3 sentences, factual, self-contained."""

PROMPTS = {
    "factual": """From this excerpt of a UHTC research paper ("{doc}"), write {n} factual
question(s) answerable from the excerpt alone. Prefer specific reported values,
mechanisms, or processing conditions.\n\n{rules}\n\nEXCERPT:\n{text}""",
    "semantic": """From this excerpt of "{doc}", write {n} question(s) that require the
excerpt to answer BUT share almost no vocabulary with it: paraphrase every key
term (e.g. if the text says "flexural strength" ask about "resistance to bending
loads"; if it says "oxidation" ask about "reaction with air at temperature").
Keep the material system named so the question stands alone.\n\n{rules}\n\nEXCERPT:\n{text}""",
    "table": """The excerpt from "{doc}" below contains serialized table data. Write {n}
question(s) whose answer is a specific numeric value or ranking read from the
table. The reference_answer must state the number with units. evidence_any spans
must come from surrounding prose or serialized cell text WITHOUT pipe characters.\n\n{rules}\n\nEXCERPT:\n{text}""",
    "multi_chunk": """The two excerpts below are from different sections of the SAME paper
("{doc}"). Write {n} question(s) that can only be answered by combining
information from BOTH excerpts (e.g. a processing condition from one and the
resulting property from the other). Put one evidence span from EACH excerpt into
evidence_any.\n\n{rules}\n\nEXCERPT A:\n{text}\n\nEXCERPT B:\n{text2}""",
    "cross_document": """The two excerpts below are from DIFFERENT papers: "{doc}" and
"{doc2}". Write {n} comparison question(s) that require BOTH papers to answer
(e.g. compare reported values, contrast processing routes, or reconcile
findings). Name both works or both material systems in the question. Put one
evidence span from EACH paper into evidence_any.\n\n{rules}\n\nPAPER 1 EXCERPT:\n{text}\n\nPAPER 2 EXCERPT:\n{text2}""",
    "multi_hop": """The excerpts below come from DIFFERENT papers on related UHTC topics.
Write {n} TWO-HOP question(s): answering requires finding a fact in one excerpt
(hop 1) and using that fact to locate/interpret a second fact in another excerpt
(hop 2). Example shape: "Which densification route produced the highest hardness
among the ZrB2-SiC composites processed by spark plasma sintering?" — the answer
cannot be retrieved by a single query because the intermediate fact determines
what to look for next. Name the material systems explicitly. Put one evidence
span from EACH excerpt used into evidence_any (2 spans).\n\n{rules}\n\n{cluster}""",
    "aggregation": """The excerpts below come from {ndocs} DIFFERENT papers that each report
the same kind of property or condition. Write {n} AGGREGATION question(s) asking
for the range, spread, or typical values of that property ACROSS the papers
(e.g. "What range of oxidation onset temperatures has been reported for HfB2-SiC
composites?"). The reference_answer must state the range/values with units,
naming at least 3 of the reported values. Put one evidence span from each of at
least 2 different excerpts into evidence_any.\n\n{rules}\n\n{cluster}""",
    "unanswerable": """The excerpt below is from "{doc}". Write {n} question(s) that SOUND
answerable by a UHTC literature corpus and are closely related to this topic,
but whose answer is NOT in the excerpt (ask about a different composition,
temperature regime, property, or test condition than any discussed). The
reference_answer must be exactly: "This information is not available in the
corpus." Set answer_keys to [["not", "no ", "unable", "cannot", "doesn't", "does not"]]
and evidence_any to [].\n\n{rules}\n\nEXCERPT:\n{text}""",
    "ambiguous": """The excerpts below come from {ndocs} DIFFERENT papers that all report
the same property for similar material systems. Write {n} deliberately
UNDERSPECIFIED question(s) that plausibly match SEVERAL of the papers at once
(e.g. "What is the flexural strength of ZrB2-SiC?" when many papers report it
for different compositions/processes). A good answer must acknowledge that
multiple values exist and enumerate or qualify them. The reference_answer must
say that multiple papers report different values and give at least 2 of them.
answer_keys MUST include one group of hedging terms like
[["multiple", "several", "vary", "varies", "range", "depend"]] plus 1-2 groups of
value substrings. Put one evidence span from each of 2 different excerpts into
evidence_any.\n\n{rules}\n\n{cluster}""",
}

BAD_PHRASES = re.compile(r"\b(this (study|paper|work|sample)|the (authors|study|paper|present work))\b", re.I)


def load_chunks():
    col = get_collection()
    res = col.get(include=["documents", "metadatas"])
    docs = defaultdict(list)  # pdf name -> [chunk texts in stored order]
    namekey = None
    for md in res["metadatas"][:5]:
        for k, v in md.items():
            if isinstance(v, str) and v.lower().endswith(".pdf"):
                namekey = k
    if not namekey:
        sys.exit("Could not find a metadata key holding the source pdf name")
    for text, md in zip(res["documents"], res["metadatas"]):
        docs[md[namekey]].append(text)
    return col, docs, namekey


def call_model(model, prompt, max_tokens=1200, retries=3):
    for i in range(retries):
        try:
            msg = client.messages.create(
                model=model, max_tokens=max_tokens, temperature=0.8,
                messages=[{"role": "user", "content": prompt}])
            return msg.content[0].text
        except anthropic.APIStatusError as e:
            if e.status_code in (429, 529, 500) and i < retries - 1:
                time.sleep(15 * (i + 1)); continue
            raise
    return None


def parse_candidates(raw):
    if not raw:
        return []
    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        out = json.loads(raw)
        return out if isinstance(out, list) else []
    except json.JSONDecodeError:
        return []


def validate(cand, category, texts):
    """texts: list of source chunk texts evidence must appear in (verbatim)."""
    q = cand.get("question", "")
    ref = cand.get("reference_answer", "")
    keys = cand.get("answer_keys", [])
    evs = cand.get("evidence_any", [])
    if len(q.split()) < 6 or not ref:
        return "too short"
    if BAD_PHRASES.search(q):
        return "non-standalone phrasing"
    if category == "unanswerable":
        return None  # no evidence to check
    if not evs:
        return "no evidence"
    if category in ("multi_hop", "aggregation", "ambiguous") and len(evs) < 2:
        return "needs 2 evidence spans"
    joined = "\n".join(texts)
    for ev in evs:
        if len(ev.split()) < 5 or "|" in ev:
            return "bad evidence span"
        if ev not in joined:
            return "evidence not verbatim"
    if not keys or not all(isinstance(g, list) and g for g in keys):
        return "bad answer_keys"
    low = ref.lower()
    for group in keys:
        if not any(s.lower() in low for s in group):
            return "answer_key not in reference"
    if category == "ambiguous":
        hedges = ("multiple", "several", "vary", "varies", "range", "depend")
        if not any(any(h in s.lower() for h in hedges) for g in keys for s in g):
            return "ambiguous missing hedge group"
    return None


def cross_doc_pairs(col, docs, namekey, n_pairs, rng):
    """Pair topically-similar chunks from different papers via the index itself."""
    names = list(docs)
    pairs, seen = [], set()
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 6:
        attempts += 1
        d1 = rng.choice(names)
        c1 = rng.choice(docs[d1])
        try:
            near = col.query(query_texts=[c1], n_results=8,
                             include=["documents", "metadatas"])
        except Exception:
            continue
        for text, md in zip(near["documents"][0], near["metadatas"][0]):
            d2 = md[namekey]
            if d2 != d1 and (d1, d2) not in seen and (d2, d1) not in seen:
                seen.add((d1, d2))
                pairs.append((d1, c1, d2, text))
                break
    return pairs


def topic_clusters(col, docs, namekey, n_clusters, min_docs, rng, k=10):
    """Sample a seed chunk, pull its index neighbours, keep one chunk per distinct
    paper. Yields clusters of (doc_name, chunk_text) with >= min_docs papers —
    the raw material for multi_hop / aggregation / ambiguous questions."""
    names = list(docs)
    clusters = []
    attempts = 0
    while len(clusters) < n_clusters and attempts < n_clusters * 8:
        attempts += 1
        d1 = rng.choice(names)
        c1 = rng.choice(docs[d1])
        try:
            near = col.query(query_texts=[c1], n_results=k,
                             include=["documents", "metadatas"])
        except Exception:
            continue
        members = {d1: c1}
        for text, md in zip(near["documents"][0], near["metadatas"][0]):
            d = md[namekey]
            if d not in members:
                members[d] = text
        if len(members) >= min_docs:
            clusters.append(list(members.items())[:6])
    return clusters


def format_cluster(cluster, per_chunk=3000):
    return "\n\n".join(f'EXCERPT {i} (from "{d}"):\n{t[:per_chunk]}'
                       for i, (d, t) in enumerate(cluster, 1))


def build_jobs(col, docs, namekey, targets, rng):
    """Deterministic job list (seeded rng): job index is the checkpoint key."""
    jobs = []  # (category, prompt, source_texts, doc_names)
    names = list(docs)
    rng.shuffle(names)

    def spread():  # yield doc names round-robin so questions spread across corpus
        i = 0
        while True:
            yield names[i % len(names)]; i += 1

    gen = spread()
    for cat in ("factual", "semantic", "unanswerable"):
        for _ in range(int(targets[cat] * OVERGEN)):
            d = next(gen)
            t = rng.choice(docs[d])
            jobs.append((cat, PROMPTS[cat].format(doc=d, n=1, rules=COMMON_RULES,
                                                  text=t[:6000]), [t], [d]))
    # table: only chunks that look like serialized tables
    tabley = [(d, t) for d in names for t in docs[d]
              if t.count("|") > 8 or re.search(r"Table \d", t)]
    rng.shuffle(tabley)
    for d, t in tabley[: int(targets["table"] * OVERGEN)]:
        jobs.append(("table", PROMPTS["table"].format(doc=d, n=1, rules=COMMON_RULES,
                                                      text=t[:6000]), [t], [d]))
    # multi_chunk: consecutive chunks of the same doc
    for _ in range(int(targets["multi_chunk"] * OVERGEN)):
        d = next(gen)
        if len(docs[d]) < 2:
            continue
        i = rng.randrange(len(docs[d]) - 1)
        a, b = docs[d][i], docs[d][i + 1]
        jobs.append(("multi_chunk", PROMPTS["multi_chunk"].format(
            doc=d, n=1, rules=COMMON_RULES, text=a[:4000], text2=b[:4000]), [a, b], [d]))
    # cross_document: similarity-paired via the index
    for d1, c1, d2, c2 in cross_doc_pairs(col, docs, namekey,
                                          int(targets["cross_document"] * OVERGEN), rng):
        jobs.append(("cross_document", PROMPTS["cross_document"].format(
            doc=d1, doc2=d2, n=1, rules=COMMON_RULES,
            text=c1[:4000], text2=c2[:4000]), [c1, c2], [d1, d2]))
    # multi_hop / aggregation / ambiguous: topical clusters spanning papers
    for cat, min_docs in (("multi_hop", 2), ("aggregation", 4), ("ambiguous", 3)):
        for cluster in topic_clusters(col, docs, namekey,
                                      int(targets[cat] * OVERGEN), min_docs, rng):
            jobs.append((cat, PROMPTS[cat].format(
                n=1, ndocs=len(cluster), rules=COMMON_RULES,
                cluster=format_cluster(cluster)),
                [t for _, t in cluster], [d for d, _ in cluster]))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(REPO_ROOT / "eval" / "golden_set_v2.json"))
    args = ap.parse_args()

    rng = random.Random(SEED)
    col, docs, namekey = load_chunks()
    print(f"{len(docs)} documents, {sum(len(v) for v in docs.values())} chunks")

    targets = {k: (2 if args.dry_run else v) for k, v in TARGETS.items()}
    jobs = build_jobs(col, docs, namekey, targets, rng)
    smart_n = sum(1 for j in jobs if j[0] in SMART_CATEGORIES)
    print(f"{len(jobs)} generation calls ({smart_n} on {SMART_MODEL}, rest on {CHEAP_MODEL})")

    ckpt_path = REPO_ROOT / "eval" / (
        "golden_v2_checkpoint.dry.jsonl" if args.dry_run else "golden_v2_checkpoint.jsonl")
    done = {}
    if ckpt_path.exists():
        for line in ckpt_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["job"]] = rec
        print(f"Resuming: {len(done)}/{len(jobs)} generation calls already checkpointed")

    todo = [(i, j) for i, j in enumerate(jobs) if i not in done]

    def run_job(idx, job):
        cat, prompt, texts, dnames = job
        model = SMART_MODEL if cat in SMART_CATEGORIES else CHEAP_MODEL
        cands = parse_candidates(call_model(model, prompt))
        accepted, rejected = [], []
        for c in cands:
            err = validate(c, cat, texts)
            (rejected if err else accepted).append(
                {**c, "error": err} if err else {**c, "source_docs": dnames})
        return {"job": idx, "category": cat, "accepted": accepted, "rejected": rejected}

    if todo:
        with ckpt_path.open("a", encoding="utf-8") as ck, \
                ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(run_job, i, j): i for i, j in todo}
            n_done = len(done)
            for fut in as_completed(futs):
                try:
                    rec = fut.result()
                except Exception as e:
                    print(f"  call failed (job {futs[fut]}): {e}")
                    continue
                ck.write(json.dumps(rec) + "\n")
                ck.flush()  # interrupt-safe: every finished call survives
                done[rec["job"]] = rec
                n_done += 1
                if n_done % 25 == 0:
                    print(f"  {n_done}/{len(jobs)} calls done")

    # Assemble the golden set from the checkpoint (deterministic: job order).
    accepted, rejected = [], []
    counts = defaultdict(int)
    seen_q = set()
    for idx in sorted(done):
        rec = done[idx]
        cat = rec["category"]
        rejected.extend({"category": cat, **c} for c in rec["rejected"])
        for c in rec["accepted"]:
            if counts[cat] >= targets[cat]:
                break
            norm_q = re.sub(r"\W+", " ", c.get("question", "").lower()).strip()
            if norm_q in seen_q:
                rejected.append({"category": cat, "error": "duplicate question", **c})
                continue
            seen_q.add(norm_q)
            counts[cat] += 1
            accepted.append({
                "id": f"v2q{len(accepted) + 1:03d}",
                "category": cat,
                "question": c["question"],
                "reference_answer": c["reference_answer"],
                "answer_keys": c.get("answer_keys", []),
                "evidence_any": c.get("evidence_any", []),
                "source_docs": c["source_docs"],
            })

    out = {
        "description": "Golden set v2: 500-paper UHTC corpus, baseline-vs-agentic "
                       "benchmark. Generated from indexed chunks (evidence verbatim), "
                       f"models {CHEAP_MODEL}/{SMART_MODEL}, seed {SEED}. FROZEN after "
                       "human review — do not tune against the test split.",
        "questions": accepted,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    Path(args.out.replace(".json", "_rejected.json")).write_text(
        json.dumps(rejected, indent=2))
    remaining = len(jobs) - len(done)
    if remaining:
        print(f"INCOMPLETE: {remaining} generation calls still pending — "
              "re-run this script to continue from the checkpoint.")
    print("accepted per category:", dict(counts))
    print(f"rejected: {len(rejected)} (see *_rejected.json)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
