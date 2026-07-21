"""Assemble golden_set_v2.json from agent-generated batch files.

Each batch file (eval/golden_v2_work/batch_*.json) is a JSON list of candidates:
  {"category": str, "question": str, "reference_answer": str,
   "answer_keys": [[str,...],...], "evidence_any": [str,...],
   "source_docs": [pdf filename, ...],
   "excerpts": [str, ...]}   # the source text passages the agent worked from

Validation (offline, no API): evidence spans must appear verbatim in the saved
excerpts; every answer_keys group must hit the reference answer; standalone
phrasing; per-category quotas; global dedup. Deterministic given the same batch
files, so re-running after adding batches just extends the set.

Run: python eval/golden_v2_work/assemble.py
"""
import difflib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

WORK = Path(__file__).resolve().parent
OUT = WORK.parent / "golden_set_v2.json"

TARGETS = {
    "factual": 60, "semantic": 40, "table": 40,
    "cross_document": 50, "multi_hop": 30, "aggregation": 25, "multi_chunk": 30,
    "unanswerable": 25, "ambiguous": 15,
}
MULTI_EVIDENCE = {"cross_document", "multi_hop", "aggregation", "ambiguous", "multi_chunk"}
BAD_PHRASES = re.compile(
    r"\b(this (study|paper|work|sample)|the (authors|study|paper|present work))\b", re.I)


def norm(s):  # whitespace/unicode-tolerant matching: PDF extraction differs
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("­", "")  # soft hyphen
    return re.sub(r"[^a-z0-9 ]+", " ", re.sub(r"\s+", " ", s).strip().lower())


def repair_span(span, excerpts):
    """Span not verbatim: find the closest 8-20-word window in the excerpts and
    substitute the excerpt's own wording (spans only locate passages, so taking
    the source text verbatim is strictly more faithful than the agent's copy)."""
    span_words = norm(span).split()
    if not span_words:
        return None
    best, best_score = None, 0.0
    for ex in excerpts:
        words = ex.split()
        n = min(max(len(span_words), 8), 20)
        for i in range(0, max(1, len(words) - n + 1)):
            window = words[i:i + n]
            score = difflib.SequenceMatcher(
                None, span_words, norm(" ".join(window)).split()).ratio()
            if score > best_score:
                best_score, best = score, " ".join(window)
    return best if best_score >= 0.7 else None


def validate(c):
    q = c.get("question", "")
    ref = c.get("reference_answer", "")
    keys = c.get("answer_keys", [])
    evs = c.get("evidence_any", [])
    cat = c.get("category", "")
    if cat not in TARGETS:
        return f"unknown category {cat!r}"
    if len(q.split()) < 6 or not ref:
        return "too short"
    if BAD_PHRASES.search(q):
        return "non-standalone phrasing"
    if not c.get("source_docs"):
        return "missing source_docs"
    if not keys or not all(isinstance(g, list) and g for g in keys):
        return "bad answer_keys"
    low = ref.lower()
    kept = [g for g in keys if any(s.lower() in low for s in g)]
    if len(kept) < 2 and cat != "unanswerable":  # repair: drop ungrounded groups
        return "answer_key not in reference"
    c["answer_keys"] = kept
    if cat == "unanswerable":
        return None
    if not evs:
        return "no evidence"
    if cat in MULTI_EVIDENCE and len(evs) < 2:
        return "needs 2 evidence spans"
    excerpts = c.get("excerpts", [])
    joined = norm("\n".join(excerpts))
    fixed = []
    for ev in evs:
        if len(ev.split()) < 5 or "|" in ev:
            return "bad evidence span"
        if norm(ev) not in joined:
            ev = repair_span(ev, excerpts)  # substitute excerpt's own wording
            if ev is None or norm(ev) not in joined:
                return "evidence not verbatim in excerpts"
        fixed.append(ev)
    c["evidence_any"] = fixed
    if cat == "ambiguous":
        hedges = ("multiple", "several", "vary", "varies", "range", "depend")
        if not any(any(h in s.lower() for h in hedges) for g in keys for s in g):
            return "ambiguous missing hedge group"
    return None


def main():
    accepted, rejected = [], []
    counts = defaultdict(int)
    seen_q = set()
    batches = sorted(WORK.glob("batch_*.json"))
    for bf in batches:
        try:
            cands = json.loads(bf.read_text())
        except json.JSONDecodeError as e:
            print(f"SKIPPING unreadable {bf.name}: {e}")
            continue
        for c in cands:
            cat = c.get("category", "?")
            err = validate(c)
            norm_q = re.sub(r"\W+", " ", c.get("question", "").lower()).strip()
            if not err and norm_q in seen_q:
                err = "duplicate question"
            if not err and counts[cat] >= TARGETS[cat]:
                err = "category quota full"
            if err:
                rejected.append({"batch": bf.name, "error": err,
                                 "category": cat, "question": c.get("question")})
                continue
            seen_q.add(norm_q)
            counts[cat] += 1
            accepted.append({
                "id": f"v2q{len(accepted) + 1:03d}",
                "category": cat,
                "question": c["question"],
                "reference_answer": c["reference_answer"],
                "answer_keys": c["answer_keys"],
                "evidence_any": c.get("evidence_any", []),
                "source_docs": c["source_docs"],
            })

    OUT.write_text(json.dumps({
        "description": "Golden set v2: 500-paper UHTC corpus, baseline-vs-agentic "
                       "benchmark. Generated by Claude agents from source PDFs "
                       "(Haiku for tier A/C, Sonnet-class for tier B), assembled by "
                       "eval/golden_v2_work/assemble.py. Evidence spans are verbatim "
                       "PDF text; run eval/retrieval_eval.py --verify against the "
                       "index and repair non-matching spans before freezing. "
                       "FROZEN after human review — do not tune against the test split.",
        "questions": accepted,
    }, indent=2))
    (WORK / "rejected.json").write_text(json.dumps(rejected, indent=2))

    print(f"batches: {len(batches)}")
    print("accepted per category (target):")
    for cat, t in TARGETS.items():
        print(f"  {cat}: {counts[cat]}/{t}")
    print(f"total accepted: {len(accepted)}  rejected: {len(rejected)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
