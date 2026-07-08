"""Retrieval evaluation over the golden Q/A set.

Usage:
    python eval/retrieval_eval.py --verify            # check evidence strings exist in the corpus
    python eval/retrieval_eval.py --retriever legacy  # run metrics against the current pipeline
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_SET = Path(__file__).resolve().parent / "golden_set.json"
METADATA_FILE = REPO_ROOT / "metadata1.json"
K_VALUES = (1, 3, 5, 10)
MRR_K = 10


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def chunk_search_text(chunk: dict) -> str:
    """Text used for evidence matching: chunk text plus serialized tables."""
    parts = [chunk.get("text", "")]
    for table in chunk.get("tables", []):
        if table.get("caption"):
            parts.append(table["caption"])
        if table.get("headers"):
            parts.append(" | ".join(str(h) for h in table["headers"]))
        for row in table.get("rows", []):
            parts.append(" | ".join(str(cell) for cell in row))
    return normalize(" ".join(parts))


def load_golden():
    return json.loads(GOLDEN_SET.read_text(encoding="utf-8"))["questions"]


def verify_evidence(corpus_texts):
    """Every evidence string must exist in the source markdown (ground truth).
    Strings present in the source but absent from the chunk corpus are warnings:
    the ingestion pipeline dropped that passage, so retrieval will (correctly)
    score a miss there until ingestion is fixed."""
    source_texts = [
        normalize(p.read_text(encoding="utf-8"))
        for p in (REPO_ROOT / "outputs").glob("*/*-referenced.md")
    ]
    bad, gaps = [], []
    for q in load_golden():
        for ev in q.get("evidence_any", []) + q.get("evidence_all", []):
            ev_norm = normalize(ev)
            if not any(ev_norm in text for text in source_texts):
                bad.append((q["id"], ev))
            elif not any(ev_norm in text for text in corpus_texts):
                gaps.append((q["id"], ev))
    if bad:
        print(f"❌ {len(bad)} evidence string(s) not found in the SOURCE documents (golden-set bugs):")
        for qid, ev in bad:
            print(f"  {qid}: {ev!r}")
        return False
    print(f"✅ All evidence strings found in source documents ({len(load_golden())} questions)")
    if gaps:
        print(f"⚠️  {len(gaps)} evidence string(s) missing from the chunk corpus (ingestion gaps — guaranteed retrieval misses):")
        for qid, ev in gaps:
            print(f"  {qid}: {ev[:80]!r}")
    return True


def question_hit(q, retrieved_texts):
    """Return rank (1-based) of the first chunk satisfying the question's evidence, or None.

    evidence_any: a chunk containing any listed string is a hit.
    evidence_all: every listed string must be found somewhere in the retrieved set;
                  rank is the worst rank among the strings' first occurrences.
    """
    if q.get("evidence_any"):
        targets = [normalize(ev) for ev in q["evidence_any"]]
        for rank, text in enumerate(retrieved_texts, 1):
            if any(t in text for t in targets):
                return rank
        return None
    ranks = []
    for ev in q["evidence_all"]:
        ev_norm = normalize(ev)
        rank = next((r for r, text in enumerate(retrieved_texts, 1) if ev_norm in text), None)
        if rank is None:
            return None
        ranks.append(rank)
    return max(ranks)


def evaluate(retriever, questions):
    rows = []
    for q in questions:
        retrieved = retriever(q["question"], MRR_K)
        texts = [chunk_search_text(c) for c in retrieved]
        rank = question_hit(q, texts)
        rows.append({"id": q["id"], "category": q["category"], "rank": rank})

    def summarize(subset):
        n = len(subset)
        out = {f"hit@{k}": sum(1 for r in subset if r["rank"] and r["rank"] <= k) / n for k in K_VALUES}
        out["mrr"] = sum(1 / r["rank"] for r in subset if r["rank"]) / n
        out["n"] = n
        return out

    categories = sorted({r["category"] for r in rows})
    report = {"overall": summarize(rows)}
    for cat in categories:
        report[cat] = summarize([r for r in rows if r["category"] == cat])
    return report, rows


def print_report(report, rows):
    header = f"{'':14s}" + "".join(f"{'hit@' + str(k):>8s}" for k in K_VALUES) + f"{'MRR':>8s}{'n':>5s}"
    print(header)
    for name, stats in report.items():
        cells = "".join(f"{stats[f'hit@{k}']:>8.2f}" for k in K_VALUES)
        print(f"{name:14s}{cells}{stats['mrr']:>8.3f}{stats['n']:>5d}")
    misses = [r["id"] for r in rows if r["rank"] is None]
    if misses:
        print(f"\nMissed entirely (no hit in top {MRR_K}): {', '.join(misses)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="only verify evidence strings against the corpus")
    parser.add_argument("--retriever", default="legacy", choices=["legacy", "dense", "hybrid", "hybrid-norerank"])
    parser.add_argument("--output", type=Path, default=None, help="write JSON report to this path")
    args = parser.parse_args()

    if args.retriever in ("dense", "hybrid", "hybrid-norerank") and not args.verify:
        sys.path.insert(0, str(REPO_ROOT))
        if args.retriever == "dense":
            from dense_adapter import DenseRetriever
            engine = DenseRetriever()
        else:
            from retrieval.retriever import HybridRetriever
            engine = HybridRetriever(rerank=args.retriever == "hybrid")
        corpus_texts = [normalize(doc) for doc in engine.collection.get()["documents"]]
        retriever = engine.search
    else:
        metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
        corpus_texts = [chunk_search_text(c) for c in metadata]

    if args.verify:
        sys.exit(0 if verify_evidence(corpus_texts) else 1)

    if not verify_evidence(corpus_texts):
        sys.exit("Fix the golden set before evaluating.")
    print()

    if args.retriever == "legacy":
        from legacy_adapter import LegacyRetriever
        retriever = LegacyRetriever(REPO_ROOT).search

    report, rows = evaluate(retriever, load_golden())
    print_report(report, rows)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"report": report, "per_question": rows}, indent=2))
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
