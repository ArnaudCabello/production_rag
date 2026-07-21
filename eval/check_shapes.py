"""Check-verdict shape spot-check (Colab GPU): run planner + round-1 retrieval
+ the real M4 check over the fixture slice and report verdict shapes per golden
category. A report, not a gate — checks structure only, never answers (golden
set stays frozen). Watch: unanswerable should trend sufficient=false with
missing≠[] and few/no queries; factual should trend sufficient=true.

Usage: python eval/check_shapes.py --model Qwen/Qwen3-14B
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIXTURE = Path(__file__).resolve().parent.parent / "agent/plans/fixtures/planner_slice.json"

EXPECT = {"unanswerable": "sufficient=false, missing!=[], queries ~0",
          "ambiguous": "sufficient often false, missing!=[]",
          "multi_hop": "sufficient=false with refinement queries",
          "factual": "sufficient=true"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None, help="generator model (as in run_benchmark)")
    ap.add_argument("--limit", type=int, default=None, help="only first N questions")
    args = ap.parse_args()

    from agentic.checker import make_check
    from agentic.graph import MAX_ROUNDS, MAX_SUB_QUERIES
    from agentic.planner import make_plan
    from generation.llm import get_llm
    from retrieval.retriever import HybridRetriever

    llm = get_llm(model_name=args.model)
    retriever = HybridRetriever()
    questions = json.load(open(FIXTURE))["questions"][: args.limit]

    by_cat = defaultdict(list)
    for i, q in enumerate(questions):
        # mirror graph round 1: plan, then search question + sub-queries, dedup chunks
        plan = make_plan(llm, q["question"])
        queries = [q["question"]] + [s for s in plan["sub_queries"]
                                     if s.strip().lower() != q["question"].strip().lower()]
        queries = queries[:1 + MAX_SUB_QUERIES]
        chunks, seen = [], set()
        for query in queries:
            for c in retriever.search(query):
                if c["chunk_id"] not in seen:
                    seen.add(c["chunk_id"])
                    chunks.append(c)
        state = {"question": q["question"], "chunks": chunks, "rounds": 1,
                 "queries_run": [s.strip().lower() for s in queries]}
        v = make_check(llm, state, MAX_ROUNDS)
        by_cat[q["category"]].append(v)
        print(f"[{i + 1}/{len(questions)}] {q['id']} {q['category']}: "
              f"sufficient={v['sufficient']} missing x{len(v['missing'])} "
              f"queries x{len(v['pending_queries'])}"
              f"{' FALLBACK' if v.get('fallback') else ''}")
        for m in v["missing"]:
            print(f"    missing: {m}")
        for pq in v["pending_queries"]:
            print(f"    query:   {pq}")

    print("\n=== per golden category ===")
    for cat in sorted(by_cat):
        vs = by_cat[cat]
        suff = Counter("true" if v["sufficient"] else "false" for v in vs)
        nq = [len(v["pending_queries"]) for v in vs]
        nm = [len(v["missing"]) for v in vs]
        fb = sum(bool(v.get("fallback")) for v in vs)
        print(f"{cat:15s} n={len(vs)} sufficient={dict(suff)} "
              f"missing mean={sum(nm) / len(nm):.1f} "
              f"queries mean={sum(nq) / len(nq):.1f} fallback={fb}"
              + (f"  [expect: {EXPECT[cat]}]" if cat in EXPECT else ""))


if __name__ == "__main__":
    main()
