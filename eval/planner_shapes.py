"""Planner shape spot-check (Colab GPU): run the real planner over the
fixture slice and report plan shapes per golden category. A report, not a
gate — checks structure only, never answers (golden set stays frozen).

Usage: python eval/planner_shapes.py --model Qwen/Qwen3-14B
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIXTURE = Path(__file__).resolve().parent.parent / "agent/plans/fixtures/planner_slice.json"

# soft expectations printed alongside the numbers (not asserted)
EXPECT = {"cross_document": ">=2 sub-queries", "multi_hop": ">=2 sub-queries",
          "aggregation": ">=2 sub-queries", "factual": "1 sub-query"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None, help="generator model (as in run_benchmark)")
    ap.add_argument("--limit", type=int, default=None, help="only first N questions")
    args = ap.parse_args()

    from agentic.planner import make_plan
    from generation.llm import get_llm

    llm = get_llm(model_name=args.model)
    questions = json.load(open(FIXTURE))["questions"][: args.limit]

    by_cat = defaultdict(list)
    for i, q in enumerate(questions):
        plan = make_plan(llm, q["question"])
        by_cat[q["category"]].append(plan)
        print(f"[{i + 1}/{len(questions)}] {q['id']} {q['category']}: "
              f"{plan['category']} x{len(plan['sub_queries'])}"
              f"{' FALLBACK' if plan.get('fallback') else ''}")
        for sq in plan["sub_queries"]:
            print(f"    - {sq}")

    print("\n=== per golden category ===")
    for cat in sorted(by_cat):
        plans = by_cat[cat]
        counts = [len(p["sub_queries"]) for p in plans]
        labels = Counter(p["category"] for p in plans)
        fb = sum(bool(p.get("fallback")) for p in plans)
        print(f"{cat:15s} n={len(plans)} labels={dict(labels)} "
              f"sub-queries mean={sum(counts) / len(counts):.1f} "
              f"min={min(counts)} max={max(counts)} fallback={fb}"
              + (f"  [expect: {EXPECT[cat]}]" if cat in EXPECT else ""))


if __name__ == "__main__":
    main()
