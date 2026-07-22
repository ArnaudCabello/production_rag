"""T0 artifact validation: eval/tuning_slice.json + eval/results_v2/T0_diagnosis_ids.json.

Validates the COMMITTED artifacts (frozen after T0), not the generator script.
Run: python tests/test_tuning_slice.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLICE = os.path.join(ROOT, "eval", "tuning_slice.json")
DIAG = os.path.join(ROOT, "eval", "results_v2", "T0_diagnosis_ids.json")
GOLDEN = os.path.join(ROOT, "eval", "golden_set_v2.json")

RED_FLAG_IDS = {"v2q079", "v2q080", "v2q154", "v2q211", "v2q249"}
UNPAIRED_IDS = {"v2q251"}

# category -> expected slice count (plan Step 1)
SLICE_COMPOSITION = {
    "aggregation": 4,
    "cross_document": 4,
    "unanswerable": 4,
    "ambiguous": 2,
    "multi_chunk": 3,
    "multi_hop": 2,
    "factual": 3,
    "semantic": 2,
    "table": 2,
}

FAILING_CATEGORIES = {"aggregation", "cross_document", "unanswerable",
                      "ambiguous", "multi_chunk"}

checks = 0


def check(cond, msg):
    global checks
    assert cond, msg
    checks += 1


def main():
    golden = {q["id"]: q for q in json.load(open(GOLDEN))["questions"]}
    slice_doc = json.load(open(SLICE))
    diag = json.load(open(DIAG))

    # --- tuning slice ---
    check("description" in slice_doc and "questions" in slice_doc,
          "slice must mirror planner_slice.json shape")
    qs = slice_doc["questions"]
    check(24 <= len(qs) <= 30, f"slice size {len(qs)} outside 24-30")
    ids = [q["id"] for q in qs]
    check(len(set(ids)) == len(ids), "duplicate ids in slice")
    for q in qs:
        check(set(q.keys()) == {"id", "question", "category"},
              f"{q.get('id')}: slice entries carry id/question/category only "
              "(no answers - golden set stays frozen)")
        check(q["id"] in golden, f"{q['id']} not in golden set")
        check(q["question"] == golden[q["id"]]["question"],
              f"{q['id']} question text mismatch vs golden set")
        check(q["category"] == golden[q["id"]]["category"],
              f"{q['id']} category mismatch vs golden set")
    from collections import Counter
    got = Counter(q["category"] for q in qs)
    check(dict(got) == SLICE_COMPOSITION,
          f"slice composition {dict(got)} != specced {SLICE_COMPOSITION}")

    # --- diagnosis ids ---
    diag_ids = [d["id"] for d in diag["questions"]]
    check(len(set(diag_ids)) == len(diag_ids), "duplicate diagnosis ids")
    check(15 <= len(diag_ids) <= 30, f"diagnosis count {len(diag_ids)} off")
    for d in diag["questions"]:
        check(d["id"] in golden, f"diag {d['id']} not in golden set")
        check(golden[d["id"]]["category"] in FAILING_CATEGORIES,
              f"diag {d['id']} in non-failing category")
        check(bool(d.get("reason")), f"diag {d['id']} missing reason")
    diag_cats = Counter(golden[i]["category"] for i in diag_ids)
    for cat in FAILING_CATEGORIES:
        check(diag_cats.get(cat, 0) >= 3, f"diagnosis under-covers {cat}")

    # --- disjointness ---
    check(not (set(ids) & set(diag_ids)),
          "slice must be disjoint from diagnosis (eyeballed) ids")
    check(not (set(ids) & (RED_FLAG_IDS | UNPAIRED_IDS)),
          "slice contains red-flag/unpaired ids")
    check(not (set(diag_ids) & (RED_FLAG_IDS | UNPAIRED_IDS)),
          "diagnosis contains red-flag/unpaired ids")

    print(f"OK - {checks} checks passed "
          f"(slice {len(ids)}, diagnosis {len(diag_ids)})")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
