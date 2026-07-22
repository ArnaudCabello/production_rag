"""T0: deterministically select diagnosis ids + the frozen tuning slice.

Reads golden_set_v2.json and the M6 full-run scored results (agentic +
baseline). Emits:
  eval/results_v2/T0_diagnosis_ids.json  (~5 failing ids per failing category,
                                          with a one-line reason each)
  eval/tuning_slice.json                 (26 ids, disjoint from diagnosis and
                                          red-flag ids; FROZEN after T0)

Run once: python eval/make_tuning_slice.py
Re-running is deterministic (fixed seed) but the committed artifacts are the
source of truth - do not regenerate after T0 closes.
"""
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, "eval", "golden_set_v2.json")
AGENTIC = os.path.join(ROOT, "eval", "results_v2", "bench_agentic_scored.json")
BASELINE = os.path.join(ROOT, "eval", "results_v2", "bench_baseline_scored.json")
OUT_DIAG = os.path.join(ROOT, "eval", "results_v2", "T0_diagnosis_ids.json")
OUT_SLICE = os.path.join(ROOT, "eval", "tuning_slice.json")

SEED = 20260722
EXCLUDE = {"v2q079", "v2q080", "v2q154", "v2q211", "v2q249",  # contamination red flags
           "v2q251"}                                          # unpaired (absent from baseline run)
DIAG_PER_CAT = 5

# slice composition: weighted toward failing categories (plan Step 1)
SLICE_COMPOSITION = {
    "aggregation": 4, "cross_document": 4, "unanswerable": 4,
    "ambiguous": 2, "multi_chunk": 3, "multi_hop": 2,
    "factual": 3, "semantic": 2, "table": 2,
}


def pick(rng, pool, n):
    pool = sorted(pool)  # determinism independent of dict order
    rng.shuffle(pool)
    return pool[:n]


def main():
    golden = json.load(open(GOLDEN))["questions"]
    ag = {r["id"]: r for r in json.load(open(AGENTIC))}
    bl = {r["id"]: r for r in json.load(open(BASELINE))}
    rng = random.Random(SEED)

    by_cat = {}
    for q in golden:
        if q["id"] not in EXCLUDE:
            by_cat.setdefault(q["category"], []).append(q["id"])

    def wrong(i):
        return ag[i].get("correctness") != "correct"

    # --- diagnosis ids: failure modes each TX needs to see ---
    diag = []

    def add(ids, reason):
        for i in ids:
            diag.append({"id": i, "category": ag[i]["category"], "reason": reason})

    # aggregation: judge-incorrect first (hard failures), then partial
    agg = [i for i in by_cat["aggregation"] if wrong(i)]
    agg_inc = [i for i in agg if ag[i]["correctness"] == "incorrect"]
    agg_par = [i for i in agg if i not in agg_inc]
    sel = pick(rng, agg_inc, DIAG_PER_CAT)
    sel += pick(rng, agg_par, DIAG_PER_CAT - len(sel))
    add(sel, "aggregation judge-incorrect/partial - recall vs synthesis diagnosis")

    # cross_document: evidence found but not converted
    xd = [i for i in by_cat["cross_document"]
          if wrong(i) and ag[i]["evidence_recall"] > 0]
    add(pick(rng, xd, DIAG_PER_CAT),
        "cross_document ev_recall>0 but judge not correct - conversion failure")

    # unanswerable: the answered-instead-of-refused ones, topped up with
    # refused-but-partial for round-burn diagnosis
    un_ans = [i for i in by_cat["unanswerable"] if not ag[i].get("refused")]
    add(un_ans, "unanswerable answered instead of refused")
    un_rest = [i for i in by_cat["unanswerable"]
               if ag[i].get("refused") and wrong(i)]
    add(pick(rng, un_rest, DIAG_PER_CAT - len(un_ans)),
        "unanswerable refused but judge partial - round-burn diagnosis")

    # ambiguous: no-ack first, then the rest (all are judge-non-correct)
    am_noack = [i for i in by_cat["ambiguous"]
                if not ag[i].get("acknowledges_multiple")]
    sel = pick(rng, am_noack, DIAG_PER_CAT)
    am_rest = [i for i in by_cat["ambiguous"] if i not in am_noack and wrong(i)]
    sel += pick(rng, am_rest, DIAG_PER_CAT - len(sel))
    add(sel, "ambiguous judge-non-correct (no-ack ids prioritized)")

    # multi_chunk: regressions vs baseline first, then any non-correct
    mc = [i for i in by_cat["multi_chunk"] if wrong(i)]
    mc_reg = [i for i in mc if bl[i].get("correctness") == "correct"]
    sel = pick(rng, mc_reg, DIAG_PER_CAT)
    sel += pick(rng, [i for i in mc if i not in mc_reg], DIAG_PER_CAT - len(sel))
    add(sel, "multi_chunk regressed vs baseline / judge-non-correct")

    diag_ids = {d["id"] for d in diag}

    # --- tuning slice: disjoint, ~half currently-correct / half wrong ---
    slice_qs = []
    gq = {q["id"]: q for q in golden}
    for cat, n in SLICE_COMPOSITION.items():
        pool = [i for i in by_cat[cat] if i not in diag_ids]
        right = [i for i in pool if not wrong(i)]
        bad = [i for i in pool if wrong(i)]
        n_right = min(n // 2, len(right))
        sel = pick(rng, right, n_right) + pick(rng, bad, n - n_right)
        if len(sel) < n:  # top up from whatever remains
            sel += pick(rng, [i for i in pool if i not in sel], n - len(sel))
        for i in sorted(sel):
            slice_qs.append({"id": i, "question": gq[i]["question"],
                             "category": cat})

    json.dump({"description":
               "T0 diagnosis ids (agentic failures per failing category) - "
               "eyeballed via --trace; NOT part of the tuning slice.",
               "questions": diag}, open(OUT_DIAG, "w"), indent=1)
    json.dump({"description":
               "T0 fixed validation slice for tuning modules T1-T4. FROZEN - "
               "chosen once, disjoint from diagnosis/red-flag ids, "
               "never tune against golden-set answers.",
               "questions": slice_qs}, open(OUT_SLICE, "w"), indent=1)

    print(f"diagnosis: {len(diag)} ids -> {OUT_DIAG}")
    print(f"slice: {len(slice_qs)} ids -> {OUT_SLICE}")
    print("diagnosis --ids:", ",".join(d["id"] for d in diag))


if __name__ == "__main__":
    main()
