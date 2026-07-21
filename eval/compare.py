"""Compare two scored benchmark runs (baseline vs agentic).

Paired comparison on the intersection of answered question ids: per-category
deltas for key_match / evidence_recall / judge-correct, cost columns, and a
paired bootstrap significance test on the headline metrics — with ~30-50
questions per category a 2-3 question swing is noise, and this says so.

Usage:
    python eval/compare.py eval/results/bench_baseline_scored.json eval/results/bench_agentic_scored.json
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load(path):
    return {r["id"]: r for r in json.loads(Path(path).read_text())}


def rate(rows, key, val=True):
    xs = [r.get(key) == val if val is not True else bool(r.get(key)) for r in rows
          if r.get(key) is not None]
    return sum(xs) / len(xs) if xs else None


def mean(rows, key):
    xs = [r[key] for r in rows if r.get(key) is not None]
    return sum(xs) / len(xs) if xs else None


def bootstrap_delta(pairs, metric, n_boot=10000, seed=42):
    """pairs: list of (a_row, b_row). Returns (delta, p) where p is the two-sided
    bootstrap probability that the observed delta is zero-consistent."""
    vals = [(float(bool(a.get(metric))), float(bool(b.get(metric)))) for a, b in pairs
            if a.get(metric) is not None and b.get(metric) is not None]
    if len(vals) < 5:
        return None, None
    obs = sum(b - a for a, b in vals) / len(vals)
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_boot):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        d = sum(b - a for a, b in sample) / len(sample)
        if (d <= 0) if obs > 0 else (d >= 0):
            hits += 1
    return obs, hits / n_boot


def fmt(x, pct=True):
    if x is None:
        return "    —"
    return f"{100 * x:5.1f}" if pct else f"{x:5.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    args = ap.parse_args()

    a_all, b_all = load(args.baseline), load(args.candidate)
    ids = sorted(set(a_all) & set(b_all))
    only_a, only_b = len(a_all) - len(ids), len(b_all) - len(ids)
    print(f"paired on {len(ids)} questions"
          + (f" (dropped {only_a} baseline-only, {only_b} candidate-only)" if only_a or only_b else ""))

    by_cat = defaultdict(list)
    for qid in ids:
        by_cat[a_all[qid]["category"]].append((a_all[qid], b_all[qid]))

    name_a, name_b = args.baseline.stem.replace("_scored", ""), args.candidate.stem.replace("_scored", "")
    print(f"\nA = {name_a}   B = {name_b}   (all numbers %; Δ = B - A)")
    header = (f"{'category':16} {'n':>3} | {'km A':>5} {'km B':>5} {'Δ':>6} | "
              f"{'ev A':>5} {'ev B':>5} | {'jd A':>5} {'jd B':>5} {'Δ':>6}")
    print(header + "\n" + "-" * len(header))

    def judge_ok(r):
        return r.get("correctness") == "correct" if r.get("correctness") else None

    for cat in sorted(by_cat):
        pairs = by_cat[cat]
        ra, rb = [a for a, _ in pairs], [b for _, b in pairs]
        km_a, km_b = rate(ra, "key_match"), rate(rb, "key_match")
        ja = rate([{"j": judge_ok(r)} for r in ra], "j")
        jb = rate([{"j": judge_ok(r)} for r in rb], "j")
        print(f"{cat:16} {len(pairs):>3} | {fmt(km_a):>5} {fmt(km_b):>5} "
              f"{fmt((km_b - km_a) if None not in (km_a, km_b) else None):>6} | "
              f"{fmt(mean(ra, 'evidence_recall')):>5} {fmt(mean(rb, 'evidence_recall')):>5} | "
              f"{fmt(ja):>5} {fmt(jb):>5} "
              f"{fmt((jb - ja) if None not in (ja, jb) else None):>6}")

    all_pairs = [p for ps in by_cat.values() for p in ps]
    print("-" * len(header))
    for metric, label in (("key_match", "key_match"),):
        d, p = bootstrap_delta(all_pairs, metric)
        if d is not None:
            sig = "SIGNIFICANT" if p < 0.05 else "not significant"
            print(f"overall {label}: Δ = {100 * d:+.1f} pts, bootstrap p = {p:.3f} ({sig})")
    jd_pairs = [({"m": judge_ok(a)}, {"m": judge_ok(b)}) for a, b in all_pairs
                if judge_ok(a) is not None and judge_ok(b) is not None]
    d, p = bootstrap_delta(jd_pairs, "m")
    if d is not None:
        sig = "SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"overall judge-correct: Δ = {100 * d:+.1f} pts, bootstrap p = {p:.3f} ({sig})")

    for name, rows in ((name_a, [a for a, _ in all_pairs]), (name_b, [b for _, b in all_pairs])):
        lat, calls = mean(rows, "latency_s"), mean(rows, "llm_calls")
        if lat is not None:
            print(f"cost {name}: {lat:.1f}s/question"
                  + (f", {calls:.1f} llm calls/question" if calls else ""))


if __name__ == "__main__":
    main()
