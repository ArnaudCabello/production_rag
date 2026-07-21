"""Score a benchmark run (eval/run_benchmark.py output) against golden_set_v2.

Three metric layers:
  1. Objective (no GPU, no API — always computed):
     - key_match: every answer_keys group hits the answer
     - evidence_recall: fraction of evidence_any spans present in the retrieved
       chunks the generator actually saw (whitespace/unicode-tolerant)
     - refusal metrics for unanswerable (correct refusal vs false answer) and
       hedge acknowledgement for ambiguous
     - cost: latency / llm_calls / retrieval_calls per question
  2. LLM judge (optional, local HF model on GPU): rubric verdicts per question —
     correctness (correct/partial/incorrect), faithfulness (grounded/unsupported).
     Judge progress checkpoints to <run>_judge.jsonl; re-running resumes.
     Use a DIFFERENT model from the generator (default Qwen2.5-14B-Instruct
     judges a Qwen3-14B generator).
  3. --judge-file: merge externally produced verdicts (same jsonl schema:
     {"id", "correctness", "faithfulness"}) instead of running a local judge.

Usage (repo root):
    python eval/score_benchmark.py eval/results/bench_baseline.jsonl            # objective only
    python eval/score_benchmark.py eval/results/bench_baseline.jsonl --judge    # + local judge (GPU)
    python eval/score_benchmark.py eval/results/bench_baseline.jsonl --judge-file verdicts.jsonl

Writes <run>_scored.json (per-question) and prints the summary table.
"""
import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

JUDGE_MODEL = "Qwen/Qwen2.5-14B-Instruct"
HEDGES = ("multiple", "several", "vary", "varies", "varying", "range", "depend", "differ")
REFUSAL = re.compile(r"\b(not (available|found|present|contain|provided|mentioned)"
                     r"|no information|cannot (be )?(answer|determin|found)"
                     r"|do(es)? not (contain|provide|mention|report)|unable to)\b", re.I)

JUDGE_PROMPT = """You are grading an answer from a document Q/A system over a corpus of
ultra-high-temperature-ceramics research papers.

Question: {question}

Reference answer (ground truth): {reference}

Sources the system retrieved (what its answer must be grounded in):
{context}

System answer: {answer}

Grade on two axes:
1. correctness vs the reference:
   - correct: states the key facts of the reference without contradicting it
   - partial: some key facts, but misses or muddles others
   - incorrect: contradicts the reference, is unrelated, or refuses although the reference exists
2. faithfulness to the retrieved sources:
   - grounded: every factual claim (numbers, names, conditions) appears in the sources
   - unsupported: at least one factual claim is absent from or contradicts the sources

Reply with EXACTLY two words separated by a space: <correctness> <faithfulness>
Example: correct grounded"""


def norm(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9 ]+", " ", re.sub(r"\s+", " ", s).strip().lower())


def key_match(answer, answer_keys):
    low = answer.lower()
    return all(any(k.lower() in low for k in group) for group in answer_keys)


def evidence_recall(chunks, evidence):
    if not evidence:
        return None
    joined = norm(" \n ".join(c["text"] for c in chunks))
    return sum(norm(ev) in joined for ev in evidence) / len(evidence)


def objective_row(q, a):
    cat = q["category"]
    row = {
        "id": q["id"], "category": cat, "question": q["question"],
        "reference": q["reference_answer"], "answer": a["answer"],
        "key_match": key_match(a["answer"], q["answer_keys"]),
        "evidence_recall": evidence_recall(a.get("chunks", []), q.get("evidence_any", [])),
        "latency_s": a.get("latency_s"), "llm_calls": a.get("llm_calls"),
        "retrieval_calls": a.get("retrieval_calls"),
    }
    if cat == "unanswerable":
        row["refused"] = bool(REFUSAL.search(a["answer"]))
    if cat == "ambiguous":
        row["acknowledges_multiple"] = any(h in a["answer"].lower() for h in HEDGES)
    return row


def run_judge(rows, answers_by_id, out_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    done = {}
    if out_path.exists():  # resume interrupted judging
        for line in out_path.read_text().splitlines():
            if line.strip():
                v = json.loads(line)
                done[v["id"]] = v
        print(f"Judge resuming: {len(done)} verdicts already saved")

    todo = [r for r in rows if r["id"] not in done]
    if todo:
        tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            JUDGE_MODEL, dtype=torch.bfloat16, device_map="auto")
        with out_path.open("a") as f:
            for i, r in enumerate(todo, 1):
                chunks = answers_by_id[r["id"]].get("chunks", [])
                context = "\n\n".join(f"[{j}] {c['text'][:1200]}"
                                      for j, c in enumerate(chunks, 1)) or "(none)"
                prompt = JUDGE_PROMPT.format(question=r["question"], reference=r["reference"],
                                             context=context, answer=r["answer"])
                inputs = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], add_generation_prompt=True,
                    return_tensors="pt", return_dict=True).to(model.device)
                out = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
                reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                         skip_special_tokens=True).strip().lower()
                corr = re.search(r"\b(correct|partial|incorrect)\b", reply)
                faith = re.search(r"\b(grounded|unsupported)\b", reply)
                v = {"id": r["id"],
                     "correctness": corr.group(1) if corr else "unparseable",
                     "faithfulness": faith.group(1) if faith else "unparseable"}
                f.write(json.dumps(v) + "\n")
                f.flush()  # judge checkpoint: nothing lost on interrupt
                done[r["id"]] = v
                print(f"\rjudging {len(done)}/{len(rows)}", end="", flush=True)
        print()
    return done


def pct(xs):
    xs = [x for x in xs if x is not None]
    return f"{100 * sum(xs) / len(xs):5.1f}%" if xs else "    —"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="bench_*.jsonl from run_benchmark.py")
    ap.add_argument("--golden", type=Path,
                    default=Path(__file__).resolve().parent / "golden_set_v2.json")
    ap.add_argument("--judge", action="store_true", help="run the local LLM judge (GPU)")
    ap.add_argument("--judge-file", type=Path, default=None,
                    help="merge externally produced verdicts instead of judging locally")
    args = ap.parse_args()

    golden = {q["id"]: q for q in json.loads(args.golden.read_text())["questions"]}
    answers = {}
    for line in args.run.read_text().splitlines():
        if line.strip():
            a = json.loads(line)
            answers[a["id"]] = a  # last write wins on re-runs

    missing = [qid for qid in golden if qid not in answers]
    rows = [objective_row(golden[qid], answers[qid]) for qid in golden if qid in answers]

    verdicts = {}
    if args.judge_file:
        for line in args.judge_file.read_text().splitlines():
            if line.strip():
                v = json.loads(line)
                verdicts[v["id"]] = v
    elif args.judge:
        verdicts = run_judge(rows, answers, args.run.with_name(args.run.stem + "_judge.jsonl"))
    for r in rows:
        v = verdicts.get(r["id"])
        if v:
            r["correctness"] = v.get("correctness")
            r["faithfulness"] = v.get("faithfulness")

    out = args.run.with_name(args.run.stem + "_scored.json")
    out.write_text(json.dumps(rows, indent=2))

    n = len(rows)
    print(f"\n=== {args.run.name}: {n} answered, {len(missing)} missing ===")
    if missing:
        print(f"    (run_benchmark.py resumes the missing ones: --ids {','.join(missing[:5])}...)")
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    header = f"{'category':16} {'n':>3}  {'key_match':>9} {'ev_recall':>9} {'judge✓':>7} {'grounded':>8}"
    print(header + "\n" + "-" * len(header))
    for cat in sorted(by_cat):
        rs = by_cat[cat]
        jm = [r.get("correctness") == "correct" for r in rs if r.get("correctness")]
        gr = [r.get("faithfulness") == "grounded" for r in rs if r.get("faithfulness")]
        print(f"{cat:16} {len(rs):>3}  {pct([r['key_match'] for r in rs]):>9} "
              f"{pct([r['evidence_recall'] for r in rs]):>9} "
              f"{pct(jm) if jm else '     —':>7} {pct(gr) if gr else '      —':>8}")
    print("-" * len(header))
    jm = [r.get("correctness") == "correct" for r in rows if r.get("correctness")]
    print(f"{'TOTAL':16} {n:>3}  {pct([r['key_match'] for r in rows]):>9} "
          f"{pct([r['evidence_recall'] for r in rows]):>9} {pct(jm) if jm else '     —':>7}")
    un = [r for r in rows if r["category"] == "unanswerable"]
    if un:
        print(f"unanswerable: correct refusal {pct([r['refused'] for r in un])}, "
              f"false answer {pct([not r['refused'] for r in un])}")
    amb = [r for r in rows if r["category"] == "ambiguous"]
    if amb:
        print(f"ambiguous: acknowledges multiple values {pct([r['acknowledges_multiple'] for r in amb])}")
    lat = [r["latency_s"] for r in rows if r.get("latency_s")]
    if lat:
        print(f"latency: mean {sum(lat)/len(lat):.1f}s, total {sum(lat)/60:.0f}min")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
