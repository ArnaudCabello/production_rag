"""Score generated answers against the golden set.

Two metrics:
  1. key_match — does the answer contain the required answer_keys substrings? (no GPU)
  2. LLM judge — a local model grades each answer against the reference (GPU; skip with --no-judge)

Usage (repo root):
    python eval/judge_answers.py eval/results/answers_legacy.jsonl
    python eval/judge_answers.py eval/results/answers_legacy.jsonl --no-judge
"""
import argparse
import json
import re
from pathlib import Path

JUDGE_MODEL = "Qwen/Qwen2.5-14B-Instruct"

JUDGE_PROMPT = """You are grading an answer produced by a document Q/A system.

Question: {question}

Reference answer (ground truth): {reference}

System answer: {answer}

Grade the system answer strictly on factual agreement with the reference:
- "correct": states the key facts of the reference without contradicting it
- "partial": states some key facts but misses or muddles others
- "incorrect": contradicts the reference, is unrelated, or refuses despite the reference existing

Reply with ONLY one word: correct, partial, or incorrect."""


def key_match(answer: str, answer_keys) -> bool:
    """Every inner list must have at least one substring present (case-insensitive)."""
    answer_lower = answer.lower()
    return all(any(k.lower() in answer_lower for k in group) for group in answer_keys)


def llm_judge(pairs):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL, dtype=torch.bfloat16, device_map="auto")

    verdicts = []
    for item in pairs:
        prompt = JUDGE_PROMPT.format(**item)
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        output = model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        reply = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
        verdict = re.search(r"correct|partial|incorrect", reply)
        verdicts.append(verdict.group(0) if verdict else "unparseable")
    return verdicts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("answers", type=Path)
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge (key_match only)")
    args = parser.parse_args()

    golden = {q["id"]: q for q in json.loads(
        (Path(__file__).resolve().parent / "golden_set.json").read_text())["questions"]}
    answers = [json.loads(line) for line in args.answers.read_text().splitlines() if line.strip()]

    results = []
    for a in answers:
        q = golden[a["id"]]
        results.append({
            "id": a["id"],
            "category": q["category"],
            "question": q["question"],
            "reference": q["reference_answer"],
            "answer": a["answer"],
            "key_match": key_match(a["answer"], q["answer_keys"]),
        })

    if not args.no_judge:
        verdicts = llm_judge(results)
        for r, v in zip(results, verdicts):
            r["judge"] = v

    n = len(results)
    print(f"key_match: {sum(r['key_match'] for r in results)}/{n}")
    if not args.no_judge:
        for verdict in ("correct", "partial", "incorrect"):
            print(f"judge {verdict}: {sum(1 for r in results if r.get('judge') == verdict)}/{n}")

    by_cat = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)
    for cat, rows in sorted(by_cat.items()):
        print(f"  {cat}: key_match {sum(r['key_match'] for r in rows)}/{len(rows)}")

    out = args.answers.with_name(args.answers.stem + "_scored.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
