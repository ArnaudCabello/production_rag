"""Generate answers with the NEW pipeline (hybrid retrieval + LangGraph + config generator).

Run on a GPU machine (Colab A100) from the repo root, after `python -m ingestion.run`:
    python eval/generate_answers_v2.py

Optional smoke test with a small model on CPU:
    python eval/generate_answers_v2.py --model Qwen/Qwen2.5-0.5B-Instruct --limit 2

Writes eval/results/answers_v2.jsonl for judging with eval/judge_answers.py.
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from generation.llm import get_llm  # noqa: E402
from generation.pipeline import build_graph  # noqa: E402
from retrieval.retriever import HybridRetriever  # noqa: E402

OUTPUT = Path(__file__).resolve().parent / "results" / "answers_v2.jsonl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="override config.GENERATOR_MODEL")
    parser.add_argument("--limit", type=int, default=None, help="only answer the first N questions")
    args = parser.parse_args()

    golden = json.loads((Path(__file__).resolve().parent / "golden_set.json").read_text())["questions"]
    if args.limit:
        golden = golden[: args.limit]

    graph = build_graph(HybridRetriever(), get_llm(args.model))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for q in golden:
            result = graph.invoke({"question": q["question"]})
            record = {
                "id": q["id"],
                "question": q["question"],
                "answer": result["answer"],
                "sources": [c["chunk_id"] for c in result["chunks"]],
            }
            f.write(json.dumps(record) + "\n")
            print(f"{q['id']}: {result['answer'][:120]}")

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
