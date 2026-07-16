"""Generate answers with the NEW pipeline (hybrid retrieval + LangGraph + config generator).

Run from the repo root, after `python -m ingestion.run`. With the local HF generator
(default) this needs a GPU (Colab A100); with an API provider it runs anywhere:
    python eval/generate_answers_v2.py --provider google_genai --model gemini-2.5-flash

Optional smoke test with a small local model on CPU:
    python eval/generate_answers_v2.py --model Qwen/Qwen2.5-0.5B-Instruct --limit 2

Writes eval/results/answers_v2.jsonl for judging with eval/judge_answers.py.
"""
import argparse
import json
import pathlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")  # provider API keys, same file the backend reads

from generation.llm import get_llm  # noqa: E402
from generation.pipeline import build_graph  # noqa: E402
from retrieval.retriever import HybridRetriever  # noqa: E402

OUTPUT = Path(__file__).resolve().parent / "results" / "answers_v2.jsonl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="override config.GENERATOR_MODEL")
    parser.add_argument("--provider", default=None,
                        help="override config.GENERATOR_PROVIDER (anthropic | openai | google_genai | huggingface)")
    parser.add_argument("--limit", type=int, default=None, help="only answer the first N questions")
    parser.add_argument("--ids", default=None, help="only answer these question ids, comma-separated")
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT, help="answers jsonl path")
    args = parser.parse_args()

    golden = json.loads((Path(__file__).resolve().parent / "golden_set.json").read_text())["questions"]
    if args.ids:
        wanted = set(args.ids.split(","))
        golden = [q for q in golden if q["id"] in wanted]
    if args.limit:
        golden = golden[: args.limit]

    graph = build_graph(HybridRetriever(), get_llm(args.model, args.provider), provider=args.provider)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.output.exists():  # resume an interrupted run; delete the file for a fresh one
        done = {json.loads(line)["id"] for line in args.output.read_text(encoding="utf-8").splitlines() if line}
        golden = [q for q in golden if q["id"] not in done]
        print(f"Resuming: {len(done)} answers already saved, {len(golden)} to go")
    with args.output.open("a", encoding="utf-8") as f:
        for q in golden:
            result = graph.invoke({"question": q["question"]})
            record = {
                "id": q["id"],
                "question": q["question"],
                "answer": result["answer"],
                "sources": [c["chunk_id"] for c in result["chunks"]],
            }
            f.write(json.dumps(record) + "\n")
            f.flush()  # a crash mid-run must not lose finished answers to buffering
            print(f"{q['id']}: {result['answer'][:120]}")

    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
