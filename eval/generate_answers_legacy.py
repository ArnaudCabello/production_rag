"""Generate baseline answers with the CURRENT pipeline (single chunk + LLaVA).

Run on a GPU machine (Colab A100) from the repo root:
    python eval/generate_answers_legacy.py

Writes eval/results/answers_legacy.jsonl for judging with eval/judge_answers.py.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rag import SimpleRAGPipeline  # noqa: E402

OUTPUT = Path(__file__).resolve().parent / "results" / "answers_legacy.jsonl"


def main():
    golden = json.loads((Path(__file__).resolve().parent / "golden_set.json").read_text())["questions"]

    pipeline = SimpleRAGPipeline()
    pipeline.initialize_models()
    if not pipeline.load_vector_store():
        sys.exit("No vector store found — run the ingestion pipeline first.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for q in golden:
            chunk = pipeline.tfidf_first_search(q["question"], top_k=20)
            answer = pipeline.generate_answer(q["question"], chunk) if chunk else "No relevant information found."
            f.write(json.dumps({"id": q["id"], "question": q["question"], "answer": answer}) + "\n")
            print(f"{q['id']}: {answer[:120]}")

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
