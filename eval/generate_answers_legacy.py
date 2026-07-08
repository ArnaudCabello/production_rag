"""Generate answers with the LEGACY pipeline (single chunk + LLaVA).

Requires the legacy index (`python legacy/run_ingestion.py`) and a GPU
(LLaVA loads in 4-bit via bitsandbytes). Run from the repo root:
    python eval/generate_answers_legacy.py

Writes eval/results/answers_legacy_54q.jsonl for judging with eval/judge_answers.py.
"""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_DIR = REPO_ROOT / "legacy"
sys.path.insert(0, str(LEGACY_DIR))

GOLDEN_SET = Path(__file__).resolve().parent / "golden_set.json"
OUTPUT = Path(__file__).resolve().parent / "results" / "answers_legacy_54q.jsonl"


def main():
    golden = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))["questions"]

    # rag.py resolves vector_store.faiss / metadata1.json / outputs relative to cwd
    os.chdir(LEGACY_DIR)
    from rag import SimpleRAGPipeline  # noqa: E402

    pipeline = SimpleRAGPipeline()
    pipeline.initialize_models()
    if not pipeline.load_vector_store():
        sys.exit("No legacy vector store found — run `python legacy/run_ingestion.py` first.")

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
