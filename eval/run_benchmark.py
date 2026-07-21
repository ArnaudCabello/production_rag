"""Run a pipeline over golden_set_v2 and record everything needed for scoring.

Pipeline-agnostic: each pipeline is an adapter — a callable taking a question
string and returning
    {"answer": str, "chunks": [{"chunk_id", "text"}, ...],
     "llm_calls": int, "retrieval_calls": int}
The runner adds latency and writes one JSONL line per question, so retrieval
metrics (evidence recall over the context the generator actually saw), answer
metrics, and cost metrics all come from the same record.

Built-in adapters:
    baseline   — the current pipeline (hybrid retrieval + LangGraph generator)
    agentic    — placeholder; wire your agentic pipeline into build_agentic()

Ablation: --top-k N overrides how many chunks the generator sees (config
default is 5), giving the "baseline with larger k" comparison for free.

Resume: answers append to the output JSONL; a re-run skips already-answered
ids. Delete the file for a fresh run. Safe to interrupt anytime.

Usage (repo root, ingested corpus, generator per config or flags):
    python eval/run_benchmark.py --pipeline baseline
    python eval/run_benchmark.py --pipeline baseline --top-k 15 --output eval/results/bench_baseline_k15.jsonl
    python eval/run_benchmark.py --pipeline agentic
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

import config  # noqa: E402


def build_baseline(model, provider, top_k, trace=False):
    from generation.llm import get_llm
    from generation.pipeline import build_graph
    from retrieval.retriever import HybridRetriever

    if top_k:
        config.RERANK_TOP_N = top_k
    graph = build_graph(HybridRetriever(), get_llm(model, provider), provider=provider)

    def answer(question: str) -> dict:
        result = graph.invoke({"question": question})
        return {
            "answer": result["answer"],
            "chunks": [{"chunk_id": c["chunk_id"], "text": c["text"]}
                       for c in result["chunks"]],
            "llm_calls": 1,
            "retrieval_calls": 1,
        }
    return answer


def build_agentic(model, provider, top_k, trace=False):
    from agentic.graph import build_agentic_graph
    from generation.llm import get_llm
    from retrieval.retriever import HybridRetriever

    if top_k:
        config.RERANK_TOP_N = top_k
    graph = build_agentic_graph(HybridRetriever(), get_llm(model, provider), trace=trace)

    def answer(question: str) -> dict:
        result = graph.invoke({"question": question, "llm_calls": 0,
                               "retrieval_calls": 0, "chunks": [], "rounds": 0,
                               "pending_queries": [], "queries_run": [],
                               "gaps": [], "citations": {}, "trace": []})
        out = {
            "answer": result["answer"],
            "chunks": [{"chunk_id": c["chunk_id"], "text": c["text"]}
                       for c in result["chunks"]],
            "llm_calls": result["llm_calls"],
            "retrieval_calls": result["retrieval_calls"],
            "gaps": result["gaps"],
            "citations": result["citations"],
        }
        if trace:
            out["trace"] = result["trace"]
        return out
    return answer


CLOSED_BOOK_SYSTEM = """You are a precise question-answering assistant.
Answer the question from your own knowledge. Rules:
- If you do not know the answer, say so plainly — never guess.
- Quote exact numbers and names; do not round or paraphrase figures.
- Write plain text only — your answer is displayed verbatim, not rendered. No Markdown
  (**bold**, # headings, | tables) and no LaTeX ($...$, \\frac, $_{x}$ subscripts).
  Write formulas with plain characters, e.g. HfC0.5N0.5, MPa m^1/2, 1800 C.
  For lists, start lines with "- "."""

CLOSED_BOOK_USER = """Question: {question}

Answer:"""


def build_closed_book(model, provider, top_k, trace=False):
    """Contamination control (PRD §3c): no retrieval, question-only prompt.

    Prompt mirrors the baseline SYSTEM_PROMPT minus the sources/citation rules,
    so the judge compares like with like.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from generation.llm import get_llm

    llm = get_llm(model, provider)

    def answer(question: str) -> dict:
        messages = [
            SystemMessage(content=CLOSED_BOOK_SYSTEM),
            HumanMessage(content=CLOSED_BOOK_USER.format(question=question)),
        ]
        return {
            "answer": llm.invoke(messages).content,
            "chunks": [],
            "llm_calls": 1,
            "retrieval_calls": 0,
        }
    return answer


PIPELINES = {"baseline": build_baseline, "agentic": build_agentic,
             "closed_book": build_closed_book}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", choices=sorted(PIPELINES), required=True)
    ap.add_argument("--golden", type=Path,
                    default=Path(__file__).resolve().parent / "golden_set_v2.json")
    ap.add_argument("--output", type=Path, default=None,
                    help="default: eval/results_v2/bench_<pipeline>.jsonl")
    ap.add_argument("--model", default=None, help="override config.GENERATOR_MODEL")
    ap.add_argument("--provider", default=None, help="override config.GENERATOR_PROVIDER")
    ap.add_argument("--top-k", type=int, default=None,
                    help="chunks shown to the generator (ablation; default config.RERANK_TOP_N)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default=None, help="comma-separated question ids")
    ap.add_argument("--trace", action="store_true",
                    help="record per-node agent trace in the output (agentic only; "
                         "use on small --ids/--limit runs, not the full set)")
    args = ap.parse_args()

    output = args.output or (Path(__file__).resolve().parent / "results_v2"
                             / f"bench_{args.pipeline}.jsonl")
    golden = json.loads(args.golden.read_text())["questions"]
    if args.ids:
        wanted = set(args.ids.split(","))
        golden = [q for q in golden if q["id"] in wanted]
    if args.limit:
        golden = golden[: args.limit]

    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.exists():  # resume an interrupted run; delete the file for a fresh one
        done = {json.loads(line)["id"] for line in
                output.read_text(encoding="utf-8").splitlines() if line.strip()}
        golden = [q for q in golden if q["id"] not in done]
        print(f"Resuming: {len(done)} answers already saved, {len(golden)} to go")
    if not golden:
        print("Nothing to do.")
        return

    answer_fn = PIPELINES[args.pipeline](args.model, args.provider, args.top_k,
                                         trace=args.trace)

    with output.open("a", encoding="utf-8") as f:
        for i, q in enumerate(golden, 1):
            t0 = time.monotonic()
            try:
                result = answer_fn(q["question"])
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"{q['id']}: FAILED ({e}) — will retry on next run")
                continue
            record = {
                "id": q["id"],
                "pipeline": args.pipeline,
                "question": q["question"],
                "answer": result["answer"],
                "chunks": result["chunks"],
                "llm_calls": result.get("llm_calls"),
                "retrieval_calls": result.get("retrieval_calls"),
                "latency_s": round(time.monotonic() - t0, 2),
            }
            for key in ("gaps", "citations", "trace"):
                if key in result:
                    record[key] = result[key]
            f.write(json.dumps(record) + "\n")
            f.flush()  # a crash mid-run must not lose finished answers to buffering
            print(f"[{len(done) + i}] {q['id']} ({record['latency_s']}s): "
                  f"{result['answer'][:100]}")

    print(f"\nSaved: {output}\nScore with: python eval/score_benchmark.py {output}")


if __name__ == "__main__":
    main()
