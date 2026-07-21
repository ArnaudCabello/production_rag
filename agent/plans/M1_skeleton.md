# M1 — Skeleton + adapter (agentic pass-through == baseline)

## Context
First module of the agentic RAG pipeline (PRD.md). Goal: an `agentic/` package with the TARGET LangGraph shape — planner → retrieve → synthesize — where every node is a trivial pass-through, so behaviour matches the baseline linear pipeline exactly on simple questions. Wired into the stubbed `build_agentic()` in `eval/run_benchmark.py`, with a parity test as the module's validation set (written FIRST, per /build workflow). M2–M5 then replace node internals without restructuring.

Decisions confirmed with the user:
- Multi-node skeleton now (not a thin wrapper around the baseline graph).
- Generator is Qwen3-14B via `--model` override on Colab; `config.py` default is stale — don't touch config.

## Design

**`agentic/graph.py`** (new, ~60 lines):
```python
class AgentState(TypedDict):
    question: str
    sub_queries: list[str]   # M1: [question]; M2 planner fills this
    chunks: list[dict]       # union across rounds, deduped by chunk_id
    answer: str
    llm_calls: int
    retrieval_calls: int

def build_agentic_graph(retriever, llm):  # -> compiled StateGraph
```
Nodes (closures, linear edges START→plan→retrieve→synthesize→END):
- `plan`: `{"sub_queries": [state["question"]]}` — no LLM call.
- `retrieve`: one `retriever.search(q)` per sub-query (M1: exactly one, default `top_k`), dedup by `chunk_id`; `retrieval_calls += 1` per search.
- `synthesize`: **import `SYSTEM_PROMPT`, `USER_TEMPLATE`, `format_context` from `generation/pipeline.py`** (don't duplicate); one `llm.invoke`; `llm_calls += 1`.

Counters live in state, initialized to 0 in the adapter's invoke input; each node returns the incremented value (linear graph, no reducers needed).

**Fanout/vision: deliberately NOT replicated.** Vision is out of PRD scope; the baseline's multi-doc fanout is what M2/M3 replace. Parity is defined on non-fanout, non-vision questions; the divergence is documented by an explicit test.

**`eval/run_benchmark.py`** — replace the `sys.exit` stub in `build_agentic(model, provider, top_k)`, mirroring `build_baseline`:
```python
if top_k: config.RERANK_TOP_N = top_k
graph = build_agentic_graph(HybridRetriever(), get_llm(model, provider))
def answer(question):
    r = graph.invoke({"question": question, "llm_calls": 0, "retrieval_calls": 0})
    return {"answer": r["answer"],
            "chunks": [{"chunk_id": c["chunk_id"], "text": c["text"]} for c in r["chunks"]],
            "llm_calls": r["llm_calls"], "retrieval_calls": r["retrieval_calls"]}
return answer
```

**`tests/test_agentic_parity.py`** (new) — plain executable script (assert + `print("...: OK")`), modelled on `tests/test_pipeline.py` with local `StubRetriever`/`StubLLM` (that file executes on import, so redefine locally):
1. Non-fanout question on identical stubs: baseline `build_graph` vs agentic graph → same `answer`, same chunk_id sequence; agentic reports `llm_calls == 1`, `retrieval_calls == 1`.
2. Prompt parity: StubLLM records messages; assert agentic system+user message text equals baseline's.
3. Retriever-call parity: agentic search call uses baseline defaults (top_k=5, pdfs=None, rerank=True).
4. Divergence doc: fanout-worded question → agentic does 1 retrieval (baseline fans out); asserted as intentional.

## Steps (validation-first)
1. Save this plan as `agent/plans/M1_skeleton.md`.
2. Write `tests/test_agentic_parity.py` → verify it fails (ImportError: no `agentic`).
3. Create `agentic/__init__.py` + `agentic/graph.py` → verify parity test passes.
4. Wire `build_agentic()` in `eval/run_benchmark.py` → verify `python eval/run_benchmark.py --help` works.
5. Regression: `python tests/test_pipeline.py` still passes (baseline untouched).
6. Update `agent/PROGRESS.md` (entry + status board: M1 done), commit + push to `agentic_pipeline`.

## Files
- new: `agentic/__init__.py`, `agentic/graph.py`, `tests/test_agentic_parity.py`, `agent/plans/M1_skeleton.md`
- edit: `eval/run_benchmark.py` (build_agentic body only), `agent/PROGRESS.md`
- read-only reuse: `generation/pipeline.py` (prompts, format_context, build_graph), `retrieval/retriever.py`, `generation/llm.py`

## Verification
```
python tests/test_agentic_parity.py   # module validation set — must pass
python tests/test_pipeline.py         # baseline regression — must pass
python eval/run_benchmark.py --help   # CLI intact
```
No GPU/API needed; the real `--pipeline agentic` run happens on Colab in M6.
