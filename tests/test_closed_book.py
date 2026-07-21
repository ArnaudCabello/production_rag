"""Closed-book adapter tests: no retrieval, question-only prompt (no models).

The closed-book run is the contamination floor (PRD §3c): the generator answers
from parametric knowledge alone, judged with the same judge as the pipelines.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import generation.llm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
import run_benchmark


class StubLLM:
    def __init__(self):
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        class R: content = "ANSWER"
        return R()


stub = StubLLM()
generation.llm.get_llm = lambda model, provider: stub

# 1. registered as a pipeline
assert "closed_book" in run_benchmark.PIPELINES
print("closed_book in PIPELINES: OK")

answer_fn = run_benchmark.build_closed_book(None, None, None)
result = answer_fn("what is X")

# 2. adapter contract: answer, no chunks, 1 llm call, 0 retrieval calls
assert result == {"answer": "ANSWER", "chunks": [], "llm_calls": 1,
                  "retrieval_calls": 0}, result
print("contract: answer, chunks=[], llm_calls=1, retrieval_calls=0: OK")

# 3. exactly one LLM call, system + user messages, question present, no sources
assert len(stub.messages) == 1
msgs = stub.messages[0]
assert len(msgs) == 2
assert "what is X" in msgs[1].content
assert "Sources" not in msgs[1].content and "{context}" not in msgs[1].content
assert "[1]" not in msgs[0].content  # no citation instruction without sources
print("prompt: single call, question only, no sources/citations: OK")

# 4. style rules parallel to baseline (same plain-text output constraints)
assert "plain text" in msgs[0].content
assert "never guess" in msgs[0].content
print("prompt: baseline-parallel style + honesty rules: OK")
