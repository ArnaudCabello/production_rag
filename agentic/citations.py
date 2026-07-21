"""M5: deterministic citation parsing/validation — no LLM calls.

The synthesis prompt numbers sources positionally ([n] = chunks[n-1], see
generation/pipeline.py format_context). extract_citations parses the [n]
markers out of an answer and splits them into in-range (valid) and
out-of-range (invalid) indices. Shared by agentic/graph.py (pipeline
validation) and eval/score_benchmark.py (citation_validity metric).
"""
import re

# [1], [2][3], [1, 2] — comma groups allowed; anything non-numeric is not a citation
CITATION_RE = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")


def extract_citations(answer: str, n_chunks: int) -> dict:
    markers = [int(n) for m in CITATION_RE.finditer(answer)
               for n in m.group(1).split(",")]
    valid, invalid = [], []
    for n in markers:
        target = valid if 1 <= n <= n_chunks else invalid
        if n not in target:
            target.append(n)
    return {"markers": markers, "valid": valid, "invalid": invalid}
