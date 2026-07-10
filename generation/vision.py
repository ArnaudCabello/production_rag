"""Figure-grounded answering with a vision-language model.

Loaded lazily on the first question whose retrieved chunks carry figures, then
kept resident. 4-bit by default so it fits alongside the text generator
(config.VISION_LOAD_IN_4BIT).
"""
import logging

import config

log = logging.getLogger(__name__)

_model = None
_processor = None

SYSTEM_PROMPT = """You are a precise document question-answering assistant.
You are given numbered text sources and the figure images they reference.
Answer using ONLY the sources and figures provided. Rules:
- Cite the sources you use inline, e.g. [1] or [2][3]; refer to figures by their label, e.g. (Figure A).
- Describe what is actually visible in the figures when the question asks about them.
- If the sources and figures do not contain the answer, say so plainly — never guess.
- Quote exact numbers and names from the sources; do not round or paraphrase figures."""

USER_TEMPLATE = """Sources:

{context}

Question: {question}

Answer (with [n] citations):"""


def _load():
    global _model, _processor
    if _model is not None:
        return
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    log.info(f"Loading vision model {config.VISION_MODEL} "
             f"({'4-bit' if config.VISION_LOAD_IN_4BIT else 'bf16'})...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # release allocator blocks the text generator no longer uses
    kwargs = {"device_map": "auto"}
    if config.VISION_LOAD_IN_4BIT:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )
    else:
        kwargs["dtype"] = torch.bfloat16
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config.VISION_MODEL, **kwargs)
    _processor = AutoProcessor.from_pretrained(config.VISION_MODEL)


def collect_figures(chunks: list[dict]) -> list[str]:
    """Unique figure paths across the retrieved chunks, retrieval order, capped."""
    paths = [p for chunk in chunks for p in chunk.get("figures", "").split(",") if p]
    return list(dict.fromkeys(paths))[: config.MAX_FIGURES_PER_ANSWER]


def answer_with_figures_api(llm, question: str, chunks: list[dict], format_context) -> str:
    """Figure-grounded answer via a multimodal API model (Claude/GPT/Gemini):
    same sources-plus-labeled-images prompt, sent as data-URI image blocks."""
    import base64

    from langchain_core.messages import HumanMessage, SystemMessage

    content = []
    for label, path in zip("ABCD", collect_figures(chunks)):
        image_b64 = base64.b64encode((config.REPO_ROOT / path).read_bytes()).decode()
        content.append({"type": "text", "text": f"Figure {label}:"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
    content.append({"type": "text", "text": USER_TEMPLATE.format(
        context=format_context(chunks), question=question)})
    reply = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=content)])
    return reply.content if isinstance(reply.content, str) else str(reply.content)


def answer_with_figures(question: str, chunks: list[dict], format_context) -> str:
    from qwen_vl_utils import process_vision_info

    _load()
    figures = collect_figures(chunks)
    content = []
    for label, path in zip("ABCD", figures):
        content.append({"type": "text", "text": f"Figure {label}:"})
        content.append({"type": "image", "image": str(config.REPO_ROOT / path)})
    content.append({"type": "text", "text": USER_TEMPLATE.format(
        context=format_context(chunks), question=question)})
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = _processor(text=[text], images=image_inputs, return_tensors="pt").to(_model.device)
    output = _model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False)
    trimmed = output[0][inputs["input_ids"].shape[1]:]
    return _processor.decode(trimmed, skip_special_tokens=True).strip()
