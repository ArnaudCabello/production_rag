"""Model-agnostic chat LLM loader. The generator is a config string —
swap models or providers in config.py without touching pipeline code.
"""
import logging

import config

log = logging.getLogger(__name__)


def get_llm(model_name: str = None):
    model_name = model_name or config.GENERATOR_MODEL

    if config.GENERATOR_PROVIDER == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

        log.info(f"Loading generator {model_name} (local HF, deterministic decoding)...")
        pipe = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="text-generation",
            device_map="auto",
            model_kwargs={"dtype": "auto"},
            pipeline_kwargs={
                "max_new_tokens": config.MAX_NEW_TOKENS,
                "do_sample": False,
                "return_full_text": False,
            },
        )
        return ChatHuggingFace(llm=pipe)

    from langchain.chat_models import init_chat_model

    return init_chat_model(model_name, model_provider=config.GENERATOR_PROVIDER)
