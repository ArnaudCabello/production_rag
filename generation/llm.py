"""Model-agnostic chat LLM loader. The generator is a config string —
swap models or providers in config.py without touching pipeline code.
"""
import logging

import config

log = logging.getLogger(__name__)


def get_llm(model_name: str = None, provider: str = None):
    model_name = model_name or config.GENERATOR_MODEL
    provider = provider or config.GENERATOR_PROVIDER

    if provider == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

        import torch

        quantize = config.GENERATOR_LOAD_IN_4BIT and torch.cuda.is_available()
        model_kwargs = {"dtype": "auto"}
        if quantize:
            from transformers import BitsAndBytesConfig

            model_kwargs = {"quantization_config": BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
            )}
        log.info(f"Loading generator {model_name} (local HF, deterministic decoding, "
                 f"{'4-bit' if quantize else 'full precision'})...")
        pipe = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="text-generation",
            device_map="auto",
            model_kwargs=model_kwargs,
            pipeline_kwargs={
                "max_new_tokens": config.MAX_NEW_TOKENS,
                "do_sample": False,
                "return_full_text": False,
            },
        )
        return ChatHuggingFace(llm=pipe)

    from langchain.chat_models import init_chat_model

    log.info(f"Loading generator {model_name} via {provider} API...")
    return init_chat_model(model_name, model_provider=provider, temperature=0)
