# model.py

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

def load_llava_model_4bit():
    model_id = "llava-hf/llava-1.5-7b-hf"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant_config
    )

    return processor, model
