from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path
import torch
from shared import CHECKPOINTS_DIR, get_versioned_dir, USE_BASE_MODEL, BASE_MODEL

FINAL_DIR = get_versioned_dir(CHECKPOINTS_DIR)

def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto"
    )

    if not USE_BASE_MODEL:
        model = PeftModel.from_pretrained(base_model, FINAL_DIR)

    model.eval()
    return model, tokenizer