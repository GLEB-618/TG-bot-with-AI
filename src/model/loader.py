from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from shared import CHECKPOINTS_DIR, get_versioned_dir, USE_BASE_MODEL, BASE_MODEL, model_logger, get_bnb_config

FINAL_DIR = get_versioned_dir(CHECKPOINTS_DIR)

model_logger.info(f"Model: {BASE_MODEL}")
if not USE_BASE_MODEL:
    model_logger.info(f"Path: {FINAL_DIR}")

def load_model():
    bnb_config = get_bnb_config()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model = base_model if USE_BASE_MODEL else PeftModel.from_pretrained(base_model, FINAL_DIR)
    model.eval()

    return model, tokenizer