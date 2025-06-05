from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
from typing import Union

def load_model(model_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto"
    )

    # Проверяем, подключен ли LoRA
    adapter_path = model_path / "adapter_config.json"
    if adapter_path.exists():
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = base_model

    model.eval()
    return model, tokenizer