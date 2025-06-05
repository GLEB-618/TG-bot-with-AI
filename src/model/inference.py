from model.loader import load_model
import torch
import json
from pathlib import Path
from shared import Timer
from shared import bot_logger

# Грузим модель и токенизатор один раз
model, tokenizer = load_model()

# Грузим параметры генерации
config_path = Path(__file__).resolve().parent / "config.json"
with open(config_path, "r", encoding="utf-8") as f:
    generation_config = json.load(f)

def generate_response(prompt: str) -> str:
    with Timer(logger=bot_logger, label="Генерация ответа"):
        full_prompt = f"<|user|>\n{prompt.strip()}\n<|assistant|>\n"
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_config,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)