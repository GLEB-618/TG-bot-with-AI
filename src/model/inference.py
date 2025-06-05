from model.loader import load_model
import torch
import json
from pathlib import Path
from shared import CHECKPOINTS_DIR, Timer

# Грузим модель и токенизатор один раз
def get_versioned_dir(base_dir: Path, prefix: str = "final_model_v") -> Path:
    i = 1
    while (base_dir / f"{prefix}{i}").exists():
        i += 1
    return base_dir / f"{prefix}{i-1}"

FINAL_DIR = get_versioned_dir(CHECKPOINTS_DIR)
model, tokenizer = load_model(FINAL_DIR)

# Грузим параметры генерации
config_path = Path(__file__).resolve().parent / "config.json"
with open(config_path, "r", encoding="utf-8") as f:
    generation_config = json.load(f)

def generate_response(prompt: str) -> str:
    with Timer(label="Генерация ответа"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_config
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)