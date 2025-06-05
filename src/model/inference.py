from model.loader import load_model
import torch
import json
from pathlib import Path
from shared import Timer
from shared import bot_logger

# Грузим модель и токенизатор один раз
model, tokenizer = load_model()
bot_logger.info("Модель загрузилась!")

# Грузим параметры генерации
config_path = Path(__file__).resolve().parent / "config.json"
with open(config_path, "r", encoding="utf-8") as f:
    generation_config = json.load(f)

async def generate_response(prompt: str) -> str:
    bot_logger.info(f"Начало генерации ответа: {prompt}")
    with Timer(logger=bot_logger, label="Генерация ответа"):
        messages = [{"role": "user", "content": prompt.strip()}]

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        encoded = tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(model.device)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        bot_logger.info(f"Сгенерированный ответ: {decoded}")

        if prompt in decoded:
            decoded = decoded.split(prompt, 1)[-1].strip()

        if "### Response:" in decoded:
            decoded = decoded.split("### Response:")[-1].strip()

        return decoded
