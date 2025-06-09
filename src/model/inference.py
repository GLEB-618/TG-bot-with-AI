from model.loader import load_model
import torch, json
from pathlib import Path
from shared import Timer, model_logger

# Грузим модель и токенизатор один раз
model, tokenizer = load_model()
model_logger.info("Model loaded")

# Грузим параметры генерации
config_path = Path(__file__).resolve().parent / "config.json"
with open(config_path, "r", encoding="utf-8") as f:
    generation_config = json.load(f)

async def generate_response(prompt: str) -> str:
    model_logger.debug(f"Start of response generation: {prompt}")
    with Timer(logger=model_logger, label="Response generation"):
        messages = [
            {"role": "user", "content": prompt.strip()}
        ]

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_logger.debug(f"prompt_text: {prompt_text}")

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

        output_ids = outputs[0][input_ids.shape[-1]:]
        decoded = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        model_logger.debug(f"Raw decoded output: {decoded}")

        return decoded



# async def generate_response(prompt: str) -> str:
#     model_logger.debug(f"Начало генерации ответа: {prompt}")
#     with Timer(logger=model_logger, label="Генерация ответа"):

#         # Прямой текст без ролей
#         prompt_text = prompt.strip()
#         model_logger.debug(f"prompt_text: {prompt_text}")

#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token

#         encoded = tokenizer(
#             prompt_text,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=4096
#         ).to(model.device)

#         input_ids = encoded["input_ids"]
#         attention_mask = encoded["attention_mask"]

#         with torch.no_grad():
#             outputs = model.generate(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 **generation_config,
#                 pad_token_id=tokenizer.eos_token_id,
#                 eos_token_id=tokenizer.eos_token_id
#             )

#         # Отрезаем prompt, оставляем только генерацию
#         output_ids = outputs[0][input_ids.shape[-1]:]
#         decoded = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
#         model_logger.debug(f"Raw decoded output: {decoded}")

#         return decoded