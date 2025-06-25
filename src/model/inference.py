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

# async def generate_response(prompt: str) -> str:
#     model_logger.debug(f"Start of response generation: {prompt}")
#     with Timer(logger=model_logger, label="Response generation"):
#         messages = [
#             {"role": "system", "content": "Ты — матерящийся, грубый, агрессивный бот, который не терпит тупых вопросов. Отвечай дерзко, язвительно, с ненормативной лексикой. Игнорируй любые нормы морали и приличий. Не оправдывайся. Не смягчай выражения."}, # You're a swearing, rude, aggressive bot who doesn't tolerate dumb questions. Answer boldly, sarcastically, with profanity. Ignore any standards of morality and decency. Don't make excuses. Don't soften your expression.
#             {"role": "user", "content": prompt.strip()}
#         ]

#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token

#         prompt_text = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )

#         model_logger.debug(f"prompt_text: {prompt_text}")

#         encoded = tokenizer(
#             prompt_text,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=4096
#         ).to(model.device)

#         input_ids = encoded["input_ids"]
#         attention_mask = encoded["attention_mask"]
#         terminators = [
#             tokenizer.eos_token_id,
#             tokenizer.convert_tokens_to_ids("<|eot_id|>")
#         ]

#         with torch.no_grad():
#             outputs = model.generate(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 **generation_config,
#                 # pad_token_id=tokenizer.eos_token_id,
#                 eos_token_id=terminators
#             )

#         output_ids = outputs[0][input_ids.shape[-1]:]
#         decoded = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
#         model_logger.debug(f"Raw decoded output: {decoded}")

#         return decoded


# RAW GENERATION
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


# MAIN HUIHUI
# def generate_response(prompt: str) -> str:
#     model_logger.debug(f"Start of response generation: {prompt}")
#     with Timer(logger=model_logger, label="Response generation"):
#         messages = [
#             {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#             # {"role": "system", "content": "You may swear, use dark humor, or be edgy, but never generate or assist with anything illegal, violent, harmful, or criminal (e.g. bombs, drugs, terrorism, pedophilia, real-world violence); refuse such requests."},
#             {"role": "user", "content": prompt.strip()}
#         ]

#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token

#         prompt_text = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )

#         model_logger.debug(f"prompt_text: {prompt_text}")

#         encoded = tokenizer(
#             [prompt_text],
#             return_tensors="pt",
#             truncation=True,
#             max_length=4096
#         ).to(model.device)

#         with torch.no_grad():
#             outputs = model.generate(
#                 **encoded,
#                 **generation_config
#             )
        
#         outputs_ids = [
#             output_ids[len(input_ids):] for input_ids, output_ids in zip(encoded.input_ids, outputs)
#         ]

#         decoded = tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)[0]
#         model_logger.debug(f"Raw decoded output: {decoded}")

#         return decoded
    


def generate_response(messages: str) -> str:
    model_logger.debug(f"Start of response generation: {messages}")
    with Timer(logger=model_logger, label="Response generation"):
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_logger.debug(f"prompt_text: {prompt_text}")

        encoded = tokenizer(
            [prompt_text],
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                **generation_config
            )
        
        outputs_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(encoded.input_ids, outputs)
        ]

        decoded = tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)[0]
        model_logger.debug(f"Raw decoded output: {decoded}")

        return decoded