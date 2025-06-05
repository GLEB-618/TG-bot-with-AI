# 🤖 Abaev Bot — Telegram-бот на Mistral 7B-Instruct v0.3

Локальный Telegram-бот, работающий на модели [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3).  
Поддерживает инференс, дообучение на своих диалогах (LoRA + 4-bit) и асинхронную работу с Telegram API через Aiogram.

---

## 📦 Установка зависимостей

```bash
pip install -r requirements.txt
```

---

## 🧠 Дообучение (SFT)

Формат тренировочного JSON:

```json
{
  "messages": [
    {"role": "user", "content": "Привет"},
    {"role": "assistant", "content": "Здорово, бро"}
  ]
}
```

Чекпоинты сохраняются в папке `checkpoints/`.

---

## 💬 Инференс

Модель использует:
- `tokenizer.apply_chat_template(...)`
- ручную установку `attention_mask`
- `pad_token = eos_token`
- генерацию через `generate(...)` с sampling

Входной промпт формируется через Telegram-хендлер → `generate_response()`.

---

## ✅ Чеклист

- [x] Поддержка 4-bit инференса
- [x] Mistral 7B Instruct v0.3
- [x] Telegram-бот (Aiogram)
- [x] LoRA дообучение
- [x] Attention mask, pad fix
- [x] Обработка ошибок инференса
