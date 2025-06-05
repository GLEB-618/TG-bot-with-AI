# 🤖 TG-bot-with-AI

Телеграм-бот с интеграцией кастомной языковой модели, дообученной на стиле общения конкретного человека (Саввчика). Используется `transformers`, `PEFT`, `LoRA` и 4-битное квантование для эффективной генерации и обучения на GPU с ограниченной памятью.

---

## 📦 Возможности

- Генерация ответов в стиле Саввчика
- Интеграция с Telegram через `aiogram`
- Дообучение LLM с помощью LoRA
- Поддержка 4bit модели (через `bitsandbytes`)
- Callback'и с логированием, генерацией и метриками (loss, perplexity)

---

## 🧱 Структура проекта

```
src/
├── bot/                    # Telegram-бот (на aiogram)
│   ├── handlers/           # Обработчики команд
├── model/                  # Загрузка модели и генерация
├── shared/                 # Константы, пути, логгеры
├── training/               # Обучение модели и LoRA
│   ├── train.py            # Основной скрипт дообучения
│   ├── prepare_data.py     # Загрузка и токенизация датасета
│   ├── callbacks.py        # Callbacks для логов и генерации
```

---

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

> Обязательно: Python 3.10+, `torch` с поддержкой CUDA, `transformers`, `peft`, `bitsandbytes`

---

### 2. Настройка переменных

Создай `.env` в корне проекта:

```
BOT_TOKEN=твой_тг_токен
```

---

### 3. Обучение модели

```bash
python src/training/train.py
```

- Использует датасет из `data/processed/`
- Сохраняет чекпоинты в `models/checkpoints/`
- Финальная модель — в `models/final_model_v*/`

---

### 4. Запуск Telegram-бота

```bash
python src/bot/main.py
```

- Ответы идут через дообученную модель
- Поддержка команд `/start`, `/reset`, кастомный чат

---

## 📈 Метрики

- `eval_loss` и `perplexity` логируются автоматически
- Генерация текста происходит после каждой эпохи
- Чекпоинты логируются в `logs/train.log`

---

## 🛠 Используемые библиотеки

- [Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
- [Aiogram](https://github.com/aiogram/aiogram)

---

## 🧠 Цель проекта

Создать Telegram-бота, способного общаться как реальный человек — на основе стиля конкретного собеседника из Telegram-истории, с сохранением характерных интонаций, фраз и эмоций.

---

## 📄 Лицензия

MIT License