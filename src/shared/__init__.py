from .logging import get_logger
from .paths import BASE_DIR, LOG_DIR, DATA_DIR, CHECKPOINTS_DIR
from .utils import Timer

bot_logger = get_logger("bot", "bot.log")
train_logger = get_logger("train", "train.log")
aiogram_logger = get_logger("aiogram", "aiogram.log")
model_logger = get_logger("model", "model.log")

__all__ = [
    "bot_logger",
    "train_logger",
    "aiogram_logger",
    "model_logger",
    "BASE_DIR",
    "LOG_DIR",
    "DATA_DIR",
    "CHECKPOINTS_DIR",
    "Timer"
]