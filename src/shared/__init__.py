from .logging import get_logger
from .paths import BASE_DIR, LOG_DIR, DATA_DIR, CHECKPOINTS_DIR

bot_logger = get_logger("bot", "bot.log")
train_logger = get_logger("train", "train.log")
aiogram_logger = get_logger("aiogram", "aiogram.log")
error_logger = get_logger("errors", "errors.log")

__all__ = [
    "bot_logger",
    "train_logger",
    "aiogram_logger",
    "error_logger",
    "BASE_DIR",
    "LOG_DIR",
    "DATA_DIR",
    "CHECKPOINTS_DIR"
]