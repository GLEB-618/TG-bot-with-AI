import logging
from shared.paths import LOG_DIR
from shared.сonfig import LOG_LEVEL


LOG_LEVEL_NUM = getattr(logging, LOG_LEVEL, logging.INFO)

LOG_DIR.mkdir(parents=True, exist_ok=True)

def get_logger(name: str, filename: str) -> logging.Logger:
    from shared import LOG_DIR, LOG_LEVEL
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL_NUM)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s | %(message)s"
        )

        file_handler = logging.FileHandler(LOG_DIR / filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(LOG_LEVEL_NUM)

        if name != "aiogram":
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            stream_handler.setLevel(LOG_LEVEL_NUM)

        logger.addHandler(file_handler)
        if name != "aiogram":
            logger.addHandler(stream_handler)

    return logger