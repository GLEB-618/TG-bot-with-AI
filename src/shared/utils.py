import time
import logging
from shared import bot_logger

class Timer:
    def __init__(self, logger: logging.Logger = bot_logger, label: str ="⏱"):
        self.label = label
        self.logger = logger

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed = time.perf_counter() - self.start
        self.logger.info(f"[Timer] {self.label} — {elapsed:.3f} сек")