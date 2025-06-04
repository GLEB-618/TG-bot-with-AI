from pathlib import Path

# Абсолютный путь до корня проекта
BASE_DIR = Path(__file__).resolve().parents[2]

# Подпапки проекта
LOG_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data" / "processed"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"