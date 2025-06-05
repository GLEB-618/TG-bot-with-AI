from pathlib import Path

# Абсолютный путь до корня проекта
BASE_DIR = Path(__file__).resolve().parents[2]

# Подпапки проекта
LOG_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data" / "processed"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

def get_next_versioned_dir(base_dir: Path, prefix: str = "final_model_v") -> Path:
    i = 1
    while (base_dir / f"{prefix}{i}").exists():
        i += 1
    return base_dir / f"{prefix}{i}"

def get_versioned_dir(base_dir: Path, prefix: str = "final_model_v") -> Path:
    i = 1
    while (base_dir / f"{prefix}{i}").exists():
        i += 1
    return base_dir / f"{prefix}{i-1}"

