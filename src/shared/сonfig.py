from dotenv import load_dotenv
import os

load_dotenv()

TOKEN = os.getenv("TOKEN", "0")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
BASE_MODEL = os.getenv("BASE_MODEL", "unsloth/DeepSeek-R1-0528-Qwen3-8B-bnb-4bit")
USE_BASE_MODEL = os.getenv("USE_BASE_MODEL", "false").lower() == "true"
DATASET_FILE = os.getenv("DATASET_FILE", "file.jsonl")
